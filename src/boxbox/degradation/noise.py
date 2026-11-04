"""Lap time noise for the strategy simulator. Lap errors are not independent: every lap of a stint shares that
stint's departure from the predicted wear rate, and every stint of a race shares the race's (modelling O7: a
real 0.027 s per lap rms). Independent per lap draws average out over a stint and give intervals that are far
too narrow for a total race time.

WearNoise, fitted on a wear model's residuals in the training seasons:
* per stint, the slope of the residual on tyre age (b) and its sampling variance (sigma^2 / Sxx);
* race spread tau_r: the covariance of b between stints of the same race (method of moments);
* stint spread tau_u: the variance of b left after the race spread and the sampling variance;
* lap noise: the residuals around each stint's own line, drawn with replacement (fat tailed, so not Gaussian).

simulate_errors draws the anchored forecast's error for a stint: a slope shift (race + stint) times tyre age,
plus lap noise, with the anchor level set from the noisy anchor laps exactly as the forecast sets it.
"""
import numpy as np
import pandas as pd

from boxbox.degradation.task import RACE, STINT


def stint_slopes(wear, laps, min_laps=8):
    """Per stint: slope of the wear model's residual on tyre age, its Sxx, and the lap residuals around the
    stint's own line. Only stints with at least min_laps clean laps (short stints give useless slopes)."""
    r = laps['LapTimeFC_s'] - wear.wear(laps) - wear.dirty_air(laps)
    keys = [laps[k] for k in STINT]
    x = laps['TyreLife'].astype(float)
    dx, dr = x - x.groupby(keys).transform('mean'), r - r.groupby(keys).transform('mean')
    g = pd.DataFrame({'xy': dx * dr, 'xx': dx * dx}).groupby(keys).sum()
    g['n'] = laps.groupby(STINT).size()
    g = g[(g['n'] >= min_laps) & (g['xx'] > 0)]
    g['b'] = g['xy'] / g['xx']
    b = pd.Series(laps.join(g['b'], on=STINT)['b'].to_numpy(), index=laps.index)
    e = (dr - b * dx)[b.notna()]
    return g.reset_index(), e


class WearNoise:
    def __init__(self, min_laps=8, race=True, stint=True, seed=0):
        self.min_laps, self.use_race, self.use_stint, self.seed = min_laps, race, stint, seed
        self.name = 'lap noise' + (', race' if race else '') + (', stint' if stint else '')

    def fit(self, wear, laps):
        g, e = stint_slopes(wear, laps, self.min_laps)
        n = g['n'].to_numpy(float)
        self.lap_resid = (e * np.sqrt(np.median(n) / (np.median(n) - 2))).to_numpy()   # undo the line's 2 df
        sigma2 = self.lap_resid.var()
        se2 = sigma2 / g['xx']
        b = g['b'] - g['b'].mean()
        # race spread: mean product of different stints' slopes in the same race
        s1, s2 = b.groupby([g[k] for k in RACE]).sum(), (b ** 2).groupby([g[k] for k in RACE]).sum()
        m = b.groupby([g[k] for k in RACE]).size()
        pairs = (m * (m - 1)).sum()
        tau_r2 = max(((s1 ** 2 - s2).sum()) / pairs, 0.0)
        tau_u2 = max(b.var() - se2.mean() - tau_r2, 0.0)
        self.tau_r, self.tau_u, self.sigma = np.sqrt(tau_r2), np.sqrt(tau_u2), np.sqrt(sigma2)
        return self

    def summary(self):
        return pd.Series({'race spread, s per lap': self.tau_r, 'stint spread, s per lap': self.tau_u,
                          'lap noise SD, s': self.sigma, 'lap noise q10': np.quantile(self.lap_resid, 0.1),
                          'lap noise q90': np.quantile(self.lap_resid, 0.9)})

    def draw_slopes(self, rng, draws, races):
        """Slope shifts, s per lap: draws x stints, races sharing their race draw."""
        out = np.zeros((draws, len(races)))
        if self.use_race:
            codes, uniq = pd.factorize(pd.Series([f'{y}_{r}' for y, r in races]))
            out += rng.normal(0, self.tau_r, (draws, len(uniq)))[:, codes]
        if self.use_stint:
            out += rng.normal(0, self.tau_u, (draws, len(races)))
        return out

    def draw_laps(self, rng, shape):
        return rng.choice(self.lap_resid, size=shape)


def simulate_errors(noise, stints, cfg, draws=1000, seed=0):
    """Simulated anchored forecast error sums (forecast minus actual, s) for each scored stint. stints:
    anchored_forecast rows (scored laps) with the stint's anchor ages in AnchorAges (a list). Returns the stint
    keys and the draws of each stint's error sum (draws x stints); stints of the same race share their race
    draw, so columns can be summed per driver race."""
    rng = np.random.default_rng(seed)
    meta = stints.groupby(STINT).agg(Ages=('TyreLife', list), Anchor=('AnchorAges', 'first')).reset_index()
    slopes = noise.draw_slopes(rng, draws, list(zip(meta['Year'], meta['Round'])))
    out = np.empty((draws, len(meta)))
    agg = cfg['degradation'].get('anchor_level', 'mean')
    for j, row in enumerate(meta.itertuples()):
        a, t = np.asarray(row.Anchor, float), np.asarray(row.Ages, float)
        d = slopes[:, [j]]
        ea, et = noise.draw_laps(rng, (draws, len(a))), noise.draw_laps(rng, (draws, len(t)))
        anchor = d * a + ea
        lvl = np.median(anchor, axis=1) if agg == 'median' else anchor.mean(axis=1)
        out[:, j] = (lvl[:, None] - d * t - et).sum(axis=1)
    return meta[STINT], out


def iid_errors(stints, oof, draws=1000, seed=0):
    """Comparison: each scored lap's error drawn independently from out of fold forecast errors in the same
    band of laps after the anchor (the conformal assumption applied lap by lap). Returns stint keys and the
    draws of each stint's error sum (draws x stints)."""
    from scipy import sparse
    from boxbox.degradation.intervals import ahead_band
    rng = np.random.default_rng(seed)
    pool = (oof['Pred_s'] - oof['LapTimeFC_s']).groupby(ahead_band(oof), observed=True).apply(np.asarray)
    band = ahead_band(stints).to_numpy()
    meta = stints[STINT].drop_duplicates().reset_index(drop=True)
    col = pd.MultiIndex.from_frame(stints[STINT]).map(pd.Series(range(len(meta)), index=pd.MultiIndex.from_frame(meta)))
    e = np.empty((draws, len(stints)))
    for b in pd.unique(band):
        on = np.flatnonzero(band == b)
        e[:, on] = rng.choice(pool[b], size=(draws, len(on)))
    m = sparse.csr_matrix((np.ones(len(stints)), (np.arange(len(stints)), np.asarray(col))), shape=(len(stints), len(meta)))
    return meta, np.asarray(m.T.dot(e.T).T)
