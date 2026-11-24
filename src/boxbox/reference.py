"""External reference tables in data/reference (notes/sim_v2_reference.md):

* pirelli_compounds.csv: the C numbers Pirelli nominated as Hard, Medium and Soft at every race 2022 to 2026
  (modelling D1 hedge; supersedes pirelli_2025_2026.csv, which is filled with the same values).
* circuit_ratings.csv: one row per race 2019 to 2026 with Pirelli's 1 to 5 track ratings (2022 to 2025), lap
  length, circuit type and street flag, turns, DRS zones, pit lane length and speed limit, and resurfacing.

Every row carries its source. Loaders return frames keyed by Year, Round and Circuit (the project's track key).
Derived quantities (an era comparable compound index, changes against a circuit's history, pit lane time at the
limit) live in features.py. The study helpers below serve scripts/tune_reference_features.py: per race and
compound wear shifts, a wear slope prior from circuit descriptors for circuits with no history, a pit loss prior
from the pit lane, and a pass logit with extra circuit columns. None of them is used by a model yet.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from boxbox import features
from boxbox.degradation.task import STINT
from boxbox.overtaking import PASS_FEATURES, PassLogit

RACE = ['Year', 'Round']
LABELS = ['HARD', 'MEDIUM', 'SOFT']
RATINGS = ['Traction', 'Braking', 'Lateral', 'TyreStress', 'AsphaltGrip', 'AsphaltAbrasion', 'TrackEvolution', 'Downforce']
RESURFACED = {'full': 2.0, 'partial': 1.0, 'minor': 0.0}   # minor: pit lane or grid only, not the racing line


def load_compounds(cfg):
    """pirelli_compounds.csv with HARD, MEDIUM and SOFT as C numbers (nullable integers: missing where Pirelli had
    not published the nomination)."""
    d = pd.read_csv(cfg['paths']['reference'] / cfg['reference']['compounds'], encoding='utf-8')
    for c in LABELS:
        d[c] = pd.to_numeric(d[c].astype('string').str[1:], errors='coerce').astype('Int64')
    return d


def compounds_long(d):
    """One row per race and compound label: Year, Round, Circuit, Compound (HARD, MEDIUM, SOFT), CNumber."""
    out = d.melt(id_vars=RACE + ['Circuit'], value_vars=LABELS, var_name='Compound', value_name='CNumber')
    return out.dropna(subset=['CNumber']).astype({'CNumber': float}).reset_index(drop=True)


def load_circuits(cfg):
    """circuit_ratings.csv: one row per race (Year, Round, Circuit) with ratings and facts; Street as 0 or 1 and
    ResurfacedLevel (2 full, 1 partial, 0 otherwise) added."""
    d = pd.read_csv(cfg['paths']['reference'] / cfg['reference']['circuits'], encoding='utf-8')
    d['ResurfacedLevel'] = d['Resurfaced'].map(RESURFACED).fillna(0.0)
    return d


def load_strategies(cfg):
    """pirelli_strategies.csv: Pirelli's predicted fastest strategy per race 2022 to 2025 from its race eve release
    (Stops, Sequence of S M H, Alternative, Source, Notes; blank where the release names no fastest plan). A benchmark
    for the strategy search, not a model input."""
    return pd.read_csv(cfg['paths']['reference'] / cfg['reference']['strategies'], encoding='utf-8')


# ---------------------------------------------------------------- degradation studies (modelling O7)

def race_compound_shifts(base, laps, min_stints=5, n_boot=300, seed=0):
    """Each race and compound's slope shift against a fitted wear model (the slope of the within stint residual on
    tyre age, s per lap per lap, not shrunk), with its sampling SD from resampling whole stints. The per compound
    version of degradation.race_shift_se, for asking whether the compound behind a label explains the shift. Cells
    with fewer than min_stints stints are dropped: one or two stints give shifts of up to 1.8 s per lap per lap."""
    x = base.f(laps['TyreLife'].clip(lower=1))
    keys = [laps[c] for c in STINT]
    dx = x - x.groupby(keys).transform('mean')
    r = laps['LapTimeFC_s'] - base.wear(laps) - base.dirty_air(laps)
    s = pd.DataFrame({'xy': dx * (r - r.groupby(keys).transform('mean')), 'xx': dx ** 2, 'Compound': laps['Compound']})
    s = s.groupby(keys).agg(xy=('xy', 'sum'), xx=('xx', 'sum'), Compound=('Compound', 'first')).reset_index()
    rng, out = np.random.default_rng(seed), []
    for (y, rd, comp), g in s.groupby(RACE + ['Compound']):
        xy, xx = g['xy'].to_numpy(), g['xx'].to_numpy()
        if xx.sum() <= 0 or len(g) < min_stints:
            continue
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        boot = xy[idx].sum(1) / np.where(xx[idx].sum(1) > 0, xx[idx].sum(1), np.nan)
        out.append({'Year': y, 'Round': rd, 'Compound': comp, 'Shift': xy.sum() / xx.sum(), 'SE': np.nanstd(boot), 'Stints': len(g)})
    return pd.DataFrame(out)


class SlopeTable:
    """A fitted linear wear model whose slope is replaced per race and compound from a table (Series indexed by
    Year, Round, Compound; missing entries keep the base slope). Offsets (race start, curvature) and dirty air
    stay the base model's. Used to score a slope prior or a slope adjustment on the anchored forecast."""

    def __init__(self, base, slopes):
        self.base, self.slopes = base, slopes.rename('TableSlope')

    def _slope(self, laps):
        s = laps.join(self.slopes, on=RACE + ['Compound'])['TableSlope']
        return s.fillna(self.base.slopes_for(laps))

    def wear(self, laps):
        x = self.base.f(laps['TyreLife'].clip(lower=1))
        other = self.base.wear(laps) - self.base.slopes_for(laps) * x
        return self._slope(laps) * x + other

    def dirty_air(self, laps):
        return self.base.dirty_air(laps)


class RatedSlopes:
    """Wear slope per circuit and compound from circuit descriptors, for a circuit with no dry history (the
    fallback is the compound slope across all circuits, modelling D1 cost). Weighted ridge across the training
    circuits' raw slopes: slope = compound mean + sum of beta_j x_j, x the descriptors standardised on the
    training cells (ratings, the era index of the C number behind the label), weights the cell's stints capped at
    max_weight, alpha the ridge penalty on beta (the compound means are not penalised)."""

    def __init__(self, feats, alpha=10.0, max_weight=50):
        self.feats, self.alpha, self.max_weight = list(feats), alpha, max_weight
        self.name = 'compound mean' if not self.feats else f"rated slopes ({', '.join(self.feats)}), alpha {alpha}"

    def _design(self, X):
        comp = pd.get_dummies(X['Compound']).reindex(columns=LABELS, fill_value=False).astype(float).to_numpy()
        z = ((X[self.feats] - self.mu) / self.sd).fillna(0.0).to_numpy(float) if self.feats else np.zeros((len(X), 0))
        return comp, z

    def fit(self, cells):
        """cells: one row per training circuit and compound with Compound, Raw (slope), Stints and the descriptors."""
        c = cells.dropna(subset=['Raw'] + self.feats)
        self.mu, self.sd = c[self.feats].mean(), c[self.feats].std(ddof=0).replace(0, 1.0)
        comp, z = self._design(c)
        w = np.minimum(c['Stints'].to_numpy(float), self.max_weight)
        A = np.hstack([comp, z]) * np.sqrt(w)[:, None]
        y = c['Raw'].to_numpy(float) * np.sqrt(w)
        pen = np.hstack([np.zeros((z.shape[1], comp.shape[1])), np.sqrt(self.alpha) * np.eye(z.shape[1])])
        b = np.linalg.lstsq(np.vstack([A, pen]), np.concatenate([y, np.zeros(z.shape[1])]), rcond=None)[0]
        self.comp_mean, self.beta = dict(zip(LABELS, b[:3])), dict(zip(self.feats, b[3:]))
        return self

    def predict(self, X):
        comp, z = self._design(X)
        return comp @ np.array([self.comp_mean[k] for k in LABELS]) + z @ np.array([self.beta[f] for f in self.feats])


# ---------------------------------------------------------------- pit loss studies (modelling D16)

class PitLanePrior:
    """Median green flag pit loss of a circuit from its pit lane: a + b x pit lane time at the limit
    (features.pit_lane_time), least squares across the training circuits' medians. The answer for a circuit with
    no stops; b near 1 means the rest of a stop (deceleration, the stop itself, rejoining) is the same everywhere."""
    name = 'pit lane prior'

    def fit(self, medians, lane_time):
        """medians: Series by Circuit (s). lane_time: Series by Circuit (s at the limit)."""
        d = pd.DataFrame({'y': medians, 'x': lane_time}).dropna()
        self.n = len(d)
        if self.n < 3:   # too few circuits for a slope: their mean
            self.b, self.a = 0.0, float(d['y'].mean()) if self.n else float(medians.median())
        else:
            self.b, self.a = np.polyfit(d['x'], d['y'], 1)
        return self

    def predict(self, lane_time):
        return self.a + self.b * np.asarray(lane_time, float)


def circuit_lengths(facts):
    """Pit lane length per circuit (median over the seasons that state one): the table states a length only for
    the seasons a source gives it, and a pit lane is a fixed structure unless PitLaneChange says otherwise."""
    return facts.dropna(subset=['PitLaneLength_m']).groupby('Circuit')['PitLaneLength_m'].median()


def limit_corrected_stops(stops, facts, target_year, scale=1.0):
    """Stops moved to the target season's pit lane: each stop's loss plus scale x (pit lane time at the target
    season's limit less at the limit of the season the stop was made, at the circuit's pit lane length,
    circuit_lengths). A changed speed limit (Marina Bay and Zandvoort 2025) then moves the circuit's median in its
    first season. facts: Circuit, Year, PitLaneLength_m, PitSpeedLimit_kmh; stops with no limit for either season
    or no length for the circuit are left as they are."""
    f = facts.dropna(subset=['PitSpeedLimit_kmh']).drop_duplicates(['Circuit', 'Year'])
    f = f.assign(L=f['Circuit'].map(circuit_lengths(facts))).dropna(subset=['L'])
    f = f.assign(T=features.pit_lane_time(f['L'], f['PitSpeedLimit_kmh']))[['Circuit', 'Year', 'T']]
    target = f[f['Year'] == target_year].set_index('Circuit')['T']
    s = stops.merge(f, on=['Circuit', 'Year'], how='left')
    delta = (s['Circuit'].map(target) - s['T']).fillna(0.0)
    return s.drop(columns='T').assign(PitLoss_s=s['PitLoss_s'] + scale * delta.to_numpy(), LimitShift_s=scale * delta.to_numpy())


# ---------------------------------------------------------------- overtaking studies

class PassLogitExtra(PassLogit):
    """overtaking.PassLogit with extra numeric pair columns appended to its features (for example the race's DRS
    zones or the street flag, joined onto the pairs by the study). Same circuit logit and ridge; extra=() gives
    the parent model's fit."""

    def __init__(self, C=1.0, k=200, feats=PASS_FEATURES, extra=()):
        super().__init__(C, k, feats)
        self.extra = list(extra)
        self.name += ''.join(f' + {e}' for e in self.extra)

    def _x(self, p):
        x = super()._x(p)
        return np.column_stack([x] + [p[c].to_numpy(float) for c in self.extra]) if self.extra else x

    def fit(self, p):
        self.rate = float(p['Pass'].mean())
        g = p.groupby('Circuit')['Pass'].agg(['sum', 'size'])
        r = (g['sum'] + self.k * self.rate) / (g['size'] + self.k)
        self.circuit_logit = np.log(r / (1 - r))
        self.all_logit = float(np.log(self.rate / (1 - self.rate)))
        m = LogisticRegression(C=self.C, max_iter=2000).fit(self._x(p), p['Pass'].to_numpy(bool))
        self.coef = dict(zip(self.feats + self.extra, m.coef_[0]))
        self.intercept = float(m.intercept_[0])
        return self

    def prob(self, p):
        z = self.intercept + self._x(p) @ np.array([self.coef[f] for f in self.feats + self.extra])
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    predict = prob
