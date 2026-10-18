"""Tyre degradation: seconds lost to tyre age, per circuit and compound (modelling D6).

Wear is slope x f(tyre age), where f is linear, square root or log (O8). Every stint gets its own
intercept (car, driver, conditions on the day), so the slope comes only from how fuel corrected lap time
changes with tyre age inside a stint. Track evolution and any error in the fuel correction land in the
slope too, so it is an effective rate, which is what the simulator needs as long as it applies the same
fuel correction.

Circuit slopes are shrunk toward the compound's slope across all circuits, weighted by how many stints
back the circuit. A circuit or compound with no history falls back to the compound slope.

Optional: recency weighting (each older season counts decay times the next one), a stint length term
(the slope changes by a per compound amount per lap of stint length away from the reference length), and lap
level offsets shared by all circuits: dirty air (seconds lost per band of gap to the car ahead) and the race
start (race laps 2 onward run slow while the field settles). Both are fitted jointly with the slopes.
Teams also pit early on worn tyres, but the effect holds on stints ended by a safety car or the finish,
so it is mostly planned tyre management (experiments.md, 2026-10-14). Keep stint lengths in the observed range.
"""
import numpy as np
import pandas as pd

from boxbox import evaluate, features
from boxbox.degradation.task import AGE_TERMS, KEY, STINT, ConstantPace, anchored_forecast, model_laps


def flat_frame(cols, index):
    """DataFrame from {tuple key: column}, keeping the tuples as a flat column index (keys of different
    lengths would otherwise become a MultiIndex)."""
    d = pd.DataFrame(dict(enumerate(cols.values())), index=index, dtype=float)
    d.columns = pd.Index(list(cols), tupleize_cols=False)
    return d


class LinearDegradation:
    """Lap time = stint level + slope x f(tyre age), slope per circuit and compound.

    by_circuit=False uses the compound slope everywhere, which is the model's answer for a new circuit.
    decay < 1 weights recent seasons more. stint_length=True lets the slope vary with stint length.
    """

    def __init__(self, shrink_stints=10, by_circuit=True, age_term='linear', decay=1.0, stint_length=False,
                 length_ref=20, dirty_air_bins=None, race_start_laps=0, passes=50):
        self.k, self.by_circuit, self.age_term, self.f = shrink_stints, by_circuit, age_term, AGE_TERMS[age_term]
        self.decay, self.stint_length, self.length_ref = decay, stint_length, length_ref
        self.bins, self.start_laps, self.passes = list(dirty_air_bins or []), race_start_laps, passes
        self.name = ((f'{age_term}, shrink {shrink_stints}' if by_circuit else f'{age_term}, compound only')
                     + (f', decay {decay}' if decay != 1 else '') + (', stint length' if stint_length else '')
                     + (', dirty air' if self.bins else '') + (', race start' if race_start_laps > 1 else ''))

    def _z(self, laps):
        if self.stint_length:
            return laps['StintLaps'] - self.length_ref
        return pd.Series(0.0, index=laps.index)

    def _design(self, laps):
        """0 or 1 per lap level offset: one column per dirty air band (from GapAhead_s) and one per race lap 2
        to race_start_laps (from RaceLap). A lap without the column gets none of these offsets."""
        none = pd.Series(np.nan, index=laps.index)
        cols = {}
        if self.bins:
            band = features.dirty_air_bin(laps.get('GapAhead_s', none), self.bins)
            cols.update({('DirtyAir', self.bins[i]): (band == i).astype(float) for i in range(len(self.bins) - 1)})
        race_lap = laps.get('RaceLap', none)
        cols.update({('RaceStart', n): (race_lap == n).astype(float) for n in range(2, self.start_laps + 1)})
        return flat_frame(cols, laps.index)

    def _penalty(self, columns):
        """Rows R of a roughness penalty on the offsets (the fit minimises the squared residual plus |R b|^2).
        None here; smooth terms in subclasses add them."""
        return np.zeros((0, len(columns)))

    def _offset(self, laps, terms, exclude=False):
        """Sum of the fitted offsets whose term is in terms (or, with exclude, not in terms)."""
        cols = [c for c in self.offsets.index if (c[0] in terms) != exclude]
        if not cols:
            return pd.Series(0.0, index=laps.index)
        return self._design(laps)[cols] @ self.offsets[cols]

    def fit(self, laps):
        """Slopes (and the stint length term) from within stint variation, then the lap level offsets by least
        squares on the within stint residual, in turn (backfitting) until the offsets move less than 1e-10 s or
        after `passes` rounds."""
        w = self.decay ** (laps['Year'].max() - laps['Year'])
        d = self._design(laps)
        self.offsets = pd.Series(0.0, index=d.columns, dtype=float)
        if len(d.columns):
            keys = [laps[k] for k in STINT]
            pen = self._penalty(d.columns)
            dd = np.vstack([(d - d.groupby(keys).transform('mean')).mul(np.sqrt(w), axis=0).to_numpy(), pen])
            x = self.f(laps['TyreLife'].clip(lower=1))
            for _ in range(self.passes):
                self._fit_slopes(laps, laps['LapTimeFC_s'] - d @ self.offsets, w)
                r = laps['LapTimeFC_s'] - self.slopes_for(laps) * x
                r = (r - r.groupby(keys).transform('mean')) * np.sqrt(w)
                rhs = np.concatenate([r.to_numpy(), np.zeros(len(pen))])
                new = pd.Series(np.linalg.lstsq(dd, rhs, rcond=None)[0], index=d.columns)
                done = (new - self.offsets).abs().max() < 1e-10
                self.offsets = new
                if done:
                    break
        self._fit_slopes(laps, laps['LapTimeFC_s'] - d @ self.offsets, w)
        return self

    def _fit_slopes(self, laps, y, w):
        x = self.f(laps['TyreLife'].clip(lower=1))
        dx = x - x.groupby([laps[k] for k in STINT]).transform('mean')
        dy = y - y.groupby([laps[k] for k in STINT]).transform('mean')
        z = self._z(laps)
        cells, comp = [laps['Circuit'], laps['Compound']], laps['Compound']

        # Stint length term, shared per compound, solved exactly by partialling out the circuit slopes
        # (Frisch Waugh): regress dy on the part of dx*z that the per circuit slopes cannot explain.
        sxx = (w * dx ** 2).groupby(cells).transform('sum')
        if self.stint_length:
            u = dx * z
            u_t = u - dx * (w * dx * u).groupby(cells).transform('sum') / sxx
            self.length_coef = (w * u_t * dy).groupby(comp).sum() / (w * u_t ** 2).groupby(comp).sum()
        else:
            self.length_coef = pd.Series(0.0, index=comp.unique())
        adj = dy - comp.map(self.length_coef) * dx * z
        raw = (w * dx * adj).groupby(cells).sum() / (w * dx ** 2).groupby(cells).sum()
        pooled = (w * dx * adj).groupby(comp).sum() / (w * dx ** 2).groupby(comp).sum()

        self.pooled = pooled.rename('Pooled')
        first = laps.drop_duplicates(STINT)
        cell = raw.rename('Raw').to_frame()
        cell['Stints'] = (self.decay ** (laps['Year'].max() - first['Year'])).groupby(
            [first['Circuit'], first['Compound']]).sum()
        cell = cell.join(self.pooled, on='Compound')
        cell['Slope'] = (cell['Stints'] * cell['Raw'] + self.k * cell['Pooled']) / (cell['Stints'] + self.k)
        if not self.by_circuit:
            cell['Slope'] = cell['Pooled']

        slope = laps.join(cell['Slope'], on=KEY)['Slope'] + comp.map(self.length_coef) * z
        cell['ResidSD_s'] = (dy - dx * slope).groupby(cells).std()
        cell['Laps'] = laps.groupby(KEY).size()
        cell['LengthCoef'] = cell.index.get_level_values('Compound').map(self.length_coef)
        cell['AgeTerm'] = self.age_term
        self.cells = cell

    def slopes_for(self, laps):
        s = laps.join(self.cells['Slope'], on=KEY)['Slope']
        s = s.fillna(laps['Compound'].map(self.pooled)).fillna(0.0)
        return s + laps['Compound'].map(self.length_coef).fillna(0.0) * self._z(laps)

    def wear(self, laps):
        """Seconds of wear at each lap's tyre age, up to a constant per stint, plus every offset but dirty air
        (the race start, and curvature in subclasses)."""
        return self.slopes_for(laps) * self.f(laps['TyreLife'].clip(lower=1)) + self._offset(laps, ['DirtyAir'], exclude=True)

    def dirty_air(self, laps):
        """Seconds each lap lost to the car ahead, from its gap. Kept out of wear: a forecast lap's gap is not
        known in advance, so forecasts are for clean air."""
        return self._offset(laps, ['DirtyAir'])

    def export(self):
        """Fitted parts as tables, for data/models: circuit and compound slopes, and the lap level offsets."""
        off = pd.DataFrame([(c[0], '/'.join(map(str, c[1:])), v) for c, v in self.offsets.items()],
                           columns=['Term', 'Key', 'Offset_s'])
        return {'cells': self.cells.reset_index(), 'offsets': off}

    def loss_after(self, laps_on_tyre):
        """Seconds lost per compound from tyre age 1 to the given age, from the compound slopes
        (for a stint of that length when the stint length term is on)."""
        c = self.length_coef * (laps_on_tyre - self.length_ref) if self.stint_length else 0.0
        return (self.pooled + c) * (self.f(laps_on_tyre) - self.f(1))


def candidates(cfg):
    c = cfg['degradation']
    models = [ConstantPace()]
    for t in c['age_terms']:
        models += [LinearDegradation(by_circuit=False, age_term=t)]
        for d in c['decay_grid']:
            for length in c['stint_length_options']:
                for lap_terms in c.get('lap_terms_options', [False]):
                    models += [LinearDegradation(k, age_term=t, decay=d, stint_length=length,
                                                 length_ref=c['stint_length_ref'], **lap_terms_kw(cfg, lap_terms))
                               for k in c['shrink_grid']]
    return models


def lap_terms_kw(cfg, on=True):
    """LinearDegradation arguments for the dirty air and race start offsets from config (off: none)."""
    c = cfg['degradation']
    return {'dirty_air_bins': c['dirty_air_bins'], 'race_start_laps': c['race_start_laps']} if on else {}


def score(laps, cfg, train_years, test_year, models):
    train, test = model_laps(laps, train_years, cfg, train=True), model_laps(laps, [test_year], cfg)
    return {m.name: anchored_forecast(m.fit(train), test, cfg).assign(Fold=test_year) for m in models}


def validate(laps, cfg, seasons):
    """Every candidate on the validation folds. Returns (validation table, chosen name)."""
    folds = [score(laps, cfg, tr, y, candidates(cfg)) for tr, y in evaluate.season_splits(seasons)['validation']]
    val_preds = {n: pd.concat([f[n] for f in folds]) for n in folds[0]}
    val = pd.concat({n: evaluate.error_table(p) for n, p in val_preds.items()}).droplevel(1)
    for f in folds:
        y = next(iter(f.values()))['Fold'].iloc[0]
        val[f'MAE_{y}'] = pd.Series({n: p['Err_s'].abs().mean() for n, p in f.items()})
    return val, val.drop(index='constant pace')['MAE_s'].idxmin()


def run(laps, cfg, seasons, reference=None):
    """Choose a candidate on the validation folds, then score only it, constant pace and an optional
    reference model on the test season (so the test season is not used for selection). The chosen model is
    refit on all seasons.

    Returns (validation table, chosen name, test predictions per model, test table, final model)."""
    split = evaluate.season_splits(seasons)
    val, best = validate(laps, cfg, seasons)
    fresh = {m.name: m for m in candidates(cfg)}
    picks = [fresh['constant pace'], fresh[best]] + [fresh[r] for r in [reference] if r and r != best]
    preds = score(laps, cfg, *split['test'], picks)
    test = pd.concat({n: evaluate.error_table(p) for n, p in preds.items()}).droplevel(1)
    final = {m.name: m for m in candidates(cfg)}[best].fit(model_laps(laps, seasons, cfg, train=True))
    return val, best, preds, test, final
