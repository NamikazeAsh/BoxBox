"""Rival pit decisions for the strategy simulator (pit_decision v1). The simulator races one car on a chosen
strategy; every rival is driven by PlanPolicy, plan first: before the race, each rival draws a plan (1, 2 or 3+
tyre stops) from the plan model; each lap, the chance it pits at the end of the lap (a discrete time hazard given
the planned stops left); and when it does, which dry tyre it fits. A per lap hazard with no plan (RivalPolicy)
was the first version: drivers in the same lap state follow different plans, so it averaged them and its replays
made too many 3+ stop races. The plan is drawn once per car, so a one stopper stays one.

At the start the simulator calls PlanPolicy.sample_plan on each rival's lap 1 state and stores the result as the
car's Plan. Then it hands the policy one row per car per lap, the car's state on that lap (STATE plus Plan):

    Year, Round, Driver, Circuit, LapNumber, RaceLaps, Compound, TyreLife (age on this lap), FreshTyre, Stint,
    UsedSoft, UsedMedium, UsedHard (dry compounds run so far, the current one included),
    Grid, QualiGap, TeamPaceForm, DriverPaceForm (pre race, from outcome.entries)

The policy computes every feature from that state (add_features), so training and the simulator see the same
features. Wear comes from the simulator's own wear model and pit loss from pitloss.PitLoss, both fitted on the
training seasons only. v1 is dry races, green flag laps and tyre change stops (features.pit_labels); gap and
position features wait for an overtaking model, safety cars for v2.

Splits are by whole season, rolling origin (evaluate.season_splits). Models are chosen on validation log loss.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer, StandardScaler

from boxbox import features, outcome
from boxbox.degradation import adjust, compare, model_laps
from boxbox.degradation.monotone import MonotoneWear
from boxbox.degradation.task import KEY
from boxbox.pitloss import PitLoss

RACE = ['Year', 'Round']
DRIVER_RACE = RACE + ['Driver']
DRY = ['SOFT', 'MEDIUM', 'HARD']
USED = {'SOFT': 'UsedSoft', 'MEDIUM': 'UsedMedium', 'HARD': 'UsedHard'}
PRE_RACE = ['Grid', 'QualiGap', 'TeamPaceForm', 'DriverPaceForm']
STATE = DRIVER_RACE + ['Circuit', 'LapNumber', 'RaceLaps', 'Compound', 'TyreLife', 'FreshTyre', 'Stint'] + list(USED.values()) + PRE_RACE

# Feature groups, in the order the ablation adds them
GROUPS = {'tyre': ['IsSoft', 'IsHard', 'TyreLife', 'FreshTyre', 'Stint'],
          'two compound rule': ['TwoCompounds'],
          'race': ['LapsLeft', 'RaceFrac', 'RaceLaps'],
          'wear': ['WearLoss', 'WearRate', 'CircuitWear'],
          'pit loss': ['PitLoss'],
          'pre race': PRE_RACE,
          'fastest lap point': ['FastestLapPoint']}
FEATURES = [f for g in GROUPS.values() for f in g]
SPLINE = ['TyreLife', 'LapsLeft', 'RaceFrac']
# Plan first (option 2): a stop count drawn per car before the race, then timing given the planned stops left
PLANS = [1, 2, 3]   # 3 is three or more
NEUTRAL = ['SafetyCar', 'VSC', 'RedFlag']   # the race had such a lap: training only, 0 in the v1 simulator
NEUTRAL_CODES = {'SafetyCar': '4', 'VSC': '[67]', 'RedFlag': '5'}
PLAN_TIMING = ['StopsLeft', 'StintTarget', 'AgeShare']
TIMING_FEATURES = FEATURES + PLAN_TIMING
TIMING_SPLINE = SPLINE + ['AgeShare']
PLAN_FEATURES = ['IsSoft', 'IsHard', 'FreshTyre', 'RaceLaps', 'SoftRate', 'MediumRate', 'HardRate', 'CircuitWear', 'PitLoss',
                 'CircuitStops', 'FastestLapPoint'] + NEUTRAL + PRE_RACE
TYRE_FEATURES = ['IsSoft', 'IsHard', 'UsedSoft', 'UsedMedium', 'UsedHard', 'TwoCompounds', 'Stint', 'LapsLeft', 'RaceFrac',
                 'SoftRate', 'MediumRate', 'HardRate']


# Data

def pre_race(results, qualifying, sprint_results, events, laps, cfg):
    """Pre race features per driver and race, from the outcome predictor's entries (no lap of the race used)."""
    return outcome.entries(results, qualifying, sprint_results, events, cfg, laps)[DRIVER_RACE + PRE_RACE]


def pit_laps(laps, pre, cfg):
    """One row per decision lap (features.pit_labels PitRow): the STATE columns, the label PitStop and, for
    stops, NextCompound (the tyre fitted). Used flags count every lap so far, decision lap or not."""
    l = laps.join(features.pit_labels(laps, cfg)).sort_values(DRIVER_RACE + ['LapNumber'])
    by = [l['Year'], l['Round'], l['Driver']]
    l['RaceLaps'] = l.groupby(RACE)['LapNumber'].transform('max')
    l['NextCompound'] = l.groupby(by)['Compound'].shift(-1)
    for comp, col in USED.items():
        l[col] = (l['Compound'] == comp).groupby(by).cummax()
    rows = l[l['PitRow']].merge(pre, on=DRIVER_RACE, how='left')
    return rows[STATE + ['PitStop', 'NextCompound']].reset_index(drop=True)


def stops_with_circuit(stops, events):
    """features.pit_loss stops with the circuit key PitLoss needs."""
    return stops.merge(events[RACE + ['Circuit']], on=RACE, how='left')


class Context:
    """What the features need from the training seasons: the simulator's wear curve (simulator.wear_model, monotone
    if simulator.monotone_wear) per circuit, compound and tyre age, and the pit loss per circuit. A circuit with no
    curve uses the compound's mean curve over circuits."""

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, laps, stops, years, circuits, weekend=None):
        """weekend: optional (sprint_laps, practice_laps, race years) to also store each of those races' weekend
        wear shift (weekend_wear_shifts), used by wear_features(shift=True)."""
        s = self.cfg['simulator']
        self.max_age = s['max_age']
        wear = compare.ladder(self.cfg)[s['wear_model']]()
        if s.get('monotone_wear', True):
            wear = MonotoneWear(wear, self.max_age)
        wear.fit(model_laps(laps, years, self.cfg, train=True))
        self.shift = pd.Series(dtype=float)
        if weekend is not None:
            base = wear.base if isinstance(wear, MonotoneWear) else wear
            self.shift = weekend_wear_shifts(base, laps, *weekend[:2], self.cfg, years, weekend[2])
        g = pd.DataFrame([(c, k) for c in sorted(set(circuits)) for k in DRY], columns=KEY)
        g = g.merge(pd.DataFrame({'TyreLife': np.arange(1, self.max_age + 1, dtype=float)}), how='cross')
        g = g.assign(StintLaps=20, RaceLap=np.nan, GapAhead_s=np.nan).sort_values(KEY + ['TyreLife'])
        w = pd.Series(wear.wear(g).to_numpy(), index=g.index)
        g['Wear_s'] = w - w.groupby([g['Circuit'], g['Compound']]).transform('first')   # loss against age 1
        self.wear = g.set_index(KEY + ['TyreLife'])['Wear_s']
        self.fallback = g.groupby(['Compound', 'TyreLife'])['Wear_s'].mean()
        p = s['pit_loss']
        self.pit = PitLoss(p['recent'], p['min_stops']).fit(stops[stops['Year'].isin(years)])
        self.cfg = None   # keep the fitted context small and free of paths
        return self

    def wear_at(self, circuit, compound, age):
        age = np.clip(np.round(np.asarray(age, dtype=float)), 1, self.max_age)
        w = self.wear.reindex(pd.MultiIndex.from_arrays([circuit, compound, age])).to_numpy().copy()
        miss = np.isnan(w)
        if miss.any():
            w[miss] = self.fallback.reindex(pd.MultiIndex.from_arrays([np.asarray(compound)[miss], age[miss]])).to_numpy()
        return w

    def race_shift(self, d):
        """The stored weekend wear shift of each row's race, s per lap (0 where none)."""
        if not len(getattr(self, 'shift', [])):
            return np.zeros(len(d))
        idx = pd.MultiIndex.from_arrays([d['Year'], d['Round']])
        return np.nan_to_num(self.shift.reindex(idx).to_numpy(dtype=float))

    def wear_features(self, d, shift=False):
        """Wear so far against a fresh tyre, wear rate now, and each compound's rate at the circuit. shift=True
        moves every rate by the race's weekend shift, as degradation.WeekendShifted does (rates kept at 0 or above)."""
        c, k, a = d['Circuit'].to_numpy(), d['Compound'].to_numpy(), d['TyreLife'].to_numpy(float)
        s = self.race_shift(d) if shift else np.zeros(len(d))
        age = np.clip(np.round(a), 1, self.max_age)
        now = self.wear_at(c, k, a) + s * (age - 1)
        nxt = self.wear_at(c, k, a + 1) + s * (np.clip(age + 1, 1, self.max_age) - 1)
        out = {'WearLoss': np.maximum(now, 0.0), 'WearRate': np.maximum(nxt - now, 0.0)}
        for comp in DRY:
            kk = np.full(len(d), comp, dtype=object)
            rate = (self.wear_at(c, kk, np.full(len(d), 11.0)) - self.wear_at(c, kk, np.ones(len(d)))) / 10 + s
            out[f'{comp.title()}Rate'] = np.maximum(rate, 0.0)
        out = pd.DataFrame(out, index=d.index)
        out['CircuitWear'] = out[['SoftRate', 'MediumRate', 'HardRate']].mean(axis=1)
        return out


def weekend_wear_shifts(base, laps, sprint_laps, prac, cfg, train_years, years):
    """Wear slope shift per (Year, Round) of years, s per lap, by the degradation model's weekend rule
    (adjust.weekend_forecast): the dry sprint's shift, else practice long runs at circuits with no dry race history
    in train_years, else 0, with the configured shrinks. base is a linear family model fitted on train_years."""
    w = cfg['degradation']['weekend_adjust']
    years = sorted(set(years))
    runs = adjust.model_runs(prac, laps, sorted(set(train_years) | set(years)), cfg, cfg['long_run']['min_laps'])
    offset = adjust.practice_offset(base, runs[runs['Year'].isin(train_years)])
    t = adjust.weekend_table(base, model_laps(laps, years, cfg), model_laps(sprint_laps, years, cfg),
                             runs[runs['Year'].isin(years)], offset, cfg)
    return adjust.weekend_variants(t, [w['practice_shrink']], w['practice_max_history'])[
        f"sprint, else practice fallback, k {w['practice_shrink']}"].rename('Shift')


def add_features(state, ctx, weekend=False):
    """Every model feature from a STATE frame (training rows or the simulator's cars). weekend=True uses the
    weekend shifted wear (Context.fit with weekend data)."""
    d = state.copy()
    d['IsSoft'] = (d['Compound'] == 'SOFT').astype(float)
    d['IsHard'] = (d['Compound'] == 'HARD').astype(float)
    d['FreshTyre'] = d['FreshTyre'].astype(float)
    d['TwoCompounds'] = (d[list(USED.values())].astype(int).sum(axis=1) >= 2).astype(float)
    d['LapsLeft'] = d['RaceLaps'] - d['LapNumber']
    d['RaceFrac'] = d['LapNumber'] / d['RaceLaps']
    d['FastestLapPoint'] = (d['Year'] <= 2024).astype(float)   # a point for fastest lap, 2019 to 2024
    d = d.join(ctx.wear_features(d, shift=weekend))
    d['PitLoss'] = ctx.pit.predict(d['Circuit'].to_numpy())
    if 'Plan' in d:   # the car's planned stop count: stops left, and the stint length that plan implies
        d['StopsLeft'] = (d['Plan'] - (d['Stint'] - 1)).clip(lower=0)
        d['StintTarget'] = (d['TyreLife'] + d['LapsLeft']) / (d['StopsLeft'] + 1)
        d['AgeShare'] = d['TyreLife'] / d['StintTarget']
    return d


def shift_logit(p, delta):
    """p moved by delta on the log odds scale."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + delta)))


def season_offsets(p, d, k):
    """Season correction per race of d: log((A + k) / (E + k)), A the real tyre stops and E the model's expected
    stops (sum of p) over the decision laps of the same season's earlier races, so k pseudo stops pull it toward
    0. 0 at a season's first race. Uses only races before the one it corrects."""
    g = pd.DataFrame({'A': d['PitStop'].astype(float).to_numpy(), 'E': p}, index=pd.MultiIndex.from_frame(d[RACE])).groupby(RACE).sum()
    before = g.groupby(level='Year').cumsum() - g
    return np.log((before['A'] + k) / (before['E'] + k)).replace([np.inf, -np.inf], np.nan).fillna(0.0).rename('Offset')


def offsets_for(d, offsets):
    """Each row's race offset (0 where none)."""
    if offsets is None or not len(offsets):
        return np.zeros(len(d))
    return np.nan_to_num(offsets.reindex(pd.MultiIndex.from_arrays([d['Year'], d['Round']])).to_numpy(dtype=float))


# Pit hazard models: fit(rows) and prob(rows), rows from add_features

class ConstantRate:
    """Baseline: the training pit rate on every lap."""
    name = 'constant rate'

    def fit(self, d):
        self.p = d['PitStop'].mean()
        return self

    def prob(self, d):
        return np.full(len(d), self.p)


class AgeRate:
    """Baseline: pit rate per compound and tyre age bin, pulled toward the compound's rate with weight k laps."""
    name = 'rate by compound and tyre age'

    def __init__(self, k=50, age_bin=3):
        self.k, self.bin = k, age_bin

    def _key(self, d):
        return pd.MultiIndex.from_arrays([d['Compound'], (d['TyreLife'] // self.bin).to_numpy()])

    def fit(self, d):
        y = d['PitStop'].astype(float)
        self.all, self.comp = y.mean(), y.groupby(d['Compound']).mean()
        g = y.groupby(self._key(d)).agg(['sum', 'size'])
        prior = self.comp.reindex(g.index.get_level_values(0)).to_numpy()
        self.rate = (g['sum'] + self.k * prior) / (g['size'] + self.k)
        return self

    def prob(self, d):
        p = pd.Series(self.rate.reindex(self._key(d)).to_numpy(), index=d.index)
        return p.fillna(d['Compound'].map(self.comp)).fillna(self.all).to_numpy()


class LogitHazard:
    """Ridge logistic regression; cubic splines on tyre age, laps left and race fraction, the rest linear."""

    def __init__(self, C, knots=6, feats=FEATURES, spline=SPLINE):
        self.C, self.knots, self.feats, self.spline = C, knots, list(feats), list(spline)
        self.name = f'logit, C {C}'

    def fit(self, d):
        spl = [f for f in self.spline if f in self.feats]
        lin = [f for f in self.feats if f not in spl]
        imp = lambda: SimpleImputer(strategy='median', keep_empty_features=True)
        parts = [('spline', make_pipeline(imp(), SplineTransformer(n_knots=self.knots), StandardScaler()), spl)] if spl else []
        parts += [('linear', make_pipeline(imp(), StandardScaler()), lin)] if lin else []
        self.m = make_pipeline(ColumnTransformer(parts), LogisticRegression(C=self.C, max_iter=5000))
        self.m.fit(d[self.feats].astype(float), d['PitStop'].astype(int))
        return self

    def prob(self, d):
        return self.m.predict_proba(d[self.feats].astype(float))[:, 1]

    def with_features(self, feats):
        return LogitHazard(self.C, self.knots, feats, self.spline)


class LGBMHazard:
    """Small gradient boosted trees on the log loss."""

    def __init__(self, num_leaves, n_estimators, min_child_samples, learning_rate, feats=FEATURES, seed=0):
        self.kw = dict(num_leaves=num_leaves, n_estimators=n_estimators, min_child_samples=min_child_samples,
                       learning_rate=learning_rate)
        self.feats, self.seed = list(feats), seed
        self.name = f'lgbm, leaves {num_leaves}, trees {n_estimators}'

    def fit(self, d):
        self.m = lgb.LGBMClassifier(**self.kw, random_state=self.seed, verbose=-1)
        self.m.fit(d[self.feats].astype(float), d['PitStop'].astype(int))
        return self

    def prob(self, d):
        return self.m.predict_proba(d[self.feats].astype(float))[:, 1]

    def with_features(self, feats):
        return LGBMHazard(**self.kw, feats=feats, seed=self.seed)


def pooled(t):
    """Per model log loss pooled over scored seasons from a (model, season) table with LogLoss and N."""
    w = t['LogLoss'] * t['N']
    return w.groupby(level=0, sort=False).sum() / t['N'].groupby(level=0, sort=False).sum()


def candidates(cfg):
    """{name: factory}: the two baselines, then the logit and LightGBM grids."""
    c = cfg['pit_decision']
    f = {ConstantRate.name: ConstantRate, AgeRate.name: lambda: AgeRate(c['age_shrink'], c['age_bin'])}
    for C in c['logit_C']:
        f[f'logit, C {C}'] = lambda C=C: LogitHazard(C, c['spline_knots'])
    g = c['lgbm']
    for leaves in g['num_leaves']:
        for trees in g['n_estimators']:
            f[f'lgbm, leaves {leaves}, trees {trees}'] = lambda leaves=leaves, trees=trees: LGBMHazard(
                leaves, trees, g['min_child_samples'], g['learning_rate'])
    return f


BASELINES = [ConstantRate.name, AgeRate.name]


# Tyre choice

def mask_tyres(probs, d, force_laps_left):
    """The two compound rule: with one dry compound used so far and fewer than force_laps_left laps to go, the
    next tyre must be a different compound."""
    late = d['LapsLeft'] < force_laps_left
    if 'StopsLeft' in d:
        late |= d['StopsLeft'] <= 1   # the last planned stop
    must = ((d['TwoCompounds'] == 0) & late).to_numpy()
    allowed = pd.DataFrame({comp: ~(must & d[USED[comp]].astype(bool).to_numpy()) for comp in DRY}, index=probs.index)
    probs = probs[DRY].where(allowed, 0.0)
    s = probs.sum(axis=1)
    even = allowed.div(allowed.sum(axis=1), axis=0)   # the model gave the allowed tyres nothing: pick evenly among them
    return probs.div(s.where(s > 0), axis=0).fillna(even)


def stop_rows(d):
    return d[d['PitStop'].astype(bool) & d['NextCompound'].isin(DRY)]


class NextCompoundRate:
    """Baseline: how often each tyre followed the current compound in training."""
    name = 'rate by current compound'

    def __init__(self, force_laps_left):
        self.force = force_laps_left

    def fit(self, d):
        s = stop_rows(d)
        self.t = pd.crosstab(s['Compound'], s['NextCompound'], normalize='index').reindex(index=DRY, columns=DRY, fill_value=0.0)
        return self

    def probs(self, d):
        p = pd.DataFrame(self.t.reindex(d['Compound']).fillna(1 / 3).to_numpy(), index=d.index, columns=DRY)
        return mask_tyres(p, d, self.force)


class TyreChoice:
    """Multinomial logit over the dry tyre fitted at a stop, then mask_tyres."""
    name = 'tyre choice, multinomial logit'

    def __init__(self, C, force_laps_left, feats=TYRE_FEATURES):
        self.C, self.force, self.feats = C, force_laps_left, list(feats)

    def fit(self, d):
        s = stop_rows(d)
        self.m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
                               LogisticRegression(C=self.C, max_iter=5000))
        self.m.fit(s[self.feats].astype(float), s['NextCompound'])
        return self

    def probs(self, d):
        p = pd.DataFrame(self.m.predict_proba(d[self.feats].astype(float)), index=d.index, columns=self.m.classes_)
        return mask_tyres(p.reindex(columns=DRY, fill_value=0.0), d, self.force)


def tyre_metrics(model, d):
    s = stop_rows(d)
    p = model.probs(s)
    y = s['NextCompound'].to_numpy()
    labels = sorted(DRY)   # log_loss reads probability columns in sorted label order
    q = np.clip(p[labels].to_numpy(), 1e-6, 1)
    return {'LogLoss': log_loss(y, q / q.sum(axis=1, keepdims=True), labels=labels),
            'Accuracy': (p.idxmax(axis=1).to_numpy() == y).mean(), 'N': len(s)}


# The policy the simulator uses

class RivalPolicy:
    """weekend: use the weekend shifted wear features. offsets: per race season correction (season_offsets), a
    Series by Year and Round added to the log odds; set it for the race being simulated."""

    def __init__(self, hazard, tyre, ctx, cfg, weekend=False, offsets=None):
        c = cfg['pit_decision']
        self.hazard, self.tyre, self.ctx, self.weekend, self.offsets = hazard, tyre, ctx, weekend, offsets
        self.first_lap, self.force_stop = c['first_lap'], c['force_stop_laps_left']

    def features(self, state):
        return add_features(state, self.ctx, self.weekend)

    def pit_prob(self, state, d=None):
        """Chance of pitting at the end of the lap. None before first_lap or on the last lap; certain when the two
        compound rule is unmet with force_stop_laps_left or fewer laps to go."""
        d = self.features(state) if d is None else d
        p = shift_logit(self.hazard.prob(d), offsets_for(d, self.offsets))
        p = np.where((d['LapNumber'] < self.first_lap) | (d['LapsLeft'] <= 0), 0.0, p)
        force = (d['TwoCompounds'] == 0) & (d['LapsLeft'] <= self.force_stop) & (d['LapsLeft'] > 0)
        return np.where(force, 1.0, p)

    def tyre_probs(self, state, d=None):
        return self.tyre.probs(self.features(state) if d is None else d)

    def choose_tyre(self, state, rng, d=None):
        p = self.tyre_probs(state, d).to_numpy()
        idx = (p.cumsum(axis=1) < rng.random(len(p))[:, None]).sum(axis=1)
        return np.array(DRY, dtype=object)[np.minimum(idx, len(DRY) - 1)]


# Plan first: the stop count before the race, then the timing given the plan

def start_state(starts):
    """The STATE of each car on lap 1 from race_starts rows."""
    s = starts.assign(LapNumber=1.0, Stint=1.0)
    for comp, col in USED.items():
        s[col] = s['Compound'] == comp
    return s


class CircuitStops:
    """Mean planned stops per circuit, pulled toward the overall mean with weight k drivers."""

    def __init__(self, k=40):
        self.k = k

    def fit(self, starts):
        self.all = starts['Plan'].mean()
        self.g = starts.groupby(['Circuit', 'Year'])['Plan'].agg(['sum', 'size'])
        return self

    def predict(self, d, leave_season_out=False):
        """leave_season_out: leave out the row's own season (for training rows, so a race never sees itself)."""
        t = self.g.groupby(level='Circuit').sum().reindex(d['Circuit']).fillna(0.0).to_numpy()
        if leave_season_out:
            t = t - self.g.reindex(pd.MultiIndex.from_frame(d[['Circuit', 'Year']])).fillna(0.0).to_numpy()
        return (t[:, 0] + self.k * self.all) / (t[:, 1] + self.k)


def plan_features(starts, ctx, cstops, leave_season_out=False):
    """PLAN_FEATURES from race_starts rows: lap 1 state, circuit wear and pit loss, circuit stop history. The
    NEUTRAL flags are 0 unless given (v1 simulates no safety cars)."""
    d = add_features(start_state(starts), ctx)
    d['CircuitStops'] = cstops.predict(d, leave_season_out)
    for n in NEUTRAL:
        d[n] = d[n] if n in d else 0.0
    return d


class PlanShare:
    """Baseline: the training share of each plan."""
    name = 'plan shares'

    def fit(self, d):
        self.p = d['Plan'].value_counts(normalize=True).reindex(PLANS, fill_value=0.0)
        return self

    def probs(self, d):
        return pd.DataFrame(np.tile(self.p.to_numpy(), (len(d), 1)), index=d.index, columns=PLANS)


def _scale_columns(x, scale):
    return np.asarray(x) * scale


class PlanLogit:
    """Multinomial logit over the plan. decay: weight decay ** (seasons before the latest training season).
    neutral_scale: the NEUTRAL flags are multiplied by this after standardising, so their penalty is that factor
    squared smaller (the flags only correct training races; shrinking them leaves safety car stops in the plan)."""

    def __init__(self, C, feats=PLAN_FEATURES, decay=1.0, neutral_scale=1.0):
        self.C, self.feats, self.decay, self.neutral_scale = C, list(feats), decay, neutral_scale
        self.name = (f'plan logit, C {C}' + (f', decay {decay}' if decay != 1 else '')
                     + (f', flags x{neutral_scale}' if neutral_scale != 1 else ''))

    def fit(self, d):
        scale = np.array([self.neutral_scale if f in NEUTRAL else 1.0 for f in self.feats])
        self.m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
                               FunctionTransformer(_scale_columns, kw_args={'scale': scale}),
                               LogisticRegression(C=self.C, max_iter=5000))
        w = self.decay ** (d['Year'].max() - d['Year']).to_numpy(float)
        self.m.fit(d[self.feats].astype(float), d['Plan'].astype(int), logisticregression__sample_weight=w)
        return self

    def probs(self, d):
        p = pd.DataFrame(self.m.predict_proba(d[self.feats].astype(float)), index=d.index, columns=self.m.classes_)
        return p.reindex(columns=PLANS, fill_value=0.0)


def plan_season_offsets(probs, d, m):
    """Season correction per race of d and plan: log((A + m) / (E + m)), A the drivers on that plan and E the
    model's expected number (sum of probs) over the same season's earlier races, so m pseudo drivers pull it toward
    0. 0 at a season's first race. d: race_starts rows of every dry race (plan features with their real NEUTRAL
    flags), probs: the plan model's probs on them."""
    y = pd.get_dummies(d['Plan'].astype(int)).reindex(columns=PLANS, fill_value=0).astype(float)
    idx = pd.MultiIndex.from_frame(d[RACE])
    a = pd.DataFrame(y.to_numpy(), index=idx, columns=PLANS).groupby(RACE).sum()
    e = pd.DataFrame(probs[PLANS].to_numpy(), index=idx, columns=PLANS).groupby(RACE).sum()
    before = lambda g: g.groupby(level='Year').cumsum() - g
    return np.log((before(a) + m) / (before(e) + m)).fillna(0.0)


def apply_plan_offsets(probs, d, offsets):
    """probs moved by each row's race offsets (plan_season_offsets) on the log scale, renormalised."""
    if offsets is None or not len(offsets):
        return probs
    o = offsets.reindex(pd.MultiIndex.from_arrays([d['Year'], d['Round']])).fillna(0.0).to_numpy()
    q = probs[PLANS].to_numpy() * np.exp(o)
    return pd.DataFrame(q / q.sum(axis=1, keepdims=True), index=probs.index, columns=PLANS)


def plan_metrics(model, d, offsets=None):
    p = apply_plan_offsets(model.probs(d), d, offsets)
    q = np.clip(p[PLANS].to_numpy(), 1e-6, 1)
    y = d['Plan'].astype(int).to_numpy()
    return {'LogLoss': log_loss(y, q / q.sum(axis=1, keepdims=True), labels=PLANS),
            'Accuracy': (p.idxmax(axis=1).to_numpy() == y).mean(),
            **{f'P({k})': p[k].mean() for k in PLANS}, **{f'Real {k}': (y == k).mean() for k in PLANS}, 'N': len(d)}


def plan_candidates(cfg):
    """{name: factory}: the plan share baseline, then the plan logit grid over C and season decay."""
    c = cfg['pit_decision']['plan']
    f = {PlanShare.name: PlanShare}
    for C in c['C']:
        for decay in c['decay']:
            m = PlanLogit(C, decay=decay, neutral_scale=c['neutral_scale'])
            f[m.name] = lambda C=C, decay=decay: PlanLogit(C, decay=decay, neutral_scale=c['neutral_scale'])
    return f


def plan_data(starts, scored, ctx, train, k):
    """(training plan rows, scored plan rows, CircuitStops): starts are race_starts rows of every dry race (the
    training races, with their NEUTRAL flags), scored the race_starts rows to score; circuit history from train."""
    st = starts[starts['Year'].isin(train)]
    cs = CircuitStops(k).fit(st)
    return plan_features(st, ctx, cs, leave_season_out=True), plan_features(scored, ctx, cs), cs


def timing_candidates(cfg):
    """{name: factory}: the plan first timing hazard, logit grid and LightGBM grid on TIMING_FEATURES."""
    c = cfg['pit_decision']
    f = {f'timing logit, C {C}': (lambda C=C: LogitHazard(C, c['spline_knots'], TIMING_FEATURES, TIMING_SPLINE))
         for C in c['timing_C']}
    g = c['lgbm']
    for leaves in g['num_leaves']:
        for trees in g['n_estimators']:
            f[f'timing lgbm, leaves {leaves}, trees {trees}'] = lambda leaves=leaves, trees=trees: LGBMHazard(
                leaves, trees, g['min_child_samples'], g['learning_rate'], feats=TIMING_FEATURES)
    return f


def timing_rows(d):
    """Timing training rows: laps with a planned stop still to come (no planned stops left means no stop)."""
    return d[d['StopsLeft'] >= 1]


class PlanPolicy(RivalPolicy):
    """RivalPolicy with a plan: sample_plan draws each car's stop count at the start (the simulator stores it as
    the car's Plan column); the hazard times the planned stops. No stop once the plan is done, unless the two
    compound rule is still unmet; a planned stop not yet made with force_stop_laps_left or fewer laps to go is made.
    oracle=True replays with the real plan (evaluation only). plan_shares, e.g. {1: 0.7, 2: 0.3, 3: 0.0}, replaces
    the plan model for every car (stops fall from season to season; set it from recent races or the tyre
    supplier's predicted strategy)."""

    def __init__(self, plan, cstops, hazard, tyre, ctx, cfg, oracle=False, plan_offsets=None, plan_shares=None):
        super().__init__(hazard, tyre, ctx, cfg)
        self.plan, self.cstops, self.oracle, self.plan_offsets = plan, cstops, oracle, plan_offsets
        self.plan_shares = plan_shares

    def plan_probs(self, starts):
        """plan_offsets: the season correction (plan_season_offsets) for the races being simulated."""
        if self.plan_shares is not None:
            w = np.array([self.plan_shares.get(k, 0.0) for k in PLANS], dtype=float)
            return pd.DataFrame(np.tile(w / w.sum(), (len(starts), 1)), index=starts.index, columns=PLANS)
        return apply_plan_offsets(self.plan.probs(plan_features(starts, self.ctx, self.cstops)), starts, self.plan_offsets)

    def sample_plan(self, starts, rng):
        p = self.plan_probs(starts).to_numpy()
        idx = (p.cumsum(axis=1) < rng.random(len(p))[:, None]).sum(axis=1)
        return np.array(PLANS, dtype=float)[np.minimum(idx, len(PLANS) - 1)]

    def pit_prob(self, state, d=None):
        d = self.features(state) if d is None else d
        p = np.where(d['StopsLeft'] >= 1, self.hazard.prob(d), 0.0)
        p = np.where((d['LapNumber'] < self.first_lap) | (d['LapsLeft'] <= 0), 0.0, p)
        due = (d['TwoCompounds'] == 0) | (d['StopsLeft'] >= 1)
        force = due & (d['LapsLeft'] <= self.force_stop) & (d['LapsLeft'] > 0)
        return np.where(force, 1.0, p)


def fit_plan_policy(laps, pre, results, stops, years, cfg):
    """The chosen PlanPolicy (pit_decision.chosen: timing and plan model names) fitted on years, as
    scripts/train_pit_decision.py fits the final one. For simulator folds and backtests."""
    c = cfg['pit_decision']
    starts = race_starts(laps, pre, results, years, cfg, green_only=False)
    rows = pit_laps(laps, pre, cfg)
    rows = rows[rows['Year'].isin(years)].merge(starts[DRIVER_RACE + ['Plan']], on=DRIVER_RACE).reset_index(drop=True)
    ctx = Context(cfg).fit(laps, stops, years, laps['Circuit'].unique())
    full = timing_rows(add_features(rows, ctx))
    ftr, _, cs = plan_data(starts, starts.head(0), ctx, years, c['plan']['circuit_shrink'])
    hazard = timing_candidates(cfg)[c['chosen']['timing']]().fit(full)
    plan = plan_candidates(cfg)[c['chosen']['plan']]().fit(ftr)
    tyre = TyreChoice(c['tyre_C'], c['force_new_compound_laps_left'], TYRE_FEATURES + ['StopsLeft']).fit(full)
    return PlanPolicy(plan, cs, hazard, tyre, ctx, cfg)


# Evaluation

def lap_metrics(d, p):
    y = d['PitStop'].to_numpy(bool)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {'LogLoss': log_loss(y, p, labels=[False, True]), 'Brier': np.mean((p - y) ** 2),
            'PR_AUC': average_precision_score(y, p), 'MeanP': p.mean(), 'Rate': y.mean(), 'N': len(y)}


def fold_data(rows, laps, stops, folds, cfg, weekend=None):
    """Per fold: (scored season, fitted context, training rows, scored rows), features from that fold's context.
    weekend: (sprint_laps, practice_laps) for weekend shifted wear features, training and scored races alike."""
    out = []
    for train, year in folds:
        wk = None if weekend is None else (*weekend, list(train) + [year])
        ctx = Context(cfg).fit(laps, stops, train, rows['Circuit'].unique(), wk)
        tr, te = rows[rows['Year'].isin(train)], rows[rows['Year'] == year]
        out.append((year, ctx, add_features(tr, ctx, wk is not None), add_features(te, ctx, wk is not None)))
    return out


def score(folds, factories):
    """{name: scored rows with p} over the folds, each model refit per fold."""
    preds = {}
    for year, _, tr, te in folds:
        for name, make in factories.items():
            m = make().fit(tr)
            preds.setdefault(name, []).append(te[DRIVER_RACE + ['LapNumber', 'Compound', 'TyreLife', 'PitStop']].assign(p=m.prob(te)))
    return {n: pd.concat(p, ignore_index=True) for n, p in preds.items()}


def adjustment_preds(plain, weekend, make, ks):
    """Scored rows of one hazard model with and without the race adjustments: weekend shifted wear (folds from
    fold_data with weekend data) and the season correction for each k in ks."""
    preds = {}
    for variant, folds in [('none', plain), ('weekend wear', weekend)]:
        for _, _, tr, te in folds:
            p = make().fit(tr).prob(te)
            cols = te[DRIVER_RACE + ['LapNumber', 'PitStop']]
            preds.setdefault(variant, []).append(cols.assign(p=p))
            for k in ks:
                q = shift_logit(p, offsets_for(te, season_offsets(p, te, k)))
                preds.setdefault(f'{variant} + season, k {k}', []).append(cols.assign(p=q))
    return {n: pd.concat(v, ignore_index=True) for n, v in preds.items()}


def table(preds, by=None):
    """Lap metrics per model, pooled over folds (by='Year' for one row per scored season)."""
    if by:
        return pd.concat({n: pd.DataFrame({k: lap_metrics(g, g['p']) for k, g in p.groupby(by)}).T for n, p in preds.items()})
    return pd.DataFrame({n: lap_metrics(p, p['p']) for n, p in preds.items()}).T


def calibration(p, bins=10):
    """Mean predicted against observed pit rate in quantile bins of the prediction."""
    q = pd.qcut(p['p'].rank(method='first'), bins, labels=False)
    g = p.groupby(q)
    return pd.DataFrame({'MeanP': g['p'].mean(), 'Observed': g['PitStop'].mean(), 'N': g.size()})


def race_starts(laps, pre, results, years, cfg, green_only=None):
    """Classified finishers of dry races (green_only, default pit_decision.replay.green_races_only: only races with
    no SC, VSC or red flag lap), with their state on lap 1, the NEUTRAL flags of the race, Neutral (any), the real
    number of tyre stops, Plan (the stops, 1 to 3, 3 meaning three or more) and the lap of the first stop."""
    if green_only is None:
        green_only = cfg['pit_decision']['replay']['green_races_only']
    l = laps[laps['Year'].isin(years) & ~laps['WetRace']]
    l = l.join(features.pit_labels(l, cfg)[['PitStop']])
    status = l.groupby(RACE)['TrackStatus'].agg(lambda x: ''.join(x.dropna().astype(str)))
    neutral = pd.DataFrame({n: status.str.contains(code).astype(float) for n, code in NEUTRAL_CODES.items()})
    neutral['Neutral'] = neutral.max(axis=1)
    if green_only:
        l = l[pd.MultiIndex.from_frame(l[RACE]).isin(neutral.index[neutral['Neutral'] == 0])]
    fin = results[pd.to_numeric(results['ClassifiedPosition'], errors='coerce').notna()][DRIVER_RACE]
    l = l.merge(fin, on=DRIVER_RACE)
    l['RaceLaps'] = l.groupby(RACE)['LapNumber'].transform('max')
    actual = l.groupby(DRIVER_RACE).agg(Stops=('PitStop', 'sum')).join(
        l[l['PitStop']].groupby(DRIVER_RACE)['LapNumber'].min().rename('FirstStop'))
    s = l[(l['LapNumber'] == 1) & l['Compound'].isin(DRY) & l['TyreLife'].notna()]
    s = s[DRIVER_RACE + ['Circuit', 'RaceLaps', 'Compound', 'TyreLife', 'FreshTyre']].merge(pre, on=DRIVER_RACE, how='left')
    s = s.join(actual, on=DRIVER_RACE).join(neutral, on=RACE)
    s['Plan'] = s['Stops'].clip(PLANS[0], PLANS[-1])
    return s.reset_index(drop=True)


def replay(policy, starts, draws=200, seed=0, record=False):
    """Simulate every start's tyre stops through the race with the policy, draws times each: start on the real
    first tyre (a PlanPolicy first draws the car's plan), pit each lap with the policy's chance, fit the policy's tyre. One row per draw and driver with
    SimStops and SimFirst (lap of the first stop, NaN if none). Ignores everything but the car's own state (v1).
    record=True also returns every stop (Draw, Driver, Lap, Compound fitted), for simulator.strategy.model_stints."""
    s = start_state(starts.loc[starts.index.repeat(draws)].reset_index(drop=True))
    s['Draw'] = np.tile(np.arange(draws), len(starts))
    s['FreshTyre'], s['SimStops'], s['SimFirst'] = s['FreshTyre'].astype(bool), 0, np.nan
    rng = np.random.default_rng(seed)
    events = []
    if isinstance(policy, PlanPolicy):   # oracle=True keeps the starts' real Plan
        s['Plan'] = s['Plan'] if policy.oracle else policy.sample_plan(s, rng)
    else:
        s = s.drop(columns='Plan', errors='ignore')
    for lap in range(1, int(s['RaceLaps'].max())):
        s['LapNumber'] = float(lap)
        on = (lap < s['RaceLaps']).to_numpy()
        if lap >= policy.first_lap and on.any():
            st = s[on]
            d = policy.features(st)
            stop = rng.random(len(st)) < policy.pit_prob(st, d)
            if stop.any():
                idx = st.index[stop]
                new = policy.choose_tyre(st[stop], rng, d[stop])
                if record:
                    events.append(pd.DataFrame({'Draw': s.loc[idx, 'Draw'].to_numpy(), 'Driver': s.loc[idx, 'Driver'].to_numpy(),
                                                'Lap': float(lap), 'Compound': new}))
                s.loc[idx, 'Compound'], s.loc[idx, 'TyreLife'], s.loc[idx, 'FreshTyre'] = new, 0.0, True
                s.loc[idx, 'Stint'] += 1
                s.loc[idx, 'SimStops'] += 1
                s.loc[idx, 'SimFirst'] = s.loc[idx, 'SimFirst'].fillna(float(lap))
                for comp, col in USED.items():
                    s.loc[idx, col] = s.loc[idx, col] | (new == comp)
        s['TyreLife'] += 1
    out = s[DRIVER_RACE + ['Draw', 'SimStops', 'SimFirst'] + (['Plan'] if 'Plan' in s else [])]
    if record:
        cols = ['Draw', 'Driver', 'Lap', 'Compound']
        return out, (pd.concat(events, ignore_index=True) if events else pd.DataFrame(columns=cols))
    return out


def replay_metrics(sim, starts):
    """Stop count shares, real against simulated (over all draws), mean stops, and the first stop lap error of
    each driver's median simulated first stop."""
    cap = lambda s: s.clip(upper=3).map({0: '0', 1: '1', 2: '2', 3: '3+'})
    real = cap(starts['Stops']).value_counts(normalize=True)
    simd = cap(sim['SimStops']).value_counts(normalize=True)
    per = sim.groupby(DRIVER_RACE).agg(SimStops=('SimStops', 'mean'), SimFirst=('SimFirst', 'median'))
    a = starts.set_index(DRIVER_RACE)[['Stops', 'FirstStop']].join(per)
    out = {f'{k} stops, real': real.get(k, 0.0) for k in ['1', '2', '3+']}
    out.update({f'{k} stops, sim': simd.get(k, 0.0) for k in ['1', '2', '3+']})
    out.update({'Mean stops, real': a['Stops'].mean(), 'Mean stops, sim': a['SimStops'].mean(),
                'Stops MAE': (a['SimStops'] - a['Stops']).abs().mean(),
                'First stop MAE, laps': (a['SimFirst'] - a['FirstStop']).abs().mean(), 'Drivers': len(a)})
    return out
