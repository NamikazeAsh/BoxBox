"""Updates to a fitted wear model from other data: earlier races of the same season (season multipliers), the
same weekend's sprint and practice long runs (weekend shifts), and stints already run in the same race (live
shifts). Each returns a wrapper with the same wear and dirty_air interface (modelling D4, O1, O7, O9).
"""
import numpy as np
import pandas as pd
from scipy import stats

from boxbox import evaluate, features
from boxbox.degradation.task import COMPOUNDS, KEY, RACE, STINT, anchored_forecast, forecast_cuts, model_laps


class SeasonScaled:
    """A fitted model whose wear is scaled by a multiplier per compound (fix 1, in season updating)."""

    def __init__(self, base, mult):
        self.base, self.mult = base, mult

    def wear(self, laps):
        return self.base.wear(laps) * laps['Compound'].map(self.mult).fillna(1.0)

    def dirty_air(self, laps):
        return self.base.dirty_air(laps)


def _wear_regression(base, laps):
    """Per lap pieces of the within stint regression of lap time change on the base model's predicted wear
    change. Summed over a set of laps, uy / uu says how much faster (above 1) or slower they wore than the
    base model predicts; n counts stints."""
    x = base.f(laps['TyreLife'].clip(lower=1))
    u = base.slopes_for(laps) * (x - x.groupby([laps[c] for c in STINT]).transform('mean'))
    dy = laps['LapTimeFC_s'] - laps.groupby(STINT)['LapTimeFC_s'].transform('mean')
    return pd.DataFrame({'uy': u * dy, 'uu': u ** 2, 'n': (~laps.duplicated(STINT)).astype(float)}, index=laps.index)


def _shrink(sums, k):
    """(n x raw + k) / (n + k): the raw multiplier pulled toward 1 with weight k stints."""
    raw = (sums['uy'] / sums['uu'].where(sums['uu'] > 0)).fillna(1.0)
    return ((sums['n'] * raw + k) / (sums['n'] + k)).fillna(1.0)


def _groups(laps, scope):
    return laps['Compound'] if scope == 'compound' else pd.Series('all', index=laps.index, name='Compound')


def season_multipliers(base, laps, scope, k):
    """Multiplier per round and compound from the same season's earlier races only (fix 1).
    Round 1 gets 1. scope 'global' shares one value across compounds."""
    r = _wear_regression(base, laps)
    per = r.groupby([laps['Round'], _groups(laps, scope)]).sum()
    rounds = sorted(laps['Round'].unique())
    per = per.unstack().reindex(rounds).fillna(0.0)
    before = per.cumsum().shift(1).fillna(0.0)          # sums over rounds strictly before each round
    m = _shrink({c: before[c] for c in ['uy', 'uu', 'n']}, k)
    if scope == 'global':
        m = pd.DataFrame({c: m['all'] for c in laps['Compound'].unique()})
    return m


def weekend_multipliers(base, sprint, scope, k):
    """Multiplier per (Year, Round) from that weekend's sprint, shrunk toward 1 with weight k stints.
    Columns are compounds, or 'all' for scope 'global'."""
    r = _wear_regression(base, sprint)
    sums = r.groupby([sprint['Year'], sprint['Round'], _groups(sprint, scope)]).sum()
    return _shrink(sums, k).unstack()


class WeekendShifted:
    """A fitted model with a per compound slope shift in seconds per lap (stable where wear is near zero)."""

    def __init__(self, base, shift):
        self.base, self.shift = base, shift

    def wear(self, laps):
        x = self.base.f(laps['TyreLife'].clip(lower=1))
        return self.base.wear(laps) + laps['Compound'].map(self.shift).fillna(0.0) * x

    def dirty_air(self, laps):
        return self.base.dirty_air(laps)


def weekend_shifts(base, sprint, scope, k):
    """Slope shift per (Year, Round) from that weekend's sprint: within stint slope of the lap time left over
    after the base model's wear, shrunk toward 0 with weight k stints. Columns as in weekend_multipliers."""
    x = base.f(sprint['TyreLife'].clip(lower=1))
    keys = [sprint[c] for c in STINT]
    dx = x - x.groupby(keys).transform('mean')
    resid = sprint['LapTimeFC_s'] - base.wear(sprint) - base.dirty_air(sprint)
    dr = resid - resid.groupby(keys).transform('mean')
    r = pd.DataFrame({'xy': dx * dr, 'xx': dx ** 2, 'n': (~sprint.duplicated(STINT)).astype(float)})
    sums = r.groupby([sprint['Year'], sprint['Round'], _groups(sprint, scope)]).sum()
    raw = sums['xy'] / sums['xx'].where(sums['xx'] > 0)
    return (sums['n'] * raw.fillna(0.0) / (sums['n'] + k)).unstack()


class StintShifted:
    """A fitted model with its own slope shift per stint, s per lap (a Series by STINT; missing means none)."""

    def __init__(self, base, shift):
        self.base, self.shift = base, shift

    def wear(self, laps):
        s = laps.join(self.shift.rename('_Shift'), on=STINT)['_Shift'].fillna(0.0)
        return self.base.wear(laps) + s * self.base.f(laps['TyreLife'].clip(lower=1))

    def dirty_air(self, laps):
        return self.base.dirty_air(laps)


def live_shifts(base, laps, cfg, k, prior=None):
    """Live update from the race itself (modelling O9): each stint's slope shift is the within stint slope of
    lap time left after the base model, over every stint of the same race, using only laps run up to that
    stint's forecast lap (forecast_cuts), so nothing after the forecast is used. Stints with fewer than
    min_fit_laps laps by then are left out. Shrunk toward prior (the weekend shift, a Series by Year and Round;
    default 0) with weight k stints. Returns (shift, stints behind it), both Series by STINT."""
    m = cfg['degradation']['min_fit_laps']
    d = laps.assign(x=base.f(laps['TyreLife'].clip(lower=1)),
                    r=laps['LapTimeFC_s'] - base.wear(laps) - base.dirty_air(laps))
    cuts = forecast_cuts(laps, cfg)
    raw, n = {}, {}
    for race, t in d.groupby(RACE):
        c = cuts.loc[race]
        for cut in c.unique():
            u = t[t['LapNumber'] <= cut]
            u = u[u.groupby(STINT)['LapNumber'].transform('size') >= m]
            keys = [u[s] for s in STINT]
            dx = u['x'] - u['x'].groupby(keys).transform('mean')
            dr = u['r'] - u['r'].groupby(keys).transform('mean')
            sxx = (dx * dx).sum()
            raw[race + (cut,)] = (dx * dr).sum() / sxx if sxx > 0 else 0.0
            n[race + (cut,)] = u.drop_duplicates(STINT).shape[0]
    at = list(zip(cuts.index.get_level_values('Year'), cuts.index.get_level_values('Round'), cuts))
    raw = pd.Series([raw[a] for a in at], index=cuts.index)
    n = pd.Series([n[a] for a in at], index=cuts.index, dtype=float)
    p = 0.0 if prior is None else pd.Series(
        [prior.get(a[:2], 0.0) for a in at], index=cuts.index).fillna(0.0)
    return ((n * raw + k * p) / (n + k)).fillna(0.0).rename('Shift'), n.rename('Stints')


def race_shift_se(base, laps, n_boot=300, seed=0):
    """Each race's slope shift against the base model (one value for all compounds, not shrunk), with its
    sampling SD from resampling whole stints within the race (modelling O7: is a race's departure from its
    circuit's history real or noise?)."""
    x = base.f(laps['TyreLife'].clip(lower=1))
    keys = [laps[c] for c in STINT]
    dx = x - x.groupby(keys).transform('mean')
    r = laps['LapTimeFC_s'] - base.wear(laps) - base.dirty_air(laps)
    s = pd.DataFrame({'xy': dx * (r - r.groupby(keys).transform('mean')), 'xx': dx ** 2}).groupby(keys).sum()
    rng, out = np.random.default_rng(seed), {}
    for key, g in s.groupby(level=RACE):
        xy, xx = g['xy'].to_numpy(), g['xx'].to_numpy()
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        out[key] = {'Shift': xy.sum() / xx.sum(), 'SE': (xy[idx].sum(1) / xx[idx].sum(1)).std(), 'Stints': len(g)}
    t = pd.DataFrame(out).T
    t.index.names = RACE
    return t


def mult_for(row, scope):
    """One weekend's multipliers as a Series by compound (missing compounds default to 1 in SeasonScaled)."""
    if scope == 'global':
        return pd.Series(row['all'], index=COMPOUNDS)
    return row.dropna()


def model_runs(prac, laps, seasons, cfg, min_laps):
    """Practice long run laps from the given seasons, comparable with the race model: fuel corrected at the
    race burn rate, tyre age known, and stint length held at the reference (a practice run is not a planned
    race stint, so the stint length term is switched off for it)."""
    runs = features.practice_runs(prac[prac['Year'].isin(seasons)], cfg, min_laps)
    runs = features.add_practice_fuel_correction(runs, laps, cfg)
    runs = runs[runs['TyreLife'].notna()].assign(StintLaps=cfg['degradation']['stint_length_ref'])
    return runs[STINT + KEY + ['Session', 'LapNumber', 'TyreLife', 'StintLaps', 'LapTimeFC_s']]


def run_shifts(base, runs):
    """Slope of each practice run's lap time left over after the base model's wear, s per lap. Theil Sen
    (median of pairwise slopes), so a lap lost in traffic, common in practice, does not tilt a run."""
    r = runs.assign(x=base.f(runs['TyreLife'].clip(lower=1)), r=runs['LapTimeFC_s'] - base.wear(runs))
    return r.groupby(STINT)[['x', 'r']].apply(lambda g: stats.theilslopes(g['r'], g['x'])[0]).rename('Shift')


def practice_offset(base, runs):
    """Mean weekend median run shift over the base model's training seasons. Practice runs are not centred on
    the race model (push level, track evolution, the burn rate assumption), so the forecast removes this."""
    return run_shifts(base, runs).groupby(RACE).median().mean()


def circuit_history(base):
    """Training stints per circuit behind the base model's slopes, all compounds."""
    return base.cells['Stints'].groupby(level='Circuit').sum()


def shrink_runs(raw, n, k):
    """A weekend shift from n runs pulled toward 0 with weight k runs (0 where there are none)."""
    s = n * raw / (n + k)
    return s.fillna(0.0) if isinstance(s, pd.Series) else float(np.nan_to_num(s))


def same_weekend_model(base, circuit, sprint, runs, offset, cfg):
    """The race model for one weekend (modelling O1): every compound slope moved by the dry sprint's shift
    when there is one, else by the practice shift when the circuit has no dry race history behind the base
    model, else the base model unchanged. sprint is that weekend's model_laps (empty if none or wet), runs
    its model_runs, offset the practice_offset on the base model's training seasons."""
    c = cfg['degradation']['weekend_adjust']
    s = 0.0
    if len(sprint):
        s = weekend_shifts(base, sprint, 'global', c['sprint_shrink'])['all'].iloc[0]
    elif len(runs) and circuit_history(base).get(circuit, 0.0) <= c['practice_max_history']:
        rs = run_shifts(base, runs)
        s = shrink_runs(rs.median() - offset, len(rs), c['practice_shrink'])
    return WeekendShifted(base, pd.Series(s, index=COMPOUNDS))


def compound_gaps(shifts, laps, min_n=3):
    """Per weekend Soft minus Medium and Medium minus Hard median run (or stint) shift, where both compounds
    have min_n of them. D4's compound deltas: fuel and track evolution cancel in a difference."""
    comp = laps.drop_duplicates(STINT).set_index(STINT)['Compound']
    g = shifts.to_frame().join(comp).groupby(RACE + ['Compound'])['Shift'].agg(['median', 'size']).unstack()
    med = g['median'].where(g['size'] >= min_n).reindex(columns=COMPOUNDS)
    return pd.DataFrame({'SoftMinusMedium': med['SOFT'] - med['MEDIUM'], 'MediumMinusHard': med['MEDIUM'] - med['HARD']})


def weekend_table(base, test, sprint, runs, offset, cfg):
    """One row per race weekend in test: circuit, training stints there (History), the dry sprint's shift
    (shrunk with weekend_adjust.sprint_shrink), the practice run count and raw practice shift (median run
    shift less offset, not shrunk), and the race's own shift against the base model, plus compound gaps in
    practice and race. Race columns are for agreement checks only and never enter a forecast."""
    w = test.drop_duplicates(RACE).set_index(RACE)[['Circuit']]
    w['History'] = w['Circuit'].map(circuit_history(base)).fillna(0.0)
    w['Sprint'] = weekend_shifts(base, sprint, 'global', cfg['degradation']['weekend_adjust']['sprint_shrink'])['all'] \
        if len(sprint) else np.nan
    rs = run_shifts(base, runs)
    w['Runs'] = rs.groupby(RACE).size()
    w['Runs'] = w['Runs'].fillna(0)
    w['PracticeRaw'] = rs.groupby(RACE).median() - offset
    w['Race'] = weekend_shifts(base, test, 'global', 0)['all']
    gaps = pd.concat({'Practice': compound_gaps(rs, runs), 'Race': compound_gaps(run_shifts(base, test), test)}, axis=1)
    gaps.columns = [a + b for a, b in gaps.columns]
    return w.join(gaps)


def forecast_with_shifts(base, test, shift, cfg):
    """Anchored forecast race by race, every compound slope moved by that weekend's shift (s per lap, a
    Series by Year and Round; missing or NaN means no shift)."""
    out = []
    for key, t in test.groupby(RACE):
        s = shift.get(key, 0.0)
        out.append(anchored_forecast(WeekendShifted(base, pd.Series(0.0 if pd.isna(s) else s, index=COMPOUNDS)), t, cfg))
    return pd.concat(out)


def weekend_variants(w, k_grid, max_history):
    """Shift per weekend for each way of using same weekend data, from a weekend_table.
    practice everywhere: every weekend with runs. practice fallback: only circuits with at most max_history
    training stints. sprint: dry sprint weekends. sprint, else practice fallback: the sprint where there is one."""
    v = {'none': pd.Series(0.0, index=w.index), 'sprint': w['Sprint'].fillna(0.0)}
    new = w['History'] <= max_history
    for k in k_grid:
        p = shrink_runs(w['PracticeRaw'], w['Runs'], k)
        v[f'practice everywhere, k {k}'] = p
        v[f'practice fallback, k {k}'] = p.where(new, 0.0)
        v[f'sprint, else practice fallback, k {k}'] = w['Sprint'].fillna(p.where(new, 0.0))
    return v


def weekend_forecast(base, laps, sprint_laps, prac, cfg, train_years, year):
    """The pre race model for one season: the anchored forecast of every race of year with each weekend's
    shift (dry sprint, else practice at circuits with no dry history in train_years, the configured shrinks).
    base is fitted on train_years."""
    w = cfg['degradation']['weekend_adjust']
    test = model_laps(laps, [year], cfg)
    runs = model_runs(prac, laps, list(train_years) + [year], cfg, cfg['long_run']['min_laps'])
    offset = practice_offset(base, runs[runs['Year'].isin(train_years)])
    t = weekend_table(base, test, model_laps(sprint_laps, [year], cfg), runs[runs['Year'] == year], offset, cfg)
    shift = weekend_variants(t, [w['practice_shrink']], w['practice_max_history'])[
        f"sprint, else practice fallback, k {w['practice_shrink']}"]
    return forecast_with_shifts(base, test, shift, cfg)


def run_weekend_adjust(laps, sprint_laps, prac, cfg, seasons, base_factory, min_laps):
    """Validation only: does same weekend data (sprint, practice long runs) improve race forecasts? Every
    race of each validation season is scored, so 'none' is the base model's validation error. The practice
    offset comes from the training seasons. Returns (predictions per variant, weekend tables)."""
    c = cfg['degradation']['weekend_adjust']
    preds, tables = {}, []
    for train_years, year in evaluate.season_splits(seasons)['validation']:
        base = base_factory().fit(model_laps(laps, train_years, cfg, train=True))
        test = model_laps(laps, [year], cfg)
        runs = model_runs(prac, laps, train_years + [year], cfg, min_laps)
        offset = practice_offset(base, runs[runs['Year'].isin(train_years)])
        w = weekend_table(base, test, model_laps(sprint_laps, [year], cfg), runs[runs['Year'] == year], offset, cfg)
        tables.append(w.assign(Offset=offset))
        for name, shift in weekend_variants(w, c['practice_shrink_grid'], c['practice_max_history']).items():
            preds.setdefault(name, []).append(forecast_with_shifts(base, test, shift, cfg).join(w['History'], on=RACE))
    return {n: pd.concat(p) for n, p in preds.items()}, pd.concat(tables)


def season_adjusted_forecast(base, laps, cfg, scope, k):
    """Anchored forecast for one season, race by race, with the multiplier from earlier races."""
    m = season_multipliers(base, laps, scope, k)
    out = [anchored_forecast(SeasonScaled(base, m.loc[r]), laps[laps['Round'] == r], cfg)
           for r in sorted(laps['Round'].unique())]
    return pd.concat(out), m


def run_season_adjust(laps, cfg, seasons, base_factory):
    """Validation only: tune the season multiplier (scope, shrink) for a fixed base model on the rolling
    validation folds. The test season is not touched. Returns (table, predictions, multipliers)."""
    c = cfg['degradation']['season_adjust']
    split = evaluate.season_splits(seasons)
    preds, mults = {}, {}
    for train_years, year in split['validation']:
        base = base_factory().fit(model_laps(laps, train_years, cfg, train=True))
        test = model_laps(laps, [year], cfg)
        preds.setdefault('no season adjustment', []).append(anchored_forecast(base, test, cfg))
        for scope in c['scopes']:
            for k in c['shrink_grid']:
                p, m = season_adjusted_forecast(base, test, cfg, scope, k)
                preds.setdefault(f'{scope}, shrink {k}', []).append(p)
                mults[(year, scope, k)] = m
    preds = {n: pd.concat(p) for n, p in preds.items()}
    table = pd.concat({n: evaluate.error_table(p) for n, p in preds.items()}).droplevel(1)
    for y in [y for _, y in split['validation']]:
        table[f'MAE_{y}'] = pd.Series({n: p.loc[p['Year'] == y, 'Err_s'].abs().mean() for n, p in preds.items()})
    return table, preds, mults


def run_sprint_adjust(laps, sprint_laps, cfg, seasons, base_factory):
    """Validation only: on sprint weekends, scale race wear by how the sprint wore against the base model.
    Scored on races whose sprint was dry. Returns (table, predictions, per weekend multipliers)."""
    c = cfg['degradation']['sprint_adjust']
    split = evaluate.season_splits(seasons)
    preds, mults = {}, {}
    for train_years, year in split['validation']:
        base = base_factory().fit(model_laps(laps, train_years, cfg, train=True))
        sp = model_laps(sprint_laps, [year], cfg)
        weekends = sorted(sp['Round'].unique())
        test = model_laps(laps, [year], cfg)
        test = test[test['Round'].isin(weekends)]
        preds.setdefault('no sprint adjustment', []).append(anchored_forecast(base, test, cfg))
        for mode in c['modes']:
            for scope in c['scopes']:
                for k in c['shrink_grid']:
                    if mode == 'scale':
                        m = weekend_multipliers(base, sp, scope, k)
                        wrap = lambda row: SeasonScaled(base, mult_for(row, scope))
                    else:
                        m = weekend_shifts(base, sp, scope, k)
                        wrap = lambda row: WeekendShifted(base, mult_for(row, scope).fillna(0.0) if scope == 'compound'
                                                          else mult_for(row, scope))
                    out = [anchored_forecast(wrap(m.loc[(year, r)]), test[test['Round'] == r], cfg) for r in weekends]
                    preds.setdefault(f'{mode}, {scope}, shrink {k}', []).append(pd.concat(out))
                    mults[(year, mode, scope, k)] = m
    preds = {n: pd.concat(p) for n, p in preds.items()}
    table = pd.concat({n: evaluate.error_table(p) for n, p in preds.items()}).droplevel(1)
    for y in [y for _, y in split['validation']]:
        table[f'MAE_{y}'] = pd.Series({n: p.loc[p['Year'] == y, 'Err_s'].abs().mean() for n, p in preds.items()})
    return table, preds, mults
