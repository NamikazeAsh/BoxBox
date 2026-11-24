"""Wet race study (notes/wet_and_degradation_2026-11-23.md). v1 leaves wet races out (modelling D6); this measures
what a wet race model would need. Study only: no model or the simulator reads it. Config block `wet`.

Wear: the degradation task (degradation/task.py) on wet tyre stints, clean laps from features.wet_clean. On a
drying or wetting track a stint's lap times move with the track as well as the tyre, so the usual within stint
slope is an effective rate (tyre plus track). TrackNetWear separates them: driver race effects and race lap
effects (the track state every car on that tyre shares) are removed together, so its slope comes from cars that
fitted fresh tyres of the same kind later in the race.

Crossover: in a window of laps where slicks and intermediates run together, lap time = lap effect + driver
effect + slick x (a + b x (lap - lap0)). a + b x (lap - lap0) is the slick minus intermediate gap; where it
crosses zero is the crossover lap. A car's switch is then priced against the best switch lap on that gap
(switch_losses), which measures how much a wet strategy call is worth.
"""
import numpy as np
import pandas as pd

from boxbox import evaluate, features
from boxbox.degradation.linear import LinearDegradation
from boxbox.degradation.task import KEY, RACE, STINT, ConstantPace, forecast, model_laps


# Wear ------------------------------------------------------------------------------------------------------------

def wet_model_laps(laps, seasons, cfg):
    """model_laps on wet tyre laps: IsClean replaced by features.wet_clean. WetRace is cleared because wet tyres
    only run in wet races (features.wet_races), which model_laps would otherwise drop."""
    seasons = [y for y in seasons if y not in cfg['holdout_seasons']]
    return model_laps(laps.assign(IsClean=features.wet_clean(laps, cfg), WetRace=False), seasons, cfg)


def demean_two_way(v, a, b, iters):
    """v with both group means removed (alternating projections): a and b are integer group codes."""
    v = np.asarray(v, dtype=float).copy()
    for _ in range(iters):
        v -= (np.bincount(a, v) / np.bincount(a))[a]
        v -= (np.bincount(b, v) / np.bincount(b))[b]
    return v


class TrackNetWear:
    """Wear per compound net of the track: lap time = driver race level + race lap effect + slope x tyre age,
    fitted within compound. Stint levels cannot be used here: within a stint tyre age is the race lap plus a
    constant, so stint and race lap effects absorb a linear wear term completely. With one level per driver and
    race, the slope comes from drivers who fitted fresh tyres of the same compound later in the race (age resets,
    the track does not). No circuit slopes (few wet races per circuit). TrackRate is the within stint slope less
    the net one: how fast the track itself moved per lap on that tyre, on average."""
    name = 'net of track'

    def __init__(self, iters=100):
        self.iters = iters

    def fit(self, laps):
        rows = {}
        for comp, d in laps.groupby('Compound'):
            st = d.groupby(STINT).ngroup().to_numpy()
            a = d.groupby(RACE + ['Driver']).ngroup().to_numpy()
            b = d.groupby(RACE + ['LapNumber']).ngroup().to_numpy()
            x, y = d['TyreLife'].to_numpy(float), d['LapTimeFC_s'].to_numpy(float)
            xs, ys = (v - (np.bincount(st, v) / np.bincount(st))[st] for v in (x, y))
            xt, yt = demean_two_way(x, a, b, self.iters), demean_two_way(y, a, b, self.iters)
            within = (xs @ ys) / (xs @ xs) if xs @ xs > 0 else np.nan
            net = (xt @ yt) / (xt @ xt) if xt @ xt > 1e-6 * len(xt) else np.nan
            rows[comp] = {'Slope': net, 'WithinStint': within, 'TrackRate': within - net,
                          'Stints': int(st.max() + 1), 'Laps': len(d),
                          'RefitDrivers': int((d.groupby(RACE + ['Driver'])['Stint'].nunique() > 1).sum())}
        self.cells = pd.DataFrame(rows).T
        return self

    def wear(self, laps):
        return laps['Compound'].map(self.cells['Slope']).astype(float).fillna(0.0) * laps['TyreLife']

    def dirty_air(self, laps):
        return pd.Series(0.0, index=laps.index)


def wear_models(cfg):
    """{name: factory}: constant pace, one wear rate per compound, circuit rates blended toward it (wet.shrink_grid),
    and the rate net of the track."""
    w = cfg['wet']
    f = {'constant pace': ConstantPace, 'linear, compound only': lambda: LinearDegradation(by_circuit=False)}
    for k in w['shrink_grid']:
        f[f'linear, shrink {k}'] = lambda k=k: LinearDegradation(k)
    f['net of track'] = lambda: TrackNetWear(w['track_iters'])
    return f


def wear_study(laps, cfg):
    """Anchored forecasts of every wear model on the wet tyre laps: {'validation': preds, 'test': preds}, each with
    a Fold column (the scored season), and the models fitted on each fold's training seasons."""
    f = wear_models(cfg)
    split = evaluate.season_splits([y for y in cfg['wet']['wear_seasons'] if y not in cfg['holdout_seasons']])
    out, fits = {}, {}
    for part, folds in [('validation', split['validation']), ('test', [split['test']])]:
        preds = []
        for train_years, year in folds:
            train, test = wet_model_laps(laps, train_years, cfg), wet_model_laps(laps, [year], cfg)
            for name, make in f.items():
                m = make().fit(train)
                fits[(year, name)] = m
                preds.append(forecast(m, test, cfg).assign(Model=name, Fold=year))
        out[part] = pd.concat(preds)
    return out, fits


def wear_table(preds, cfg):
    """MAE and bias per model and compound (compounds with fewer than wet.min_stints scored stints in a fold are
    marked), pooled over the folds in preds."""
    g = preds.groupby(['Compound', 'Model'])
    t = pd.DataFrame({'MAE_s': g['Err_s'].apply(lambda e: e.abs().mean()), 'Bias_s': g['Err_s'].mean(), 'Laps': g.size(),
                      'Stints': g.apply(lambda d: d[STINT].drop_duplicates().shape[0] // d['Model'].nunique())})
    t['Enough'] = t['Stints'] >= cfg['wet']['min_stints'] * preds['Fold'].nunique()
    return t


# Crossover -------------------------------------------------------------------------------------------------------

def crossover_laps(laps, cfg):
    """Laps that can price slicks against the wet group: valid, not lap 1, not an in or out lap, no safety car,
    VSC or red flag (local yellows kept when allow_yellow: wet races have many), on a slick or wet group tyre
    (features.tyre_group), in the crossover seasons, and no slower than outlier_pct times the median of the same
    race lap on the same tyre group. Traffic is kept (both groups run in it)."""
    c = cfg['wet']['crossover']
    seasons = [y for y in c['seasons'] if y not in cfg['holdout_seasons']]
    d = laps[laps['Year'].isin(seasons)].assign(Group=features.tyre_group(laps['Compound'], cfg))
    d = d[d['LapTime_s'].notna() & (d['LapNumber'] > 1) & ~d['PitIn'].astype(bool) & ~d['PitOut'].astype(bool)
          & d['Group'].notna()]
    st = d['TrackStatus'].fillna('').astype(str)
    d = d[~st.str.contains('[4-7]') if c.get('allow_yellow', True) else st == '1']
    med = d.groupby(RACE + ['LapNumber', 'Group'])['LapTime_s'].transform('median')
    return d[d['LapTime_s'] <= c['outlier_pct'] * med]


def windows(cl, cfg):
    """Crossover windows: runs of race laps where at least min_cars cars run on each tyre group, overlap laps up to
    max_gap_laps apart joined, at least min_window_laps overlap laps. One row per window."""
    c = cfg['wet']['crossover']
    n = cl.groupby(RACE + ['LapNumber', 'Group'])['Driver'].nunique().unstack('Group', fill_value=0)
    if not {'slick', 'wet'} <= set(n.columns):
        return pd.DataFrame(columns=RACE + ['Start', 'End', 'OverlapLaps'])
    ov = n[(n['slick'] >= c['min_cars']) & (n['wet'] >= c['min_cars'])].reset_index()
    rows = []
    for (y, r), d in ov.groupby(RACE):
        lapnos = d['LapNumber'].to_numpy()
        cut = np.flatnonzero(np.diff(lapnos) > c['max_gap_laps'] + 1) + 1
        for seg in np.split(lapnos, cut):
            if len(seg) >= c['min_window_laps']:
                rows.append({'Year': y, 'Round': r, 'Start': int(seg[0]), 'End': int(seg[-1]), 'OverlapLaps': len(seg)})
    return pd.DataFrame(rows, columns=RACE + ['Start', 'End', 'OverlapLaps'])


def fit_gap(rows, lap0, driver_effects=True):
    """Least squares on one window's laps: lap time = lap effect + driver effect + slick x (a + b x (lap - lap0)).
    Returns (a, b), or (nan, nan) when the slick terms are not identified (no car on both groups with driver
    effects, or no lap with both groups)."""
    lapno, drv = rows['LapNumber'].to_numpy(), rows['Driver'].to_numpy()
    slick = (rows['Group'] == 'slick').to_numpy(float)
    cols = [pd.get_dummies(lapno).to_numpy(float)]
    if driver_effects:
        cols.append(pd.get_dummies(drv, drop_first=True).to_numpy(float))
    cols += [slick[:, None], (slick * (lapno - lap0))[:, None]]
    X = np.hstack(cols)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return np.nan, np.nan
    beta = np.linalg.lstsq(X, rows['LapTime_s'].to_numpy(float), rcond=None)[0]
    return beta[-2], beta[-1]


def fit_lap_gaps(rows, driver_effects=True):
    """The slick minus wet group gap on each overlap lap, free per lap (no line): lap time = lap effect + driver
    effect + slick x lap gap. Laps without both groups get no gap. Returns a Series by lap number (empty when not
    identified)."""
    both = rows.groupby('LapNumber')['Group'].nunique()
    both = both[both == 2].index
    lapno, slick = rows['LapNumber'].to_numpy(), (rows['Group'] == 'slick').to_numpy(float)
    cols = [pd.get_dummies(lapno).to_numpy(float)]
    if driver_effects:
        cols.append(pd.get_dummies(rows['Driver'].to_numpy(), drop_first=True).to_numpy(float))
    cols.append(np.column_stack([slick * (lapno == l) for l in both]))
    X = np.hstack(cols)
    if not len(both) or np.linalg.matrix_rank(X) < X.shape[1]:
        return pd.Series(dtype=float)
    beta = np.linalg.lstsq(X, rows['LapTime_s'].to_numpy(float), rcond=None)[0]
    return pd.Series(beta[-len(both):], index=both.astype(int))


def switch_losses(laps, year, rnd, lo, hi, gap, trend):
    """Time each car lost by its first tyre group switch against the best switch lap on the fitted gap
    (gap(lap) = slick minus wet group seconds), over the window's overlap laps lo to hi only (the gap is not
    measured outside them). Only switches the gap prices: to slicks when the gap falls (trend < 0, drying), to the
    wet group when it rises. A switch at lap s puts laps s onward on the new group; a switch before lo or after hi
    counts as lo or hi + 1 (Clipped). Loss = cost(s) - min cost over switch laps lo to hi + 1, cost(s) = sum over
    window laps l >= s of the new group's time minus the old one's. The stop itself costs the same whenever it is
    made, so it is left out."""
    to = 'slick' if trend < 0 else 'wet'
    d = laps[(laps['Year'] == year) & (laps['Round'] == rnd)]
    d = d[d['Group'].notna()].sort_values(['Driver', 'LapNumber'])
    ls = np.arange(lo, hi + 1)
    per_lap = gap(ls) if to == 'slick' else -gap(ls)    # new minus old, per lap
    cost = np.array([per_lap[ls >= t].sum() for t in np.arange(lo, hi + 2)])
    best = lo + int(np.argmin(cost))
    rows = []
    for drv, x in d.groupby('Driver'):
        grp, lap = x['Group'].to_numpy(), x['LapNumber'].to_numpy()
        ch = np.flatnonzero((grp[1:] != grp[:-1]) & (grp[1:] == to) & (lap[1:] >= lo - 3) & (lap[1:] <= hi + 4))
        if not len(ch):
            continue
        s = int(lap[ch[0] + 1])
        t = min(max(s, lo), hi + 1)
        rows.append({'Year': year, 'Round': rnd, 'Driver': drv, 'To': to, 'SwitchLap': s, 'BestLap': best,
                     'LapsOff': s - best, 'Loss_s': cost[t - lo] - cost.min(), 'Clipped': t != s,
                     'BestAtEdge': best in (lo, hi + 1)})
    return pd.DataFrame(rows)


def lateness_cost(trend, laps_off):
    """Seconds lost by switching laps_off laps from the crossover on a gap that moves trend s per lap per lap
    (the line's own answer: |trend| x k^2 / 2 for k laps either side)."""
    return abs(trend) * np.asarray(laps_off, dtype=float) ** 2 / 2


def crossover_study(laps, cfg, seed=0):
    """Every crossover window: the fitted gap, the crossover lap with a bootstrap interval over drivers, and every
    car's switch loss. Returns (windows table, switches table)."""
    c = cfg['wet']['crossover']
    cl = crossover_laps(laps, cfg)
    allg = laps[laps['Year'].isin(cl['Year'].unique())].assign(Group=features.tyre_group(laps['Compound'], cfg))
    rng = np.random.default_rng(seed)
    win, sw = windows(cl, cfg), []
    out = []
    for w in win.itertuples(index=False):
        rows = cl[(cl['Year'] == w.Year) & (cl['Round'] == w.Round)
                  & cl['LapNumber'].between(w.Start - c['pad_laps'], w.End + c['pad_laps'])]
        lap0 = (w.Start + w.End) / 2
        a, b = fit_gap(rows, lap0, c['driver_effects'])
        cross = lap0 - a / b if b else np.nan
        found = np.isfinite(cross) and w.Start - c['max_extrapolate_laps'] <= cross <= w.End + c['max_extrapolate_laps']
        drivers = rows['Driver'].unique()
        boot = []
        for _ in range(c['bootstrap']):
            pick = rng.choice(drivers, len(drivers))
            r = pd.concat([rows[rows['Driver'] == p].assign(Driver=f'{p}_{i}') for i, p in enumerate(pick)])
            ba, bb = fit_gap(r, lap0, c['driver_effects'])
            if np.isfinite(bb) and bb != 0:
                boot.append(lap0 - ba / bb)
        circuit = rows['Circuit'].iloc[0] if 'Circuit' in rows else ''
        n = rows.groupby('Group')['Driver'].nunique()
        out.append({**w._asdict(), 'Circuit': circuit, 'SlickCars': n.get('slick', 0), 'WetCars': n.get('wet', 0),
                    'GapAtMid_s': a, 'GapTrend_s': b, 'Crossover': cross, 'Found': found,
                    'Cross_lo': np.percentile(boot, 10) if boot else np.nan,
                    'Cross_hi': np.percentile(boot, 90) if boot else np.nan,
                    'Cost1Lap_s': lateness_cost(b, 1), 'Cost3Laps_s': lateness_cost(b, 3)})
        if np.isfinite(b) and b != 0:
            s = switch_losses(allg, w.Year, w.Round, w.Start, w.End, lambda l, a=a, b=b, lap0=lap0: a + b * (l - lap0), b)
            per = fit_lap_gaps(rows, c['driver_effects']).reindex(range(w.Start, w.End + 1)).interpolate().bfill().ffill()
            if len(s) and per.notna().all():
                sp = switch_losses(allg, w.Year, w.Round, w.Start, w.End, lambda l, per=per: per.reindex(l).to_numpy(), b)
                s = s.merge(sp[['Driver', 'BestLap', 'Loss_s']].rename(columns={'BestLap': 'BestLapPerLap', 'Loss_s': 'LossPerLap_s'}),
                            on='Driver', how='left')
            if len(s):
                sw.append(s.assign(Start=w.Start, End=w.End, Found=found))
    return pd.DataFrame(out), (pd.concat(sw, ignore_index=True) if sw else pd.DataFrame())
