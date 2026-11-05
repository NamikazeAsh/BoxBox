"""Fixed strategies. A strategy is a list of stints, each a dry compound, the lap it starts on and the tyre's age
on that lap; a stop is made at the end of the lap before each later stint starts (in lap; the pit loss covers the
in and out laps). Every source gives the same stints frame (Draw, Driver, Stint, Compound, StartLap, StartAge):

* real_stints: what each car ran in a past race (data/processed/stints.parquet);
* model_stints: drawn from the rival pit model (pitdecision.PlanPolicy), one strategy per draw. Valid because
  v1 rivals never react to other cars, so drawing the whole race up front is the same as deciding lap by lap;
* legal_strategies: every legal strategy for the user's car on a grid of stop laps;
* override: a table the user supplies replaces given cars.

to_arrays turns a stints frame into the engine's padded arrays.
"""
from itertools import combinations, product

import numpy as np
import pandas as pd

from boxbox.degradation.task import COMPOUNDS

STINTS = ['Draw', 'Driver', 'Stint', 'Compound', 'StartLap', 'StartAge']
CODE = {c: i for i, c in enumerate(COMPOUNDS)}


def real_stints(stints, year, rnd, drivers=None):
    """A past race's stints (draw 0), dry compounds only. A driver with a non dry stint is left out."""
    s = stints[(stints['Year'] == year) & (stints['Round'] == rnd)]
    if drivers is not None:
        s = s[s['Driver'].isin(drivers)]
    bad = s.loc[~s['Compound'].isin(COMPOUNDS), 'Driver'].unique()
    s = s[~s['Driver'].isin(bad)].sort_values(['Driver', 'Stint'])
    out = s.assign(Draw=0, Stint=s.groupby('Driver').cumcount() + 1, StartAge=s['TyreLifeStart'].fillna(1.0))
    return out[STINTS].reset_index(drop=True)


def model_stints(policy, starts, draws, seed=0):
    """Strategies drawn from the pit model for each start (race_starts rows): one per draw and driver."""
    from boxbox import pitdecision
    _, events = pitdecision.replay(policy, starts, draws, seed, record=True)
    first = pd.DataFrame({'Draw': np.tile(np.arange(draws), len(starts)),   # replay's row order
                          'Driver': np.repeat(starts['Driver'].to_numpy(), draws),
                          'Compound': np.repeat(starts['Compound'].to_numpy(), draws),
                          'StartLap': 1.0, 'StartAge': np.repeat(starts['TyreLife'].to_numpy(float), draws)})
    later = events.assign(StartLap=events['Lap'] + 1.0, StartAge=1.0)[['Draw', 'Driver', 'Compound', 'StartLap', 'StartAge']]
    s = pd.concat([first, later], ignore_index=True).sort_values(['Draw', 'Driver', 'StartLap'])
    s['Stint'] = s.groupby(['Draw', 'Driver']).cumcount() + 1
    return s[STINTS].reset_index(drop=True)


def override(stints, table):
    """Replace the strategies of the cars in table (same columns, any draws: a table with Draw 0 only applies
    to every draw) and keep the rest."""
    draws = stints['Draw'].unique()
    t = table if table['Draw'].nunique() > 1 else pd.concat([table.assign(Draw=d) for d in draws], ignore_index=True)
    return pd.concat([stints[~stints['Driver'].isin(table['Driver'])], t[STINTS]], ignore_index=True)


def stint_caps(stints, events, wet, years, cfg):
    """Longest stint, as tyre age at the stint's end, each compound has run at each circuit: the
    simulator.race.stint_cap.quantile of end ages over the dry races of years, pulled toward the compound's all
    circuit quantile with weight shrink stints. A table (Circuit, Compound, Cap, Stints) plus the all circuit
    caps under Circuit None. Strategies past what teams have run lean on wear curves fitted on few laps (Softs at
    Monza), so the search keeps within it."""
    c = cfg['simulator']['race']['stint_cap']
    s = stints[stints['Year'].isin(years) & stints['Compound'].isin(COMPOUNDS)]
    s = s[[(y, r) not in wet for y, r in zip(s['Year'], s['Round'])]].merge(events[['Year', 'Round', 'Circuit']], on=['Year', 'Round'])
    s = s.assign(EndAge=s['TyreLifeStart'].fillna(1.0) + s['Laps'] - 1)
    q = c['quantile']
    glob = s.groupby('Compound')['EndAge'].quantile(q)
    g = s.groupby(['Circuit', 'Compound'])['EndAge'].agg(Q=lambda x: x.quantile(q), Stints='size').reset_index()
    g['Cap'] = (g['Stints'] * g['Q'] + c['shrink'] * g['Compound'].map(glob)) / (g['Stints'] + c['shrink'])
    allc = pd.DataFrame({'Circuit': None, 'Compound': glob.index, 'Cap': glob.to_numpy(), 'Stints': s.groupby('Compound').size().to_numpy()})
    return pd.concat([g[['Circuit', 'Compound', 'Cap', 'Stints']], allc], ignore_index=True)


def caps_for(caps, circuit):
    """{compound: cap, laps} at a circuit (the all circuit cap for a compound it has never run)."""
    allc = caps[caps['Circuit'].isna()].set_index('Compound')['Cap']
    here = caps[caps['Circuit'] == circuit].set_index('Compound')['Cap']
    return {k: float(here.get(k, allc.get(k, np.inf))) for k in COMPOUNDS}   # never run anywhere: no cap


def legal_strategies(race_laps, cfg, start_compounds=COMPOUNDS, start_age=1.0, caps=None):
    """Every legal strategy for one car: 1 to max_stops stops, two dry compounds at least, stops at the end of
    laps on a grid of stop_lap_step, every stint at least min_stint_laps and at most simulator.max_age laps
    (counting the start tyre's age), and, with caps ({compound: laps}, caps_for), no stint past its compound's
    cap. Returns a list of (compounds, stop laps)."""
    r, max_age = cfg['simulator']['race'], cfg['simulator']['max_age']
    step, lo = r['stop_lap_step'], r['min_stint_laps']
    out = []
    for n in range(1, r['max_stops'] + 1):
        laps = range(lo, race_laps - lo + 1, step * (2 if n >= 3 else 1))   # a coarser grid for 3 stops
        for stops in combinations(laps, n):
            edges = (0,) + stops + (race_laps,)
            lengths = np.diff(edges)
            if lengths.min() < lo or lengths[0] + start_age - 1 > max_age or lengths[1:].max() > max_age:
                continue
            ends = lengths + np.r_[start_age - 1, np.zeros(n)]   # tyre age at each stint's end
            for comps in product(COMPOUNDS, repeat=n + 1):
                if comps[0] not in start_compounds or len(set(comps)) < 2:
                    continue
                if caps is None or all(e <= caps[k] for e, k in zip(ends, comps)):
                    out.append((comps, stops))
    return out


def strategy_stints(comps, stops, driver, draw=0, start_age=1.0):
    """One strategy as stints rows."""
    starts = (1,) + tuple(s + 1 for s in stops)
    return pd.DataFrame({'Draw': draw, 'Driver': driver, 'Stint': np.arange(1, len(comps) + 1), 'Compound': comps,
                         'StartLap': np.asarray(starts, float), 'StartAge': [start_age] + [1.0] * len(stops)})


def to_arrays(stints, drivers, draws):
    """Padded arrays (draw x car x stint): Stop (the lap a stint ends with a stop, inf for the last stint),
    Comp (compound index) and Age (tyre age on the stint's first lap). Draw 0 only applies to every draw."""
    s = stints[stints['Driver'].isin(drivers)]
    if s['Draw'].nunique() == 1 and draws > 1:
        s = pd.concat([s.assign(Draw=d) for d in range(draws)], ignore_index=True)
    s = s.sort_values(['Draw', 'Driver', 'Stint'])
    k = int(s.groupby(['Draw', 'Driver']).size().max())
    car = pd.Series(range(len(drivers)), index=list(drivers))
    d, c = s['Draw'].to_numpy(int), car.reindex(s['Driver']).to_numpy()
    j = s.groupby(['Draw', 'Driver']).cumcount().to_numpy()
    if np.isnan(c.astype(float)).any() or s.groupby('Draw')['Driver'].nunique().min() < len(drivers):
        raise ValueError('every car needs a strategy in every draw')
    stop = np.full((draws, len(drivers), k), np.inf)
    comp = np.zeros((draws, len(drivers), k), int)
    age = np.ones((draws, len(drivers), k))
    nxt = s.groupby(['Draw', 'Driver'])['StartLap'].shift(-1).to_numpy()
    stop[d, c, j] = nxt - 1          # NaN for the last stint
    stop[np.isnan(stop)] = np.inf
    comp[d, c, j] = s['Compound'].map(CODE).to_numpy()
    age[d, c, j] = s['StartAge'].to_numpy(float)
    return {'Stop': stop, 'Comp': comp, 'Age': age}
