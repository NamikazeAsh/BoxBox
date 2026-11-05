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


def legal_strategies(race_laps, cfg, start_compounds=COMPOUNDS, start_age=1.0):
    """Every legal strategy for one car: 1 to max_stops stops, two dry compounds at least, stops at the end of
    laps on a grid of stop_lap_step, every stint at least min_stint_laps and at most simulator.max_age laps
    (counting the start tyre's age). Returns a list of (compounds, stop laps)."""
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
            for comps in product(COMPOUNDS, repeat=n + 1):
                if comps[0] in start_compounds and len(set(comps)) >= 2:
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
