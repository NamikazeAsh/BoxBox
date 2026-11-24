"""Fixed strategies. A strategy is a list of stints, each a dry compound, the lap it starts on and the tyre's age
on that lap; a stop is made at the end of the lap before each later stint starts (in lap; the pit loss covers the
in and out laps). Every source gives the same stints frame (Draw, Driver, Stint, Compound, StartLap, StartAge):

* real_stints: what each car ran in a past race (data/processed/stints.parquet);
* model_stints: drawn from the rival pit model (pitdecision.PlanPolicy), one strategy per draw. Valid because
  v1 rivals never react to other cars, so drawing the whole race up front is the same as deciding lap by lap;
* legal_strategies: every legal strategy for the user's car on a grid of stop laps;
* override: a table the user supplies replaces given cars.

to_arrays turns a stints frame into the engine's padded arrays; strategy_grid lays many strategies of one car out
lap by lap for timing them all at once.
"""
from itertools import combinations, product

import numpy as np
import pandas as pd

from boxbox.degradation.task import COMPOUNDS

STINTS = ['Draw', 'Driver', 'Stint', 'Compound', 'StartLap', 'StartAge']
CODE = {c: i for i, c in enumerate(COMPOUNDS)}


def real_stints(stints, year, rnd, drivers=None, compounds=COMPOUNDS):
    """A past race's stints (draw 0), dry compounds only. A driver with a non dry stint is left out. A stint that
    carries on the same tyre after a pit lane pass is merged into the one before. compounds: the tyres allowed (the
    wet scenario tool, simulator/wet.py, passes the wet tyres as well)."""
    s = stints[(stints['Year'] == year) & (stints['Round'] == rnd)]
    if drivers is not None:
        s = s[s['Driver'].isin(drivers)]
    bad = s.loc[~s['Compound'].isin(compounds), 'Driver'].unique()
    s = s[~s['Driver'].isin(bad)].sort_values(['Driver', 'Stint'])
    # a new stint on the same tyre (same compound, age carrying on: the field led through the pit lane under a safety
    # car) is not a stop: merge it into the stint before (features.pit_labels, pit_decision.tyre_change_only)
    prev = s.groupby('Driver')
    end_age = prev['TyreLifeStart'].shift(1) + (s['StartLap'] - prev['StartLap'].shift(1))   # the old tyre's age on this lap
    same = (prev['Compound'].shift(1) == s['Compound']) & (s['TyreLifeStart'] == end_age)
    s = s[~same]
    out = s.assign(Draw=0, Stint=s.groupby('Driver').cumcount() + 1, StartAge=s['TyreLifeStart'].fillna(1.0))
    return out[STINTS].reset_index(drop=True)


def model_stints(policy, starts, draws, seed=0, status=None):
    """Strategies drawn from the pit model for each start (race_starts rows): one per draw and driver. status: the
    track status of each draw's laps (draws x laps + 1, features.NEUTRAL_CODES; see pitdecision.replay), so rivals
    react to safety cars and VSCs; None is every lap green (v1)."""
    from boxbox import pitdecision
    _, events = pitdecision.replay(policy, starts, draws, seed, record=True, status=status)
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


def usage_ratio(s):
    """Per circuit and compound (every compound at every circuit in s): the compound's share of the circuit's dry
    race laps over its share of all circuits' laps, capped at 1. Near 0 where teams barely ran it there."""
    laps = s.groupby(['Circuit', 'Compound'])['Laps'].sum().unstack(fill_value=0).reindex(columns=COMPOUNDS, fill_value=0)
    here = laps.div(laps.sum(axis=1), axis=0)
    overall = laps.sum() / laps.to_numpy().sum()
    return here.div(overall, axis=1).clip(upper=1.0).stack().rename('Usage')


def stint_caps(stints, events, wet, years, cfg):
    """Longest stint, as tyre age at the stint's end, each compound has run at each circuit: the
    simulator.race.stint_cap.quantile of end ages over the dry races of years, pulled toward a prior cap with
    weight shrink stints. A table (Circuit, Compound, Cap, Stints) plus the all circuit caps under Circuit None.
    Strategies past what teams have run lean on wear curves fitted on few laps (Softs at Monza), so the search
    keeps within it.

    The prior (simulator.search.cap_prior): 'global' (v1) is the compound's all circuit quantile, so a compound
    teams almost never ran at a circuit (Softs at Austin, 1 to 3 stints) gets nearly the all circuit cap and the
    search runs it on wear and gaps borrowed from other circuits. 'usage' scales that quantile by
    min(1, usage ratio / cap_usage_full) (usage_ratio): teams' avoidance of a compound there is evidence the
    borrowed estimate is too kind, and the lap data cannot say otherwise (no laps). A compound run at least
    cap_usage_full times its usual rate keeps the v1 prior. A compound never run at a circuit with training races
    gets cap 0 there under 'usage'; a circuit with no training races uses the all circuit caps."""
    c = cfg['simulator']['race']['stint_cap']
    search = cfg['simulator'].get('search', {})
    prior = search.get('cap_prior', 'global')
    s = stints[stints['Year'].isin(years) & stints['Compound'].isin(COMPOUNDS)]
    s = s[[(y, r) not in wet for y, r in zip(s['Year'], s['Round'])]].merge(events[['Year', 'Round', 'Circuit']], on=['Year', 'Round'])
    s = s.assign(EndAge=s['TyreLifeStart'].fillna(1.0) + s['Laps'] - 1)
    q = c['quantile']
    glob = s.groupby('Compound')['EndAge'].quantile(q)
    g = s.groupby(['Circuit', 'Compound'])['EndAge'].agg(Q=lambda x: x.quantile(q), Stints='size')
    if prior == 'usage':
        g = g.reindex(usage_ratio(s).index).join(usage_ratio(s))
        g['Stints'] = g['Stints'].fillna(0).astype(int)
        g['Q'] = g['Q'].fillna(0.0)
        scale = np.minimum(g['Usage'].to_numpy() / search.get('cap_usage_full', 0.5), 1.0)
        target = g.index.get_level_values('Compound').map(glob).to_numpy() * scale
    elif prior == 'global':
        target = g.index.get_level_values('Compound').map(glob).to_numpy()
    else:
        raise ValueError(f'unknown simulator.search.cap_prior {prior!r}')
    g = g.reset_index()
    g['Cap'] = (g['Stints'] * g['Q'] + c['shrink'] * target) / (g['Stints'] + c['shrink'])
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
    cap. The start tyre is given, not chosen, so its cap never goes below a min_stint_laps opening stint (a car
    can always pit early off a tyre the caps would not pick). Returns a list of (compounds, stop laps)."""
    r, max_age = cfg['simulator']['race'], cfg['simulator']['max_age']
    step, lo = r['stop_lap_step'], r['min_stint_laps']
    out = []
    first_floor = lo + start_age - 1
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
                if caps is None or _fits(ends, comps, caps, first_floor):
                    out.append((comps, stops))
    return out


def strategy_grid(strategies, race_laps, start_age=1.0):
    """Lap by lap arrays for many strategies of one car (strategies x laps): Comp (compound index), Age (tyre age
    on the lap, the start tyre counting its old laps) and the stop count per strategy. With LapModel.free_air on
    them, a whole legal set is timed in free air at once (the coarse pass without the engine)."""
    n = len(strategies)
    comp, age, stops = np.zeros((n, race_laps), int), np.zeros((n, race_laps)), np.zeros(n)
    for i, (comps, stop_laps) in enumerate(strategies):
        edges = (0,) + tuple(stop_laps) + (race_laps,)
        for j, (c, a, b) in enumerate(zip(comps, edges[:-1], edges[1:])):
            comp[i, a:b] = CODE[c]
            age[i, a:b] = np.arange(1, b - a + 1) + (start_age - 1 if j == 0 else 0)
        stops[i] = len(stop_laps)
    return {'Comp': comp, 'Age': age, 'Stops': stops}


def within_caps(comps, stops, race_laps, caps, cfg, start_age=1.0):
    """Whether one strategy keeps every stint within the caps as legal_strategies applies them (the start tyre's
    floor included); True with no caps."""
    if caps is None:
        return True
    lengths = np.diff((0,) + tuple(stops) + (race_laps,))
    ends = lengths + np.r_[start_age - 1, np.zeros(len(stops))]
    return _fits(ends, comps, caps, cfg['simulator']['race']['min_stint_laps'] + start_age - 1)


def _fits(ends, comps, caps, first_floor):
    """Each stint's end age within its compound's cap; the start tyre's cap is at least first_floor."""
    return bool(ends[0] <= max(caps[comps[0]], first_floor) and all(e <= caps[k] for e, k in zip(ends[1:], comps[1:])))


def strategy_stints(comps, stops, driver, draw=0, start_age=1.0):
    """One strategy as stints rows."""
    starts = (1,) + tuple(s + 1 for s in stops)
    return pd.DataFrame({'Draw': draw, 'Driver': driver, 'Stint': np.arange(1, len(comps) + 1), 'Compound': comps,
                         'StartLap': np.asarray(starts, float), 'StartAge': [start_age] + [1.0] * len(stops)})


def to_arrays(stints, drivers, draws, codes=None):
    """Padded arrays (draw x car x stint): Stop (the lap a stint ends with a stop, inf for the last stint),
    Comp (compound index) and Age (tyre age on the stint's first lap). Draw 0 only applies to every draw. codes:
    compound to index map (default CODE; simulator/wet.py adds the wet tyres)."""
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
    comp[d, c, j] = s['Compound'].map(CODE if codes is None else codes).to_numpy()
    age[d, c, j] = s['StartAge'].to_numpy(float)
    return {'Stop': stop, 'Comp': comp, 'Age': age}
