"""Strategy Room data (app/views/strategy_room.py): the strategy search for the test season's races, built ahead
of time by scripts/build_strategy_room.py because one search takes several seconds per driver.

Honest inputs: the simulator inputs and the rival pit model are fitted on the seasons before the test season
(evaluate.season_splits on the overtaking seasons, as scripts/backtest_simulator.py does), never on the final fit
that includes it. Every setting comes from config.yaml at build time: the simulator block for the engine, the
passing rule, the pace model and the search; dashboard.strategy_room for the races, draws and drivers.

Per race the field is drawn once (Field): the cars (classified finishers with a dry strategy, in grid order),
each car's pace draws and the rivals' strategies drawn from the pit model. backtest.strategy_search draws the
same field for every driver of a race, so it is drawn once here and each driver's search, the place
distributions and the page's strategy builder all race on the same draws with the same seed: a strategy raced
in the builder lands exactly on its row of the search.
"""
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from boxbox import features, overtaking as ot, pitdecision as pdm
from boxbox.config import seasons_for
from boxbox.degradation.task import COMPOUNDS
from boxbox.evaluate import season_splits
from boxbox.simulator import backtest as bt, pace as pc, race as rc, strategy as st
from boxbox.simulator.inputs import SimInputs

RACE = ['Year', 'Round']
LETTER = {c[0]: c for c in COMPOUNDS}   # S, M, H as race.best_strategies writes them
TABLES = ['meta', 'cars', 'bases', 'rivals', 'search', 'places']
OPTIONAL = ['status']   # neutral laps per draw, written only when a field carries a status
NEEDS = ['laps', 'results', 'qualifying', 'sprint_results', 'events', 'pit_loss_stops', 'stints']


def settings(cfg):
    """dashboard.strategy_room with the simulator defaults filled in."""
    r = cfg['simulator']['race']
    s = dict(cfg['dashboard'].get('strategy_room') or {})
    defaults = {'races': 'green', 'draws': r['draws'], 'seed': r['seed'], 'drivers': None, 'chart_strategies': 3,
                'table_rows': 10, 'use_caps': True}
    return {**defaults, **{k: v for k, v in s.items() if v is not None or k == 'drivers'}}


def split(cfg):
    """(training seasons, test season) for the inputs and the rival pit model."""
    return season_splits(seasons_for(cfg, 'overtaking'))['test']


def races(laps, years, kind='green'):
    """Races the room covers. green: dry races with no safety car, VSC or red flag lap (what simulator v1
    races); dry: every dry race, for a simulator that samples neutral laps. Widen here when the simulator does."""
    if kind == 'green':
        return bt.green_races(laps, years)
    if kind == 'dry':
        l = laps[laps['Year'].isin(years) & ~laps['WetRace']]
        return l[RACE + ['Circuit']].drop_duplicates().sort_values(RACE).reset_index(drop=True)
    raise ValueError(f'unknown race kind {kind!r} (dashboard.strategy_room.races: green or dry)')


def fit(t, pre, stops, train, cfg):
    """SimInputs and the rival PlanPolicy fitted on train, with the passing rule and pace model config.yaml names."""
    s = cfg['simulator']
    rule = ot.candidates(cfg)[s['overtaking']['chosen']]
    pace = pc.candidates(cfg)[s['pace']['chosen']]
    inputs = SimInputs(cfg).fit(t, pre, train, rule, pace)
    policy = pdm.fit_plan_policy(t['laps'], pre, t['results'], stops, train, cfg)
    return inputs, policy


@dataclass
class Field:
    """One race's field. cars: Driver, Team, Lineage, Grid, Finish, StartCompound, StartAge in grid order (the
    engine's car order); bases: draws x cars, s; rivals: stints frame (strategy.STINTS) for every car."""
    year: int
    rnd: int
    circuit: str
    race_laps: int
    cars: pd.DataFrame
    bases: np.ndarray
    rivals: pd.DataFrame
    seed: int = 0
    status: np.ndarray = None   # neutral laps, draws x race laps + 1 (race.simulate's status); None for green races

    @property
    def drivers(self):
        return self.cars['Driver'].tolist()

    @property
    def draws(self):
        return self.bases.shape[0]

    @property
    def kw(self):
        """Extra engine keyword arguments: none for a green race, status once the field carries one."""
        return {} if self.status is None else {'status': self.status}


def draw_field(inputs, policy, t, pre, starts, year, rnd, circuit, draws, seed, cfg):
    """The field as backtest.strategy_search draws it: model pace, rivals from the pit model (real strategies
    for cars with no lap 1 tyre data). starts: pitdecision.race_starts for the season (green_only False).
    Returns the Field and the race's real stints."""
    laps = t['laps']
    truth = bt.race_truth(laps, t['results'], year, rnd)
    real = st.real_stints(t['stints'], year, rnd, truth['Driver'])
    truth = truth[truth['Driver'].isin(real['Driver'])].reset_index(drop=True)
    drivers = truth['Driver'].tolist()
    race_laps = int(laps.loc[(laps['Year'] == year) & (laps['Round'] == rnd), 'LapNumber'].max())
    cars = truth[['Driver']].assign(Year=year, Round=rnd).merge(pre, on=RACE + ['Driver'], how='left')
    rng = np.random.default_rng(seed)
    ref = None
    quali = (getattr(inputs, 'reference', 'last_season') == 'quali' and getattr(inputs, 'quali_ref', None) is not None
             and inputs.quali_ref.ref(year, rnd, circuit) is not None)
    if circuit not in inputs.refs and not quali:   # places only depend on relative pace; the race's own median sets the scale
        b = inputs.lap.driver_bases(inputs.lap.stint_levels(laps, [year]))
        ref = float(b[(b['Year'] == year) & (b['Round'] == rnd)]['Base'].median())
    bases = inputs.sample_bases(cars, circuit, draws, rng, ref=ref)
    status = bt.sample_status(inputs, circuit, race_laps, draws, seed)   # sampled safety cars, as backtest.strategy_search
    s = starts[(starts['Year'] == year) & (starts['Round'] == rnd) & starts['Driver'].isin(drivers)].reset_index(drop=True)
    rivals = bt.model_strategies(policy, s, draws, seed, status)
    missing = set(drivers) - set(s['Driver'])
    rivals = st.override(rivals, real[real['Driver'].isin(missing)]) if missing else rivals
    first = real.sort_values('Stint').groupby('Driver').first()
    res = t['results']
    team = res[(res['Year'] == year) & (res['Round'] == rnd)].set_index('Driver')['Team']
    out = truth[['Driver', 'Grid', 'Finish', 'Road', 'Dsq']].assign(
        Team=truth['Driver'].map(team), StartCompound=truth['Driver'].map(first['Compound']),
        StartAge=truth['Driver'].map(first['StartAge']).astype(float))
    out['Lineage'] = features.team_lineage(out['Team'], cfg)
    return Field(year, rnd, circuit, race_laps, out, bases, rivals.reset_index(drop=True), seed, status), real


def real_strategy(real, driver):
    """(compounds, stop laps) a driver ran, as backtest.strategy_search writes it."""
    mine = real[real['Driver'] == driver].sort_values('Stint')
    return tuple(mine['Compound']), tuple(int(x) - 1 for x in mine['StartLap'].iloc[1:])


def label(comps, stops):
    """Compounds and stop laps in race.best_strategies' strings ('M-H', '22')."""
    return '-'.join(c[0] for c in comps), ', '.join(map(str, stops))


def parse(compounds, stop_laps):
    """Back from race.best_strategies' strings to (compounds, stop laps)."""
    comps = tuple(LETTER[x] for x in compounds.split('-'))
    stops = tuple(int(x) for x in str(stop_laps).split(',') if x.strip())
    return comps, stops


def _arrays(fld, driver, comps, stops, cfg=None):
    """Engine arrays with driver on (comps, stops) and every rival on its drawn strategy, built as
    race.best_strategies builds them (so the same seed gives the same race), including its move of planned stops
    onto a neutral lap when the field has a status."""
    car = fld.cars.set_index('Driver').loc[driver]
    start_age = float(car['StartAge']) if comps[0] == car['StartCompound'] else 1.0   # another start tyre is a new set
    rivals = fld.rivals[fld.rivals['Driver'] != driver]
    mine = st.strategy_stints(tuple(comps), tuple(stops), driver, 0, start_age)
    if rivals['Draw'].nunique() != 1:
        mine = pd.concat([mine.assign(Draw=d) for d in rivals['Draw'].unique()])
    arr = st.to_arrays(pd.concat([rivals, mine]), fld.drivers, fld.draws)
    r = (cfg or {}).get('simulator', {}).get('race', {})
    if fld.status is not None and r.get('neutral_react_window'):
        arr = rc.react_to_neutral(arr, fld.status, [fld.drivers.index(driver)], r['neutral_react_window'], r['min_stint_laps'])
    return arr


def race_strategy(inputs, fld, driver, comps, stops, cfg=None):
    """Race driver on (comps, stops) against the field: per draw finishing place and race time, s. cfg: needed only
    for a field with a status (simulator.race.neutral_react_window)."""
    arr = _arrays(fld, driver, comps, stops, cfg)
    u = fld.drivers.index(driver)
    # as race.best_strategies races it: rivals may answer the searched car's stop (simulator.engine.cover_search)
    res = rc.simulate(inputs, fld.circuit, fld.race_laps, fld.bases, arr, seed=fld.seed, **fld.kw,
                      cover=bool(rc.engine_options(inputs).get('cover_search', False)), fixed_cars=[u])
    return res['Place'][:, u].astype(int), res['Time'][:, u]


def search(inputs, fld, real, driver, cfg, use_caps=True):
    """race.best_strategies for driver against the field, starting on its real tyre, its real strategy raced too
    (tidy: one row per distinct strategy, best first, with Rank and Real)."""
    comps, stops = real_strategy(real, driver)
    car = fld.cars.set_index('Driver').loc[driver]
    free = cfg['simulator'].get('search', {}).get('start_tyre', 'real') == 'free'
    best = rc.best_strategies(inputs, fld.circuit, fld.race_laps, fld.drivers, fld.bases, fld.rivals, driver, cfg,
                              start_age=float(car['StartAge']), start_compounds=None if free else [car['StartCompound']],
                              start_ages={car['StartCompound']: float(car['StartAge'])} if free else None,
                              extra=[(comps, stops)], use_caps=use_caps, seed=fld.seed, **fld.kw)
    return tidy(best)


def tidy(best):
    """A best_strategies table with each strategy once. The real strategy is raced as an extra row and may also be a
    searched candidate (same race, same result): keep the better ranked copy and mark it Real."""
    b = best.copy()
    b['Real'] = b.groupby(['Compounds', 'StopLaps'])['Extra'].transform('any')
    b = b.drop_duplicates(['Compounds', 'StopLaps'], keep='first').drop(columns='Extra').reset_index(drop=True)
    b.insert(0, 'Rank', np.arange(1, len(b) + 1))
    return b


def chart_rows(table, top):
    """Ranks of the strategies the page charts: the top ones and the real one."""
    return sorted(set(table['Rank'].head(top)) | set(table.loc[table['Real'], 'Rank']))


def race_places(inputs, fld, driver, table, top, cfg=None):
    """Per draw places of the charted strategies (chart_rows): Rank, Draw, Place. Each strategy's mean must match
    its search row (common random numbers); a mismatch raises, as it means the field is not the search's."""
    out = []
    for rank in chart_rows(table, top):
        row = table.loc[table['Rank'] == rank].iloc[0]
        place, _ = race_strategy(inputs, fld, driver, *parse(row['Compounds'], row['StopLaps']), cfg=cfg)
        if not np.isclose(place.mean(), row['MeanPlace']):
            raise RuntimeError(f'{driver} {row["Compounds"]} {row["StopLaps"]}: raced again {place.mean():.3f}, '
                               f'search {row["MeanPlace"]:.3f}; the field differs from the search\'s')
        out.append(pd.DataFrame({'Rank': rank, 'Draw': np.arange(len(place)), 'Place': place}))
    return pd.concat(out, ignore_index=True)


def check_strategy(comps, stops, race_laps, min_stint):
    """Problems with a strategy the user built (empty when it can be raced): one compound per stint, stops in
    order inside the race, stints of at least min_stint laps, two dry compounds at least."""
    out = []
    if len(comps) != len(stops) + 1:
        out.append('Pick one compound per stint.')
    edges = (0,) + tuple(stops) + (race_laps,)
    if list(stops) != sorted(set(stops)) or min(stops, default=1) < 1 or max(stops, default=0) >= race_laps:
        out.append(f'Stop laps must rise from one stop to the next, between lap 1 and lap {race_laps - 1}.')
    elif min(np.diff(edges)) < min_stint:
        out.append(f'Every stint needs at least {min_stint} laps.')
    if len(set(comps)) < 2:
        out.append('Use at least two different dry compounds (the two compound rule).')
    return out


def summary(place, best_place=None):
    """Mean place, 10th and 90th percentile, P(win), P(podium), P(points) of per draw places, and against a
    paired best (same draws): the mean place difference and the share of draws ahead of it (ties half)."""
    p = np.asarray(place, float)
    out = {'MeanPlace': p.mean(), 'Place10': np.quantile(p, 0.1), 'Place90': np.quantile(p, 0.9),
           'Win': (p == 1).mean(), 'Podium': (p <= 3).mean(), 'Points': (p <= 10).mean()}
    if best_place is not None:
        d = p - np.asarray(best_place, float)
        out['DeltaPlace'] = d.mean()
        out['PBeatsBest'] = (d < 0).mean() + 0.5 * (d == 0).mean()
    return out


# Build (scripts/build_strategy_room.py)

def build(t, cfg, rounds=None, drivers=None, check=True, log=print):
    """Every table of the room and the inputs it raced on. t: the processed tables (NEEDS). rounds: only these
    rounds of the test season; drivers: override dashboard.strategy_room.drivers. check: run the first search
    through backtest.strategy_search too and stop if the two disagree (the field here must stay the search's)."""
    s = settings(cfg)
    train, year = split(cfg)
    laps = t['laps']
    t0 = time.time()
    pre_race = getattr(pc, 'pre_race', pdm.pre_race)   # simulator pace rows: pitdecision's plus the finish form
    pre = pre_race(t['results'], t['qualifying'], t['sprint_results'], t['events'], laps, cfg)
    stops = pdm.stops_with_circuit(t['pit_loss_stops'], t['events'])
    inputs, policy = fit(t, pre, stops, train, cfg)
    log(f'inputs and rival pit model fitted on {train[0]} to {train[-1]} in {time.time() - t0:.0f} s')
    starts = pdm.race_starts(laps, pre, t['results'], [year], cfg, green_only=False)
    todo = races(laps, [year], s['races'])
    if rounds:
        todo = todo[todo['Round'].isin(rounds)]
    names = t['events'].set_index(RACE)['EventName']
    engine = getattr(rc, 'engine_options', lambda i: {})(inputs)
    out = {k: [] for k in TABLES}
    top = drivers if drivers is not None else s['drivers']
    for race in todo.itertuples():
        t1 = time.time()
        fld, real = draw_field(inputs, policy, t, pre, starts, year, race.Round, race.Circuit, s['draws'], s['seed'], cfg)
        who = fld.cars.sort_values('Road' if 'Road' in fld.cars else 'Finish')['Driver'].tolist()[:top or None]
        for d in who:
            table = search(inputs, fld, real, d, cfg, s['use_caps'])
            if check:
                theirs, _, _ = bt.strategy_search(inputs, policy, t, pre, year, race.Round, race.Circuit, d, cfg, s['draws'],
                                                  use_caps=s['use_caps'], seed=s['seed'])
                theirs = tidy(theirs)
                same = len(theirs) == len(table) and np.allclose(theirs['MeanPlace'], table['MeanPlace']) and \
                    (theirs['StopLaps'] == table['StopLaps']).all()
                if not same:
                    raise RuntimeError('the room\'s search no longer matches backtest.strategy_search: update draw_field')
                log('  checked against backtest.strategy_search: same table')
                check = False
            out['search'].append(table.assign(Year=year, Round=race.Round, Driver=d))
            out['places'].append(race_places(inputs, fld, d, table, s['chart_strategies'], cfg).assign(Year=year, Round=race.Round, Driver=d))
        cars, bases, rivals, status = field_tables(fld)
        out['cars'].append(cars.assign(Searched=cars['Driver'].isin(who)))
        out['bases'].append(bases)
        out['rivals'].append(rivals)
        if status is not None:
            out.setdefault('status', []).append(status)
        secs = time.time() - t1
        out['meta'].append({'Year': year, 'Round': race.Round, 'Circuit': race.Circuit, 'EventName': names.get((year, race.Round), ''),
                            'RaceLaps': fld.race_laps, 'Cars': len(fld.cars), 'Searched': len(who), 'Draws': fld.draws,
                            'Seed': fld.seed, 'Races': s['races'], 'TrainFrom': train[0], 'TrainTo': train[-1],
                            'Neutral': fld.status is not None, 'Rule': inputs.rule.name, 'Pace': inputs.pace.name, 'WearModel': cfg['simulator']['wear_model'],
                            'Bootstrap': inputs.lap.n_boot, 'Engine': ', '.join(f'{k} {v}' for k, v in engine.items()),
                            'UseCaps': s['use_caps'], 'StartTyre': cfg['simulator'].get('search', {}).get('start_tyre', 'real'), 'Seconds': round(secs, 1), 'Built': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')})
        log(f'{year} round {race.Round} {race.Circuit}: {len(who)} drivers searched in {secs:.0f} s')
    first = ['Year', 'Round', 'Driver']
    tables = {k: (pd.DataFrame(v) if k == 'meta' else pd.concat(v, ignore_index=True)) for k, v in out.items()}
    for k in ['search', 'places']:
        tables[k] = tables[k][first + [c for c in tables[k].columns if c not in first]]
    return tables, inputs.release()


def save(tables, inputs, cfg):
    d = cfg['paths']['models']
    d.mkdir(exist_ok=True)
    for k in OPTIONAL:
        if k not in tables:
            (d / f'strategy_room_{k}.parquet').unlink(missing_ok=True)   # no stale status from an earlier build
    for k, v in tables.items():
        v.to_parquet(d / f'strategy_room_{k}.parquet', index=False)
    joblib.dump(inputs, d / 'strategy_room_inputs.joblib')
    return d


def load(cfg):
    """The saved tables (FileNotFoundError if not built)."""
    d = cfg['paths']['models']
    out = {k: pd.read_parquet(d / f'strategy_room_{k}.parquet') for k in TABLES}
    more = [k for k in OPTIONAL if (d / f'strategy_room_{k}.parquet').exists()]
    return {**out, **{k: pd.read_parquet(d / f'strategy_room_{k}.parquet') for k in more}}


def load_inputs(cfg):
    return joblib.load(cfg['paths']['models'] / 'strategy_room_inputs.joblib')


# Saved tables and back

def field_tables(fld):
    """A Field as tidy rows: cars (with Car, the engine's car index), bases (Draw, Driver, Base_s), rivals, and
    status (Draw, Lap, Code for every neutral lap; None for a green field)."""
    key = {'Year': fld.year, 'Round': fld.rnd}
    cars = fld.cars.assign(**key, Car=np.arange(len(fld.cars)))
    d, n = fld.bases.shape
    bases = pd.DataFrame({**key, 'Draw': np.repeat(np.arange(d), n), 'Driver': np.tile(fld.drivers, d), 'Base_s': fld.bases.ravel()})
    status = None
    if fld.status is not None:
        s = np.broadcast_to(np.asarray(fld.status), (d, fld.race_laps + 1))
        draw, lap = np.nonzero(s)
        status = pd.DataFrame({**key, 'Draw': draw, 'Lap': lap, 'Code': s[draw, lap].astype(int)})
    return cars, bases, fld.rivals.assign(**key), status


def field_from_tables(tables, year, rnd):
    """The Field of one race from the saved tables (the page's strategy builder)."""
    pick = lambda d: d[(d['Year'] == year) & (d['Round'] == rnd)]
    m = pick(tables['meta']).iloc[0]
    c = pick(tables['cars']).sort_values('Car').reset_index(drop=True)
    b = pick(tables['bases']).pivot(index='Draw', columns='Driver', values='Base_s')[c['Driver']].to_numpy()
    r = pick(tables['rivals'])[st.STINTS].reset_index(drop=True)
    cols = [k for k in ['Driver', 'Team', 'Lineage', 'Grid', 'Finish', 'Road', 'Dsq', 'StartCompound', 'StartAge'] if k in c]
    status = None
    if bool(m.get('Neutral', False)):
        status = np.zeros((b.shape[0], int(m['RaceLaps']) + 1), np.int8)
        s = pick(tables['status'])
        status[s['Draw'].to_numpy(int), s['Lap'].to_numpy(int)] = s['Code'].to_numpy(np.int8)
    return Field(int(year), int(rnd), m['Circuit'], int(m['RaceLaps']), c[cols], b, r, int(m['Seed']), status)
