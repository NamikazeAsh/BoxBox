"""Wet races in the simulator as a scenario tool, not a forecast (simulator/wet.py, notes/wet_simulation.md).

    python scripts/simulate_wet.py --scenario --race 2024 Barcelona --path "dry,wet@20,dry@35" --driver HAM [--neutral]
    python scripts/simulate_wet.py --replay 2024 Montreal [--driver VER --shift -2 [--nth 1]] [--search VER] [--track]
    python scripts/simulate_wet.py --backtest

--scenario: a past dry race's field (grid, oracle pace fitted on its laps, real strategies as dry plans) on a track the user gives:
the state on lap 1, then each lap from which the other tyre is faster. Rivals switch in a wave around each change;
the strategy search ranks the driver's switch laps. --neutral samples dry safety cars (simulator.wet.scenario_neutral).
--replay: a real wet race on its estimated track with real strategies, real neutral laps and oracle pace; --shift
moves one car's nth tyre switch (counterfactual), --search ranks that car's switch laps, --track prints the path.
--backtest: every wet race 2022 to 2025 replayed (real track path and the no track path baseline), place MAE
against the grid, as the dry backtest reports.
The saved simulator inputs (data/models/simulator_inputs.joblib, dry races 2022 to 2025) race every mode.
"""
import copy
import sys
import unicodedata
import warnings

import joblib
import numpy as np
import pandas as pd

from boxbox import overtaking as ot
from boxbox.config import load_config
from boxbox.simulator import backtest as bt, race as rc, strategy as st, wet

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.width', 220)
cfg = copy.deepcopy(load_config())
cfg['simulator']['wet']['enabled'] = True   # this script races the wet track by definition
w = cfg['simulator']['wet']
args = sys.argv[1:]
opt = lambda name, n=1: args[args.index(name) + 1: args.index(name) + 1 + n] if name in args else None
t = bt.load_tables(cfg)
laps = t['laps']
inputs = joblib.load(cfg['paths']['models'] / 'simulator_inputs.joblib')
draws, seed = int(w['draws']), int(w['seed'])


def find_race(year, name):
    plain = lambda x: unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode().lower()
    l = laps[laps['Year'] == int(year)]
    l = l[l['Circuit'].map(plain).str.contains(plain(name), regex=False)]
    if l.empty:
        raise SystemExit(f'no {year} race at a circuit matching {name}')
    return int(l['Round'].iloc[0]), l['Circuit'].iloc[0]


def show(tab, n=12):
    cols = ['Strategy', 'MeanPlace', 'Place10', 'Place90', 'Points', 'DeltaPlace', 'DeltaLo', 'DeltaHi', 'DeltaTime_s', 'Tier']
    print(tab[cols].head(n).round(3).to_string(index=False))


if '--scenario' in args:
    year, name = opt('--race', 2)
    rnd, circuit = find_race(year, name)
    year = int(year)
    path, user = opt('--path')[0], opt('--driver')[0]
    start, changes = wet.parse_path(path)
    truth = bt.race_truth(laps, t['results'], year, rnd)
    real = st.real_stints(t['stints'], year, rnd, truth['Driver'])
    truth = truth[truth['Driver'].isin(real['Driver'])].reset_index(drop=True)
    drivers = truth['Driver'].tolist()
    race_laps = int(wet.race_laps_of(laps, year, rnd)['LapNumber'].max())
    ob = wet.race_track(laps, inputs.lap, circuit, year, rnd, race_laps, cfg, group_terms=False)['bases']   # oracle pace
    known = truth['Driver'].map(ob).to_numpy(float)
    bases = np.tile(np.where(np.isnan(known), np.nanmedian(known), known), (draws, 1))
    rng = np.random.default_rng(seed)
    gap = wet.scenario_gap(race_laps, start, changes, wet.draw_trends(rng, draws, len(changes), cfg), cfg)
    plans = {d: wet.plan_of(real, d) for d in drivers}
    caps = st.caps_for(inputs.caps, circuit) if getattr(inputs, 'caps', None) is not None else None
    rivals = wet.scenario_stints(plans, start, changes, draws, race_laps, np.random.default_rng(seed + 1), cfg, caps)
    neutral = '--neutral' in args or bool(w['scenario_neutral'])
    status = inputs.neutral.sample(circuit, race_laps, draws, np.random.default_rng(seed + 2)) if neutral else None
    print(f'SCENARIO {year} {circuit} ({race_laps} laps), path {path!r}, searched car {user}, {draws} draws, '
          f'safety cars {"sampled (dry model)" if neutral else "off"}')
    q = np.quantile(gap, [0.1, 0.5, 0.9], axis=0)
    lapsel = sorted({1, race_laps} | {min(max(c + d, 1), race_laps) for c, _ in changes for d in (-3, -1, 0, 1, 3)})
    print('track gap (wet tyre minus slick, s; 10th, median, 90th percentile over draws):')
    print(pd.DataFrame({'Lap': lapsel, 'P10': q[0, lapsel], 'Median': q[1, lapsel], 'P90': q[2, lapsel]}).round(2).to_string(index=False))
    g = wet.group_of(rivals['Compound'], cfg)
    first_wet = rivals[g.to_numpy() == 'wet'].groupby(['Draw', 'Driver'])['StartLap'].min()
    print('rival wave, first lap on the wet tyre (share of cars x draws):')
    print((first_wet.value_counts(normalize=True).sort_index().round(3)).to_string())
    print(f'cars that never fit the wet tyre: {1 - len(first_wet) / (draws * len(drivers)):.3f}')
    tab = wet.search_switches(inputs, circuit, race_laps, bases, rivals, user, drivers, plans[user], start, changes,
                              wet.engine_track(gap, cfg), cfg, status=status, seed=seed, caps=caps)
    print(f'\n{user} real dry plan: {wet.label(plans[user])}; {tab["Candidates"].iloc[0]} candidates, '
          f'{len(tab)} raced after the free air pass')
    show(tab, 15)

elif '--replay' in args:
    year, name = opt('--replay', 2)
    rnd, circuit = find_race(year, name)
    year = int(year)
    s = wet.replay_setup(inputs, t, year, rnd, circuit, cfg, draws)
    print(f'REPLAY {year} {circuit} ({s["race_laps"]} laps), {len(s["drivers"])} cars, oracle pace, real strategies, '
          f'real neutral laps, {draws} draws')
    if '--track' in args:
        print(s['track']['table'].round(2).to_string())
    base = wet.replay(inputs, s, circuit, cfg, seed=seed)
    summ = rc.summarise(base, s['drivers']).assign(Grid=s['truth']['Grid'].to_numpy(), Finish=s['truth']['Finish'].to_numpy())
    print(summ[['Driver', 'Grid', 'Finish', 'MeanPlace', 'GapToWinner_s']].round(2).to_string(index=False))
    if '--shift' in args:
        drv, k = opt('--driver')[0], int(opt('--shift')[0])
        nth = int(opt('--nth')[0]) if '--nth' in args else 1
        moved = wet.shift_switch(s['real'], drv, k, cfg, nth)
        res = wet.replay(inputs, s, circuit, cfg, stints=moved, seed=seed)
        u = s['drivers'].index(drv)
        dt = res['Time'][:, u] - base['Time'][:, u]
        print(f'\ncounterfactual: {drv} switch {nth} moved {k:+d} laps ({wet.switches_of(s["real"], drv, cfg)[nth - 1][1]} '
              f'to {wet.switches_of(s["real"], drv, cfg)[nth - 1][1] + k}): mean place {base["Place"][:, u].mean():.2f} -> '
              f'{res["Place"][:, u].mean():.2f}, race time {np.median(dt):+.2f} s (10th to 90th {np.quantile(dt, 0.1):+.2f} '
              f'to {np.quantile(dt, 0.9):+.2f})')
    if '--search' in args:
        drv = opt('--search')[0]
        cands = wet.replay_candidates(s['real'], drv, s['race_laps'], cfg)
        tab = wet.race_candidates(inputs, circuit, s['race_laps'], s['bases'], s['real'], drv, s['drivers'], cands,
                                  s['engine'], cfg, status=s['status'], free=s['free'], seed=seed)
        print(f'\n{drv}: {len(cands)} switch lap combinations raced')
        show(tab, 12)
        print(tab[tab['Strategy'].str.startswith('real')][['Strategy', 'MeanPlace', 'DeltaPlace', 'DeltaTime_s', 'Tier']].round(3).to_string(index=False))

elif '--backtest' in args:
    races = wet.wet_races(laps, [2022, 2023, 2024, 2025])
    print(f'WET REPLAY BACKTEST: {len(races)} wet races 2022 to 2025, oracle pace (fitted on the race with its track), '
          f'real strategies, real neutral laps, chosen passing rule, {draws} draws')
    rows, tracks = [], []
    for race in races.itertuples():
        try:
            for name, gt in [('real track path', True), ('no track path (wet tyre = fresh Medium)', False)]:
                s = wet.replay_setup(inputs, t, race.Year, race.Round, race.Circuit, cfg, draws, group_terms=gt)
                res = wet.replay(inputs, s, race.Circuit, cfg, seed=seed)
                rows.append(wet.replay_rows(res, s, race.Year, race.Round, race.Circuit, name, laps))
                if gt:
                    tb = s['track']['table']
                    real_sw = sum(len(wet.switches_of(s['real'], d, cfg)) for d in s['drivers'])
                    tracks.append({'Year': race.Year, 'Round': race.Round, 'Circuit': race.Circuit, 'Cars': len(s['drivers']),
                                   'Switches': real_sw, 'Measured': int((tb['Source'] == 'measured').sum()),
                                   'Interp': int((tb['Source'] == 'interpolated').sum()),
                                   'MinGap': tb.loc[tb['Source'] == 'measured', 'Gap'].min(),
                                   'MaxGap': tb.loc[tb['Source'] == 'measured', 'Gap'].max(),
                                   'WetLaps': int((tb['WetShare'] >= 0.5).sum()), 'Laps': s['race_laps']})
                    keep = s
            s = keep
            res = wet.replay(inputs, s, race.Circuit, cfg, seed=seed, rule=ot.NoPassing())
            rows.append(wet.replay_rows(res, s, race.Year, race.Round, race.Circuit, 'real track path, no passing', laps))
        except (KeyError, ValueError, IndexError) as e:
            print(f'skipped {race.Year} round {race.Round} {race.Circuit}: {type(e).__name__}: {e}')
    rows = pd.concat(rows, ignore_index=True)
    g = rows.drop_duplicates(bt.RACE + ['Driver']).assign(Variant='grid order', MeanPlace=lambda d: d['Grid'].astype(float))
    allr = pd.concat([rows, g], ignore_index=True)
    print('\ntrack paths (gap = wet tyre minus slick, s, on measured laps):')
    print(pd.DataFrame(tracks).round(2).to_string(index=False))
    print('\nplaces among classified finishers (outcome.summary), gaps and winner time:')
    m = bt.metrics(allr)
    print(m.round(3).to_string())
    print('\nper race place MAE:')
    print(bt.race_errors(allr).round(2).to_string())
    print('\npaired against the grid order (change in place MAE, 95% race bootstrap):')
    print(bt.paired(allr, 'grid order', [v for v in rows['Variant'].unique()]).round(3).to_string())
    print('\npaired, real track path against no track path:')
    print(bt.paired(allr, 'no track path (wet tyre = fresh Medium)', ['real track path']).round(3).to_string())
else:
    print(__doc__)
