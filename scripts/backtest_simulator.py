"""Backtest the strategy simulator v1 on historical green flag dry races (simulator/backtest.py).

    python scripts/backtest_simulator.py [--validation] [--search]

Each scored season is simulated with inputs (lap model, passing rule, pace model) and a rival pit policy fitted
on the seasons before it. Variants: pace (oracle: each car's real race base; model: drawn from the pace model)
x strategies (real; model: drawn from the rival pit model) x passing rule (the chosen circuit threshold, one
global threshold, free passing, no passing). Places are scored as the finishing position predictor is (place
among the classified finishers), against the grid order and the outcome ridge model trained on the same seasons.
--search also runs the strategy search for the top 3 of every race, with and without stint caps, and reports
where the strategy they really ran ranks (about 20 minutes for the two validation seasons).
--validation scores 2023 and 2024 only.
"""
import sys

import joblib
import numpy as np
import pandas as pd

from boxbox import evaluate, features, outcome, overtaking as ot, pitdecision as pdm
from boxbox.config import load_config, seasons_for
from boxbox.simulator import backtest as bt, pace as pc, race as rc, strategy as st
from boxbox.simulator.inputs import SimInputs

cfg = load_config()
s, r = cfg['simulator'], cfg['simulator']['race']
seasons = seasons_for(cfg, 'overtaking')
P = cfg['paths']['processed']
t = {n: pd.read_parquet(P / f'{n}.parquet') for n in ['laps', 'results', 'qualifying', 'sprint_results', 'events', 'pit_loss_stops', 'stints']}
laps = t['laps']
pre = pdm.pre_race(t['results'], t['qualifying'], t['sprint_results'], t['events'], laps, cfg)
stops = pdm.stops_with_circuit(t['pit_loss_stops'], t['events'])
split = evaluate.season_splits(seasons)
folds = split['validation'] + ([] if '--validation' in sys.argv else [split['test']])
rules, paces = ot.candidates(cfg), pc.candidates(cfg)
chosen_rule, chosen_pace = s['overtaking']['chosen'], s['pace']['chosen']
glob_name = 'global threshold, ' + chosen_rule.split(', ')[-1]
entries = outcome.entries(t['results'], t['qualifying'], t['sprint_results'], t['events'], cfg, laps)
alpha = joblib.load(cfg['paths']['models'] / 'outcome_ridge.joblib').alpha
pd.set_option('display.width', 250)

all_rows, all_passes, spots = [], [], []
for train, year in folds:
    inputs = SimInputs(cfg).fit(t, pre, train, rules[chosen_rule], paces[chosen_pace])
    policy = pdm.fit_plan_policy(laps, pre, t['results'], stops, train, cfg)
    glob = rules[glob_name]().fit(inputs.train_pairs)
    variants = [('oracle', 'real', 'circuit threshold', inputs.rule), ('oracle', 'real', 'global threshold', glob),
                ('oracle', 'real', 'free passing', ot.FreePassing()), ('oracle', 'real', 'no passing', ot.NoPassing()),
                ('model', 'real', 'circuit threshold', inputs.rule), ('oracle', 'model', 'circuit threshold', inputs.rule),
                ('model', 'model', 'circuit threshold', inputs.rule)]
    bases = inputs.lap.driver_bases(inputs.lap.stint_levels(laps, [year]))
    real = bt.real_passes(laps, inputs.lap, year, inputs.window_s)
    races = bt.green_races(laps, [year])
    for race in races.itertuples():
        ob = bases[(bases['Year'] == year) & (bases['Round'] == race.Round)].set_index('Driver')['Base']
        rows, passes = bt.run_race(inputs, t, pre, year, race.Round, race.Circuit, ob, policy, variants, r['draws'], r['seed'], cfg)
        all_rows.append(rows)
        all_passes.append({'Year': year, 'Round': race.Round, 'Circuit': race.Circuit, **passes,
                           'real passes': real['Passes'].get((year, race.Round), np.nan),
                           'real pairs': real['Pairs'].get((year, race.Round), np.nan)})
    # outcome predictor trained on the outcome seasons before the scored one, same races and drivers
    ent = entries[entries['Year'].isin([y for y in seasons_for(cfg, 'outcome') if y < year])]
    m = outcome.RidgeModel(alpha).fit(ent)
    yr = pd.concat(all_rows)
    yr = yr[(yr['Year'] == year)].drop_duplicates(['Year', 'Round', 'Driver'])
    te = entries.merge(yr[['Year', 'Round', 'Driver']], on=['Year', 'Round', 'Driver'])
    all_rows.append(outcome.ranked(te, m.score(te)).assign(Variant='outcome predictor (ridge)', MeanPlace=lambda d: d['Pred'],
                                                            NewCircuit=True)[['Year', 'Round', 'Driver', 'Variant', 'MeanPlace', 'Finish', 'Grid', 'NewCircuit']])
    spots.append((inputs, policy, year))

rows = pd.concat(all_rows, ignore_index=True)
sim = rows[rows['Variant'] != 'outcome predictor (ridge)']
grid = sim.drop_duplicates(['Year', 'Round', 'Driver']).assign(Variant='grid order', MeanPlace=lambda d: d['Grid'].astype(float))
out = []
for y, g in pd.concat([sim, grid, rows[rows['Variant'] == 'outcome predictor (ridge)']]).groupby('Year'):
    m = bt.metrics(g.assign(Time_s=g.get('Time_s'), GapToWinner_s=g.get('GapToWinner_s'), RealGap_s=g.get('RealGap_s'),
                            RaceTime_s=g.get('RaceTime_s')))
    out.append(m.assign(Year=y))
tab = pd.concat(out).reset_index(names='Variant').set_index(['Year', 'Variant'])
cols = ['MAE', 'Spearman', 'Podium', 'Winner', 'Gap MAE, s', 'Winner time error, s', 'Winner time MAE, s', 'Races', 'N']
print('backtest by scored season (places among classified finishers; gaps for cars on the lead lap)')
print(tab[cols].round(3).to_string(), '\n')

pas = pd.DataFrame(all_passes)
pcols = [c for c in pas.columns if c.startswith('oracle pace, real strategies') or c.startswith('real')]
print('on track passes and close pairs per race, oracle pace and real strategies (mean over races)')
print(pas.groupby('Year')[pcols].mean().T.round(1).to_string(), '\n')

if '--search' not in sys.argv:
    sys.exit()

# Strategy search check: for the top 3 of every green race, where the strategy they really ran ranks among the
# searched ones (model pace, rivals from the pit model), with and without the stint caps
print('stint caps: share of the scored seasons\' real stints within them')
for inputs, policy, year in spots:
    print(f'  {year}: {bt.cap_coverage(t["stints"], t["events"], inputs.caps, features.wet_races(laps), [year]):.3f}')
checks = []
for inputs, policy, year in spots:
    for race in bt.green_races(laps, [year]).itertuples():
        truth = bt.race_truth(laps, t['results'], year, race.Round)
        for driver in truth.sort_values('Finish')['Driver'].head(3):
            for caps in [True, False]:
                try:
                    best, actual, grid = bt.strategy_search(inputs, policy, t, pre, year, race.Round, race.Circuit, driver, cfg,
                                                            r['coarse_draws'] * 4, use_caps=caps, seed=r['seed'])
                except (IndexError, ValueError):
                    continue
                me = best[best['Extra']].iloc[0]
                real_tier = me['Tier']
                top = best.iloc[0]
                checks.append({'Year': year, 'Round': race.Round, 'Circuit': race.Circuit, 'Driver': driver, 'Caps': caps,
                               'Real': '-'.join(c[0] for c in actual[0]) + ' ' + ','.join(map(str, actual[1])),
                               'Best': top['Compounds'] + ' ' + top['StopLaps'], 'RealRank': int(best.index[best['Extra']][0]) + 1,
                               'Raced': len(best), 'RealStops': len(actual[1]), 'BestStops': int(top['Stops']),
                               'PlaceLoss': me['MeanPlace'] - top['MeanPlace'], 'RealNearTie': real_tier != 'worse',
                               'TieGroup': int((best['Tier'] != 'worse').sum()),
                               'BestSoftLaps': top['Compounds'].split('-').count('S')})
c = pd.DataFrame(checks)
summary = c.groupby(['Year', 'Caps']).agg(Searches=('Driver', 'size'), RealRankMedian=('RealRank', 'median'),
                                          RealInTop3=('RealRank', lambda x: (x <= 3).mean()),
                                          SameStops=('RealStops', lambda x: (x == c.loc[x.index, 'BestStops']).mean()),
                                          PlaceLoss=('PlaceLoss', 'mean'), Raced=('Raced', 'median'),
                                          RealNearTie=('RealNearTie', 'mean'), TieGroup=('TieGroup', 'median'))
print('strategy search on real top 3 finishers (RealRank: rank of their real strategy; PlaceLoss: its mean place minus the best\'s; RealNearTie: it is in the near tie group)')
print(summary.round(3).to_string(), '\n')
print('best stop count vs real, with caps'); print(pd.crosstab(c.loc[c['Caps'], 'RealStops'], c.loc[c['Caps'], 'BestStops']).to_string(), '\n')
m = c[(c['Circuit'] == 'Monza')]
print('Monza'); print(m[['Year', 'Driver', 'Caps', 'Real', 'Best', 'RealRank', 'Raced', 'PlaceLoss']].round(3).to_string(index=False))
