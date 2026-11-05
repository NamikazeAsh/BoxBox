"""Backtest the strategy simulator v1 on historical green flag dry races (simulator/backtest.py).

    python scripts/backtest_simulator.py [--validation]

Each scored season is simulated with inputs (lap model, passing rule, pace model) and a rival pit policy fitted
on the seasons before it. Variants: pace (oracle: each car's real race base; model: drawn from the pace model)
x strategies (real; model: drawn from the rival pit model) x passing rule (the chosen circuit threshold, one
global threshold, free passing, no passing). Places are scored as the finishing position predictor is (place
among the classified finishers), against the grid order and the outcome ridge model trained on the same seasons.
Then a spot check: the strategy search for one race's winner, with the strategy they really ran added.
--validation scores 2023 and 2024 only.
"""
import sys

import joblib
import numpy as np
import pandas as pd

from boxbox import evaluate, outcome, overtaking as ot, pitdecision as pdm
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

all_rows, all_passes, spot = [], [], None
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
    if spot is None and year == 2024:
        spot = (inputs, policy, year)

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

# Spot check: the strategy search for the winner of Monza 2024 (or the season's first green race)
inputs, policy, year = spot
races = bt.green_races(laps, [year])
race = races[races['Circuit'] == 'Monza'].iloc[0] if (races['Circuit'] == 'Monza').any() else races.iloc[0]
truth = bt.race_truth(laps, t['results'], year, race['Round'])
real = st.real_stints(t['stints'], year, race['Round'], truth['Driver'])
truth = truth[truth['Driver'].isin(real['Driver'])].reset_index(drop=True)
drivers = truth['Driver'].tolist()
winner = truth.loc[truth['Finish'] == 1, 'Driver'].iloc[0]
race_laps = int(laps.loc[(laps['Year'] == year) & (laps['Round'] == race['Round']), 'LapNumber'].max())
cars = truth[['Driver']].assign(Year=year, Round=race['Round']).merge(pre, on=['Year', 'Round', 'Driver'], how='left')
rng = np.random.default_rng(r['seed'])
bases = inputs.sample_bases(cars, race['Circuit'], r['refine_draws'], rng)
starts = pdm.race_starts(laps, pre, t['results'], [year], cfg, green_only=False)
starts = starts[(starts['Round'] == race['Round']) & starts['Driver'].isin(drivers)].reset_index(drop=True)
rivals = st.model_stints(policy, starts, r['refine_draws'], r['seed'])
missing = set(drivers) - set(starts['Driver'])
rivals = st.override(rivals, real[real['Driver'].isin(missing)]) if missing else rivals
mine = real[real['Driver'] == winner].sort_values('Stint')
actual = (tuple(mine['Compound']), tuple(int(x) - 1 for x in mine['StartLap'].iloc[1:]))
w = mine.iloc[0]
best = rc.best_strategies(inputs, race['Circuit'], race_laps, drivers, bases, rivals, winner, cfg,
                          start_age=float(w['StartAge']), start_compounds=[w['Compound']], extra=[actual])
print(f"strategy search, {winner} at {race['Circuit']} {year} (grid {int(truth.loc[truth['Driver'] == winner, 'Grid'].iloc[0])}, "
      f"model pace, rivals from the pit model, {r['refine_draws']} draws); real: {'-'.join(c[0] for c in actual[0])} stopping {actual[1]}")
print(best.head(12).round(3).to_string())
print('real strategy:', best.index[best['Extra']].tolist(), 'of', len(best), 'raced')
