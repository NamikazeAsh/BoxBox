"""Backtest the strategy simulator on historical dry races (simulator/backtest.py).

    python scripts/backtest_simulator.py [--validation] [--search] [--green]

Each scored season is simulated with inputs (lap model, passing rule, pace model, safety car model) and a rival pit
policy fitted on the seasons before it. Races: every dry race (v2; safety cars and VSCs raced), or the green flag
races only with --green (v1's set). Variants: pace (oracle: each car's real race base; model: drawn from the pace
model) x strategies (real; model: drawn from the rival pit model) x passing rule x neutral laps (real: the race's
own safety cars and VSCs, for replays; sample: drawn from the safety car model, as before the race; green: none,
as v1). The fully predicted simulator is model pace, model strategies, sampled neutral laps. The v1 system (v1 pace
model, last season reference, no safety cars) is raced on the same races for the before and after. Places are
scored as the finishing position predictor is (place among the classified finishers), against the grid order and
the outcome ridge model trained on the same seasons, and a blend of the simulator's mean place with the
predictor's place (weights simulator.backtest.blend_weights).
--search also runs the strategy search for the top 3 of every green race, with and without stint caps, and reports
where the strategy they really ran ranks (slow).
--validation scores 2023 and 2024 only; --test-only scores the test season 2025 only (once, after every choice).
--shift scores 2026 (holdout tables appended, train 2022 to 2025; the shift study's measure 5, run once) and prints
the simulator and safety car configuration first. --check sets up and races every race with a few draws and stops
before scoring (run it first with --shift: a race that fails would otherwise be skipped silently).
"""
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from boxbox import evaluate, features, outcome, overtaking as ot, pitdecision as pdm
from boxbox.config import load_config, seasons_for
from boxbox.simulator import backtest as bt, pace as pc, strategy as st
from boxbox.simulator.inputs import SimInputs

cfg = load_config()
s, r = cfg['simulator'], cfg['simulator']['race']
b = s.get('backtest', {})
seasons = seasons_for(cfg, 'overtaking')
shift = '--shift' in sys.argv   # the 2026 shift study, measure 5 (notes/shift_study.md): holdout tables, train 2022 to 2025
t = bt.load_tables(cfg, holdout=shift)
laps = t['laps']
pre = pc.pre_race(t['results'], t['qualifying'], t['sprint_results'], t['events'], laps, cfg)
stops = pdm.stops_with_circuit(t['pit_loss_stops'], t['events'])
split = evaluate.season_splits(seasons)
folds = [split['test']] if '--test-only' in sys.argv else split['validation'] + ([] if '--validation' in sys.argv else [split['test']])
if shift:
    folds = bt.shift_folds(cfg)
    print('SHIFT STUDY (measure 5): ' + ', '.join(f'train {tr} score {y}' for tr, y in folds))
    print('configuration at the time of the run:')
    print(bt.config_record(cfg))
green_only = '--green' in sys.argv
rules, paces = ot.candidates(cfg), pc.candidates(cfg)
chosen_rule, chosen_pace = s['overtaking']['chosen'], s['pace']['chosen']
v1_pace = b.get('v1_pace', 'ridge, alpha 100')
entries = outcome.entries(t['results'], t['qualifying'], t['sprint_results'], t['events'], cfg, laps)
alpha = joblib.load(cfg['paths']['models'] / 'outcome_ridge.joblib').alpha
weights = b.get('blend_weights', [0.25, 0.5, 0.75])
FULL = 'model pace, model strategies, {rule}, sample neutral laps'
pd.set_option('display.width', 250)
warnings.filterwarnings('ignore', category=RuntimeWarning)   # empty means for variants with no race times
warnings.filterwarnings('ignore', category=FutureWarning)

all_rows, all_passes, spots = [], [], []
for train, year in folds:
    inputs = SimInputs(cfg).fit(t, pre, train, rules[chosen_rule], paces[chosen_pace])
    policy = pdm.fit_plan_policy(laps, pre, t['results'], stops, train, cfg)
    neutral = inputs.neutral is not None and not green_only
    nl = ['real', 'sample'] if neutral else ['green']
    variants = [('oracle', 'real', 'chosen rule', inputs.rule, nl[0]), ('oracle', 'real', 'no passing', ot.NoPassing(), nl[0]),
                ('model', 'real', 'chosen rule', inputs.rule, nl[0]), ('oracle', 'model', 'chosen rule', inputs.rule, nl[0]),
                ('model', 'model', 'chosen rule', inputs.rule, nl[-1])]
    if neutral:
        variants.append(('model', 'model', 'chosen rule', inputs.rule, 'green'))   # the same system with no safety cars
    # the v1 system: v1 pace model and reference, the v1 engine, no safety cars
    v1 = SimInputs.__new__(SimInputs)
    v1.__dict__.update(inputs.__dict__)
    targets = pc.pace_targets(inputs.lap.driver_bases(inputs.lap.stint_levels(laps, train)), pre, t['events'])
    v1.pace, v1.reference = paces[v1_pace]().fit(targets), 'last_season'
    v1.engine = {'reference': 'adjacent', 'held_gap': 'fixed', 'pace_gap': 'drawn', 'close_air': 'flat', 'lap1': 'grid'}
    bases = inputs.lap.driver_bases(inputs.lap.stint_levels(laps, [year]))
    real = bt.real_passes(laps, inputs.lap, year, inputs.window_s)
    races = bt.green_races(laps, [year]) if green_only else bt.dry_races(laps, [year])
    if '--check' in sys.argv:   # set up and race every race with a few draws before any scoring (new circuits)
        print(f'setup check, {year}:')
        print(bt.setup_check(inputs, policy, t, pre, year, races, bases, variants, cfg).to_string(index=False), chr(10))
        continue
    green = set(map(tuple, bt.green_races(laps, [year])[['Year', 'Round']].to_numpy()))
    for race in races.itertuples():
        ob = bases[(bases['Year'] == year) & (bases['Round'] == race.Round)].set_index('Driver')['Base']
        try:
            rows, passes = bt.run_race(inputs, t, pre, year, race.Round, race.Circuit, ob, policy, variants, r['draws'], r['seed'], cfg)
            rows1, _ = bt.run_race(v1, t, pre, year, race.Round, race.Circuit, ob, policy,
                                   [('model', 'model', 'v1 system', v1.rule, 'green')], r['draws'], r['seed'], cfg)
        except (KeyError, ValueError) as e:   # a race the simulator cannot set up (no strategy data for a car, ...)
            print(f'skipped {year} round {race.Round} {race.Circuit}: {e}')
            continue
        rows = pd.concat([rows, rows1.assign(Variant='v1 system (v1 pace, last season reference, no safety cars)')])
        all_rows.append(rows.assign(Green=(year, race.Round) in green))
        all_passes.append({'Year': year, 'Round': race.Round, 'Circuit': race.Circuit, 'Green': (year, race.Round) in green,
                           **{k: v for k, v in passes.items() if not k.endswith('tenth')},
                           'real passes': real['Passes'].get((year, race.Round), np.nan)})
    # outcome predictor trained on the outcome seasons before the scored one, same races and drivers
    ent = entries[entries['Year'].isin([y for y in seasons_for(cfg, 'outcome') if y < year])]
    m = outcome.RidgeModel(alpha).fit(ent)
    yr = pd.concat(all_rows)
    yr = yr[(yr['Year'] == year)].drop_duplicates(['Year', 'Round', 'Driver'])
    te = entries.merge(yr[['Year', 'Round', 'Driver', 'Green']], on=['Year', 'Round', 'Driver'])
    all_rows.append(outcome.ranked(te, m.score(te)).assign(Variant='outcome predictor (ridge)', MeanPlace=lambda d: d['Pred'],
                                                            NewCircuit=True)[['Year', 'Round', 'Driver', 'Variant', 'MeanPlace', 'Finish', 'Grid', 'NewCircuit', 'Green']])
    spots.append((inputs, policy, year))

if '--check' in sys.argv:
    sys.exit()
rows = pd.concat(all_rows, ignore_index=True)
rule_name = 'chosen rule'
full = FULL.format(rule=rule_name) if not green_only and any('sample neutral laps' in v for v in rows['Variant'].unique()) \
    else f'model pace, model strategies, {rule_name}'
pred = rows[rows['Variant'] == 'outcome predictor (ridge)'].set_index(['Year', 'Round', 'Driver'])['MeanPlace']
blends = []
for w in weights:
    f = rows[rows['Variant'] == full].copy()
    f['MeanPlace'] = w * f['MeanPlace'] + (1 - w) * f.set_index(['Year', 'Round', 'Driver']).index.map(pred).to_numpy(float)
    blends.append(f.dropna(subset=['MeanPlace']).assign(Variant=f'blend: {w} x simulator + {1 - w} x predictor'))
sim = rows[rows['Variant'] != 'outcome predictor (ridge)']
grid = sim.drop_duplicates(['Year', 'Round', 'Driver']).assign(Variant='grid order', MeanPlace=lambda d: d['Grid'].astype(float))
everything = pd.concat([sim, grid, rows[rows['Variant'] == 'outcome predictor (ridge)']] + blends, ignore_index=True)
cols = ['MAE', 'Spearman', 'Podium', 'Winner', 'Gap MAE, s', 'Winner time error, s', 'Winner time MAE, s', 'Races', 'N']
for label, part in [('every dry race' if not green_only else 'green flag races', everything), ('green flag races only', everything[everything['Green']]),
                    ('races with a safety car or VSC', everything[~everything['Green']])]:
    if part.empty or (green_only and label != 'green flag races'):
        continue
    out = []
    for y, g in part.groupby('Year'):
        out.append(bt.metrics(g.assign(Time_s=g.get('Time_s'), GapToWinner_s=g.get('GapToWinner_s'), RealGap_s=g.get('RealGap_s'),
                                       RaceTime_s=g.get('RaceTime_s'))).assign(Year=y))
    out.append(bt.metrics(part).assign(Year='pooled'))
    tab = pd.concat(out).reset_index(names='Variant').set_index(['Year', 'Variant'])
    print(f'backtest, {label} (places among classified finishers; gaps for cars on the lead lap)')
    print(tab[cols].round(3).to_string(), '\n')

v1_name = 'v1 system (v1 pace, last season reference, no safety cars)'
others = [v for v in everything['Variant'].unique() if v.startswith(('model pace, model strategies', 'blend', 'outcome', 'grid'))]
for label, part in [('every race', everything), ('green flag races', everything[everything['Green']])]:
    if green_only and label != 'green flag races':
        continue
    if part.empty:
        print(f'paired against the v1 system, {label}: no races\n')
        continue
    print(f'paired against the v1 system, {label} (change in place MAE, 95% bootstrap interval over races)')
    print(bt.paired(part, v1_name, others).round(3).to_string(), chr(10))
print('paired against the outcome predictor, every race')
print(bt.paired(everything, 'outcome predictor (ridge)', [v for v in others if not v.startswith('outcome')] + [v1_name]).round(3).to_string(), chr(10))

pas = pd.DataFrame(all_passes)
pcols = [c for c in pas.columns if c.startswith('oracle pace, real strategies, chosen rule') or c.startswith('real')]
print('on track passes and close pairs per race, oracle pace and real strategies (mean over races)')
print(pas.groupby(['Year', 'Green'])[pcols].mean().T.round(1).to_string(), '\n')

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
                top = best.iloc[0]
                checks.append({'Year': year, 'Round': race.Round, 'Circuit': race.Circuit, 'Driver': driver, 'Caps': caps,
                               'Real': '-'.join(c[0] for c in actual[0]) + ' ' + ','.join(map(str, actual[1])),
                               'Best': top['Compounds'] + ' ' + top['StopLaps'], 'RealRank': int(best.index[best['Extra']][0]) + 1,
                               'Raced': len(best), 'RealStops': len(actual[1]), 'BestStops': int(top['Stops']),
                               'PlaceLoss': me['MeanPlace'] - top['MeanPlace'], 'RealNearTie': me['Tier'] != 'worse',
                               'TieGroup': int((best['Tier'] != 'worse').sum())})
c = pd.DataFrame(checks)
summary = c.groupby(['Year', 'Caps']).agg(Searches=('Driver', 'size'), RealRankMedian=('RealRank', 'median'),
                                          RealInTop3=('RealRank', lambda x: (x <= 3).mean()),
                                          SameStops=('RealStops', lambda x: (x == c.loc[x.index, 'BestStops']).mean()),
                                          PlaceLoss=('PlaceLoss', 'mean'), Raced=('Raced', 'median'),
                                          RealNearTie=('RealNearTie', 'mean'), TieGroup=('TieGroup', 'median'))
print('strategy search on real top 3 finishers (RealRank: rank of their real strategy; PlaceLoss: its mean place minus the best\'s; RealNearTie: it is in the near tie group)')
print(summary.round(3).to_string(), '\n')
print('best stop count vs real, with caps'); print(pd.crosstab(c.loc[c['Caps'], 'RealStops'], c.loc[c['Caps'], 'BestStops']).to_string(), '\n')
