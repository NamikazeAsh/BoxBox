"""Choose and evaluate the simulator's passing rule and car pace model, then fit every simulator input on all
seasons (simulator/inputs.py).

    python scripts/train_simulator.py [--validation]

Both are chosen on the rolling validation folds (each scored season with a lap model fitted on the seasons
before it): the passing rule by the error in passes per race over close pairs (overtaking.pass_pairs), the pace
model by MAE in % pace. The test season is scored only for the chosen ones and the baselines. Writes
data/models/simulator_inputs.joblib (SimInputs). --validation stops before the test.
"""
import sys

import joblib
import pandas as pd

from boxbox import evaluate, overtaking as ot, pitdecision as pdm
from boxbox.config import load_config, seasons_for
from boxbox.simulator import pace as pc
from boxbox.simulator.inputs import SimInputs, lap1_terms
from boxbox.simulator.laptime import LapModel

cfg = load_config()
o = cfg['simulator']['overtaking']
seasons = seasons_for(cfg, 'overtaking')
P = cfg['paths']['processed']
t = {n: pd.read_parquet(P / f'{n}.parquet') for n in ['laps', 'results', 'qualifying', 'sprint_results', 'events', 'pit_loss_stops', 'stints']}
laps = t['laps']
pre = pdm.pre_race(t['results'], t['qualifying'], t['sprint_results'], t['events'], laps, cfg)
split = evaluate.season_splits(seasons)
validation_only = '--validation' in sys.argv
pd.set_option('display.width', 250)
rules, paces = ot.candidates(cfg), pc.candidates(cfg)
print('validation folds:', ', '.join(f'train {tr[0]} to {tr[-1]} score {y}' for tr, y in split['validation']), '\n')


def fold(train, year):
    lm = LapModel(cfg).fit(laps, train, laps['Circuit'].unique())
    pairs = {}
    for name, yrs in [('train', train), ('test', [year])]:
        free = ot.free_air_laps(laps, lm, yrs)
        pairs[name] = ot.pass_pairs(laps.loc[free.index], free, o['window_s'])
    bases = {n: lm.driver_bases(lm.stint_levels(laps, yrs)) for n, yrs in [('train', train), ('test', [year])]}
    targets = {n: pc.pace_targets(b, pre, t['events']) for n, b in bases.items()}
    return lm, pairs, targets, bases


def score(folds, rule_names, pace_names):
    rp, pp, extra = {}, {}, []
    for year, (lm, pairs, targets, bases) in folds.items():
        for n in rule_names:
            m = rules[n]().fit(pairs['train'])
            rp.setdefault(n, []).append(pairs['test'].assign(Pred=m.predict(pairs['test'])))
        for n in pace_names:
            m = paces[n]().fit(targets['train'])
            te = targets['test'].dropna(subset=['Pace'])
            pp.setdefault(n, []).append(te.assign(Pred=m.predict(te)))
        gaps, ex, ex_all = lap1_terms(laps, lm, bases['train'], sorted(set(pairs['train']['Year'])))
        extra.append({'Season': year, 'Follow gap, s (train)': ot.follow_gap(pairs['train']),
                      'Follow gap, s (scored)': ot.follow_gap(pairs['test']), 'Pass rate (train)': pairs['train']['Pass'].mean(),
                      'Pass rate (scored)': pairs['test']['Pass'].mean(), 'Lap 1 extra, s': ex_all,
                      'Lap 1 gap to P2, s': gaps[1], 'Lap 1 gap to P10, s': gaps[9]})
    rt = pd.DataFrame({n: ot.pass_metrics(p := pd.concat(v, ignore_index=True), p['Pred']) for n, v in rp.items()}).T
    pt = pd.DataFrame({n: pc.pace_metrics(p := pd.concat(v, ignore_index=True), p['Pred'].to_numpy()) for n, v in pp.items()}).T
    return rt, pt, pd.DataFrame(extra).set_index('Season')


val = {year: fold(train, year) for train, year in split['validation']}
rt, pt, extra = score(val, list(rules), list(paces))
print('passing rule, validation (close pairs, B within window_s at the line)')
print(rt.sort_values('Race MAE, passes').round(3).to_string(), '\n')
best_rule = rt.drop([ot.FreePassing.name, ot.NoPassing.name]).sort_values('Race MAE, passes').index[0]
print(f'chosen rule: {best_rule}\n')
print('car pace, validation (% behind the race median base)'); print(pt.sort_values('MAE, %').round(4).to_string(), '\n')
best_pace = pt.drop(pc.ConstantPace.name).sort_values('MAE, %').index[0]
print(f'chosen pace: {best_pace}\n')
print('other inputs by fold'); print(extra.round(3).to_string(), '\n')
if validation_only:
    sys.exit()

train, year = split['test']
glob = next(n for n in rules if n.startswith('global threshold') and n.endswith(best_rule.split(', ')[-1]))
rt, pt, extra = score({year: fold(train, year)}, list(dict.fromkeys([ot.FreePassing.name, ot.NoPassing.name, glob, best_rule])),
                      [pc.ConstantPace.name, 'ridge, alpha 10, QualiGap', best_pace])
print(f'test: train {train[0]} to {train[-1]}, score {year}')
print('passing rule, test'); print(rt.round(3).to_string(), '\n')
print('car pace, test'); print(pt.round(4).to_string(), '\n')
print(extra.round(3).to_string(), '\n')

inputs = SimInputs(cfg).fit(t, pre, seasons, rules[best_rule], paces[best_pace]).release()
out = cfg['paths']['models']
out.mkdir(exist_ok=True)
joblib.dump(inputs, out / 'simulator_inputs.joblib')
print(f"final fit on {seasons[0]} to {seasons[-1]}: {inputs.rule.name}, {inputs.pace.name}, follow gap {inputs.follow_gap_s:.2f} s, "
      f"written to {out / 'simulator_inputs.joblib'}")
