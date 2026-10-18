"""Choose and evaluate the degradation model (notes/evaluation.md) and write the final fit.

    python scripts/train_degradation.py [--validation]

Linear candidates are compared on the validation folds only; the test season is scored for the chosen model,
constant pace and the previous final model, then for the averages in degradation.ensembles and their curved
and Bayesian members (chosen in scripts/compare_degradation.py, modelling D11). Writes to data/models:
degradation.parquet and degradation_offsets.parquet (the chosen linear model, fit on all degradation seasons),
degradation_<member>_<part>.parquet for each member of the simulator average, and degradation_test_preds.parquet.
--validation prints the linear validation table and stops, test unscored.
"""
import sys

import pandas as pd

from boxbox import degradation, evaluate
from boxbox.config import load_config, seasons_for
from boxbox.degradation import compare

REFERENCE = 'linear, shrink 10, stint length, dirty air, race start'   # previous final model (modelling D8)
SIMULATOR = 'simulator average'   # the wear model the strategy simulator uses (modelling D11)

cfg = load_config()
seasons = seasons_for(cfg, 'degradation')
laps = pd.read_parquet(cfg['paths']['processed'] / 'laps.parquet')
split = evaluate.season_splits(seasons)
pd.set_option('display.width', 200)
if '--validation' in sys.argv:
    val, best = degradation.validate(laps, cfg, seasons)
    print('validation folds:', ', '.join(f'train {tr} score {y}' for tr, y in split['validation']))
    print(val.sort_values('MAE_s').round(4).to_string(), '\n')
    print(f'chosen: {best}')
    sys.exit()
val, best, preds, test, final = degradation.run(laps, cfg, seasons, REFERENCE)

print('validation folds:', ', '.join(f'train {tr} score {y}' for tr, y in split['validation']))
print(val.sort_values('MAE_s').round(3).to_string(), '\n')
print(f'chosen: {best}\n')
print(f"test: train {split['test'][0]}, score {split['test'][1]}")
print(test.round(3).to_string(), '\n')

# Pre race model on the test season: chosen and previous final, each with the same weekend shifts
P = cfg['paths']['processed']
sprint, prac = pd.read_parquet(P / 'sprint_laps.parquet'), pd.read_parquet(P / 'practice_laps.parquet')
train_years, year = split['test']
for name in [best, REFERENCE]:
    m = next(c for c in degradation.candidates(cfg) if c.name == name).fit(degradation.model_laps(laps, train_years, cfg, train=True))
    preds[f'{name}, weekend'] = degradation.weekend_forecast(m, laps, sprint, prac, cfg, train_years, year)
weekend = pd.concat({n: evaluate.error_table(p) for n, p in preds.items() if n.endswith(', weekend')}).droplevel(1)
print('test, pre race model (sprint, else practice at circuits with no dry history)')
print(weekend.round(3).to_string(), '\n')

# The averages and their members, scored once on the test season (modelling D11)
ladder = compare.ladder(cfg)
ens = [n for n, e in (cfg['degradation'].get('ensembles') or {}).items() if e['members']]
members = list(dict.fromkeys(m for n in ens for m in cfg['degradation']['ensembles'][n]['members']))
train_laps, test_laps = degradation.model_laps(laps, train_years, cfg, train=True), degradation.model_laps(laps, [year], cfg)
for name in members + ens:
    preds[name] = degradation.forecast(ladder[name]().fit(train_laps), test_laps, cfg)
avg = pd.concat({n: evaluate.error_table(preds[n]) for n in members + ens}).droplevel(1)
print('test, averages and their members'); print(avg.round(3).to_string(), '\n')

for by in ['LapsAhead', 'Compound']:
    t = pd.concat({n: evaluate.error_table(p, by) for n, p in preds.items()}, axis=1)
    print(f'test by {by}'); print(t.round(3).to_string(), '\n')
circ = pd.concat({n: evaluate.error_table(p, 'Circuit')['MAE_s'] for n, p in preds.items()}, axis=1)
print('test MAE by circuit'); print(circ.sort_values(best).round(3).to_string(), '\n')

print(f'final fit on {seasons}, seconds lost from tyre age 1 to age N, compound slopes:')
print(pd.DataFrame({n: final.loss_after(n) for n in [10, 20, 30]}).round(2).to_string(), '\n')
print(final.cells[['Stints', 'Raw', 'Slope', 'ResidSD_s']].round(3).to_string(), '\n')
print('lap level offsets, s (dirty air: gap band start; race start: race lap)')
print(final.export()['offsets'].round(3).to_string(index=False))

out = cfg['paths']['models']
out.mkdir(exist_ok=True)
parts = final.export()
parts['cells'].to_parquet(out / 'degradation.parquet', index=False)
parts['offsets'].to_parquet(out / 'degradation_offsets.parquet', index=False)

# Members of the simulator average, refit on all degradation seasons
for name in (cfg['degradation'].get('ensembles') or {}).get(SIMULATOR, {}).get('members', []):
    m = ladder[name]().fit(degradation.model_laps(laps, seasons, cfg, train=True))
    for part, table in m.export().items():
        table.to_parquet(out / f"degradation_{name.replace(', ', '_').replace(' ', '_')}_{part}.parquet", index=False)
    if 'curve' in m.export():
        print(f'\n{name}: fitted bend per compound (s, straight line part removed)')
        print(m.export()['curve'].set_index('TyreLife').loc[[1, 5, 10, 20, 30, 40]].round(3).to_string())
    if hasattr(m, 'summary'):
        print(f'\n{name}: posterior summary'); print(m.summary().round(4).to_string())
pd.concat(preds, names=['Model']).reset_index(level=0).astype({'LapsAhead': str}).to_parquet(out / 'degradation_test_preds.parquet', index=False)
