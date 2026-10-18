"""Choose and evaluate the finishing position predictor (notes/evaluation.md) and write the final fit.

    python scripts/train_outcome.py [--validation]

Candidates are compared on the rolling validation folds (each season scored by a model trained on the seasons
before it); the test season is scored only for the chosen model and the grid baseline. Writes
data/models/outcome_test_preds.parquet and the final model (outcome_ridge.joblib, outcome_blend.joblib or
outcome_lgbm.txt). --validation prints the validation table and feature sets and stops, test unscored.
"""
import sys

import joblib
import pandas as pd

from boxbox import evaluate, features, outcome
from boxbox.config import load_config, seasons_for

cfg = load_config()
seasons = seasons_for(cfg, 'outcome')
P = cfg['paths']['processed']
t = {n: pd.read_parquet(P / f'{n}.parquet') for n in ['results', 'qualifying', 'sprint_results', 'events', 'laps']}
df = outcome.entries(t['results'], t['qualifying'], t['sprint_results'], t['events'], cfg, t['laps'])
df = df[df['Year'].isin(seasons)]
wet = features.wet_races(t['laps'])
split = evaluate.season_splits(seasons)
validation_only = '--validation' in sys.argv
if validation_only:
    val, val_preds, best = outcome.validate(df, cfg, seasons)
    final = next(m for m in outcome.candidates(cfg) if m.name == best)
else:
    val, val_preds, best, test_preds, test, final = outcome.run(df, cfg, seasons)
pd.set_option('display.width', 220)
raw = lambda p: (p['Grid'] - p['Finish']).abs().mean()

fin = df[df['Finish'].notna()]
print(f"{len(df)} starters, {len(fin)} classified finishers, {df.groupby(['Year', 'Round']).ngroups} races, {seasons[0]} to {seasons[-1]}")
print('validation folds:', ', '.join(f'train {tr[0]} to {tr[-1]} score {y}' for tr, y in split['validation']))
g = val_preds['grid']
print(f"raw grid (not ranked among finishers), validation MAE {raw(g):.3f}\n")
print(val.sort_values('MAE').round(3).to_string(), '\n')
print(f'chosen: {best}\n')

# Which features carry it: the chosen model refit with nested feature sets, validation only
sets = {'grid': ['Grid'], '+ qualifying gap': ['Grid', 'QualiGap'],
        '+ form': ['Grid', 'QualiGap', 'DriverFinishForm', 'DriverGainForm', 'TeamFinishForm', 'TeamGainForm'],
        '+ circuit grid hold': ['Grid', 'QualiGap', 'DriverFinishForm', 'DriverGainForm', 'TeamFinishForm', 'TeamGainForm', 'CircuitGridHold'],
        '+ sprint, starters (v1)': outcome.FEATURES_V1,
        '+ race pace form (all)': outcome.FEATURES}
ridge = final if isinstance(final, outcome.RidgeModel) else outcome.RidgeModel(1000)
abl = outcome.score_folds(df, split['validation'], [lambda f=f: ridge.with_features(f) for f in sets.values()])
abl = outcome.table(abl)
abl.index = list(sets)
print(f'feature sets, {ridge.name}, validation'); print(abl.round(3).to_string(), '\n')
if validation_only:
    sys.exit()

print(f"test: train {split['test'][0][0]} to {split['test'][0][-1]}, score {split['test'][1]}")
print(f"raw grid (not ranked), test MAE {raw(test_preds['grid']):.3f}")
print(test.round(3).to_string(), '\n')
for name, p in test_preds.items():
    w = p.assign(Wet=[(y, r) in wet for y, r in zip(p['Year'], p['Round'])])
    print(f"{name}: test MAE dry {w.loc[~w['Wet'], 'Err'].abs().mean():.3f} ({(~w['Wet']).sum()} finishers), "
          f"wet {w.loc[w['Wet'], 'Err'].abs().mean():.3f} ({w['Wet'].sum()})")
per = pd.concat({n: outcome.race_metrics(p)['MAE'] for n, p in test_preds.items()}, axis=1)
per['Circuit'] = df.groupby(['Year', 'Round'])['Circuit'].first()
print('\ntest MAE per race'); print(per.round(2).to_string(), '\n')

out = cfg['paths']['models']
out.mkdir(exist_ok=True)
if isinstance(final, outcome.LGBMModel):
    imp = pd.Series(final.m.booster_.feature_importance('gain'), index=final.feats)
    print(f'final fit on {seasons[0]} to {seasons[-1]}, share of split gain'); print((imp / imp.sum()).sort_values(ascending=False).round(3).to_string())
    final.m.booster_.save_model(str(out / 'outcome_lgbm.txt'))
elif isinstance(final, outcome.Blend):
    joblib.dump(final, out / 'outcome_blend.joblib')
elif isinstance(final, outcome.RidgeModel):
    coef = pd.Series(final.m[-1].coef_, index=final.feats)
    print(f'final fit on {seasons[0]} to {seasons[-1]}, places per SD of each feature (standardised ridge coefficients)')
    print(coef.sort_values().round(3).to_string())
    joblib.dump(final, out / 'outcome_ridge.joblib')
pd.concat(test_preds, names=['Model']).reset_index(level=0).to_parquet(out / 'outcome_test_preds.parquet', index=False)
