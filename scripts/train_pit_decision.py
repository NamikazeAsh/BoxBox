"""Choose and evaluate the rival pit decision model (pitdecision.py, pit_decision v1, plan first) and write the
final fit.

    python scripts/train_pit_decision.py [--validation]

Three parts, each chosen on the rolling validation folds (2026-11-05 experiments):
- timing: the per lap hazard given the planned stops left, scored with the real plan on laps with a planned stop
  still to come, against the hazard with no plan (the first version) and the baselines;
- plan: the stop count (1, 2, 3+) per classified finisher, scored on green flag races (v1 has no safety cars);
- tyre choice at a stop, with and without the planned stops left.
The replay simulates every classified finisher's stops through the green flag races of a scored season with the
plan drawn from the plan model, and with the real plan (the timing alone), against the no plan hazard. The test
season is scored only for the chosen models. Writes data/models/pit_decision.joblib (a PlanPolicy fit on all
pit_decision seasons) and pit_test_preds.parquet (timing, test). --validation stops before the test.
The season correction and weekend shifted wear of the no plan hazard were reported 2026-11-04 and are not used.
"""
import sys

import joblib
import pandas as pd

from boxbox import evaluate, pitdecision as pdm
from boxbox.config import load_config, seasons_for

cfg = load_config()
c = cfg['pit_decision']
seasons = seasons_for(cfg, 'pit_decision')
P = cfg['paths']['processed']
t = {n: pd.read_parquet(P / f'{n}.parquet') for n in ['laps', 'results', 'qualifying', 'sprint_results', 'events', 'pit_loss_stops']}
laps = t['laps']
pre = pdm.pre_race(t['results'], t['qualifying'], t['sprint_results'], t['events'], laps, cfg)
starts = pdm.race_starts(laps, pre, t['results'], seasons, cfg, green_only=False)   # every dry race, with its plan
rows = pdm.pit_laps(laps, pre, cfg)
rows = rows[rows['Year'].isin(seasons)].merge(starts[pdm.DRIVER_RACE + ['Plan']], on=pdm.DRIVER_RACE).reset_index(drop=True)
stops = pdm.stops_with_circuit(t['pit_loss_stops'], t['events'])
split = evaluate.season_splits(seasons)
validation_only = '--validation' in sys.argv
draws, seed = c['replay']['draws'], c['replay']['seed']
pd.set_option('display.width', 250)

print(f"{len(rows)} decision laps of classified finishers, {int(rows['PitStop'].sum())} tyre stops, "
      f"{rows.groupby(['Year', 'Round']).ngroups} races, {seasons[0]} to {seasons[-1]}")
print('plans, every dry race:', starts['Plan'].value_counts(normalize=True).sort_index().round(3).to_dict(),
      '| green races:', starts.loc[starts['Neutral'] == 0, 'Plan'].value_counts(normalize=True).sort_index().round(3).to_dict())
print('validation folds:', ', '.join(f'train {tr[0]} to {tr[-1]} score {y}' for tr, y in split['validation']), '\n')

timing = pdm.timing_candidates(cfg)
no_plan = {pdm.ConstantRate.name: pdm.ConstantRate, pdm.AgeRate.name: lambda: pdm.AgeRate(c['age_shrink'], c['age_bin']),
           'no plan logit, C 0.01': lambda: pdm.LogitHazard(0.01, c['spline_knots'])}
tyre_plain = lambda: pdm.TyreChoice(c['tyre_C'], c['force_new_compound_laps_left'])
tyre_plan = lambda: pdm.TyreChoice(c['tyre_C'], c['force_new_compound_laps_left'], pdm.TYRE_FEATURES + ['StopsLeft'])
plans = pdm.plan_candidates(cfg)


def timing_preds(folds, factories):
    """Scored timing rows with p per model. No plan models fit on every decision lap (they know no plan)."""
    out = {}
    for _, _, tr, te in folds:
        et = pdm.timing_rows(te)
        for name, make in factories.items():
            m = make().fit(tr if name in no_plan else pdm.timing_rows(tr))
            out.setdefault(name, []).append(et[pdm.DRIVER_RACE + ['LapNumber', 'Compound', 'TyreLife', 'Plan', 'PitStop']].assign(p=m.prob(et)))
    return {n: pd.concat(v, ignore_index=True) for n, v in out.items()}


def plan_table(folds, factories):
    out = {}
    for year, ctx, tr, _ in folds:
        scored = pdm.race_starts(laps, pre, t['results'], [year], cfg)
        ftr, fte, _ = pdm.plan_data(starts, scored, ctx, sorted(set(tr['Year'])), c['plan']['circuit_shrink'])
        for name, make in factories.items():
            out[(name, year)] = pdm.plan_metrics(make().fit(ftr), fte)
    return pd.DataFrame(out).T


def tyre_table(folds):
    out = {}
    for year, _, tr, te in folds:
        tt, et = pdm.timing_rows(tr), pdm.timing_rows(te)
        for name, make in [('no plan', tyre_plain), ('+ planned stops left', tyre_plan)]:
            out[(name, year)] = pdm.tyre_metrics(make().fit(tt), et)
    return pd.DataFrame(out).T


def replay_table(folds, timing_name, plan_name):
    out = {}
    for year, ctx, tr, _ in folds:
        scored = pdm.race_starts(laps, pre, t['results'], [year], cfg)
        ftr, _, cs = pdm.plan_data(starts, scored, ctx, sorted(set(tr['Year'])), c['plan']['circuit_shrink'])
        hz, ty, pm = timing[timing_name]().fit(pdm.timing_rows(tr)), tyre_plan().fit(pdm.timing_rows(tr)), plans[plan_name]().fit(ftr)
        policies = {'no plan logit, C 0.01': pdm.RivalPolicy(no_plan['no plan logit, C 0.01']().fit(tr), tyre_plain().fit(tr), ctx, cfg),
                    f'{timing_name}, real plan': pdm.PlanPolicy(None, None, hz, ty, ctx, cfg, oracle=True),
                    f'{timing_name}, {plan_name}': pdm.PlanPolicy(pm, cs, hz, ty, ctx, cfg)}
        for name, pol in policies.items():
            out[(name, year)] = pdm.replay_metrics(pdm.replay(pol, scored, draws, seed), scored)
    return pd.DataFrame(out).T


val_folds = pdm.fold_data(rows, laps, stops, split['validation'], cfg)
val = timing_preds(val_folds, {**no_plan, **timing})
vt = pdm.table(val)
print('pit timing, validation, laps with a planned stop left (real plan)'); print(vt.sort_values('LogLoss').round(4).to_string(), '\n')
print('by scored season'); print(pdm.table(val, 'Year')['LogLoss'].unstack().round(4).to_string(), '\n')
best = vt.loc[list(timing)].sort_values('LogLoss').index[0]
print(f'chosen timing: {best}\n')

# Which feature groups carry the timing model: nested groups, validation only
sets, acc = {}, []
for name, group in {**pdm.GROUPS, 'plan': pdm.PLAN_TIMING}.items():
    acc = acc + group
    sets[('' if not sets else '+ ') + name] = list(acc)
ablation = pdm.table(timing_preds(val_folds, {n: (lambda f=f: timing[best]().with_features(f)) for n, f in sets.items()}))
print(f'feature groups, {best}, validation'); print(ablation.loc[list(sets), ['LogLoss', 'Brier', 'PR_AUC']].round(4).to_string(), '\n')

pt = plan_table(val_folds, plans)
print('plan, validation, classified finishers of green flag races'); print(pt.round(3).to_string(), '\n')
pp = pdm.pooled(pt)
print('pooled log loss'); print(pp.sort_values().round(4).to_string(), '\n')
best_plan = pp.drop(pdm.PlanShare.name).idxmin()
print(f'chosen plan: {best_plan}\n')

print('tyre choice at a stop, validation'); print(tyre_table(val_folds).round(3).to_string(), '\n')
cols = ['1 stops, real', '1 stops, sim', '2 stops, real', '2 stops, sim', '3+ stops, real', '3+ stops, sim',
        'Mean stops, real', 'Mean stops, sim', 'Stops MAE', 'First stop MAE, laps', 'Drivers']
print('replay, validation (green flag races, classified finishers)')
print(replay_table(val_folds, best, best_plan)[cols].round(2).to_string(), '\n')
if validation_only:
    sys.exit()

test_fold = pdm.fold_data(rows, laps, stops, [split['test']], cfg)
test = timing_preds(test_fold, {**no_plan, best: timing[best]})
print(f"test: train {split['test'][0][0]} to {split['test'][0][-1]}, score {split['test'][1]}")
print('pit timing, test'); print(pdm.table(test).round(4).to_string(), '\n')
print(f'calibration, {best}, test'); print(pdm.calibration(test[best]).round(4).to_string(), '\n')
print('plan, test'); print(plan_table(test_fold, {pdm.PlanShare.name: plans[pdm.PlanShare.name], best_plan: plans[best_plan]}).round(3).to_string(), '\n')
print('tyre choice at a stop, test'); print(tyre_table(test_fold).round(3).to_string(), '\n')
print('replay, test'); print(replay_table(test_fold, best, best_plan)[cols].round(2).to_string(), '\n')

# Final fit on every pit_decision season
ctx = pdm.Context(cfg).fit(laps, stops, seasons, rows['Circuit'].unique())
full = pdm.timing_rows(pdm.add_features(rows, ctx))
ftr, _, cs = pdm.plan_data(starts, starts.head(0), ctx, seasons, c['plan']['circuit_shrink'])
policy = pdm.PlanPolicy(plans[best_plan]().fit(ftr), cs, timing[best]().fit(full), tyre_plan().fit(full), ctx, cfg)
out = cfg['paths']['models']
out.mkdir(exist_ok=True)
joblib.dump(policy, out / 'pit_decision.joblib')
pd.concat(test, names=['Model']).reset_index(level=0).to_parquet(out / 'pit_test_preds.parquet', index=False)
print(f"final fit on {seasons[0]} to {seasons[-1]}: {best_plan}, {best} and {policy.tyre.name}, written to {out / 'pit_decision.joblib'}")
