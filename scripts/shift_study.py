"""The 2026 shift study (notes/shift_study.md). Logic in src/boxbox/shift.py.

    python scripts/shift_study.py --rehearsal    # train 2019 to 2021, score 2022 (and the 2021 reference); chooses SeasonScaled
    python scripts/shift_study.py --study        # train 2022 to 2025, score 2026 (and the 2025 reference); needs shift_study.season_adjust
    [--no-figures]

The study reads every 2026 round in data/processed/holdout/, so rerunning adds later rounds.
"""
import sys

import pandas as pd

from boxbox import outcome, shift
from boxbox.config import ROOT, load_config
from boxbox.degradation.task import RACE

cfg = load_config()
study = 'study' if '--study' in sys.argv else 'rehearsal' if '--rehearsal' in sys.argv else sys.exit(__doc__)
ss = cfg.get('shift_study') or {}
if study == 'study' and not ss.get('season_adjust'):
    sys.exit('shift_study.season_adjust is not set: run --rehearsal and freeze its choice first')
t = shift.load_tables(cfg, holdout=study == 'study')
laps = t['laps']
sp = shift.splits(cfg, study)
pd.set_option('display.width', 220)
out = {}
print(f'{study}: ' + ', '.join(f'{n} train {tr} test {te}' for n, (tr, te) in sp.items()))
print(f"test rounds in data: {sorted(laps.loc[laps['Year'] == sp['shift'][1], 'Round'].unique().tolist())}\n")

# Measure 1: degradation
flags = {}
if study == 'study':
    from boxbox import reference
    wide = reference.load_compounds(cfg)
    for name, (tr, te) in sp.items():
        flags[name] = shift.allocation_flags(wide, te)
        f = flags[name]
        print(f"D23 {te}: flagged {f.loc[f['Flagged'], 'Circuit'].tolist()}; no {te - 1} row {f.loc[f['New'], 'Circuit'].tolist()}")
    print()
for name, (tr, te) in sp.items():
    preds, fitted = shift.degradation(laps, cfg, tr, te)
    out[name] = {'preds': preds, 'fitted': fitted}
    print(f'== measure 1, {name}: train {tr}, test {te}')
    print(shift.score_table(preds).round(3).to_string())
    for by in ['Compound', 'LapsAhead']:
        print(shift.score_table(preds, by)[['MAE_s', 'Bias_s', 'Gain', 'N']].unstack('Model').round(3).to_string())
    if name in flags:
        fl = flags[name]
        kept = shift.drop_races(preds, fl.loc[fl['Flagged'], RACE].to_numpy())
        print(f'without D23 flagged circuits:'); print(shift.score_table(kept).round(3).to_string())
    new = shift.new_circuits(laps, cfg, tr, te)
    print(f'new circuits (no training laps): {new}')
    if new and study == 'study':
        fb, _ = shift.practice_fallback(fitted['linear'], t, cfg, tr, te)
        on = lambda p: p[p['Circuit'].isin(new)]
        print(shift.score_table({'constant pace': on(preds['constant pace']), 'linear (compound slope)': on(preds['linear']),
                                 'linear, practice fallback': on(fb)}).round(3).to_string())
    print('wear slopes, s per lap of tyre age (linear): training fit, test season alone')
    out[name]['rates'] = shift.wear_rates(laps, cfg, tr, te, fitted['linear'])
    print(out[name]['rates'].round(4).to_string(), '\n')

# Measure 2: in season recovery
if study == 'rehearsal':
    tr, te = sp['shift']
    grid = shift.season_grid(out['shift']['fitted']['linear'], laps, cfg, te)
    g = pd.Series({f'{s}, shrink {k}': p['Err_s'].abs().mean() for (s, k), p in grid.items()})
    print(f'== SeasonScaled grid on the rehearsal shift ({te}), pooled MAE'); print(g.round(4).to_string())
    scope, k = shift.choose_season_adjust(grid)
    print(f'chosen: scope {scope}, shrink {k}  ->  freeze in config shift_study.season_adjust: {{scope: {scope}, shrink: {k}}}\n')
else:
    scope, k = ss['season_adjust']['scope'], ss['season_adjust']['shrink']
    print(f'SeasonScaled frozen from the rehearsal: scope {scope}, shrink {k}\n')
for name, (tr, te) in sp.items():
    rec = shift.recovery(laps, cfg, tr, te, scope, k, base=out[name]['fitted']['linear'])
    out[name]['recovery'] = rec
    per, cum = shift.round_mae(rec)
    print(f'== measure 2, {name}: test {te}, MAE by round block')
    print(shift.block_table(rec).round(3).to_string())
    for v in ['season scaled', 'rolling refit']:
        print(f'{v}: stays ahead of frozen on cumulative MAE from round {shift.stays_ahead(cum, v)}')
    print('MAE per round'); print(per.round(3).T.to_string())
    if name in flags:
        fl = flags[name]
        kept = shift.drop_races(rec, fl.loc[fl['Flagged'], RACE].to_numpy())
        print('without D23 flagged circuits'); print(shift.block_table(kept).round(3).to_string())
    print()

# Measure 3: fresh tyre gaps and pit loss (study only: the rehearsal has no simulator inputs)
if study == 'study':
    from boxbox import reference
    ev = t['events'][RACE + ['Circuit']].drop_duplicates(RACE)
    stops = t['pit_loss_stops'].merge(ev, on=RACE, how='left')
    facts = reference.load_circuits(cfg)
    for name, (tr, te) in sp.items():
        print(f'== measure 3, {name}: test {te}')
        print('fresh tyre gap to Medium, s per lap (simulator wear model and compound pace):')
        print(shift.compound_gaps(laps, cfg, tr, te, out[name]['fitted']['simulator wear']).round(3).to_string())
        ptr = shift.component_train(cfg, 'pit_loss', tr, te)
        tab, races = shift.pit_loss(stops, cfg, ptr, te, facts)
        print(f'pit loss, train {ptr[0]} to {ptr[-1]}, race medians, D26 corrected:'); print(tab.round(3).to_string())
        print(races.round(2).to_string(index=False), '\n')

# Measure 4: finishing predictor
df = outcome.entries(t['results'], t['qualifying'], t['sprint_results'], t['events'], cfg, laps)
name_m = ss.get('outcome_model', 'ridge, alpha 1000')
for name, (tr, te) in sp.items():
    d2 = shift.component_train(cfg, 'outcome', tr, te)
    for label, years in [('D2 seasons', d2)] + ([('split seasons', tr)] if list(tr) != d2 else []):
        tab, _ = shift.finishing(df, cfg, years, te, name_m)
        print(f'== measure 4, {name}, {label}: train {years[0]} to {years[-1]}, test {te}')
        print(tab.round(3).to_string(), '\n')

# Measure 6: safety cars (study only)
if study == 'study':
    rows = {}
    for name, (tr, te) in sp.items():
        rows[f'{name} {te}'] = shift.safety_cars(laps, cfg, shift.component_train(cfg, 'safety_car', tr, te), te)
    print('== measure 6, NeutralModel'); print(pd.DataFrame(rows).T.round(3).to_string(), '\n')

if '--no-figures' not in sys.argv:
    paths = shift.save_figures(out, sp, study, ROOT / 'notes' / 'shift_figures')
    print('figures:', ', '.join(str(p) for p in paths))
