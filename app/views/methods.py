"""How every number in the app was made: data, splits, metrics, baselines, test scores."""
import streamlit as st

import data
from boxbox import dashboard, evaluate
from components import definitions, page_header, section, stats

cfg = data.cfg()
yrs = data.seasons()
deg_seasons, out_seasons = data.training('degradation'), data.training('outcome')
deg = dashboard.degradation_scores(data.model_output('degradation_test_preds')).set_index('Model')
out = dashboard.outcome_scores(data.model_output('outcome_test_preds')).set_index('Model')
ridge = next(m for m in out.index if m != 'grid')
split = lambda s: evaluate.season_splits(s)
seq = lambda ys: ', '.join(str(y) for y in ys)

page_header('Methods', 'Research', 'Every split is by whole season, never by random lap')

section('Data')
definitions([
    ('Sources', 'FastF1 (official live timing archive) and the Jolpica API (results, grids, pit stops).'),
    ('Seasons', f'{yrs[0]} to {yrs[-1]}. {seq(cfg["holdout_seasons"])} is held out for the shift study and not read by the app.'),
    ('Clean laps', f'Lap 1, pit laps, safety car laps, laps within {cfg["clean_lap"]["traffic_gap_s"]} s of the car ahead and laps '
                   f'over {cfg["clean_lap"]["outlier_pct"]:.2f} x the driver\'s race median are left out.'),
    ('Fuel', f'Assumed: {cfg["fuel"]["start_kg"]:.0f} kg burned evenly over the race, {cfg["fuel"]["s_per_kg"]} s per kg. No fuel data is public.'),
    ('Circuit key', 'One name per track across renamed venues and changed layouts.'),
])

section('Tyre degradation', f'test season {split(deg_seasons)["test"][1]}')
stats([
    {'value': f'{deg.loc["simulator average", "MAE_s"]:.3f}', 'unit': 's', 'key': True, 'label': 'lap error, simulator average'},
    {'value': f'{deg.loc["constant pace", "MAE_s"]:.3f}', 'unit': 's', 'label': 'lap error, no tyre wear'},
    {'value': f'{int(deg.loc["simulator average", "N"]):,}', 'label': 'laps scored'},
])
definitions([
    ('Training', f'{deg_seasons[0]} to {deg_seasons[-1]}: the 18 inch tyre, ground effect era.'),
    ('Validation', f'Train on the seasons before, score {seq(y for _, y in split(deg_seasons)["validation"])}. Models are chosen here.'),
    ('Test', f'Train on {seq(split(deg_seasons)["test"][0])}, score {split(deg_seasons)["test"][1]} once for the chosen models.'),
    ('Metric', 'Mean absolute error of each forecast lap in seconds, plus interval hit rate where a model gives intervals.'),
    ('Baseline', 'No tyre wear: every lap as fast as the stint\'s first clean laps.'),
])

section('Finishing position', f'test season {split(out_seasons)["test"][1]}')
stats([
    {'value': f'{out.loc[ridge, "MAE"]:.2f}', 'key': True, 'label': 'places off, the model'},
    {'value': f'{out.loc["grid", "MAE"]:.2f}', 'label': 'places off, the grid order'},
    {'value': f'{100 * out.loc[ridge, "Podium"]:.0f}', 'unit': '%', 'label': 'of podium finishers found'},
])
definitions([
    ('Training', f'{out_seasons[0]} to {out_seasons[-1]}, classified finishers only.'),
    ('Validation', f'Rolling: each of {seq(y for _, y in split(out_seasons)["validation"])} scored by a model trained on the seasons before it.'),
    ('Features', 'Grid slot, qualifying gap, recent driver and team form, race pace form, how well the circuit holds grid order.'),
    ('Metric', 'Mean absolute error in places, Spearman correlation per race, podium and top ten hit rates.'),
    ('Baseline', 'The grid order, ranked among the cars that finished.'),
])
