"""Tables shaped for the Streamlit app (app/). Everything here is plain pandas on saved tables and model
outputs; definitions come from features, evaluate, outcome and degradation, never redone here.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from boxbox import evaluate, outcome
from boxbox.degradation.task import AGE_TERMS

RACE = ['Year', 'Round']

# Saved model names to the names the app shows. Anything not listed keeps its own name.
MODEL_NAMES = {
    'constant pace': 'No tyre wear',
    'linear, shrink 20, stint length, dirty air, race start': 'Linear',
    'linear, shrink 10, stint length, dirty air, race start': 'Linear (previous)',
    'linear, shrink 20, stint length, dirty air, race start, weekend': 'Linear + weekend',
    'linear, shrink 10, stint length, dirty air, race start, weekend': 'Linear (previous) + weekend',
    'quadratic': 'Quadratic',
    'bayes': 'Hierarchical Bayes',
    'lgbm, leaves 8, trees 300': 'LightGBM',
    'simulator average': 'Simulator average',
    'live average': 'Live average',
    'grid': 'Grid order',
    'ridge, alpha 1000': 'Ridge',
}
BASELINES = {'constant pace', 'grid'}


def table(cfg, name, holdout=False, columns=None):
    """A processed table. Holdout seasons are refused until dashboard.show_holdout is switched on."""
    if holdout and not cfg['dashboard']['show_holdout']:
        raise PermissionError('holdout seasons are locked until the shift study (dashboard.show_holdout)')
    base = cfg['paths']['processed'] / ('holdout' if holdout else '')
    return pd.read_parquet(base / f'{name}.parquet', columns=columns)


def model_output(cfg, name):
    return pd.read_parquet(cfg['paths']['models'] / f'{name}.parquet')


def season_summary(events, pit_loss, sc):
    """Per season: races, median green flag pit loss over races, and races with a safety car or a VSC."""
    races = events.groupby('Year').size().rename('Races')
    pit = pit_loss.groupby('Year')['Median_s'].median().rename('PitLoss_s')
    flags = sc.groupby('Year')[['SafetyCar', 'VSC']].sum()
    return pd.concat([races, pit, flags], axis=1).reset_index()


def pit_loss_by_circuit(pit_loss, year):
    """Median pit loss per circuit in one season, cheapest first."""
    p = pit_loss[pit_loss['Year'] == year]
    return p.groupby('Circuit')['Median_s'].median().sort_values().rename('PitLoss_s').reset_index()


def wear_curve(cells, circuit, compound, stint_laps, length_ref, level=0.8):
    """Seconds lost against a tyre on its first lap, at each age through a stint of stint_laps, from the saved
    linear model: (circuit slope + stint length term) x (f(age) - f(1)). The band holds `level` of laps under
    the cell's lap noise. A circuit without the compound falls back to the compound's pooled slope."""
    c = cells[(cells['Circuit'] == circuit) & (cells['Compound'] == compound)]
    pooled = cells[cells['Compound'] == compound]
    row = c.iloc[0] if len(c) else pooled.iloc[0]
    slope = row['Slope'] if len(c) else pooled['Pooled'].iloc[0]
    sd = row['ResidSD_s'] if len(c) else pooled['ResidSD_s'].median()
    f = AGE_TERMS[row['AgeTerm']]
    age = pd.Series(range(1, int(stint_laps) + 1), dtype=float)
    loss = (slope + row['LengthCoef'] * (stint_laps - length_ref)) * (f(age) - f(1.0))
    half = norm.ppf((1 + level) / 2) * sd
    return pd.DataFrame({'Age': age, 'Loss_s': loss, 'Lo_s': loss - half, 'Hi_s': loss + half,
                         'FromCircuit': bool(len(c))})


def offsets(cfg, model='degradation'):
    """Lap level offsets of a saved wear model, split into dirty air (by gap band start) and race start laps."""
    o = model_output(cfg, f'{model}_offsets')
    return {t: g.assign(Key=g['Key'].astype(float)).drop(columns='Term') for t, g in o.groupby('Term')}


def degradation_scores(preds):
    """Test season error per model (evaluate.error_table), best first, with the 80% interval hit rate where the
    model saved quantiles."""
    t = pd.concat({n: evaluate.error_table(p) for n, p in preds.groupby('Model')}).droplevel(1)
    q = preds[preds['Q10'].notna()]
    t['Cover80'] = q['LapTimeFC_s'].between(q['Q10'], q['Q90']).groupby(q['Model']).mean()
    return t.sort_values('MAE_s').rename_axis('Model').reset_index()


def degradation_by_circuit(preds, models):
    """Test season MAE per circuit for the given models, one column each."""
    p = preds[preds['Model'].isin(models)]
    return p.groupby(['Circuit', 'Model'])['Err_s'].apply(lambda e: e.abs().mean()).unstack()


def gain(model_err, base_err):
    """Share of the baseline's error a model removes (0.2 = 20% less error)."""
    return 1 - model_err / base_err


def outcome_scores(preds):
    """Test season summary per model (outcome.summary), best first."""
    t = pd.DataFrame({n: outcome.summary(p) for n, p in preds.groupby('Model')}).T
    return t.sort_values('MAE').rename_axis('Model').reset_index()


def outcome_races(preds, events):
    """Races in the outcome predictions with their event names, in calendar order."""
    r = preds[RACE].drop_duplicates()
    return r.merge(events[RACE + ['EventName', 'Circuit']], on=RACE, how='left').sort_values(RACE).reset_index(drop=True)


def race_prediction(preds, year, rnd, model):
    """One race: each classified finisher's predicted and actual place under `model`, and the grid order's
    prediction for comparison."""
    p = preds[(preds['Year'] == year) & (preds['Round'] == rnd)]
    m = p[p['Model'] == model].set_index('Driver')
    g = p[p['Model'] == 'grid'].set_index('Driver')['Pred'].rename('GridPred')
    out = m[['Team', 'Lineage', 'Grid', 'Finish', 'Pred', 'Err']].join(g)
    return out.sort_values('Pred').reset_index()


def stint_plan(stints, year, rnd):
    """Stints each driver ran in one race (the stints table), in order."""
    s = stints[(stints['Year'] == year) & (stints['Round'] == rnd)]
    return s.sort_values(['Driver', 'Stint']).reset_index(drop=True)


def track_maps(cfg):
    """Every circuit outline pulled by `pull_data.py --track-maps` (empty when none are pulled)."""
    files = sorted((cfg['paths']['interim'] / 'track_maps').glob('*.parquet'))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def track_outline(maps, circuit, points=240):
    """One circuit drawn upright: X, Y rotated by FastF1's map angle, scaled so the longer side runs 0 to 1
    (Y pointing down, as screens draw it), thinned to about `points` points. None if not pulled.
    The first point is the start of the lap, the finish line."""
    t = maps[maps['Circuit'] == circuit] if len(maps) else maps
    if len(t) == 0:
        return None
    a = np.deg2rad(t['Rotation'].iloc[0])
    xy = t[['X', 'Y']].to_numpy(float) @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    xy = xy[::max(1, len(xy) // points)]
    xy = (xy - xy.min(axis=0)) / np.ptp(xy, axis=0).max()
    xy[:, 1] = xy[:, 1].max() - xy[:, 1]
    return pd.DataFrame(xy, columns=['X', 'Y'])
