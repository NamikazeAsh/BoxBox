"""The degradation evaluation task, shared by every wear model (notes/evaluation.md): clean dry race laps
(model_laps), and the anchored forecast, where each stint's first clean laps set its pace level and a model
predicts the rest. A model only needs wear(laps) (seconds of wear up to a constant per stint) and
dirty_air(laps) (seconds lost to the car ahead, 0 for models without it).
"""
import numpy as np
import pandas as pd

from boxbox import features

STINT = ['Year', 'Round', 'Driver', 'Stint']
RACE = ['Year', 'Round']
KEY = ['Circuit', 'Compound']
COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
AHEAD_BINS = [0, 5, 10, 20, 100]
AGE_TERMS = {'linear': lambda a: a, 'sqrt': np.sqrt, 'log': np.log}


def model_laps(laps, seasons, cfg, train=False):
    """Clean dry race laps from the given seasons, in stints with enough clean laps to inform a slope.
    StintLaps is the full stint length in laps, clean or not. RaceLap is the lap of the race (practice runs
    have none, so the race start offsets never apply to them). RaceLaps is the race distance in laps. FuelKg
    and TrackTemp are carried for models that use them.

    train=True also drops laps deleted for track limits and laps run close behind a backmarker
    (degradation.train_clean). Only for fitting: scored laps stay the same, so the test does not get easier."""
    laps = laps.assign(StintLaps=laps.groupby(STINT)['LapNumber'].transform('size'),
                       RaceLaps=laps.groupby(RACE)['LapNumber'].transform('max'))
    d = laps[laps['Year'].isin(seasons) & laps['IsClean'] & ~laps['WetRace'] & laps['TyreLife'].notna()]
    if train:
        c = cfg['degradation'].get('train_clean', {})
        if c.get('drop_deleted'):
            d = d[~d['Deleted'].fillna(False).astype(bool)]
        if c.get('backmarker_gap_s'):
            d = d[~(features.backmarker_gap(laps[laps['Year'].isin(seasons)]).loc[d.index] < c['backmarker_gap_s'])]
    d = d[d.groupby(STINT)['LapNumber'].transform('size') >= cfg['degradation']['min_fit_laps']]
    d = d.assign(RaceLap=d['LapNumber'])
    cols = STINT + KEY + ['Team', 'LapNumber', 'RaceLap', 'RaceLaps', 'TyreLife', 'StintLaps', 'GapAhead_s', 'FuelKg', 'TrackTemp',
                          'LapTimeFC_s']
    return d[[c for c in cols if c in d]].sort_values(STINT + ['TyreLife'])


class ConstantPace:
    """Baseline: no degradation, every lap of a stint at its anchor pace."""
    name = 'constant pace'

    def fit(self, laps):
        return self

    def wear(self, laps):
        return pd.Series(0.0, index=laps.index)

    def dirty_air(self, laps):
        return pd.Series(0.0, index=laps.index)


def anchored_forecast(model, laps, cfg):
    """The evaluation task: the first anchor_laps clean laps of each stint set its pace level, the model
    predicts the rest. Stints with fewer than min_eval_laps laps after the anchor are not scored. The level is
    clean air pace (anchor laps less their dirty air), and so is the forecast. degradation.anchor_level picks
    the mean or the median over the anchor laps (the median ignores one slow lap; lap noise is skewed slow)."""
    c = cfg['degradation']
    i = laps.groupby(STINT).cumcount()
    anchor, rest = laps[i < c['anchor_laps']], laps[i >= c['anchor_laps']].copy()
    rest = rest[rest.groupby(STINT)['LapNumber'].transform('size') >= c['min_eval_laps']]

    level = (anchor['LapTimeFC_s'] - model.wear(anchor) - model.dirty_air(anchor)).groupby(
        [anchor[k] for k in STINT]).agg(c.get('anchor_level', 'mean')).rename('Level')
    last = anchor.groupby(STINT)['TyreLife'].max().rename('AnchorAge')
    rest = rest.join(level, on=STINT).join(last, on=STINT)
    rest['Pred_s'] = rest['Level'] + model.wear(rest)
    rest['Err_s'] = rest['Pred_s'] - rest['LapTimeFC_s']
    rest['LapsAhead'] = pd.cut(rest['TyreLife'] - rest['AnchorAge'], AHEAD_BINS)
    return rest


def forecast_cuts(laps, cfg):
    """Race lap at which each stint is forecast: its last anchor lap (anchored_forecast's anchor)."""
    i = laps.groupby(STINT).cumcount()
    return laps[i < cfg['degradation']['anchor_laps']].groupby(STINT)['LapNumber'].max().rename('Cut')


def forecast(model, laps, cfg):
    """A fitted model's anchored forecast: its own forecast method if it has one (models that predict the laps
    after the anchor directly, such as trees), else anchored_forecast on its wear."""
    return model.forecast(laps, cfg) if hasattr(model, 'forecast') else anchored_forecast(model, laps, cfg)
