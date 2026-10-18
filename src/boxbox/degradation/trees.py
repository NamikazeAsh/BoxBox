"""LightGBM on the anchored task (the proposal's comparison with the linear model). A tree model predicts each
lap after the anchor directly: target is the lap minus the stint's anchor mean, from what is known at the
anchor. Variants: plain, linear leaves (each leaf fits a line, so predictions can keep rising beyond the tyre
ages seen in training), monotone in laps after the anchor, and quantile (one model per quantile level, for
intervals).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from boxbox.degradation.intervals import quantile_columns
from boxbox.degradation.task import COMPOUNDS, STINT, ConstantPace, anchored_forecast

FEATURES = ['LapsAheadN', 'TyreLife', 'AnchorAge', 'CompoundId', 'StintLaps', 'RaceFrac', 'FuelKg', 'TrackTemp',
            'CircuitId', 'AnchorGap', 'Stint']
CATEGORICAL = ['CircuitId', 'CompoundId']


def tree_frame(d, cfg, circuits):
    """One row per scored lap of model_laps d: the anchored task's level (anchor mean), the target, and the
    features. circuits maps a circuit to its id (unseen circuits get none)."""
    p = anchored_forecast(ConstantPace(), d, cfg)
    i = d.groupby(STINT).cumcount()
    a = d[i < cfg['degradation']['anchor_laps']]
    gap = a.get('GapAhead_s', pd.Series(np.nan, index=a.index)).fillna(99).clip(upper=5).groupby([a[k] for k in STINT]).mean().rename('AnchorGap')
    p = p.join(gap, on=STINT)
    return p.assign(LapsAheadN=p['TyreLife'] - p['AnchorAge'], CompoundId=p['Compound'].map(COMPOUNDS.index),
                    RaceFrac=p['LapNumber'] / p['RaceLaps'], CircuitId=p['Circuit'].map(circuits),
                    Target=p['LapTimeFC_s'] - p['Level'])


class LGBMDegradation:
    """LightGBM on the anchored task. objective 'l1' (median, matches the MAE metric), 'huber' (squared error
    within huber_s seconds, absolute beyond) or 'quantile' with levels (one booster per level, Pred_s is the
    median, quantile columns Q10 and so on). linear_tree fits a line in each leaf; monotone forces the forecast
    to rise with laps after the anchor (LightGBM allows it with huber, not with l1 or quantile)."""

    def __init__(self, cfg, num_leaves=8, n_estimators=300, min_child_samples=200, learning_rate=0.03,
                 objective='l1', levels=None, linear_tree=False, monotone=False, huber_s=1.0, seed=0):
        self.cfg, self.objective, self.levels, self.huber_s = cfg, objective, list(levels or []), huber_s
        self.params = dict(num_leaves=num_leaves, n_estimators=n_estimators, min_child_samples=min_child_samples,
                           learning_rate=learning_rate, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           linear_tree=linear_tree, random_state=seed, n_jobs=4, verbose=-1)
        if monotone:
            self.params['monotone_constraints'] = [1 if f == 'LapsAheadN' else 0 for f in FEATURES]
        self.name = (f'lgbm {objective}, leaves {num_leaves}, trees {n_estimators}' + (', linear leaves' if linear_tree else '')
                     + (', monotone' if monotone else ''))

    def fit(self, laps):
        self.circuits = {c: n for n, c in enumerate(sorted(laps['Circuit'].unique()))}
        f = tree_frame(laps, self.cfg, self.circuits)
        fit = lambda **kw: lgb.LGBMRegressor(**self.params, **kw).fit(f[FEATURES], f['Target'], categorical_feature=CATEGORICAL)
        if self.objective == 'quantile':
            self.models = {q: fit(objective='quantile', alpha=q) for q in self.levels}
        elif self.objective == 'huber':
            self.models = {None: fit(objective='huber', alpha=self.huber_s)}
        else:
            self.models = {None: fit(objective=self.objective)}
        return self

    def forecast(self, laps, cfg):
        p = tree_frame(laps, cfg, self.circuits)
        if self.objective == 'quantile':
            q = np.sort(np.column_stack([self.models[l].predict(p[FEATURES]) for l in self.levels]), axis=1)  # no crossing
            for col, v in zip(quantile_columns(self.levels), q.T):
                p[col] = p['Level'] + v
            p['Pred_s'] = p['Level'] + q[:, len(self.levels) // 2]
        else:
            p['Pred_s'] = p['Level'] + self.models[None].predict(p[FEATURES])
        p['Err_s'] = p['Pred_s'] - p['LapTimeFC_s']
        return p.drop(columns=['Target'])

    def importance(self):
        m = next(iter(self.models.values()))
        g = pd.Series(m.booster_.feature_importance('gain'), index=FEATURES)
        return g / g.sum()
