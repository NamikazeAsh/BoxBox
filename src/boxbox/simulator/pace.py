"""Each car's base pace for the simulator: its level on a fresh Medium at race fraction 0 with empty tanks
(LapModel.driver_bases), as a % gap to the race's median base. Predicted before the race from the pre race
features (pitdecision.pre_race: qualifying gap, team and driver race pace form) by ridge regression; the
simulator draws each car's pace from the prediction plus a normal error with the training residual SD, so
rival pace is not known exactly. A circuit's reference time (the median base, s) is its latest training season's.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RACE = ['Year', 'Round']
DRIVER_RACE = RACE + ['Driver']
FEATURES = ['QualiGap', 'TeamPaceForm', 'DriverPaceForm']


def pace_targets(bases, pre, events):
    """Driver race bases with Pace (% behind the race's median base), the circuit and the pre race features."""
    b = bases.merge(events[RACE + ['Circuit']], on=RACE, how='left')
    med = b.groupby(RACE)['Base'].transform('median')
    b['Pace'] = 100 * (b['Base'] / med - 1)
    b['RaceMedian'] = med
    return b.merge(pre, on=DRIVER_RACE, how='left')


class ConstantPace:
    """Baseline: every car at the race median."""
    name = 'race median'

    def fit(self, d):
        self.sd = float(d['Pace'].std())
        return self

    def predict(self, d):
        return np.zeros(len(d))


class CarPace:
    """Ridge of Pace on the pre race features (missing form filled with the training median)."""

    def __init__(self, alpha=10.0, feats=FEATURES):
        self.alpha, self.feats = alpha, list(feats)
        self.name = f'ridge, alpha {alpha}' + ('' if self.feats == FEATURES else f", {'+'.join(self.feats)}")

    def fit(self, d):
        d = d[d['Pace'].notna()]
        self.m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
                               Ridge(alpha=self.alpha)).fit(d[self.feats].astype(float), d['Pace'])
        self.sd = float(np.std(d['Pace'] - self.m.predict(d[self.feats].astype(float))))
        return self

    def predict(self, d):
        return self.m.predict(d[self.feats].astype(float))


def candidates(cfg):
    f = {ConstantPace.name: ConstantPace,
         'ridge, alpha 10, QualiGap': lambda: CarPace(10.0, ['QualiGap'])}
    for a in cfg['simulator']['pace']['ridge_alpha']:
        f[CarPace(a).name] = lambda a=a: CarPace(a)
    return f


def pace_metrics(d, pred):
    e = pred - d['Pace'].to_numpy()
    rank = lambda x: pd.Series(x, index=d.index).groupby([d['Year'], d['Round']]).rank()
    sp = pd.DataFrame({'p': rank(pred), 'y': rank(d['Pace'].to_numpy()), 'Y': d['Year'], 'R': d['Round']}).groupby(['Y', 'R']).apply(
        lambda g: g['p'].corr(g['y']), include_groups=False)
    return {'MAE, %': np.abs(e).mean(), 'RMSE, %': np.sqrt((e ** 2).mean()), 'Bias, %': e.mean(),
            'Spearman per race': sp.mean(), 'N': len(d)}


def reference_times(targets):
    """Per circuit: the race median base, s, of the circuit's latest season in targets."""
    r = targets.drop_duplicates(RACE).sort_values('Year')
    return r.groupby('Circuit')['RaceMedian'].last()
