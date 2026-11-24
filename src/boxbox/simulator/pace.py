"""Each car's base pace for the simulator: its level on a fresh Medium at race fraction 0 with empty tanks
(LapModel.driver_bases), as a % gap to the race's median base. Predicted before the race from the pre race
features (pre_race: v1 qualifying gap, team and driver race pace form; v2 qualifying gap, grid slot and the
driver's and team's finish form, modelling D18) by ridge regression; the
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
FEATURES = ['QualiGap', 'TeamPaceForm', 'DriverPaceForm']                    # v1
FEATURES_V2 = ['QualiGap', 'Grid', 'TeamFinishForm', 'DriverFinishForm']     # v2 (modelling D18): finish form for race pace form
EXTRA = ['TeamFinishForm', 'DriverFinishForm']


def pre_race(results, qualifying, sprint_results, events, laps, cfg):
    """pitdecision.pre_race plus the outcome predictor's finish form (mean classified finish over the previous
    form_races, driver and team by lineage), all known before the start. Every caller of the simulator builds its
    pre race rows here, so the pace model and the rival pit model read the same table."""
    from boxbox import outcome, pitdecision
    e = outcome.entries(results, qualifying, sprint_results, events, cfg, laps)
    return e[DRIVER_RACE + pitdecision.PRE_RACE + EXTRA + ['Lineage']]


def pace_targets(bases, pre, events):
    """Driver race bases with Pace (% behind the race's median base), the circuit and the pre race features."""
    b = bases.merge(events[RACE + ['Circuit']], on=RACE, how='left')
    med = b.groupby(RACE)['Base'].transform('median')
    b['Pace'] = 100 * (b['Base'] / med - 1)
    b['RaceMedian'] = med
    return b.merge(pre, on=DRIVER_RACE, how='left')


def team_share(d):
    """Share of the pace error two teammates have in common (their residuals' covariance over the residual
    variance, races where both have a pace), between 0 and 0.9; 0 without a Lineage column. The simulator draws that
    share once per team, so a car the model overrates takes its teammate with it."""
    if 'Lineage' not in d:
        return 0.0
    two = d.dropna(subset=['Res', 'Lineage']).sort_values('Driver').groupby(RACE + ['Lineage'])['Res'].agg(list)
    two = two[two.map(len) == 2]
    if len(two) < 10:
        return 0.0
    a, b = np.array([x[0] for x in two]), np.array([x[1] for x in two])
    return float(np.clip(np.cov(a, b)[0, 1] / d['Res'].var(), 0.0, 0.9))


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
        res = d['Pace'] - self.m.predict(d[self.feats].astype(float))
        self.sd = float(np.std(res))
        self.team_share = team_share(d.assign(Res=res))
        return self

    def predict(self, d):
        return self.m.predict(d[self.feats].astype(float))


def candidates(cfg):
    """{name: factory}: the baselines, v1 ridge over its alpha grid, and every simulator.pace.feature_sets entry
    over its alpha grid."""
    p = cfg['simulator']['pace']
    f = {ConstantPace.name: ConstantPace,
         'ridge, alpha 10, QualiGap': lambda: CarPace(10.0, ['QualiGap'])}
    for a in p['ridge_alpha']:
        f[CarPace(a).name] = lambda a=a: CarPace(a)
    for feats in p.get('feature_sets', []):
        for a in p.get('feature_set_alpha', p['ridge_alpha']):
            f[CarPace(a, feats).name] = lambda a=a, feats=feats: CarPace(a, feats)
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


def quali_medians(qualifying):
    """Per race: the median over drivers of their best qualifying lap, s (known before the start)."""
    best = qualifying[['Q1_s', 'Q2_s', 'Q3_s']].min(axis=1)
    return best.groupby([qualifying['Year'], qualifying['Round']]).median().rename('QualiMedian')


class QualiReference:
    """A race's reference time (the median base, s) from its own qualifying: the qualifying median times the
    circuit's ratio of race median base to qualifying median over the training races, pulled toward the all
    circuit ratio with weight k races. Cars get faster through a season and between seasons, so last season's base
    at the circuit (v1) runs slow; same weekend qualifying carries the current speed."""

    def __init__(self, k=1.0):
        self.k = k

    def fit(self, targets, qualifying):
        self.medians = quali_medians(qualifying)
        r = targets.drop_duplicates(RACE)[RACE + ['Circuit', 'RaceMedian']].join(self.medians, on=RACE).dropna()
        ratio = r['RaceMedian'] / r['QualiMedian']
        self.all = float(ratio.median())
        g = ratio.groupby(r['Circuit']).agg(['sum', 'size'])
        self.ratio = (g['sum'] + self.k * self.all) / (g['size'] + self.k)
        return self

    def ref(self, year, rnd, circuit):
        """Reference time, s, or None when the race has no qualifying times."""
        q = self.medians.get((year, rnd), np.nan)
        return None if not np.isfinite(q) else float(self.ratio.get(circuit, self.all) * q)
