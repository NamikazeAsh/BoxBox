"""Every model family on the same rolling folds and the same scored laps: point error, error by horizon and on
old tyres, and interval scores where a model has quantiles (its own, or conformal)."""
import pandas as pd

from boxbox import evaluate
from boxbox.degradation.bayes import BayesDegradation
from boxbox.degradation.ensemble import KINDS
from boxbox.degradation.gam import QuadraticDegradation, SplineDegradation
from boxbox.degradation.intervals import Conformal, interval_scores
from boxbox.degradation.linear import LinearDegradation, lap_terms_kw
from boxbox.degradation.task import ConstantPace, forecast, model_laps
from boxbox.degradation.trees import LGBMDegradation


def ladder(cfg):
    """{name: factory} for the comparison: baselines, the linear model, the quadratic and GAM curves, LightGBM
    variants, the Bayesian model, the averages in degradation.ensembles (members by name) and conformal
    intervals for the models in degradation.intervals.conformal_models."""
    c = cfg['degradation']
    lv, folds = c['intervals']['levels'], c['intervals']['conformal_folds']
    lin_kw = dict(stint_length=True, length_ref=c['stint_length_ref'], **lap_terms_kw(cfg))
    g, t = c['gam'], c['lgbm']
    tree = lambda **kw: LGBMDegradation(cfg, num_leaves=t['num_leaves'][0], n_estimators=t['n_estimators'][0],
                                        min_child_samples=t['min_child_samples'], learning_rate=t['learning_rate'], **kw)
    f = {'constant pace': ConstantPace,
         'linear v1': lambda: LinearDegradation(10),
         'linear': lambda: LinearDegradation(10, **lin_kw),
         'quadratic': lambda: QuadraticDegradation(10, max_age=g['max_age'], **lin_kw)}
    for lam in g['lam_grid']:
        f[f'gam, lam {lam}'] = lambda lam=lam: SplineDegradation(10, lam=lam, knot_step=g['knot_step'], max_age=g['max_age'], **lin_kw)
    for leaves in t['num_leaves']:
        for trees in t['n_estimators']:
            f[f'lgbm, leaves {leaves}, trees {trees}'] = lambda leaves=leaves, trees=trees: LGBMDegradation(
                cfg, num_leaves=leaves, n_estimators=trees, min_child_samples=t['min_child_samples'], learning_rate=t['learning_rate'])
    f['lgbm, linear leaves'] = lambda: tree(linear_tree=True)
    f['lgbm huber'] = lambda: tree(objective='huber')
    f['lgbm huber, monotone'] = lambda: tree(objective='huber', monotone=True)
    f['lgbm huber, linear leaves, monotone'] = lambda: tree(objective='huber', linear_tree=True, monotone=True)
    f['lgbm quantile'] = lambda: tree(objective='quantile', levels=lv)
    f['bayes'] = lambda: BayesDegradation(cfg, c['bayes']['warmup'], c['bayes']['samples'], c['bayes']['chains'])
    for name, e in (c.get('ensembles') or {}).items():
        if e['members']:
            f[name] = lambda name=name, e=e: KINDS[e['kind']]([f[m]() for m in e['members']], name)
    for name in c['intervals'].get('conformal_models', []):
        f[f'{name}, conformal'] = lambda name=name: Conformal(f[name], cfg, lv, folds)
    return f


def run(laps, cfg, seasons, names=None, log=print):
    """Forecasts of every model (or the named ones) on the validation folds. Returns {name: predictions}."""
    f = ladder(cfg)
    preds = {}
    for train_years, year in evaluate.season_splits(seasons)['validation']:
        train, test = model_laps(laps, train_years, cfg, train=True), model_laps(laps, [year], cfg)
        for name in names or f:
            preds.setdefault(name, []).append(forecast(f[name]().fit(train), test, cfg))
            log(f'  {year}: {name}')
    return {n: pd.concat(p) for n, p in preds.items()}


def table(preds, cfg, old_tyre=30):
    """One row per model: MAE, bias, MAE per validation season, 21+ laps after the anchor, laps on tyres older
    than old_tyre, and interval scores."""
    lv = cfg['degradation']['intervals']['levels']
    rows = {}
    for n, p in preds.items():
        e = p['Err_s']
        r = {'MAE_s': e.abs().mean(), 'Bias_s': e.mean(), 'N': len(p)}
        r.update({f'MAE_{y}': e[p['Year'] == y].abs().mean() for y in sorted(p['Year'].unique())})
        r['MAE_21+ahead'] = e[p['TyreLife'] - p['AnchorAge'] > 20].abs().mean()
        r[f'MAE_age>{old_tyre}'] = e[p['TyreLife'] > old_tyre].abs().mean()
        r.update(interval_scores(p, lv).to_dict())
        rows[n] = r
    return pd.DataFrame(rows).T
