"""Overtaking page tables (app/views/overtaking.py), built by scripts/build_overtaking.py.

The learned pass model (overtaking.PassLogit, the name in dashboard.overtaking.model) fitted on the close pairs
(overtaking.pass_pairs: B within the passing window of the car ahead at the line, green laps, no pit lap) of the
seasons before the test season, so the page is honest for a 2025 view: 2022 to 2024 with the current settings.
Free air pace comes from a lap model (simulator.laptime.LapModel) fitted on the same seasons. Also scored on the
rolling validation folds inside those seasons (train on the seasons before, score the next) against the
circuit's pass rate. The test season's pairs are never built or scored here.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from boxbox import overtaking as ot
from boxbox.config import seasons_for
from boxbox.evaluate import season_splits
from boxbox.simulator.laptime import LapModel

TABLES = ['circuits', 'curves', 'effects', 'coefs', 'validation', 'calibration', 'meta']
NEEDS = ['laps']


def settings(cfg):
    s = dict(cfg['dashboard'].get('overtaking') or {})
    defaults = {'model': 'pass logit, C 1.0, k 200', 'gap_s': 0.6, 'delta_s': 0.5, 'delta_range': [-0.5, 2.5],
                'contrast': ['Sakhir', 'Monaco'], 'curve_circuits': ['Monaco', 'Barcelona', 'Monza', 'Sakhir'],
                'calibration_bins': 10}
    return {**defaults, **s}


def pairs(laps, years, cfg):
    """Close pairs of years, free air pace from a lap model fitted on the same seasons."""
    lm = LapModel(cfg).fit(laps, years, laps['Circuit'].unique())
    return pairs_with(laps, lm, years, cfg)


def pairs_with(laps, lm, years, cfg):
    free = ot.free_air_laps(laps, lm, years)
    return ot.pass_pairs(laps.loc[free.index], free, cfg['simulator']['overtaking']['window_s'])


def model(cfg):
    """A fresh pass model of the configured name (overtaking.candidates)."""
    name = settings(cfg)['model']
    c = ot.candidates(cfg)
    if name not in c:
        raise KeyError(f'dashboard.overtaking.model {name!r} is not an overtaking candidate (simulator.overtaking.logit grid)')
    return c[name]()


def circuit_rate(m, circuits):
    """The circuit pass rate the model leans on (shrunk toward the all circuit rate)."""
    z = m._logit(circuits)
    return 1 / (1 + np.exp(-z))


def grid(m, circuit, **features):
    """Pass probability at one circuit (None: the all circuit rate) over the one feature given as an array, every
    other feature (Delta, Gap, AgeDiff, CompDiff, StuckRun) held at the value given."""
    (name, values), = [(k, v) for k, v in features.items() if np.ndim(v)]
    ref = {k: v for k, v in features.items() if not np.ndim(v)}
    d = pd.DataFrame({name: values})
    for k, v in ref.items():
        d[k] = v
    d['Circuit'] = circuit
    return d.assign(Prob=m.prob(d))


def calibration(p, prob, bins):
    """Mean predicted pass probability against the observed pass rate in quantile bins of the prediction."""
    q = pd.qcut(pd.Series(prob).rank(method='first'), bins, labels=False)
    g = pd.DataFrame({'Pred': prob, 'Real': p['Pass'].to_numpy(float), 'Bin': q.to_numpy()}).groupby('Bin')
    return pd.DataFrame({'Pred': g['Pred'].mean(), 'Real': g['Real'].mean(), 'Pairs': g.size()}).reset_index()


def build(t, cfg, log=print):
    """Every table of the page. t: processed tables (NEEDS)."""
    s = settings(cfg)
    laps = t['laps']
    split = season_splits(seasons_for(cfg, 'overtaking'))
    train, test_year = split['test']
    t0 = time.time()
    val, cal = [], []
    for tr, year in split['validation']:   # inside the training seasons; the test season is never scored here
        lm = LapModel(cfg).fit(laps, tr, laps['Circuit'].unique())
        a, b = pairs_with(laps, lm, tr, cfg), pairs_with(laps, lm, [year], cfg)
        m = model(cfg).fit(a)
        y = b['Pass'].to_numpy(bool)
        pm, pc_ = np.clip(m.prob(b), 1e-6, 1 - 1e-6), np.clip(circuit_rate(m, b['Circuit']), 1e-6, 1 - 1e-6)
        val.append({'Year': year, 'TrainFrom': tr[0], 'TrainTo': tr[-1], 'Pairs': len(b), 'Rate': y.mean(),
                    'LogLoss': log_loss(y, pm, labels=[False, True]), 'LogLossCircuit': log_loss(y, pc_, labels=[False, True]),
                    'LogLossAll': log_loss(y, np.full(len(y), a['Pass'].mean()), labels=[False, True]),
                    'Passes': int(y.sum()), 'PassesPred': float(pm.sum())})
        cal.append(calibration(b, pm, s['calibration_bins']).assign(Year=year))
        log(f'validation {year} (train {tr[0]} to {tr[-1]}): log loss {val[-1]["LogLoss"]:.3f}, circuit rate {val[-1]["LogLossCircuit"]:.3f}')
    p = pairs(laps, train, cfg)
    m = model(cfg).fit(p)
    log(f'pass model fitted on {len(p):,} close pairs of {train[0]} to {train[-1]} ({time.time() - t0:.0f} s so far)')

    races = p.groupby('Circuit')[['Year', 'Round']].apply(lambda g: len(g.drop_duplicates()))
    c = p.groupby('Circuit')['Pass'].agg(Passes='sum', Pairs='size').join(races.rename('Races'))
    c['Rate'] = c['Passes'] / c['Pairs']
    c['ModelRate'] = circuit_rate(m, c.index)
    ref = {'Gap': s['gap_s'], 'AgeDiff': 0.0, 'CompDiff': 0.0, 'StuckRun': 0.0}
    c['ProbRef'] = [grid(m, ci, Delta=np.array([s['delta_s']]), **ref)['Prob'].iloc[0] for ci in c.index]
    c = c.reset_index().sort_values('Rate', ascending=False).reset_index(drop=True)

    lo, hi = s['delta_range']
    deltas = np.round(np.arange(lo, hi + 1e-9, 0.05), 2)
    curves = pd.concat([grid(m, ci, Delta=deltas, **ref).assign(Circuit=ci) for ci in c['Circuit']] +
                       [grid(m, None, Delta=deltas, **ref).assign(Circuit='All circuits')], ignore_index=True)

    base = {'Delta': s['delta_s'], **ref}
    eff = []
    for name, values in [('AgeDiff', np.arange(-20, 21, 1.0)), ('StuckRun', np.arange(0, ot.STUCK_CAP + 1, 1.0)),
                         ('CompDiff', np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))]:
        g = grid(m, None, **{**{k: v for k, v in base.items() if k != name}, name: values})
        eff.append(pd.DataFrame({'Feature': name, 'Value': values, 'Prob': g['Prob'].to_numpy()}))
    eff = pd.concat(eff, ignore_index=True)

    coefs = pd.DataFrame({'Feature': list(m.coef), 'Coef': list(m.coef.values())})
    coefs['OddsRatio'] = np.exp(coefs['Coef'])
    meta = pd.DataFrame([{'TrainFrom': train[0], 'TrainTo': train[-1], 'TestYear': test_year, 'Model': m.name, 'C': m.C, 'k': m.k,
                          'Features': ', '.join(m.feats), 'Intercept': m.intercept, 'Pairs': len(p), 'Passes': int(p['Pass'].sum()),
                          'Races': int(p[['Year', 'Round']].drop_duplicates().shape[0]), 'Rate': m.rate,
                          'WindowS': cfg['simulator']['overtaking']['window_s'], 'GapS': s['gap_s'], 'DeltaS': s['delta_s'],
                          'EngineShift': getattr(m, 'engine_shift', 0.0), 'Seconds': round(time.time() - t0, 1),
                          'Built': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}])
    return {'circuits': c, 'curves': curves, 'effects': eff, 'coefs': coefs, 'validation': pd.DataFrame(val),
            'calibration': pd.concat(cal, ignore_index=True) if cal else pd.DataFrame(), 'meta': meta}


def save(tables, cfg):
    d = cfg['paths']['models']
    d.mkdir(exist_ok=True)
    for k, v in tables.items():
        v.to_parquet(d / f'overtaking_{k}.parquet', index=False)
    return d


def load(cfg):
    d = cfg['paths']['models']
    return {k: pd.read_parquet(d / f'overtaking_{k}.parquet') for k in TABLES}
