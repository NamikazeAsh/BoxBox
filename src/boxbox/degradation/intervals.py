"""Intervals for any wear model, and how to score them. The simulator samples lap times, so a model is judged
on its spread as well as its point forecast.

Conformal: the model is refit on the training seasons with whole races held out in turn, its out of fold
errors are collected per band of laps after the anchor, and their quantiles are added to the forecast. No
assumption about the error's shape; coverage holds if the test season errs like the training seasons.

Scores: coverage and width of each central interval (10 to 90 is the 80% interval), and the quantile score
(mean pinball loss over the levels, an approximation of CRPS; lower is better).
"""
import numpy as np
import pandas as pd

from boxbox.degradation.task import AHEAD_BINS, RACE, forecast


def quantile_columns(levels):
    return [f'Q{round(100 * q)}' for q in levels]


def out_of_fold(make, laps, cfg, folds=5, seed=0):
    """Anchored forecasts of the given laps, each race forecast by a model fitted without it (folds of whole
    races)."""
    races = laps[RACE].drop_duplicates().sample(frac=1, random_state=seed)
    fold = pd.Series(np.arange(len(races)) % folds, index=pd.MultiIndex.from_frame(races))
    g = fold.reindex(pd.MultiIndex.from_frame(laps[RACE])).to_numpy()
    return pd.concat([forecast(make().fit(laps[g != f]), laps[g == f], cfg) for f in range(folds)])


def ahead_band(p):
    return pd.cut(p['TyreLife'] - p['AnchorAge'], AHEAD_BINS)


class Conformal:
    """Wraps a model factory: the point forecast of the model fitted on all training laps, plus the quantiles
    of its out of fold errors (lap minus forecast) in the same band of laps after the anchor."""

    def __init__(self, make, cfg, levels, folds=5):
        self.make, self.cfg, self.levels, self.folds = make, cfg, list(levels), folds
        self.name = make().name + ', conformal'

    def fit(self, laps):
        self.model = self.make().fit(laps)
        o = out_of_fold(self.make, laps, self.cfg, self.folds)
        resid = o['LapTimeFC_s'] - o['Pred_s']
        self.q = resid.groupby(ahead_band(o), observed=True).quantile(self.levels).unstack()
        return self

    def forecast(self, laps, cfg):
        p = forecast(self.model, laps, cfg)
        q = self.q.reindex(ahead_band(p)).to_numpy()
        for col, v in zip(quantile_columns(self.levels), q.T):
            p[col] = p['Pred_s'].to_numpy() + v
        return p


def interval_scores(p, levels):
    """Coverage and mean width of each central interval, and the quantile score, for forecasts with quantile
    columns (NaN without)."""
    cols, y, out = quantile_columns(levels), p['LapTimeFC_s'], {}
    if not set(cols) <= set(p.columns):
        return pd.Series(dtype=float)
    for lo, hi in zip(levels, levels[::-1]):
        if lo >= hi:
            break
        a, b = p[f'Q{round(100 * lo)}'], p[f'Q{round(100 * hi)}']
        out[f'Cover{round(100 * (hi - lo))}'] = ((y >= a) & (y <= b)).mean()
        out[f'Width{round(100 * (hi - lo))}_s'] = (b - a).mean()
    pin = [np.maximum(q * (y - p[c]), (q - 1) * (y - p[c])).mean() for q, c in zip(levels, cols)]
    out['QuantileScore'] = float(np.mean(pin))
    return pd.Series(out)
