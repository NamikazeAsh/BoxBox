"""Evaluation helpers shared by every model. Splits are by whole season, never by lap (modelling D5)."""
import pandas as pd


def season_splits(seasons):
    """Rolling origin. Validation folds score every season except the first and last, each trained on the
    seasons before it; models are chosen on these. Test trains on all but the last season and scores the
    last, and is only run for the chosen model and baselines."""
    s = sorted(seasons)
    return {'validation': [(s[:i], s[i]) for i in range(1, len(s) - 1)], 'test': (s[:-1], s[-1])}


def error_table(df, by=None, err='Err_s'):
    """MAE, mean error (bias) and count of an error column, overall or per group."""
    g = (df.groupby(by, observed=True) if by else df.assign(_all='all').groupby('_all'))[err]
    return pd.DataFrame({'MAE_s': g.apply(lambda e: e.abs().mean()), 'Bias_s': g.mean(), 'N': g.size()})
