"""Pit loss per circuit for the strategy simulator, from the green flag stops of `features.pit_loss` (the one
definition of a stop's loss). A circuit's estimate is the median stop over its most recent `recent` seasons
with stops (all seasons when recent is None), so a changed pit lane (Marina Bay and Zandvoort in 2025,
data_issues.md) drops out after a season or two. A circuit with too few stops falls back to all its seasons,
then to the global median. The simulator draws a stop's loss from the same stops (`sample`), so its spread
matches the data.
"""
import numpy as np
import pandas as pd

RACE = ['Year', 'Round']


class PitLoss:
    def __init__(self, recent=None, min_stops=10):
        self.recent, self.min_stops = recent, min_stops
        self.name = f'circuit median, last {recent} seasons' if recent else 'circuit median, all seasons'

    def fit(self, stops):
        """stops: features.pit_loss rows with a Circuit column."""
        self.global_median = stops['PitLoss_s'].median()
        self.pools = {}
        for circ, g in stops.groupby('Circuit'):
            years = sorted(g['Year'].unique())[-self.recent:] if self.recent else sorted(g['Year'].unique())
            pool = g[g['Year'].isin(years)]['PitLoss_s']
            if len(pool) < self.min_stops:
                pool = g['PitLoss_s']
            self.pools[circ] = pool.to_numpy()
        self.median = pd.Series({c: np.median(p) for c, p in self.pools.items()}, name='PitLoss_s').rename_axis('Circuit')
        return self

    def predict(self, circuits):
        return pd.Series(circuits).map(self.median).fillna(self.global_median).to_numpy()

    def sample(self, circuit, rng, size):
        pool = self.pools.get(circuit)
        if pool is None:
            return np.full(size, self.global_median)
        return rng.choice(pool, size=size)


class GlobalPitLoss:
    """Baseline: one median for every circuit."""
    name = 'global median'

    def fit(self, stops):
        self.global_median = stops['PitLoss_s'].median()
        return self

    def predict(self, circuits):
        return np.full(len(circuits), self.global_median)


def race_medians(stops, min_stops=5):
    """The test target: each race's median stop, for races with at least min_stops green flag stops."""
    g = stops.groupby(RACE + ['Circuit'])['PitLoss_s']
    t = pd.DataFrame({'Stops': g.size(), 'Median_s': g.median()}).reset_index()
    return t[t['Stops'] >= min_stops]
