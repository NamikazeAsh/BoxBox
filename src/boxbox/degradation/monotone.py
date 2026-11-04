"""Wear that never falls with tyre age, for the strategy simulator. A fitted curve can bend down late in a stint
(the quadratic Medium bend, low or negative circuit slopes such as Lusail Medium), and a simulator that
optimises over stint length would exploit that. MonotoneWear wraps any wear model: the age part of its wear is
replaced by its running maximum over ages 1 to max_age per circuit and compound, and the lap level offsets
(race start, dirty air) are left as they are.
"""
import numpy as np
import pandas as pd

from boxbox.degradation.task import KEY


class MonotoneWear:
    def __init__(self, base, max_age=60):
        self.base, self.max_age = base, max_age
        self.name = base.name + ', monotone'

    def fit(self, laps):
        self.base.fit(laps)
        return self

    def _age_part(self, laps):
        """Base wear without the race start offsets (RaceLap removed, so no lap gets them)."""
        return self.base.wear(laps.assign(RaceLap=np.nan))

    def curve(self, cells):
        """Monotone age wear on a grid: one row per circuit, compound and age 1 to max_age."""
        ages = np.arange(1, self.max_age + 1, dtype=float)
        g = cells[KEY].drop_duplicates().merge(pd.DataFrame({'TyreLife': ages}), how='cross')
        g = g.assign(StintLaps=20, RaceLap=np.nan, GapAhead_s=np.nan)
        g['Wear_s'] = self._age_part(g).to_numpy()
        g['Wear_s'] = g.sort_values('TyreLife').groupby(KEY)['Wear_s'].cummax()
        return g[KEY + ['TyreLife', 'Wear_s']]

    def wear(self, laps):
        g = self.curve(laps).set_index(KEY + ['TyreLife'])['Wear_s']
        age = laps['TyreLife'].clip(1, self.max_age).round()
        idx = pd.MultiIndex.from_arrays([laps['Circuit'], laps['Compound'], age])
        mono = pd.Series(g.reindex(idx).to_numpy(), index=laps.index)
        return mono + (self.base.wear(laps) - self._age_part(laps))

    def dirty_air(self, laps):
        return self.base.dirty_air(laps)
