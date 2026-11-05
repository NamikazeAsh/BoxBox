"""The free air lap time of a car, the one formula the engine and the overtaking calibration share:

    lap time = base + fuel(lap) + compound gap + wear(age) + evolution x lap / race laps + race start(lap) [+ noise]

base is the car's level on a fresh Medium at race fraction 0 with empty tanks (pace.py). Wear is the
simulator's wear model (simulator.wear_model, monotone if simulator.monotone_wear); the compound gaps and track
evolution come from degradation.compound.CompoundPace; the race start (laps 2 to race_start_laps) and dirty air
offsets are the wear model's lap level offsets; lap noise is degradation.noise.WearNoise. All fitted on the
training seasons only.
"""
import numpy as np
import pandas as pd

from boxbox.degradation import compare, model_laps
from boxbox.degradation.compound import CompoundPace, stint_levels
from boxbox.degradation.monotone import MonotoneWear
from boxbox.degradation.noise import WearNoise
from boxbox.degradation.task import COMPOUNDS, KEY, STINT

DRIVER_RACE = ['Year', 'Round', 'Driver']


class LapModel:
    def __init__(self, cfg):
        s = cfg['simulator']
        self.max_age = s['max_age']
        self.fuel_start, self.s_per_kg = cfg['fuel']['start_kg'], cfg['fuel']['s_per_kg']
        self.min_level_laps = s['compound_pace']['min_laps']
        self._cfg = cfg

    def fit(self, laps, years, circuits):
        """laps: the clean laps table; years: training seasons; circuits: every circuit the model will be asked
        about (one with no training laps uses the compound's mean curve and the all circuit gaps)."""
        cfg, s = self._cfg, self._cfg['simulator']
        base = compare.ladder(cfg)[s['wear_model']]()
        self.wear_model = MonotoneWear(base, self.max_age) if s.get('monotone_wear', True) else base
        self.wear_model.fit(model_laps(laps, years, cfg, train=True))
        fit_laps = model_laps(laps, years, cfg)
        c = s['compound_pace']
        self.pace = CompoundPace(c['shrink'], c['min_laps']).fit(self.wear_model, fit_laps)
        self.noise = WearNoise(s['noise']['min_laps']).fit(self.wear_model, fit_laps)
        self.evolution = float(self.pace.coef['Frac'])
        b = base.offsets
        self.race_start = {int(k[1]): float(v) for k, v in b.items() if k[0] == 'RaceStart'}
        self.dirty_edges = list(getattr(base, 'bins', []))
        self.dirty = np.array([float(b.get(('DirtyAir', e), 0.0)) for e in self.dirty_edges[:-1]])
        # wear grid: circuit x compound x age 1..max_age, loss against age 1 (the level is at age 0 to 1)
        g = pd.DataFrame([(ci, k) for ci in sorted(set(circuits)) for k in COMPOUNDS], columns=KEY)
        g = g.merge(pd.DataFrame({'TyreLife': np.arange(1, self.max_age + 1, dtype=float)}), how='cross')
        g = g.assign(StintLaps=20, RaceLap=np.nan, GapAhead_s=np.nan)
        g['Wear_s'] = self.wear_model.wear(g).to_numpy()
        fallback = g.groupby(['Compound', 'TyreLife'])['Wear_s'].mean()
        seen = set(fit_laps['Circuit'])
        new = ~g['Circuit'].isin(seen)
        g.loc[new, 'Wear_s'] = fallback.reindex(pd.MultiIndex.from_frame(g.loc[new, ['Compound', 'TyreLife']])).to_numpy()
        self.circuits = sorted(set(circuits))
        self.wear_grid = {ci: np.stack([g[(g['Circuit'] == ci) & (g['Compound'] == k)].sort_values('TyreLife')['Wear_s'].to_numpy()
                                        for k in COMPOUNDS]) for ci in self.circuits}
        self._cfg = None   # keep the fitted model free of paths
        return self

    def release(self):
        """Drop the wear model (the engine only needs the wear grid; the model is not picklable). stint_levels and
        overtaking.free_air_laps need it, so call this only once fitting is done."""
        self.wear_model = None
        return self

    # Vectorised pieces. compound arrays are indices into COMPOUNDS.

    def wear(self, circuit, compound, age):
        """Wear, s, at tyre age (laps on the tyre, 1 on its first lap), clipped to 1..max_age."""
        a = np.clip(np.rint(np.asarray(age, dtype=float)), 1, self.max_age).astype(int) - 1
        return self.wear_grid[circuit][np.asarray(compound), a]

    def gap(self, circuit, compound):
        """Fresh tyre gap to a fresh Medium, s per lap, per compound index."""
        g = self.pace.gap(pd.Series([circuit] * len(COMPOUNDS)), pd.Series(COMPOUNDS)).to_numpy()
        return g[np.asarray(compound)]

    def fuel(self, lap, race_laps):
        """Fuel effect on lap (1 based), s: linear burn over race_laps, as features.add_fuel_correction."""
        kg = self.fuel_start * (1 - (np.asarray(lap, dtype=float) - 1) / race_laps)
        return self.s_per_kg * np.clip(kg, 0, None)

    def start_offset(self, lap):
        """Race start offset, s, for a lap number or an array of them."""
        lap = np.asarray(lap)
        if lap.ndim == 0:
            return self.race_start.get(int(lap), 0.0)
        return np.array([self.race_start.get(int(x), 0.0) for x in lap.ravel()]).reshape(lap.shape)

    def dirty_air(self, gap):
        """Seconds lost behind a car gap seconds ahead (0 beyond the last dirty air edge or with no car ahead).
        Below the first edge the first band's offset applies (the overtaking rule handles closer running)."""
        gap = np.asarray(gap, dtype=float)
        if not len(self.dirty):
            return np.zeros_like(gap)
        idx = np.searchsorted(self.dirty_edges, gap, side='right') - 1
        idx = np.clip(idx, 0, len(self.dirty) - 1)
        out = self.dirty[idx]
        return np.where(np.isnan(gap) | (gap >= self.dirty_edges[-1]), 0.0, out)

    def free_air(self, base, circuit, compound, age, lap, race_laps):
        """Free air lap time without noise, s."""
        return (base + self.fuel(lap, race_laps) + self.gap(circuit, compound) + self.wear(circuit, compound, age)
                + self.evolution * np.asarray(lap, dtype=float) / race_laps + self.start_offset(lap))

    # Levels in the data (for pace targets and the overtaking calibration)

    def stint_levels(self, laps, years):
        """Per stint with enough clean laps (simulator.compound_pace.min_laps): its level (s, empty tanks, wear
        and dirty air removed) and its base, the level less the compound gap and evolution (fresh Medium,
        race fraction 0)."""
        s = stint_levels(self.wear_model, model_laps(laps, years, {'degradation': {'min_fit_laps': 1}}), self.min_level_laps)
        return s.assign(Base=s['Level'] - self.pace.offset(s))

    def driver_bases(self, levels):
        """Each driver's race base: the clean lap weighted mean of their stints' bases."""
        w = levels.assign(W=levels['Base'] * levels['N'])
        g = w.groupby(DRIVER_RACE)
        return (g['W'].sum() / g['N'].sum()).rename('Base').reset_index()
