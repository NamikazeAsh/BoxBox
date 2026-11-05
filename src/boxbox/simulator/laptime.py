"""The free air lap time of a car, the one formula the engine and the overtaking calibration share:

    lap time = base + fuel(lap) + compound gap + wear(age) + evolution x lap / race laps + race start(lap) [+ noise]

base is the car's level on a fresh Medium at race fraction 0 with empty tanks (pace.py). Wear is the
simulator's wear model (simulator.wear_model, monotone if simulator.monotone_wear); the compound gaps and track
evolution come from degradation.compound.CompoundPace; the race start (laps 2 to race_start_laps) and dirty air
offsets are the wear model's lap level offsets; lap noise is degradation.noise.WearNoise. All fitted on the
training seasons only.

Uncertainty in the compound inputs (simulator.uncertainty.bootstrap > 0): the wear curves, compound gaps and
track evolution are refitted on training races resampled with replacement, one wear grid and gap table per
refit. The engine gives each Monte Carlo draw one refit (rep), so a draw's cars all race on the same version of
the tyre model and strategies compare on common draws.
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

    def _fit_tyres(self, train_laps, fit_laps, circuits):
        """Wear model, compound pace and the wear grid (circuit -> compound x age) from one set of laps."""
        cfg, s = self._cfg, self._cfg['simulator']
        base = compare.ladder(cfg)[s['wear_model']]()
        wear = MonotoneWear(base, self.max_age) if s.get('monotone_wear', True) else base
        wear.fit(train_laps)
        c = s['compound_pace']
        pace = CompoundPace(c['shrink'], c['min_laps']).fit(wear, fit_laps)
        g = pd.DataFrame([(ci, k) for ci in sorted(set(circuits)) for k in COMPOUNDS], columns=KEY)
        g = g.merge(pd.DataFrame({'TyreLife': np.arange(1, self.max_age + 1, dtype=float)}), how='cross')
        g = g.assign(StintLaps=20, RaceLap=np.nan, GapAhead_s=np.nan, Code=g['Compound'].map({k: i for i, k in enumerate(COMPOUNDS)}))
        g = g.sort_values(['Circuit', 'Code', 'TyreLife'])   # compounds in COMPOUNDS order, the engine's index
        g['Wear_s'] = wear.wear(g).to_numpy()
        fallback = g.groupby(['Compound', 'TyreLife'])['Wear_s'].mean()
        new = ~g['Circuit'].isin(set(fit_laps['Circuit']))
        g.loc[new, 'Wear_s'] = fallback.reindex(pd.MultiIndex.from_frame(g.loc[new, ['Compound', 'TyreLife']])).to_numpy()
        w = g['Wear_s'].to_numpy().reshape(len(set(circuits)), len(COMPOUNDS), self.max_age)
        grid = dict(zip(sorted(set(circuits)), w))
        gaps = {ci: pace.gap(pd.Series([ci] * len(COMPOUNDS)), pd.Series(COMPOUNDS)).to_numpy() for ci in sorted(set(circuits))}
        return base, wear, pace, grid, gaps

    def fit(self, laps, years, circuits, bootstrap=0, seed=0):
        """laps: the clean laps table; years: training seasons; circuits: every circuit the model will be asked
        about (one with no training laps uses the compound's mean curve and the all circuit gaps). bootstrap:
        number of refits on resampled training races (module docstring)."""
        cfg, s = self._cfg, self._cfg['simulator']
        train_laps, fit_laps = model_laps(laps, years, cfg, train=True), model_laps(laps, years, cfg)
        base, self.wear_model, self.pace, self.wear_grid, self.gaps = self._fit_tyres(train_laps, fit_laps, circuits)
        self.noise = WearNoise(s['noise']['min_laps']).fit(self.wear_model, fit_laps)
        self.evolution = float(self.pace.coef['Frac'])
        self.boot = None
        if bootstrap:
            rng = np.random.default_rng(seed)
            races = fit_laps[['Year', 'Round']].drop_duplicates().to_numpy()
            reps = []
            for _ in range(bootstrap):
                pick = races[rng.integers(0, len(races), len(races))]
                parts = []
                for d in (train_laps, fit_laps):
                    k = pd.DataFrame(pick, columns=['Year', 'Round']).assign(Copy=np.arange(len(pick)))
                    r = d.merge(k, on=['Year', 'Round'])
                    parts.append(r.assign(Round=r['Round'] * 1000 + r['Copy']).drop(columns='Copy'))   # copies are distinct races
                _, _, pace, grid, gaps = self._fit_tyres(parts[0], parts[1], circuits)
                reps.append((grid, gaps, float(pace.coef['Frac'])))
            self.boot = {'wear': {ci: np.stack([r[0][ci] for r in reps]) for ci in self.wear_grid},
                         'gap': {ci: np.stack([r[1][ci] for r in reps]) for ci in self.wear_grid},
                         'evolution': np.array([r[2] for r in reps])}
        b = base.offsets
        self.race_start = {int(k[1]): float(v) for k, v in b.items() if k[0] == 'RaceStart'}
        self.dirty_edges = list(getattr(base, 'bins', []))
        self.dirty = np.array([float(b.get(('DirtyAir', e), 0.0)) for e in self.dirty_edges[:-1]])
        self.circuits = sorted(set(circuits))
        self._cfg = None   # keep the fitted model free of paths
        return self

    def release(self):
        """Drop the wear model (the engine only needs the wear grid; the model is not picklable). stint_levels and
        overtaking.free_air_laps need it, so call this only once fitting is done."""
        self.wear_model = None
        return self

    # Vectorised pieces. compound arrays are indices into COMPOUNDS.

    @property
    def n_boot(self):
        return 0 if getattr(self, 'boot', None) is None else len(self.boot['evolution'])

    def wear(self, circuit, compound, age, rep=None):
        """Wear, s, at tyre age (laps on the tyre, 1 on its first lap), clipped to 1..max_age. rep: per draw
        refit index (draws,) for draws x cars arrays, None for the main fit."""
        a = np.clip(np.rint(np.asarray(age, dtype=float)), 1, self.max_age).astype(int) - 1
        if rep is None:
            return self.wear_grid[circuit][np.asarray(compound), a]
        return self.boot['wear'][circuit][np.asarray(rep)[:, None], np.asarray(compound), a]

    def gap(self, circuit, compound, rep=None):
        """Fresh tyre gap to a fresh Medium, s per lap, per compound index."""
        if rep is None:
            return self.gaps[circuit][np.asarray(compound)] if circuit in self.gaps else \
                self.pace.gap(pd.Series([circuit] * len(COMPOUNDS)), pd.Series(COMPOUNDS)).to_numpy()[np.asarray(compound)]
        return self.boot['gap'][circuit][np.asarray(rep)[:, None], np.asarray(compound)]

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

    def free_air(self, base, circuit, compound, age, lap, race_laps, rep=None):
        """Free air lap time without noise, s. rep: per draw refit index (see wear)."""
        evo = self.evolution if rep is None else self.boot['evolution'][np.asarray(rep)][:, None]
        return (base + self.fuel(lap, race_laps) + self.gap(circuit, compound, rep) + self.wear(circuit, compound, age, rep)
                + evo * np.asarray(lap, dtype=float) / race_laps + self.start_offset(lap))

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
