"""The free air lap time of a car, the one formula the engine and the overtaking calibration share:

    lap time = base + fuel(lap) + compound gap + wear(age) + evolution x E(lap, age) + race start(lap) [+ noise]

E is the race fraction the track evolution term reads (simulator.compound_pace.evolution): 'lap' (v1) uses
lap / race laps, so evolution also runs within a stint; 'stint_start' uses the stint's start lap,
(lap - age) / race laps, floored at 0, because the wear slope is fitted within stints and already carries the
evolution and fuel error of the laps a stint runs (v1 counted that part twice, making long stints too fast;
notes/sim_v2_search.md). The compound gaps are then fitted with the race fraction at the stint start too.

base is the car's level on a fresh Medium at race fraction 0 with empty tanks (pace.py). Wear is the
simulator's wear model (simulator.wear_model, monotone if simulator.monotone_wear); the compound gaps and track
evolution come from degradation.compound.CompoundPace; the race start (laps 2 to race_start_laps) and dirty air
offsets are the wear model's lap level offsets; lap noise is degradation.noise.WearNoise. All fitted on the
training seasons only.

Uncertainty in the compound inputs (simulator.uncertainty.bootstrap > 0): the wear curves, compound gaps and
track evolution are refitted on training races resampled with replacement, one wear grid and gap table per
refit. The engine gives each Monte Carlo draw one refit (rep), so a draw's cars all race on the same version of
the tyre model and strategies compare on common draws.

Race spread of the gaps (simulator.compound_pace.race_spread): a race's own Soft and Hard gaps depart from the
circuit's fitted gaps by about 0.2 to 0.3 s per lap beyond sampling error (compound.race_gap_spread), far more
than the refits move them. With it on, each rep also gets one shift per compound drawn from that spread, shared
by every car in the draw (it is the race's tyre behaviour, not one car's). Without refits, the reps are the main
fit plus the shifts (race_spread_reps of them).

Compound order (simulator.compound_pace.compound_order, off by default): at each circuit and tyre age the wear of
a softer compound is raised to at least the next harder one's (monotone.order_compounds).

New set terms (simulator.new_set, off by default; notes/sim_v3_loose_ends.md): on a new set fitted at a stop, an
out lap cost per compound (out_lap: the within race compound effect on the out lap against the model, centred on
the training stops' compound mix because the pit loss already holds the average out lap) and an offset per tyre
age for the first early_laps flying laps (ages 2 on, pooled over compounds: the lap against the same stint's laps
at ages 4 to 10). Both measured on the training seasons' green stops (new_set_rows).
"""
import numpy as np
import pandas as pd

from boxbox.degradation import compare, model_laps
from boxbox.degradation.compound import CompoundPace, race_gap_spread, stint_levels
from boxbox.degradation.monotone import MonotoneWear, order_compounds
from boxbox.degradation.noise import WearNoise
from boxbox.degradation.task import COMPOUNDS, KEY, STINT

DRIVER_RACE = ['Year', 'Round', 'Driver']


class LapModel:
    def __init__(self, cfg):
        s = cfg['simulator']
        self.max_age = s['max_age']
        self.fuel_start, self.s_per_kg = cfg['fuel']['start_kg'], cfg['fuel']['s_per_kg']
        self.min_level_laps = s['compound_pace']['min_laps']
        self.evolution_at = s['compound_pace'].get('evolution', 'lap')
        self._cfg = cfg

    def _fit_tyres(self, train_laps, fit_laps, circuits):
        """Wear model, compound pace and the wear grid (circuit -> compound x age) from one set of laps."""
        cfg, s = self._cfg, self._cfg['simulator']
        base = compare.ladder(cfg)[s['wear_model']]()
        wear = MonotoneWear(base, self.max_age) if s.get('monotone_wear', True) else base
        wear.fit(train_laps)
        c = s['compound_pace']
        frac = {'frac': 'start'} if c.get('evolution', 'lap') == 'stint_start' else {}   # v1 keeps the middle lap
        pace = CompoundPace(c['shrink'], c['min_laps'], **frac).fit(wear, fit_laps)
        g = pd.DataFrame([(ci, k) for ci in sorted(set(circuits)) for k in COMPOUNDS], columns=KEY)
        g = g.merge(pd.DataFrame({'TyreLife': np.arange(1, self.max_age + 1, dtype=float)}), how='cross')
        g = g.assign(StintLaps=20, RaceLap=np.nan, GapAhead_s=np.nan, Code=g['Compound'].map({k: i for i, k in enumerate(COMPOUNDS)}))
        g = g.sort_values(['Circuit', 'Code', 'TyreLife'])   # compounds in COMPOUNDS order, the engine's index
        g['Wear_s'] = wear.wear(g).to_numpy()
        fallback = g.groupby(['Compound', 'TyreLife'])['Wear_s'].mean()
        new = ~g['Circuit'].isin(set(fit_laps['Circuit']))
        g.loc[new, 'Wear_s'] = fallback.reindex(pd.MultiIndex.from_frame(g.loc[new, ['Compound', 'TyreLife']])).to_numpy()
        w = g['Wear_s'].to_numpy().reshape(len(set(circuits)), len(COMPOUNDS), self.max_age)
        if c.get('compound_order', False):
            w = order_compounds(w)
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
        c = s['compound_pace']
        self.gap_spread = race_gap_spread(self.pace, self.wear_model, fit_laps, self.min_level_laps)[0] if c.get('race_spread') else None
        rng = np.random.default_rng(seed)
        reps = []
        if bootstrap:
            races = fit_laps[['Year', 'Round']].drop_duplicates().to_numpy()
            for _ in range(bootstrap):
                pick = races[rng.integers(0, len(races), len(races))]
                parts = []
                for d in (train_laps, fit_laps):
                    k = pd.DataFrame(pick, columns=['Year', 'Round']).assign(Copy=np.arange(len(pick)))
                    r = d.merge(k, on=['Year', 'Round'])
                    parts.append(r.assign(Round=r['Round'] * 1000 + r['Copy']).drop(columns='Copy'))   # copies are distinct races
                _, _, pace, grid, gaps = self._fit_tyres(parts[0], parts[1], circuits)
                reps.append((grid, gaps, float(pace.coef['Frac'])))
        elif self.gap_spread is not None:
            reps = [(self.wear_grid, self.gaps, self.evolution)] * c.get('race_spread_reps', 30)
        if self.gap_spread is not None:
            sd = np.array([self.gap_spread.get(k, 0.0) for k in COMPOUNDS])   # Medium is the reference: no shift
            shift = rng.standard_normal((len(reps), len(COMPOUNDS))) * sd
            reps = [(grid, {ci: g + shift[i] for ci, g in gaps.items()}, evo) for i, (grid, gaps, evo) in enumerate(reps)]
        if reps:
            self.boot = {'wear': {ci: np.stack([r[0][ci] for r in reps]) for ci in self.wear_grid},
                         'gap': {ci: np.stack([r[1][ci] for r in reps]) for ci in self.wear_grid},
                         'evolution': np.array([r[2] for r in reps])}
        b = base.offsets
        self.race_start = {int(k[1]): float(v) for k, v in b.items() if k[0] == 'RaceStart'}
        self.dirty_edges = list(getattr(base, 'bins', []))
        self.dirty = np.array([float(b.get(('DirtyAir', e), 0.0)) for e in self.dirty_edges[:-1]])
        self.circuits = sorted(set(circuits))
        self.out_cost, self.early = None, None
        ns = s.get('new_set', {}) or {}
        if ns.get('out_lap') or ns.get('early_laps', 0):
            from boxbox import features
            stops = features.pit_loss(laps[laps['Year'].isin(years)], features.wet_races(laps), cfg)
            self.out_cost, self.early = new_set_terms(new_set_rows(self, laps, stops, years), ns.get('out_lap', False),
                                                      int(ns.get('early_laps', 0)))
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

    def race_fraction(self, lap, age, race_laps):
        """The race fraction the evolution term reads (module docstring): lap / race laps (v1, 'lap') or the
        stint's start, (lap - age) / race laps floored at 0 ('stint_start'; exact for a new set, while a used set
        counts its old laps as part of the stint)."""
        lap = np.asarray(lap, dtype=float)
        if getattr(self, 'evolution_at', 'lap') == 'stint_start':
            return np.clip(lap - np.asarray(age, dtype=float), 0, None) / race_laps
        return lap / race_laps

    def free_air(self, base, circuit, compound, age, lap, race_laps, rep=None):
        """Free air lap time without noise, s. rep: per draw refit index (see wear)."""
        evo = self.evolution if rep is None else self.boot['evolution'][np.asarray(rep)][:, None]
        return (base + self.fuel(lap, race_laps) + self.gap(circuit, compound, rep) + self.wear(circuit, compound, age, rep)
                + evo * self.race_fraction(lap, age, race_laps) + self.start_offset(lap))

    def new_set(self, compound, age, out_lap, fresh):
        """New set terms, s (module docstring; 0 when simulator.new_set is off): the out lap cost of the compound on
        an out lap, and the early lap offset at ages 2 on of a new set fitted at a stop (fresh)."""
        extra = np.zeros(np.broadcast(np.asarray(compound), np.asarray(age, float)).shape)
        oc, ea = getattr(self, 'out_cost', None), getattr(self, 'early', None)
        if oc is not None:
            extra = extra + np.where(out_lap, oc[np.asarray(compound)], 0.0)
        if ea is not None and len(ea):
            a = np.rint(np.asarray(age, float)).astype(int)
            on = np.asarray(fresh, bool) & (a >= 2) & (a < 2 + len(ea))
            extra = extra + np.where(on, ea[np.clip(a - 2, 0, len(ea) - 1)], 0.0)
        return extra

    @property
    def has_new_set(self):
        return getattr(self, 'out_cost', None) is not None or getattr(self, 'early', None) is not None

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


def _residuals(lm, d):
    """Clean laps less the model's free air lap (base 0) and dirty air, s."""
    from boxbox.simulator.strategy import CODE
    comp = d['Compound'].map(CODE).to_numpy()
    pred = np.empty(len(d))
    for ci, idx in d.groupby('Circuit').indices.items():
        x = d.iloc[idx]
        pred[idx] = lm.free_air(0.0, ci, comp[idx], x['TyreLife'].to_numpy(), x['LapNumber'].to_numpy(), x['RaceLaps'].to_numpy())
    return d['LapTime_s'].to_numpy() - pred - lm.dirty_air(d['GapAhead_s'].to_numpy())


def new_set_rows(lm, laps, stops, years, max_dev_s=10.0):
    """One row per green stop onto a new set (stops: features.pit_loss rows): X, the out lap less the model's free
    air lap at age 1 and the new stint's level (its clean laps at ages 2 to 8, 3 or more), and W2, W3, its clean
    laps at ages 2 and 3 less its clean laps at ages 4 to 10, all against the model (s). Out laps more than
    max_dev_s from the race median are dropped (traffic, a slow release)."""
    from boxbox.simulator.strategy import CODE
    stint = ['Year', 'Round', 'Driver', 'Stint']
    l = laps[laps['Year'].isin(years)]
    l = l.assign(RaceLaps=l.groupby(['Year', 'Round'])['LapNumber'].transform('max'))
    o = l[l['PitOut']][['Year', 'Round', 'Driver', 'LapNumber', 'Stint', 'Compound', 'TyreLife', 'Circuit', 'RaceLaps']]
    o = stops[stops['Year'].isin(years)].assign(LapNumber=lambda x: x['InLap'] + 1).merge(o, on=['Year', 'Round', 'Driver', 'LapNumber'])
    o = o[o['Compound'].isin(COMPOUNDS) & (o['TyreLife'] <= 1.5)]
    d = model_laps(laps, years, {'degradation': {'min_fit_laps': 1}})
    d = d[d['Compound'].isin(COMPOUNDS)]
    d = d.assign(LapTime_s=laps.loc[d.index, 'LapTime_s'].to_numpy())
    d = d.assign(R=_residuals(lm, d))
    lvl = d[d['TyreLife'].between(2, 8)].groupby(stint)['R'].agg(Level='mean', NL='size')
    later = d[d['TyreLife'].between(4, 10)].groupby(stint)['R'].mean().rename('Later')
    o = o.join(lvl, on=stint).join(later, on=stint)
    for a in (2, 3):
        o = o.join(d[d['TyreLife'] == a].groupby(stint)['R'].mean().rename(f'R{a}'), on=stint)
        o[f'W{a}'] = o[f'R{a}'] - o['Later']
    fa = np.empty(len(o))
    for ci, idx in o.groupby('Circuit').indices.items():
        x = o.iloc[idx]
        fa[idx] = lm.free_air(0.0, ci, x['Compound'].map(CODE).to_numpy(), np.ones(len(idx)), x['LapNumber'].to_numpy(), x['RaceLaps'].to_numpy())
    o['X'] = o['OutLap_s'].to_numpy() - fa - o['Level'].to_numpy()
    o = o[o['NL'] >= 3].dropna(subset=['X'])
    o = o[(o['X'] - o.groupby(['Year', 'Round'])['X'].transform('median')).abs() < max_dev_s]
    return o.reset_index(drop=True)


def compound_effects(rows, col='X'):
    """Within race compound effects on col (Medium 0): OLS on race demeaned Soft and Hard indicators."""
    race = [rows['Year'], rows['Round']]
    y = rows[col] - rows.groupby(race)[col].transform('mean')
    X = pd.DataFrame({c: (rows['Compound'] == c).astype(float) for c in ('SOFT', 'HARD')})
    X = X - X.groupby(race).transform('mean')
    b = np.linalg.lstsq(X.to_numpy(), y.to_numpy(), rcond=None)[0]
    return {'SOFT': float(b[0]), 'MEDIUM': 0.0, 'HARD': float(b[1])}


def new_set_terms(rows, out_lap=True, early_laps=0):
    """(out lap cost per compound in COMPOUNDS order, centred on the rows' compound mix, or None; early lap offsets
    for ages 2 .. 1 + early_laps, pooled over compounds, or None). Ages past 3 are not measured: capped at 2 laps."""
    oc = None
    if out_lap:
        e = compound_effects(rows)
        mix = rows['Compound'].value_counts(normalize=True)
        centre = sum(e[c] * mix.get(c, 0.0) for c in COMPOUNDS)
        oc = np.array([e[c] - centre for c in COMPOUNDS])
    ea = None
    if early_laps:
        ea = np.array([float(rows[f'W{a}'].mean()) for a in range(2, 2 + min(int(early_laps), 2))])
    return oc, ea
