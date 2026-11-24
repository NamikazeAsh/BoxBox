"""Strategy search inputs (notes/sim_v2_search.md): evolution at the stint start, the race spread of the compound
gaps, compound order on the wear grid, usage aware stint caps and the vectorised free air pass."""
import numpy as np
import pandas as pd
import pytest

from boxbox import degradation
from boxbox.degradation.compound import CompoundPace, race_gap_spread
from boxbox.degradation.monotone import order_compounds
from boxbox.simulator import race as rc, strategy as st
from boxbox.simulator.laptime import LapModel

GAP = {'SOFT': -0.5, 'MEDIUM': 0.0, 'HARD': 0.3}
CFG = {'simulator': {'max_age': 60, 'race': {'stop_lap_step': 2, 'min_stint_laps': 5, 'max_stops': 2,
                                             'stint_cap': {'quantile': 0.95, 'shrink': 5}}, 'search': {}}}


def stint(rnd, driver, n, compound, start, laps, level, slope=0.05, evo=0.0, race_laps=60, circuit='Monza', noise=0.0, rng=None):
    """Clean laps of one stint: level + gap + slope x age + evo x (stint start - 1) / race laps, as the stint
    start model has it (the within stint evolution is part of the slope)."""
    age = np.arange(1, laps + 1, dtype=float)
    e = 0.0 if rng is None else noise * rng.standard_normal(laps)
    return pd.DataFrame({'Year': 2024, 'Round': rnd, 'Driver': driver, 'Stint': n, 'Circuit': circuit, 'Compound': compound,
                         'LapNumber': start + age - 1, 'TyreLife': age, 'StintLaps': laps, 'RaceLaps': race_laps,
                         'LapTimeFC_s': level + GAP[compound] + slope * age + evo * (start - 1) / race_laps + e})


def lap_model(evolution_at):
    lm = LapModel.__new__(LapModel)
    lm.evolution_at, lm.evolution, lm.max_age, lm.fuel_start, lm.s_per_kg = evolution_at, -0.6, 60, 0.0, 0.0
    lm.wear_grid = {'Monza': np.tile(0.05 * np.arange(1, 61), (3, 1))}
    lm.gaps = {'Monza': np.array([GAP['SOFT'], 0.0, GAP['HARD']])}
    lm.race_start, lm.boot = {}, None
    return lm


def test_evolution_at_stint_start_runs_once_per_stint():
    lap, age = np.arange(21, 41, dtype=float), np.arange(1, 21, dtype=float)   # a stint from lap 21
    per_lap = lap_model('lap').free_air(0.0, 'Monza', np.ones(20, int), age, lap, 60)
    start = lap_model('stint_start').free_air(0.0, 'Monza', np.ones(20, int), age, lap, 60)
    assert np.allclose(np.diff(start), 0.05)                                  # within the stint only the wear slope
    assert np.allclose(np.diff(per_lap), 0.05 - 0.6 / 60)                     # v1 adds evolution on top of it
    assert start[0] == pytest.approx(0.05 - 0.6 * 20 / 60)                    # evolution at the stint's start lap


def test_compound_pace_at_stint_start_recovers_evolution_and_gaps():
    """Stints of different lengths: the level moves with the start lap, so the middle lap mixes in stint length."""
    plans = [(('MEDIUM', 10), ('HARD', 30)), (('SOFT', 25), ('MEDIUM', 15)), (('HARD', 12), ('SOFT', 28)), (('MEDIUM', 30), ('SOFT', 10))]
    parts = []
    for r in range(8):
        for i, plan in enumerate(plans):
            start = 1
            for n, (comp, ln) in enumerate(plan):
                parts.append(stint(r, f'D{i}', n + 1, comp, start, ln, 80.0 + i, evo=-0.6))
                start += ln
    laps = pd.concat(parts, ignore_index=True)
    wear = degradation.LinearDegradation(0).fit(laps)
    m = CompoundPace(0, frac='start').fit(wear, laps)
    assert m.coef['Frac'] == pytest.approx(-0.6, abs=1e-6)
    assert m.gaps['SOFT'] == pytest.approx(-0.5, abs=1e-6) and m.gaps['HARD'] == pytest.approx(0.3, abs=1e-6)
    mid = CompoundPace(0).fit(wear, laps)
    assert abs(mid.gaps['HARD'] - 0.3) > 1e-3 or abs(mid.coef['Frac'] + 0.6) > 1e-3     # the v1 definition is off here


def test_race_gap_spread_finds_a_shared_race_shift():
    rng = np.random.default_rng(1)
    parts = []
    for r in range(60):
        shift = rng.normal(0, 0.3)          # this race's Soft runs 0.3 s per lap away from the circuit gap (SD)
        for i in range(8):
            first, second = ('SOFT', 'MEDIUM') if i % 2 else ('MEDIUM', 'SOFT')
            for n, (comp, start) in enumerate([(first, 1), (second, 21)]):
                s = stint(r, f'D{i}', n + 1, comp, start, 20, 80.0 + i, noise=0.3, rng=rng)
                s['LapTimeFC_s'] += shift * (comp == 'SOFT')
                parts.append(s)
    laps = pd.concat(parts, ignore_index=True)
    wear = degradation.LinearDegradation(0).fit(laps)
    pace = CompoundPace(20).fit(wear, laps)
    spread, table = race_gap_spread(pace, wear, laps)
    assert spread['SOFT'] == pytest.approx(0.3, rel=0.3)
    assert spread['HARD'] == 0.0 and len(table) == 60


def test_order_compounds_never_lets_a_softer_tyre_wear_less():
    w = np.array([[[1.0, 2.0], [3.0, 1.0], [2.0, 2.5]]])        # one circuit: Soft, Medium, Hard rows, two ages
    o = order_compounds(w)
    assert np.allclose(o[0], [[3.0, 2.5], [3.0, 2.5], [2.0, 2.5]])
    assert np.allclose(w[0, 0], [1.0, 2.0])                     # input untouched


def stints_table():
    """Austin: Mediums and Hards only but for one short Soft; Monza: all three used. Plus the events table."""
    rows = []
    for y in [2022, 2023]:
        for d in range(10):
            rows += [(y, 1, f'D{d}', 1, 'MEDIUM', 20), (y, 1, f'D{d}', 2, 'HARD', 36)]
            rows += [(y, 2, f'D{d}', 1, 'SOFT', 18), (y, 2, f'D{d}', 2, 'MEDIUM', 22), (y, 2, f'D{d}', 3, 'HARD', 13)]
    rows.append((2023, 1, 'D0', 3, 'SOFT', 4))
    s = pd.DataFrame(rows, columns=['Year', 'Round', 'Driver', 'Stint', 'Compound', 'Laps']).assign(TyreLifeStart=1.0)
    ev = pd.DataFrame({'Year': [2022, 2022, 2023, 2023], 'Round': [1, 2, 1, 2], 'Circuit': ['Austin', 'Monza', 'Austin', 'Monza']})
    return s, ev


def test_usage_caps_bound_a_compound_teams_avoid_and_keep_the_rest():
    s, ev = stints_table()
    v1 = st.stint_caps(s, ev, set(), [2022, 2023], CFG)
    cfg = {'simulator': {**CFG['simulator'], 'search': {'cap_prior': 'usage', 'cap_usage_full': 0.5}}}
    use = st.stint_caps(s, ev, set(), [2022, 2023], cfg)
    a1, a2 = st.caps_for(v1, 'Austin'), st.caps_for(use, 'Austin')
    assert a1['SOFT'] > 15 and a2['SOFT'] < 5                       # 1 short Soft stint: v1 borrows the all circuit cap
    assert a2['HARD'] == pytest.approx(a1['HARD']) and a2['MEDIUM'] == pytest.approx(a1['MEDIUM'])
    assert st.caps_for(use, 'Monza') == pytest.approx(st.caps_for(v1, 'Monza'))
    assert st.caps_for(use, 'Baku') == pytest.approx(st.caps_for(v1, 'Baku'))   # no history: all circuit caps
    soft = st.legal_strategies(50, cfg, ['MEDIUM'], caps=a2)
    assert soft and not any('SOFT' in c for c, _ in soft)


def test_start_tyre_keeps_a_short_opening_stint_under_a_tiny_cap():
    caps = {'SOFT': 2.0, 'MEDIUM': 40.0, 'HARD': 45.0}
    out = st.legal_strategies(50, CFG, ['SOFT'], caps=caps)
    assert out and all(c[0] == 'SOFT' and s[0] <= 5 for c, s in out)
    assert all(st.within_caps(c, s, 50, caps, CFG) for c, s in out)
    assert not st.within_caps(('SOFT', 'MEDIUM'), (9,), 50, caps, CFG)


class StubPit:
    def predict(self, circuits):
        return np.full(len(circuits), 22.0)


def test_strategy_grid_times_match_the_search_coarse_pass():
    inputs = type('I', (), {'lap': lap_model('stint_start'), 'pit': StubPit()})()
    strategies = st.legal_strategies(30, CFG, ['MEDIUM'], start_age=3.0)[:40]
    g = st.strategy_grid(strategies, 30, start_age=3.0)
    fast = inputs.lap.free_air(80.0, 'Monza', g['Comp'], g['Age'], np.arange(1, 31, dtype=float)[None, :], 30).sum(1) + 22.0 * g['Stops']
    slow = rc.free_air_race_time(inputs, 'Monza', 30, 80.0, strategies, start_age=3.0)
    assert np.allclose(fast, slow)
