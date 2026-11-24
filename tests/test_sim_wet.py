import numpy as np
import pandas as pd
import pytest

from boxbox.config import load_config
from boxbox.simulator import race as rc, strategy as st, wet

CFG = {'simulator': {'race': {'tie_places': 0.25}, 'wet': {
    'wet_compounds': ['INTERMEDIATE', 'WET'], 'ref_compound': 'MEDIUM', 'wear_s_per_lap': 0.0, 'hardness': 0.0,
    'trends': [2.0], 'dry_level_s': 5.0, 'wet_level_s': -10.0, 'switch_offsets': {0: 1}, 'min_wet_laps': 2,
    'min_stint_laps': 5, 'search_laps': 2, 'refine_top': 5, 'replay': {'min_cars': 1, 'outlier_pct': 1.07, 'max_interp_laps': 3}}},
    'neutral': {'lap_share': 0.5}}


class NoisyLap:
    """Base pace, linear wear by compound, real random noise (so a changed random stream would show)."""
    class noise:
        @staticmethod
        def draw_slopes(rng, draws, races):
            return 0.01 * rng.standard_normal((draws, len(races)))

        @staticmethod
        def draw_laps(rng, shape):
            return 0.3 * rng.standard_normal(shape)

    def free_air(self, base, circuit, compound, age, lap, race_laps, rep=None):
        c = np.asarray(compound)
        return np.asarray(base, float) + 0.05 * (c + 1) * np.asarray(age, float) - 0.3 * (c == 0)

    def dirty_air(self, gap):
        return np.where(np.asarray(gap, float) < 1.0, 0.2, 0.0)


class Pit:
    def sample(self, circuit, rng, size):
        return 20.0 + rng.standard_normal(size)

    def predict(self, circuits):
        return np.full(len(circuits), 20.0)


class Rule:
    beta, hold = 0.5, True

    def threshold(self, circuit):
        return 0.4


class Inputs:
    def __init__(self):
        self.lap, self.pit, self.rule = NoisyLap(), Pit(), Rule()
        self.window_s, self.follow_gap_s = 1.0, 0.4

    def start_gap(self, n):
        return np.arange(n) * 0.6

    def lap1_extra(self, circuit):
        return 2.0


def field(n=6, laps=30, draws=8):
    drivers = [f'D{i}' for i in range(n)]
    s = pd.concat([st.strategy_stints(('SOFT', 'HARD') if i % 2 else ('MEDIUM', 'HARD'), (8 + 2 * i,), d)
                   for i, d in enumerate(drivers)])
    bases = 80 + np.tile(np.linspace(0, 1.5, n), (draws, 1)) + 0.1 * np.random.default_rng(1).standard_normal((draws, n))
    return drivers, s, bases


def test_wet_switch_is_off_by_default():
    cfg = load_config()
    assert cfg['simulator']['wet']['enabled'] is False and not wet.enabled(cfg)


def test_dry_race_is_reproduced_exactly_with_the_wet_track_on_and_no_wet_tyres():
    drivers, s, bases = field()
    arr = st.to_arrays(s, drivers, 8)
    dry = rc.simulate(Inputs(), 'X', 30, bases, arr, seed=3)
    gap = wet.scenario_gap(30, 'dry', [(10, 'wet'), (20, 'dry')], np.full((8, 2), 2.0), CFG)
    on = rc.simulate(Inputs(), 'X', 30, bases, st.to_arrays(s, drivers, 8, wet.codes(CFG)), seed=3,
                     wet=wet.engine_track(gap, CFG))
    for k in ('Time', 'Place', 'Passes', 'Pairs', 'PairsByLap'):
        assert np.array_equal(dry[k], on[k]), k


def test_a_wet_tyre_lap_adds_the_track_gap():
    s = pd.concat([st.strategy_stints(('MEDIUM', 'INTERMEDIATE', 'MEDIUM'), (9, 19), 'A'), st.strategy_stints(('MEDIUM',), (), 'B')])
    s['StartAge'] = 1.0
    bases = np.array([[80.0, 200.0]])           # B far behind: no interaction
    gap = np.zeros(31)
    gap[10:20] = -3.0
    track = {**wet.engine_track(gap, CFG), 'common': np.zeros(31)}

    class Flat(NoisyLap):
        class noise:
            draw_slopes = staticmethod(lambda rng, d, r: np.zeros((d, len(r))))
            draw_laps = staticmethod(lambda rng, shape: np.zeros(shape))

        def free_air(self, base, circuit, compound, age, lap, race_laps, rep=None):
            return np.broadcast_to(np.asarray(base, float), np.broadcast(np.asarray(compound), np.asarray(base)).shape) + 0.0

    inp = Inputs()
    inp.lap = Flat()
    inp.pit.sample = lambda circuit, rng, size: np.full(size, 20.0)
    arr = st.to_arrays(s, ['A', 'B'], 1, wet.codes(CFG))
    with_gap = rc.simulate(inp, 'X', 30, bases, arr, wet=track)
    no_gap = rc.simulate(inp, 'X', 30, bases, arr, wet={**track, 'gap': np.zeros(31)})
    assert np.isclose(with_gap['Time'][0, 0] - no_gap['Time'][0, 0], -30.0)   # laps 10 to 19 on the wet tyre


def test_scenario_gap_crosses_zero_at_the_change_laps_and_is_clipped():
    g = wet.scenario_gap(40, 'dry', [(10, 'wet'), (30, 'dry')], np.array([[2.0, 3.0]]), CFG)[0]
    assert g[9] > 0 > g[10] and g[29] < 0 < g[30]
    assert g.max() == 5.0 and g.min() == -10.0 and np.isclose(g[10], -1.0) and np.isclose(g[30], 1.5)
    assert (wet.scenario_gap(20, 'wet', [], np.zeros((1, 0)), CFG) == -10.0).all()


def test_parse_path():
    assert wet.parse_path('dry,wet@20,dry@35') == ('dry', [(20, 'wet'), (35, 'dry')])
    for bad in ('wet@3', 'dry,dry@4', 'dry,wet@20,dry@10'):
        with pytest.raises(ValueError):
            wet.parse_path(bad)


def test_overlay_replaces_close_and_wet_phase_stops():
    plan = [('MEDIUM', 1, 1.0), ('HARD', 18, 1.0), ('SOFT', 26, 1.0), ('SOFT', 45, 1.0)]
    out = wet.overlay(plan, 'dry', [(20, 'wet'), (30, 'dry')], 50, CFG)
    # 18 is within 5 laps of the switch at 20, 26 falls in the wet spell; 45 is kept
    assert out == [('MEDIUM', 1, 1.0), (wet.WET, 20, 1.0), ('SOFT', 30, 1.0), ('SOFT', 45, 1.0)]
    assert wet.overlay(plan, 'dry', [(20, 'wet'), (30, 'dry')], 50, CFG, after='HARD', later_stops=False)[-1] == ('HARD', 30, 1.0)
    assert wet.replan(plan, 'dry', [(20, 'wet'), (30, 'dry')], 50, CFG, caps={'SOFT': 15, 'MEDIUM': 25, 'HARD': 35})[-1] == ('MEDIUM', 30, 1.0)
    assert wet.overlay(plan, 'wet', [(6, 'dry')], 50, CFG)[0] == (wet.WET, 1, 1.0)


def test_rival_switches_skip_short_spells():
    assert wet.rival_switches('dry', [(20, 'wet'), (21, 'dry')], [0, 0], 50, CFG) == []
    assert wet.rival_switches('dry', [(20, 'wet'), (30, 'dry')], [2, -1], 50, CFG) == [(22, 'wet'), (29, 'dry')]


def race_laps_frame():
    """Six cars, laps 2 to 20: lap effect 10 - 0.5 x lap, driver bases 0 to 2.5; cars 0 to 2 switch to the wet tyre
    from lap 8 to 14, where it is 3 s slower than slicks (gap +3)."""
    rows = []
    for i in range(6):
        for lap in range(1, 21):
            wet_on = i < 3 and 8 <= lap <= 14
            t = 80 + 0.5 * i + 10 - 0.5 * lap + (3.0 if wet_on else 0.0)
            rows.append({'Year': 2024, 'Round': 1, 'Driver': f'D{i}', 'LapNumber': lap, 'LapTime_s': t, 'Stint': 1 + (lap >= 8) + (lap >= 15) if i < 3 else 1,
                         'Compound': 'INTERMEDIATE' if wet_on else 'MEDIUM', 'TyreLife': np.nan, 'PitIn': False, 'PitOut': False,
                         'TrackStatus': '1'})
    return pd.DataFrame(rows)


def test_track_fit_recovers_the_gap_and_bases():
    class Zero:
        def free_air(self, base, circuit, compound, age, lap, race_laps, rep=None):
            return np.zeros(np.shape(compound))
    l = race_laps_frame()
    gap, common, base, _ = wet.fit_track(l, Zero(), 'X', 20, CFG)
    assert set(gap.index) == set(range(8, 15)) and np.allclose(gap, 3.0)
    assert np.allclose(base.sort_index().diff().dropna(), 0.5)
    assert np.allclose(common.diff().dropna(), -0.5)


def test_shift_switch_moves_one_car():
    s = pd.concat([st.strategy_stints(('MEDIUM', 'INTERMEDIATE', 'HARD'), (9, 19), 'A'), st.strategy_stints(('MEDIUM',), (), 'B')])
    moved = wet.shift_switch(s, 'A', -2, CFG)
    assert moved.loc[moved['Driver'] == 'A', 'StartLap'].tolist() == [1.0, 8.0, 20.0]
    with pytest.raises(ValueError):
        wet.shift_switch(s, 'A', 12, CFG)
