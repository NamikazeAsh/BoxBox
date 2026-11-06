import numpy as np
import pandas as pd

from boxbox import overtaking as ot
from boxbox.simulator import race as rc, strategy as st

CFG = {'simulator': {'max_age': 30, 'race': {'stop_lap_step': 2, 'min_stint_laps': 5, 'max_stops': 2, 'refine_top': 3}}}


class StubLap:
    """Every lap at the car's base (no fuel, wear or compound gap), no noise, no dirty air."""
    class noise:
        @staticmethod
        def draw_slopes(rng, draws, races):
            return np.zeros((draws, len(races)))

        @staticmethod
        def draw_laps(rng, shape):
            return np.zeros(shape)

    def free_air(self, base, circuit, compound, age, lap, race_laps, rep=None):
        return np.broadcast_to(base, np.shape(compound)).astype(float) + 0.0

    def dirty_air(self, gap):
        return np.zeros_like(np.asarray(gap, float))


class StubPit:
    def sample(self, circuit, rng, size):
        return np.full(size, 20.0)

    def predict(self, circuits):
        return np.full(len(circuits), 20.0)


class StubRule:
    def __init__(self, threshold, beta=0.0):
        self.t, self.beta = threshold, beta

    def threshold(self, circuit):
        return self.t


class StubInputs:
    def __init__(self, threshold=0.5):
        self.lap, self.pit, self.rule = StubLap(), StubPit(), StubRule(threshold)
        self.window_s, self.follow_gap_s = 1.0, 0.4

    def start_gap(self, n):
        return np.arange(n) * 0.5

    def lap1_extra(self, circuit):
        return 0.0


def one_stop(driver, stop=10, comps=('MEDIUM', 'HARD')):
    return st.strategy_stints(comps, (stop,), driver)


def race(delta, threshold=0.5, laps=20, strategies=None, seed=0):
    """A (grid 1) at 80 s a lap, B (grid 2) delta s a lap faster, both one stop on lap 10 unless given."""
    s = strategies if strategies is not None else pd.concat([one_stop('A'), one_stop('B')])
    arr = st.to_arrays(s, ['A', 'B'], 2)
    bases = np.array([[80.0, 80.0 - delta]] * 2)
    return rc.simulate(StubInputs(threshold), 'Monza', laps, bases, arr, seed=seed)


def test_faster_car_is_held_under_the_threshold_and_passes_over_it():
    held = race(0.3)
    assert (held['Place'][:, 1] == 2).all() and (held['Passes'] == 0).all()
    assert np.allclose(held['Time'][:, 1] - held['Time'][:, 0], 0.4)       # exactly follow_gap_s behind
    passed = race(0.8)
    assert (passed['Place'][:, 1] == 1).all() and (passed['Passes'] == 1).all()


def test_pit_loss_is_added_on_the_in_lap():
    two = pd.concat([st.strategy_stints(('MEDIUM', 'HARD', 'MEDIUM'), (6, 12), 'A'), one_stop('B')])
    r = race(-5.0, strategies=two)   # B far slower: no interaction
    one = race(-5.0)
    assert np.allclose(r['Time'][:, 0] - one['Time'][:, 0], 20.0)


def test_fixed_strategies_give_their_stops_and_compounds():
    s = pd.concat([st.strategy_stints(('SOFT', 'HARD'), (12,), 'A', start_age=3.0), one_stop('B', 8, ('MEDIUM', 'SOFT'))])
    a = st.to_arrays(s, ['A', 'B'], 3)
    assert a['Stop'].shape == (3, 2, 2) and (a['Stop'][:, 0, 0] == 12).all() and np.isinf(a['Stop'][:, :, 1]).all()
    assert (a['Comp'][:, 0] == [st.CODE['SOFT'], st.CODE['HARD']]).all() and (a['Age'][:, 0, 0] == 3).all()


def test_legal_strategies_meet_the_two_compound_rule_and_stint_limits():
    out = st.legal_strategies(40, CFG)
    assert out and all(len(set(c)) >= 2 for c, _ in out)
    for comps, stops in out:
        lengths = np.diff((0,) + stops + (40,))
        assert len(comps) == len(stops) + 1 and lengths.min() >= 5 and lengths.max() <= 30


def test_real_stints_rebuild_a_race():
    stints = pd.DataFrame({'Year': 2024, 'Round': 1, 'Driver': ['A', 'A', 'B'], 'Stint': [1.0, 2.0, 1.0],
                           'Compound': ['MEDIUM', 'HARD', 'INTERMEDIATE'], 'StartLap': [1, 19, 1], 'TyreLifeStart': [2.0, 1.0, 1.0]})
    s = st.real_stints(stints, 2024, 1)
    assert s['Driver'].unique().tolist() == ['A']                       # a non dry stint drops the driver
    a = st.to_arrays(s, ['A'], 1)
    assert a['Stop'][0, 0, 0] == 18 and a['Age'][0, 0, 0] == 2


def test_same_seed_same_race():
    r1, r2 = race(0.3, seed=4), race(0.3, seed=4)
    assert np.array_equal(r1['Time'], r2['Time']) and np.array_equal(r1['Place'], r2['Place'])


def test_threshold_reproduces_each_circuits_pass_rate():
    rng = np.random.default_rng(0)
    p = pd.DataFrame({'Circuit': np.repeat(['Monaco', 'Monza'], 2000), 'Year': 2024, 'Round': 1,
                      'Gap': rng.uniform(0, 1, 4000), 'Delta': rng.normal(0, 1, 4000)})
    p['Pass'] = rng.random(4000) < np.where(p['Circuit'] == 'Monaco', 0.02, 0.2)
    m = ot.PassThreshold(k=0, beta=1.0).fit(p)
    rate = pd.Series(m.predict(p)).groupby(p['Circuit']).mean()
    real = p.groupby('Circuit')['Pass'].mean()
    assert np.allclose(rate, real, atol=0.002) and m.threshold('Monaco') > m.threshold('Monza')


def test_stint_caps_shrink_toward_the_compound_cap_and_bound_the_search():
    cfg = {'simulator': {**CFG['simulator'], 'race': {**CFG['simulator']['race'], 'stint_cap': {'quantile': 1.0, 'shrink': 10}}}}
    rows = [('Monza', 'SOFT', 10)] * 10 + [('Spa', 'SOFT', 30)] * 30 + [('Monza', 'HARD', 25)] * 10
    stints = pd.DataFrame([{'Year': 2024, 'Round': 1 if c == 'Monza' else 2, 'Driver': f'D{i}', 'Stint': 1.0, 'Compound': k,
                            'StartLap': 1, 'Laps': n, 'TyreLifeStart': 1.0} for i, (c, k, n) in enumerate(rows)])
    events = pd.DataFrame({'Year': 2024, 'Round': [1, 2], 'Circuit': ['Monza', 'Spa']})
    caps = st.stint_caps(stints, events, set(), [2024], cfg)
    monza = st.caps_for(caps, 'Monza')
    assert np.isclose(monza['SOFT'], (10 * 10 + 10 * 30) / 20)     # 10 stints at 10 laps, pulled toward 30
    assert st.caps_for(caps, 'Baku')['SOFT'] == 30                   # a new circuit gets the all circuit cap
    out = st.legal_strategies(40, cfg, caps=monza)
    for comps, stops in out:
        ends = np.diff((0,) + stops + (40,))
        assert all(e <= monza[k] for e, k in zip(ends, comps))
    assert len(out) < len(st.legal_strategies(40, cfg))


def test_search_groups_near_ties_and_flags_clearly_worse_strategies(monkeypatch):
    cfg = {'simulator': {**CFG['simulator'], 'race': {**CFG['simulator']['race'], 'tie_places': 0.25}}}

    def fake(inputs, circuit, race_laps, bases, arr, rule=None, seed=0):
        stops = np.isfinite(arr['Stop'][:, 1]).sum(axis=1)               # the user's car is the second
        place = np.where(stops == 1, 1, 3)
        return {'Time': np.zeros(bases.shape), 'Place': np.stack([np.full(len(place), 2), place], axis=1)}
    monkeypatch.setattr(rc, 'simulate', fake)
    rivals = one_stop('A')
    best = rc.best_strategies(StubInputs(), 'Monza', 30, ['A', 'B'], np.full((20, 2), 80.0), rivals, 'B', cfg)
    one, two = best[best['Stops'] == 1], best[best['Stops'] == 2]
    assert best.loc[0, 'Tier'] == 'best' and (one['Tier'] != 'worse').all() and (one['DeltaPlace'] == 0).all()
    assert (two['Tier'] == 'worse').all() and np.allclose(two['DeltaPlace'], 2) and (two['PBeatsBest'] == 0).all()
    assert (rc.tie_summary(best)['Stops'] == 1).all()


def test_each_draw_reads_its_own_tyre_model_refit():
    from boxbox.simulator.laptime import LapModel
    lm = LapModel.__new__(LapModel)
    lm.max_age = 3
    lm.wear_grid = {'Monza': np.zeros((3, 3))}
    lm.gaps = {'Monza': np.zeros(3)}
    lm.boot = {'wear': {'Monza': np.stack([np.zeros((3, 3)), np.ones((3, 3))])},
               'gap': {'Monza': np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.5]])}, 'evolution': np.array([0.0, 0.0])}
    comp, age = np.array([[0, 2], [0, 2]]), np.ones((2, 2))
    assert lm.n_boot == 2
    assert np.array_equal(lm.wear('Monza', comp, age, rep=np.array([0, 1])), [[0, 0], [1, 1]])
    assert np.array_equal(lm.gap('Monza', comp, rep=np.array([1, 0])), [[-1.0, 0.5], [0.0, 0.0]])
    assert np.array_equal(lm.wear('Monza', comp, age), np.zeros((2, 2)))   # no rep: the main fit


def test_wear_grid_rows_follow_the_engine_compound_order():
    """The grid's compound axis must be COMPOUNDS order (Soft, Medium, Hard), not alphabetical."""
    from boxbox.simulator.laptime import LapModel

    class Wear:
        def fit(self, laps):
            return self

        def wear(self, g):
            return g['Compound'].map({'SOFT': 3.0, 'MEDIUM': 2.0, 'HARD': 1.0}) * g['TyreLife']

    class Pace:
        def gap(self, c, k):
            return pd.Series(0.0, index=k.index)
    lm = LapModel.__new__(LapModel)
    lm.max_age = 4
    lm._cfg = {'simulator': {'wear_model': 'x', 'compound_pace': {'shrink': 1, 'min_laps': 1}}}
    import boxbox.simulator.laptime as lt
    orig = (lt.compare.ladder, lt.CompoundPace)
    lt.compare.ladder = lambda cfg: {'x': Wear}
    lt.CompoundPace = lambda *a: type('P', (), {'fit': lambda self, w, l: Pace()})()
    lm._cfg['simulator']['monotone_wear'] = False
    try:
        _, _, _, grid, _ = lm._fit_tyres(None, pd.DataFrame({'Circuit': ['Monza']}), ['Monza', 'Baku'])
    finally:
        lt.compare.ladder, lt.CompoundPace = orig
    assert np.allclose(grid['Monza'][:, 0], [3.0, 2.0, 1.0])     # Soft, Medium, Hard at age 1
