import numpy as np
import pandas as pd

from boxbox.simulator import laptime as lt, race as rc, strategy as st
from test_simulator import StubInputs, StubLap, one_stop


def stops(hard=0.4, soft=-0.2, races=4):
    """Two Medium, two Hard and one Soft stop per race; each race has its own out lap constant (the pit exit)."""
    rows = []
    for r in range(races):
        for c, x in [('MEDIUM', 0.0), ('MEDIUM', 0.0), ('HARD', hard), ('HARD', hard), ('SOFT', soft)]:
            rows.append({'Year': 2023, 'Round': r, 'Compound': c, 'X': 20.0 + 3 * r + x, 'W2': -0.1, 'W3': -0.05})
    return pd.DataFrame(rows)


def test_compound_effects_are_within_race_and_medium_is_zero():
    e = lt.compound_effects(stops())
    assert np.isclose(e['HARD'], 0.4) and np.isclose(e['SOFT'], -0.2) and e['MEDIUM'] == 0.0


def test_out_lap_cost_is_centred_on_the_stop_mix_and_early_laps_pool():
    rows = stops()
    oc, ea = lt.new_set_terms(rows, out_lap=True, early_laps=2)
    mix = rows['Compound'].map({c: i for i, c in enumerate(lt.COMPOUNDS)}).to_numpy()
    assert np.isclose(oc[mix].mean(), 0.0)                 # the pit loss already holds the average out lap
    assert np.isclose(oc[lt.COMPOUNDS.index('HARD')] - oc[lt.COMPOUNDS.index('MEDIUM')], 0.4)
    assert np.allclose(ea, [-0.1, -0.05])
    assert lt.new_set_terms(rows, out_lap=False, early_laps=0) == (None, None)


def test_new_set_terms_apply_on_the_out_lap_and_early_laps_of_a_fresh_set_only():
    lm = lt.LapModel.__new__(lt.LapModel)
    lm.out_cost, lm.early = np.array([0.1, 0.0, 0.3]), np.array([-0.2, -0.1])
    comp = np.array([2, 2, 2, 2, 2])
    age = np.array([1, 2, 3, 4, 2])
    out = np.array([True, False, False, False, False])
    fresh = np.array([True, True, True, True, False])
    assert np.allclose(lm.new_set(comp, age, out, fresh), [0.3, -0.2, -0.1, 0.0, 0.0])
    assert lm.has_new_set
    lm.out_cost, lm.early = None, None
    assert not lm.has_new_set and np.allclose(lm.new_set(comp, age, out, fresh), 0.0)


class CostLap(StubLap):
    has_new_set = True

    def new_set(self, compound, age, out_lap, fresh):
        return np.where(out_lap, 0.5, 0.0) + np.where(fresh & (np.asarray(age) == 2), -0.2, 0.0)


def test_the_engine_adds_the_new_set_terms_to_the_stopping_car():
    s = pd.concat([one_stop('A'), one_stop('B')])
    arr = st.to_arrays(s, ['A', 'B'], 2)
    bases = np.array([[80.0, 90.0]] * 2)    # far apart: no interaction
    plain = rc.simulate(StubInputs(), 'Monza', 20, bases, arr)
    inputs = StubInputs()
    inputs.lap = CostLap()
    cost = rc.simulate(inputs, 'Monza', 20, bases, arr)
    assert np.allclose(cost['Time'] - plain['Time'], 0.3)   # one out lap (+0.5) and one age 2 lap (-0.2) each
    t = rc.free_air_race_time(inputs, 'Monza', 20, 80.0, [(('MEDIUM', 'HARD'), (10,))])
    t0 = rc.free_air_race_time(StubInputs(), 'Monza', 20, 80.0, [(('MEDIUM', 'HARD'), (10,))])
    assert np.allclose(t - t0, 0.3)
