import numpy as np
import pandas as pd
import pytest

from boxbox import degradation
from boxbox.degradation.compound import CompoundPace, SameLevel, level_forecast
from boxbox.degradation.monotone import MonotoneWear
from boxbox.degradation.noise import WearNoise, simulate_errors
from boxbox.pitloss import GlobalPitLoss, PitLoss, race_medians

CFG = {'degradation': {'min_fit_laps': 3, 'anchor_laps': 3, 'min_eval_laps': 3, 'anchor_level': 'median'}}
GAP = {'SOFT': -0.5, 'MEDIUM': 0.0, 'HARD': 0.3}


def stint(driver, stint_no, compound, level, slope, start, laps=15, rnd=1, circuit='Monza', noise=0.0, seed=0):
    age = np.arange(1, laps + 1, dtype=float)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'Year': 2024, 'Round': rnd, 'Driver': driver, 'Stint': stint_no, 'Circuit': circuit,
                         'Compound': compound, 'LapNumber': start + age - 1, 'TyreLife': age, 'StintLaps': laps,
                         'RaceLaps': 60, 'LapTimeFC_s': level + slope * age + noise * rng.standard_normal(laps)})


def two_stop_race(rnd, driver, pace, first, second, circuit='Monza'):
    """A driver's race: first compound for laps 1 to 15, second for 16 to 30, same car pace."""
    return pd.concat([stint(driver, 1, first, pace + GAP[first], 0.05, 1, rnd=rnd, circuit=circuit),
                      stint(driver, 2, second, pace + GAP[second], 0.05, 16, rnd=rnd, circuit=circuit)])


def test_monotone_wear_never_falls_and_keeps_rising_curves():
    laps = pd.concat([stint('A', 1, 'MEDIUM', 80.0, -0.02, 1, laps=30), stint('B', 1, 'HARD', 80.0, 0.05, 1, laps=30)])
    m = MonotoneWear(degradation.LinearDegradation(0), max_age=40).fit(laps)
    w = m.wear(laps)
    med = w[laps['Compound'] == 'MEDIUM'].to_numpy()
    assert np.all(np.diff(med) >= -1e-12) and np.ptp(med) < 1e-9          # falling wear is held flat
    hard = laps['Compound'] == 'HARD'
    assert np.allclose(w[hard], m.base.wear(laps[hard]))                 # rising wear is unchanged


def test_compound_pace_recovers_gaps_within_driver_races():
    pairs = [('SOFT', 'MEDIUM'), ('MEDIUM', 'HARD'), ('SOFT', 'HARD'), ('HARD', 'MEDIUM')]
    laps = pd.concat([two_stop_race(r, d, 80.0 + i, *pairs[(r + i) % 4]) for r in range(6) for i, d in enumerate('ABCD')])
    wear = degradation.LinearDegradation(0).fit(laps)
    m = CompoundPace(shrink=0).fit(wear, laps)
    assert m.gaps['SOFT'] == pytest.approx(-0.5, abs=1e-6)
    assert m.gaps['HARD'] == pytest.approx(0.3, abs=1e-6)
    p = level_forecast(m, wear, laps)
    assert p['Err_s'].abs().max() < 1e-6
    assert level_forecast(SameLevel(), wear, laps)['Err_s'].abs().mean() > 0.2


def test_compound_pace_shrinks_a_sparse_circuit_toward_the_overall_gap():
    laps = pd.concat([two_stop_race(r, d, 80.0, 'SOFT', 'MEDIUM') for r in range(6) for d in 'ABCD']
                     + [two_stop_race(9, 'A', 90.0, 'SOFT', 'MEDIUM', circuit='Baku').assign(
                         LapTimeFC_s=lambda d: d['LapTimeFC_s'] - 1.0 * (d['Compound'] == 'SOFT'))])
    wear = degradation.LinearDegradation(0).fit(laps)
    m = CompoundPace(shrink=10).fit(wear, laps)
    baku = m.circuit_gaps.loc['Baku', 'SOFT']
    assert -1.5 < baku < m.gaps['SOFT'] < -0.5


def test_noise_finds_a_shared_race_slope_shift():
    rng = np.random.default_rng(0)
    races = []
    for r in range(40):
        shift = rng.normal(0, 0.03)
        races += [stint(f'D{i}', 1, 'MEDIUM', 80.0, 0.05 + shift + rng.normal(0, 0.01), 1, laps=25, rnd=r, noise=0.3, seed=100 * r + i)
                  for i in range(10)]
    laps = pd.concat(races)
    wear = degradation.LinearDegradation(1000, by_circuit=False).fit(laps)
    n = WearNoise().fit(wear, laps)
    assert n.tau_r == pytest.approx(0.03, rel=0.35)
    assert n.tau_u < 0.02
    assert n.sigma == pytest.approx(0.3, rel=0.15)


def test_simulated_errors_grow_with_a_race_spread():
    laps = stint('A', 1, 'MEDIUM', 80.0, 0.05, 1, laps=25)
    p = degradation.anchored_forecast(degradation.LinearDegradation(0).fit(laps), laps, CFG).assign(AnchorAges=[[1.0, 2.0, 3.0]] * 22)
    n = WearNoise()
    n.lap_resid, n.tau_u = np.random.default_rng(0).normal(0, 0.3, 5000), 0.0
    n.tau_r = 0.0
    quiet = simulate_errors(n, p, CFG, draws=2000)[1].std()
    n.tau_r = 0.03
    loud = simulate_errors(n, p, CFG, draws=2000)[1].std()
    assert loud > 1.5 * quiet


def test_pit_loss_uses_recent_seasons_and_falls_back():
    stops = pd.DataFrame({'Year': [2022] * 12 + [2023] * 12 + [2024] * 3, 'Round': 1,
                          'Circuit': ['Marina Bay'] * 24 + ['Baku'] * 3, 'PitLoss_s': [30.0] * 12 + [23.0] * 12 + [20.0] * 3})
    recent = PitLoss(recent=1, min_stops=10).fit(stops)
    assert recent.predict(['Marina Bay'])[0] == 23.0
    assert PitLoss(None, 10).fit(stops).predict(['Marina Bay'])[0] == pytest.approx(26.5)
    assert recent.predict(['Baku'])[0] == 20.0                  # too few recent stops: all its seasons
    assert recent.predict(['Madrid'])[0] == stops['PitLoss_s'].median()
    assert GlobalPitLoss().fit(stops).predict(['Baku'])[0] == stops['PitLoss_s'].median()
    assert len(race_medians(stops, min_stops=5)) == 2
