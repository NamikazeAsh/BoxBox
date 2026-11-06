import numpy as np
import pandas as pd

from boxbox import features, pitdecision as pdm

CFG = {'clean_lap': {'dry_compounds': ['SOFT', 'MEDIUM', 'HARD']},
       'pit_decision': {'first_lap': 2, 'neutral_cooldown_laps': 1, 'force_stop_laps_left': 3,
                        'force_new_compound_laps_left': 20}}


def race(driver, n=12, stop_after=None, pit_in=(), status=None, first='MEDIUM', second='HARD'):
    """One driver's race laps: a tyre change at the end of lap stop_after; pit_in adds PitIn laps with no
    change (a penalty, or retiring in the pits on the last lap)."""
    lap = np.arange(1, n + 1, dtype=float)
    after = lap > stop_after if stop_after else np.zeros(n, bool)
    age = np.where(after, lap - (stop_after or 0), lap)
    return pd.DataFrame({'Year': 2024, 'Round': 1, 'Driver': driver, 'Circuit': 'Monza', 'LapNumber': lap,
                         'Stint': np.where(after, 2.0, 1.0), 'Compound': np.where(after, second, first),
                         'TyreLife': age, 'FreshTyre': True, 'PitIn': (lap == stop_after) | np.isin(lap, pit_in),
                         'TrackStatus': status if status is not None else ['1'] * n, 'WetRace': False})


def test_labels_keep_tyre_stops_and_drop_retirements_penalties_and_neutral_laps():
    status = ['1'] * 12
    status[4] = '4'                                             # safety car on lap 5
    laps = pd.concat([race('A', stop_after=6), race('B', n=8, pit_in=[8]), race('C', stop_after=9, pit_in=[4]),
                      race('D', status=status)], ignore_index=True)
    lab = features.pit_labels(laps, CFG).join(laps[['Driver', 'LapNumber']])
    a = lab[lab['Driver'] == 'A'].set_index('LapNumber')
    assert a.loc[6, 'PitStop'] and a.loc[6, 'PitRow'] and a['PitStop'].sum() == 1
    assert not a.loc[1, 'PitRow']                               # lap 1 stops are damage
    b = lab[lab['Driver'] == 'B'].set_index('LapNumber')
    assert not b.loc[8, 'PitStop'] and not b.loc[8, 'PitRow']   # retired in the pits
    c = lab[lab['Driver'] == 'C'].set_index('LapNumber')
    assert not c.loc[4, 'PitStop'] and not c.loc[4, 'PitRow']   # penalty: same stint after
    assert c.loc[9, 'PitStop']
    d = lab[lab['Driver'] == 'D'].set_index('LapNumber')
    assert not d.loc[5, 'PitRow'] and not d.loc[6, 'PitRow'] and d.loc[7, 'PitRow']   # SC lap and cooldown


def test_pit_laps_used_flags_never_look_ahead():
    laps = race('A', stop_after=6)
    pre = pd.DataFrame({'Year': [2024], 'Round': [1], 'Driver': ['A'], 'Grid': [3], 'QualiGap': [0.2],
                        'TeamPaceForm': [0.1], 'DriverPaceForm': [0.0]})
    rows = pdm.pit_laps(laps, pre, CFG).set_index('LapNumber')
    assert not rows.loc[6, 'UsedHard'] and rows.loc[7, 'UsedHard'] and rows.loc[6, 'UsedMedium']
    assert rows.loc[6, 'NextCompound'] == 'HARD' and rows.loc[6, 'PitStop']
    assert rows['RaceLaps'].eq(12).all() and rows.loc[6, 'Grid'] == 3


def test_season_offsets_use_only_earlier_races():
    d = pd.DataFrame({'Year': 2024, 'Round': [1, 1, 2, 2, 3, 3], 'PitStop': [1, 1, 0, 0, 1, 0]})
    p = np.full(6, 0.5)                                   # expects one stop a race
    off = pdm.season_offsets(p, d, k=0)
    assert off.loc[(2024, 1)] == 0                       # first race: nothing earlier
    assert np.isclose(off.loc[(2024, 2)], np.log(2 / 1))  # race 1: two real stops against one expected
    assert np.isclose(off.loc[(2024, 3)], np.log(2 / 2))  # races 1 and 2: two against two
    assert np.isclose(pdm.season_offsets(p, d, k=100).loc[(2024, 2)], np.log(102 / 101))   # pulled toward 0


def test_tyre_mask_enforces_the_two_compound_rule():
    d = pd.DataFrame({'TwoCompounds': [0.0, 0.0, 1.0], 'LapsLeft': [10, 30, 10],
                      'UsedSoft': [False, False, True], 'UsedMedium': [True, True, True], 'UsedHard': False})
    p = pd.DataFrame({'SOFT': 0.2, 'MEDIUM': 0.6, 'HARD': 0.2}, index=d.index)
    m = pdm.mask_tyres(p, d, force_laps_left=20)
    assert m.loc[0, 'MEDIUM'] == 0 and np.isclose(m.loc[0].sum(), 1)   # one compound, few laps left
    assert np.allclose(m.loc[1], p.loc[1]) and np.allclose(m.loc[2], p.loc[2])


class StubContext:
    """No wear and a fixed pit loss, so policy tests need no fitted models."""
    def wear_features(self, d, shift=False):
        cols = ['WearLoss', 'WearRate', 'SoftRate', 'MediumRate', 'HardRate', 'CircuitWear']
        return pd.DataFrame(0.0, index=d.index, columns=cols)

    class pit:
        @staticmethod
        def predict(c):
            return np.full(len(c), 22.0)


class Never:
    def prob(self, d):
        return np.zeros(len(d))


class AlwaysSoft:
    def probs(self, d):
        return pdm.mask_tyres(pd.DataFrame({'SOFT': 1.0, 'MEDIUM': 0.0, 'HARD': 0.0}, index=d.index), d, 20)


def test_replay_forces_one_legal_stop_when_the_hazard_never_pits():
    policy = pdm.RivalPolicy(Never(), AlwaysSoft(), StubContext(), CFG)
    starts = pd.DataFrame({'Year': 2024, 'Round': 1, 'Driver': ['A', 'B'], 'Circuit': 'Monza', 'RaceLaps': 30,
                           'Compound': ['SOFT', 'MEDIUM'], 'TyreLife': 1.0, 'FreshTyre': True, 'Grid': [1, 2],
                           'QualiGap': 0.0, 'TeamPaceForm': 0.0, 'DriverPaceForm': 0.0})
    sim = pdm.replay(policy, starts, draws=5)
    assert sim['SimStops'].eq(1).all()
    assert sim['SimFirst'].eq(30 - CFG['pit_decision']['force_stop_laps_left']).all()   # forced as late as allowed
    # A starts on Soft: the forced stop cannot fit Soft again, even though the tyre model always picks it
    state = starts.assign(LapNumber=27.0, Stint=1.0, UsedSoft=[True, False], UsedMedium=[False, True], UsedHard=False)
    choice = policy.choose_tyre(state, np.random.default_rng(0))
    assert choice[0] != 'SOFT' and choice[1] == 'SOFT'


class Always:
    def prob(self, d):
        return np.ones(len(d))


def starts_with_plan(plan):
    return pd.DataFrame({'Year': 2024, 'Round': 1, 'Driver': ['A', 'B', 'C'], 'Circuit': 'Monza', 'RaceLaps': 30,
                         'Compound': 'MEDIUM', 'TyreLife': 1.0, 'FreshTyre': True, 'Grid': [1, 2, 3], 'QualiGap': 0.0,
                         'TeamPaceForm': 0.0, 'DriverPaceForm': 0.0, 'Plan': plan})


def test_plan_policy_makes_exactly_the_planned_stops():
    starts = starts_with_plan([1.0, 2.0, 3.0])
    for hazard in [Always(), Never()]:   # pits whenever a stop is left, or only when forced near the end
        policy = pdm.PlanPolicy(None, None, hazard, AlwaysSoft(), StubContext(), CFG, oracle=True)
        sim = pdm.replay(policy, starts, draws=3)
        assert (sim['SimStops'] == sim['Plan']).all()


def test_plan_shares_override_the_plan_model():
    policy = pdm.PlanPolicy(None, None, Never(), AlwaysSoft(), StubContext(), CFG, plan_shares={1: 1.0})
    plan = policy.sample_plan(starts_with_plan(np.nan), np.random.default_rng(0))
    assert (plan == 1).all()


def test_last_planned_stop_must_bring_a_second_compound():
    d = pd.DataFrame({'TwoCompounds': [0.0, 0.0], 'LapsLeft': [40, 40], 'StopsLeft': [1, 2],
                      'UsedSoft': False, 'UsedMedium': True, 'UsedHard': False})
    p = pd.DataFrame({'SOFT': 0.2, 'MEDIUM': 0.6, 'HARD': 0.2}, index=d.index)
    m = pdm.mask_tyres(p, d, force_laps_left=20)
    assert m.loc[0, 'MEDIUM'] == 0 and np.allclose(m.loc[1], p.loc[1])   # a stop still to come after this one


def test_plan_timing_features_and_circuit_history_leave_own_season_out():
    d = pdm.add_features(starts_with_plan([2.0, 2.0, 1.0]).assign(LapNumber=11.0, Stint=[1.0, 2.0, 2.0], TyreLife=10.0,
                                                                   UsedSoft=False, UsedMedium=True, UsedHard=False), StubContext())
    assert d['StopsLeft'].tolist() == [2, 1, 0]
    assert np.isclose(d.loc[0, 'StintTarget'], (10 + 19) / 3) and np.isclose(d.loc[0, 'AgeShare'], 10 / ((10 + 19) / 3))
    s = pd.DataFrame({'Circuit': 'Monza', 'Year': [2022, 2022, 2023], 'Plan': [3.0, 3.0, 1.0]})
    cs = pdm.CircuitStops(k=0).fit(s)
    assert np.allclose(cs.predict(s, leave_season_out=True), [1.0, 1.0, 3.0])
    assert np.allclose(cs.predict(s), 7 / 3)
