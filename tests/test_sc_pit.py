"""Rival stops under a safety car or VSC (pitdecision NeutralPits, replay with status), on small synthetic races."""
import numpy as np
import pandas as pd
import pytest

from boxbox import pitdecision as pdm
from boxbox.simulator import strategy as st

CFG = {'clean_lap': {'dry_compounds': ['SOFT', 'MEDIUM', 'HARD']},
       'pit_decision': {'first_lap': 2, 'neutral_cooldown_laps': 1, 'force_stop_laps_left': 3,
                        'force_new_compound_laps_left': 20,
                        'neutral': {'lap_dummies': 3, 'plan_flags': 'zero', 'replan': 'none', 'exclude_red_flag': True}}}
LAPS = 30


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


class Rate:
    def __init__(self, p):
        self.p = p

    def prob(self, d):
        return np.full(len(d), self.p)


class OnNeutral:
    """Pits at the end of every SC or VSC lap on a tyre older than 2 laps, never on a cooldown lap."""
    def prob(self, d):
        return np.where((d['LapStatus'] > 0) & (d['TyreLife'] > 2), 1.0, 0.0)


class Keep:
    """Tyre choice: the current compound whenever the two compound rule allows it."""
    def probs(self, d):
        p = pd.DataFrame({k: (d['Compound'] == k).astype(float) for k in pdm.DRY}, index=d.index)
        return pdm.mask_tyres(p, d, 20)


def starts(plan, n=1):
    return pd.DataFrame({'Year': 2024, 'Round': 1, 'Driver': [f'D{i}' for i in range(n)], 'Circuit': 'Monza',
                         'RaceLaps': LAPS, 'Compound': 'MEDIUM', 'TyreLife': 1.0, 'FreshTyre': True, 'Grid': 1,
                         'QualiGap': 0.0, 'TeamPaceForm': 0.0, 'DriverPaceForm': 0.0, 'Plan': float(plan)})


class Hard:
    def probs(self, d):
        return pdm.mask_tyres(pd.DataFrame({'SOFT': 0.0, 'MEDIUM': 0.0, 'HARD': 1.0}, index=d.index), d, 20)


def policy(green, neutral=None, again=None, replan='none', oracle=True, tyre=None):
    cfg = {**CFG, 'pit_decision': {**CFG['pit_decision'], 'neutral': {**CFG['pit_decision']['neutral'], 'replan': replan}}}
    np_ = None if neutral is None else pdm.NeutralPits(neutral, again, cfg)
    return pdm.PlanPolicy(None, None, green, tyre or Keep(), StubContext(), CFG, oracle=oracle, neutral=np_)


def status(*runs, draws=1):
    """Status codes (draws x laps + 1) with each (first, last, code) run of laps neutral."""
    a = np.zeros((draws, LAPS + 1), np.int8)
    for first, last, code in runs:
        a[:, first:last + 1] = code
    return a


def test_status_none_and_all_green_status_replay_as_v1():
    s = starts(2, n=3)
    plain = pdm.replay(policy(Rate(0.08)), s, draws=20, seed=3)
    attached = pdm.replay(policy(Rate(0.08), OnNeutral()), s, draws=20, seed=3)
    green = pdm.replay(policy(Rate(0.08), OnNeutral()), s, draws=20, seed=3, status=status())
    assert list(plain.columns) == pdm.DRIVER_RACE + ['Draw', 'SimStops', 'SimFirst', 'Plan']
    assert plain.equals(attached)                          # a NeutralPits is ignored without status
    assert plain.equals(green[plain.columns])              # every lap green: the same random draws, the same stops
    assert green['SimNeutral'].eq(0).all()


def test_a_high_neutral_pit_chance_moves_the_stop_onto_the_safety_car():
    sim = pdm.replay(policy(Never(), OnNeutral()), starts(1), draws=4, status=status((10, 12, 1)))
    assert sim['SimFirst'].eq(10).all() and sim['SimNeutral'].eq(1).all()
    assert sim['SimStops'].eq(1).all()                     # the plan is done: nothing forced near the end
    green = pdm.replay(policy(Never(), OnNeutral()), starts(1), draws=4, status=status())
    assert green['SimFirst'].eq(LAPS - CFG['pit_decision']['force_stop_laps_left']).all()


def test_a_neutral_stop_uses_up_a_planned_stop_and_an_extra_one_adds_to_the_plan():
    two = pdm.replay(policy(Never(), OnNeutral()), starts(2), draws=3, status=status((10, 11, 2)))
    assert two['SimStops'].eq(2).all() and two['SimNeutral'].eq(1).all()   # one under the VSC, one forced late
    extra = pdm.replay(policy(Never(), OnNeutral()), starts(1), draws=3, status=status((8, 9, 1), (20, 21, 1)))
    assert extra['SimStops'].eq(2).all() and extra['SimNeutral'].eq(2).all()   # the second SC stop is an extra one


def test_replan_after_a_neutral_stop():
    s = status((6, 7, 1))
    more = pdm.replay(policy(Never(), OnNeutral(), Rate(1.0), 'redraw'), starts(1), draws=3, status=s)
    assert more['SimStops'].eq(2).all() and more['PlanEnd'].eq(2).all() and more['Plan'].eq(1).all()
    fewer = pdm.replay(policy(Never(), OnNeutral(), Rate(0.0), 'redraw', tyre=Hard()), starts(2), draws=3, status=s)
    assert fewer['SimStops'].eq(1).all() and fewer['PlanEnd'].eq(1).all()   # two compounds run: no stop is needed
    forced = pdm.replay(policy(Never(), OnNeutral(), Rate(0.0), 'redraw'), starts(2), draws=3, status=s)
    assert forced['SimStops'].eq(2).all()                  # Medium again under the SC: the rule still forces a stop
    kept = pdm.replay(policy(Never(), OnNeutral(), Rate(0.0), 'add', tyre=Hard()), starts(2), draws=3, status=s)
    assert kept['SimStops'].eq(2).all()                    # 'add' never drops a planned stop


def test_per_draw_status_shapes_and_the_two_compound_rule():
    draws = 4
    s = status(draws=draws)
    s[1, 10:13] = 1                                        # a safety car in draw 1 only
    s[3, 15:17] = 2                                        # a VSC in draw 3 only
    pol = policy(Never(), OnNeutral())
    sim, events = pdm.replay(pol, starts(1, n=2), draws=draws, record=True, status=s)
    assert len(sim) == 2 * draws and set(events.columns) >= {'Draw', 'Driver', 'Lap', 'Compound', 'LapStatus'}
    assert sim.groupby('Draw')['SimNeutral'].max().tolist() == [0, 1, 0, 1]
    assert set(events.loc[events['LapStatus'] > 0, 'Lap']) == {10.0, 15.0}
    stints = st.model_stints(pol, starts(1, n=2), draws, status=s)
    assert (stints.groupby(['Draw', 'Driver'])['Compound'].nunique() >= 2).all()   # Keep never changes tyre unless forced
    assert set(stints.loc[stints['Draw'] == 1, 'StartLap']) == {1.0, 11.0}
    with pytest.raises(ValueError):
        pdm.replay(pol, starts(1), draws=draws, status=np.zeros((2, LAPS + 1), np.int8))   # neither 1 row nor draws
    with pytest.raises(ValueError):
        pdm.replay(pol, starts(1), draws=1, status=np.zeros((1, LAPS - 5), np.int8))      # too few laps


def test_status_features_runs_last_lap_and_cooldown():
    f = pdm.status_features(np.array([[0, 0, 1, 1, 0, 0, 2, 0]]), cooldown=1)
    assert f['NeutralLap'][0].tolist() == [0, 0, 1, 2, 0, 0, 1, 0]
    assert f['NeutralLast'][0].tolist() == [False, False, False, True, False, False, True, False]
    assert f['Cooldown'][0].tolist() == [False, False, False, False, True, False, False, True]


def race(driver, compounds, ages, pit_in, status):
    n = len(compounds)
    stint = 1.0 + np.r_[0, np.cumsum(np.asarray(pit_in[:-1], bool))]
    return pd.DataFrame({'Year': 2024, 'Round': 1, 'Driver': driver, 'Circuit': 'Monza', 'LapNumber': np.arange(1.0, n + 1),
                         'Stint': stint, 'Compound': compounds, 'TyreLife': ages, 'FreshTyre': True, 'PitIn': pit_in,
                         'TrackStatus': status, 'WetRace': False})


def test_neutral_rows_count_tyre_changes_only_and_cover_neutral_and_cooldown_laps():
    status = ['1'] * 5 + ['4', '4', '1', '1', '1']          # a safety car on laps 6 and 7
    a = race('A', ['MEDIUM'] * 6 + ['HARD'] * 4, [1, 2, 3, 4, 5, 6, 1, 2, 3, 4], [False] * 5 + [True] + [False] * 4, status)
    b = race('B', ['MEDIUM'] * 10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [False] * 6 + [True] + [False] * 3, status)   # led through the pit lane
    laps = pd.concat([a, b], ignore_index=True)
    starts_ = pd.DataFrame({'Year': 2024, 'Round': 1, 'Driver': ['A', 'B'], 'Plan': 1.0})
    pre = starts_[pdm.DRIVER_RACE].assign(Grid=1, QualiGap=0.0, TeamPaceForm=0.0, DriverPaceForm=0.0)
    rows, stops = pdm.neutral_rows(laps, pre, starts_, {**CFG, 'neutral': {'lap_share': 0.5}})
    r = rows.set_index(['Driver', 'LapNumber'])
    assert sorted(r.loc['A'].index) == [6.0, 7.0, 8.0]      # two SC laps and the cooldown lap
    assert r.loc[('A', 6.0), 'PitStop'] and not r.loc[('B', 7.0), 'PitStop']   # B kept its tyre: no stop
    assert r.loc[('B', 8.0), 'Stint'] == 1 and r.loc[('A', 8.0), 'Stint'] == 2
    assert r.loc[('A', 8.0), 'Cooldown'] and r.loc[('A', 6.0), 'NeutralLap'] == 1
    assert len(stops) == 1 and stops.loc[0, 'Compound'] == 'HARD' and stops.loc[0, 'TyreLife'] == 1 and not stops.loc[0, 'Again']
