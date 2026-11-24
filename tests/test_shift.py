import numpy as np
import pandas as pd
import pytest

from boxbox import degradation, shift

CFG = {'degradation': {'min_fit_laps': 3, 'anchor_laps': 3, 'min_eval_laps': 3},
       'training_seasons': {'shift_study_train': [2022, 2023, 2024, 2025], 'shift_study_test': [2026],
                            'shift_rehearsal_train': [2019, 2020, 2021], 'shift_rehearsal_test': [2022],
                            'outcome': [2019, 2020, 2021, 2022, 2023, 2024, 2025], 'degradation': [2022, 2023, 2024, 2025]},
       'holdout_seasons': [2026]}


def stint(driver, slope, year, rnd, laps=12, circuit='Monza'):
    age = np.arange(1, laps + 1)
    return pd.DataFrame({'Year': year, 'Round': rnd, 'Driver': driver, 'Stint': 1, 'Circuit': circuit, 'Compound': 'HARD',
                         'Team': 'X', 'LapNumber': age, 'TyreLife': age.astype(float), 'IsClean': True, 'WetRace': False,
                         'LapTimeFC_s': 80.0 + slope * age})


def test_splits_reference_drops_the_last_training_season():
    s = shift.splits(CFG, 'study')
    assert s == {'reference': ([2022, 2023, 2024], 2025), 'shift': ([2022, 2023, 2024, 2025], 2026)}
    assert shift.splits(CFG, 'rehearsal')['reference'] == ([2019, 2020], 2021)


def test_component_train_keeps_d2_seasons_before_the_test():
    assert shift.component_train(CFG, 'outcome', [2022, 2023, 2024, 2025], 2026) == list(range(2019, 2026))
    assert shift.component_train(CFG, 'degradation', [2019, 2020], 2021) == [2019, 2020]


def test_allocation_flags_compare_with_the_previous_season():
    wide = pd.DataFrame({'Year': [2025, 2025, 2026, 2026, 2026], 'Round': [1, 2, 1, 2, 3],
                         'Circuit': ['A', 'B', 'A', 'B', 'C'], 'HARD': [1, 4, 1, 3, 2], 'MEDIUM': [2, 5, 2, 4, 3],
                         'SOFT': [3, 6, 3, 5, 4]})
    f = shift.allocation_flags(wide, 2026).set_index('Circuit')
    assert not f.loc['A', 'Flagged'] and f.loc['B', 'Flagged'] and f.loc['C', 'New'] and not f.loc['C', 'Flagged']
    assert shift.allocation_flags(wide, 2025).empty


def test_drop_races_and_block_table():
    p = pd.DataFrame({'Year': 2026, 'Round': [1, 2, 7, 13], 'Err_s': [1.0, -1.0, 2.0, 4.0]})
    kept = shift.drop_races({'m': p}, [(2026, 13)])['m']
    assert list(kept['Round']) == [1, 2, 7]
    t = shift.block_table({'m': p})
    assert t.loc['m', '1 to 6'] == 1.0 and t.loc['m', '13 on'] == 4.0 and t.loc['m', 'bias'] == 1.5


def test_stays_ahead_finds_the_first_round_of_the_final_lead():
    cum = pd.DataFrame({'frozen': [1.0, 1.0, 1.0, 1.0], 'upd': [0.9, 1.1, 0.95, 0.9]}, index=[1, 2, 3, 4])
    assert shift.stays_ahead(cum, 'upd') == 3
    assert shift.stays_ahead(cum.assign(upd=[0.9, 0.9, 0.9, 1.2]), 'upd') is None


def test_round_mae_is_cumulative_over_laps():
    p = pd.DataFrame({'Round': [1, 1, 2], 'Err_s': [1.0, 3.0, 2.0]})
    per, cum = shift.round_mae({'m': p})
    assert per.loc[1, 'm'] == 2.0 and cum.loc[2, 'm'] == pytest.approx(2.0)


def test_choose_season_adjust_picks_the_lowest_mae():
    grid = {('global', 0): pd.DataFrame({'Err_s': [1.0]}), ('compound', 50): pd.DataFrame({'Err_s': [-0.5]})}
    assert shift.choose_season_adjust(grid) == ('compound', 50)


def test_score_table_gain_over_constant_pace():
    a = pd.DataFrame({'Err_s': [1.0, -1.0], 'Compound': 'HARD'})
    b = pd.DataFrame({'Err_s': [0.5, -0.5], 'Compound': 'HARD'})
    t = shift.score_table({'constant pace': a, 'linear': b})
    assert t.loc['linear', 'Gain'] == pytest.approx(0.5) and t.loc['constant pace', 'Gain'] == 0.0
    assert shift.score_table({'constant pace': a, 'linear': b}, 'Compound').loc[('linear', 'HARD'), 'Gain'] == pytest.approx(0.5)


def test_rolling_refit_learns_from_earlier_rounds_only():
    train = pd.concat([stint(d, 0.1, 2025, 1) for d in 'AB'])
    test = pd.concat([stint(d, 0.3, 2026, r) for r in [1, 2, 3] for d in 'AB'])
    laps = pd.concat([train, test], ignore_index=True)
    make = lambda: degradation.LinearDegradation(0)
    p = shift.rolling_refit(laps, CFG, [2025], 2026, make)
    frozen = degradation.anchored_forecast(make().fit(degradation.model_laps(laps, [2025], CFG)),
                                           degradation.model_laps(laps, [2026], CFG), CFG)
    err = lambda d, r: d.loc[d['Round'] == r, 'Err_s'].abs().mean()
    assert err(p, 1) == pytest.approx(err(frozen, 1))     # round 1 sees no test season laps
    assert err(p, 3) < err(p, 2) < err(frozen, 2)          # each round adds the ones before it


def test_excluded_races_and_drop_rounds():
    ev = pd.DataFrame({'Year': [2026, 2026, 2026], 'Round': [13, 14, 15], 'Circuit': ['Spa', 'Madrid', 'Monza']})
    skip = shift.excluded_races(ev, ['Madrid'])
    assert skip == {(2026, 14)}
    assert shift.excluded_races(ev, []) == set()
    laps = pd.DataFrame({'Year': [2026] * 3, 'Round': [13, 14, 15], 'LapTime_s': [90.0, 91.0, 92.0]})
    assert shift.drop_rounds(laps, skip)['Round'].tolist() == [13, 15]
    other = pd.DataFrame({'Circuit': ['Madrid']})
    assert shift.drop_rounds(other, skip) is other
