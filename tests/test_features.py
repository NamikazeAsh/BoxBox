import pandas as pd
import pytest

from boxbox import clean, features

CFG = {
    'clean_lap': {'traffic_gap_s': 1.0, 'outlier_pct': 1.07, 'dry_compounds': ['SOFT', 'MEDIUM', 'HARD']},
    'fuel': {'start_kg': 100.0, 's_per_kg': 0.03},
    'pit_loss': {'min_s': 5.0, 'max_s': 60.0},
    'long_run': {'min_laps': 3},
}


def make_laps(rows):
    cols = ['Driver', 'LapNumber', 'Time_s', 'LapTime_s', 'PitIn', 'PitOut', 'TrackStatus', 'Compound']
    df = pd.DataFrame(rows, columns=cols)
    df['Year'], df['Round'], df['Session'] = 2025, 1, 'R'
    return df


def two_car_race():
    # A leads B. B is 0.5 s behind on lap 2 and 3 s behind on lap 3. A pits on lap 3.
    return make_laps([
        ('A', 1, 100.0, 100.0, False, False, '1', 'MEDIUM'),
        ('B', 1, 101.0, 101.0, False, False, '1', 'MEDIUM'),
        ('A', 2, 190.0, 90.0, False, False, '1', 'MEDIUM'),
        ('B', 2, 190.5, 89.5, False, False, '1', 'MEDIUM'),
        ('A', 3, 300.0, 110.0, True, False, '1', 'MEDIUM'),
        ('B', 3, 280.0, 89.5, False, False, '1', 'MEDIUM'),
        ('A', 4, 420.0, 120.0, False, True, '1', 'HARD'),
        ('B', 4, 370.0, 90.0, False, False, '4', 'MEDIUM'),
    ])


def test_gap_ahead_is_within_lap_and_ordered_by_line_time():
    laps = features.add_gap_ahead(two_car_race())
    gap = laps.set_index(['Driver', 'LapNumber'])['GapAhead_s']
    assert gap[('B', 2)] == pytest.approx(0.5)
    assert pd.isna(gap[('A', 2)])
    # On lap 3 B crosses the line first, so A is the one behind
    assert gap[('A', 3)] == pytest.approx(20.0)


def test_clean_flags_drop_lap1_pits_traffic_and_non_green():
    laps = features.add_clean_flags(features.add_gap_ahead(two_car_race()), CFG)
    clean_laps = set(laps.loc[laps['IsClean'], ['Driver', 'LapNumber']].itertuples(index=False, name=None))
    # A lap 2 leads (clean). B lap 2 is in traffic. B lap 3 is clean. A laps 3 and 4 are pit laps. B lap 4 is SC.
    assert clean_laps == {('A', 2), ('B', 3)}


def test_fuel_correction_is_linear_to_zero():
    laps = make_laps([('A', n, 0, 90.0, False, False, '1', 'SOFT') for n in range(1, 11)])
    laps = features.add_fuel_correction(laps, CFG)
    assert laps['FuelKg'].iloc[0] == pytest.approx(100.0)
    assert laps['FuelKg'].iloc[-1] == pytest.approx(10.0)
    assert laps['LapTimeFC_s'].iloc[0] == pytest.approx(87.0)


def test_pit_loss_uses_in_and_out_lap_against_clean_median():
    laps = features.add_clean_flags(features.add_gap_ahead(two_car_race()), CFG)
    stops = features.pit_loss(laps, wet=set(), cfg=CFG)
    assert len(stops) == 1
    # A's only clean lap is 90 s, so loss = 110 + 120 - 2 * 90
    assert stops['PitLoss_s'].iloc[0] == pytest.approx(50.0)
    assert features.pit_loss(laps, wet={(2025, 1)}, cfg=CFG).empty


def practice(rows, stint):
    prac = make_laps(rows)
    prac['Session'], prac['Stint'], prac['Team'], prac['TyreLife'], prac['TrackTemp'] = 'FP2', stint, 'X', 1, 40
    return prac


def test_long_run_drops_a_slow_lap_instead_of_ending_the_run():
    rows = [('A', n, 0, 90.0 + 0.1 * n, False, False, '1', 'MEDIUM') for n in range(1, 8)]
    rows[4] = ('A', 5, 0, 150.0, False, False, '1', 'MEDIUM')   # traffic
    short = [('A', 8, 0, 95.0, False, True, '1', 'SOFT'), ('A', 9, 0, 89.0, False, False, '1', 'SOFT'),
             ('A', 10, 0, 99.0, True, False, '1', 'SOFT')]      # a new stint with one timed lap
    runs = features.long_runs(pd.concat([practice(rows, 1), practice(short, 2)]), CFG)
    assert runs['Laps'].tolist() == [6]
    assert runs['Slope_s_per_lap'].iloc[0] == pytest.approx(0.1)


def test_practice_fuel_correction_uses_the_race_burn_rate():
    race = make_laps([('A', n, 0.0, 90.0, False, False, '1', 'MEDIUM') for n in range(1, 51)])
    prac = practice([('A', n, 0, 90.0, False, False, '1', 'MEDIUM') for n in range(11, 16)], 1)
    runs = features.add_practice_fuel_correction(features.practice_runs(prac, CFG), race, CFG)
    # 2 kg per lap at 0.03 s per kg: flat raw times are 0.06 s per lap slower once fuel is held fixed
    assert runs['LapTimeFC_s'].tolist() == pytest.approx([90.0, 90.06, 90.12, 90.18, 90.24])


def test_pitstop_duration_parses_seconds_and_minutes():
    raw = pd.DataFrame({'Year': 2025, 'Round': 1, 'driverId': 'x', 'lap': '10', 'stop': '1',
                        'duration': ['23.456', '1:02.500', '']})
    out = clean.clean_pitstops(raw)
    assert out['PitLane_s'].iloc[0] == pytest.approx(23.456)
    assert out['PitLane_s'].iloc[1] == pytest.approx(62.5)
    assert pd.isna(out['PitLane_s'].iloc[2])


def test_wet_race_needs_wet_tyres_not_just_rain():
    laps = pd.concat([
        make_laps([('A', 1, 90.0, 90.0, False, False, '1', 'MEDIUM')]).assign(Round=1, Rainfall=True),
        make_laps([('A', 1, 90.0, 90.0, False, False, '1', 'INTERMEDIATE')]).assign(Round=2, Rainfall=False),
    ])
    assert features.wet_races(laps) == {(2025, 2)}


def test_circuit_merges_renamed_venues_and_splits_sakhir_outer():
    ev = pd.DataFrame({'Year': [2019, 2022, 2020, 2020],
                       'EventName': ['Monaco Grand Prix', 'Monaco Grand Prix', 'Bahrain Grand Prix', 'Sakhir Grand Prix'],
                       'Location': ['Monte Carlo', 'Monaco', 'Sakhir', 'Sakhir']})
    cfg = {'circuits': {'aliases': {'Monte Carlo': 'Monaco'},
                        'layouts': [{'year': 2020, 'event': 'Sakhir Grand Prix', 'circuit': 'Sakhir Outer'}]}}
    assert features.add_circuit(ev, cfg)['Circuit'].tolist() == ['Monaco', 'Monaco', 'Sakhir', 'Sakhir Outer']


def test_sprint_fuel_scales_with_distance():
    race = make_laps([('A', n, 0.0, 90.0, False, False, '1', 'MEDIUM') for n in range(1, 61)])
    sprint = make_laps([('A', n, 0.0, 90.0, False, False, '1', 'MEDIUM') for n in range(1, 21)])
    start = features.sprint_start_fuel(sprint, race, CFG)
    assert start.iloc[0] == pytest.approx(100.0 / 3)
    out = features.add_fuel_correction(sprint.copy(), CFG, start)
    assert out['FuelKg'].iloc[0] == pytest.approx(100.0 / 3)
