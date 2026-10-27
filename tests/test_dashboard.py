import pandas as pd
import pytest

from boxbox import dashboard, features


def test_safety_car_races_reads_every_status_on_a_lap():
    laps = pd.DataFrame({'Year': 2025, 'Round': [1, 1, 2, 2, 3],
                         'TrackStatus': ['1', '41', '1', '6', '1']})
    f = features.safety_car_races(laps).set_index('Round')
    assert f.loc[1, 'SafetyCar'] and not f.loc[1, 'VSC']
    assert f.loc[2, 'VSC'] and not f.loc[2, 'SafetyCar']
    assert not f.loc[3, 'SafetyCar'] and not f.loc[3, 'VSC']


def test_holdout_is_locked_until_switched_on(tmp_path):
    cfg = {'dashboard': {'show_holdout': False}, 'paths': {'processed': tmp_path}}
    with pytest.raises(PermissionError):
        dashboard.table(cfg, 'laps', holdout=True)


CELLS = pd.DataFrame({'Circuit': ['A', 'A'], 'Compound': ['SOFT', 'MEDIUM'], 'Raw': [0.1, 0.05], 'Stints': [10.0, 10.0],
                      'Pooled': [0.08, 0.04], 'Slope': [0.1, 0.05], 'ResidSD_s': [0.5, 0.5], 'Laps': [100, 100],
                      'LengthCoef': [0.002, 0.001], 'AgeTerm': 'linear'})


def test_wear_curve_starts_at_zero_and_follows_the_slope():
    w = dashboard.wear_curve(CELLS, 'A', 'SOFT', stint_laps=20, length_ref=20)
    assert w['Loss_s'].iloc[0] == 0
    assert w['Loss_s'].iloc[-1] == pytest.approx(0.1 * 19)
    assert (w['Hi_s'] > w['Loss_s']).all() and (w['Lo_s'] < w['Loss_s']).all()


def test_wear_curve_stint_length_changes_the_slope():
    w = dashboard.wear_curve(CELLS, 'A', 'SOFT', stint_laps=30, length_ref=20)
    assert w['Loss_s'].iloc[-1] == pytest.approx((0.1 + 0.002 * 10) * 29)


def test_wear_curve_falls_back_to_the_pooled_slope():
    cells = CELLS.assign(Circuit=['B', 'A'])
    w = dashboard.wear_curve(cells, 'A', 'SOFT', stint_laps=11, length_ref=11)
    assert not w['FromCircuit'].iloc[0]
    assert w['Loss_s'].iloc[-1] == pytest.approx(0.08 * 10)


def test_race_prediction_joins_the_grid_baseline():
    p = pd.DataFrame({'Model': ['ridge'] * 2 + ['grid'] * 2, 'Year': 2025, 'Round': 1, 'Driver': ['X', 'Y'] * 2,
                      'Team': 'T', 'Lineage': 'T', 'Grid': [2, 1] * 2, 'Finish': [1, 2] * 2,
                      'Pred': [1, 2, 2, 1], 'Err': [0, 0, 1, -1]})
    r = dashboard.race_prediction(p, 2025, 1, 'ridge')
    assert list(r['Driver']) == ['X', 'Y']
    assert list(r['GridPred']) == [2, 1]


def test_track_outline_rotates_and_scales_to_a_unit_box():
    # An L shape 200 wide and 100 tall, rotated 90 degrees: it should come back 100 wide and 200 tall
    maps = pd.DataFrame({'Circuit': 'A', 'X': [0.0, 200.0, 200.0], 'Y': [0.0, 0.0, 100.0], 'Rotation': 90.0})
    o = dashboard.track_outline(maps, 'A')
    assert o['X'].max() == pytest.approx(0.5) and o['Y'].max() == pytest.approx(1.0)
    assert o['X'].min() == pytest.approx(0) and o['Y'].min() == pytest.approx(0)
    assert dashboard.track_outline(maps, 'B') is None


def test_track_map_races_takes_the_latest_race_outside_the_holdout():
    from boxbox.ingest import track_map_races
    ev = pd.DataFrame({'Circuit': ['A', 'A', 'B', 'A'], 'Year': [2023, 2025, 2024, 2026], 'Round': [3, 2, 5, 1]})
    r = track_map_races(ev, {'holdout_seasons': [2026]}).set_index('Circuit')
    assert r.loc['A', 'Year'] == 2025 and r.loc['B', 'Year'] == 2024
