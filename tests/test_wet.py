import numpy as np
import pandas as pd
import pytest

from boxbox import degradation, features, wet

CFG = {'holdout_seasons': [2026],
       'clean_lap': {'traffic_gap_s': 1.0, 'dry_compounds': ['SOFT', 'MEDIUM', 'HARD']},
       'degradation': {'min_fit_laps': 3, 'anchor_laps': 3, 'min_eval_laps': 3},
       'wet': {'compounds': ['INTERMEDIATE', 'WET'], 'outlier_pct': 1.07,
               'crossover': {'wet_group': ['INTERMEDIATE'], 'outlier_pct': 1.07, 'min_cars': 2, 'max_gap_laps': 2,
                             'min_window_laps': 2, 'allow_yellow': True}}}


def lap_rows(n=6, compound='INTERMEDIATE', **kw):
    d = pd.DataFrame({'Year': 2024, 'Round': 1, 'Driver': 'A', 'Stint': 1, 'LapNumber': np.arange(2, n + 2),
                      'Compound': compound, 'LapTime_s': 100.0, 'PitIn': False, 'PitOut': False, 'TrackStatus': '1',
                      'GapAhead_s': 5.0})
    return d.assign(**kw)


def test_wet_clean_keeps_green_inter_laps_and_drops_the_rest():
    d = lap_rows()
    d.loc[1, 'PitIn'] = True
    d.loc[2, 'TrackStatus'] = '4'
    d.loc[3, 'LapTime_s'] = 120.0      # a spin: over 107% of the stint median
    d.loc[4, 'GapAhead_s'] = 0.5       # traffic
    assert features.wet_clean(d, CFG).tolist() == [True, False, False, False, False, True]
    assert not features.wet_clean(lap_rows(compound='MEDIUM'), CFG).any()


def test_tyre_group():
    g = features.tyre_group(pd.Series(['SOFT', 'INTERMEDIATE', 'WET', 'UNKNOWN']), CFG)
    assert g.iloc[0] == 'slick' and g.iloc[1] == 'wet' and g.iloc[2:].isna().all()


def test_net_of_track_separates_tyre_wear_from_a_drying_track():
    # Track gets 0.3 s per lap faster, tyres 0.1 s per lap slower; drivers refit fresh inters at different laps.
    rows = []
    for drv, stop, offset in [('A', 8, 0.0), ('B', 12, 1.0), ('C', 15, -0.5)]:
        for lap in range(2, 25):
            stint, age = (1, lap) if lap < stop else (2, lap - stop + 1)
            rows.append({'Year': 2024, 'Round': 1, 'Driver': drv, 'Stint': stint, 'Compound': 'INTERMEDIATE',
                         'LapNumber': lap, 'TyreLife': float(age), 'LapTimeFC_s': 100 + offset - 0.3 * lap + 0.1 * age})
    m = wet.TrackNetWear(iters=500).fit(pd.DataFrame(rows))
    assert m.cells.loc['INTERMEDIATE', 'Slope'] == pytest.approx(0.1, abs=1e-3)
    assert m.cells.loc['INTERMEDIATE', 'WithinStint'] == pytest.approx(-0.2, abs=1e-6)


def drying_race(switch_laps, gap=lambda l: 2.0 - (l - 5)):
    """Cars switch from inters to slicks at the given laps; slick minus inter gap is gap(lap)."""
    rows = []
    for i, s in enumerate(switch_laps):
        for lap in range(2, 12):
            on_slick = lap >= s
            t = 100 - 0.5 * lap + 0.3 * i + (gap(lap) if on_slick else 0.0)
            rows.append({'Year': 2024, 'Round': 1, 'Driver': f'D{i}', 'LapNumber': lap, 'LapTime_s': t,
                         'Group': 'slick' if on_slick else 'wet', 'Compound': 'MEDIUM' if on_slick else 'INTERMEDIATE'})
    return pd.DataFrame(rows)


def test_fit_gap_recovers_the_line_and_crossover_with_driver_offsets():
    d = drying_race([3, 4, 5, 6, 7, 8, 9])
    a, b = wet.fit_gap(d, lap0=5.0)
    assert a == pytest.approx(2.0) and b == pytest.approx(-1.0)
    assert 5.0 - a / b == pytest.approx(7.0)
    per = wet.fit_lap_gaps(d)
    assert per.loc[4] == pytest.approx(3.0) and per.loc[8] == pytest.approx(-1.0)


def test_windows_find_the_overlap_laps():
    w = wet.windows(drying_race([3, 4, 5, 6, 7, 8, 9]), CFG)
    assert len(w) == 1 and (w.loc[0, 'Start'], w.loc[0, 'End']) == (4, 7)   # at least 2 cars on each group


def test_switch_losses_price_a_late_and_an_early_switch():
    d = drying_race([7, 9, 5])
    s = wet.switch_losses(d, 2024, 1, 3, 10, lambda l: 2.0 - (l - 5.0), -1.0).set_index('Driver')
    # Gap is 0 at lap 7 and negative after: switching at 7 or 8 is best (lap 7 costs nothing either way)
    assert s.loc['D0', 'Loss_s'] == pytest.approx(0.0)
    assert s.loc['D1', 'Loss_s'] == pytest.approx(1.0)          # laps 8 on inters while slicks 1 s faster
    assert s.loc['D2', 'Loss_s'] == pytest.approx(2.0 + 1.0)    # laps 5 and 6 on slicks while 2 and 1 s slower
    assert wet.lateness_cost(-1.0, 2) == pytest.approx(2.0)


def stint(driver, level, slope, laps=10, rnd=1):
    age = np.arange(1, laps + 1)
    return pd.DataFrame({'Year': 2024, 'Round': rnd, 'Driver': driver, 'Stint': 1, 'Circuit': 'Monza',
                         'Compound': 'HARD', 'LapNumber': age, 'RaceLap': age, 'TyreLife': age.astype(float),
                         'GapAhead_s': np.where(age % 2 == 0, 1.2, 5.0), 'LapTimeFC_s': level + slope * age})


def test_lap_terms_pace_has_no_wear_and_keeps_the_offsets():
    laps = pd.concat([stint(d, 80.0 + i, 0.1) for i, d in enumerate('ABC')])
    laps['LapTimeFC_s'] += np.where(laps['GapAhead_s'] < 1.5, 0.3, 0.0)
    kw = dict(dirty_air_bins=[1.0, 1.5, 3.0], race_start_laps=0)
    lin = degradation.LinearDegradation(0, **kw).fit(laps)
    own = degradation.LapTermsPace(0, own=True, **kw).fit(laps)
    from_lin = degradation.LapTermsPace(0, own=False, **kw).fit(laps)
    assert (own.wear(laps) == 0).all() and (from_lin.wear(laps) == 0).all()
    assert from_lin.offsets.to_numpy() == pytest.approx(lin.offsets.to_numpy())
    assert lin.offsets.iloc[0] == pytest.approx(0.3, abs=1e-6)
