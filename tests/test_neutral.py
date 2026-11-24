import numpy as np
import pandas as pd

from boxbox import features
from boxbox import neutral as N

H = {'shrink': 2, 'lap1': True, 'lap1_shrink': 5, 'open_laps': 1, 'restart_laps': 0, 'frac_bins': [], 'season_decay': 1.0}
CFG = {'neutral': {'lap_share': 0.75, 'exclude_races': [], 'hazard': {'sc': H, 'vsc': H}, 'multiplier_prior': 1.0,
                   'duration': 'pool', 'max_period_laps': 20, 'conversion': True, 'draws': 2000, 'seed': 0,
                   'effects': {'close_alpha': 0.05, 'close_max': 6, 'min_gaps': 50, 'neighbours': 2, 'min_stayout': 3,
                               'vsc_gap_max_s': 10.0}},
       'pit_loss': {'min_s': 5.0, 'max_s': 60.0}}


def race(rnd, statuses, circuit='Monza', cars=4, year=2024):
    """Lap rows of one race: statuses has one entry per lap, a string for every car or a list with one per car."""
    rows = []
    for t, st in enumerate(statuses, start=1):
        for i, s in enumerate(st if isinstance(st, list) else [st] * cars):
            rows.append({'Year': year, 'Round': rnd, 'Driver': f'D{i}', 'LapNumber': float(t), 'TrackStatus': s,
                         'Circuit': circuit, 'WetRace': False})
    return pd.DataFrame(rows)


def test_status_codes_follow_the_share_of_cars():
    laps = race(1, ['1', ['4', '4', '4', '1'], ['4', '1', '1', '1'], ['6', '6', '67', '1'], ['45', '45', '5', '1'],
                    ['4', '6', '1', '1'], ''])
    half = features.neutral_laps(laps, 0.5).set_index('LapNumber')['Neutral']
    assert half.tolist() == [0, 1, 0, 2, 1, 1, 0]             # lap 6: a quarter each, half neutral: the tie goes to SC
    anyone = features.neutral_laps(laps, 0).set_index('LapNumber')['Neutral']
    assert anyone.tolist() == [0, 1, 1, 2, 1, 1, 0]           # the first rule: one car is enough
    assert half.dtype == np.int8


def test_race_status_rebuilds_a_known_race():
    laps = pd.concat([race(1, ['1', '4', '4', '41', '1', '6', '67', '1']), race(2, ['1'] * 5)])
    assert N.race_status(laps, 2024, 1).tolist() == [0, 0, 1, 1, 1, 0, 2, 2, 0]
    assert N.race_status(laps, 2024, 2).tolist() == [0] * 6


def training(L=40):
    """10 busy races (SC 3 laps, SC 5 laps, a VSC that turns into a SC) and 10 calm, all green, races."""
    busy = {t: '4' for t in [10, 11, 12, 20, 21, 22, 23, 24, 35, 36, 37]} | {33: '6', 34: '6'}
    out = [race(r, [busy.get(t, '1') for t in range(1, L + 1)], 'Busy') for r in range(1, 11)]
    out += [race(r, ['1'] * L, 'Calm') for r in range(11, 21)]
    return pd.concat(out, ignore_index=True)


def test_periods_and_risk_table():
    codes = N.race_codes(training())
    p = N.periods({k: v for k, v in codes.items() if k[1] == 1})
    assert p[['Code', 'Start', 'Laps']].values.tolist() == [[1, 10, 3], [1, 20, 5], [2, 33, 2], [1, 35, 3]]
    assert p['Converted'].tolist() == [False, False, True, False]
    r = N.risk_table({(2024, 1): codes[(2024, 1)]})
    assert 11 not in r['Lap'].values and 13 not in r['Lap'].values and 14 in r['Lap'].values   # the lap after a period is not at risk
    assert r.set_index('Lap').loc[14, 'Since'] == 1 and r.set_index('Lap').loc[15, 'Since'] == 2


def test_sample_shape_codes_durations_and_no_overlap():
    m = N.NeutralModel(CFG).fit(training(), [2024], effects=False)
    assert 0 < m.p_convert <= 1
    s = m.sample('Busy', 40, 600, np.random.default_rng(1))
    assert s.shape == (600, 41) and s.dtype == np.int8
    assert set(np.unique(s)) <= {0, 1, 2} and (s[:, 0] == 0).all()
    pool = {N.SC: set(m.duration[N.SC].values), N.VSC: set(m.duration[N.VSC].values)}
    seen = 0
    for row in s:
        p = N.periods({(0, 0): row})
        for prev, nxt in zip(p.itertuples(), list(p.itertuples())[1:]):
            gap = nxt.Start - prev.End - 1
            assert gap >= 1 or (prev.Code == N.VSC and nxt.Code == N.SC)     # back to back only when a VSC turns into a SC
        done = p[~p['Censored']]
        assert all(n in pool[c] for c, n in zip(done['Code'], done['Laps']))
        seen += len(done)
    assert seen > 300


def test_busy_circuit_gets_more_safety_cars_than_calm():
    m = N.NeutralModel(CFG).fit(training(), [2024], effects=False)
    busy, calm = m.race_probs('Busy', 40), m.race_probs('Calm', 40)
    assert busy['p_sc'] > calm['p_sc'] + 0.3 and busy['sc_laps'] > calm['sc_laps']
    new = m.race_probs('Elsewhere', 40)                      # no history: the all circuit rate, in between
    assert calm['p_sc'] < new['p_sc'] < busy['p_sc']
    assert m.circuit_group('Busy') == 'high' and m.circuit_group('Elsewhere') == 'new'


def test_global_hazard_ignores_the_circuit():
    cfg = {**CFG, 'neutral': {**CFG['neutral'], 'hazard': {k: {**H, 'shrink': None, 'lap1_shrink': None} for k in ('sc', 'vsc')}}}
    m = N.NeutralModel(cfg).fit(training(), [2024], effects=False)
    assert np.allclose(m.hazard[N.SC].lap_rates('Busy', 40), m.hazard[N.SC].lap_rates('Calm', 40))


def test_pit_loss_under_a_safety_car_against_cars_staying_out():
    """Six cars, 1 s apart, 90 s green laps and 130 s safety car laps 3 and 4. D0 stops under the safety car
    losing 12 s over its in and out lap, D5 stops on green laps 5 and 6 losing 22 s: the ratio is 12 / 22."""
    rows = []
    for i in range(6):
        t, stint = float(i), 1
        for lap in range(1, 8):
            lt = 130.0 if lap in (3, 4) else 90.0
            pit_in, pit_out = (i == 0 and lap == 3) or (i == 5 and lap == 5), (i == 0 and lap == 4) or (i == 5 and lap == 6)
            if pit_in or pit_out:
                lt += 6.0 if i == 0 else 11.0
            stint += pit_out
            t += lt
            rows.append({'Year': 2024, 'Round': 1, 'Driver': f'D{i}', 'LapNumber': float(lap), 'Stint': float(stint),
                         'TrackStatus': '4' if lap in (3, 4) else '1', 'LapTime_s': lt, 'Time_s': t, 'PitIn': pit_in,
                         'PitOut': pit_out, 'IsClean': lt == 90.0 and lap > 1, 'GapAhead_s': np.nan if i == 0 else 1.0,
                         'Circuit': 'Monza', 'WetRace': False})
    laps = pd.DataFrame(rows)
    a = N.annotate(laps, N.race_codes(laps))
    s = N.neutral_pit_loss(a, CFG['neutral']['effects'], CFG['pit_loss'])
    s = s[s['Use']].set_index('Driver')
    assert np.isclose(s.loc['D5', 'Loss_s'], 22.0) and s.loc['D5', 'Kind'] == 0
    assert np.isclose(s.loc['D0', 'Loss_s'], 12.0) and s.loc['D0', 'Kind'] == N.SC
    assert np.isclose(s.loc['D0', 'Ratio'], 12.0 / 22.0)


def test_scores():
    assert np.isclose(N.crps(np.zeros(100), 0.0), 0.0) and np.isclose(N.crps(np.zeros(100), 2.0), 2.0)
    assert N.ks_distance([1, 2, 3], [1, 2, 3]) == 0 and N.ks_distance([1, 2], [5, 6]) == 1
    st = N.race_draw_stats(np.array([[0, 1, 1, 0, 2, 1, 0], [0, 0, 0, 0, 0, 0, 0]], np.int8))
    assert st['SCLaps'].tolist() == [3, 0] and st['SCPeriods'].tolist() == [2, 0] and st['VSCPeriods'].tolist() == [1, 0]
