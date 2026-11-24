"""Reference tables (data/reference) and the features and study helpers built on them."""
import numpy as np
import pandas as pd
import pytest

from boxbox import features, reference as rf
from boxbox.config import load_config

CFG = {'reference': {'c_rename': {2022: {1: 0}}, 'tyre_eras': [[2022, 2025], [2026, 2026]]}}


def test_compound_index_renames_2022_c1_and_blocks_other_eras():
    idx = features.compound_index([2022, 2022, 2023, 2025, 2026, 2021], [1, 3, 1, 6, 3, 2], CFG)
    assert idx[:4].tolist() == [0.0, 3.0, 1.0, 6.0]
    assert idx[4] == 3.0                      # 2026 is its own era: index kept, compared only within 2026
    assert np.isnan(idx[5])                   # 13 inch era: no index


def test_allocation_change_against_history_mean():
    long = pd.DataFrame({'Year': [2022, 2023, 2024, 2024, 2026], 'Round': [1, 1, 1, 2, 1],
                         'Circuit': ['A', 'A', 'A', 'B', 'A'], 'Compound': ['HARD'] * 5, 'CNumber': [1.0, 1.0, 2.0, 3.0, 1.0]})
    out = features.allocation_change(long, CFG, [2022, 2023], [2024, 2026]).set_index(['Year', 'Circuit'])
    assert out.loc[(2024, 'A'), 'HistIndex'] == pytest.approx(0.5)    # 2022 C1 (now C0) and 2023 C1
    assert out.loc[(2024, 'A'), 'DeltaC'] == pytest.approx(1.5)
    assert np.isnan(out.loc[(2024, 'B'), 'DeltaC'])                  # no history at B
    assert np.isnan(out.loc[(2026, 'A'), 'DeltaC'])                  # different tyre era


def test_rating_change_and_pit_lane_time():
    c = pd.DataFrame({'Year': [2022, 2023, 2024], 'Round': [1, 1, 1], 'Circuit': ['A'] * 3, 'AsphaltAbrasion': [4, 2, 2]})
    out = features.rating_change(c, ['AsphaltAbrasion'], [2022, 2023], [2024])
    assert out['DeltaAsphaltAbrasion'].iloc[0] == pytest.approx(-1.0)
    assert features.pit_lane_time([400.0], [80.0])[0] == pytest.approx(18.0)
    assert features.pit_lane_time(400.0, 60.0) - features.pit_lane_time(400.0, 80.0) == pytest.approx(6.0)


def test_limit_corrected_stops_moves_history_to_new_limit():
    facts = pd.DataFrame({'Circuit': ['A', 'A', 'B'], 'Year': [2024, 2025, 2025], 'PitLaneLength_m': [400.0, 400.0, 300.0],
                          'PitSpeedLimit_kmh': [60.0, 80.0, 80.0]})
    stops = pd.DataFrame({'Circuit': ['A', 'B', 'C'], 'Year': [2024, 2024, 2024], 'PitLoss_s': [30.0, 20.0, 22.0]})
    out = rf.limit_corrected_stops(stops, facts, 2025)
    assert out['PitLoss_s'].tolist() == pytest.approx([24.0, 20.0, 22.0])   # A: 6 s faster at 80; B, C: no facts for 2024


def test_rated_slopes_recover_a_descriptor_effect():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        abr = rng.integers(1, 6)
        for comp, mean in [('HARD', 0.03), ('MEDIUM', 0.05), ('SOFT', 0.08)]:
            rows.append({'Circuit': f'c{i}', 'Compound': comp, 'AsphaltAbrasion': float(abr), 'Stints': 20.0,
                         'Raw': mean + 0.01 * (abr - 3) + rng.normal(0, 0.001)})
    cells = pd.DataFrame(rows)
    m = rf.RatedSlopes(['AsphaltAbrasion'], alpha=0.0).fit(cells)
    pred = m.predict(pd.DataFrame({'Compound': ['SOFT', 'HARD'], 'AsphaltAbrasion': [5.0, 1.0]}))
    sd = cells['AsphaltAbrasion'].std(ddof=0)
    assert m.beta['AsphaltAbrasion'] == pytest.approx(0.01 * sd, rel=0.05)
    assert pred == pytest.approx([0.08 + 0.02, 0.03 - 0.02], abs=0.001)   # the true slopes at abrasion 5 and 1
    mean_only = rf.RatedSlopes([]).fit(cells)
    assert mean_only.comp_mean['SOFT'] == pytest.approx(cells.loc[cells['Compound'] == 'SOFT', 'Raw'].mean(), abs=1e-6)


class _Base:
    """A minimal linear wear model: slope per compound, no offsets."""
    f = staticmethod(lambda a: a)

    def __init__(self, slopes):
        self.s = slopes

    def slopes_for(self, laps):
        return laps['Compound'].map(self.s).astype(float)

    def wear(self, laps):
        return self.slopes_for(laps) * laps['TyreLife']

    def dirty_air(self, laps):
        return pd.Series(0.0, index=laps.index)


def _laps(n_stints, slope, compound='SOFT'):
    rows = []
    for s in range(n_stints):
        for a in range(1, 11):
            rows.append({'Year': 2024, 'Round': 1, 'Driver': f'{compound}{s}', 'Stint': 1, 'Compound': compound, 'TyreLife': float(a),
                         'LapTimeFC_s': 90.0 + s * 0.1 + slope * a})
    return pd.DataFrame(rows)


def test_race_compound_shifts_measure_the_extra_slope_and_drop_thin_cells():
    laps = pd.concat([_laps(6, 0.10, 'SOFT'), _laps(2, 0.10, 'HARD')], ignore_index=True)
    out = rf.race_compound_shifts(_Base({'SOFT': 0.07, 'HARD': 0.03}), laps, min_stints=5)
    assert out['Compound'].tolist() == ['SOFT']
    assert out['Shift'].iloc[0] == pytest.approx(0.03)


def test_slope_table_replaces_only_listed_cells():
    laps = pd.concat([_laps(1, 0.1, 'SOFT'), _laps(1, 0.1, 'HARD')], ignore_index=True)
    base = _Base({'SOFT': 0.07, 'HARD': 0.03})
    t = rf.SlopeTable(base, pd.Series({(2024, 1, 'SOFT'): 0.2}))
    w = t.wear(laps)
    assert w[laps['Compound'] == 'SOFT'].to_numpy() == pytest.approx(0.2 * laps.loc[laps['Compound'] == 'SOFT', 'TyreLife'].to_numpy())
    assert w[laps['Compound'] == 'HARD'].to_numpy() == pytest.approx(0.03 * laps.loc[laps['Compound'] == 'HARD', 'TyreLife'].to_numpy())


def test_pass_logit_extra_without_extras_matches_parent():
    from boxbox.overtaking import PASS_FEATURES, PassLogit
    rng = np.random.default_rng(1)
    n = 600
    p = pd.DataFrame({'Circuit': rng.choice(['A', 'B', 'C'], n), 'Delta': rng.normal(0.3, 0.5, n), 'Gap': rng.uniform(0, 1, n),
                      'AgeDiff': rng.normal(0, 5, n), 'CompDiff': rng.choice([-1.0, 0.0, 1.0], n), 'StuckRun': rng.integers(0, 3, n).astype(float),
                      'DRSZones': rng.choice([1.0, 2.0, 3.0], n)})
    p['Pass'] = rng.uniform(size=n) < 1 / (1 + np.exp(-(-2 + 2 * p['Delta'] - p['Gap'])))
    a = PassLogit(1.0, 50).fit(p).prob(p)
    b = rf.PassLogitExtra(1.0, 50, PASS_FEATURES).fit(p).prob(p)
    assert np.allclose(a, b)
    m = rf.PassLogitExtra(1.0, 50, PASS_FEATURES, extra=['DRSZones']).fit(p)
    assert 'DRSZones' in m.coef and m.prob(p).shape == (n,)


def test_reference_tables_load_and_are_sane():
    cfg = load_config()
    c = rf.load_compounds(cfg)
    have = c.dropna(subset=['HARD'])
    assert len(c) >= 112 and len(have) >= 110
    assert ((have['HARD'] < have['MEDIUM']) & (have['MEDIUM'] < have['SOFT'])).all()
    assert have[['HARD', 'SOFT']].stack().between(0, 6).all()
    assert have['Source'].notna().all()
    assert not c.duplicated(['Year', 'Round']).any()
    r = rf.load_circuits(cfg)
    rated = r.dropna(subset=['Traction'])
    assert rated[rf.RATINGS].stack().between(1, 5).all()
    assert rated['RatingsSource'].notna().all()
    assert not r.duplicated(['Year', 'Round']).any()
    for col, src in [('PitSpeedLimit_kmh', 'PitSpeedLimitSource'), ('PitLaneLength_m', 'PitLaneLengthSource'), ('DRSZones', 'DRSSource')]:
        assert r.loc[r[col].notna(), src].notna().all(), col
    assert set(r['Street'].unique()) <= {0, 1}


def test_pit_lane_prior_and_circuit_lengths():
    facts = pd.DataFrame({'Circuit': ['A', 'A', 'B'], 'Year': [2023, 2024, 2024], 'PitLaneLength_m': [300.0, np.nan, 500.0]})
    assert rf.circuit_lengths(facts).to_dict() == {'A': 300.0, 'B': 500.0}
    lane = pd.Series({'A': 13.5, 'B': 22.5, 'C': 18.0})
    med = 5.0 + lane                                     # loss = 5 s plus the time at the limit
    m = rf.PitLanePrior().fit(med, lane)
    assert (m.a, m.b) == pytest.approx((5.0, 1.0))
    few = rf.PitLanePrior().fit(med[:2], lane[:2])       # under three circuits: their mean, no slope
    assert few.b == 0.0 and few.predict([30.0])[0] == pytest.approx(med[:2].mean())


def test_pirelli_strategies_load():
    s = rf.load_strategies(load_config())
    assert len(s) >= 90 and not s.duplicated(['Year', 'Round']).any()
    named = s.dropna(subset=['Stops'])
    assert named['Stops'].isin([1, 2, 3]).all() and named['Source'].notna().all()
