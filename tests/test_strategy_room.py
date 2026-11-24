import numpy as np
import pandas as pd
import pytest

from boxbox import dashboard, strategy_room as sr
from boxbox.config import ROOT, load_config
from boxbox.simulator import strategy as st
from test_simulator import CFG, StubInputs, StubLap, StubPit

SEARCH_CFG = {'simulator': {**CFG['simulator'], 'race': {**CFG['simulator']['race'], 'tie_places': 0.25}}}


class NoisyLap(StubLap):
    class noise:
        @staticmethod
        def draw_slopes(rng, draws, races):
            return np.zeros((draws, len(races)))

        @staticmethod
        def draw_laps(rng, shape):
            return rng.normal(0, 0.6, shape)


class NoisyPit(StubPit):
    def sample(self, circuit, rng, size):
        return rng.normal(20.0, 2.0, size)


class NoisyInputs(StubInputs):
    def __init__(self):
        super().__init__(threshold=0.3)
        self.lap, self.pit = NoisyLap(), NoisyPit()


def small_field(draws=24, seed=3):
    """Three cars on a 30 lap race: A and C one stop rivals, B the car searched, close on pace."""
    rng = np.random.default_rng(1)
    cars = pd.DataFrame({'Driver': ['A', 'B', 'C'], 'Team': 'T', 'Lineage': 'T', 'Grid': [1.0, 2.0, 3.0], 'Finish': [1.0, 2.0, 3.0],
                         'StartCompound': 'MEDIUM', 'StartAge': 1.0})
    bases = 80.0 + rng.normal(0, 0.15, (draws, 3))
    rivals = pd.concat([st.strategy_stints(('MEDIUM', 'HARD'), (12,), 'A'), st.strategy_stints(('MEDIUM', 'HARD'), (16,), 'C')])
    return sr.Field(2025, 1, 'Monza', 30, cars, bases, rivals.reset_index(drop=True), seed)


def real_stints():
    return st.strategy_stints(('MEDIUM', 'HARD'), (13,), 'B')


def test_tidy_keeps_one_row_per_strategy_and_marks_the_real_one():
    best = pd.DataFrame({'Extra': [False, False, True], 'Compounds': ['M-H', 'M-H', 'M-H'], 'StopLaps': ['20', '22', '22'],
                         'MeanPlace': [2.0, 2.1, 2.1], 'Tier': ['best', 'near tie', 'near tie']})
    t = sr.tidy(best)
    assert list(t['Rank']) == [1, 2] and list(t['Real']) == [False, True] and 'Extra' not in t


def test_label_and_parse_round_trip():
    comps, stops = ('MEDIUM', 'SOFT', 'HARD'), (18, 37)
    assert sr.label(comps, stops) == ('M-S-H', '18, 37')
    assert sr.parse(*sr.label(comps, stops)) == (comps, stops)


def test_check_strategy_flags_each_problem():
    assert sr.check_strategy(['MEDIUM', 'HARD'], [20], 50, 1) == []
    assert sr.check_strategy(['MEDIUM', 'MEDIUM'], [20], 50, 1)            # two compound rule
    assert sr.check_strategy(['MEDIUM', 'HARD', 'SOFT'], [30, 20], 50, 1)  # stops out of order
    assert sr.check_strategy(['MEDIUM', 'HARD'], [48], 50, 5)              # last stint too short
    assert sr.check_strategy(['MEDIUM', 'HARD'], [50], 50, 1)              # stop on the last lap


def test_summary_pairs_against_the_best():
    m = sr.summary([1, 2, 3, 4], best_place=[2, 2, 2, 2])
    assert m['MeanPlace'] == 2.5 and m['Podium'] == 0.75 and m['Win'] == 0.25
    assert m['DeltaPlace'] == 0.5 and m['PBeatsBest'] == pytest.approx(0.25 + 0.5 * 0.25)


def test_builder_race_lands_on_the_search_row():
    """Common random numbers: the same strategy raced again against the field gives the search's numbers."""
    fld = small_field()
    table = sr.search(NoisyInputs(), fld, real_stints(), 'B', SEARCH_CFG, use_caps=False)
    assert table['Real'].sum() == 1 and table['Rank'].is_monotonic_increasing
    places = sr.race_places(NoisyInputs(), fld, 'B', table, 2)              # raises if any mean differs
    assert set(places['Rank']) == set(sr.chart_rows(table, 2))
    row = table.iloc[1]
    place, _ = sr.race_strategy(NoisyInputs(), fld, 'B', *sr.parse(row['Compounds'], row['StopLaps']))
    assert place.mean() == pytest.approx(row['MeanPlace']) and np.quantile(place, 0.9) == pytest.approx(row['Place90'])


def test_field_survives_the_saved_tables():
    fld = small_field()
    cars, bases, rivals, status = sr.field_tables(fld)
    meta = pd.DataFrame([{'Year': 2025, 'Round': 1, 'Circuit': 'Monza', 'RaceLaps': 30, 'Seed': 3}])
    back = sr.field_from_tables({'meta': meta, 'cars': cars, 'bases': bases, 'rivals': rivals}, 2025, 1)
    assert status is None and back.status is None and back.kw == {}
    assert back.drivers == fld.drivers and np.array_equal(back.bases, fld.bases) and back.seed == fld.seed
    pd.testing.assert_frame_equal(back.rivals, fld.rivals[st.STINTS].reset_index(drop=True), check_dtype=False)


def test_a_status_is_saved_and_moves_the_cars_stop_as_the_search_does():
    fld = small_field(draws=4)
    fld.status = np.zeros((4, 31), np.int8)
    fld.status[1, 10:13] = 1                              # a safety car on laps 10 to 12 of draw 1 only
    cars, bases, rivals, status = sr.field_tables(fld)
    meta = pd.DataFrame([{'Year': 2025, 'Round': 1, 'Circuit': 'Monza', 'RaceLaps': 30, 'Seed': 3, 'Neutral': True}])
    back = sr.field_from_tables({'meta': meta, 'cars': cars, 'bases': bases, 'rivals': rivals, 'status': status}, 2025, 1)
    assert np.array_equal(back.status, fld.status) and 'status' in back.kw
    cfg = {'simulator': {'race': {'neutral_react_window': 3, 'min_stint_laps': 5}}}
    stop = sr._arrays(back, 'B', ('MEDIUM', 'HARD'), (12,), cfg)['Stop'][:, 1, 0]
    assert list(stop) == [12, 10, 12, 12]                 # boxes under the safety car in draw 1 only


def test_races_refuses_an_unknown_kind():
    with pytest.raises(ValueError):
        sr.races(pd.DataFrame(columns=['Year', 'Round', 'Circuit', 'WetRace', 'TrackStatus']), [2025], 'all')


def test_place_shares_sum_to_one_over_every_place():
    p = pd.DataFrame({'Rank': [1, 1, 1, 2, 2], 'Place': [1, 1, 3, 2, 2]})
    s = dashboard.place_shares(p, 4)
    assert len(s) == 8 and np.allclose(s.groupby('Rank')['Share'].sum(), 1)
    assert s.set_index(['Rank', 'Place']).loc[(1, 1), 'Share'] == pytest.approx(2 / 3)


def test_strategy_plan_and_words():
    plan = dashboard.strategy_plan([('Best', 'M-H', '25')], 57)
    assert list(plan['StartLap']) == [1, 26] and list(plan['EndLap']) == [25, 57] and list(plan['Compound']) == ['MEDIUM', 'HARD']
    assert dashboard.describe_strategy('M-H', '25') == 'one stop on lap 25, Medium then Hard'
    assert dashboard.describe_strategy('M-S-S', '21, 41') == 'two stops on laps 21 and 41, Medium, Soft then Soft'


def test_verdict_reads_the_real_row():
    t = pd.DataFrame({'Rank': [1, 2, 3], 'MeanPlace': [2.0, 2.1, 3.0], 'Tier': ['best', 'near tie', 'worse'], 'Real': [False, False, True]})
    v = dashboard.room_verdict(t)
    assert v['real_rank'] == 3 and v['real_tier'] == 'worse' and v['ties'] == 1 and v['gain'] == pytest.approx(1.0)


BUILT = (load_config()['paths']['models'] / 'strategy_room_search.parquet').exists()


@pytest.mark.skipif(not BUILT, reason='needs python scripts/build_strategy_room.py')
def test_page_loads_and_the_builder_races():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app' / 'main.py'), default_timeout=120).run()
    at.switch_page('views/strategy_room.py').run()
    assert not at.exception
    next(b for b in at.button if b.label == 'Race it').click().run()   # on the strategy the driver ran
    assert not at.exception and not at.warning
    assert any('Rank among strategies' in h.proto.body for h in at.get('html'))   # the result tiles


def test_room_overview_lists_each_driver_with_what_they_ran_and_the_best():
    import pandas as pd
    from boxbox import dashboard
    search = pd.DataFrame({'Year': 2025, 'Round': 1, 'Driver': ['A', 'A', 'B', 'B'], 'Rank': [1, 2, 1, 2],
                           'Compounds': ['M-H', 'M-H-H', 'S-H', 'S-M'], 'StopLaps': ['20', '15, 35', '18', '22'],
                           'MeanPlace': [2.0, 2.5, 5.0, 5.4], 'Tier': ['best', 'near tie', 'best', 'worse'], 'Real': [False, True, True, False]})
    cars = pd.DataFrame({'Year': 2025, 'Round': 1, 'Driver': ['A', 'B'], 'Team': ['X', 'Y'], 'Grid': [2, 1], 'Finish': [1, 2], 'Searched': True})
    o = dashboard.room_overview(search, cars, 2025, 1)
    assert o['Driver'].tolist() == ['A', 'B'] and o.loc[0, 'Ran'] == 'M H H, laps 15, 35' and o.loc[0, 'Gain'] == 0.5
    assert o.loc[1, 'Tier'] == 'best' and o.loc[1, 'Best'] == 'S H, lap 18'
