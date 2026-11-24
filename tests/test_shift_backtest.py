import pandas as pd
import yaml

from boxbox.simulator import backtest as bt

CFG = {'training_seasons': {'shift_study_train': [2022, 2023, 2024, 2025], 'shift_study_test': [2026]},
       'simulator': {'compound_pace': {'evolution': 'stint_start'}}, 'neutral': {'lap_share': 0.75}}


def test_shift_fold_trains_on_2022_to_2025_and_scores_2026(monkeypatch):
    import boxbox.config as c
    monkeypatch.setattr(c, 'seasons_for', lambda cfg, comp: cfg['training_seasons'][comp])
    assert bt.shift_folds(CFG) == [([2022, 2023, 2024, 2025], 2026)]


def test_config_record_holds_the_simulator_and_neutral_settings():
    rec = yaml.safe_load(bt.config_record(CFG))
    assert rec['simulator']['compound_pace']['evolution'] == 'stint_start' and rec['neutral']['lap_share'] == 0.75


def test_setup_check_reports_a_race_that_would_be_skipped(monkeypatch):
    def run(inputs, t, pre, year, rnd, circuit, ob, policy, variants, draws, seed, cfg):
        if circuit == 'Madrid':
            raise KeyError('Madrid')
        return None, None
    monkeypatch.setattr(bt, 'run_race', run)
    inputs = type('I', (), {'refs': {'Monza': 1.0}, 'pit': type('P', (), {'pools': {'Monza': [20.0]}})()})()
    races = pd.DataFrame({'Round': [1, 2], 'Circuit': ['Monza', 'Madrid']})
    bases = pd.DataFrame({'Year': [2026], 'Round': [1], 'Driver': ['VER'], 'Base': [80.0]})
    out = bt.setup_check(inputs, None, {}, None, 2026, races, bases, [], {})
    assert out['Status'].tolist()[0] == 'ok' and out['Status'].iloc[1].startswith('KeyError')
    assert out['NewCircuit'].tolist() == [False, True] and out['PitLossHistory'].tolist() == [True, False]
