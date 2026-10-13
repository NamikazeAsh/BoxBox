import pytest

from boxbox.config import load_config, seasons_for


def test_real_config_has_no_holdout_leak():
    cfg = load_config()
    for component in cfg['training_seasons']:
        seasons_for(cfg, component)


def test_holdout_season_is_rejected_outside_shift_study():
    cfg = {'holdout_seasons': [2026],
           'training_seasons': {'degradation': [2025, 2026], 'shift_study_test': [2026]}}
    with pytest.raises(ValueError):
        seasons_for(cfg, 'degradation')
    assert seasons_for(cfg, 'shift_study_test') == [2026]
