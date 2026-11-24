import numpy as np
import pandas as pd
import pytest

from boxbox import dashboard, overtaking as ot, overtaking_view as ov
from boxbox.config import ROOT, load_config


def synthetic_pairs(n=4000, seed=0):
    """Close pairs at two circuits: passing easier at Sakhir, harder from further back, easier when faster."""
    rng = np.random.default_rng(seed)
    p = pd.DataFrame({'Circuit': np.where(rng.random(n) < 0.5, 'Sakhir', 'Monaco'), 'Year': 2024, 'Round': 1,
                      'Gap': rng.uniform(0, 1, n), 'Delta': rng.normal(0.3, 0.6, n), 'AgeDiff': rng.normal(0, 6, n),
                      'CompDiff': rng.integers(-1, 2, n).astype(float), 'StuckRun': rng.integers(0, 5, n).astype(float)})
    z = -1.5 + 1.5 * p['Delta'] - 3 * p['Gap'] + np.where(p['Circuit'] == 'Sakhir', 1.0, -2.0) - 0.15 * p['StuckRun']
    p['Pass'] = rng.random(n) < 1 / (1 + np.exp(-z))
    return p


def test_grid_varies_one_feature_and_follows_the_fit():
    m = ot.PassLogit(1.0, 50).fit(synthetic_pairs())
    d = ov.grid(m, 'Sakhir', Delta=np.array([0.0, 0.5, 1.0]), Gap=0.6, AgeDiff=0.0, CompDiff=0.0, StuckRun=0.0)
    assert d['Prob'].is_monotonic_increasing and (d['Gap'] == 0.6).all()
    hard = ov.grid(m, 'Monaco', Delta=np.array([0.5]), Gap=0.6, AgeDiff=0.0, CompDiff=0.0, StuckRun=0.0)['Prob'].iloc[0]
    assert hard < d['Prob'].iloc[1]
    stuck = ov.grid(m, None, StuckRun=np.arange(0, 5.0), Delta=0.5, Gap=0.6, AgeDiff=0.0, CompDiff=0.0)['Prob']
    assert stuck.is_monotonic_decreasing
    assert 0 < ov.circuit_rate(m, ['Monaco'])[0] < ov.circuit_rate(m, ['Sakhir'])[0] < 1


def test_calibration_bins_cover_every_pair():
    p = synthetic_pairs(1000)
    c = ov.calibration(p, np.linspace(0, 1, len(p)), 10)
    assert len(c) == 10 and c['Pairs'].sum() == len(p) and c['Pred'].is_monotonic_increasing


def test_page_numbers_from_the_tables():
    circuits = pd.DataFrame({'Circuit': ['Sakhir', 'Monaco'], 'ProbRef': [0.2, 0.02]})
    assert dashboard.pass_ratio(circuits, 'Sakhir', 'Monaco') == pytest.approx(10)
    coefs = pd.DataFrame({'Feature': ['StuckRun', 'AgeDiff'], 'Coef': [np.log(0.86), np.log(1.05)]})
    assert dashboard.odds_change(coefs, 'StuckRun') == pytest.approx(-0.14)
    assert dashboard.odds_change(coefs, 'AgeDiff', 2) == pytest.approx(1.05 ** 2 - 1)


def test_spread_labels_keeps_order_and_spacing():
    y = dashboard.spread_labels([0.1, 0.11, 0.5, 0.12], 0.04)
    assert list(np.argsort(y)) == [0, 1, 3, 2] and y[2] == 0.5 and y[0] == 0.1
    assert np.diff(np.sort(y)).min() >= 0.04 - 1e-12
    top = dashboard.spread_labels([0.9, 0.95, 0.97], 0.04, top=0.97)
    assert top.max() <= 0.97 and np.diff(top).min() >= 0.04 - 1e-12


BUILT = (load_config()['paths']['models'] / 'overtaking_meta.parquet').exists()


@pytest.mark.skipif(not BUILT, reason='needs python scripts/build_overtaking.py')
def test_page_loads():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app' / 'main.py'), default_timeout=120).run()
    at.switch_page('views/overtaking.py').run()
    assert not at.exception and not at.error
