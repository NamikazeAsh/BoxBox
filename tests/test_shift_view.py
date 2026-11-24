import numpy as np
import pandas as pd
import pytest

from boxbox import shift_view as sv


def synthetic_preds(seed=0):
    """Anchored forecast errors for three models on two rounds and two compounds."""
    rng = np.random.default_rng(seed)
    n = 200
    base = pd.DataFrame({'Year': 2026, 'Round': np.repeat([1, 2], n // 2), 'Circuit': np.repeat(['Sakhir', 'Jeddah'], n // 2),
                         'Compound': np.tile(['SOFT', 'MEDIUM'], n // 2)})
    return {'constant pace': base.assign(Err_s=rng.normal(-0.5, 1.0, n)),
            'linear': base.assign(Err_s=rng.normal(-0.1, 0.7, n)),
            'simulator wear': base.assign(Err_s=rng.normal(-0.1, 0.7, n))}


def tables():
    rows, other = [], []
    for split, test, shift_by in [('reference', 2025, 0.0), ('shift', 2026, 0.3)]:
        p = {k: v.assign(Err_s=v['Err_s'] - shift_by, Year=test) for k, v in synthetic_preds(test).items()}
        rows.append(sv.scores_rows(p, 'study', split, test).assign(Train='2022 to 2024'))
        other += sv._other('study', split, test, 'predictor', {'MAE': 2.1, 'GridMAE': 2.5 if test == 2026 else 2.4})
        other += sv._other('study', split, test, 'pit loss', {'MAE': 0.7 if test == 2025 else 1.6, 'GlobalMAE': 1.5})
    return {'scores': pd.concat(rows, ignore_index=True), 'other': pd.DataFrame(other)}


def test_scores_rows_match_the_error_table():
    p = synthetic_preds()
    s = sv.scores_rows(p, 'study', 'shift', 2026)
    lin = s[(s['Model'] == 'linear') & (s['Compound'] == 'all')].iloc[0]
    assert lin['MAE_s'] == pytest.approx(p['linear']['Err_s'].abs().mean())
    assert lin['Gain'] == pytest.approx(1 - lin['MAE_s'] / p['constant pace']['Err_s'].abs().mean())
    assert set(s['Compound']) == {'all', 'SOFT', 'MEDIUM'} and (s.loc[s['Model'] == 'constant pace', 'Gain'] == 0).all()


def test_headline_and_verdicts():
    t = tables()
    h = sv.headline(t)
    assert h['mae_change'] == pytest.approx(h['shift']['MAE_s'] / h['ref']['MAE_s'] - 1)
    assert h['predictor']['shift']['Gain'] == pytest.approx(1 - 2.1 / 2.5)
    v = {x['label']: x for x in sv.verdicts(t)}
    assert v['Pit loss']['value'] == 'Breaks' and v['Finishing predictor']['value'] == 'Holds'
    rows = sv.score_rows(t['scores'], 'study')
    assert [r['Model'] for r in rows] == ['constant pace', 'linear', 'simulator wear'] and 'Bias 2026' in rows[0]


def test_round_series_and_blocks():
    p = synthetic_preds()
    rec = {'frozen': p['linear'], 'season scaled': p['simulator wear'], 'rolling refit': p['linear']}
    r = sv.rounds_rows(rec, 'study', 'shift', 2026)
    s = sv.round_series(r, 'study')
    assert list(s) == sv.VERSIONS and list(s['frozen']['Circuit']) == ['Sakhir', 'Jeddah']
    last = s['frozen']['CumMAE_s'].iloc[-1]
    assert last == pytest.approx(p['linear']['Err_s'].abs().mean())
    b = sv.blocks_rows(rec, 'study', 'shift', 2026)
    assert b.loc[(b['Version'] == 'frozen') & (b['Block'] == 'all'), 'Value'].iloc[0] == pytest.approx(last)


def test_wet_costs_follow_the_square_law():
    wet = pd.DataFrame({'Year': [2023, 2024, 2025], 'Round': [1, 2, 3], 'Circuit': ['A', 'B', 'C'], 'Start': [1, 5, 9],
                        'End': [3, 6, 10], 'GapTrend_s': [-2.0, 4.0, 1.0], 'Found': [True, True, False]})
    costs, med = sv.wet_costs(wet, 2)
    assert list(costs['Cost_s']) == [8.0, 4.0] and med == pytest.approx(3.0 * 4 / 2)


def _built():
    from boxbox.config import load_config
    d = load_config()['paths']['models']
    return all((d / f'shift_{k}.parquet').exists() for k in sv.TABLES)


@pytest.mark.skipif(not _built(), reason='needs python scripts/build_shift_page.py')
def test_page_loads_and_switches_the_wet_call():
    from streamlit.testing.v1 import AppTest
    from boxbox.config import ROOT
    at = AppTest.from_file(str(ROOT / 'app' / 'main.py'), default_timeout=120).run()
    at.switch_page('views/shift_study.py').run()
    assert not at.exception and not at.error
    html = ' '.join(h.proto.body for h in at.get('html'))
    assert 'Tyre wear forecast' in html and 'Method check' in html and 'Wet call' in html
    at.button_group[0].set_value(3).run() if at.button_group else None
    assert not at.exception
