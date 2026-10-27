"""Every component with sample data, for design work (dashboard.pages.style_guide in config.yaml)."""
import pandas as pd
import streamlit as st

import theme
from components import bars, page_header, radio, score_table, section, stats, stint_bars, tower, tyre

page_header('Style guide', 'Design', 'Sample data, not results')

section('Stats', 'one red key number per page')
stats([{'value': '151', 'label': 'a plain number'}, {'value': '29', 'unit': '%', 'key': True, 'label': 'the key number'},
       {'value': '2.19', 'label': 'a decimal'}])

section('Radio card', 'the one boxed element')
radio('Box box. Undercut is on, the car ahead is on old hards.')

c1, c2 = st.columns(2, gap='large')
with c1:
    section('Timing tower')
    tower([{'code': 'NOR', 'lineage': 'McLaren', 'team': 'McLaren', 'right': 'P1', 'delta': 'exact', 'delta_class': 'bb-purple'},
           {'code': 'VER', 'lineage': 'Red Bull Racing', 'team': 'Red Bull Racing', 'right': 'P3', 'delta': '-1', 'delta_class': 'bb-green'},
           {'code': 'LEC', 'lineage': 'Ferrari', 'team': 'Ferrari', 'right': 'P7', 'delta': '-4', 'delta_class': 'bb-yellow'}])
with c2:
    section('Stint bars')
    plan = pd.DataFrame({'Driver': ['NOR', 'NOR', 'VER', 'VER', 'VER'], 'StartLap': [1, 23, 1, 15, 36],
                         'EndLap': [22, 57, 14, 35, 57], 'Compound': ['MEDIUM', 'HARD', 'SOFT', 'MEDIUM', 'HARD']})
    stint_bars(plan, ['NOR', 'VER'], 57)
    st.html('<p>Tyres: ' + ' '.join(tyre(c) for c in theme.COMPOUND_NAMES) + '</p>')

section('Score table', 'purple marks the best')
score_table([{'Model': 'Simulator average', 'Tag': 'used by the simulator', 'MAE': 0.578},
             {'Model': 'Linear', 'MAE': 0.594}, {'Model': 'No tyre wear', 'Tag': 'baseline', 'MAE': 0.819}],
            [('Model', 'Model', None), ('MAE', 'Error, s per lap', '{:.3f}')], best=['MAE'], base={'No tyre wear'})

section('Bars')
bars([('1.0 to 1.5 s', 0.26), ('1.5 to 2.0 s', 0.18), ('2.0 to 2.5 s', 0.12)])

section('Streamlit widgets', 'themed by .streamlit/config.toml')
a, b = st.columns(2)
a.segmented_control('Compound', theme.COMPOUNDS, format_func=theme.COMPOUND_NAMES.get, default='MEDIUM', key='compound')
b.slider('Stint length', 5, 45, 20)
st.tabs(['First', 'Second', 'Third'])
