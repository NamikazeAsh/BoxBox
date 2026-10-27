"""Home: what the data covers, how the models score, and the way into every other page."""
import plotly.graph_objects as go
import streamlit as st

import data
import theme
from boxbox import dashboard
from components import page_header, radio, section, stats, tower
from registry import PAGES

yrs = data.seasons()
cfg = data.cfg()
events = data.processed('events')
deg = dashboard.degradation_scores(data.model_output('degradation_test_preds')).set_index('Model')['MAE_s']
op = data.model_output('outcome_test_preds')
out = dashboard.outcome_scores(op).set_index('Model')['MAE']
ridge = next(m for m in out.index if m != 'grid')
test_year = int(op['Year'].max())

page_header('Paddock', f'Seasons {yrs[0]} to {yrs[-1]}',
            f'{", ".join(map(str, cfg["holdout_seasons"]))} held out for the shift study')

stats([
    {'value': f'{len(events)}', 'label': f'races pulled and cleaned, {yrs[0]} to {yrs[-1]}'},
    {'value': f'{100 * dashboard.gain(deg["simulator average"], deg["constant pace"]):.0f}', 'unit': '%', 'key': True,
     'label': f'less lap time error than assuming no tyre wear ({test_year} test season)'},
    {'value': f'{100 * dashboard.gain(out[ridge], out["grid"]):.0f}', 'unit': '%',
     'label': f'fewer places off than predicting the grid order ({test_year} test season)'},
])

left, right = st.columns([1.1, 1], gap='large')
with left:
    rnd = int(op.loc[op['Year'] == test_year, 'Round'].max())
    name = events.loc[(events['Year'] == test_year) & (events['Round'] == rnd), 'EventName'].iloc[0]
    section('Predicted finish', f'{name} {test_year}, top ten')
    rp = dashboard.race_prediction(op, test_year, rnd, ridge).head(10)
    tower([{'code': r.Driver, 'lineage': r.Lineage, 'team': r.Team, 'right': f'finished P{int(r.Finish)}',
            'delta': 'exact' if r.Err == 0 else f'{int(r.Err):+d}',
            'delta_class': 'bb-purple' if r.Err == 0 else 'bb-green' if abs(r.Err) <= 2 else 'bb-yellow'}
           for r in rp.itertuples()])
    st.caption('Purple: exact place. Green: within two places. Yellow: further off.')
with right:
    section('Explore')
    for e in PAGES:
        if e.key in ('paddock', 'style_guide') or not cfg['dashboard']['pages'].get(e.key):
            continue
        if e.status == 'ready':
            st.page_link(f'views/{e.key}.py', label=f'**{e.title}**', icon=e.icon)
        else:
            st.html(f'<div style="padding:4px 0 0 2px;opacity:.45"><b>{e.title}</b></div>')
        st.caption(e.blurb)

flags = data.race_flags()
summary = dashboard.season_summary(events, data.processed('pit_loss'), flags)
last = summary.iloc[-1]
radio(f'Safety car came out in {int(last.SafetyCar)} of {int(last.Races)} races in {int(last.Year)}, '
      f'and a virtual safety car in {int(last.VSC)}. Every strategy needs a plan for it.')

c1, c2 = st.columns(2, gap='large')
with c1:
    section('Seasons', 'races, and races with a safety car')
    fig = go.Figure()
    fig.add_bar(x=summary['Year'], y=summary['Races'], name='Races', marker_color=theme.rgba(theme.muted(), 0.35),
                hovertemplate='%{x}: %{y} races<extra></extra>')
    fig.add_bar(x=summary['Year'], y=summary['SafetyCar'], name='Safety car', marker_color=theme.RED,
                hovertemplate='%{x}: safety car in %{y} races<extra></extra>')
    fig.update_layout(barmode='overlay', bargap=0.35)
    fig.update_xaxes(dtick=1)
    theme.plot(fig, height=320, key='seasons')
with c2:
    section('Pit loss by circuit', 'median green flag stop')
    year = st.selectbox('Season', yrs[::-1], key='pit_year')
    p = dashboard.pit_loss_by_circuit(data.processed('pit_loss'), year)
    fig = go.Figure(go.Bar(x=p['PitLoss_s'], y=p['Circuit'], orientation='h', marker_color=theme.rgba(theme.muted(), 0.55),
                           hovertemplate='%{y}: %{x:.1f} s<extra></extra>'))
    fig.update_xaxes(title_text='seconds')
    fig.update_yaxes(autorange='reversed')
    theme.plot(fig, height=max(320, 22 * len(p)), key='pitloss')
