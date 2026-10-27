"""Finishing position predictor: one race at a time on the test season, against the grid order."""
import plotly.graph_objects as go
import streamlit as st

import data
import theme
from boxbox import dashboard, evaluate, outcome
from components import circuit_map, page_header, section, stats, stint_bars, tower

op = data.model_output('outcome_test_preds')
events = data.processed('events')
ridge = next(m for m in op['Model'].unique() if m != 'grid')
train, test_year = evaluate.season_splits(data.training('outcome'))['test']
races = dashboard.outcome_races(op, events)

page_header('Race Predictor', f'{test_year} test season', f'Trained on {train[0]} to {train[-1]}, before each race')

labels = {i: f'Round {r.Round} · {r.EventName}' for i, r in races.iterrows()}
pick = st.selectbox('Race', list(labels), index=len(races) - 1, format_func=labels.get)
race = races.loc[pick]
year, rnd = int(race.Year), int(race.Round)
rp = dashboard.race_prediction(op, year, rnd, ridge)

top = lambda col: set(rp.nsmallest(3, col)['Driver'])
mae, grid_mae = rp['Err'].abs().mean(), (rp['GridPred'] - rp['Finish']).abs().mean()
winner = rp.loc[rp['Pred'] == 1, 'Driver'].iloc[0]
numbers, track = st.columns([5, 1], vertical_alignment='center')
with numbers:
    stats([
        {'value': f'{mae:.2f}', 'key': True, 'label': 'places off on average, the model'},
        {'value': f'{grid_mae:.2f}', 'label': 'places off on average, the grid order'},
        {'value': f'{len(top("Pred") & top("Finish"))}', 'unit': '/3', 'label': 'podium finishers found'},
        {'value': winner, 'label': 'predicted winner' + (', and won' if rp.loc[rp['Pred'] == 1, 'Finish'].iloc[0] == 1 else '')},
    ])
with track:
    circuit_map(dashboard.track_outline(data.track_maps(), race.Circuit), race.Circuit)

left, right = st.columns(2, gap='large')
with left:
    section('Predicted', 'against where they finished')
    tower([{'code': r.Driver, 'lineage': r.Lineage, 'team': r.Team, 'right': f'P{int(r.Finish)}',
            'delta': 'exact' if r.Err == 0 else f'{int(r.Err):+d}',
            'delta_class': 'bb-purple' if r.Err == 0 else 'bb-green' if abs(r.Err) <= 2 else 'bb-yellow'}
           for r in rp.itertuples()])
    st.caption('Purple: exact place. Green: within two places. Yellow: further off. Classified finishers only.')
with right:
    section('Result', 'places gained from the grid')
    res = rp.sort_values('Finish')
    tower([{'code': r.Driver, 'lineage': r.Lineage, 'team': r.Team, 'right': f'from P{int(r.Grid)}',
            'delta': '0' if r.Grid == r.Finish else f'{int(r.Grid - r.Finish):+d}',
            'delta_class': 'bb-eq' if r.Grid == r.Finish else 'bb-up' if r.Grid > r.Finish else 'bb-down'}
           for r in res.itertuples()])

section('Strategies', 'what the top ten ran')
plan = dashboard.stint_plan(data.processed('stints'), year, rnd)
order = [d for d in res['Driver'].head(10) if d in set(plan['Driver'])]
if order:
    stint_bars(plan, order, int(plan['EndLap'].max()))

section('Season', f'every {test_year} race, places off on average')
per = {m: outcome.race_metrics(p)['MAE'] for m, p in op.groupby('Model')}
names = races.set_index(['Year', 'Round'])['EventName'].str.replace(' Grand Prix', '')
fig = go.Figure()
for m, colour, width in [('grid', theme.muted(), 1.5), (ridge, theme.RED, 3)]:
    s = per[m]
    fig.add_scatter(x=[names[i] for i in s.index], y=s.values, mode='lines+markers', name=dashboard.MODEL_NAMES.get(m, m),
                    line=dict(color=colour, width=width), marker=dict(size=6),
                    hovertemplate='%{x}: %{y:.2f} places<extra>' + dashboard.MODEL_NAMES.get(m, m) + '</extra>')
fig.update_layout(showlegend=True, legend=dict(orientation='h', y=1.08, x=0))
fig.update_yaxes(title_text='places off')
theme.plot(fig, height=340, key='season')
