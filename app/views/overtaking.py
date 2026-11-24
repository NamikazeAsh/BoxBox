"""Overtaking: how hard passing is circuit by circuit, from the learned pass model the simulator uses (fitted on the
seasons before the test season). Tables from scripts/build_overtaking.py."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import data
import theme
from boxbox import dashboard, overtaking_view as ov
from components import definitions, page_header, score_table, section, stats

cfg = data.cfg()
s = ov.settings(cfg)
o = data.overtaking()
meta = o['meta'].iloc[0]
circuits, curves, effects, coefs = o['circuits'], o['curves'], o['effects'], o['coefs']
years = f'{int(meta.TrainFrom)} to {int(meta.TrainTo)}'
gap, delta = float(meta.GapS), float(meta.DeltaS)
ink, soft = theme.ink(), theme.rgba(theme.muted(), 0.5)

page_header('Overtaking', 'Passing', f'Learned pass model fitted on {years}, {int(meta.Pairs):,} close laps, before {int(meta.TestYear)}')

a, b = s['contrast']
have = set(circuits['Circuit'])
stuck = dashboard.odds_change(coefs, 'StuckRun')
older = dashboard.odds_change(coefs, 'AgeDiff', 5)
stats([
    *([{'value': f'{dashboard.pass_ratio(circuits, a, b):.0f}', 'unit': '×', 'key': True,
        'label': f'likelier to pass at {a} than at {b}, {delta} s a lap faster from {gap} s behind'}] if {a, b} <= have else []),
    {'value': f'{meta.Rate:.0%}', 'label': f'of laps starting within {meta.WindowS:.0f} s of the car ahead end in a pass, {years}'},
    {'value': f'{-stuck:.0%}', 'label': 'lower odds of passing for every lap already stuck behind the same car'},
    {'value': f'{older:+.0%}', 'label': 'odds when the tyres ahead are 5 laps older, on top of the pace gap'},
])

section('Pace advantage', f'chance of a pass from {gap} s behind, on a first attempt, same tyre age and compound')
order = circuits['Circuit'].tolist()
picked = st.pills('Circuits', order, selection_mode='multi', default=[c for c in s['curve_circuits'] if c in have],
                  key='pass_circuits') or []
fig = go.Figure()
allc = curves[curves['Circuit'] == 'All circuits']
fig.add_scatter(x=allc['Delta'], y=allc['Prob'], mode='lines', line=dict(color=soft, width=1.5, dash='dot'),
                hovertemplate='All circuits, %{x:.2f} s faster: %{y:.0%}<extra></extra>')
ends = []
for c in picked:
    d = curves[curves['Circuit'] == c]
    fig.add_scatter(x=d['Delta'], y=d['Prob'], mode='lines', line=dict(color=ink, width=2),
                    hovertemplate=f'{c}, %{{x:.2f}} s faster: %{{y:.0%}}<extra></extra>')
    ends.append((c, float(d['Prob'].iloc[-1])))
labels = [('All circuits', float(allc['Prob'].iloc[-1]))] + ends
ys = dashboard.spread_labels([y for _, y in labels], 0.045, top=1.0)
x_end = float(curves['Delta'].max())
for (c, _), y in zip(labels, ys):
    fig.add_annotation(x=x_end, y=y, text=c, showarrow=False, xanchor='left', xshift=8, font=dict(size=12, color=ink if c != 'All circuits' else theme.muted()))
fig.add_vline(x=delta, line=dict(color=theme.grid(), width=1))
fig.update_xaxes(title_text='pace advantage over the car ahead, s per lap', range=[float(curves['Delta'].min()), x_end + 0.6],
                 tickvals=np.arange(float(curves['Delta'].min()), x_end + 1e-9, 0.5))
fig.update_yaxes(title_text='chance of a pass on the lap', tickformat='.0%', range=[0, 1.02])
theme.plot(fig, height=420, key='curves')
st.caption(f'The model\'s curves at a {gap} s gap; dotted: the all circuit rate. Pace advantage is the free air lap time '
           f'gap, so it already includes tyre wear. The thin line marks {delta} s, the pace gap behind the numbers above.')

section('Pass rate by circuit', f'share of close laps that end in a pass, {years}')
c = circuits.sort_values('Rate')
fig = go.Figure()
fig.add_bar(x=c['Rate'], y=c['Circuit'], orientation='h', marker=dict(color=[ink if x in picked else soft for x in c['Circuit']],
            line_width=0, cornerradius=3), customdata=np.c_[c['Passes'], c['Pairs'], c['Races']],
            hovertemplate='%{y}: %{x:.1%} of %{customdata[1]:,} close laps, %{customdata[0]} passes in %{customdata[2]} races<extra></extra>')
fig.add_scatter(x=c['ModelRate'], y=c['Circuit'], mode='markers', marker=dict(color=theme.muted(), size=7, symbol='line-ns-open', line_width=2),
                hovertemplate='%{y}: the model leans on %{x:.1%}<extra></extra>')
fig.update_xaxes(title_text='close laps ending in a pass', tickformat='.0%')
fig.update_layout(bargap=0.3)
theme.plot(fig, height=max(360, 22 * len(c)), key='rates')
st.caption(f'Bars: real pass rate (highlighted: the circuits picked above). Ticks: the rate the model uses, pulled toward the all circuit '
           f'rate with the weight of {int(meta.k)} close laps, so a circuit seen in few races leans on the rest.')

section('Tyres and getting stuck', f'all circuit rate, {delta} s faster from {gap} s behind')
left, right = st.columns(2, gap='large')
with left:
    e = effects[effects['Feature'] == 'AgeDiff']
    fig = go.Figure(go.Scatter(x=e['Value'], y=e['Prob'], mode='lines', line=dict(color=ink, width=2),
                               hovertemplate='tyres ahead %{x:+.0f} laps older: %{y:.1%}<extra></extra>'))
    fig.update_xaxes(title_text='how much older the tyres ahead are, laps')
    fig.update_yaxes(title_text='chance of a pass on the lap', tickformat='.0%', rangemode='tozero')
    theme.plot(fig, height=300, key='age')
    st.caption('Older tyres ahead help beyond the lap time gap: worn tyres are easier to pass on.')
with right:
    e = effects[effects['Feature'] == 'StuckRun']
    fig = go.Figure(go.Bar(x=e['Value'], y=e['Prob'], marker=dict(color=soft, line_width=0, cornerradius=3),
                           hovertemplate='%{x:.0f} laps stuck: %{y:.1%}<extra></extra>'))
    fig.update_xaxes(title_text='laps already stuck behind the same car', dtick=1)
    fig.update_yaxes(title_text='chance of a pass on the lap', tickformat='.0%', rangemode='tozero')
    fig.update_layout(bargap=0.3)
    theme.plot(fig, height=300, key='stuck')
    comp = dashboard.odds_change(coefs, 'CompDiff')
    st.caption(f'A car that has not got by is less likely to on the next lap: the odds fall {-stuck:.0%} a lap. A harder '
               f'compound ahead changes the odds by {comp:+.0%} a step.')

section('How it works', 'logistic regression on close laps')
v = o['validation']
how, score = st.columns([1.25, 1], gap='large')
with how:
    definitions([
        ('Close laps', f'Car B starts a lap within {meta.WindowS:.0f} s of the car ahead at the line, both laps green for both cars, '
                       'neither pitting, lap 2 or later, dry races. A pass: B ahead at the next line.'),
        ('Model', f'Logistic regression ({meta.Model}) on the pace advantage (free air lap times from the simulator\'s lap '
                  'model and each stint\'s real level), the gap, the circuit\'s pass rate, the tyre age and compound '
                  'differences and the laps already stuck behind.'),
        ('Seasons', f'Fitted on {years}; {int(meta.TestYear)} is not used, so the simulator\'s {int(meta.TestYear)} races are '
                    'out of sample.'),
        ('Validation', 'Rolling: ' + '; '.join(f'trained on {int(r.TrainFrom)}' + (f' to {int(r.TrainTo)}' if r.TrainTo != r.TrainFrom else '')
                                               + f', scored on {int(r.Year)}' for r in v.itertuples()) + '.'),
        ('In the simulator', 'When the simulator races on this model, each close pair\'s pass is drawn against it, with the logit '
                             f'shifted {"down" if meta.EngineShift < 0 else "up"} by {abs(meta.EngineShift):.1f} so simulated '
                             'races have the real number of passes.'),
        ('Limits', 'No DRS or straight line speed data, no team orders or blue flags; a pass the model sees is any change of '
                   'order at the line, including mistakes.'),
    ])
with score:
    yrs = [str(int(y)) for y in v['Year']]
    rows = [{'Model': 'Pass model', **{y: x for y, x in zip(yrs, v['LogLoss'])}},
            {'Model': 'Circuit pass rate', **{y: x for y, x in zip(yrs, v['LogLossCircuit'])}},
            {'Model': 'One pass rate', 'Tag': 'baseline', **{y: x for y, x in zip(yrs, v['LogLossAll'])}}]
    score_table(rows, [('Model', 'Log loss', None)] + [(y, y, '{:.3f}') for y in yrs], best=yrs, base={'One pass rate'},
                note='Validation seasons, lower is better. Purple marks the best.')
    cal = o['calibration']
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color=theme.grid(), width=1), hoverinfo='skip', showlegend=False)
    for (yr, g), colour in zip(cal.groupby('Year'), [soft, ink]):
        fig.add_scatter(x=g['Pred'], y=g['Real'], mode='lines+markers', line=dict(color=colour, width=2), marker=dict(size=8),
                        name=str(int(yr)), hovertemplate=f'{int(yr)}: predicted %{{x:.0%}}, real %{{y:.0%}}<extra></extra>')
    fig.update_layout(showlegend=True, legend=dict(orientation='h', y=1.12, x=0))
    fig.update_xaxes(title_text='predicted chance of a pass', tickformat='.0%', range=[0, 0.85])
    fig.update_yaxes(title_text='real share passed', tickformat='.0%', range=[0, 0.85])
    theme.plot(fig, height=300, key='calibration')
    st.caption(f'Calibration on the validation seasons, ten bins of close laps each. {int(v["Year"].iloc[-1])}: '
               f'{int(v["PassesPred"].iloc[-1]):,} passes predicted, {int(v["Passes"].iloc[-1]):,} real.')
