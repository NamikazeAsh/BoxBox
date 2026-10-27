"""Degradation: wear curves per circuit and compound from the saved model, and how every model scored."""
import plotly.graph_objects as go
import streamlit as st

import data
import theme
from boxbox import dashboard, evaluate
from components import bars, circuit_map, definitions, page_header, score_table, section, stats

cfg = data.cfg()
dg, dash = cfg['degradation'], cfg['dashboard']
cells = data.model_output('degradation')
preds = data.model_output('degradation_test_preds')
train, test_year = evaluate.season_splits(data.training('degradation'))['test']
seasons = data.training('degradation')

page_header('Tyre Lab', 'Degradation', f'Wear rates fitted on {seasons[0]} to {seasons[-1]}, {int(cells["Laps"].sum()):,} clean laps')

circuits = sorted(cells['Circuit'].unique())
c1, c2, c3 = st.columns([1.2, 1.4, 1.2], gap='medium')
circuit = c1.selectbox('Circuit', circuits, index=circuits.index('Sakhir') if 'Sakhir' in circuits else 0)
with c2:
    comp = st.segmented_control('Compound', theme.COMPOUNDS, format_func=theme.COMPOUND_NAMES.get,
                                default='MEDIUM', key='compound') or 'MEDIUM'
laps = c3.slider('Stint length, laps', 5, 45, dash['stint_laps_default'])

curves = {c: dashboard.wear_curve(cells, circuit, c, laps, dg['stint_length_ref'], dash['interval_level'])
          for c in theme.COMPOUNDS}
cur = curves[comp]
row = cells[(cells['Circuit'] == circuit) & (cells['Compound'] == comp)]
name = theme.COMPOUND_NAMES[comp]

numbers, track = st.columns([5, 1], vertical_alignment='center')
with numbers:
    stats([
        {'value': f'{cur["Loss_s"].iloc[-1]:.1f}', 'unit': 's', 'key': True, 'label': f'lost on the {name.lower()} after {laps} laps'},
        {'value': f'{cur["Loss_s"].diff().iloc[-1]:.3f}', 'unit': 's', 'label': 'more each lap'},
        {'value': f'{int(row["Stints"].iloc[0]) if len(row) else 0}', 'label': f'{name.lower()} stints at {circuit} behind the rate'},
    ])
with track:
    circuit_map(dashboard.track_outline(data.track_maps(), circuit), circuit)

fig = go.Figure()
col = theme.compound(comp)
fig.add_scatter(x=list(cur['Age']) + list(cur['Age'][::-1]), y=list(cur['Hi_s']) + list(cur['Lo_s'][::-1]),
                fill='toself', fillcolor=theme.rgba(col, 0.14), line=dict(width=0), hoverinfo='skip')
for c, d in curves.items():
    if c != comp:
        fig.add_scatter(x=d['Age'], y=d['Loss_s'], mode='lines', line=dict(color=theme.rgba(theme.muted(), 0.6), width=1.5),
                        hovertemplate=f'{theme.COMPOUND_NAMES[c]}, lap %{{x}}: %{{y:.2f}} s<extra></extra>')
fig.add_scatter(x=cur['Age'], y=cur['Loss_s'], mode='lines', line=dict(color=col, width=3.5),
                hovertemplate=f'{name}, lap %{{x}}: %{{y:.2f}} s<extra></extra>')
fig.add_scatter(x=[cur['Age'].iloc[-1]], y=[cur['Loss_s'].iloc[-1]], mode='markers', marker=dict(color=col, size=10),
                hoverinfo='skip')
fig.update_xaxes(title_text='tyre age, laps', range=[0.5, laps + 0.5])
fig.update_yaxes(title_text='seconds lost against lap one')
theme.plot(fig, height=400, key='wear')
st.caption(f'Shaded: where {dash["interval_level"]:.0%} of {name.lower()} laps land around the curve. Other compounds in grey, '
           f'for the same stint length.' + ('' if cur['FromCircuit'].iloc[0] else
                                            f' No {name.lower()} history at {circuit}: this is the average over all circuits.'))

section('How the models score', f'{test_year} test season')
models, air, circ, how = st.tabs(['Model comparison', 'Dirty air and race start', 'By circuit', 'Methods'])
with models:
    s = dashboard.degradation_scores(preds)
    tags = {'simulator average': 'used by the simulator', 'live average': 'needs laps from the race', 'constant pace': 'baseline'}
    rows = [{'Model': dashboard.MODEL_NAMES.get(r.Model, r.Model), 'Tag': tags.get(r.Model, ''), 'MAE_s': r.MAE_s,
             'Bias_s': r.Bias_s, 'Cover80': None if r.Cover80 != r.Cover80 else r.Cover80, 'N': r.N}
            for r in s.itertuples()]
    score_table(rows, [('Model', 'Model', None), ('MAE_s', 'Error, s per lap', '{:.3f}'), ('Bias_s', 'Bias, s', '{:+.3f}'),
                       ('Cover80', '80% band hit rate', '{:.0%}'), ('N', 'Laps', '{:,.0f}')],
                best=['MAE_s'], base={'No tyre wear'},
                note=f'Trained on {train[0]} to {train[-1]}, scored once on {test_year}. Purple marks the lowest error. '
                     'Each stint\'s first clean laps set its pace; the models forecast the rest.')
with air:
    off = dashboard.offsets(cfg)
    edges = dg['dirty_air_bins']
    da = off.get('DirtyAir')
    if da is not None:
        bars([(f'{k:.1f} to {edges[edges.index(k) + 1]:.1f} s', v) for k, v in zip(da['Key'], da['Offset_s'])],
             note=f'Extra time per lap by gap to the car ahead. Laps closer than {edges[0]:.1f} s count as traffic and are left out; '
                  f'beyond {edges[-1]:.1f} s is clean air.')
    rs = off.get('RaceStart')
    if rs is not None:
        bars([(f'Race lap {int(k)}', v) for k, v in zip(rs['Key'], rs['Offset_s'])],
             note='Extra time on the first racing laps while the field settles.')
with circ:
    linear = next(m for m in preds['Model'].unique() if dashboard.MODEL_NAMES.get(m) == 'Linear')
    show = ['constant pace', linear, 'simulator average']
    t = dashboard.degradation_by_circuit(preds, show).sort_values('simulator average')
    colours = {'constant pace': theme.rgba(theme.muted(), 0.45), linear: theme.muted(), 'simulator average': theme.RED}
    fig = go.Figure()
    for m in show:
        fig.add_scatter(x=t[m], y=t.index, mode='markers', name=dashboard.MODEL_NAMES[m], marker=dict(color=colours[m], size=9),
                        hovertemplate=f'{dashboard.MODEL_NAMES[m]}, %{{y}}: %{{x:.3f}} s<extra></extra>')
    fig.update_layout(showlegend=True, legend=dict(orientation='h', y=1.06, x=0))
    fig.update_xaxes(title_text='error, s per lap')
    fig.update_yaxes(autorange='reversed')
    theme.plot(fig, height=max(360, 24 * len(t)), key='bycircuit')
    st.caption('Sorted by the simulator average, best first. Races swing for reasons nothing before the race predicts.')
with how:
    definitions([
        ('Laps', f'Dry race laps from {seasons[0]} to {seasons[-1]}, without lap 1, pit laps, safety cars, laps within '
                 f'{cfg["clean_lap"]["traffic_gap_s"]} s of the car ahead and outliers. Fuel corrected at '
                 f'{cfg["fuel"]["s_per_kg"]} s per kg.'),
        ('Model', f'Lap time = stint pace + wear rate x tyre age, one rate per circuit and compound, pulled toward the '
                  f'compound average where a circuit has few stints. The rate changes with planned stint length '
                  f'(reference {dg["stint_length_ref"]} laps).'),
        ('Task', f'The first {dg["anchor_laps"]} clean laps of a stint set its pace; the model forecasts every later clean lap.'),
        ('Split', f'Whole seasons, never random laps. Models chosen on {", ".join(str(y) for _, y in evaluate.season_splits(seasons)["validation"])}; '
                  f'{test_year} scored once at the end.'),
        ('Baseline', 'No tyre wear: every lap as fast as the stint\'s first laps.'),
    ])
