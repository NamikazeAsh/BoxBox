"""2026 Shift: do the models fitted on 2022 to 2025 hold on the 2026 cars? Tables from scripts/build_shift_page.py
(the shift study code, boxbox/shift.py); the page never reads the holdout season itself."""
import plotly.graph_objects as go
import streamlit as st

import data
import theme
from boxbox import shift_view as sv
from components import bars, definitions, page_header, score_table, section, stats, tiles

t = data.shift()
meta = t['meta'].iloc[0]
h = sv.headline(t)
ref, sh = h['ref'], h['shift']
y0, y1 = int(ref['Test']), int(sh['Test'])
ink, muted, soft = theme.ink(), theme.muted(), theme.rgba(theme.muted(), 0.5)
LINES = {'frozen': dict(color=ink, width=2.5), 'season scaled': dict(color=muted, width=2, dash='dash'),
         'rolling refit': dict(color=soft, width=2, dash='dot')}

page_header('2026 Shift', 'Distribution shift',
            f'Trained on {sh["Train"]}, scored on {y1} rounds 1 to {int(meta.LastRound)} ({int(meta.Rounds)} dry races); '
            f'reference: {ref["Train"]} scored on {y0}')

p, pl = h['predictor'], h['pit loss']
stats([
    {'value': f'{h["mae_change"]:+.0%}', 'key': True,
     'label': f'tyre wear forecast error on {y1} against {y0} ({ref["MAE_s"]:.3f} to {sh["MAE_s"]:.3f} s a lap)'},
    {'value': f'{sh["Bias_s"]:+.2f}', 'unit': 's',
     'label': f'bias on {y1} ({ref["Bias_s"]:+.2f} on {y0}): negative means tyres wore faster than forecast'},
    *([{'value': f'{p["shift"]["Gain"]:.1%}', 'label': f'finishing predictor better than grid order on {y1} ({p["reference"]["Gain"]:.1%} on {y0})'}]
      if p['shift'] else []),
    *([{'value': f'{pl["shift"]["MAE"]:.2f}', 'unit': 's', 'label': f'pit loss error per race on {y1} ({pl["reference"]["MAE"]:.2f} s on {y0})'}]
      if pl['shift'] else []),
])
tiles(sv.verdicts(t))


def round_chart(study, key, height=340):
    fig = go.Figure()
    for v, d in sv.round_series(t['rounds'], study).items():
        fig.add_scatter(x=d['Round'], y=d['MAE_s'], mode='lines+markers', name=v, line=LINES[v], marker=dict(size=6),
                        customdata=d['Circuit'], hovertemplate=f'{v}, round %{{x}} %{{customdata}}: %{{y:.3f}} s<extra></extra>')
    fig.update_layout(showlegend=True, legend=dict(orientation='h', y=1.14, x=0))
    fig.update_xaxes(title_text='round', dtick=1)
    fig.update_yaxes(title_text='forecast error, s per lap', rangemode='tozero')
    theme.plot(fig, height=height, key=key)


def block_note(study):
    b = t['blocks']
    b = b[(b['Study'] == study) & (b['Split'] == 'shift') & (b['Block'] == 'all')].set_index('Version')['Value']
    return ', '.join(f'{v} {b[v]:.3f}' for v in sv.VERSIONS if v in b) + ' s over all rounds.'


section('Tyre wear forecast', f'{y0} against {y1}, laps forecast from a stint\'s early laps')
left, right = st.columns([1.5, 1], gap='large')
cols = [('Model', 'Model', None)] + [(f'{m} {y}', f'{m} {y}', f) for y in (y0, y1)
                                    for m, f in [('MAE', '{:.3f}'), ('Bias', '{:+.3f}'), ('Gain', '{:.1%}')]]
with left:
    score_table(sv.score_rows(t['scores'], 'study'), cols, best=[f'MAE {y0}', f'MAE {y1}'], base={'constant pace'},
                note='Seconds per lap. Bias is predicted minus actual. Gain: error saved against constant pace on the same laps.')
with right:
    rows = [{'Model': theme.COMPOUND_NAMES[c], **{f'Gain {y}': sv.score(t['scores'], 'study', s, 'linear', c)['Gain']
                                                 for y, s in [(y0, 'reference'), (y1, 'shift')]}} for c in theme.COMPOUNDS]
    score_table(rows, [('Model', 'Linear, by compound', None), (f'Gain {y0}', f'Gain {y0}', '{:.1%}'), (f'Gain {y1}', f'Gain {y1}', '{:.1%}')],
                note='The Soft forecast loses most of its value on the new cars.')
st.caption('The error rises and the bias flips sign. Gain over constant pace rises only because constant pace gets worse '
           'when tyres wear faster, so bias and wear rates, not gain, show the shift.')

section('Wear rate per compound', 'linear model, all circuits pooled')
r = sv.rates_wide(t['rates'], 'study')
series = [c for c in r.columns if not c.startswith('Matched')]
fig = go.Figure()
for c, colour in zip(series, [theme.rgba(muted, 0.3), soft, ink]):
    fig.add_bar(x=[theme.COMPOUND_NAMES[k] for k in r.index], y=r[c], name=c if c != 'Training fit' else f'training fit, {sh["Train"]}',
                marker=dict(color=colour, line_width=0, cornerradius=3), hovertemplate=f'{c}, %{{x}}: %{{y:.3f}} s per lap<extra></extra>')
fig.update_layout(barmode='group', bargap=0.3, showlegend=True, legend=dict(orientation='h', y=1.14, x=0))
fig.update_yaxes(title_text='s per lap of tyre age', rangemode='tozero')
theme.plot(fig, height=320, key='rates')
m = r[f'Matched change {y1}']
st.caption(f'On circuits raced in both years, {y1} against the training fit: '
           + ', '.join(f'{theme.COMPOUND_NAMES[k]} {m[k]:+.3f}' for k in r.index) + ' s per lap.')

section('Error by round', f'{y1}: does updating on the season\'s own races help?')
round_chart('study', 'rounds')
st.caption('Frozen: the model fitted before the season. Season scaled: one wear multiplier per compound from the earlier '
           'rounds, set on the 2022 rehearsal before 2026 was scored. Rolling refit: refitted after every round. '
           + block_note('study'))

section('Method check', 'the same code on 2022, the last big rule change')
rr = sv.headline(t, 'rehearsal')['ref'], sv.headline(t, 'rehearsal')['shift']
r0, r1 = int(rr[0]['Test']), int(rr[1]['Test'])
left, right = st.columns([1.2, 1], gap='large')
with left:
    cols = [('Model', 'Model', None)] + [(f'{m} {y}', f'{m} {y}', f) for y in (r0, r1)
                                        for m, f in [('MAE', '{:.3f}'), ('Bias', '{:+.3f}'), ('Gain', '{:.1%}')]]
    score_table(sv.score_rows(t['scores'], 'rehearsal'), cols, best=[f'MAE {r0}', f'MAE {r1}'], base={'constant pace'},
                note=f'Trained on {rr[1]["Train"]}, scored on {r1}; reference: {rr[0]["Train"]} scored on {r0}.')
    st.caption(f'In {r1} the gain rose while the bias moved from {rr[0]["Bias_s"]:+.2f} to {rr[1]["Bias_s"]:+.2f} s: a known shift '
               'shows in bias and wear rates, not in gain. That rule, and the season scaled setting, were fixed here before '
               f'{y1} was scored.')
with right:
    round_chart('rehearsal', 'rehearsal_rounds', height=300)
    st.caption(block_note('rehearsal'))

wet = t['wet']
if len(wet):
    section('Wet call', 'scenario: the cost of switching tyres off the crossover lap')
    k = st.segmented_control('Laps off the crossover', [1, 2, 3, 4], default=1, key='wet_laps') or 1
    costs, med = sv.wet_costs(wet, k)
    a, b = st.columns([1, 2.4], gap='large')
    with a:
        stats([{'value': f'{med:.1f}', 'unit': 's', 'label': f'lost switching {k} lap{"s" if k > 1 else ""} early or late, median race'}],
              vertical=True)
    with b:
        bars(list(zip(costs['Race'], costs['Cost_s'])), fmt='{:.1f} s',
             note='Each wet race window where slicks and intermediates ran together (2019 to 2025).')
    st.caption('The gap between slicks and intermediates moves by about the same amount each lap as the track dries or '
               'gets wetter. Switching k laps from where it crosses zero costs that trend times k squared over 2. '
               'A scenario, not a forecast: there is no rain input.')

section('How it works', 'declared before 2026 was scored')
definitions([
    ('Splits', f'Shift: trained on {sh["Train"]}, scored on {y1}. Reference: trained on {ref["Train"]}, scored on {y0}, '
               'a season with no rule change. The shift is the change between the two.'),
    ('Tyre wear', 'Each stint is forecast from its early laps (anchored forecast), dry races only. Linear is the chosen '
                  'model; simulator wear is the one the strategy simulator races on.'),
    ('Pit loss', 'Race median time lost in the pit lane, per circuit median of earlier seasons, against one global median.'),
    ('Safety cars', 'Chance of a safety car or virtual safety car in a race, from earlier seasons.'),
    ('Protocol', 'Every setting was fixed on the 2022 rehearsal and earlier seasons; nothing is retuned on 2026. '
                 f'Built {meta.Built}.'),
])
