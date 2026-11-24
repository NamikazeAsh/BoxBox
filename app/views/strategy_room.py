"""Strategy Room: the strategy search for one driver of a test season race against the strategy they ran, every
driver of the race at a glance, and a builder that races any strategy against the same field. Tables from
scripts/build_strategy_room.py."""
from html import escape

import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots

import data
import theme
from boxbox import dashboard, strategy_room as sr
from components import call_panel, page_header, race_table, section, stats, stint_bars, strategy, strategy_cards, strategy_table, tiles, tyre, tyre_guide

cfg = data.cfg()
room = data.room()
s, r = sr.settings(cfg), cfg['simulator']['race']
meta = dashboard.room_races(room['meta'])
test_year = int(meta['Year'].iloc[0])
season = data.processed('events', ['Year'])
total = int((season['Year'] == test_year).sum())


# Start tyres of the 381 searched cars of 2025: what teams fitted, and what the search picked when free to choose
# (D28; notes/runs/2026-11-11_strategy_room_build_free_start.txt). Why the search keeps the real start tyre.
FREE_START = {'HARD': (71, 169), 'MEDIUM': (235, 166), 'SOFT': (75, 46)}


def start_tyre_mode(race):
    """real or free: the start tyre rule the tables were built with (older tables: the config's)."""
    x = race.get('StartTyre')
    return x if isinstance(x, str) else cfg['simulator'].get('search', {}).get('start_tyre', 'real')


def why_fixed():
    """Why the search keeps the real start tyre: teams' start tyres against the free search's, as paired bars."""
    n = sum(a for a, _ in FREE_START.values())
    top = max(max(a, b) for a, b in FREE_START.values())
    rows = ''.join(
        f'<div class="r"><span class="k">{tyre(k)}{escape(theme.COMPOUND_NAMES.get(k, k.title()))}</span>'
        f'<span class="bars"><span class="b team"><i style="width:{a / top * 100:.0f}%"></i><b>{a}</b></span>'
        f'<span class="b free"><i style="width:{b / top * 100:.0f}%"></i><b>{b}</b></span></span></div>'
        for k, (a, b) in FREE_START.items())
    return ('<div class="bb-why"><h5>Why the start tyre is fixed</h5>'
            '<p>Since 2022 teams may start on any dry tyre, and BoxBox once searched every start tyre. It started far too many '
            f'cars on Hards: of the {n} cars searched this season, teams started {FREE_START["HARD"][0]} on Hards and BoxBox '
            f'picked {FREE_START["HARD"][1]}. The cause is its weakest input: it rates a new Hard nearly as fast as a new Medium '
            'and slower to wear, so a Hard start looked free. Keeping the start tyre the team chose takes that bias out of the '
            'BoxBox pick. You can still try any start tyre in Build your own strategy.</p>'
            f'<div class="bb-why-chart">{rows}</div>'
            '<div class="bb-why-key"><span class="team">teams started on</span>'
            '<span class="free">BoxBox picked, free to choose</span></div></div>')


@st.dialog('How the Strategy Room works', width='large')
def how_it_works(race, driver, cars, cars_start=None):
    from html import escape
    free = start_tyre_mode(race) == 'free'
    start = f'{theme.COMPOUND_NAMES.get(cars_start, str(cars_start).title()).lower()} start' if cars_start else 'start tyre'
    steps = [('flag', 'Set up the race', f'The real {test_year} grid. Pace from qualifying and recent form, not the real result.'),
             ('alt_route', 'Try every strategy', ('Any start tyre' if free else f'From {driver}\'s real {start}') +
                                                 f', 1 to {r["max_stops"]} stops every {r["stop_lap_step"]} laps, '
                                                 f'only stint lengths teams have run at {race.Circuit}.'),
             ('replay', f'Race each one {int(race.Draws)} times', 'Same rivals, same safety cars, same luck for every strategy.'),
             ('leaderboard', 'Rank by average finish', 'The best average finish is the BoxBox pick.')]
    gloss = [('Average', '<span class="bb-pill"><b>P3.8</b></span>', f'mean finishing place over the {int(race.Draws)} races'),
             ('8 in 10 races', '<span class="bb-mini-range"><i></i></span>', 'the range most races end in'),
             ('Against the best', '<span class="bb-pill"><b>49%</b></span>', 'how often it beats the BoxBox pick'),
             ('Their call', '<span class="bb-badge best">Spot on</span>', 'the team ran the BoxBox pick'),
             ('', '<span class="bb-badge near">Even</span>', f'what the team ran is within {r["tie_places"]} of a place of the pick'),
             ('', '<span class="bb-badge worse">BoxBox better</span>', 'the pick clearly finishes higher than what the team ran')]
    know = [('event_available', f'Fitted on {race.TrainFrom} to {race.TrainTo} only, so every {test_year} race is a fair test.'),
            ('do_not_disturb_on', 'No retirements, no team orders, no wet races.'),
            ('speed', 'Places come from model pace, so they are not the real result.'),
            ('tire_repair', 'The search picks compounds less well than Pirelli\'s own race eve prediction, so BoxBox better is '
                            'what its tyre model believes, not proof the team got it wrong.'),
            ('swap_horiz', 'The search tries every start tyre, and leans to Hard starts more than teams do.' if free else
             'The search keeps the start tyre the team chose; the builder lets you change it. Why: below.')]
    cards = [f'<div class="bb-step"><span class="num">{n}</span><span class="ic">{icon}</span><b>{escape(title)}</b>'
             f'<p>{escape(text)}</p></div>' for n, (icon, title, text) in enumerate(steps, 1)]
    st.html(f'<p class="bb-how-lead">Every strategy {escape(driver)} could have run, raced {int(race.Draws)} times against the '
            'field, then ranked by where it finishes on average.</p>'
            '<div class="bb-steps">' + '<span class="go">arrow_forward</span>'.join(cards) + '</div>'
            '<div class="bb-how-cols"><div><h5>Reading the numbers</h5><dl class="bb-gloss">' + ''.join(
                f'<dt>{escape(k)}</dt><dd class="ex">{ex}</dd><dd>{escape(v)}</dd>' for k, ex, v in gloss) + '</dl></div>'
            '<div><h5>Good to know</h5><ul class="bb-know">' + ''.join(
                f'<li><span class="ic">{icon}</span><span>{escape(x)}</span></li>' for icon, x in know) + '</ul></div></div>'
            + ('' if free else why_fixed()))
    tiles = [('Passing', 'threshold per circuit' if 'threshold' in str(race.Rule) else str(race.Rule)),
             ('Car pace', 'qualifying gap, grid, finishing form' if 'FinishForm' in str(race.Pace) else str(race.Pace)),
             ('Tyre wear', f'{race.WearModel} curves, {int(race.Bootstrap)} refits'),
             ('Safety cars', 'drawn per circuit'),
             ('Restarts', 'the train lines up'),
             ('Red flags', 'free tyre change'),
             ('Stop under a safety car', 'moved onto it if due'),
             ('Rivals', 'may answer a stop next lap'),
             ('Start tyre', 'any, a new set' if free else 'real in the search, any in the builder'),
             ('Built', f'{pd.Timestamp(race.Built):%d %B %Y}')]
    with st.expander('Technical settings', icon=':material/tune:'):
        st.html('<div class="bb-settings">' + ''.join(f'<div><span>{escape(k)}</span><b>{escape(v)}</b></div>' for k, v in tiles) + '</div>')


page_header('Strategy Room', f'{test_year} test season',
            f'Simulator and rival pit model fitted on {meta["TrainFrom"].iloc[0]} to {meta["TrainTo"].iloc[0]}, before the season')

labels = {i: f'Round {m.Round} · {m.EventName}' for i, m in meta.iterrows()}
c1, c2, c3 = st.columns([1.4, 1, 0.45], gap='medium', vertical_alignment='bottom')
pick = c1.selectbox('Race', list(labels), index=len(meta) - 1, format_func=labels.get,
                    help=f'{len(meta)} of {total} {test_year} races: the dry ones. The simulator races dry races, '
                         'with safety cars and VSCs drawn for each one; wet races are left out.')
race = meta.loc[pick]
year, rnd, race_laps, cars = int(race.Year), int(race.Round), int(race.RaceLaps), int(race.Cars)
drivers = dashboard.room_drivers(room['cars'], year, rnd)
names = {d.Driver: f'{d.Driver} · {d.Team} · ' + (f'disqualified, P{int(d.Road)} on the road' if d.Dsq else f'finished P{int(d.Finish)}')
         for d in drivers.itertuples()}
driver = c2.selectbox('Driver', list(names), format_func=names.get, key=f'driver_{year}_{rnd}',
                      help='The best strategy depends on the car: where it starts and how fast it is, so BoxBox searches '
                           'for each driver. The Whole race tab below shows every driver at once. Cars that retired are left out '
                           '(the simulator has no crashes or failures); cars disqualified after the race are in, at their place on the road.')
if c3.button('How it works', icon=':material/info:', width='stretch'):
    how_it_works(race, driver, cars, str(drivers.set_index('Driver').loc[driver, 'StartCompound']))

table = dashboard.room_table(room['search'], year, rnd, driver)
v = dashboard.room_verdict(table)
best, real = v['best'], v['real']
car = drivers.set_index('Driver').loc[driver]
place = lambda x: f'P{x:.1f}'
pit_loss = float(data.room_inputs().pit.predict([race.Circuit])[0])

numbers, talk = st.columns([1, 3.2], gap='large')
with numbers:
    stats([
        {'value': place(best['MeanPlace']), 'key': True, 'label': 'average finish on the BoxBox pick'},
        {'value': place(real['MeanPlace']), 'label': f'on the strategy {driver} ran, rank {v["real_rank"]} of {v["raced"]}'},
        {'value': dashboard.finish_label(car), 'label': (f'disqualified after the race; P{int(car.Road)} on the road from P{int(car.Grid)}'
                                                          if car.Dsq else f'real finish, from P{int(car.Grid)} on the grid')},
    ], vertical=True)
def chips(row):
    return strategy(dashboard.strategy_compounds(row['Compounds']), dashboard.strategy_stops(row['StopLaps']))


def pill(label, value, tone=''):
    return f'<span class="bb-pill {tone}">{escape(label)} <b>{escape(value)}</b></span>'


def tone(x, eps=0.5):
    return 'good' if x > eps else ('bad' if x < -eps else '')


def head_to_head(a, b, a_name, b_name):
    rows = [('Average', place(a['MeanPlace']), place(b['MeanPlace']), a['MeanPlace'] < b['MeanPlace'], b['MeanPlace'] < a['MeanPlace']),
            ('Win', f'{a["Win"]:.0%}', f'{b["Win"]:.0%}', a['Win'] > b['Win'], b['Win'] > a['Win']),
            ('Podium', f'{a["Podium"]:.0%}', f'{b["Podium"]:.0%}', a['Podium'] > b['Podium'], b['Podium'] > a['Podium'])]
    out = f'<div class="bb-vs"><span></span><span class="head">{escape(a_name)}</span><span></span><span class="head">{escape(b_name)}</span>'
    for k, x, y, xw, yw in rows:
        out += (f'<span class="k">{k}</span><span class="a{" win" if xw else ""}">{x}</span><span class="mid">vs</span>'
                f'<span class="b{" win" if yw else ""}">{y}</span>')
    return out + '</div>'


def meter(pick, pick_name, other_name, draws):
    """Head to head on the same draws: the share of races each strategy finishes ahead of the other (ties split)."""
    n = round(pick * draws)
    if 0.45 <= pick <= 0.55:
        say = f'<b>Coin toss.</b> {escape(pick_name)} finishes ahead in {n} of {draws} races: about half.'
    elif pick > 0.55:
        say = (f'<b>{"Clear call" if pick >= 0.65 else "Lean"}.</b> {escape(pick_name)} finishes ahead in {n} of {draws} races.')
    else:
        say = f'<b>Against the pick.</b> {escape(other_name)} finishes ahead in {draws - n} of {draws} races.'
    bar = lambda name, x, cls: (f'<div class="row"><span class="nm">{escape(name)}</span><span class="bar {cls}"><i style="width:{x * 100:.0f}%">'
                                f'</i></span><b>{x:.0%}</b></div>')
    return (f'<div class="bb-h2h">{bar(pick_name, pick, "a")}{bar(other_name, 1 - pick, "b")}</div>'
            f'<p class="bb-sure-say">{say}</p>')


with talk:
    call = dashboard.room_call(table, pit_loss)
    ran_row = (f'<div class="bb-real"><span class="pos">{dashboard.finish_label(car)}</span><span>'
               f'{f"P{int(car.Road)} on the road, " if car.Dsq else ""}from P{int(car.Grid)} on</span>{chips(real)}</div>')
    if call['tier'] == 'best':
        nxt = call['next']
        blocks = [('Worth', '<div class="bb-pills">' + pill('Average', place(real['MeanPlace'])) + pill('Win', f'{real["Win"]:.0%}')
                   + pill('Podium', f'{real["Podium"]:.0%}') + '</div>')]
        if nxt is not None:
            blocks.append(('Next best', f'<div class="bb-pills">{chips(nxt)}' + pill('behind', f'{nxt["DeltaPlace"]:.2f} places') + '</div>'))
            blocks.append(('How sure, inside the simulation', meter(1 - float(nxt['PBeatsBest']), f'{driver}\'s strategy', 'Next best',
                                                                  int(race.Draws))))
        blocks.append(('Real race', ran_row))
        call_panel('Right call', f'{driver} ran the BoxBox pick', chips(real), blocks)
    else:
        if call['extra_stops']:
            why = (pill('Pit stops', f'{call["pit"]:+.0f} s', tone(call['pit'])) + pill('Fresher tyres' if call['tyres'] >= 0 else 'Older tyres',
                   f'{call["tyres"]:+.0f} s', tone(call['tyres'])))
        else:
            why = pill('Stop timing and tyres', f'{call["clean"]:+.0f} s', tone(call['clean']))
        why = (f'<div class="bb-pills">{why}<span class="arrow">=</span>' + pill('Alone on track', f'{call["clean"]:+.0f} s', tone(call['clean']))
               + pill('Racing the field', f'{call["traffic"]:+.0f} s', tone(call['traffic'])) + '</div>')
        blocks = [('Why: seconds the call gains', why),
                  ('Worth', head_to_head(best, real, 'BoxBox pick', f'{driver} ran')),
                  ('How sure, inside the simulation', meter(1 - call['beats'], 'BoxBox pick', f'{driver} ran', int(race.Draws))),
                  ('Real race', ran_row)]
        if call['tier'] == 'near tie':
            call_panel('Too close to call', f'What {driver} ran is as good as the BoxBox pick', chips(best), blocks)
        else:
            words = dashboard.plan_words(best['Compounds'], best['StopLaps'])
            call_panel('The call', words[0].upper() + words[1:], chips(best), blocks)


def tag(row):
    return 'Ran' if row['Real'] else {1: 'Best', 2: '2nd', 3: '3rd'}.get(int(row['Rank']), f'#{int(row["Rank"])}')


def card(label, compounds, stop_laps, mean, p10, p90, win, podium, against, real=False, best=False):
    return {'label': label, 'compounds': dashboard.strategy_compounds(compounds), 'stops': dashboard.strategy_stops(stop_laps),
            'mean': place(mean), 'range': f'P{round(p10)} to P{round(p90)}', 'win': f'{win:.0%}', 'podium': f'{podium:.0%}',
            'against': against, 'real': real, 'best': best}


def against(row):
    return 'the pick' if int(row['Rank']) == 1 else f'ahead {row["PBeatsBest"]:.0%} of races'


shares = dashboard.room_place_shares(room['places'], year, rnd, driver, cars)
charted = table[table['Rank'].isin(shares['Rank'].unique())]
section('Best strategies against what they ran', f'{race_laps} laps; numbers under each bar are the laps a stop is made on')
strategy_cards([card(tag(x), x['Compounds'], x['StopLaps'], x['MeanPlace'], x['Place10'], x['Place90'], x['Win'], x['Podium'],
                     against(x), real=bool(x['Real']), best=int(x['Rank']) == 1) for _, x in charted.iterrows()], race_laps)


def place_chart(rows, key):
    """Small multiples: share of simulated races finishing in each place, one row per strategy.
    rows: (name, shares by place, mean place, strong) with strong drawn in ink and the rest muted."""
    k = len(rows)
    fig = make_subplots(rows=k, cols=1, shared_xaxes=True, vertical_spacing=0.12 / k)
    top = max(float(sh.max()) for _, sh, _, _ in rows) * 1.25
    ink, soft = theme.ink(), theme.rgba(theme.muted(), 0.55)
    for i, (name, sh, mean, strong) in enumerate(rows, 1):
        fig.add_bar(x=sh.index, y=sh.values, marker=dict(color=ink if strong else soft, line_width=0, cornerradius=3),
                    hovertemplate=f'{name}, P%{{x}}: %{{y:.0%}} of races<extra></extra>', row=i, col=1)
        fig.add_shape(type='line', x0=mean, x1=mean, y0=0, y1=top, line=dict(color=ink, width=1.5, dash='dot'), row=i, col=1)
        fig.add_annotation(text=f'<b>{name}</b>   average P{mean:.1f}', x=0, y=1, xref='x domain', yref='y domain',
                           xanchor='left', yanchor='top', showarrow=False, font=dict(size=13), row=i, col=1)
        fig.update_yaxes(range=[0, top], tickformat='.0%', nticks=3, row=i, col=1)
    fig.update_layout(bargap=0.25)
    fig.update_xaxes(tickvals=list(range(1, cars + 1)), ticktext=[f'P{p}' for p in range(1, cars + 1)], range=[0.4, cars + 0.6])
    theme.plot(fig, height=40 + 92 * k, key=key)


whole, every, spread = st.tabs(['Whole race', 'Every strategy', 'Finishing spread'])

with whole:
    o = dashboard.room_overview(room['search'], room['cars'], year, rnd)
    race_table([{'finish': x.Finish, 'driver': x.Driver, 'team': x.Team, 'lineage': x.Lineage, 'grid': x.Grid,
                 'ran': (x.RanCompounds, x.RanStops), 'ran_mean': x.RanPlace, 'ran_rank': x.RanRank, 'raced': x.Raced,
                 'best': (x.BestCompounds, x.BestStops), 'best_mean': x.BestPlace, 'gain': x.Gain, 'tier': x.Tier,
                 'expected': x.Expected, 'beat': x.BeatSim} for x in o.itertuples()], selected=driver)
    near = int((o['Tier'] != 'worse').sum())
    beat, worse, miss = int((o['BeatSim'] > 0).sum()), int((o['BeatSim'] < 0).sum()), float(o['BeatSim'].abs().mean())
    st.markdown(f'**{near} of {len(o)}** teams were spot on or even with the BoxBox pick. Against BoxBox\'s expected order, '
                f'**{beat}** cars finished higher and **{worse}** lower; a car finished **{miss:.1f} places** from its expected '
                'position on average.')
    st.caption('BoxBox prefers its own pick for most cars in most races. Teams with full strategy departments rarely leave a place '
               'behind, so this mostly reflects BoxBox\'s weakest input: how fast each tyre is when new. Read BoxBox better as what '
               'its tyre model believes; Beat the sim by is the check against the real race.')
    st.html('<dl class="bb-legend">'
            '<dt>Ran</dt><dd>the strategy the team really ran: tyres and the laps they stopped on.</dd>'
            f'<dt>Average</dt><dd>BoxBox\'s average finishing place for that strategy over {int(race.Draws)} simulated races, with '
            'car pace predicted before the race.</dd>'
            '<dt>Rank</dt><dd>where the team\'s strategy ranks among the strategies BoxBox raced for that car (1 is the pick).</dd>'
            '<dt>BoxBox pick</dt><dd>the strategy with the best average finish for that car.</dd>'
            '<dt>Places it gains</dt><dd>the team\'s average less the pick\'s: how much higher the pick finishes, in places.</dd>'
            '<dt>Their call</dt><dd>the team\'s strategy judged inside the simulation: spot on (it is the pick), even (within '
            f'{r["tie_places"]} of a place, or not clearly worse), BoxBox better (clearly worse).</dd>'
            '<dt>Expected</dt><dd>BoxBox\'s expected finishing position: the car\'s rank in the race by its average on the strategy '
            'it ran.</dd>'
            '<dt>Beat the sim by</dt><dd>expected position less the real place on the road. Positive: the car finished higher than '
            'BoxBox expected. This is where BoxBox loses: either it undervalued the strategy or the car was faster than its pace '
            'model predicted; this column cannot tell the two apart.</dd></dl>')


with every:
    show_all = st.toggle(f'Show all {v["raced"]}', key=f'all_{year}_{rnd}_{driver}') if v['raced'] > s['table_rows'] else True
    rows = table if show_all else table[(table['Rank'] <= s['table_rows']) | table['Real']]
    strategy_table([{'rank': x.Rank, 'compounds': dashboard.strategy_compounds(x.Compounds), 'stops': dashboard.strategy_stops(x.StopLaps),
                     'mean': place(x.MeanPlace), 'range': f'P{round(x.Place10)} to P{round(x.Place90)}', 'podium': f'{x.Podium:.0%}',
                     'points': f'{x.Points:.0%}', 'delta': '' if x.Rank == 1 else f'{x.DeltaPlace:+.2f}', 'tier': x.Tier,
                     'real': x.Real, 'tag': f'{driver} ran' if x.Real else ''} for x in rows.itertuples()],
                   note=f'Numbers between tyres are the laps a stop is made on. Likely range: 8 in 10 simulated races finish there. '
                        f'Against the best: average places lost on the same draws. Near tie: within {r["tie_places"]} of a place, '
                        f'or not clearly worse.')

with spread:
    st.markdown(f'**How to read it.** Each panel is one strategy. Every bar is the share of the {int(race.Draws)} simulated races '
                'that ended in that position, and the dotted line is the average finish. Tall bars on the left mean the strategy '
                'often wins or podiums; a long tail to the right means it sometimes goes wrong (traffic after a stop, a safety car '
                'at the wrong time). Two strategies with the same average can carry very different risk.')
    place_chart([(tag(x), shares[shares['Rank'] == x['Rank']].set_index('Place')['Share'], x['MeanPlace'], bool(x['Real']))
                 for _, x in charted.iterrows()], key='places')


def guide():
    inputs = data.room_inputs()
    from boxbox.simulator import strategy as stg
    years = list(range(int(race.TrainFrom), int(race.TrainTo) + 1))
    wear = lambda k, age: float(inputs.lap.wear(race.Circuit, [stg.CODE[k]], [float(age)])[0])
    g = dashboard.tyre_guide(data.processed('stints'), data.processed('events'), race.Circuit, years,
                             stg.caps_for(inputs.caps, race.Circuit), wear)
    lap = lambda x: '' if x != x else f'{x:.0f} laps'
    tyre_guide([{'compound': x.Compound, 'typical': lap(x.Typical), 'range': '' if x.Low != x.Low else f'usually {x.Low:.0f} to {x.High:.0f}',
                 'longest': lap(x.Longest), 'wear': f'{x.Wear10:.1f} s', 'all': bool(x.All)} for x in g.itertuples()],
               note=f'{race.Circuit}, dry races {years[0]} to {years[-1]}: tyre age when a stint ended, and how much slower a tyre is '
                    'after 10 laps than when new. BoxBox searches stints up to ' +
                    ', '.join(f'{x.Compound.title()} {x.Cap:.0f}' for x in g.itertuples()) + ' laps here (what teams have run, '
                    'pulled toward the norm where a compound is rarely raced).')


@st.fragment
def builder():
    key = f'{year}_{rnd}_{driver}'
    real_comps, real_stops = sr.parse(real['Compounds'], real['StopLaps'])
    left, right = st.columns([1.25, 1], gap='large')
    with left:
        n = st.segmented_control('Stops', [1, 2, 3], default=min(max(len(real_stops), 1), 3), key=f'stops_{key}') or 1
        if n == len(real_stops):
            comps0, stops0 = list(real_comps), list(real_stops)
        else:
            comps0 = [car.StartCompound] + [('HARD' if car.StartCompound != 'HARD' else 'MEDIUM')] * n
            stops0 = [round(race_laps * (i + 1) / (n + 1)) for i in range(n)]
        comps, stops = [], []
        for i in range(n + 1):
            a, b = st.columns([2, 1], vertical_alignment='bottom')
            with a:
                c = st.segmented_control('Start tyre' if i == 0 else f'Stint {i + 1}', theme.COMPOUNDS,
                                         format_func=theme.COMPOUND_NAMES.get, default=comps0[i], key=f'tyres_{i}_{n}_{key}')
                comps.append(c or comps0[i])
            if i < n:
                with b:
                    stops.append(int(st.number_input('Box at the end of lap', 1, race_laps - 1, stops0[i], key=f'lap_{i}_{n}_{key}')))
        go_race = st.button('Race it', type='primary', icon=':material/sports_score:', width='stretch')
    with right:
        problems = sr.check_strategy(comps, stops, race_laps, 1)
        label = sr.label(comps, stops)
        st.caption('Your plan against what was run')
        stint_bars(dashboard.strategy_plan([('Yours', *label), ('Ran', real['Compounds'], real['StopLaps'])], race_laps),
                   ['Yours', 'Ran'], race_laps, {'Yours': dashboard.short_strategy(*label), 'Ran': dashboard.short_strategy(real['Compounds'], real['StopLaps'])})
        if problems:
            st.warning(' '.join(problems))
        if go_race and not problems:
            with st.spinner(f'Racing it {int(race.Draws)} times'):
                st.session_state['built'] = (key, comps, stops, data.room_race(year, rnd, driver, tuple(comps), tuple(stops)))
        b = st.session_state.get('built')
        if not b or b[0] != key:
            st.caption(f'Pick the start tyre, the tyres after each stop and the laps you box on, then race it against the same '
                       f'{int(race.Draws)} draws of the field BoxBox used: same pace draws, same rival strategies, same safety cars. '
                       f'{driver}\'s real start tyre keeps its real age; any other start tyre is a new set.'
                       + ('' if start_tyre_mode(race) == 'free' else
                          f' The BoxBox pick only tried {driver}\'s real '
                          f'{theme.COMPOUND_NAMES.get(car.StartCompound, car.StartCompound).lower()} start (How it works says why); '
                          'here you can start on anything.'))
            return
    # the result, full width under the controls
    _, bcomps, bstops, mine = b
    pl = room['places']
    pl = pl[(pl['Year'] == year) & (pl['Round'] == rnd) & (pl['Driver'] == driver)]
    best_p = pl.loc[pl['Rank'] == best['Rank'], 'Place'].to_numpy()
    m = sr.summary(mine, best_p)
    blabel = sr.label(bcomps, bstops)
    same = table[(table['Compounds'] == blabel[0]) & (table['StopLaps'] == blabel[1])]
    rank = int(same['Rank'].iloc[0]) if len(same) else int((table['MeanPlace'] < m['MeanPlace']).sum()) + 1
    d_best, d_ran = float(m['DeltaPlace']), float(m['MeanPlace'] - real['MeanPlace'])
    side = lambda x: 'level' if abs(x) < 0.005 else (f'{abs(x):.2f} behind' if x > 0 else f'{abs(x):.2f} ahead')
    tiles([{'label': 'Against the BoxBox pick', 'value': side(d_best), 'sub': 'places, on average', 'tone': 'bad' if d_best > 0.005 else 'good'},
           {'label': 'Finishes ahead of the pick', 'value': f'{m["PBeatsBest"]:.0%}', 'sub': f'of {int(race.Draws)} races (ties half)',
            'bar': float(m['PBeatsBest'])},
           {'label': 'Rank among strategies', 'value': f'{rank} of {v["raced"] + (0 if len(same) else 1)}', 'sub': '1 is the BoxBox pick'},
           {'label': f'Against what {driver} ran', 'value': side(d_ran), 'sub': 'places, on average',
            'tone': 'bad' if d_ran > 0.005 else ('good' if d_ran < -0.005 else '')}])
    if bcomps[0] != car.StartCompound and start_tyre_mode(race) != 'free':
        st.caption(f'This starts on {theme.COMPOUND_NAMES.get(bcomps[0], bcomps[0]).lower()}s, a tyre the BoxBox pick did not try, '
                   'so its rank is among strategies that all start on the real tyre. BoxBox rates a new Hard nearly as fast as '
                   'a Medium, so a gain from a Hard start is likely too generous.')
    strategy_cards([card('Yours', blabel[0], blabel[1], m['MeanPlace'], m['Place10'], m['Place90'], float((mine == 1).mean()),
                         m['Podium'], f'ahead {m["PBeatsBest"]:.0%} of races'),
                    card('Best', best['Compounds'], best['StopLaps'], best['MeanPlace'], best['Place10'], best['Place90'],
                         best['Win'], best['Podium'], 'the pick', best=True),
                    card('Ran', real['Compounds'], real['StopLaps'], real['MeanPlace'], real['Place10'], real['Place90'],
                         real['Win'], real['Podium'], against(real), real=True)], race_laps)


section('Build your own strategy', f'race it against the same {int(race.Draws)} draws of the field')
with st.container(border=True):
    guide()
    builder()
