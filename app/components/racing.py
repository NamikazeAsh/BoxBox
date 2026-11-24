"""F1 pieces: tyre dot, timing tower, stint bars, circuit map."""
import base64
from html import escape

import streamlit as st

import theme


LETTER = {'SOFT': 'S', 'MEDIUM': 'M', 'HARD': 'H', 'INTERMEDIATE': 'I', 'WET': 'W'}


def tyre(compound):
    """F1 style tyre marker: the compound's colour as a ring with its letter (S, M, H, I, W)."""
    return (f'<span class="bb-tyre" style="--c:{theme.compound(compound)}" title="{escape(compound.title())}">'
            f'{LETTER.get(compound, "?")}</span>')


def tower(rows):
    """Timing tower. rows: dicts with code, lineage, team (shown small), right (small text), delta (text) and
    delta_class (a bb-* colour class), in display order."""
    out = ''
    for i, r in enumerate(rows, 1):
        out += (f'<div class="bb-trow"><span class="p">{i}</span><span class="stripe" style="background:{theme.team(r["lineage"])}"></span>'
                f'<span class="who">{escape(r["code"])}<span>{escape(r.get("team", ""))}</span></span>'
                f'<span class="g">{escape(r.get("right", ""))}</span>'
                f'<span class="d {r.get("delta_class", "")}">{escape(r.get("delta", ""))}</span></div>')
    st.html(f'<div class="bb-tower">{out}</div>')


def stint_bars(plan, order, race_laps, right=None):
    """Classic tyre strategy graphic. plan: stints table rows (Driver, StartLap, EndLap, Compound) for one race;
    order: drivers top to bottom; race_laps: laps in the race, for the bar scale. right: optional {driver: text}
    shown after each bar (stop laps, say)."""
    rows = ''
    for d in order:
        s = plan[plan['Driver'] == d]
        segs = ''.join(f'<i style="flex:{(r.EndLap - r.StartLap + 1) / race_laps:.4f};background:{theme.compound(r.Compound)}" '
                       f'title="{escape(r.Compound.title())}, laps {int(r.StartLap)} to {int(r.EndLap)}">'
                       f'<b>{LETTER.get(r.Compound, "")}</b></i>' for r in s.itertuples())
        note = f'<em>{escape(right.get(d, ""))}</em>' if right else ''
        rows += f'<div class="bb-srow{" wide" if right else ""}"><span>{escape(d)}</span><div class="bb-sbar">{segs}</div>{note}</div>'
    used = [c for c in theme.COMPOUND_NAMES if c in set(plan['Compound'])]
    key = ''.join(f'<span>{tyre(c)} {theme.COMPOUND_NAMES[c]}</span>' for c in used)
    st.html(f'<div class="bb-stints">{rows}</div><div class="bb-skey">{key}</div>')


def circuit_map(outline, label=''):
    """Small circuit outline with the finish line in red. outline: dashboard.track_outline (None draws nothing).
    Drawn as an SVG image because st.html strips inline SVG; colours follow the current theme."""
    if outline is None:
        return
    pad, w, h = 0.06, outline['X'].max(), outline['Y'].max()
    pts = ' '.join(f'{x:.4f},{y:.4f}' for x, y in zip(outline['X'], outline['Y']))
    x0, y0 = outline.iloc[0]
    ink = '#ECECEF' if theme.mode() == 'dark' else '#17171B'
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-pad} {-pad} {w + 2 * pad} {h + 2 * pad}">'
           f'<polygon points="{pts}" fill="none" stroke="{ink}" stroke-opacity=".85" stroke-width=".022" '
           f'stroke-linejoin="round"/><circle cx="{x0:.4f}" cy="{y0:.4f}" r=".034" fill="{theme.RED}"/></svg>')
    src = 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()
    cap = f'<div class="bb-map-label">{escape(label)}</div>' if label else ''
    st.html(f'<div class="bb-map"><img src="{src}" alt="{escape(label or "circuit map")}">{cap}</div>')


def strategy(compounds, stops):
    """A strategy inline: tyre dots with the stop laps between them. compounds: names; stops: laps."""
    parts = tyre(compounds[0]) + ''.join(f'<small>{int(lap)}</small>{tyre(c)}' for c, lap in zip(compounds[1:], stops))
    return f'<span class="bb-strat">{parts}</span>'


TIER_CLASS = {'best': 'bb-purple', 'near tie': 'bb-green', 'worse': 'bb-faint'}


def strategy_table(rows, note=''):
    """Ranked strategies (Strategy Room). rows: dicts with rank, compounds, stops, mean, range, podium, points,
    delta (text), tier ('best', 'near tie' or 'worse'), real (True marks the strategy the car ran) and tag."""
    head = ''.join(f'<th>{h}</th>' for h in ['#', 'Strategy', 'Mean place', 'Likely range', 'Podium', 'Points',
                                             'Against the best', 'Tier'])
    body = ''
    for r in rows:
        tag = f'<span class="tag">{escape(r["tag"])}</span>' if r.get('tag') else ''
        cells = [f'<td class="bb-faint">{int(r["rank"])}</td>', f'<td>{strategy(r["compounds"], r["stops"])}{tag}</td>',
                 f'<td{" class=best" if r["tier"] == "best" else ""}>{escape(r["mean"])}</td>',
                 f'<td>{escape(r["range"])}</td>', f'<td>{escape(r["podium"])}</td>', f'<td>{escape(r["points"])}</td>',
                 f'<td>{escape(r["delta"])}</td>', f'<td class="tier {TIER_CLASS.get(r["tier"], "")}">{escape(r["tier"])}</td>']
        cls = ' class="real"' if r.get('real') else ''
        body += f'<tr{cls}>{"".join(cells)}</tr>'
    st.html(f'<div class="bb-table-wrap"><table class="bb-table bb-strats"><thead><tr>{head}</tr></thead><tbody>{body}'
            f'</tbody></table></div>{f"<p class=bb-note>{escape(note)}</p>" if note else ""}')



def _bar(compounds, stops, race_laps):
    """One strategy as a stint bar with letters, and its stop laps marked under it."""
    edges = [0] + [int(x) for x in stops] + [race_laps]
    segs = ''.join(f'<i style="flex:{(b - a) / race_laps:.4f};background:{theme.compound(c)}" title="{escape(c.title())}, '
                   f'laps {a + 1} to {b}"><b>{LETTER.get(c, "")}</b></i>' for c, a, b in zip(compounds, edges[:-1], edges[1:]))
    ticks = ''.join(f'<span style="left:{lap / race_laps * 100:.2f}%">{lap}</span>' for lap in stops)
    return f'<div class="bb-cbar"><div class="bb-sbar">{segs}</div><div class="bb-ticks">{ticks}</div></div>'


def strategy_cards(rows, race_laps):
    """Strategies side by side, one row each: label, stint bar with its stop laps, and its numbers. rows: dicts with
    label, compounds (names), stops (laps), mean (text), range (text), win, podium (text), against (text, head to
    head with the best), real and best (flags)."""
    head = ('<div class="bb-crow head"><span></span><span>Strategy</span><span>Average</span><span>8 in 10 races</span>'
            '<span>Win</span><span>Podium</span><span>Against the best</span></div>')
    body = ''
    for r in rows:
        cls = ' real' if r.get('real') else (' best' if r.get('best') else '')
        body += (f'<div class="bb-crow{cls}"><span class="lab">{escape(r["label"])}</span>{_bar(r["compounds"], r["stops"], race_laps)}'
                 f'<span class="big">{escape(r["mean"])}</span><span>{escape(r["range"])}</span><span>{escape(r["win"])}</span>'
                 f'<span>{escape(r["podium"])}</span><span class="vs">{escape(r["against"])}</span></div>')
    st.html(f'<div class="bb-cards">{head}{body}</div>')


def _beat(x):
    """Real place against the simulated average: +3.4 (green, finished higher than expected) or -2.1 (amber)."""
    if x is None or x != x:
        return '<td></td>'
    cls = 'up' if x > 0 else ('down' if x < 0 else 'flat')
    return f'<td class="num"><span class="bb-beat {cls}">{"+" if x > 0 else ""}{int(x)}</span></td>'


VERDICT = {'best': ('best', 'Spot on'), 'near tie': ('near', 'Even'), 'worse': ('worse', 'BoxBox better')}   # the team's strategy against the pick


def race_table(rows, selected=None):
    """Every driver of a race (Strategy Room): finish, driver with team stripe, grid, what they ran and how it
    scored, the BoxBox pick, the places it gains (bar) and a verdict badge. rows: dicts with finish, driver,
    team, lineage, grid, ran (compounds, stops), ran_mean, ran_rank, raced, best (compounds, stops), best_mean,
    gain, tier. selected: the driver to highlight."""
    top = max([r['gain'] for r in rows] + [0.5])
    head = ''.join(f'<th>{h}</th>' for h in ['', 'Driver', 'Grid', 'Ran', 'Average', 'Rank', 'BoxBox pick', 'Average',
                                             'Places it gains', 'Their call', 'Expected', 'Beat the sim by'])
    body = ''
    for r in rows:
        cls, word = VERDICT.get(r['tier'], ('worse', r['tier']))
        gain = (f'<span class="bb-gain"><i style="width:{max(r["gain"], 0) / top * 100:.0f}%"></i></span>'
                f'<span class="n">{r["gain"]:.2f}</span>')
        body += (f'<tr class="{"sel" if r["driver"] == selected else ""}">'
                 f'<td class="pos">{escape(str(r["finish"]))}</td>'
                 f'<td class="who"><span class="stripe" style="background:{theme.team(r["lineage"])}"></span>'
                 f'<b>{escape(r["driver"])}</b><small>{escape(str(r["team"]))}</small></td>'
                 f'<td class="num">P{int(r["grid"])}</td><td>{strategy(*r["ran"])}</td>'
                 f'<td class="num">P{r["ran_mean"]:.1f}</td><td class="num faint">{int(r["ran_rank"])} of {int(r["raced"])}</td>'
                 f'<td>{strategy(*r["best"])}</td><td class="num">P{r["best_mean"]:.1f}</td><td class="gaincell">{gain}</td>'
                 f'<td><span class="bb-badge {cls}">{word}</span></td>'
                 f'<td class="num faint">{"P" + str(int(r["expected"])) if r.get("expected") is not None else ""}</td>{_beat(r.get("beat"))}</tr>')
    st.html(f'<div class="bb-table-wrap"><table class="bb-table bb-race"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


def tyre_guide(rows, note=''):
    """One small card per compound: typical stint, the longest teams ran and wear after 10 laps. rows: dicts with
    compound, typical, range, longest, wear (texts) and all (the circuit had too few stints: every circuit used)."""
    cards = ''.join(
        f'<div class="bb-tg"><div class="hd">{tyre(r["compound"])}<b>{escape(r["compound"].title())}</b>'
        f'{"<span class=all>all circuits</span>" if r.get("all") else ""}</div>'
        f'<dl><dt>Typical stint</dt><dd><b>{escape(r["typical"])}</b> <span>{escape(r["range"])}</span></dd>'
        f'<dt>Longest teams ran</dt><dd><b>{escape(r["longest"])}</b></dd>'
        f'<dt>After 10 laps</dt><dd><b>{escape(r["wear"])}</b> <span>slower per lap</span></dd></dl></div>' for r in rows)
    st.html(f'<div class="bb-tgs">{cards}</div>{f"<p class=bb-note>{escape(note)}</p>" if note else ""}')
