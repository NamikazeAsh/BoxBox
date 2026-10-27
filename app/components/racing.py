"""F1 pieces: tyre dot, timing tower, stint bars, circuit map."""
import base64
from html import escape

import streamlit as st

import theme


def tyre(compound):
    return f'<span class="bb-tyre" style="--c:{theme.compound(compound)}" title="{escape(compound.title())}"></span>'


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


def stint_bars(plan, order, race_laps):
    """Classic tyre strategy graphic. plan: stints table rows (Driver, StartLap, EndLap, Compound) for one race;
    order: drivers top to bottom; race_laps: laps in the race, for the bar scale."""
    rows = ''
    for d in order:
        s = plan[plan['Driver'] == d]
        segs = ''.join(f'<i style="flex:{(r.EndLap - r.StartLap + 1) / race_laps:.4f};background:{theme.compound(r.Compound)}" '
                       f'title="{escape(r.Compound.title())}, laps {int(r.StartLap)} to {int(r.EndLap)}"></i>'
                       for r in s.itertuples())
        rows += f'<div class="bb-srow"><span>{escape(d)}</span><div class="bb-sbar">{segs}</div></div>'
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
