"""Design tokens and the shared stylesheet. Base colours also live in .streamlit/config.toml; keep them in step.

Red is the signature and has fixed jobs: the top strip, page title bar, section ticks, the current page,
one key number per page, and selected controls. Everything else stays neutral, tyre colours or team colours.
"""
from pathlib import Path

import streamlit as st

RED = '#E10600'
# Timing colours (purple best, green better, yellow worse) live in style.css and config.toml only

COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
COMPOUND_NAMES = {'SOFT': 'Soft', 'MEDIUM': 'Medium', 'HARD': 'Hard', 'INTERMEDIATE': 'Inter', 'WET': 'Wet'}
_COMPOUND = {
    'dark': {'SOFT': '#FF3B3B', 'MEDIUM': '#FFD12E', 'HARD': '#E9E9EC', 'INTERMEDIATE': '#43B047', 'WET': '#3AA8FF'},
    'light': {'SOFT': '#E0201B', 'MEDIUM': '#D9A400', 'HARD': '#8A8C95', 'INTERMEDIATE': '#2E8B3A', 'WET': '#1F7FD1'},
}

# Keyed by team lineage (config teams.lineage), so renamed teams keep one colour
TEAMS = {
    'McLaren': '#FF8000', 'Red Bull Racing': '#3671C6', 'Mercedes': '#27F4D2', 'Ferrari': '#E8002D',
    'Williams': '#64C4FF', 'Aston Martin': '#229971', 'Alpine': '#FF87BC', 'Racing Bulls': '#6692FF',
    'Sauber': '#52E252', 'Haas F1 Team': '#B6BABD',
}

_CSS = Path(__file__).with_name('style.css')


def mode():
    """'dark' or 'light', from the viewer's current Streamlit theme (dark before the browser reports one)."""
    t = getattr(st.context, 'theme', None)
    return 'light' if t is not None and t.type == 'light' else 'dark'


def compound(c):
    return _COMPOUND[mode()].get(c, '#9A9CA5')


def team(lineage):
    return TEAMS.get(lineage, '#9A9CA5')


def rgba(hex_colour, alpha):
    h = hex_colour.lstrip('#')
    return f'rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})'


def inject():
    """The shared stylesheet, once per run (st.html with only a style tag adds no visible element)."""
    st.html(f'<style>{_CSS.read_text(encoding="utf-8")}</style>')


def muted():
    return '#8B8E98' if mode() == 'dark' else '#62646E'


def grid():
    return 'rgba(255,255,255,0.06)' if mode() == 'dark' else 'rgba(0,0,0,0.07)'


def plot(fig, height=360, key=None):
    """A Plotly figure in the house style: Streamlit theme colours, quiet grid, no mode bar."""
    if fig.layout.showlegend is None:
        fig.update_layout(showlegend=False)
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8),
                      font=dict(family='Barlow, sans-serif', size=13), hoverlabel=dict(font_family='Barlow, sans-serif'),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(gridcolor=grid(), zeroline=False, linecolor=grid(), tickfont=dict(color=muted()),
                     title_font=dict(color=muted(), size=12))
    fig.update_yaxes(gridcolor=grid(), zeroline=False, linecolor=grid(), tickfont=dict(color=muted()),
                     title_font=dict(color=muted(), size=12))
    st.plotly_chart(fig, theme='streamlit', config={'displayModeBar': False}, key=key)
