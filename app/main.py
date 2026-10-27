"""BoxBox dashboard.

    streamlit run app/main.py

Pages come from registry.PAGES, switched on under dashboard.pages in config.yaml. The menu is drawn here
(not by Streamlit) so it can carry the wordmark and sections.
"""
from pathlib import Path

import streamlit as st

import data
import theme
from components import page_header, soon
from registry import PAGES

st.set_page_config(page_title='BoxBox', page_icon=':material/flag:', layout='wide')
theme.inject()

VIEWS = Path(__file__).with_name('views')


def placeholder(entry):
    def run():
        page_header(entry.title, entry.section)
        soon(entry)
    run.__name__ = f'soon_{entry.key}'
    return run


switches = data.cfg()['dashboard']['pages']
entries = [e for e in PAGES if switches.get(e.key, False)]
# The first page is the default and lives at the root URL; the rest get /<key>
pages = {e.key: st.Page(VIEWS / f'{e.key}.py' if e.status == 'ready' else placeholder(e), title=e.title,
                        icon=e.icon, **({'default': True} if i == 0 else {'url_path': e.key}))
         for i, e in enumerate(entries)}
current = st.navigation(list(pages.values()), position='hidden')

with st.sidebar:
    st.html('<div class="bb-wordmark">BOX<span>BOX</span></div><div class="bb-tagline">Pit wall analytics</div>')
    last = None
    for e in entries:
        if e.section != last:
            st.html(f'<div class="bb-navsec">{e.section}</div>')
            last = e.section
        on = current.title == e.title
        key = 'nav-current' if on else f'nav-{"soon-" if e.status == "soon" else ""}{e.key}'
        with st.container(key=key):
            st.page_link(pages[e.key], label=e.title + ('  ·  soon' if e.status == 'soon' else ''), icon=e.icon)
    st.html('<div class="bb-foot">Data: FastF1 and Jolpica, 2019 to 2025.</div>')

current.run()
