"""Cached loaders for the app. Views read data only through here; shaping lives in boxbox.dashboard."""
import streamlit as st

from boxbox import dashboard, features
from boxbox.config import load_config, seasons_for


@st.cache_resource
def cfg():
    return load_config()


def _need(load, *args, **kw):
    """Run a loader; a missing file stops the page with the command that builds it."""
    try:
        return load(*args, **kw)
    except FileNotFoundError as e:
        st.error(f'Missing data: {e.filename}. Build it with `python scripts/build_tables.py`, then '
                 '`python scripts/train_degradation.py` and `python scripts/train_outcome.py`.')
        st.stop()


@st.cache_data(show_spinner=False)
def processed(name, columns=None):
    return _need(dashboard.table, cfg(), name, columns=columns)


@st.cache_data(show_spinner=False)
def model_output(name):
    return _need(dashboard.model_output, cfg(), name)


@st.cache_data(show_spinner=False)
def race_flags():
    return features.safety_car_races(processed('laps', ['Year', 'Round', 'TrackStatus']))


@st.cache_data(show_spinner=False)
def seasons():
    """Seasons with processed data (holdout seasons live elsewhere and are never read here)."""
    return sorted(processed('events', ['Year'])['Year'].unique().tolist())


@st.cache_data(show_spinner=False)
def track_maps():
    """Circuit outlines; optional, so an empty table (not a stop) when they have not been pulled."""
    return dashboard.track_maps(cfg())


def training(component):
    return seasons_for(cfg(), component)


def _need_room(load):
    """Strategy Room loaders import the simulator only when that page runs, so the other pages never depend on it."""
    try:
        return load()
    except FileNotFoundError as e:
        st.error(f'Missing data: {e.filename}. Build it with `python scripts/build_strategy_room.py` '
                 '(about 15 minutes), then restart the app.')
        st.stop()


@st.cache_data(show_spinner=False)
def room():
    """Strategy Room tables (strategy_room.TABLES)."""
    from boxbox import strategy_room
    return _need_room(lambda: strategy_room.load(cfg()))


@st.cache_resource(show_spinner=False)
def room_inputs():
    """The simulator inputs the room's searches raced on (fitted before the test season)."""
    from boxbox import strategy_room
    return _need_room(lambda: strategy_room.load_inputs(cfg()))


@st.cache_resource(show_spinner=False)
def room_field(year, rnd):
    from boxbox import strategy_room
    return strategy_room.field_from_tables(room(), year, rnd)


@st.cache_data(show_spinner=False, max_entries=200)
def room_race(year, rnd, driver, comps, stops):
    """Per draw places of one strategy raced against the saved field (the strategy builder)."""
    from boxbox import strategy_room
    place, _ = strategy_room.race_strategy(room_inputs(), room_field(year, rnd), driver, comps, stops, cfg())
    return place


@st.cache_data(show_spinner=False)
def shift():
    """2026 Shift page tables (shift_view.TABLES), precomputed: the app never reads the holdout season itself."""
    from boxbox import shift_view
    try:
        return shift_view.load(cfg())
    except FileNotFoundError as e:
        st.error(f'Missing data: {e.filename}. Build it with `python scripts/build_shift_page.py` (about 3 minutes), '
                 'then restart the app.')
        st.stop()


@st.cache_data(show_spinner=False)
def overtaking():
    """Overtaking page tables (overtaking_view.TABLES)."""
    from boxbox import overtaking_view
    try:
        return overtaking_view.load(cfg())
    except FileNotFoundError as e:
        st.error(f'Missing data: {e.filename}. Build it with `python scripts/build_overtaking.py` (a few minutes), '
                 'then restart the app.')
        st.stop()
