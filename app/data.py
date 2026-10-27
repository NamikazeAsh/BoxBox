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
