"""Turn the per session interim files into typed, season spanning tables."""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger('boxbox.clean')

LAP_TIMES = {'LapTime': 'LapTime_s', 'Sector1Time': 'S1_s', 'Sector2Time': 'S2_s',
             'Sector3Time': 'S3_s', 'Time': 'Time_s', 'LapStartTime': 'LapStart_s'}
LAP_KEEP = ['Driver', 'DriverNumber', 'Team', 'LapNumber', 'Stint', 'Compound', 'TyreLife',
            'FreshTyre', 'TrackStatus', 'Position', 'IsAccurate', 'Deleted', 'SpeedST']


def seconds(s):
    return pd.to_timedelta(s, errors='coerce').dt.total_seconds()


def as_bool(s):
    return s.astype(str).str.lower().isin(['true', '1'])


def session_files(cfg, years, table, kinds):
    """Yield (year, round, session, path) for every finished session of the given kinds."""
    for year in years:
        ydir = cfg['paths']['interim'] / str(year)
        for d in sorted(ydir.glob('*_*')):
            rnd, name = d.name.split('_', 1)
            f = d / f'{table}.parquet'
            if name in kinds and (d / '_done').exists() and f.exists():
                yield year, int(rnd), name, f


def read_table(cfg, years, table, kinds):
    parts = []
    for year, rnd, name, f in session_files(cfg, years, table, kinds):
        df = pd.read_parquet(f)
        df.insert(0, 'Session', name)
        df.insert(0, 'Round', rnd)
        df.insert(0, 'Year', year)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def events(cfg, years):
    rows = []
    for year in years:
        f = cfg['paths']['interim'] / str(year) / 'schedule.parquet'
        if f.exists():
            s = pd.read_parquet(f)
            rows.append(pd.DataFrame({
                'Year': year, 'Round': s['RoundNumber'].astype(int), 'EventName': s['EventName'],
                'Location': s['Location'], 'Country': s['Country'],
                'EventFormat': s['EventFormat'].astype(str), 'EventDate': pd.to_datetime(s['EventDate'])}))
    return pd.concat(rows, ignore_index=True)


def clean_laps(raw):
    df = raw[['Year', 'Round', 'Session'] + LAP_KEEP].copy()
    for src, dst in LAP_TIMES.items():
        df[dst] = seconds(raw[src])
    df['PitIn'] = raw['PitInTime'].notna()
    df['PitOut'] = raw['PitOutTime'].notna()
    df['TrackStatus'] = df['TrackStatus'].astype(str).replace({'nan': '', 'None': ''})
    df['Compound'] = df['Compound'].astype(str).str.upper().replace({'NAN': None, 'NONE': None})
    df['DriverNumber'] = df['DriverNumber'].astype(str)
    for c in ['FreshTyre', 'IsAccurate', 'Deleted']:
        df[c] = as_bool(df[c])
    for c in ['LapNumber', 'Stint', 'TyreLife', 'Position', 'SpeedST']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def clean_weather(raw):
    df = raw[['Year', 'Round', 'Session']].copy()
    df['Time_s'] = seconds(raw['Time'])
    for c in ['AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'WindDirection']:
        df[c] = pd.to_numeric(raw[c], errors='coerce')
    df['Rainfall'] = as_bool(raw['Rainfall'])
    return df


def join_weather(laps, weather):
    """Attach the latest weather reading at or before each lap's end."""
    cols = ['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed', 'Rainfall']
    out = []
    for key, g in laps.groupby(['Year', 'Round', 'Session'], sort=False):
        w = weather[(weather['Year'] == key[0]) & (weather['Round'] == key[1]) & (weather['Session'] == key[2])]
        g = g.sort_values('Time_s')
        if len(w) and g['Time_s'].notna().any():
            m = pd.merge_asof(g[g['Time_s'].notna()], w.sort_values('Time_s')[['Time_s'] + cols],
                              on='Time_s', direction='backward')
            g = pd.concat([m, g[g['Time_s'].isna()]])
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    return df.sort_values(['Year', 'Round', 'Session', 'Driver', 'LapNumber'], ignore_index=True)


def clean_results(raw):
    df = pd.DataFrame({
        'Year': raw['Year'], 'Round': raw['Round'], 'Session': raw['Session'],
        'Driver': raw['Abbreviation'], 'DriverNumber': raw['DriverNumber'].astype(str),
        'DriverId': raw['DriverId'], 'Team': raw['TeamName'],
        'Position': pd.to_numeric(raw['Position'], errors='coerce'),
        'ClassifiedPosition': raw['ClassifiedPosition'].astype(str),
        'GridPosition': pd.to_numeric(raw['GridPosition'], errors='coerce'),
        'Status': raw['Status'].astype(str), 'Points': pd.to_numeric(raw['Points'], errors='coerce'),
        'Laps': pd.to_numeric(raw['Laps'], errors='coerce')})
    for q in ['Q1', 'Q2', 'Q3']:
        df[f'{q}_s'] = seconds(raw[q])
    return df


def clean_pitstops(raw):
    df = raw[['Year', 'Round', 'driverId']].rename(columns={'driverId': 'DriverId'})
    df['Lap'] = pd.to_numeric(raw['lap'], errors='coerce')
    df['Stop'] = pd.to_numeric(raw['stop'], errors='coerce')
    # Jolpica duration is pit lane time, entry to exit. Values over a minute come as "m:ss.sss".
    d = raw['duration'].astype(str)
    mins = d.str.extract(r'^(\d+):(\d+\.?\d*)$').astype(float)
    df['PitLane_s'] = pd.to_numeric(d, errors='coerce').fillna(mins[0] * 60 + mins[1])
    return df


def clean_rcm(raw):
    df = raw[['Year', 'Round', 'Category', 'Message', 'Status', 'Flag', 'Scope']].copy()
    df['Time'] = pd.to_datetime(raw['Time'], errors='coerce')
    df['Lap'] = pd.to_numeric(raw['Lap'], errors='coerce')
    df['RacingNumber'] = raw['RacingNumber'].astype(str)
    return df

