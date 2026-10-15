"""Derived quantities on the clean tables. Every definition used by the models lives here."""
import numpy as np
import pandas as pd

RACE = ['Year', 'Round']
WET = ['INTERMEDIATE', 'WET']


def add_gap_ahead(laps):
    """Seconds behind the car ahead at the line, from cumulative session time on the same lap.

    Lapped cars are not caught because the comparison is within one lap number.
    """
    laps = laps.sort_values(RACE + ['Session', 'LapNumber', 'Time_s'])
    laps['GapAhead_s'] = laps.groupby(RACE + ['Session', 'LapNumber'])['Time_s'].diff()
    return laps.sort_index()


def add_circuit(events, cfg):
    """Circuit is the stable track key. Location alone is not: FastF1 renamed some venues between
    seasons, and the 2020 Sakhir Grand Prix shares its Location with Bahrain but ran the outer layout."""
    c = cfg['circuits']
    events = events.copy()
    events['Circuit'] = events['Location'].replace(c['aliases'])
    for o in c['layouts']:
        events.loc[(events['Year'] == o['year']) & (events['EventName'] == o['event']), 'Circuit'] = o['circuit']
    return events


def wet_races(laps):
    """(Year, Round) pairs where anyone ran intermediates or wets.

    A rainfall reading alone does not count: it is often recorded before the start or as a few drops
    that never change tyres (modelling O5). Per lap Rainfall stays on the laps table as a feature.
    """
    tyres = laps['Compound'].isin(WET).groupby([laps['Year'], laps['Round']]).any()
    return set(tyres[tyres].index)


def add_fuel_correction(laps, cfg, start_kg=None):
    """LapTimeFC_s is lap time with the estimated fuel mass effect removed (as if on empty tanks).

    Assumes linear burn from the start load to zero over the race distance, taken as the most laps
    completed by anyone. Shortened races are therefore treated as full distance. start_kg defaults to the
    configured race load; pass a per lap Series for sprints (see sprint_start_fuel).
    """
    f = cfg['fuel']
    start = f['start_kg'] if start_kg is None else start_kg
    total = laps.groupby(RACE)['LapNumber'].transform('max')
    fuel_kg = start * (1 - (laps['LapNumber'] - 1) / total)
    laps['FuelKg'] = fuel_kg.clip(lower=0)
    laps['LapTimeFC_s'] = laps['LapTime_s'] - f['s_per_kg'] * laps['FuelKg']
    return laps


def sprint_start_fuel(sprint, race, cfg):
    """Sprint start load per lap: the race load scaled by sprint distance over race distance at the same
    event (assumes teams fuel for the distance). Falls back to a third of the race load."""
    ratio = sprint.groupby(RACE)['LapNumber'].max() / race.groupby(RACE)['LapNumber'].max()
    r = sprint.join(ratio.rename('r'), on=RACE)['r'].fillna(1 / 3)
    return cfg['fuel']['start_kg'] * r


def add_clean_flags(laps, cfg):
    """One boolean per filter step plus IsClean. Rows are never dropped, so thresholds can change."""
    c = cfg['clean_lap']
    laps['f_valid'] = laps['LapTime_s'].notna()
    laps['f_not_lap1'] = laps['LapNumber'] > 1
    laps['f_not_pit'] = ~laps['PitIn'] & ~laps['PitOut']
    laps['f_green'] = laps['TrackStatus'] == '1'
    laps['f_dry'] = laps['Compound'].isin(c['dry_compounds'])
    laps['f_no_traffic'] = ~(laps['GapAhead_s'] < c['traffic_gap_s'])
    base = laps[['f_valid', 'f_not_lap1', 'f_not_pit', 'f_green', 'f_dry', 'f_no_traffic']].all(axis=1)
    med = laps['LapTime_s'].where(base).groupby([laps['Year'], laps['Round'], laps['Driver']]).transform('median')
    laps['f_not_outlier'] = ~(laps['LapTime_s'] > c['outlier_pct'] * med)
    laps['IsClean'] = base & laps['f_not_outlier']
    return laps


def stints(laps):
    g = laps.groupby(RACE + ['Driver', 'Stint'])
    return g.agg(Team=('Team', 'first'), Compound=('Compound', 'first'), StartLap=('LapNumber', 'min'),
                 EndLap=('LapNumber', 'max'), Laps=('LapNumber', 'size'), TyreLifeStart=('TyreLife', 'min'),
                 FreshTyre=('FreshTyre', 'first'), CleanLaps=('IsClean', 'sum')).reset_index()


def pit_loss(laps, wet, cfg):
    """Time a green flag stop costs: in lap plus out lap minus two of the driver's median clean laps.

    Uses raw lap times (fuel effect is near identical across the two laps and the reference).
    Wet races and stops with either lap under yellow, SC, VSC or red are excluded.
    """
    ref = laps[laps['IsClean']].groupby(RACE + ['Driver'])['LapTime_s'].median().rename('Ref_s')
    lp = laps.join(ref, on=RACE + ['Driver'])
    green = lp['TrackStatus'] == '1'
    inl = lp[lp['PitIn'] & green][RACE + ['Driver', 'LapNumber', 'LapTime_s', 'Ref_s']]
    outl = lp[lp['PitOut'] & green][RACE + ['Driver', 'LapNumber', 'LapTime_s']]
    outl = outl.assign(LapNumber=outl['LapNumber'] - 1).rename(columns={'LapTime_s': 'OutLap_s'})
    st = inl.merge(outl, on=RACE + ['Driver', 'LapNumber'])
    st['PitLoss_s'] = st['LapTime_s'] + st['OutLap_s'] - 2 * st['Ref_s']
    st = st[st['PitLoss_s'].between(cfg['pit_loss']['min_s'], cfg['pit_loss']['max_s'])]
    st = st[[(y, r) not in wet for y, r in zip(st['Year'], st['Round'])]]
    return st.rename(columns={'LapNumber': 'InLap', 'LapTime_s': 'InLap_s'})


def pit_loss_summary(stops):
    g = stops.groupby(RACE)['PitLoss_s']
    return pd.DataFrame({'Stops': g.size(), 'Median_s': g.median(),
                         'IQR_s': g.quantile(.75) - g.quantile(.25)}).reset_index()


def long_runs(prac, cfg):
    """Practice runs of at least min_laps consecutive representative laps on one set of dry tyres.

    Pace is the raw median lap time. Practice fuel loads are unknown, so no fuel correction.
    """
    p = prac.sort_values(RACE + ['Driver', 'LapNumber']).copy()
    ok = (p['LapTime_s'].notna() & ~p['PitIn'] & ~p['PitOut'] & (p['TrackStatus'] == '1')
          & p['Compound'].isin(cfg['clean_lap']['dry_compounds']))
    stint_med = p['LapTime_s'].where(ok).groupby([p['Year'], p['Round'], p['Driver'], p['Stint']]).transform('median')
    ok &= p['LapTime_s'] <= cfg['clean_lap']['outlier_pct'] * stint_med
    new_stint = p.groupby(RACE + ['Driver'])['Stint'].diff() != 0
    p['RunId'] = (~ok | new_stint).cumsum()
    runs = p[ok].groupby(RACE + ['Session', 'Driver', 'RunId']).agg(
        Team=('Team', 'first'), Compound=('Compound', 'first'), Laps=('LapTime_s', 'size'),
        TyreLifeStart=('TyreLife', 'min'), Median_s=('LapTime_s', 'median'),
        Slope_s_per_lap=('LapTime_s', lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if len(s) > 2 else np.nan),
        TrackTemp=('TrackTemp', 'median'))
    runs = runs[runs['Laps'] >= cfg['long_run']['min_laps']].reset_index().drop(columns='RunId')
    return runs
