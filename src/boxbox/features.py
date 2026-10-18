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


def backmarker_gap(laps, look_back=6):
    """Seconds behind the nearest car a lap or more down that crossed the line just before this one (a
    backmarker about to be lapped), NaN if none among the previous look_back crossings. GapAhead_s misses
    these because it compares cars on the same lap number."""
    out = pd.Series(np.nan, index=laps.index)
    for _, t in laps[laps['Time_s'].notna()].groupby(RACE):
        t = t.sort_values('Time_s')
        tm, ln = t['Time_s'].to_numpy(), t['LapNumber'].to_numpy()
        g = np.full(len(t), np.nan)
        for d in range(look_back, 0, -1):   # nearest last, so it wins
            gap = tm[d:] - tm[:-d]
            hit = ln[:-d] < ln[d:]
            g[d:] = np.where(hit, gap, g[d:])
        out.loc[t.index] = g
    return out


def add_circuit(events, cfg):
    """Circuit is the stable track key. Location alone is not: FastF1 renamed some venues between
    seasons, and the 2020 Sakhir Grand Prix shares its Location with Bahrain but ran the outer layout."""
    c = cfg['circuits']
    events = events.copy()
    events['Circuit'] = events['Location'].replace(c['aliases'])
    for o in c['layouts']:
        events.loc[(events['Year'] == o['year']) & (events['EventName'] == o['event']), 'Circuit'] = o['circuit']
    return events


def dirty_air_bin(gap, edges):
    """Index of the dirty air band a lap ran in, from its gap to the car ahead: band i covers edges[i] to
    edges[i+1] seconds. NaN beyond the last edge (clean air) or with no car ahead on the same lap."""
    return pd.cut(gap, list(edges), right=False, labels=False)


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


def race_conditions(laps, prac):
    """Weather per race: median track and air temperature, humidity and wind over the race laps, the track
    temperature spread during the race (90th minus 10th percentile), and whether the weekend's practice
    session was wet (intermediates, wets or a rainfall reading), a proxy for a green track on race day."""
    g = laps.groupby(RACE)
    out = pd.DataFrame({'TrackTemp': g['TrackTemp'].median(), 'AirTemp': g['AirTemp'].median(),
                        'Humidity': g['Humidity'].median(), 'WindSpeed': g['WindSpeed'].median(),
                        'TrackTempSpread': g['TrackTemp'].quantile(.9) - g['TrackTemp'].quantile(.1)})
    wet = (prac['Compound'].isin(WET) | prac['Rainfall'].fillna(False).astype(bool)).groupby([prac['Year'], prac['Round']]).any()
    out['WetPractice'] = wet.reindex(out.index).astype('boolean')
    return out


def grid_slot(results):
    """Starting slot: the grid position, with a pit lane start (FastF1 grid 0) placed behind the last grid slot."""
    last = results.groupby(RACE)['GridPosition'].transform('max')
    return results['GridPosition'].where(results['GridPosition'] > 0, last + 1)


def classified_finish(results):
    """Place among the classified finishers, 1 to N, NaN for the rest (retired, disqualified, withdrawn).
    Ranked again within the race because a couple of races have gaps in the classified positions."""
    pos = pd.to_numeric(results['ClassifiedPosition'], errors='coerce')
    return pos.groupby([results['Year'], results['Round']]).rank(method='first')


def quali_gap(qualifying):
    """Each driver's best qualifying lap (Q1, Q2 or Q3) as a percent gap to the fastest lap of the session."""
    best = qualifying[['Q1_s', 'Q2_s', 'Q3_s']].min(axis=1)
    return 100 * (best / best.groupby([qualifying['Year'], qualifying['Round']]).transform('min') - 1)


def race_pace(laps, min_laps=5):
    """Race pace per driver in each dry race: median clean fuel corrected lap as a percent gap to the fastest
    driver's median (at least min_laps clean laps). Wet races have none."""
    c = laps[laps['IsClean'] & ~laps['WetRace']]
    g = c.groupby(RACE + ['Driver'])['LapTimeFC_s'].agg(['median', 'size'])
    g = g[g['size'] >= min_laps]['median']
    return (100 * (g / g.groupby(level=RACE).transform('min') - 1)).rename('Pace').reset_index()


def race_overtakes(laps):
    """Green flag places gained per car per lap in each race (a passing rate). A lap counts only if it and the
    lap before were green and no car pitted on either, so places changed by pit stops are left out."""
    l = laps[laps['Position'].notna()].sort_values(RACE + ['Driver', 'LapNumber'])
    gain = l.groupby(RACE + ['Driver'])['Position'].diff() * -1
    lap = l.groupby(RACE + ['LapNumber']).agg(Pit=('PitIn', 'any'), Out=('PitOut', 'any'),
                                              Green=('TrackStatus', lambda s: (s == '1').all()))
    ok = lap['Green'] & ~lap['Pit'] & ~lap['Out']
    ok = ok & ok.groupby(level=RACE).shift(1, fill_value=False)
    l = l.assign(Gain=gain.clip(lower=0)).join(ok.rename('Ok'), on=RACE + ['LapNumber'])
    l = l[l['Ok'] & (l['LapNumber'] > 2) & l['Gain'].notna()]
    return l.groupby(RACE)['Gain'].mean().rename('Overtakes')


def team_lineage(team, cfg):
    """Team key that survives renames (Toro Rosso to Racing Bulls and so on), so form carries across seasons."""
    return team.replace(cfg['teams']['lineage'])


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


def practice_runs(prac, cfg, min_laps=None):
    """Representative laps of practice long runs. A run is one practice stint (pit out to pit in) on dry tyres.
    Its representative laps have a time, are not in or out laps, are green, and are within outlier_pct of the
    run's median; a run needs min_laps of them (default long_run.min_laps). A slow lap (traffic, a mistake)
    is dropped instead of ending the run. RunLaps counts the run's representative laps."""
    min_laps = cfg['long_run']['min_laps'] if min_laps is None else min_laps
    run = RACE + ['Driver', 'Stint']
    p = prac[prac['LapTime_s'].notna() & ~prac['PitIn'] & ~prac['PitOut'] & (prac['TrackStatus'] == '1')
             & prac['Compound'].isin(cfg['clean_lap']['dry_compounds']) & prac['Stint'].notna()]
    med = p.groupby(run)['LapTime_s'].transform('median')
    p = p[p['LapTime_s'] <= cfg['clean_lap']['outlier_pct'] * med]
    p = p.assign(RunLaps=p.groupby(run)['LapNumber'].transform('size'))
    return p[p['RunLaps'] >= min_laps].sort_values(run + ['LapNumber'])


def add_practice_fuel_correction(runs, race, cfg):
    """Practice fuel loads are unknown, but a slope within a run only needs the burn rate. Assumes the race's
    burn rate at the same event (start load over race distance). LapTimeFC_s is lap time at the run's
    starting load, so only its change within a run compares with race LapTimeFC_s, not its level. Runs at
    events without race laps are dropped."""
    burn = cfg['fuel']['start_kg'] / race.groupby(RACE)['LapNumber'].max().rename('BurnKgPerLap')
    runs = runs.join(burn, on=RACE, how='inner')
    since = runs['LapNumber'] - runs.groupby(RACE + ['Driver', 'Stint'])['LapNumber'].transform('min')
    runs['LapTimeFC_s'] = runs['LapTime_s'] + cfg['fuel']['s_per_kg'] * runs['BurnKgPerLap'] * since
    return runs


def long_runs(prac, cfg):
    """One row per practice long run (see practice_runs). Pace is the raw median lap time and the slope is raw
    seconds per lap, no fuel correction (practice loads are unknown)."""
    p = practice_runs(prac, cfg)
    slope = lambda g: np.polyfit(g['LapNumber'], g['LapTime_s'], 1)[0] if len(g) > 2 else np.nan
    g = p.groupby(RACE + ['Session', 'Driver', 'Stint'])
    runs = g.agg(Team=('Team', 'first'), Compound=('Compound', 'first'), Laps=('LapTime_s', 'size'),
                 TyreLifeStart=('TyreLife', 'min'), Median_s=('LapTime_s', 'median'), TrackTemp=('TrackTemp', 'median'))
    runs['Slope_s_per_lap'] = g[['LapNumber', 'LapTime_s']].apply(slope)
    return runs.reset_index().drop(columns='Stint')
