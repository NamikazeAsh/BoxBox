"""Build processed tables from data/interim. Holdout seasons are written to a separate folder."""
import logging

import pandas as pd

from boxbox import checks, clean, features

log = logging.getLogger('boxbox.pipeline')


def build(cfg, years):
    ev = features.add_circuit(clean.events(cfg, years), cfg)
    ev_cols = ['Year', 'Round', 'EventName', 'Location', 'Circuit', 'EventFormat', 'EventDate']

    weather = clean.clean_weather(clean.read_table(cfg, years, 'weather', {'R', 'FP1', 'FP2'}))
    race_w = weather[weather['Session'] == 'R']

    laps = clean.clean_laps(clean.read_table(cfg, years, 'laps', {'R'}))
    laps = clean.join_weather(laps, race_w)
    laps = laps.merge(ev[ev_cols], on=['Year', 'Round'], how='left')
    laps = features.add_gap_ahead(laps)
    laps = features.add_fuel_correction(laps, cfg)
    laps = features.add_clean_flags(laps, cfg)
    wet = features.wet_races(laps)
    laps['WetRace'] = [(y, r) in wet for y, r in zip(laps['Year'], laps['Round'])]

    prac = clean.clean_laps(clean.read_table(cfg, years, 'laps', {'FP1', 'FP2'}))
    prac = clean.join_weather(prac, weather[weather['Session'] != 'R'])
    prac = prac.merge(ev[ev_cols], on=['Year', 'Round'], how='left')

    results = clean.clean_results(clean.read_table(cfg, years, 'results', {'R', 'Q'}))
    pitstops = clean.clean_pitstops(clean.read_table(cfg, years, 'pitstops', {'R'}))
    stops = features.pit_loss(laps, wet, cfg)

    t = {
        'events': ev,
        'laps': laps,
        'practice_laps': prac,
        'stints': features.stints(laps),
        'results': results[results['Session'] == 'R'].drop(columns=['Q1_s', 'Q2_s', 'Q3_s']),
        'qualifying': results[results['Session'] == 'Q'].drop(columns=['GridPosition', 'Points', 'Laps']),
        'weather': weather,
        'race_control': clean.clean_rcm(clean.read_table(cfg, years, 'rcm', {'R'})),
        'pitstops': pitstops,
        'pit_loss_stops': stops,
        'pit_loss': features.pit_loss_summary(stops).merge(ev[['Year', 'Round', 'EventName', 'Location', 'Circuit']],
                                                           on=['Year', 'Round']),
        'long_runs': features.long_runs(prac, cfg),
    }
    checks.run_checks(t, ev)
    return t


def write(cfg, t):
    hold = set(cfg['holdout_seasons'])
    out = cfg['paths']['processed']
    for sub, keep in [('', lambda y: ~y.isin(hold)), ('holdout', lambda y: y.isin(hold))]:
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        for name, df in t.items():
            part = df[keep(df['Year'])]
            if len(part):
                part.to_parquet(d / f'{name}.parquet', index=False)
                log.info('wrote %s/%s.parquet (%s rows)', sub or '.', name, len(part))
