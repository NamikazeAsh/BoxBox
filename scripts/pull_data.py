"""Pull raw sessions into data/interim. Safe to rerun: finished sessions are skipped.

    python scripts/pull_data.py                  all seasons in config.yaml
    python scripts/pull_data.py --seasons 2025   one season
    python scripts/pull_data.py --track-maps     circuit outlines for the dashboard (needs data/processed)
"""
import argparse
import logging

import fastf1
import pandas as pd

from boxbox.config import load_config
from boxbox.ingest import pull_season, pull_track_outline, track_map_races

ap = argparse.ArgumentParser()
ap.add_argument('--seasons', type=int, nargs='+')
ap.add_argument('--sessions', nargs='+', help='any of R Q FP')
ap.add_argument('--track-maps', action='store_true', help='pull one circuit outline per track instead')
args = ap.parse_args()

cfg = load_config()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
logging.getLogger('fastf1').setLevel(logging.ERROR)
cfg['paths']['fastf1_cache'].mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cfg['paths']['fastf1_cache']))

if args.track_maps:
    races = track_map_races(pd.read_parquet(cfg['paths']['processed'] / 'events.parquet'), cfg)
    for r in races.itertuples():
        try:
            print(r.Circuit, r.Year, r.Round, pull_track_outline(cfg, r.Circuit, r.Year, r.Round))
        except Exception as e:
            print(r.Circuit, r.Year, r.Round, 'failed:', repr(e)[:200])
    raise SystemExit

failures = []
for year in args.seasons or cfg['seasons']:
    failures += pull_season(cfg, year, args.sessions)

log_path = cfg['paths']['interim'] / 'failures.csv'
pd.DataFrame(failures, columns=['Year', 'Round', 'Event', 'Session', 'Error']).to_csv(log_path, index=False)
print(f'{len(failures)} failed sessions, listed in {log_path}')
