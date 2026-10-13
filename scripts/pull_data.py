"""Pull raw sessions into data/interim. Safe to rerun: finished sessions are skipped.

    python scripts/pull_data.py                  all seasons in config.yaml
    python scripts/pull_data.py --seasons 2025   one season
"""
import argparse
import logging

import fastf1
import pandas as pd

from boxbox.config import load_config
from boxbox.ingest import pull_season

ap = argparse.ArgumentParser()
ap.add_argument('--seasons', type=int, nargs='+')
ap.add_argument('--sessions', nargs='+', help='any of R Q FP')
args = ap.parse_args()

cfg = load_config()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
logging.getLogger('fastf1').setLevel(logging.ERROR)
cfg['paths']['fastf1_cache'].mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cfg['paths']['fastf1_cache']))

failures = []
for year in args.seasons or cfg['seasons']:
    failures += pull_season(cfg, year, args.sessions)

log_path = cfg['paths']['interim'] / 'failures.csv'
pd.DataFrame(failures, columns=['Year', 'Round', 'Event', 'Session', 'Error']).to_csv(log_path, index=False)
print(f'{len(failures)} failed sessions, listed in {log_path}')
