"""Build data/processed from data/interim.

    python scripts/build_tables.py                  all seasons in config.yaml
    python scripts/build_tables.py --seasons 2025
"""
import argparse
import logging

from boxbox import pipeline
from boxbox.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument('--seasons', type=int, nargs='+')
args = ap.parse_args()

cfg = load_config()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
years = [y for y in (args.seasons or cfg['seasons']) if (cfg['paths']['interim'] / str(y)).exists()]
pipeline.write(cfg, pipeline.build(cfg, years))
