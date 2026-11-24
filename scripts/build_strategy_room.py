"""Build the Strategy Room's tables for the dashboard (boxbox/strategy_room.py, app/views/strategy_room.py).

    python scripts/build_strategy_room.py [--rounds 2 3] [--drivers 3] [--no-check]

Fits the simulator inputs and the rival pit model on the seasons before the test season (2022 to 2024 for 2025),
then for every race the room covers (dashboard.strategy_room.races) draws the field once and runs the strategy
search for its drivers (dashboard.strategy_room.drivers: that many top finishers, null for every classified
finisher). Engine, passing rule, pace model and search settings are read from config.yaml at run time, so rerun
this after the simulator changes. Writes data/models/strategy_room_*.parquet and strategy_room_inputs.joblib.
About 2 minutes of fitting, then about 6 s per driver at 200 draws (about 15 minutes for six green races).
--rounds and --drivers build a smaller set for a quick look; --no-check skips the check against
backtest.strategy_search on the first search.
"""
import argparse
import time

import pandas as pd

from boxbox import strategy_room as sr
from boxbox.config import load_config

p = argparse.ArgumentParser()
p.add_argument('--rounds', type=int, nargs='*')
p.add_argument('--drivers', type=int)
p.add_argument('--no-check', action='store_true')
a = p.parse_args()

cfg = load_config()
t0 = time.time()
t = {n: pd.read_parquet(cfg['paths']['processed'] / f'{n}.parquet') for n in sr.NEEDS}
tables, inputs = sr.build(t, cfg, rounds=a.rounds, drivers=a.drivers, check=not a.no_check, log=lambda m: print(m, flush=True))
out = sr.save(tables, inputs, cfg)
m = tables['meta']
print(f'{len(m)} races, {int(m["Searched"].sum())} searches, {len(tables["search"])} strategy rows, '
      f'{(time.time() - t0) / 60:.1f} min; written to {out}')
