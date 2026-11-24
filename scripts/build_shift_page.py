"""Build the 2026 Shift page's tables for the dashboard (boxbox/shift_view.py, app/views/shift_study.py).

    python scripts/build_shift_page.py [--no-wet]

Reruns the shift study code (shift.py) on the 2026 study and the 2022 rehearsal, with SeasonScaled frozen at
config shift_study.season_adjust, and the wet study's crossover windows (skipped with --no-wet). Reads
data/processed/holdout/ (every 2026 round there), so the page shows 2026 without the app ever reading the holdout.
Writes data/models/shift_*.parquet.
"""
import sys
import time

from boxbox import shift, shift_view
from boxbox.config import load_config

cfg = load_config()
start = time.time()
t = shift.load_tables(cfg, holdout=True)
tables = shift_view.build(t, cfg, log=lambda m: print(m, flush=True), wet_study='--no-wet' not in sys.argv)
out = shift_view.save(tables, cfg)
m = tables['meta'].iloc[0]
print(f'2026 rounds scored: {int(m["Rounds"])} (last {int(m["LastRound"])}); {len(tables["wet"])} wet windows; '
      f'{time.time() - start:.0f} s; written to {out}')
h = shift_view.headline(tables)
print(f"linear 2025 {h['ref']['MAE_s']:.3f} / {h['ref']['Bias_s']:+.3f}, 2026 {h['shift']['MAE_s']:.3f} / {h['shift']['Bias_s']:+.3f}")
