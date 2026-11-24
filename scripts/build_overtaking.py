"""Build the Overtaking page's tables for the dashboard (boxbox/overtaking_view.py, app/views/overtaking.py).

    python scripts/build_overtaking.py

Fits the learned pass model (dashboard.overtaking.model, an overtaking.PassLogit name) on the close pairs of the
seasons before the test season (2022 to 2024 for 2025), with free air pace from a lap model fitted on the same
seasons, and scores it on the rolling validation folds (2023 and 2024) against the circuit pass rate. The test
season is not scored. Writes data/models/overtaking_*.parquet. A few minutes (three lap model fits).
"""
import pandas as pd

from boxbox import overtaking_view as ov
from boxbox.config import load_config

cfg = load_config()
t = {n: pd.read_parquet(cfg['paths']['processed'] / f'{n}.parquet') for n in ov.NEEDS}
tables = ov.build(t, cfg, log=lambda m: print(m, flush=True))
out = ov.save(tables, cfg)
m = tables['meta'].iloc[0]
print(f'{m["Model"]}: {int(m["Pairs"]):,} pairs, {len(tables["circuits"])} circuits, {m["Seconds"]:.0f} s; written to {out}')
