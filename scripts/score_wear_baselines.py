"""The simulator's wear curve and the fairer no wear baselines on validation and on the test season, in one run
(degradation_writeup step 18). The test season here is a further look at 2025 for models already chosen on
validation; nothing is chosen from it.

    python scripts/score_wear_baselines.py
"""
import numpy as np
import pandas as pd

from boxbox import degradation as dg, evaluate
from boxbox.config import load_config, seasons_for
from boxbox.degradation import compare

NAMES = ['constant pace', 'linear v1', 'constant pace, lap terms', 'constant pace, lap terms from linear', 'linear',
         'linear, monotone', 'quadratic', 'quadratic, monotone']
BASE = 'constant pace, lap terms from linear'

cfg = load_config()
seasons = seasons_for(cfg, 'degradation')
laps = pd.read_parquet(cfg['paths']['processed'] / 'laps.parquet')
split = evaluate.season_splits(seasons)
pd.set_option('display.width', 220)
ladder = compare.ladder(cfg)

val = compare.run(laps, cfg, seasons, NAMES, log=lambda m: None)
train_years, year = split['test']
train, test = dg.model_laps(laps, train_years, cfg, train=True), dg.model_laps(laps, [year], cfg)
tst = {n: dg.forecast(ladder[n]().fit(train), test, cfg) for n in NAMES}


def paired(preds, base):
    """Per race MAE minus the base model's, mean and 95% bootstrap CI over races."""
    race = lambda p: p.groupby(['Year', 'Round'])['Err_s'].apply(lambda e: e.abs().mean())
    b, rows = race(preds[base]), {}
    for n, p in preds.items():
        d = (race(p) - b).to_numpy()
        rng = np.random.default_rng(0)
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        rows[n] = {'vs_base_s': d.mean(), 'CI_lo': np.percentile(boot, 2.5), 'CI_hi': np.percentile(boot, 97.5),
                   'Better': int((d < 0).sum()), 'Races': len(d)}
    return pd.DataFrame(rows).T


for label, preds in [('validation (train 2022 score 2023, train 2022 and 2023 score 2024, pooled)', val),
                     (f'test (train {train_years}, score {year}; a further look at {year})', tst)]:
    t = compare.table(preds, cfg)[['MAE_s', 'Bias_s', 'N', 'MAE_21+ahead', 'MAE_age>30']]
    cp, lt = t.loc['constant pace', 'MAE_s'], t.loc[BASE, 'MAE_s']
    t['gain_vs_constant_%'] = 100 * (1 - t['MAE_s'] / cp)
    t['gain_vs_lap_terms_%'] = 100 * (1 - t['MAE_s'] / lt)
    print(label)
    print(t.join(paired(preds, BASE)).sort_values('MAE_s').round(4).to_string(), '\n')
    by = pd.concat({n: evaluate.error_table(preds[n], 'Compound')['MAE_s'] for n in
                    ['constant pace', BASE, 'linear', 'quadratic, monotone']}, axis=1)
    print('MAE by compound'); print(by.round(3).to_string(), '\n')

# Lap term offsets: fitted with no wear vs taken from the linear model (full fit on the training seasons)
own, lin = ladder['constant pace, lap terms']().fit(train), ladder['linear']().fit(train)
print(f'offsets, train {train_years} (s)')
print(pd.DataFrame({'no wear fit': own.offsets, 'linear fit': lin.offsets}).round(3).to_string())
