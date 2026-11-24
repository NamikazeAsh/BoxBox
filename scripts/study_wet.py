"""Wet race study (src/boxbox/wet.py, notes/wet_and_degradation_2026-11-23.md): wet tyre wear on the anchored task
(validation 2023 and 2024, test 2025) and the slick to intermediate crossover in every pre 2026 wet race.

    python scripts/study_wet.py
"""
import pandas as pd

from boxbox import wet
from boxbox.config import load_config

cfg = load_config()
laps = pd.read_parquet(cfg['paths']['processed'] / 'laps.parquet')
laps = laps[~laps['Year'].isin(cfg['holdout_seasons'])]
pd.set_option('display.width', 220)

# Wear
seasons = cfg['wet']['wear_seasons']
ml = wet.wet_model_laps(laps, seasons, cfg)
print('clean wet tyre laps in stints with enough of them, by season and compound')
print(ml.groupby(['Year', 'Compound']).agg(Races=('Round', 'nunique'), Stints=('Stint', 'size')).assign(
    Stints=ml.drop_duplicates(['Year', 'Round', 'Driver', 'Stint']).groupby(['Year', 'Compound']).size(),
    Laps=ml.groupby(['Year', 'Compound']).size()).to_string(), '\n')

preds, fits = wet.wear_study(laps, cfg)
for part, p in preds.items():
    print(f'{part}: anchored forecast on wet tyre stints (folds scored: {sorted(p["Fold"].unique())})')
    t = wet.wear_table(p, cfg)
    t['vs_constant_%'] = 100 * (1 - t['MAE_s'] / t.xs('constant pace', level='Model')['MAE_s'].reindex(t.index, level='Compound'))
    print(t.round(3).to_string())
    by = p.groupby(['Fold', 'Compound', 'Model'])['Err_s'].apply(lambda e: e.abs().mean()).unstack('Model')
    print('MAE per fold'); print(by.round(3).to_string(), '\n')

print('wear rates per fold (s per lap of tyre age): within stint (effective), net of the track, track rate')
rates = []
for (year, name), m in fits.items():
    if name == 'net of track':
        rates.append(m.cells.assign(Fold=year))
    elif name == 'linear, compound only':
        rates.append(m.pooled.rename('LinearPooled').to_frame().assign(Fold=year))
r = pd.concat(rates).groupby([lambda i: i, 'Fold']).first() if rates else pd.DataFrame()
print(r.round(4).to_string(), '\n')

# All 2022 to 2025 wet laps, for the rates the note quotes (descriptive, no score)
allfit = wet.TrackNetWear(cfg['wet']['track_iters']).fit(ml)
print(f'net of track fit on every season in {seasons}'); print(allfit.cells.round(4).to_string(), '\n')

# Crossover
win, sw = wet.crossover_study(laps, cfg)
print('crossover windows (gap = slick minus intermediate lap time, s; trend per lap; crossover lap, 80% interval over drivers)')
print(win.round(3).to_string(), '\n')
if len(sw):
    print(f'switches the crossover prices: {len(sw)} (to slick {int((sw["To"] == "slick").sum())}, to inter '
          f'{int((sw["To"] == "wet").sum())}); in windows with a crossover found: {int(sw["Found"].sum())}')
    f = sw[sw['Found']]
    print(f['Loss_s'].describe(percentiles=[.25, .5, .75, .9]).round(2).to_string(), '\n')
    print('loss by laps off the best switch lap (negative = early)')
    print(f.assign(Off=f['LapsOff'].clip(-5, 5)).groupby('Off')['Loss_s'].agg(['size', 'median', 'mean']).round(2).to_string(), '\n')
    print('every priced switch, gap free per lap (no line, no extrapolation; Silverstone 2024 included)')
    print(sw['LossPerLap_s'].describe(percentiles=[.25, .5, .75, .9]).round(2).to_string(), '\n')
    print(sw.groupby(['Year', 'Round', 'Start', 'To']).agg(Switches=('Loss_s', 'size'), Line_med=('Loss_s', 'median'),
                                                         PerLap_med=('LossPerLap_s', 'median'), PerLap_max=('LossPerLap_s', 'max'),
                                                         Clipped=('Clipped', 'mean')).round(2).to_string(), '\n')
    print('per window: switches, median and max loss, share within one lap of best, best lap at the window edge')
    print(f.groupby(['Year', 'Round', 'Start']).agg(Switches=('Loss_s', 'size'), Median_s=('Loss_s', 'median'),
                                                   Max_s=('Loss_s', 'max'), Within1=('LapsOff', lambda x: (x.abs() <= 1).mean()),
                                                   Edge=('BestAtEdge', 'first')).round(2).to_string())
