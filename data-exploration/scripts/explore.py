from pathlib import Path
import numpy as np, pandas as pd

O = str(Path(__file__).resolve().parent.parent / 'data') + '/'
L = pd.read_parquet(O + 'laps.parquet')
P = pd.read_parquet(O + 'practice.parquet')
W = pd.read_parquet(O + 'weather.parquet')
R = pd.read_parquet(O + 'results.parquet')
M = pd.read_parquet(O + 'rcm.parquet')
J = pd.read_parquet(O + 'jolpica_pitstops.parquet')
pd.set_option('display.width', 200)


def sec(s):
    return pd.to_timedelta(s).dt.total_seconds()


for df in (L, P):
    df['lt'] = sec(df['LapTime'])
    df['t_end'] = sec(df['Time'])
    df['pit_in'] = df['PitInTime'].notna()
    df['pit_out'] = df['PitOutTime'].notna()
    df['ts'] = df['TrackStatus'].astype(str).replace({'nan': '', 'None': ''})

print('== RACE LAPS ==')
print('races', L['Round'].nunique(), 'laps', len(L), 'drivers', L['Driver'].nunique(), 'teams', L['Team'].nunique())
print('stints', L.groupby(['Round', 'Driver', 'Stint']).ngroups, 'cols', L.shape[1])
print('practice laps', len(P), 'sessions', P['Round'].nunique())
miss = L[['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'Compound', 'TyreLife', 'Stint', 'Position', 'SpeedST']].isna().mean() * 100
print('missing %\n', miss.round(2).to_string())
print('compound laps\n', L['Compound'].value_counts().to_string())
print('FreshTyre False share', round((L['FreshTyre'] == False).mean() * 100, 1))
print('IsAccurate False share', round((~L['IsAccurate']).mean() * 100, 1))

# track status: any code char present on the lap
codes = {'2': 'Yellow', '4': 'SC', '5': 'Red', '6': 'VSC', '7': 'VSC ending'}
for c, n in codes.items():
    print(f'laps with {n}: {L["ts"].str.contains(c).mean()*100:.1f}%')
sc_races = L.groupby('Round')['ts'].apply(lambda s: s.str.contains('4').any()).sum()
vsc_races = L.groupby('Round')['ts'].apply(lambda s: s.str.contains('6').any()).sum()
red_races = L.groupby('Round')['ts'].apply(lambda s: s.str.contains('5').any()).sum()
print('races with SC', sc_races, 'VSC', vsc_races, 'red', red_races)
wet = W.groupby('Round')['Rainfall'].apply(lambda s: (s.astype(str) == 'True').any())
print('races with rainfall', wet.sum(), list(L.loc[L['Round'].isin(wet[wet].index), 'Event'].unique()))
print('wet tyre laps', L['Compound'].isin(['INTERMEDIATE', 'WET']).sum())
print('track temp C range', W['TrackTemp'].min(), W['TrackTemp'].max(), 'race median range',
      W.groupby('Round')['TrackTemp'].median().min(), W.groupby('Round')['TrackTemp'].median().max())

# traffic: gap to car ahead at the line, from cumulative session time
L = L.sort_values(['Round', 'LapNumber', 't_end'])
L['gap_ahead'] = L.groupby(['Round', 'LapNumber'])['t_end'].diff()

# clean lap funnel
f = pd.Series(True, index=L.index)
steps = []
def step(name, mask):
    global f
    f = f & mask
    steps.append((name, int(f.sum()), round(f.mean() * 100, 1)))
steps.append(('All race laps', len(L), 100.0))
step('Valid lap time', L['lt'].notna())
step('Drop lap 1', L['LapNumber'] > 1)
step('Drop in and out laps', ~L['pit_in'] & ~L['pit_out'])
step('Green flag only', L['ts'] == '1')
step('Dry compounds only', L['Compound'].isin(['SOFT', 'MEDIUM', 'HARD']))
step('FastF1 IsAccurate', L['IsAccurate'])
step('Not in traffic (gap > 1.5 s)', ~(L['gap_ahead'] < 1.5))
med = L[f].groupby(['Round', 'Driver'])['lt'].transform('median')
step('Within 107% of driver median', ~(L['lt'] > 1.07 * L.groupby(['Round', 'Driver'])['lt'].transform('median')))
print('\n== CLEAN LAP FUNNEL ==')
print(pd.DataFrame(steps, columns=['Step', 'Laps', 'Pct']).to_string(index=False))
print('traffic share of otherwise clean laps',
      round((L.loc[f | (L['gap_ahead'] < 1.5), 'gap_ahead'] < 1.5).mean() * 100, 1))
L['clean'] = f

# stints
st = L.groupby(['Round', 'Driver', 'Stint']).agg(comp=('Compound', 'first'), n=('LapNumber', 'size'),
                                                  clean=('clean', 'sum'), age0=('TyreLife', 'min'))
print('\n== STINTS ==')
print(st.groupby('comp')['n'].describe()[['count', 'mean', '50%', 'max']].round(1).to_string())
print('stints with >=10 clean laps', (st['clean'] >= 10).sum(), 'of', len(st))
stops = L.groupby(['Round', 'Driver'])['Stint'].max() - 1
print('stops per driver race\n', stops.value_counts().sort_index().to_string())

# pit loss (green flag, from FastF1): in lap + out lap - 2 x driver race median clean lap
dm = L[L['clean']].groupby(['Round', 'Driver'])['lt'].median().rename('ref')
L2 = L.join(dm, on=['Round', 'Driver'])
inl = L2[L2['pit_in'] & (L2['ts'] == '1')][['Round', 'Event', 'Driver', 'LapNumber', 'lt', 'ref']]
outl = L2[L2['pit_out'] & (L2['ts'] == '1')][['Round', 'Driver', 'LapNumber', 'lt']]
outl = outl.assign(LapNumber=outl['LapNumber'] - 1).rename(columns={'lt': 'lt_out'})
pl = inl.merge(outl, on=['Round', 'Driver', 'LapNumber'])
pl['loss'] = pl['lt'] + pl['lt_out'] - 2 * pl['ref']
pl = pl[pl['loss'].between(5, 60)]
by = pl.groupby('Event')['loss'].agg(['count', 'median']).sort_values('median')
print('\n== PIT LOSS (green flag) ==')
print('stops used', len(pl), 'of', int(L['pit_in'].sum()), 'pit in laps; median', round(pl['loss'].median(), 1))
print(by.round(1).to_string())

# Jolpica pit lane duration
J['dur'] = pd.to_numeric(J['duration'], errors='coerce')
print('\n== JOLPICA PIT STOPS ==')
print('rows', len(J), 'unparsed durations', J['dur'].isna().sum(), 'median', J['dur'].median(),
      'p95', J['dur'].quantile(.95).round(1))

# practice long runs: >=8 consecutive laps, no pit in/out, green, within 107% of stint median
P = P.sort_values(['Round', 'Driver', 'LapNumber'])
P['ok'] = P['lt'].notna() & ~P['pit_in'] & ~P['pit_out'] & P['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])
P['ok'] &= P['lt'] <= 1.07 * P.groupby(['Round', 'Driver', 'Stint'])['lt'].transform('median')
P['run'] = (~P['ok'] | (P.groupby(['Round', 'Driver'])['Stint'].diff() != 0)).cumsum()
runs = P[P['ok']].groupby(['Round', 'Driver', 'run']).agg(n=('lt', 'size'), comp=('Compound', 'first'))
lr = runs[runs['n'] >= 8]
print('\n== PRACTICE LONG RUNS (>=8 laps) ==')
print('long runs', len(lr), 'driver weekends covered', lr.reset_index().groupby(['Round', 'Driver']).ngroups,
      'of', P.groupby(['Round', 'Driver']).ngroups)
print(lr['comp'].value_counts().to_string())
print('per session median runs', lr.groupby('Round').size().median(), 'min', lr.groupby('Round').size().min())
sess = P.groupby('Round')['Session'].first().value_counts()
print(sess.to_string())

# results
R['grid'] = pd.to_numeric(R['GridPosition'], errors='coerce')
R['pos'] = pd.to_numeric(R['Position'], errors='coerce')
print('\n== RESULTS ==')
print('entries', len(R), 'status\n', R['Status'].value_counts().head(8).to_string())
g = R[(R['grid'] > 0) & R['pos'].notna()]
print('pit lane starts', (R['grid'] == 0).sum())
print('Spearman grid vs finish', round(g[['grid', 'pos']].corr(method='spearman').iloc[0, 1], 3),
      'grid-only MAE', round((g['grid'] - g['pos']).abs().mean(), 2))
print('rcm rows', len(M), 'categories', M['Category'].value_counts().to_dict())
