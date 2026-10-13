import time, logging
from pathlib import Path
import pandas as pd, fastf1, requests

BASE = Path(__file__).resolve().parent.parent  # data-exploration/
DATA = BASE / 'data'
fastf1.Cache.enable_cache(str(BASE / 'cache'))
logging.getLogger('fastf1').setLevel(logging.ERROR)
YR = 2025
out = {k: [] for k in ['laps', 'practice', 'weather', 'results', 'rcm']}
fails = []


def tag(df, k):
    df = pd.DataFrame(df).copy()
    for c, v in k.items():
        df[c] = v
    return df


sch = fastf1.get_event_schedule(YR, include_testing=False)
for _, ev in sch.iterrows():
    k = dict(Year=YR, Round=int(ev['RoundNumber']), Event=ev['EventName'], Format=ev['EventFormat'])
    try:
        s = fastf1.get_session(YR, k['Round'], 'R')
        s.load(telemetry=False, weather=True, messages=True)
        out['laps'].append(tag(s.laps.drop(columns=['LapStartDate'], errors='ignore'), k))
        out['weather'].append(tag(s.weather_data, k))
        out['results'].append(tag(s.results, k))
        out['rcm'].append(tag(s.race_control_messages, k))
        # long run pace session: FP2, or FP1 on sprint weekends
        fp = 'FP1' if 'sprint' in str(k['Format']) else 'FP2'
        p = fastf1.get_session(YR, k['Round'], fp)
        p.load(telemetry=False, weather=True, messages=False)
        out['practice'].append(tag(p.laps.drop(columns=['LapStartDate'], errors='ignore'), {**k, 'Session': fp}))
        print(k['Round'], k['Event'], len(s.laps), len(p.laps), flush=True)
    except Exception as e:
        fails.append((k['Round'], k['Event'], repr(e)[:200]))
        print('FAIL', k['Round'], repr(e)[:200], flush=True)

for name, lst in out.items():
    df = pd.concat(lst, ignore_index=True)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    df.to_parquet(DATA / f'{name}.parquet')
sch.astype(str).to_parquet(DATA / 'schedule.parquet')

# Jolpica pit stops (stationary duration, per stop)
rows = []
for r in range(1, len(sch) + 1):
    js = requests.get(f'https://api.jolpi.ca/ergast/f1/{YR}/{r}/pitstops.json?limit=200', timeout=30).json()
    for race in js['MRData']['RaceTable']['Races']:
        for p in race['PitStops']:
            rows.append(dict(Round=r, Event=race['raceName'], **p))
    time.sleep(0.4)
pd.DataFrame(rows).to_parquet(DATA / 'jolpica_pitstops.parquet')
print('fails', fails)
print('DONE')
