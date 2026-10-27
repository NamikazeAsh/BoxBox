"""Raw pulls from FastF1 and Jolpica into data/interim, one folder per session.

Resumable: a session folder with a _done marker is skipped, so rerunning after a crash
or a rate limit only fetches what is missing.
"""
import logging
import time

import fastf1
import pandas as pd
import requests
from fastf1.mvapi.api import get_circuit
from fastf1.req import RateLimitExceededError

log = logging.getLogger('boxbox.ingest')


def practice_session(event):
    # Sprint weekends only have FP1
    return 'FP1' if 'sprint' in str(event['EventFormat']) else 'FP2'


def session_dir(cfg, year, rnd, name):
    return cfg['paths']['interim'] / str(year) / f'{rnd:02d}_{name}'


def write_parquet(df, path):
    df = pd.DataFrame(df)
    try:
        df.to_parquet(path)
    except Exception:
        # Mixed type object columns (str and None and numbers) trip pyarrow
        obj = df.select_dtypes('object').columns
        df[obj] = df[obj].astype(str)
        df.to_parquet(path)


def with_retry(fn, cfg):
    tries = cfg['ingest']['rate_limit_tries']
    for i in range(tries):
        try:
            return fn()
        except RateLimitExceededError:
            wait = cfg['ingest']['rate_limit_wait_s']
            log.warning('rate limited, sleeping %ss (try %s of %s)', wait, i + 1, tries)
            time.sleep(wait)
    return fn()


def pull_session(cfg, year, event, name):
    rnd = int(event['RoundNumber'])
    out = session_dir(cfg, year, rnd, name)
    if (out / '_done').exists():
        return 'cached'
    is_race = name in ('R', 'S')   # sprints race too: messages, results, weather

    def load():
        s = fastf1.get_session(year, rnd, name)
        s.load(telemetry=False, weather=name != 'Q', messages=is_race)
        return s

    s = with_retry(load, cfg)
    if s.laps is None or len(s.laps) == 0:
        raise ValueError('no laps returned')
    out.mkdir(parents=True, exist_ok=True)
    write_parquet(s.laps.drop(columns=['LapStartDate'], errors='ignore'), out / 'laps.parquet')
    if name in ('R', 'S', 'Q'):
        write_parquet(s.results, out / 'results.parquet')
    if name != 'Q':
        write_parquet(s.weather_data, out / 'weather.parquet')
    if is_race:
        write_parquet(s.race_control_messages, out / 'rcm.parquet')
        if name == 'R':
            write_parquet(pull_jolpica_pitstops(cfg, year, rnd), out / 'pitstops.parquet')
    (out / '_done').touch()
    return 'pulled'


def pull_jolpica_pitstops(cfg, year, rnd):
    url = f"{cfg['ingest']['jolpica_url']}/{year}/{rnd}/pitstops.json?limit=200"
    for attempt in range(5):
        r = requests.get(url, timeout=30)
        if r.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        break
    time.sleep(cfg['ingest']['jolpica_delay_s'])
    races = r.json()['MRData']['RaceTable']['Races']
    rows = [p for race in races for p in race['PitStops']]
    return pd.DataFrame(rows, columns=['driverId', 'lap', 'stop', 'time', 'duration'])


def track_map_races(events, cfg):
    """The latest race at each circuit outside the holdout seasons: the layout the outline is drawn from."""
    e = events[~events['Year'].isin(cfg['holdout_seasons'])].sort_values(['Year', 'Round'])
    return e.groupby('Circuit').tail(1)[['Circuit', 'Year', 'Round']].reset_index(drop=True)


def pull_track_outline(cfg, circuit, year, rnd):
    """Circuit outline X, Y and map rotation from the MultiViewer circuit API (the source of FastF1's circuit
    info), traced from one real lap, so the first point is the finish line. One small request per circuit
    and no telemetry; the live timing archive started refusing telemetry with 401 (data_issues.md).
    One file per race, so reruns skip it."""
    out = cfg['paths']['interim'] / 'track_maps' / f'{year}_{rnd:02d}.parquet'
    if out.exists():
        return 'cached'

    def load():
        s = fastf1.get_session(year, rnd, cfg['ingest']['track_map_session'])
        s.load(laps=False, telemetry=False, weather=False, messages=False)
        return s

    key = with_retry(load, cfg).session_info['Meeting']['Circuit']['Key']
    c = get_circuit(year=year, circuit_key=key)
    if not c or not c.get('x'):
        raise ValueError(f'no outline from MultiViewer for circuit key {key}')
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'X': c['x'], 'Y': c['y']}).assign(Circuit=circuit, Year=year, Round=rnd,
                                                    Rotation=float(c['rotation'])).to_parquet(out)
    return 'pulled'


def pull_season(cfg, year, sessions=None):
    sessions = sessions or cfg['ingest']['sessions']
    sched = fastf1.get_event_schedule(year, include_testing=False)
    sched_dir = cfg['paths']['interim'] / str(year)
    sched_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(sched, sched_dir / 'schedule.parquet')

    now = pd.Timestamp.now(tz='UTC').tz_localize(None)
    failures = []
    for _, ev in sched.iterrows():
        for sess in sessions:
            name = practice_session(ev) if sess == 'FP' else sess
            try:
                when = ev.get_session_date(name, utc=True)
            except ValueError:
                continue  # session not part of this weekend
            if pd.isna(when) or when > now:
                continue
            try:
                status = pull_session(cfg, year, ev, name)
                log.info('%s R%02d %s %s', year, ev['RoundNumber'], name, status)
            except Exception as e:
                log.error('%s R%02d %s failed: %r', year, ev['RoundNumber'], name, e)
                failures.append(dict(Year=year, Round=int(ev['RoundNumber']), Event=ev['EventName'],
                                     Session=name, Error=repr(e)[:300]))
    return failures
