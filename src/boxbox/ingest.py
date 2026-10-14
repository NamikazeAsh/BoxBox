"""Raw pulls from FastF1 and Jolpica into data/interim, one folder per session.

Resumable: a session folder with a _done marker is skipped, so rerunning after a crash
or a rate limit only fetches what is missing.
"""
import logging
import time

import fastf1
import pandas as pd
import requests
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
    is_race = name == 'R'

    def load():
        s = fastf1.get_session(year, rnd, name)
        s.load(telemetry=False, weather=name != 'Q', messages=is_race)
        return s

    s = with_retry(load, cfg)
    if s.laps is None or len(s.laps) == 0:
        raise ValueError('no laps returned')
    out.mkdir(parents=True, exist_ok=True)
    write_parquet(s.laps.drop(columns=['LapStartDate'], errors='ignore'), out / 'laps.parquet')
    if name in ('R', 'Q'):
        write_parquet(s.results, out / 'results.parquet')
    if name != 'Q':
        write_parquet(s.weather_data, out / 'weather.parquet')
    if is_race:
        write_parquet(s.race_control_messages, out / 'rcm.parquet')
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
