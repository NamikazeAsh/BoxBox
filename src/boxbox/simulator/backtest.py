"""Replays of historical green flag dry races (no SC, VSC or red flag lap) with the simulator, scored against
what happened. Cars are the race's classified finishers on their real grid order. Variants: pace (oracle: each
car's real race base; model: drawn from the pace model) x strategies (real, or drawn from the rival pit
model) x passing rule. Finishing places are scored as outcome.ranked does (place among the classified
finishers), so the simulator compares directly with the finishing position predictor.
"""
import numpy as np
import pandas as pd

from boxbox import features, outcome, overtaking as ot, pitdecision as pdm
from boxbox.simulator import race as rc, strategy as st

RACE = ['Year', 'Round']


def green_races(laps, years):
    l = laps[laps['Year'].isin(years) & ~laps['WetRace']]
    status = l.groupby(RACE + ['Circuit'])['TrackStatus'].agg(lambda x: ''.join(x.dropna().astype(str)))
    return status[~status.str.contains('[4567]')].reset_index()[RACE + ['Circuit']]


def race_truth(laps, results, year, rnd, include_dsq=True):
    """The cars that raced to the end, in grid order: classified finishers with Finish (place among them, the target
    every model is scored on) and, with include_dsq, cars disqualified after the race that covered at least 90% of the
    distance (Dsq True, Finish NaN: they raced and are part of the field, but are not scored; China 2025 lost Leclerc,
    Hamilton and Gasly this way). Road: place on the road at the flag (laps completed, then line time). Real race time
    and gap to the winner (NaN for lapped cars)."""
    r = results[(results['Year'] == year) & (results['Round'] == rnd)].copy()
    r['Finish'] = features.classified_finish(r)
    r['Grid'] = features.grid_slot(r)
    l = laps[(laps['Year'] == year) & (laps['Round'] == rnd)]
    last = l.dropna(subset=['Time_s']).sort_values('LapNumber').groupby('Driver').last()
    done = r['Driver'].map(last['LapNumber']).fillna(0) >= 0.9 * last['LapNumber'].max()
    r['Dsq'] = r['Status'].astype(str).eq('Disqualified') & done & include_dsq
    r = r[r['Finish'].notna() | r['Dsq']]
    road = last.reindex(r['Driver']).sort_values(['LapNumber', 'Time_s'], ascending=[False, True])
    r['Road'] = r['Driver'].map({d: i + 1 for i, d in enumerate(road.index)})
    r = r.sort_values(['Grid', 'Road'])
    start = l.loc[l['LapNumber'] == 1, 'LapStart_s'].min()
    full = last['LapNumber'] == last['LapNumber'].max()
    r['RaceTime_s'] = r['Driver'].map((last['Time_s'] - start).where(full))
    r['RealGap_s'] = r['RaceTime_s'] - r['RaceTime_s'].min()
    return r[['Driver', 'Grid', 'Finish', 'Road', 'Dsq', 'RaceTime_s', 'RealGap_s']].reset_index(drop=True)


def real_status(laps, year, rnd, share=0.5):
    """The race's neutral laps as one status row (race_laps + 1,), features.neutral_laps codes, index = lap."""
    l = laps[(laps['Year'] == year) & (laps['Round'] == rnd)]
    n = features.neutral_laps(l, share)
    out = np.zeros(int(l['LapNumber'].max()) + 1, np.int8)
    out[n['LapNumber'].astype(int).to_numpy()] = n['Neutral'].to_numpy()
    return out


def real_red_laps(laps, year, rnd, share=0.5):
    """The race's red flag laps as one boolean row (race_laps + 1,), index = lap: laps where at least share of the
    cars saw track status 5 (features.neutral_laps folds red flags into the safety car code; this keeps them apart
    so a replay can make a red flag tyre change free)."""
    l = laps[(laps['Year'] == year) & (laps['Round'] == rnd)]
    red = l['TrackStatus'].fillna('').astype(str).str.contains('5').groupby(l['LapNumber']).mean() >= share
    out = np.zeros(int(l['LapNumber'].max()) + 1, bool)
    out[red.index[red.to_numpy()].astype(int)] = True
    return out


def dry_races(laps, years):
    """Every dry race of years (safety cars included), with its circuit: the v2 backtest set."""
    l = laps[laps['Year'].isin(years) & ~laps['WetRace']]
    return l[RACE + ['Circuit']].drop_duplicates().sort_values(RACE).reset_index(drop=True)


def run_race(inputs, t, pre, year, rnd, circuit, oracle_bases, policy, variants, draws, seed, cfg):
    """One race, every variant: rows per variant and car (MeanPlace, Win, sim gap, real Finish ...) and per
    variant the race's mean simulated passes. A variant is (pace, strategies, rule name, rule) or, in v2, with a
    fifth item, the neutral laps: 'green' (none, v1), 'real' (the race's own safety cars and VSCs, for replays) or
    'sample' (drawn from inputs.neutral, as before the race). Pit model strategies are drawn with the same status,
    so rivals react to the safety cars they race under."""
    truth = race_truth(t['laps'], t['results'], year, rnd)
    real = st.real_stints(t['stints'], year, rnd, truth['Driver'])
    truth = truth[truth['Driver'].isin(real['Driver'])].reset_index(drop=True)
    drivers = truth['Driver'].tolist()
    race_laps = int(t['laps'].loc[(t['laps']['Year'] == year) & (t['laps']['Round'] == rnd), 'LapNumber'].max())
    cars = truth[['Driver']].assign(Year=year, Round=rnd).merge(pre, on=['Year', 'Round', 'Driver'], how='left')
    known = cars['Driver'].map(oracle_bases).to_numpy(float)
    rng = np.random.default_rng(seed)
    new = circuit not in inputs.refs
    quali = (getattr(inputs, 'reference', 'last_season') == 'quali' and getattr(inputs, 'quali_ref', None) is not None
             and inputs.quali_ref.ref(year, rnd, circuit) is not None)
    ref = float(np.nanmedian(known)) if new and not quali else None
    new = new and not quali     # a qualifying reference makes a new circuit's race times honest too
    bases = {'model': inputs.sample_bases(cars, circuit, draws, rng, ref=ref),
             'oracle': inputs.sample_bases(cars, circuit, draws, rng, known=known, ref=ref)}
    status, free = {'green': None}, None
    teams = pd.factorize(cars['Lineage'].fillna(cars['Driver']))[0] if 'Lineage' in cars else None
    sources = {v[4] if len(v) > 4 else 'green' for v in variants}
    if 'real' in sources:
        share = cfg.get('neutral', {}).get('lap_share', 0.5)
        status['real'] = np.tile(real_status(t['laps'], year, rnd, share)[:race_laps + 1], (draws, 1))
        free = real_red_laps(t['laps'], year, rnd, share)[:race_laps + 1]
    if 'sample' in sources and getattr(inputs, 'neutral', None) is not None:
        status['sample'] = inputs.neutral.sample(circuit, race_laps, draws, np.random.default_rng(seed + 1))
    lr = t['laps'][(t['laps']['Year'] == year) & (t['laps']['Round'] == rnd)]
    red = bool(lr['TrackStatus'].astype(str).str.contains('5').any())   # real race time includes the stoppage
    starts = None
    strats = {}
    rows, passes = [], {}
    for v in variants:
        pace, strat, rule_name, rule = v[:4]
        src = v[4] if len(v) > 4 else 'green'
        if src not in status:
            continue
        if strat == 'real':
            strats[(strat, src)] = real
        elif (strat, src) not in strats:
            if policy is None:
                continue
            if starts is None:
                starts = pdm.race_starts(t['laps'], pre, t['results'], [year], cfg, green_only=False)
                starts = starts[(starts['Round'] == rnd) & starts['Driver'].isin(drivers)].reset_index(drop=True)
            model = model_strategies(policy, starts, draws, seed, status[src])
            missing = set(drivers) - set(starts['Driver'])   # no lap 1 tyre data: keep their real strategy
            strats[(strat, src)] = st.override(model, real[real['Driver'].isin(missing)]) if missing else model
        arr = st.to_arrays(strats[(strat, src)], drivers, draws)
        cover = strat == 'model' and bool(rc.engine_options(inputs).get('cover_model', False))
        res = rc.simulate(inputs, circuit, race_laps, bases[pace], arr, rule=rule, seed=seed, status=status[src], cover=cover,
                          free=free if src == 'real' else None, teams=teams)
        s = rc.summarise(res, drivers)
        name = f'{pace} pace, {strat} strategies, {rule_name}' + ('' if src == 'green' else f', {src} neutral laps')
        rows.append(s.assign(Variant=name, Year=year, Round=rnd, Circuit=circuit, NewCircuit=new, RedFlag=red, Finish=truth['Finish'].to_numpy(),
                             Grid=truth['Grid'].to_numpy(), RaceTime_s=truth['RaceTime_s'].to_numpy(),
                             RealGap_s=truth['RealGap_s'].to_numpy()))
        passes[name] = res['Passes'].mean()
        passes[name + ' | pairs'] = res['Pairs'].mean()
        passes[name + ' | pairs by tenth'] = pairs_by_tenth(res['PairsByLap'], len(drivers))
    return pd.concat(rows, ignore_index=True), passes


def model_strategies(policy, starts, draws, seed, status=None):
    """Rival strategies from the pit model; with a status array, rivals pit under the neutral laps they race under
    (strategy.model_stints with status, when the pit model supports it)."""
    if status is None:
        return st.model_stints(policy, starts, draws, seed)
    try:
        return st.model_stints(policy, starts, draws, seed, status=status)
    except TypeError:   # a pit model without the neutral reaction
        return st.model_stints(policy, starts, draws, seed)


def pairs_by_tenth(by_lap, cars):
    """Share of car laps (lap 2 on) that start within window_s of the car ahead, per tenth of the race (10 values)."""
    laps = by_lap.shape[1] - 1
    share = by_lap[:, 2:].mean(axis=0) / max(cars - 1, 1)
    tenth = np.minimum((np.arange(2, laps + 1) / laps * 10).astype(int), 9)
    return pd.Series(share).groupby(tenth).mean().reindex(range(10)).to_numpy()


def real_pairs_by_tenth(laps, year, rnd, window_s):
    """The same share in a real race: car laps from lap 2 starting within window_s of the car ahead on the same lap
    (every car, pit laps and lapped cars included, as in the engine)."""
    x = laps[(laps['Year'] == year) & (laps['Round'] == rnd)].dropna(subset=['Time_s']).sort_values(['LapNumber', 'Time_s'])
    gap = x.groupby('LapNumber')['Time_s'].diff()      # gap at the line that ends the lap = the start of the next one
    n = x['LapNumber'].max()
    close = pd.DataFrame({'Lap': x['LapNumber'] + 1, 'Close': gap < window_s})
    close = close[close['Lap'].between(2, n)]
    cars = x.groupby('LapNumber').size().reindex(close['Lap'] - 1).to_numpy()
    share = close.groupby('Lap')['Close'].sum() / (pd.Series(cars, index=close['Lap']).groupby(level=0).first() - 1)
    tenth = np.minimum((share.index.to_numpy() / n * 10).astype(int), 9)
    return share.groupby(tenth).mean().reindex(range(10)).to_numpy()


def metrics(rows):
    """Per variant: outcome.summary of the places (rank of mean simulated place among the finishers), gap to
    the winner MAE (cars on the lead lap) and winner race time error."""
    out = {}
    for v, g in rows.groupby('Variant', sort=False):
        p = outcome.ranked(g, g['MeanPlace'])
        s = outcome.summary(p)
        lead = g.dropna(subset=['RealGap_s'])
        s['Gap MAE, s'] = (lead['GapToWinner_s'] - lead['RealGap_s']).abs().mean()
        known = g[~g['NewCircuit'] & ~g.get('RedFlag', pd.Series(False, index=g.index)).fillna(False).astype(bool)]
        # a new circuit's reference came from the race itself; a red flag's stoppage is in the real race time
        sim_win = known.groupby(RACE)['Time_s'].min()
        real_win = known[known['Finish'] == 1].set_index(RACE)['RaceTime_s'].reindex(sim_win.index)
        s['Winner time error, s'] = (sim_win - real_win).mean()
        s['Winner time MAE, s'] = (sim_win - real_win).abs().mean()
        out[v] = s
    return pd.DataFrame(out).T


def race_errors(rows):
    """Per variant and race: place MAE among the classified finishers (outcome.ranked on MeanPlace)."""
    out = []
    for v, g in rows.groupby('Variant', sort=False):
        e = outcome.ranked(g, g['MeanPlace'])
        out.append(e.groupby(RACE)['Err'].apply(lambda x: x.abs().mean()).rename(v))
    return pd.concat(out, axis=1)


def paired(rows, base, others, draws=2000, seed=0):
    """Each variant in others against base, race by race: mean change in place MAE (others less base, per finisher,
    races weighted by finishers), its 95% bootstrap interval over races, and races better / worse."""
    e = race_errors(rows)
    n = rows[rows['Variant'] == base].groupby(RACE).size().reindex(e.index)
    rng = np.random.default_rng(seed)
    out = {}
    for v in others:
        ok = e[[base, v]].dropna().index
        d, w = (e.loc[ok, v] - e.loc[ok, base]).to_numpy(), n.loc[ok].to_numpy(float)
        idx = rng.integers(0, len(d), (draws, len(d)))
        boot = (d[idx] * w[idx]).sum(1) / w[idx].sum(1)
        out[v] = {'Change': (d * w).sum() / w.sum(), 'Lo': np.quantile(boot, 0.025), 'Hi': np.quantile(boot, 0.975),
                  'Better': int((d < 0).sum()), 'Worse': int((d > 0).sum()), 'Races': len(d)}
    return pd.DataFrame(out).T


def grid_metrics(rows):
    g = rows.drop_duplicates(RACE + ['Driver'])
    return outcome.summary(outcome.ranked(g, g['Grid'].astype(float)))


def real_passes(laps, lap_model, year, window_s):
    """On track passes and close pairs per race in the data (overtaking.pass_pairs)."""
    free = ot.free_air_laps(laps, lap_model, [year])
    p = ot.pass_pairs(laps.loc[free.index], free, window_s)
    return p.groupby(RACE)['Pass'].agg(Passes='sum', Pairs='size')


def sample_status(inputs, circuit, race_laps, draws, seed):
    """Neutral laps drawn before the race (inputs.neutral.sample, seed + 1 as run_race), or None with no neutral model:
    the pre race view the strategy search and the Strategy Room race under."""
    if getattr(inputs, 'neutral', None) is None:
        return None
    return inputs.neutral.sample(circuit, race_laps, draws, np.random.default_rng(seed + 1))


def strategy_search(inputs, policy, t, pre, year, rnd, circuit, driver, cfg, draws, use_caps=True, seed=0):
    """The strategy search (race.best_strategies) for one car of a past race: model pace, rivals from the pit
    model (real strategies for cars with no lap 1 tyre data), the car's real start tyre, and its real strategy
    raced too (row with Extra). Returns the ranked table."""
    laps = t['laps']
    truth = race_truth(laps, t['results'], year, rnd)
    real = st.real_stints(t['stints'], year, rnd, truth['Driver'])
    truth = truth[truth['Driver'].isin(real['Driver'])].reset_index(drop=True)
    drivers = truth['Driver'].tolist()
    race_laps = int(laps.loc[(laps['Year'] == year) & (laps['Round'] == rnd), 'LapNumber'].max())
    cars = truth[['Driver']].assign(Year=year, Round=rnd).merge(pre, on=['Year', 'Round', 'Driver'], how='left')
    rng = np.random.default_rng(seed)
    ref = None
    quali = (getattr(inputs, 'reference', 'last_season') == 'quali' and getattr(inputs, 'quali_ref', None) is not None
             and inputs.quali_ref.ref(year, rnd, circuit) is not None)
    if circuit not in inputs.refs and not quali:
        b = inputs.lap.driver_bases(inputs.lap.stint_levels(laps, [year]))
        ref = float(b[(b['Year'] == year) & (b['Round'] == rnd)]['Base'].median())
    bases = inputs.sample_bases(cars, circuit, draws, rng, ref=ref)
    status = sample_status(inputs, circuit, race_laps, draws, seed)   # v2: the search races sampled safety cars
    starts = pdm.race_starts(laps, pre, t['results'], [year], cfg, green_only=False)
    starts = starts[(starts['Round'] == rnd) & starts['Driver'].isin(drivers)].reset_index(drop=True)
    rivals = model_strategies(policy, starts, draws, seed, status)
    missing = set(drivers) - set(starts['Driver'])
    rivals = st.override(rivals, real[real['Driver'].isin(missing)]) if missing else rivals
    mine = real[real['Driver'] == driver].sort_values('Stint')
    actual = (tuple(mine['Compound']), tuple(int(x) - 1 for x in mine['StartLap'].iloc[1:]))
    w = mine.iloc[0]
    free = cfg['simulator'].get('search', {}).get('start_tyre', 'real') == 'free'   # since 2022 the start tyre is a free choice
    best = rc.best_strategies(inputs, circuit, race_laps, drivers, bases, rivals, driver, cfg, start_age=float(w['StartAge']),
                              start_compounds=None if free else [w['Compound']],
                              start_ages={w['Compound']: float(w['StartAge'])} if free else None,
                              extra=[actual], use_caps=use_caps, seed=seed, status=status)
    return best, actual, int(truth.loc[truth['Driver'] == driver, 'Grid'].iloc[0])


def cap_coverage(stints, events, caps, wet, years):
    """Share of real stints of years within the caps (tyre age at the stint's end)."""
    s = stints[stints['Year'].isin(years) & stints['Compound'].isin(st.COMPOUNDS)]
    s = s[[(y, r) not in wet for y, r in zip(s['Year'], s['Round'])]].merge(events[['Year', 'Round', 'Circuit']], on=['Year', 'Round'])
    end = s['TyreLifeStart'].fillna(1.0) + s['Laps'] - 1
    cap = [st.caps_for(caps, c)[k] for c, k in zip(s['Circuit'], s['Compound'])]
    return float((end <= np.array(cap)).mean())


# Decision level check (simulator v2.1): undercut attempts in real races, raced from the real state before the stop

def undercut_events(laps, years, cfg, gap_max=3.0, react_laps=3):
    """Green flag undercut attempts in the dry races of years: car B pits at the end of lap L (a tyre change,
    features.pit_labels) while it started that lap within gap_max of car A, and A, which had not stopped in the
    react_laps laps before, pits within react_laps laps after. Every lap from L - 1 to two laps after A's stop is
    green for both. Success: B is ahead of A at the line that ends A's out lap. One row per attempt with Gap, the
    tyre ages on lap L, the compounds fitted and the lag (A's stop less B's)."""
    l = laps[laps['Year'].isin(years) & ~laps['WetRace']]
    l = l.join(features.pit_labels(l, cfg)[['PitStop']])
    l = l.dropna(subset=['Time_s']).sort_values(RACE + ['LapNumber', 'Time_s'])
    l = l.assign(Ahead=l.groupby(RACE + ['LapNumber'])['Driver'].shift(1), Gap=l.groupby(RACE + ['LapNumber'])['Time_s'].diff())
    key = l.set_index(RACE + ['Driver', 'LapNumber'])
    nxt = l.sort_values(RACE + ['Driver', 'LapNumber']).groupby(RACE + ['Driver'])['Compound'].shift(-1)
    key['NextCompound'] = nxt.reindex(l.index).to_numpy()
    stops = key[key['PitStop'].astype(bool)].reset_index()
    by_car = stops.groupby(RACE + ['Driver'])['LapNumber'].apply(lambda x: sorted(int(v) for v in x)).to_dict()
    rows = []
    for r in stops.itertuples():
        L = int(r.LapNumber)
        if (r.Year, r.Round, r.Driver, L - 1) not in key.index:
            continue
        prev = key.loc[(r.Year, r.Round, r.Driver, L - 1)]
        a, gap = prev['Ahead'], prev['Gap']
        if not isinstance(a, str) or not gap < gap_max:
            continue
        sa = by_car.get((r.Year, r.Round, a), [])
        if any(L - react_laps <= x <= L for x in sa):
            continue
        resp = [x for x in sa if L < x <= L + react_laps]
        if not resp:
            continue
        la, end = resp[0], resp[0] + 2
        cells = [(r.Year, r.Round, d, w) for d in (r.Driver, a) for w in range(L - 1, end + 1)]
        if any(c not in key.index or key.loc[c, 'TrackStatus'] != '1' for c in cells):
            continue
        b0, a0 = key.loc[(r.Year, r.Round, r.Driver, L)], key.loc[(r.Year, r.Round, a, L)]
        rows.append({'Year': r.Year, 'Round': r.Round, 'Circuit': b0['Circuit'], 'B': r.Driver, 'A': a, 'StopB': L, 'StopA': la,
                     'Lag': la - L, 'Gap': gap, 'AgeB': b0['TyreLife'], 'AgeA': a0['TyreLife'],
                     'NewB': key.loc[(r.Year, r.Round, r.Driver, L), 'NextCompound'], 'NewA': key.loc[(r.Year, r.Round, a, la), 'NextCompound'],
                     'End': end, 'Success': key.loc[(r.Year, r.Round, r.Driver, end), 'Time_s'] < key.loc[(r.Year, r.Round, a, end), 'Time_s']})
    return pd.DataFrame(rows)


def real_state(laps, real, drivers, year, rnd, lap):
    """The race state at the line that ends lap (simulate's start): race time per car (Time_s less the race start),
    the stint each car is on for the next lap, its tyre age then, and whether the next lap is an out lap, from the
    real stints (strategy.real_stints). Cars in the order of drivers."""
    l = laps[(laps['Year'] == year) & (laps['Round'] == rnd)]
    t0 = l.loc[l['LapNumber'] == 1, 'LapStart_s'].min()
    at = l[l['LapNumber'] == lap].set_index('Driver')['Time_s']
    T, k, age, out = [], [], [], []
    for d in drivers:
        st = real[real['Driver'] == d].sort_values('Stint')
        i = int((st['StartLap'] <= lap + 1).sum()) - 1
        row = st.iloc[i]
        T.append(at.get(d, np.nan) - t0)
        k.append(i)
        age.append(float(row['StartAge']) + (lap + 1 - float(row['StartLap'])))
        out.append(i > 0 and int(row['StartLap']) == lap + 1)
    T = np.array(T, float)
    T = np.where(np.isnan(T), np.nanmax(T) + 60.0, T)   # a car with no time at the line (lapped, retired) runs at the back
    return {'lap': lap, 'T': T, 'k': np.array(k), 'age': np.array(age), 'out': np.array(out)}


def undercut_prob(inputs, t, pre, ev, oracle_bases, draws, seed, kind='oracle'):
    """P(B ahead of A at the line after A's out lap), raced from the real state at the end of lap StopB - 1 with
    every car on its real strategy (the decision is B's stop; A's answer is the one A made). kind: oracle (each
    car's real race base) or model (the pace model)."""
    truth = race_truth(t['laps'], t['results'], ev.Year, ev.Round)
    real = st.real_stints(t['stints'], ev.Year, ev.Round)
    drivers = [d for d in real['Driver'].unique() if d in set(truth['Driver']) or d in (ev.A, ev.B)]
    race_laps = int(t['laps'].loc[(t['laps']['Year'] == ev.Year) & (t['laps']['Round'] == ev.Round), 'LapNumber'].max())
    cars = pd.DataFrame({'Driver': drivers}).assign(Year=ev.Year, Round=ev.Round).merge(pre, on=['Year', 'Round', 'Driver'], how='left')
    rng = np.random.default_rng(seed)
    known = cars['Driver'].map(oracle_bases).to_numpy(float) if kind == 'oracle' else None
    ref = float(np.nanmedian(cars['Driver'].map(oracle_bases))) if ev.Circuit not in inputs.refs else None
    bases = inputs.sample_bases(cars, ev.Circuit, draws, rng, known=known, ref=ref)
    arr = st.to_arrays(real[real['Driver'].isin(drivers)], drivers, draws)
    state = real_state(t['laps'], real, drivers, ev.Year, ev.Round, int(ev.StopB) - 1)
    res = rc.simulate(inputs, ev.Circuit, race_laps, bases, arr, seed=seed, start=state, last_lap=int(ev.End))
    b, a = drivers.index(ev.B), drivers.index(ev.A)
    return float((res['Time'][:, b] < res['Time'][:, a]).mean())


# The 2026 shift backtest (notes/shift_study.md, measure 5)

TABLES = ['laps', 'results', 'qualifying', 'sprint_results', 'events', 'pit_loss_stops', 'stints']


def load_tables(cfg, holdout=False):
    """The tables a backtest reads; holdout=True appends data/processed/holdout (the shift study only)."""
    from boxbox import shift
    return shift.load_tables(cfg, holdout=holdout, names=TABLES)


def shift_folds(cfg):
    """[(train 2022 to 2025, 2026)]: the shift study's split (config training_seasons)."""
    from boxbox.config import seasons_for
    return [(seasons_for(cfg, 'shift_study_train'), seasons_for(cfg, 'shift_study_test')[0])]


def config_record(cfg):
    """The simulator and safety car settings as YAML, for the top of a scored printout."""
    import yaml
    return yaml.safe_dump({'simulator': cfg['simulator'], 'neutral': cfg.get('neutral')}, sort_keys=False, default_flow_style=None)


def setup_check(inputs, policy, t, pre, year, races, bases, variants, cfg, draws=4, seed=0):
    """Every race set up and raced once with a few draws, before any scoring: one row per race with its status
    ('ok' or the error that the scoring loop would skip it for) and whether its circuit has training laps. Prints
    no errors against the real result."""
    out = []
    for race in races.itertuples():
        ob = bases[(bases['Year'] == year) & (bases['Round'] == race.Round)].set_index('Driver')['Base']
        try:
            run_race(inputs, t, pre, year, race.Round, race.Circuit, ob, policy, variants, draws, seed, cfg)
            status = 'ok'
        except (KeyError, ValueError, IndexError) as e:
            status = f'{type(e).__name__}: {e}'
        out.append({'Round': race.Round, 'Circuit': race.Circuit, 'NewCircuit': race.Circuit not in inputs.refs,
                    'PitLossHistory': race.Circuit in getattr(inputs.pit, 'pools', {}), 'Status': status})
    return pd.DataFrame(out)
