"""Wet races in the simulator as a scenario tool, not a forecast (notes/wet_simulation.md, config simulator.wet).
There is no rain forecast input, so the track is given: by the user (scenario) or estimated from a real race
(replay).

Track: gap(lap), the wet tyre minus slick lap time gap, s (negative on a wet track: the wet tyre is faster). Every
car shares the track, so a common wet slowdown cancels in the places; what a strategy turns on is the gap, the pit
loss and wear. In the engine (race.simulate wet) a slick lap is the dry free air lap plus a common term, and a wet
tyre lap is a fresh Medium's free air lap plus the common term plus gap(lap) plus wet wear x (age - 1). The
intermediate and the full wet are one wet tyre (D30: full wets have too little green running to tell apart).

Scenario: the user names the laps the state changes ('dry,wet@20,dry@35': intermediates faster from lap 20,
slicks faster again from lap 35). Around each change the gap is a line through zero half a lap before it, with a
slope drawn per Monte Carlo draw from the wet study's fitted crossover trends (simulator.wet.trends), clipped to a
dry and a wet level. Rivals keep their dry plan and switch tyres in a wave: each car's switch lap is the change
lap plus an offset drawn from the wet study's real switches (switch_offsets). search_switches ranks the searched
car's switch laps (and its slick compound after the last change) against that field, common random numbers.

Replay: race_track fits, per real race, lap time less the dry model's terms = driver base + lap effect + wet tyre x
gap on laps where both tyre groups run (least squares). The lap effect is the common term; laps where only one
group runs are interpolated over short gaps, or else take the field share path (the scenario line through the
laps where half the field is on each group). Bases from the same fit are the replay's oracle pace.
"""
import numpy as np
import pandas as pd

from boxbox import features
from boxbox.degradation.task import COMPOUNDS
from boxbox.simulator import backtest as bt, race as rc, strategy as st

WET = 'INTERMEDIATE'          # the wet tyre's name in strategies built here
WET_CODE = len(COMPOUNDS)     # its compound index in the engine


def settings(cfg):
    return cfg['simulator']['wet']


def enabled(cfg):
    """simulator.wet.enabled (off by default: callers race the dry engine unchanged)."""
    return bool(cfg['simulator'].get('wet', {}).get('enabled', False))


def codes(cfg):
    """Compound to engine index, the wet tyres included."""
    return {**st.CODE, **{c: WET_CODE for c in settings(cfg)['wet_compounds']}}


def group_of(compound, cfg):
    """'slick', 'wet' or NaN per compound."""
    c = pd.Series(compound)
    return pd.Series(np.where(c.isin(COMPOUNDS), 'slick', np.where(c.isin(settings(cfg)['wet_compounds']), 'wet', None)),
                     index=c.index)


def engine_track(gap, cfg, common=0.0):
    """The engine's wet argument (race.simulate) for a gap path (draws x race_laps + 1, or one row)."""
    w = settings(cfg)
    return {'gap': np.asarray(gap, float), 'common': np.asarray(common, float), 'wear': float(w['wear_s_per_lap']),
            'ref': st.CODE[w['ref_compound']], 'hardness': float(w['hardness'])}


# Scenario track -------------------------------------------------------------------------------------------------

def parse_path(text):
    """'dry,wet@20,dry@35' to ('dry', [(20, 'wet'), (35, 'dry')]): the state on lap 1, then each lap from which the
    other tyre is faster. States alternate and laps increase."""
    parts = [p.strip().lower() for p in str(text).split(',') if p.strip()]
    if not parts or parts[0] not in ('dry', 'wet'):
        raise ValueError(f'a path starts with dry or wet: {text!r}')
    state, out = parts[0], []
    for p in parts[1:]:
        s, lap = p.split('@')
        if s == state or s not in ('dry', 'wet'):
            raise ValueError(f'states must alternate: {text!r}')
        if out and int(lap) <= out[-1][0]:
            raise ValueError(f'change laps must increase: {text!r}')
        out.append((int(lap), s))
        state = s
    return parts[0], out


def scenario_gap(race_laps, start, changes, trends, cfg):
    """Gap path, draws x race_laps + 1 (index = lap). changes: [(lap, state)] (parse_path); trends: draws x
    len(changes) positive slopes, s per lap per lap. A change at lap c is a line through zero at c - 0.5 (lap c is the
    first lap the new state's tyre is faster), rising toward dry and falling toward wet; phases combine as the
    larger (wet phase) or smaller (dry phase) of the lines either side, clipped to [wet_level_s, dry_level_s]."""
    w = settings(cfg)
    lo, hi = float(w['wet_level_s']), float(w['dry_level_s'])
    trends = np.atleast_2d(np.asarray(trends, float))
    laps = np.arange(race_laps + 1)[None, :]
    gap = np.full((trends.shape[0], race_laps + 1), hi if start == 'dry' else lo)
    for i, (c, s) in enumerate(changes):
        line = (1.0 if s == 'dry' else -1.0) * trends[:, i:i + 1] * (laps - c + 0.5)
        gap = line if i == 0 else (np.minimum(gap, line) if s == 'wet' else np.maximum(gap, line))
    return np.clip(gap, lo, hi)


def draw_trends(rng, draws, n, cfg):
    return rng.choice(np.abs(np.asarray(settings(cfg)['trends'], float)), size=(draws, n))


def draw_offsets(rng, size, cfg):
    """Rival switch offsets, laps off the change lap, from the wet study's real switches."""
    o = settings(cfg)['switch_offsets']
    k = np.array([int(x) for x in o.keys()])
    p = np.array(list(o.values()), float)
    return rng.choice(k, size=size, p=p / p.sum())


# Strategies on a wet track ---------------------------------------------------------------------------------------

def plan_of(stints, driver, draw=0):
    """A car's dry plan, [(compound, start lap, start age)], from a stints frame."""
    s = stints[(stints['Driver'] == driver) & (stints['Draw'] == draw)].sort_values('StartLap')
    return list(zip(s['Compound'], s['StartLap'].astype(int), s['StartAge'].astype(float)))


def overlay(plan, start, switches, race_laps, cfg, after=None, later_stops=True):
    """One car's stints, [(compound, start lap, start age)], on its dry plan with wet switches. start: the state on
    lap 1 ('dry' or 'wet'); switches: [(lap, state)], from lap on the car runs that state's tyre. A switch to slicks
    fits a new set of after (or the plan's compound on that lap); a planned dry stop is kept only while the car is on
    slicks and at least min_stint_laps from every switch (later_stops False: none after the last switch). Switches
    outside laps 2 to race_laps are dropped."""
    m = int(settings(cfg)['min_stint_laps'])
    sw = [(int(l), s) for l, s in switches if 2 <= l <= race_laps]
    state_at = lambda lap: ([start] + [s for l, s in sw if l <= lap])[-1]
    comp_at = lambda lap: [c for c, l, _ in plan if l <= lap][-1]
    ev = {1: (WET, 1.0) if start == 'wet' else (plan[0][0], plan[0][2])}
    for c, l, a in plan[1:]:
        late = sw and l > sw[-1][0] and not later_stops
        if state_at(l) == 'dry' and all(abs(l - x) >= m for x, _ in sw) and l <= race_laps and not late:
            ev[l] = (c, a)
    for l, s in sw:
        ev[l] = (WET, 1.0) if s == 'wet' else (after or comp_at(l), 1.0)
    return [(c, l, a) for l, (c, a) in sorted(ev.items())]


def rows_of(stints, driver, draw):
    return [(draw, driver, i + 1, c, float(l), float(a)) for i, (c, l, a) in enumerate(stints)]


def frame(rows):
    return pd.DataFrame(rows, columns=st.STINTS)


def rival_switches(start, changes, offsets, race_laps, cfg):
    """One car's switches for one draw: each change lap plus its offset, kept in order; a detour (a wet or dry
    spell) shorter than min_wet_laps is skipped (the car stays on the tyre it is on)."""
    mw = int(settings(cfg)['min_wet_laps'])
    laps = [min(max(int(c + o), 2), race_laps + 1) for (c, _), o in zip(changes, offsets)]
    out, i = [], 0
    while i < len(changes):
        lap, s = laps[i], changes[i][1]
        if i + 1 < len(changes) and laps[i + 1] - lap < mw:
            i += 2
            continue
        if out and lap <= out[-1][0]:
            lap = out[-1][0] + 1
        out.append((lap, s))
        i += 1
    return out


def finish_on(stints, lap, comp):
    """stints with the stint starting on lap fitted with comp."""
    return [(comp if l == lap else c, l, a) for c, l, a in stints]


def replan(plan, start, sw, race_laps, cfg, caps=None):
    """A rival's stints for its switches: after its last switch to slicks it runs to the flag on the softest
    compound whose stint cap (caps, {compound: laps}) covers the laps left, with no more stops; when none does, or
    with no caps, it resumes its dry plan on a new set."""
    if caps and sw and sw[-1][1] == 'dry':
        left = race_laps - sw[-1][0] + 1
        fit = [c for c in COMPOUNDS if caps.get(c, np.inf) >= left]
        if fit:
            return finish_on(overlay(plan, start, sw, race_laps, cfg, later_stops=False), sw[-1][0], fit[0])
    return overlay(plan, start, sw, race_laps, cfg)


def scenario_stints(plans, start, changes, draws, race_laps, rng, cfg, caps=None):
    """Every car's stints per draw on a scenario track: its dry plan with a rival switch wave (rival_switches,
    replan). plans: {driver: plan}; caps: stint caps at the circuit (strategy.caps_for)."""
    rows = []
    off = draw_offsets(rng, (draws, len(plans), len(changes)), cfg)
    for i, (drv, plan) in enumerate(plans.items()):
        for d in range(draws):
            sw = rival_switches(start, changes, off[d, i], race_laps, cfg)
            rows += rows_of(replan(plan, start, sw, race_laps, cfg, caps), drv, d)
    return frame(rows)


def lap_table(stints, race_laps):
    """Per lap 1..race_laps of one car's stints: compound, tyre age and whether a stop ends the lap."""
    comp, age, stop = np.empty(race_laps + 1, object), np.zeros(race_laps + 1), np.zeros(race_laps + 1, bool)
    for i, (c, l, a) in enumerate(stints):
        end = stints[i + 1][1] - 1 if i + 1 < len(stints) else race_laps
        comp[l:end + 1], age[l:end + 1] = c, a + np.arange(end - l + 1)
        if i + 1 < len(stints):
            stop[end] = True
    return comp[1:], age[1:], stop[1:]


def free_air_time(inputs, circuit, race_laps, base, stints, gap_row, cfg):
    """One car alone on a strategy: free air race time with the median pit loss and no noise (the coarse pass)."""
    lm, w = inputs.lap, settings(cfg)
    comp, age, stop = lap_table(stints, race_laps)
    code = pd.Series(comp).map(codes(cfg)).to_numpy(int)
    on = code == WET_CODE
    laps = np.arange(1, race_laps + 1)
    ref = st.CODE[w['ref_compound']]
    t = lm.free_air(base, circuit, np.where(on, ref, code), np.where(on, 1.0, age), laps, race_laps)
    t = t + np.where(on, np.asarray(gap_row)[laps] + w['wear_s_per_lap'] * (age - 1), 0.0)
    return float(t.sum() + stop.sum() * float(inputs.pit.predict([circuit])[0]))


def label(stints):
    return ', '.join(f'{c[0] if c != WET else "I"} from {l}' for c, l, _ in stints)


def race_candidates(inputs, circuit, race_laps, bases, rivals, user, drivers, cands, track, cfg, status=None,
                    free=None, seed=0):
    """Race each candidate strategy of the user's car (cands: {label: stints list}) against the same rivals, noise
    and track (common random numbers). One row per candidate, best (lowest mean place, then time) first, with the
    place and time differences to the best draw by draw and a near tie tier as race.best_strategies."""
    r = cfg['simulator']['race']
    D = bases.shape[0]
    u = list(drivers).index(user)
    out, places, times = [], [], []
    for name, s in cands.items():
        mine = frame([row for d in sorted(rivals['Draw'].unique()) for row in rows_of(s, user, d)])
        arr = st.to_arrays(pd.concat([rivals[rivals['Driver'] != user], mine]), drivers, D, codes(cfg))
        res = rc.simulate(inputs, circuit, race_laps, bases, arr, seed=seed, status=status, free=free, wet=track)
        P, T = res['Place'][:, u].astype(float), res['Time'][:, u]
        places.append(P)
        times.append(T)
        out.append({'Strategy': name, 'Stops': len(s) - 1, 'MeanPlace': P.mean(), 'Place10': np.quantile(P, 0.1),
                    'Place90': np.quantile(P, 0.9), 'Win': (P == 1).mean(), 'Podium': (P <= 3).mean(),
                    'Points': (P <= 10).mean(), 'Time_s': np.median(T)})
    out = pd.DataFrame(out)
    order = out.sort_values(['MeanPlace', 'Time_s']).index.to_numpy()
    out = out.loc[order].reset_index(drop=True)
    P, T = np.array(places)[order], np.array(times)[order]
    d = P - P[0]
    se = d.std(axis=1, ddof=1) / np.sqrt(D) if D > 1 else np.zeros(len(d))
    out['DeltaPlace'] = d.mean(axis=1)
    out['DeltaLo'], out['DeltaHi'] = out['DeltaPlace'] - 1.96 * se, out['DeltaPlace'] + 1.96 * se
    out['DeltaTime_s'] = np.median(T - T[0], axis=1)
    near = (out['DeltaPlace'] <= r['tie_places']) | (out['DeltaLo'] <= 0)
    out['Tier'] = np.where(out.index == 0, 'best', np.where(near, 'near tie', 'worse'))
    return out


def coarse(inputs, circuit, race_laps, base, cands, gap_row, cfg, keep=()):
    """The refine_top candidates by free air time, plus any named in keep."""
    t = {k: free_air_time(inputs, circuit, race_laps, base, s, gap_row, cfg) for k, s in cands.items()}
    top = sorted(t, key=t.get)[:int(settings(cfg)['refine_top'])]
    return {k: cands[k] for k in dict.fromkeys(top + [k for k in keep if k in cands])}, t


def switch_candidates(plan, start, changes, race_laps, cfg):
    """The searched car's strategies on a scenario: each change lap plus or minus search_laps (in order, wet or dry
    spells at least min_wet_laps), the slick compound after the last change to dry, with or without the plan's
    later dry stops, and staying out (no switch)."""
    from itertools import product
    w = settings(cfg)
    k = int(w['search_laps'])
    grids = [range(max(2, c - k), min(race_laps, c + k) + 1) for c, _ in changes]
    last_dry = max([i for i, (_, s) in enumerate(changes) if s == 'dry'], default=None)
    cands = {'no switch: ' + label(overlay(plan, start, [], race_laps, cfg)): overlay(plan, start, [], race_laps, cfg)}
    for laps in product(*grids):
        if any(b - a < int(w['min_wet_laps']) for a, b in zip(laps, laps[1:])):
            continue
        sw = [(l, s) for l, (_, s) in zip(laps, changes)]
        for later in (True, False):
            s0 = overlay(plan, start, sw, race_laps, cfg, later_stops=later)
            for comp in (COMPOUNDS if last_dry is not None else [None]):
                # the last switch to slicks on a chosen compound
                s = s0 if comp is None else finish_on(s0, sw[last_dry][0], comp)
                cands.setdefault(label(s), s)
    return cands


def search_switches(inputs, circuit, race_laps, bases, rivals, user, drivers, plan, start, changes, track, cfg,
                    status=None, seed=0, caps=None):
    """Rank the user's switch laps on a scenario track against the rival wave (switch_candidates, a free air pass,
    then the best refine_top raced; the switch at exactly each change lap and staying out are always raced)."""
    cands = switch_candidates(plan, start, changes, race_laps, cfg)
    gap_row = np.median(np.asarray(track['gap'], float).reshape(-1, race_laps + 1), axis=0)
    on_time = label(replan(plan, start, changes, race_laps, cfg, caps))
    base = float(np.median(bases[:, list(drivers).index(user)]))
    pick, t = coarse(inputs, circuit, race_laps, base, cands, gap_row, cfg, keep=[on_time, next(iter(cands))])
    out = race_candidates(inputs, circuit, race_laps, bases, rivals, user, drivers, pick, track, cfg, status=status, seed=seed)
    return out.assign(FreeAir_s=out['Strategy'].map(t), Candidates=len(cands))


# Replay track ----------------------------------------------------------------------------------------------------

def race_laps_of(laps, year, rnd):
    return laps[(laps['Year'] == year) & (laps['Round'] == rnd)]


def fit_track(l, lm, circuit, race_laps, cfg, group_terms=True):
    """One race's clean laps, lap time less the dry model's terms (a wet tyre as a fresh ref compound plus wet
    wear), in two steps: driver base + lap effect by least squares on the laps where one tyre group ran; then on
    each lap both groups ran (min_cars each), the slick level and the gap as medians of the cars' residuals from
    their bases. Returns (gap Series by measured lap, lap effect Series by lap with data, bases Series by driver, wet
    share of the fitted laps by lap). group_terms False: one step on every lap, no gap (the no track path baseline)."""
    w, r = settings(cfg), settings(cfg)['replay']
    d = l.assign(Group=group_of(l['Compound'], cfg).to_numpy()).sort_values(['Driver', 'LapNumber'])
    # tyre age where the timing feed lost it: laps into the stint (a used set counts as new)
    d['TyreLife'] = d['TyreLife'].fillna(d.groupby(['Driver', 'Stint']).cumcount() + 1.0)
    st_ = d['TrackStatus'].fillna('').astype(str)
    d = d[d['LapTime_s'].notna() & (d['LapNumber'] > 1) & ~d['PitIn'].astype(bool) & ~d['PitOut'].astype(bool)
          & d['Group'].notna() & ~st_.str.contains('[4-7]')]
    n = features.neutral_laps(l, cfg.get('neutral', {}).get('lap_share', 0.5))
    d = d[~d['LapNumber'].isin(n.loc[n['Neutral'] > 0, 'LapNumber'])]    # laps the field ran neutral (a car's own status can miss it)
    med = d.groupby(['LapNumber', 'Group'])['LapTime_s'].transform('median')
    d = d[d['LapTime_s'] <= r['outlier_pct'] * med]
    on = (d['Group'] == 'wet').to_numpy()
    code = d['Compound'].map(st.CODE).fillna(st.CODE[w['ref_compound']]).to_numpy(int)
    age = d['TyreLife'].to_numpy(float)
    lap = d['LapNumber'].to_numpy(int)
    dry = lm.free_air(0.0, circuit, code, np.where(on, 1.0, age), lap, race_laps)
    y = d['LapTime_s'].to_numpy(float) - dry - np.where(on, w['wear_s_per_lap'] * (age - 1), 0.0)
    n = d.groupby(['LapNumber', 'Group'])['Driver'].nunique().unstack(fill_value=0)
    both = n.index[(n.get('slick', 0) >= r['min_cars']) & (n.get('wet', 0) >= r['min_cars'])] if group_terms else []
    mixed = np.isin(lap, both)
    # step 1: driver bases and lap effects from the laps where one tyre group ran (all laps without group terms)
    pure = ~mixed
    drv, drivers = pd.factorize(d['Driver'].to_numpy()[pure])
    lapc, lapn = pd.factorize(lap[pure], sort=True)
    X = np.hstack([np.eye(len(drivers))[drv], np.eye(len(lapn))[lapc]])
    beta = np.linalg.lstsq(X, y[pure], rcond=None)[0]
    base, common = beta[:len(drivers)], beta[len(drivers):]
    k = np.median(common)
    base = pd.Series(base + k, index=drivers)                # the split between base and lap effect is free
    common = pd.Series(common - k, index=lapn.astype(int))
    # step 2: on each lap both groups ran, the slick level and the gap from the cars with a base (medians)
    m = d[mixed].assign(R=y[mixed] - d.loc[mixed, 'Driver'].map(base).to_numpy(), Wet=on[mixed]).dropna(subset=['R'])
    lvl = m.groupby(['LapNumber', 'Wet'])['R'].median().unstack()
    lvl = lvl.dropna() if {False, True} <= set(lvl.columns) else lvl.iloc[:0]
    gap = pd.Series((lvl[True] - lvl[False]).to_numpy() if len(lvl) else [], index=lvl.index.astype(int), dtype=float)
    if len(lvl):
        common = pd.concat([common, pd.Series(lvl[False].to_numpy(), index=lvl.index.astype(int))]).sort_index()
    wet_rows = pd.Series(on, index=lap).groupby(level=0).mean()
    return gap, common, base, wet_rows


def field_share_path(l, race_laps, cfg):
    """The scenario line through the laps where the field's share on the wet tyre crosses one half, with the median
    trend, and the state on lap 1."""
    g = l.assign(W=(group_of(l['Compound'], cfg) == 'wet').to_numpy())
    share = g.groupby('LapNumber')['W'].mean().reindex(range(1, race_laps + 1)).ffill().bfill()
    wet = (share >= 0.5).to_numpy()
    start = 'wet' if wet[0] else 'dry'
    changes = [(int(i + 1), 'wet' if wet[i] else 'dry') for i in range(1, len(wet)) if wet[i] != wet[i - 1]]
    b = np.full((1, len(changes)), float(np.median(np.abs(settings(cfg)['trends']))))
    return scenario_gap(race_laps, start, changes, b, cfg)[0], start, changes


def fill_short(x, max_run):
    """Linear interpolation over runs of at most max_run missing values between two known ones; longer runs and the
    ends stay missing."""
    full = x.interpolate(limit_area='inside')
    miss = x.isna()
    run = miss.groupby((~miss).cumsum()).transform('sum')
    return x.where(~miss, full.where(run <= max_run))


def race_track(laps, lm, circuit, year, rnd, race_laps, cfg, group_terms=True):
    """A real race's track for the engine: gap and common rows (race_laps + 1), the oracle bases (Series by driver),
    and a table per lap (Gap, Source: measured, interpolated or field share; Common; WetShare). group_terms False:
    gap 0 everywhere (no track path baseline)."""
    l = race_laps_of(laps, year, rnd)
    gm, cm, base, wet_rows = fit_track(l, lm, circuit, race_laps, cfg, group_terms)
    idx = pd.RangeIndex(race_laps + 1)
    fs, start, changes = field_share_path(l, race_laps, cfg)
    gap = gm.reindex(idx)
    src = pd.Series(np.where(gap.notna(), 'measured', ''), index=idx)
    filled = fill_short(gap, int(settings(cfg)['replay']['max_interp_laps']))
    src[(src == '') & filled.notna()] = 'interpolated'
    gap = filled.fillna(pd.Series(fs, index=idx))
    src[src == ''] = 'field share'
    if not group_terms:
        gap[:], src[:] = 0.0, 'none'
    # the lap effect is the slick level where slicks ran; on a lap with only wet tyres in the fit it is the wet
    # tyre's level, so the common term is that less the gap the engine adds back
    only_wet = (wet_rows.reindex(cm.index) >= 0.5) & ~cm.index.isin(gm.index)
    cm = cm - np.where(only_wet, gap.reindex(cm.index).to_numpy(), 0.0)
    common = cm.reindex(idx).interpolate(limit_direction='both').fillna(0.0)
    share = l.assign(W=(group_of(l['Compound'], cfg) == 'wet').to_numpy()).groupby('LapNumber')['W'].mean()
    table = pd.DataFrame({'Gap': gap, 'Source': src, 'Common': common, 'WetShare': share.reindex(idx)}).iloc[1:]
    return {'gap': gap.to_numpy(), 'common': common.to_numpy(), 'bases': base, 'table': table,
            'start': start, 'changes': changes}


def wet_races(laps, years):
    """Every race with a wet tyre lap (features.wet_races), with its circuit."""
    l = laps[laps['Year'].isin(years) & laps['WetRace']]
    return l[bt.RACE + ['Circuit']].drop_duplicates().sort_values(bt.RACE).reset_index(drop=True)


def replay_setup(inputs, t, year, rnd, circuit, cfg, draws, group_terms=True):
    """Everything a replay races: truth (grid order), drivers, real stints (wet tyres kept), race laps, bases
    (draws x cars, oracle from race_track), the engine track, the real neutral laps and red flag laps."""
    laps = t['laps']
    truth = bt.race_truth(laps, t['results'], year, rnd)
    allowed = COMPOUNDS + list(settings(cfg)['wet_compounds'])
    real = st.real_stints(t['stints'], year, rnd, truth['Driver'], compounds=allowed)
    truth = truth[truth['Driver'].isin(real['Driver'])].reset_index(drop=True)
    drivers = truth['Driver'].tolist()
    race_laps = int(race_laps_of(laps, year, rnd)['LapNumber'].max())
    tr = race_track(laps, inputs.lap, circuit, year, rnd, race_laps, cfg, group_terms)
    known = truth['Driver'].map(tr['bases']).to_numpy(float)
    known = np.where(np.isnan(known), np.nanmedian(known), known)
    share = cfg.get('neutral', {}).get('lap_share', 0.5)
    status = np.tile(bt.real_status(laps, year, rnd, share)[:race_laps + 1], (draws, 1))
    free = bt.real_red_laps(laps, year, rnd, share)[:race_laps + 1]
    return {'truth': truth, 'drivers': drivers, 'real': real, 'race_laps': race_laps, 'track': tr,
            'bases': np.tile(known, (draws, 1)), 'status': status, 'free': free,
            'engine': engine_track(tr['gap'], cfg, tr['common'])}


def replay(inputs, setup, circuit, cfg, stints=None, seed=0, rule=None):
    """Race a replay setup (real strategies unless stints given)."""
    s = setup
    arr = st.to_arrays(s['real'] if stints is None else stints, s['drivers'], s['bases'].shape[0], codes(cfg))
    return rc.simulate(inputs, circuit, s['race_laps'], s['bases'], arr, rule=rule, seed=seed, status=s['status'],
                       free=s['free'], wet=s['engine'])


def replay_rows(res, setup, year, rnd, circuit, variant, laps):
    """run_race style rows for backtest.metrics."""
    tr = setup['truth']
    red = bool(race_laps_of(laps, year, rnd)['TrackStatus'].astype(str).str.contains('5').any())
    return rc.summarise(res, setup['drivers']).assign(
        Variant=variant, Year=year, Round=rnd, Circuit=circuit, NewCircuit=False, RedFlag=red,
        Finish=tr['Finish'].to_numpy(), Grid=tr['Grid'].to_numpy(), RaceTime_s=tr['RaceTime_s'].to_numpy(),
        RealGap_s=tr['RealGap_s'].to_numpy())


def switches_of(stints, driver, cfg):
    """A car's tyre group switches: [(stint index, start lap, to group)]."""
    s = stints[(stints['Driver'] == driver) & (stints['Draw'] == 0)].sort_values('StartLap').reset_index(drop=True)
    g = group_of(s['Compound'], cfg).to_numpy()
    return [(i, int(s.loc[i, 'StartLap']), g[i]) for i in range(1, len(s)) if g[i] != g[i - 1]]


def shift_switch(stints, driver, k, cfg, nth=1):
    """The car's nth tyre group switch moved k laps (negative earlier). Raises ValueError when it would cross the
    stints either side."""
    sw = switches_of(stints, driver, cfg)
    if len(sw) < nth:
        raise ValueError(f'{driver} has {len(sw)} tyre group switches')
    i = sw[nth - 1][0]
    mine = stints[(stints['Driver'] == driver)].sort_values('StartLap').reset_index(drop=True)
    new = mine.loc[i, 'StartLap'] + k
    lo = mine.loc[i - 1, 'StartLap'] + 1
    hi = mine.loc[i + 1, 'StartLap'] - 1 if i + 1 < len(mine) else np.inf
    if not lo <= new <= hi:
        raise ValueError(f'switch {nth} of {driver} cannot move {k} laps (stints either side)')
    mine.loc[i, 'StartLap'] = float(new)
    return pd.concat([stints[stints['Driver'] != driver], mine], ignore_index=True)


def replay_candidates(stints, driver, race_laps, cfg, laps_each=None):
    """The car's real strategy with each tyre group switch moved by up to search_laps laps (all combinations of
    its first two switches), as stints lists for race_candidates."""
    from itertools import product
    k = int(settings(cfg)['search_laps'] if laps_each is None else laps_each)
    sw = switches_of(stints, driver, cfg)[:2]
    mine = stints[stints['Driver'] == driver].sort_values('StartLap')
    base = list(zip(mine['Compound'], mine['StartLap'].astype(int), mine['StartAge'].astype(float)))
    out = {}
    for shifts in product(*[range(-k, k + 1)] * len(sw)):
        s = [list(x) for x in base]
        for (i, _, _), d in zip(sw, shifts):
            s[i][1] += d
            if i > 0:
                s[i][2] = 1.0
        ls = [x[1] for x in s]
        if ls[0] != 1 or any(b <= a for a, b in zip(ls, ls[1:])) or ls[-1] > race_laps:
            continue
        name = ('real' if not any(shifts) else 'switches moved ' + ', '.join(f'{d:+d}' for d in shifts)) + ': ' + label(s)
        out[name] = [tuple(x) for x in s]
    return out
