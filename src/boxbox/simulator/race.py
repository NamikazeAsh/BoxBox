"""The lap by lap engine and the strategy search (strategy simulator v1).

simulate races every car on its fixed strategy, vectorised over Monte Carlo draws:
* lap 1: cars cross in grid order, the leader at its free air time plus the circuit's lap 1 extra, the rest at
  the median lap 1 gap of their place (no position changes on lap 1 in v1);
* later laps: each car's free air time (LapModel) plus its stint's wear slope draw x tyre age, a lap noise
  draw, dirty air from its gap at the start of the lap, and the pit loss on an in lap. Cars are then resolved
  front to back against the car ahead at the start of the lap: a car within window_s that clears the passing
  rule (boxbox.overtaking) runs its own time; one that would catch the car ahead but does not pass is held
  follow_gap_s behind it. A car on an in or out lap never interacts with one on track (the pit lane is free);
  two cars pitting on the same lap keep their order rules;
* the result: race time, finishing place and on track passes per draw.

Rivals never react to other cars and nobody retires (v1). Cars that would be lapped are not slowed.
"""
import numpy as np
import pandas as pd

from boxbox.simulator import strategy as st


ENGINE = {'held_gap': 'fixed', 'reference': 'adjacent', 'pass_margin_s': 0.3, 'pace_gap': 'drawn', 'close_air': 'flat', 'lap1': 'grid',
          'cover_gap_s': 3.0, 'cover_p': 0.10, 'cover_laps': 10,
          'sc_restart': 'train', 'red_flag_free': False, 'double_stack': False, 'stack_s': {0: 0.6, 1: 2.1, 2: 3.2}}   # v1 behaviour (simulator.engine); cover only when asked
STINT_DRAWS = 6   # stints drawn per car whatever the strategies (simulate)
HARD = np.array([0.0, 1.0, 2.0])   # compound hardness by COMPOUNDS index, as overtaking.HARDNESS


def engine_options(inputs):
    """simulator.engine settings stored on the inputs (v1 defaults for inputs saved before v2)."""
    return {**ENGINE, **(getattr(inputs, 'engine', None) or {})}


def simulate(inputs, circuit, race_laps, bases, arrays, rule=None, seed=0, use_boot=True, record_pairs=0, status=None,
             cover=False, fixed_cars=(), free=None, teams=None, start=None, last_lap=None, wet=None):
    """bases: draws x cars, s (cars in grid order); arrays: strategy.to_arrays for the same cars. rule: the
    passing rule (default inputs.rule). Returns {'Time': draws x cars, 'Place': draws x cars (1 is the
    winner), 'Passes': draws, 'Pairs': draws (cars starting a lap within window_s of the car ahead)}.

    Engine options (simulator.engine, engine_options): reference 'adjacent' (v1: a car races the car that
    started the lap directly ahead of it) or 'max' (v2: the slowest car already resolved ahead of it in the same
    pit state, so a car that was just passed is still the one to beat); held_gap 'fixed' (v1: a held car crosses
    follow_gap_s behind) or 'draw' (v2: from the real held gaps, inputs.held_gaps). A rule with pass_prob
    (overtaking.PassLogit) draws each close pair's pass; a drawn pass puts the car at least pass_margin_s ahead.
    record_pairs: log every close pair of the first record_pairs draws (Draw, Lap, Delta, Gap, AgeDiff, CompDiff, Pass)
    under 'PairLog', for checking the engine's pairs against real ones.

    status (simulator v2): neutral laps, features.NEUTRAL_CODES per lap (draws x race_laps + 1, or one row for
    every draw; None: all green). Needs inputs.neutral (neutral.NeutralModel). On a safety car lap the first car on
    track runs at sc_leader_ratio times its free air lap (the train's pace) and every car behind runs at
    vsc_lap_ratio (the delta time pace, so the field closes up in about two laps as in the data) until it joins the train, one drawn train gap (sc_train_gaps) behind the car ahead; on a VSC lap every
    car runs vsc_lap_ratio times its free air lap, so the order holds. Nobody passes on a neutral lap, close pairs are
    not counted, and a stop on one costs pit_loss_ratio times its drawn green loss.

    cover (v2): rivals answer a stop. When a car pits at the end of a green lap, the car that started the lap
    directly ahead of it within simulator.engine.cover_gap_s brings its next planned stop forward to the next lap
    with probability cover_p, if that stop was due within cover_laps laps (notes/simulator_v2.md section 8: the car
    ahead pits next lap about 3 times as often as its tyres explain). Cars in fixed_cars (the searched car) never
    move; use it only for drawn strategies (real ones already hold the real answers).

    Safety car detail (v2.1, simulator.engine): sc_restart 'measured' lines the train up at the restart gaps
    (neutral.sc_restart_gaps, median 0.45 s) on the last safety car lap instead of the train gaps; free (draws x
    race_laps + 1 booleans, or one row) marks red flag laps, where a stop costs nothing (red_flag_free); teams (one
    id per car) with double_stack charges the second of two teammates stopping on the same lap stack_s by status
    (measured: green 0.6 s, safety car 2.1, VSC 3.2).

    start (v2.1): race from a real state instead of the grid, a dict with lap (the last lap completed), T (race
    time at its line per car), k (stint index each car is on for the next lap), age (tyre age on the next lap) and
    out (the next lap is an out lap); backtest.real_state builds it. last_lap stops the race early (the result is the
    order at that line).

    wet (simulator/wet.py, simulator.wet; None is the dry engine, unchanged): the track as a dict with gap (the
    wet tyre minus slick lap time gap per lap, s, draws x race_laps + 1 or one row), common (a lap term every car
    shares, same shape or 0), wear (wet tyre s per lap of age), ref (the dry compound index a wet tyre's lap is
    built on, at age 1) and hardness (the wet tyre's hardness for the pass rule). Compound index len(HARD) is the
    wet tyre: its lap is the free air lap of a fresh ref tyre plus gap plus wear x (age - 1), with no wear slope
    draw. Every car gets common. With no car on a wet tyre and common 0 the race is the dry race exactly."""
    lm, rule = inputs.lap, rule or inputs.rule
    eng = engine_options(inputs)
    rng = np.random.default_rng(seed)
    D, N = bases.shape
    K = arrays['Stop'].shape[2]
    rows = np.arange(D)[:, None]
    r0 = np.arange(D)
    # draws for a fixed number of stints (KD), so two strategies with different stop counts race the same noise
    # (common random numbers in the strategy search); each strategy reads the first K
    KD = max(K, STINT_DRAWS)
    slope = lm.noise.draw_slopes(rng, D, [(0, 0)] * (N * KD)).reshape(D, N, KD)[:, :, :K]   # one race draw shared by all
    loss = inputs.pit.sample(circuit, rng, D * N * KD).reshape(D, N, KD)[:, :, :K]
    prob_rule = hasattr(rule, 'pass_prob')
    thr, beta = (np.nan, 0.0) if prob_rule else (rule.threshold(circuit), rule.beta)
    B = getattr(lm, 'n_boot', 0) if use_boot else 0
    rp = np.arange(D) % B if B else None          # each draw races on one refit of the tyre model
    hold, window, fg = getattr(rule, 'hold', True), inputs.window_s, inputs.follow_gap_s
    pool = getattr(inputs, 'held_gaps', None) if eng['held_gap'] == 'draw' else None
    adjacent, margin = eng['reference'] == 'adjacent', eng['pass_margin_s']
    dirty = lm.dirty_air
    if eng['close_air'] == 'measured' and getattr(inputs, 'close_air', None) is not None:
        edges, cost = inputs.close_air          # running within the window (overtaking.close_air); the wear model's bands beyond

        def dirty(gap):
            close = np.searchsorted(edges, gap, side='right') - 1
            inside = (gap >= edges[0]) & (gap < edges[-1])
            return np.where(inside, cost[np.clip(close, 0, len(cost) - 1)], lm.dirty_air(gap))

    nm = getattr(inputs, 'neutral', None)
    if status is not None and nm is not None:
        status = np.broadcast_to(np.asarray(status, np.int8), (D, race_laps + 1))
        sc_r, vsc_r = float(getattr(nm, 'sc_leader_ratio', nm.sc_lap_ratio)), float(nm.vsc_lap_ratio)   # train pace; catch up at the delta time pace
        pit_r = np.array([1.0, nm.pit_loss_ratio[1], nm.pit_loss_ratio[2]])
        train = np.asarray(nm.sc_train_gaps, float)
        restart = np.asarray(getattr(nm, 'sc_restart_gaps', nm.sc_train_gaps), float) if eng['sc_restart'] == 'measured' else train
    else:
        status = None

    if cover:
        arrays = {**arrays, 'Stop': arrays['Stop'].copy()}   # stops move during the race; leave the caller's arrays alone
        movable = np.ones(N, bool)
        movable[list(fixed_cars)] = False
        cover_gap, cover_p, cover_laps = eng['cover_gap_s'], eng['cover_p'], eng['cover_laps']
        rng_cover = np.random.default_rng(seed + 7919)   # its own stream: how many answers are drawn depends on the strategies

    new_set = bool(getattr(lm, 'has_new_set', False))
    free_laps = None
    if free is not None and eng['red_flag_free']:
        free_laps = np.broadcast_to(np.asarray(free, bool), (D, race_laps + 1))
    stack = np.asarray(teams) if teams is not None and eng['double_stack'] else None

    hard_of = HARD
    if wet is not None:   # wet tyres on a scenario or replayed track (simulator/wet.py); no draws taken here
        wgap = np.broadcast_to(np.asarray(wet['gap'], float), (D, race_laps + 1))
        wcom = np.broadcast_to(np.asarray(wet.get('common', 0.0), float), (D, race_laps + 1))
        wrate, wref = float(wet.get('wear', 0.0)), int(wet.get('ref', 1))
        hard_of = np.append(HARD, float(wet.get('hardness', 0.0)))

    def air(t, comp, age, dry=False):
        """Free air lap without noise (the lap model's, or with wet the wet tyre lap and the common term; dry: the
        lap model's alone, a wet tyre as a fresh ref tyre)."""
        if wet is None:
            return lm.free_air(bases, circuit, comp, age, t, race_laps, rp)
        on = comp >= len(HARD)
        m = lm.free_air(bases, circuit, np.where(on, wref, comp), np.where(on, 1.0, age), t, race_laps, rp)
        return m if dry else m + wcom[:, t][:, None] + np.where(on, wgap[:, t][:, None] + wrate * (age - 1), 0.0)

    def stint_slope():
        s = take(slope)
        return s if wet is None else np.where(comp >= len(HARD), 0.0, s)

    k = np.zeros((D, N), int)
    take = lambda a: np.take_along_axis(a, k[..., None], axis=2)[..., 0]
    comp, age = take(arrays['Comp']), take(arrays['Age'])
    passes, pairs = np.zeros(D, int), np.zeros(D, int)
    by_lap = np.zeros((D, race_laps + 1), int)
    out_lap = np.zeros((D, N), bool)
    log = []
    last_ref = np.full((D, N), -1)                # per car (grid index): the car it was stuck behind last lap, -1 none
    run = np.zeros((D, N))                        # laps in a row stuck behind that car (overtaking.add_stuck)

    def pit(t, T):
        nonlocal comp, age
        now = take(arrays['Stop']) == t
        T = T + np.where(now, take(loss), 0.0)
        k[...] = np.where(now, np.minimum(k + 1, K - 1), k)
        comp = np.where(now, take(arrays['Comp']), comp)
        age = np.where(now, take(arrays['Age']), age + 1)
        return T, now

    if start is None:
        # lap 1, grid order
        free = air(1, comp, age) + stint_slope() * age
        gaps = inputs.start_gap(N)
        if eng['lap1'] == 'shuffle' and getattr(inputs, 'lap1', None) is not None:
            # v2: lap 1 order from grid, pulled toward pace order, plus noise (inputs.lap1_shuffle); v1 keeps grid order
            beta, sigma = inputs.lap1
            grid = np.arange(N)[None, :]
            pace_rank = np.argsort(np.argsort(bases, axis=1), axis=1)
            pos = np.argsort(np.argsort(grid + beta * (pace_rank - grid) + sigma * rng.standard_normal((D, N)), axis=1), axis=1)
            lead = np.take_along_axis(free, np.argmin(pos, axis=1)[:, None], axis=1)
            T = lead + inputs.lap1_extra(circuit) + gaps[pos]
        else:
            T = free[:, :1] + inputs.lap1_extra(circuit) + gaps[None, :]
        T, out_lap = pit(1, T)

        t0 = 2
    else:   # a real mid race state (backtest.real_state)
        T = np.broadcast_to(np.asarray(start['T'], float), (D, N)).copy()
        k[...] = np.broadcast_to(np.asarray(start['k'], int), (D, N))
        comp, age = take(arrays['Comp']), np.broadcast_to(np.asarray(start['age'], float), (D, N)).copy()
        out_lap = np.broadcast_to(np.asarray(start['out'], bool), (D, N)).copy()
        t0 = int(start['lap']) + 1
    end = race_laps if last_lap is None else min(int(last_lap), race_laps)

    for t in range(t0, end + 1):
        model = air(t, comp, age)
        free = model + stint_slope() * age
        if new_set:   # simulator.new_set: out lap cost and early laps of a new set fitted at a stop (laptime.LapModel.new_set)
            free = free + lm.new_set(comp if wet is None else np.minimum(comp, len(HARD) - 1), age, out_lap,
                                     (k > 0) & (take(arrays['Age']) <= 1))
        stop_now = take(arrays['Stop']) == t
        code = status[:, t] if status is not None else None
        ploss = np.where(stop_now, take(loss), 0.0) * (pit_r[code][:, None] if code is not None else 1.0)
        if free_laps is not None:
            ploss = np.where(free_laps[:, t][:, None], 0.0, ploss)           # a red flag tyre change is free
        if stack is not None and stop_now.any():
            ploss = ploss + stack_cost(stop_now, T, stack, eng['stack_s'], code)
        own = free + lm.noise.draw_laps(rng, (D, N)) + ploss
        order = np.argsort(T, axis=1, kind='stable')
        g = lambda a: np.take_along_axis(a, order, axis=1)
        # the passing rules are fitted on pace gaps from the model wear (overtaking.free_air_laps), so they read the
        # same here (pace_gap: model, v2); v1 read the gap with each stint's wear slope draw in it (pace_gap: drawn)
        To, fo, oo, free_o = g(T), g(own), g(stop_now | out_lap), g(model if eng['pace_gap'] == 'model' else free)
        ageo, hardo = g(age), hard_of[g(comp)]
        gap = np.diff(To, axis=1, prepend=np.nan)
        new = To + fo + np.nan_to_num(dirty(gap))
        h = rng.choice(pool, size=(D, N)) if pool is not None else np.full((D, N), fg)
        u = rng.random((D, N)) if prob_rule else None
        mt = np.full((D, 2), -np.inf)             # slowest resolved time ahead, per pit state (0 on track, 1 in the pits)
        mi = np.zeros((D, 2), int)
        for j in range(1, N):
            if adjacent:
                ri = np.full(D, j - 1)
                same = oo[:, j] == oo[:, j - 1]   # one car in the pits and one on track never interact; two pitting do
                a_new, gj = new[:, j - 1], gap[:, j]
            else:
                grp = oo[:, j - 1].astype(int)
                up = new[:, j - 1] > mt[r0, grp]
                mt[r0[up], grp[up]], mi[r0[up], grp[up]] = new[up, j - 1], j - 1
                gj_grp = oo[:, j].astype(int)
                a_new, ri = mt[r0, gj_grp], mi[r0, gj_grp]
                same = np.isfinite(a_new)
                a_new = np.where(same, a_new, np.inf)
                gj = To[:, j] - To[r0, ri]
            b, hj = new[:, j], h[:, j]
            close = same & (gj < window) & (True if code is None else code == 0)
            pairs += close
            by_lap[:, t] += close
            delta = free_o[r0, ri] - free_o[:, j]
            if prob_rule:
                cid, rid = order[:, j], order[r0, ri]
                sr = np.where(close & (last_ref[r0, cid] == rid), run[r0, cid] + 1, 0.0)
                p = rule.pass_prob(circuit, delta, gj, ageo[r0, ri] - ageo[:, j], hardo[r0, ri] - hardo[:, j], sr)
                ok = close & (u[:, j] < p)
                held = same & ~ok & (b < a_new + hj)
                passes += ok
                new[:, j] = np.where(ok, np.minimum(b, a_new - margin), np.where(held, a_new + hj, b))
                stay = close & ~ok
                last_ref[r0, cid], run[r0, cid] = np.where(stay, rid, -1), np.where(stay, sr, 0.0)
            else:
                meet = same & (b < a_new + hj)
                ok = meet & close & (delta - beta * gj > thr)
                passes += ok & (b < a_new)
                if hold:
                    new[:, j] = np.where(meet & ~ok, np.maximum(b, a_new + hj), b)
            if record_pairs:
                m = close[:record_pairs]
                if m.any():
                    w = np.flatnonzero(m)
                    log.append(pd.DataFrame({'Draw': w, 'Lap': t, 'Delta': delta[w], 'Gap': gj[w], 'AgeDiff': (ageo[r0, ri] - ageo[:, j])[w],
                                             'CompDiff': (hardo[r0, ri] - hardo[:, j])[w], 'Pass': (ok if prob_rule else ok & (b < a_new))[w],
                                             'StuckRun': (sr if prob_rule else np.zeros(D))[w]}))
        if code is not None and (code > 0).any():
            nxt = status[:, t + 1] if t < race_laps else np.zeros(D, int)
            last_sc = (code == 1) & (nxt == 0)                                # last safety car lap: restart gaps
            # a neutral lap runs at a ratio of the dry lap: the safety car's pace does not slow with a wet track
            new = neutral_lap(new, code, To, g(model if wet is None else air(t, comp, age, dry=True)), g(ploss), oo, sc_r, vsc_r, train, rng,
                              restart if last_sc.any() else None, last_sc)
            green = code == 0
            last_ref[~green], run[~green] = -1, 0.0
        if cover and t < race_laps - 1:
            # the car directly ahead at the start of the lap (in running order) of each car pitting now
            pit_o = g(stop_now)
            ahead = np.roll(order, 1, axis=1)
            green_lap = np.ones(D, bool) if code is None else code == 0
            trig = pit_o & (gap < cover_gap) & green_lap[:, None]
            trig[:, 0] = False
            if trig.any():
                d_i, j_i = np.nonzero(trig)
                a_i = ahead[d_i, j_i]
                nxt = arrays['Stop'][d_i, a_i, k[d_i, a_i]]
                ok = (movable[a_i] & ~stop_now[d_i, a_i] & np.isfinite(nxt) & (nxt > t + 1) & (nxt <= t + cover_laps)
                      & (rng_cover.random(len(d_i)) < cover_p))
                arrays['Stop'][d_i[ok], a_i[ok], k[d_i[ok], a_i[ok]]] = t + 1
        T = np.empty_like(T)
        T[rows, order] = new
        _, out_lap = pit(t, np.zeros_like(T))   # tyre change after the in lap (its loss is already in own)
    place = np.argsort(np.argsort(T, axis=1, kind='stable'), axis=1) + 1
    out = {'Time': T, 'Place': place, 'Passes': passes, 'Pairs': pairs, 'PairsByLap': by_lap}
    if record_pairs:
        out['PairLog'] = pd.concat(log, ignore_index=True) if log else pd.DataFrame()
    return out


def stack_cost(stop_now, T, teams, stack_s, code):
    """Extra pit loss for the second of two teammates stopping on the same lap (the one further back at the start of
    the lap), stack_s by the lap's status code (0 green, 1 safety car, 2 VSC)."""
    D, N = stop_now.shape
    extra = np.zeros((D, N))
    cost = np.array([stack_s.get(k, stack_s.get(str(k), 0.0)) for k in range(3)], float)
    lap_cost = cost[code] if code is not None else np.full(D, cost[0])
    for tm in np.unique(teams):
        cars = np.flatnonzero(teams == tm)
        if len(cars) != 2:
            continue
        a, b = cars
        both = stop_now[:, a] & stop_now[:, b]
        if both.any():
            behind = np.where(T[:, a] > T[:, b], a, b)
            extra[np.flatnonzero(both), behind[both]] = lap_cost[both]
    return extra


def neutral_lap(new, code, To, model_o, ploss_o, pitting_o, sc_r, vsc_r, train, rng, restart=None, at_restart=None):
    """Resolved lap times (cars in running order) for the draws whose lap is neutral (simulate's status). Green
    draws keep new. Safety car: the first car on track at sc_r x its free air lap, every car behind at vsc_r x its
    own until it is one train gap behind the slowest car on track ahead of it; cars in the pits are not held.
    VSC: every car at vsc_r x its free air lap."""
    sc, vsc = code == 1, code == 2
    out = new.copy()
    if vsc.any():
        out[vsc] = (To + vsc_r * model_o + ploss_o)[vsc]
    if sc.any():
        D, N = new.shape
        own = To + vsc_r * model_o + ploss_o
        head = To + sc_r * model_o + ploss_o
        gaps = rng.choice(train, size=(D, N))
        if restart is not None and at_restart is not None and at_restart.any():
            gaps[at_restart] = rng.choice(restart, size=(int(at_restart.sum()), N))   # lined up for the restart
        mt = np.full(D, -np.inf)                 # slowest resolved car on track ahead
        res = np.empty_like(new)
        for j in range(N):
            on = ~pitting_o[:, j]
            first = on & ~np.isfinite(mt)
            t_j = np.where(first, head[:, j], np.where(on, np.maximum(own[:, j], mt + gaps[:, j]), own[:, j]))
            res[:, j] = t_j
            mt = np.where(on, np.maximum(mt, t_j), mt)
        out[sc] = res[sc]
    return out


def summarise(res, drivers):
    """Per car: mean place, P(win), P(podium), P(points, top 10), race time median and gap to the winner."""
    P, T = res['Place'], res['Time']
    return pd.DataFrame({'Driver': list(drivers), 'MeanPlace': P.mean(0), 'Win': (P == 1).mean(0),
                         'Podium': (P <= 3).mean(0), 'Points': (P <= 10).mean(0), 'Time_s': np.median(T, 0),
                         'GapToWinner_s': np.median(T - T.min(1, keepdims=True), 0)})


def react_to_neutral(arrays, status, cars, window, min_stint=1):
    """A fixed strategy that pits under a safety car or VSC (simulator v2): for the cars given (indices into the
    arrays' car axis), a planned stop due within window laps after the start of a neutral period is brought forward
    to the first neutral lap of that period (its in lap), per draw; later stops keep their laps. A stop is never
    moved to within min_stint laps of the stop before it. status: draws x race laps + 1 (simulate's)."""
    stop = arrays['Stop'].copy()
    D, _, K = stop.shape
    L = status.shape[1]
    idx = np.where(np.asarray(status) > 0, np.arange(L)[None, :], L)
    nxt = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]      # first neutral lap at or after each lap
    rows = np.arange(D)
    for c in cars:
        prev = np.zeros(D)
        for k in range(K):
            s = stop[:, c, k]
            fin = np.isfinite(s)
            lo = np.clip(np.where(fin, s - window, 0), 0, L - 1).astype(int)
            cand = nxt[rows, lo].astype(float)
            move = fin & (cand < s) & (cand >= prev + min_stint)
            stop[:, c, k] = np.where(move, cand, s)
            prev = np.where(fin, stop[:, c, k], prev)
    return {**arrays, 'Stop': stop}


def free_air_race_time(inputs, circuit, race_laps, base, strategies, start_age=1.0, start_ages=None):
    """Expected race time of one car on each strategy with no other cars and no noise (median pit loss): the
    coarse pass of the strategy search. start_ages: {compound: tyre age on lap 1} when the start tyre is free (a
    compound not in it starts new); otherwise every strategy starts at start_age."""
    lm = inputs.lap
    loss = float(inputs.pit.predict([circuit])[0])
    out = np.empty(len(strategies))
    for i, (comps, stops) in enumerate(strategies):
        edges = (0,) + tuple(stops) + (race_laps,)
        total = 0.0
        for n, (c, a, b) in enumerate(zip(comps, edges[:-1], edges[1:])):
            laps = np.arange(a + 1, b + 1)
            first = start_ages.get(comps[0], 1.0) if start_ages else start_age
            age = laps - a + (first - 1 if n == 0 else 0)
            total += lm.free_air(base, circuit, np.full(len(laps), st.CODE[c]), age, laps, race_laps).sum()
            if n > 0 and getattr(lm, 'has_new_set', False):
                total += lm.new_set(np.full(len(laps), st.CODE[c]), age, laps == a + 1, np.ones(len(laps), bool)).sum()
        out[i] = total + loss * len(stops)
    return out


def best_strategies(inputs, circuit, race_laps, drivers, bases, rival_stints, user, cfg, start_age=1.0,
                    start_compounds=None, seed=0, extra=(), use_caps=True, status=None, start_ages=None):
    """Rank every legal strategy for the user's car (a driver in drivers) against the rivals' strategies
    (stints frame, one strategy per draw or one for all). Coarse pass: free air race time of every legal
    strategy; the refine_top best per stop count are raced against the field with the same rival and noise
    draws (common random numbers). bases: draws x cars. extra: more (compounds, stop laps) to race whatever
    their coarse rank (e.g. the strategy the car really ran). use_caps: keep stints within the circuit's stint
    caps (inputs.caps, strategy.stint_caps).

    Returns one row per raced strategy, best (lowest mean place, then time) first: the user's mean place and its
    10th to 90th percentile, P(win), P(podium), P(points), race time percentiles, and against the best strategy
    draw by draw: DeltaPlace (mean place difference) with a 95% interval, PBeatsBest (share of draws it
    finishes ahead of the best's place, ties half) and Tier: 'best', 'near tie' (within
    simulator.race.tie_places of the best, or not clearly worse: the interval reaches 0) or 'worse'. The draws
    cover lap noise, wear slope noise, car pace and rival strategies, not the error in the fitted compound gaps
    and wear curves, so near ties are where the model cannot separate strategies.

    status (v2): neutral laps per draw (simulate's), so every strategy races the same safety car draws; with
    simulator.race.neutral_react_window, the user's planned stops move onto an early neutral lap (react_to_neutral),
    as teams box under a safety car when a stop is due soon."""
    r = cfg['simulator']['race']
    D = bases.shape[0]
    u = list(drivers).index(user)
    caps = st.caps_for(inputs.caps, circuit) if getattr(inputs, 'caps', None) is not None and use_caps else None
    if start_ages:   # free start tyre: each start compound at its own age (simulator.search.start_tyre)
        cand = [x for c in (start_compounds or st.COMPOUNDS) for x in st.legal_strategies(race_laps, cfg, [c], start_ages.get(c, 1.0), caps)]
    else:
        cand = st.legal_strategies(race_laps, cfg, start_compounds or st.COMPOUNDS, start_age, caps)
    cand = cand + [tuple(map(tuple, e)) for e in extra]
    age_of = lambda comps: start_ages.get(comps[0], 1.0) if start_ages else start_age
    t = free_air_race_time(inputs, circuit, race_laps, float(np.median(bases[:, u])), cand, start_age, start_ages)
    c = pd.DataFrame({'i': range(len(cand)), 'Stops': [len(s) for _, s in cand], 'FreeAir_s': t})
    keep = pd.concat([c.sort_values('FreeAir_s').groupby('Stops').head(r['refine_top'])['i'],
                      pd.Series(range(len(cand) - len(extra), len(cand)))]).drop_duplicates()
    rivals = rival_stints[rival_stints['Driver'] != user]
    out, places = [], []
    for i in keep:
        comps, stops = cand[i]
        mine = st.strategy_stints(comps, stops, user, 0, age_of(comps))
        arr = st.to_arrays(pd.concat([rivals, mine if rivals['Draw'].nunique() == 1 else
                                      pd.concat([mine.assign(Draw=d) for d in rivals['Draw'].unique()])]), drivers, D)
        if status is not None and r.get('neutral_react_window'):
            arr = react_to_neutral(arr, status, [u], r['neutral_react_window'], r['min_stint_laps'])
        res = simulate(inputs, circuit, race_laps, bases, arr, seed=seed, status=status,
                       cover=bool(engine_options(inputs).get('cover_search', False)), fixed_cars=[u])
        P, T = res['Place'][:, u].astype(float), res['Time'][:, u]
        places.append(P)
        out.append({'Extra': i >= len(cand) - len(extra), 'Stops': len(stops), 'Compounds': '-'.join(x[0] for x in comps),
                    'StopLaps': ', '.join(map(str, stops)), 'FreeAir_s': t[i], 'MeanPlace': P.mean(),
                    'Place10': np.quantile(P, 0.1), 'Place90': np.quantile(P, 0.9), 'Win': (P == 1).mean(),
                    'Podium': (P <= 3).mean(), 'Points': (P <= 10).mean(), 'Time10_s': np.quantile(T, 0.1),
                    'Time_s': np.median(T), 'Time90_s': np.quantile(T, 0.9)})
    out = pd.DataFrame(out)
    order = out.sort_values(['MeanPlace', 'Time_s']).index
    out, places = out.loc[order].reset_index(drop=True), [places[j] for j in order]
    best = places[0]
    d = np.array([p - best for p in places])                       # strategies x draws, paired
    se = d.std(axis=1, ddof=1) / np.sqrt(D) if D > 1 else np.zeros(len(d))
    out['DeltaPlace'] = d.mean(axis=1)
    out['DeltaLo'], out['DeltaHi'] = out['DeltaPlace'] - 1.96 * se, out['DeltaPlace'] + 1.96 * se
    out['PBeatsBest'] = (d < 0).mean(axis=1) + 0.5 * (d == 0).mean(axis=1)
    near = (out['DeltaPlace'] <= r['tie_places']) | (out['DeltaLo'] <= 0)
    out['Tier'] = np.where(out.index == 0, 'best', np.where(near, 'near tie', 'worse'))
    return out


def tie_summary(best):
    """The near tie group of a best_strategies table in one line per strategy family (stops and compounds), with
    the range of stop laps in the group."""
    g = best[best['Tier'] != 'worse']
    return g.groupby(['Stops', 'Compounds'], sort=False).agg(Strategies=('StopLaps', 'size'), StopLaps=('StopLaps', lambda x: ' | '.join(x)),
                                                              MeanPlace=('MeanPlace', 'min'), DeltaPlace=('DeltaPlace', 'max')).reset_index()
