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


def simulate(inputs, circuit, race_laps, bases, arrays, rule=None, seed=0):
    """bases: draws x cars, s (cars in grid order); arrays: strategy.to_arrays for the same cars. rule: the
    passing rule (default inputs.rule). Returns {'Time': draws x cars, 'Place': draws x cars (1 is the
    winner), 'Passes': draws, 'Pairs': draws (cars starting a lap within window_s of the car ahead)}."""
    lm, rule = inputs.lap, rule or inputs.rule
    rng = np.random.default_rng(seed)
    D, N = bases.shape
    K = arrays['Stop'].shape[2]
    rows = np.arange(D)[:, None]
    slope = lm.noise.draw_slopes(rng, D, [(0, 0)] * (N * K)).reshape(D, N, K)   # one race draw shared by all
    loss = inputs.pit.sample(circuit, rng, D * N * K).reshape(D, N, K)
    thr, beta = rule.threshold(circuit), rule.beta
    hold, window, fg = getattr(rule, 'hold', True), inputs.window_s, inputs.follow_gap_s

    k = np.zeros((D, N), int)
    take = lambda a: np.take_along_axis(a, k[..., None], axis=2)[..., 0]
    comp, age = take(arrays['Comp']), take(arrays['Age'])
    passes, pairs = np.zeros(D, int), np.zeros(D, int)
    out_lap = np.zeros((D, N), bool)

    def pit(t, T):
        nonlocal comp, age
        now = take(arrays['Stop']) == t
        T = T + np.where(now, take(loss), 0.0)
        k[...] = np.where(now, np.minimum(k + 1, K - 1), k)
        comp = np.where(now, take(arrays['Comp']), comp)
        age = np.where(now, take(arrays['Age']), age + 1)
        return T, now

    # lap 1, grid order
    free = lm.free_air(bases, circuit, comp, age, 1, race_laps) + take(slope) * age
    gaps = inputs.start_gap(N)
    T = free[:, :1] + inputs.lap1_extra(circuit) + gaps[None, :]
    T, out_lap = pit(1, T)

    for t in range(2, race_laps + 1):
        free = lm.free_air(bases, circuit, comp, age, t, race_laps) + take(slope) * age
        stop_now = take(arrays['Stop']) == t
        own = free + lm.noise.draw_laps(rng, (D, N)) + np.where(stop_now, take(loss), 0.0)
        order = np.argsort(T, axis=1, kind='stable')
        g = lambda a: np.take_along_axis(a, order, axis=1)
        To, fo, oo, free_o = g(T), g(own), g(stop_now | out_lap), g(free)
        gap = np.diff(To, axis=1, prepend=np.nan)
        new = To + fo + np.nan_to_num(lm.dirty_air(gap))
        for j in range(1, N):
            a_new, b = new[:, j - 1], new[:, j]
            same = oo[:, j] == oo[:, j - 1]   # one car in the pits and one on track never interact; two pitting do
            meet = same & (b < a_new + fg)
            close = same & (gap[:, j] < window)
            pairs += close
            ok = meet & close & (free_o[:, j - 1] - free_o[:, j] - beta * gap[:, j] > thr)
            passes += ok & (b < a_new)
            if hold:
                new[:, j] = np.where(meet & ~ok, np.maximum(b, a_new + fg), b)
        T = np.empty_like(T)
        T[rows, order] = new
        _, out_lap = pit(t, np.zeros_like(T))   # tyre change after the in lap (its loss is already in own)
    place = np.argsort(np.argsort(T, axis=1, kind='stable'), axis=1) + 1
    return {'Time': T, 'Place': place, 'Passes': passes, 'Pairs': pairs}


def summarise(res, drivers):
    """Per car: mean place, P(win), P(podium), P(points, top 10), race time median and gap to the winner."""
    P, T = res['Place'], res['Time']
    return pd.DataFrame({'Driver': list(drivers), 'MeanPlace': P.mean(0), 'Win': (P == 1).mean(0),
                         'Podium': (P <= 3).mean(0), 'Points': (P <= 10).mean(0), 'Time_s': np.median(T, 0),
                         'GapToWinner_s': np.median(T - T.min(1, keepdims=True), 0)})


def free_air_race_time(inputs, circuit, race_laps, base, strategies, start_age=1.0):
    """Expected race time of one car on each strategy with no other cars and no noise (median pit loss): the
    coarse pass of the strategy search."""
    lm = inputs.lap
    loss = float(inputs.pit.predict([circuit])[0])
    out = np.empty(len(strategies))
    for i, (comps, stops) in enumerate(strategies):
        edges = (0,) + tuple(stops) + (race_laps,)
        total = 0.0
        for n, (c, a, b) in enumerate(zip(comps, edges[:-1], edges[1:])):
            laps = np.arange(a + 1, b + 1)
            age = laps - a + (start_age - 1 if n == 0 else 0)
            total += lm.free_air(base, circuit, np.full(len(laps), st.CODE[c]), age, laps, race_laps).sum()
        out[i] = total + loss * len(stops)
    return out


def best_strategies(inputs, circuit, race_laps, drivers, bases, rival_stints, user, cfg, start_age=1.0,
                    start_compounds=None, seed=0, extra=()):
    """Rank every legal strategy for the user's car (a driver in drivers) against the rivals' strategies
    (stints frame, one strategy per draw or one for all). Coarse pass: free air race time of every legal
    strategy; the refine_top best per stop count and the overall best are raced against the field with the same
    rival and noise draws (common random numbers). bases: draws x cars. extra: more (compounds, stop laps) to
    race whatever their coarse rank (e.g. the strategy the car really ran). Returns one row per raced strategy,
    best first: stops, compounds, stop laps, the user's mean place, P(win), P(podium), P(points), median time."""
    r = cfg['simulator']['race']
    D = bases.shape[0]
    u = list(drivers).index(user)
    cand = st.legal_strategies(race_laps, cfg, start_compounds or st.COMPOUNDS, start_age) + [tuple(map(tuple, e)) for e in extra]
    t = free_air_race_time(inputs, circuit, race_laps, float(np.median(bases[:, u])), cand, start_age)
    c = pd.DataFrame({'i': range(len(cand)), 'Stops': [len(s) for _, s in cand], 'FreeAir_s': t})
    keep = pd.concat([c.sort_values('FreeAir_s').groupby('Stops').head(r['refine_top'])['i'],
                      pd.Series(range(len(cand) - len(extra), len(cand)))]).drop_duplicates()
    rivals = rival_stints[rival_stints['Driver'] != user]
    out = []
    for i in keep:
        comps, stops = cand[i]
        mine = st.strategy_stints(comps, stops, user, 0, start_age)
        arr = st.to_arrays(pd.concat([rivals, mine if rivals['Draw'].nunique() == 1 else
                                      pd.concat([mine.assign(Draw=d) for d in rivals['Draw'].unique()])]), drivers, D)
        res = simulate(inputs, circuit, race_laps, bases, arr, seed=seed)
        s = summarise(res, drivers).iloc[u]
        out.append({'Extra': i >= len(cand) - len(extra), 'Stops': len(stops), 'Compounds': '-'.join(x[0] for x in comps), 'StopLaps': ', '.join(map(str, stops)),
                    'FreeAir_s': t[i], 'MeanPlace': s['MeanPlace'], 'Win': s['Win'], 'Podium': s['Podium'],
                    'Points': s['Points'], 'Time_s': s['Time_s']})
    return pd.DataFrame(out).sort_values(['MeanPlace', 'Time_s']).reset_index(drop=True)
