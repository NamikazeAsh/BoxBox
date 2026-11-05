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


def race_truth(laps, results, year, rnd):
    """Classified finishers in grid order with Finish (place among finishers), real race time and gap to the
    winner (NaN for lapped cars)."""
    r = results[(results['Year'] == year) & (results['Round'] == rnd)].copy()
    r['Finish'] = features.classified_finish(r)
    r['Grid'] = features.grid_slot(r)
    r = r[r['Finish'].notna()].sort_values(['Grid', 'Finish'])
    l = laps[(laps['Year'] == year) & (laps['Round'] == rnd)]
    start = l.loc[l['LapNumber'] == 1, 'LapStart_s'].min()
    last = l.sort_values('LapNumber').groupby('Driver').last()
    full = last['LapNumber'] == last['LapNumber'].max()
    r['RaceTime_s'] = r['Driver'].map((last['Time_s'] - start).where(full))
    r['RealGap_s'] = r['RaceTime_s'] - r['RaceTime_s'].min()
    return r[['Driver', 'Grid', 'Finish', 'RaceTime_s', 'RealGap_s']].reset_index(drop=True)


def run_race(inputs, t, pre, year, rnd, circuit, oracle_bases, policy, variants, draws, seed, cfg):
    """One race, every variant: rows per variant and car (MeanPlace, Win, sim gap, real Finish ...) and per
    variant the race's mean simulated passes."""
    truth = race_truth(t['laps'], t['results'], year, rnd)
    real = st.real_stints(t['stints'], year, rnd, truth['Driver'])
    truth = truth[truth['Driver'].isin(real['Driver'])].reset_index(drop=True)
    drivers = truth['Driver'].tolist()
    race_laps = int(t['laps'].loc[(t['laps']['Year'] == year) & (t['laps']['Round'] == rnd), 'LapNumber'].max())
    cars = truth[['Driver']].assign(Year=year, Round=rnd).merge(pre, on=['Year', 'Round', 'Driver'], how='left')
    known = cars['Driver'].map(oracle_bases).to_numpy(float)
    rng = np.random.default_rng(seed)
    new = circuit not in inputs.refs
    ref = float(np.nanmedian(known)) if new else None
    bases = {'model': inputs.sample_bases(cars, circuit, draws, rng, ref=ref),
             'oracle': inputs.sample_bases(cars, circuit, draws, rng, known=known, ref=ref)}
    strats = {'real': real}
    if policy is not None and any(v[1] == 'model' for v in variants):
        starts = pdm.race_starts(t['laps'], pre, t['results'], [year], cfg, green_only=False)
        starts = starts[(starts['Round'] == rnd) & starts['Driver'].isin(drivers)]
        model = st.model_stints(policy, starts.reset_index(drop=True), draws, seed)
        missing = set(drivers) - set(starts['Driver'])   # no lap 1 tyre data: keep their real strategy
        strats['model'] = st.override(model, real[real['Driver'].isin(missing)]) if missing else model
    rows, passes = [], {}
    for pace, strat, rule_name, rule in variants:
        if strat not in strats:
            continue
        arr = st.to_arrays(strats[strat], drivers, draws)
        res = rc.simulate(inputs, circuit, race_laps, bases[pace], arr, rule=rule, seed=seed)
        s = rc.summarise(res, drivers)
        name = f'{pace} pace, {strat} strategies, {rule_name}'
        rows.append(s.assign(Variant=name, Year=year, Round=rnd, Circuit=circuit, NewCircuit=new, Finish=truth['Finish'].to_numpy(),
                             Grid=truth['Grid'].to_numpy(), RaceTime_s=truth['RaceTime_s'].to_numpy(),
                             RealGap_s=truth['RealGap_s'].to_numpy()))
        passes[name] = res['Passes'].mean()
        passes[name + ' | pairs'] = res['Pairs'].mean()
    return pd.concat(rows, ignore_index=True), passes


def metrics(rows):
    """Per variant: outcome.summary of the places (rank of mean simulated place among the finishers), gap to
    the winner MAE (cars on the lead lap) and winner race time error."""
    out = {}
    for v, g in rows.groupby('Variant', sort=False):
        p = outcome.ranked(g, g['MeanPlace'])
        s = outcome.summary(p)
        lead = g.dropna(subset=['RealGap_s'])
        s['Gap MAE, s'] = (lead['GapToWinner_s'] - lead['RealGap_s']).abs().mean()
        known = g[~g['NewCircuit']]   # a new circuit's reference came from the race itself
        sim_win = known.groupby(RACE)['Time_s'].min()
        real_win = known[known['Finish'] == 1].set_index(RACE)['RaceTime_s'].reindex(sim_win.index)
        s['Winner time error, s'] = (sim_win - real_win).mean()
        s['Winner time MAE, s'] = (sim_win - real_win).abs().mean()
        out[v] = s
    return pd.DataFrame(out).T


def grid_metrics(rows):
    g = rows.drop_duplicates(RACE + ['Driver'])
    return outcome.summary(outcome.ranked(g, g['Grid'].astype(float)))


def real_passes(laps, lap_model, year, window_s):
    """On track passes and close pairs per race in the data (overtaking.pass_pairs)."""
    free = ot.free_air_laps(laps, lap_model, [year])
    p = ot.pass_pairs(laps.loc[free.index], free, window_s)
    return p.groupby(RACE)['Pass'].agg(Passes='sum', Pairs='size')
