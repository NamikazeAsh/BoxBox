"""Tables shaped for the Streamlit app (app/). Everything here is plain pandas on saved tables and model
outputs; definitions come from features, evaluate, outcome and degradation, never redone here.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from boxbox import evaluate, outcome
from boxbox.degradation.task import AGE_TERMS

RACE = ['Year', 'Round']

# Saved model names to the names the app shows. Anything not listed keeps its own name.
MODEL_NAMES = {
    'constant pace': 'No tyre wear',
    'linear, shrink 10, dirty air, race start': 'Linear',
    'linear, shrink 10, dirty air, race start, weekend': 'Linear + weekend',
    'quadratic': 'Quadratic',
    'bayes': 'Hierarchical Bayes',
    'lgbm, leaves 8, trees 300': 'LightGBM',
    'simulator average': 'Simulator average',
    'live average': 'Live average',
    'grid': 'Grid order',
    'ridge, alpha 1000': 'Ridge',
}
BASELINES = {'constant pace', 'grid'}


def table(cfg, name, holdout=False, columns=None):
    """A processed table. Holdout seasons are refused until dashboard.show_holdout is switched on."""
    if holdout and not cfg['dashboard']['show_holdout']:
        raise PermissionError('holdout seasons are locked until the shift study (dashboard.show_holdout)')
    base = cfg['paths']['processed'] / ('holdout' if holdout else '')
    return pd.read_parquet(base / f'{name}.parquet', columns=columns)


def model_output(cfg, name):
    return pd.read_parquet(cfg['paths']['models'] / f'{name}.parquet')


def season_summary(events, pit_loss, sc):
    """Per season: races, median green flag pit loss over races, and races with a safety car or a VSC."""
    races = events.groupby('Year').size().rename('Races')
    pit = pit_loss.groupby('Year')['Median_s'].median().rename('PitLoss_s')
    flags = sc.groupby('Year')[['SafetyCar', 'VSC']].sum()
    return pd.concat([races, pit, flags], axis=1).reset_index()


def pit_loss_by_circuit(pit_loss, year):
    """Median pit loss per circuit in one season, cheapest first."""
    p = pit_loss[pit_loss['Year'] == year]
    return p.groupby('Circuit')['Median_s'].median().sort_values().rename('PitLoss_s').reset_index()


def wear_curve(cells, circuit, compound, stint_laps, length_ref, level=0.8):
    """Seconds lost against a tyre on its first lap, at each age through a stint of stint_laps, from the saved
    linear model: (circuit slope + stint length term, 0 when off) x (f(age) - f(1)). The band holds `level` of laps under
    the cell's lap noise. A circuit without the compound falls back to the compound's pooled slope."""
    c = cells[(cells['Circuit'] == circuit) & (cells['Compound'] == compound)]
    pooled = cells[cells['Compound'] == compound]
    row = c.iloc[0] if len(c) else pooled.iloc[0]
    slope = row['Slope'] if len(c) else pooled['Pooled'].iloc[0]
    sd = row['ResidSD_s'] if len(c) else pooled['ResidSD_s'].median()
    f = AGE_TERMS[row['AgeTerm']]
    age = pd.Series(range(1, int(stint_laps) + 1), dtype=float)
    loss = (slope + row['LengthCoef'] * (stint_laps - length_ref)) * (f(age) - f(1.0))
    half = norm.ppf((1 + level) / 2) * sd
    return pd.DataFrame({'Age': age, 'Loss_s': loss, 'Lo_s': loss - half, 'Hi_s': loss + half,
                         'FromCircuit': bool(len(c))})


def offsets(cfg, model='degradation'):
    """Lap level offsets of a saved wear model, split into dirty air (by gap band start) and race start laps."""
    o = model_output(cfg, f'{model}_offsets')
    return {t: g.assign(Key=g['Key'].astype(float)).drop(columns='Term') for t, g in o.groupby('Term')}


def degradation_scores(preds):
    """Test season error per model (evaluate.error_table), best first, with the 80% interval hit rate where the
    model saved quantiles."""
    t = pd.concat({n: evaluate.error_table(p) for n, p in preds.groupby('Model')}).droplevel(1)
    q = preds[preds['Q10'].notna()]
    t['Cover80'] = q['LapTimeFC_s'].between(q['Q10'], q['Q90']).groupby(q['Model']).mean()
    return t.sort_values('MAE_s').rename_axis('Model').reset_index()


def degradation_by_circuit(preds, models):
    """Test season MAE per circuit for the given models, one column each."""
    p = preds[preds['Model'].isin(models)]
    return p.groupby(['Circuit', 'Model'])['Err_s'].apply(lambda e: e.abs().mean()).unstack()


def gain(model_err, base_err):
    """Share of the baseline's error a model removes (0.2 = 20% less error)."""
    return 1 - model_err / base_err


def outcome_scores(preds):
    """Test season summary per model (outcome.summary), best first."""
    t = pd.DataFrame({n: outcome.summary(p) for n, p in preds.groupby('Model')}).T
    return t.sort_values('MAE').rename_axis('Model').reset_index()


def outcome_races(preds, events):
    """Races in the outcome predictions with their event names, in calendar order."""
    r = preds[RACE].drop_duplicates()
    return r.merge(events[RACE + ['EventName', 'Circuit']], on=RACE, how='left').sort_values(RACE).reset_index(drop=True)


def race_prediction(preds, year, rnd, model):
    """One race: each classified finisher's predicted and actual place under `model`, and the grid order's
    prediction for comparison."""
    p = preds[(preds['Year'] == year) & (preds['Round'] == rnd)]
    m = p[p['Model'] == model].set_index('Driver')
    g = p[p['Model'] == 'grid'].set_index('Driver')['Pred'].rename('GridPred')
    out = m[['Team', 'Lineage', 'Grid', 'Finish', 'Pred', 'Err']].join(g)
    return out.sort_values('Pred').reset_index()


def stint_plan(stints, year, rnd):
    """Stints each driver ran in one race (the stints table), in order."""
    s = stints[(stints['Year'] == year) & (stints['Round'] == rnd)]
    return s.sort_values(['Driver', 'Stint']).reset_index(drop=True)


def track_maps(cfg):
    """Every circuit outline pulled by `pull_data.py --track-maps` (empty when none are pulled)."""
    files = sorted((cfg['paths']['interim'] / 'track_maps').glob('*.parquet'))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def track_outline(maps, circuit, points=240):
    """One circuit drawn upright: X, Y rotated by FastF1's map angle, scaled so the longer side runs 0 to 1
    (Y pointing down, as screens draw it), thinned to about `points` points. None if not pulled.
    The first point is the start of the lap, the finish line."""
    t = maps[maps['Circuit'] == circuit] if len(maps) else maps
    if len(t) == 0:
        return None
    a = np.deg2rad(t['Rotation'].iloc[0])
    xy = t[['X', 'Y']].to_numpy(float) @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    xy = xy[::max(1, len(xy) // points)]
    xy = (xy - xy.min(axis=0)) / np.ptp(xy, axis=0).max()
    xy[:, 1] = xy[:, 1].max() - xy[:, 1]
    return pd.DataFrame(xy, columns=['X', 'Y'])


# Strategy Room: tables from boxbox.strategy_room (scripts/build_strategy_room.py)

STOP_WORDS = {0: 'no stop', 1: 'one stop', 2: 'two stops', 3: 'three stops', 4: 'four stops'}
COMPOUND_LETTERS = {'S': 'SOFT', 'M': 'MEDIUM', 'H': 'HARD'}


def room_races(meta):
    """Races in the Strategy Room tables, in calendar order."""
    return meta.sort_values(RACE).reset_index(drop=True)


def room_drivers(cars, year, rnd):
    """Searched drivers of one race in finishing order: Driver, Team, Lineage, Grid, Finish."""
    c = cars[(cars['Year'] == year) & (cars['Round'] == rnd) & cars['Searched']].copy()
    if 'Road' not in c:   # tables built before disqualified cars joined the field
        c = c.assign(Road=c['Finish'], Dsq=False)
    c['Dsq'] = c['Dsq'].fillna(False).astype(bool)
    return c.sort_values('Road').reset_index(drop=True)


def finish_label(car):
    """How a car's real result reads: "P3", or "DSQ" for a car disqualified after the race."""
    return 'DSQ' if bool(getattr(car, 'Dsq', False)) else f'P{int(car.Finish)}'


def room_table(search, year, rnd, driver):
    """One driver's ranked strategies (best first)."""
    s = search[(search['Year'] == year) & (search['Round'] == rnd) & (search['Driver'] == driver)]
    return s.sort_values('Rank').reset_index(drop=True)


def short_strategy(compounds, stop_laps):
    """A strategy in a few characters: compound initials and stop laps ("M H, lap 23"; "M H H, laps 17, 37")."""
    laps = strategy_stops(stop_laps)
    when = f'lap {laps[0]}' if len(laps) == 1 else 'laps ' + ', '.join(map(str, laps))
    return f"{' '.join(str(compounds).split('-'))}, {when}" if laps else ' '.join(str(compounds).split('-'))


def room_overview(search, cars, year, rnd):
    """Every searched driver of a race in finishing order: grid, real finish, what they ran and its average
    simulated finish and rank, the search's best and its average, places the best gains (ran less best) and the
    tier of what they ran (best, near tie, worse)."""
    s = search[(search['Year'] == year) & (search['Round'] == rnd)]
    c = room_drivers(cars, year, rnd).set_index('Driver')
    rows = []
    for d, g in s.groupby('Driver'):
        g = g.sort_values('Rank')
        best, ran = g.iloc[0], g[g['Real']]
        if ran.empty or d not in c.index:
            continue
        ran = ran.iloc[0]
        rows.append({'Finish': finish_label(c.loc[d]), 'Road': int(c.loc[d, 'Road']) if 'Road' in c else int(c.loc[d, 'Finish']),
                     'Driver': d, 'Team': c.loc[d, 'Team'], 'Grid': int(c.loc[d, 'Grid']),
                     'Lineage': c.loc[d, 'Lineage'] if 'Lineage' in c else c.loc[d, 'Team'],
                     'RanCompounds': strategy_compounds(ran['Compounds']), 'RanStops': strategy_stops(ran['StopLaps']),
                     'BestCompounds': strategy_compounds(best['Compounds']), 'BestStops': strategy_stops(best['StopLaps']),
                     'Ran': short_strategy(ran['Compounds'], ran['StopLaps']), 'RanPlace': ran['MeanPlace'], 'RanRank': int(ran['Rank']),
                     'Raced': len(g), 'Best': short_strategy(best['Compounds'], best['StopLaps']), 'BestPlace': best['MeanPlace'],
                     'Gain': ran['MeanPlace'] - best['MeanPlace'], 'Tier': ran['Tier']})
    o = pd.DataFrame(rows)
    # BoxBox's expected position: the car's rank in the race by its average on the strategy it ran (averages bunch toward the
    # middle, so a real place is compared with this rank, not with the average itself); positive BeatSim: finished higher
    o['Expected'] = o['RanPlace'].rank(method='first').astype(int)
    o['BeatSim'] = o['Expected'] - o['Road']
    return o.sort_values('Road').reset_index(drop=True)


NUMBER = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
PLURAL = {'SOFT': 'Softs', 'MEDIUM': 'Mediums', 'HARD': 'Hards', 'INTERMEDIATE': 'Inters', 'WET': 'Wets'}


def _laps(laps):
    return f'lap {laps[0]}' if len(laps) == 1 else 'laps ' + ', '.join(map(str, laps[:-1])) + f' and {laps[-1]}'


def plan_words(compounds, stop_laps):
    """The stops of a strategy as a call: "box on lap 17 and 37, Hards both times"; "box on lap 18 for Softs and
    lap 40 for Hards"."""
    comps, laps = strategy_compounds(compounds)[1:], strategy_stops(stop_laps)
    if not laps:
        return 'run to the flag without stopping'
    if len(set(comps)) == 1:
        each = {1: '', 2: ' both times'}.get(len(laps), ' each time')
        return f'box on {_laps(laps)}, {PLURAL.get(comps[0], comps[0])}{each}'
    return 'box on ' + ' and '.join(f'lap {lap} for {PLURAL.get(c, c)}' for c, lap in zip(comps, laps))


def ran_words(compounds, stop_laps):
    """A strategy as a description: "a one stop on lap 23, Mediums to Hards"."""
    comps, laps = strategy_compounds(compounds), strategy_stops(stop_laps)
    seq = ', '.join(PLURAL.get(c, c) for c in comps)
    return f'a {NUMBER.get(len(laps), len(laps))} stop on {_laps(laps)}, {seq}' if laps else f'no stop, {seq}'


def room_story(table, pit_loss, driver, finish, grid):
    """What the pit wall would say about one driver's search: a headline (the call) and short (label, text) lines:
    where the time comes from (the stops' pit loss against the tyre time, in clean air), what it is worth racing the
    field (average place, win and podium chances), how sure the call is (how often what the driver ran still
    finishes ahead) and what really happened. table: room_table rows, best first."""
    v = room_verdict(table)
    best, real = v['best'], v['real']
    pct = lambda x: f'{x:.0%}'
    happened = f'{driver} finished P{finish} from P{grid} on {ran_words(real["Compounds"], real["StopLaps"])}.'
    if v['real_tier'] == 'best':
        lines = [('Worth', f'Average finish P{real["MeanPlace"]:.1f}; wins {pct(real["Win"])} and podiums {pct(real["Podium"])} '
                           'of the simulated races.')]
        if len(table) > 1:
            nxt = table.iloc[1]
            lines.append(('Next best', f'{ran_words(nxt["Compounds"], nxt["StopLaps"])[0].upper()}{ran_words(nxt["Compounds"], nxt["StopLaps"])[1:]}, '
                                       f'{nxt["DeltaPlace"]:.2f} places behind on average.'))
        return {'headline': f'Right call: {ran_words(real["Compounds"], real["StopLaps"])}.', 'lines': lines + [('Real race', happened)]}

    k = len(strategy_stops(best['StopLaps'])) - len(strategy_stops(real['StopLaps']))
    clean = float(real['FreeAir_s'] - best['FreeAir_s'])         # seconds the best saves alone on track
    tyres = clean + k * pit_loss                                  # of which tyres, the stops' pit loss held aside
    traffic = float(real['Time_s'] - best['Time_s'])              # median race time saved racing the field
    if k > 0:
        cost = f'{"an extra stop costs" if k == 1 else f"{NUMBER.get(k, k)} extra stops cost"} about {k * pit_loss:.0f} s in the pit lane'
        trade = f'{cost} and fresher tyres win back about {max(tyres, 0):.0f} s'
    elif k < 0:
        trade = (f'{"one stop fewer saves" if k == -1 else f"{NUMBER.get(-k, -k)} stops fewer save"} about {-k * pit_loss:.0f} s in '
                 f'the pit lane and the older tyres give back about {max(-tyres, 0):.0f} s')
    else:
        trade = 'the stops cost the same, and the timing and compounds change the tyre time'
    if clean >= 0:
        why = f'Alone on track it is {clean:.0f} s faster: {trade}.'
    else:
        why = (f'Alone on track it is {-clean:.0f} s slower ({trade}); the gain comes from track position, racing the field '
               f'({traffic:+.0f} s on the median race time).')
    worth = f'Average finish P{best["MeanPlace"]:.1f} against P{real["MeanPlace"]:.1f}'
    win_b, win_r = float(best['Win']), float(real['Win'])
    worth += (f'; wins {pct(win_b)} against {pct(win_r)}' + (' (it wins less often, but finishes higher on average)' if win_b < win_r else '')
              + f'; podiums {pct(best["Podium"])} against {pct(real["Podium"])}.')
    beats = float(real['PBeatsBest'])
    if v['real_tier'] == 'near tie':
        head = f'Too close to call: {ran_words(real["Compounds"], real["StopLaps"])} is as good as anything the search found.'
        sure = f'What {driver} ran finishes ahead of the best in {pct(beats)} of the simulated races.'
    else:
        head = f'The search would {plan_words(best["Compounds"], best["StopLaps"])}.'
        sure = (f'What {driver} ran still finishes ahead in {pct(beats)} of races: a lean, not a certainty.' if beats >= 0.35 else
                f'What {driver} ran finishes ahead in only {pct(beats)} of races: a clear call.')
    return {'headline': head, 'lines': [('Why', why), ('Worth', worth), ('How sure', sure), ('Real race', happened)]}


def room_call(table, pit_loss):
    """The pit wall call as numbers, for the call panel: tier, the best and the real rows, the next best, and for a
    different best the time ledger (extra stops x pit loss, tyre time, net alone on track, gain racing the field, s;
    positive favours the best) and the head to heads (average, win, podium) and how often what was run still wins."""
    v = room_verdict(table)
    best, real = v['best'], v['real']
    out = {'tier': v['real_tier'], 'best': best, 'real': real, 'next': table.iloc[1] if len(table) > 1 else None}
    k = len(strategy_stops(best['StopLaps'])) - len(strategy_stops(real['StopLaps']))
    clean = float(real['FreeAir_s'] - best['FreeAir_s'])
    out.update({'extra_stops': k, 'pit': -k * pit_loss, 'tyres': clean + k * pit_loss, 'clean': clean,
                'traffic': float(real['Time_s'] - best['Time_s']), 'beats': float(real['PBeatsBest'])})
    return out


def room_place_shares(places, year, rnd, driver, cars):
    """Share of draws finishing in each place 1 to cars, per charted strategy: Rank, Place, Share."""
    p = places[(places['Year'] == year) & (places['Round'] == rnd) & (places['Driver'] == driver)]
    return place_shares(p, cars)


def place_shares(p, cars, by='Rank'):
    """Share of draws in each place 1 to cars of per draw places (columns by, Place)."""
    n = p.groupby(by)['Place'].size()
    c = p.groupby([by, 'Place']).size().unstack(fill_value=0).reindex(columns=range(1, int(cars) + 1), fill_value=0)
    return c.div(n, axis=0).stack().rename('Share').reset_index()


def strategy_compounds(compounds):
    """'M-H' (race.best_strategies) to ['MEDIUM', 'HARD']."""
    return [COMPOUND_LETTERS[x] for x in compounds.split('-')]


def strategy_stops(stop_laps):
    """'21, 41' to [21, 41]."""
    return [int(x) for x in str(stop_laps).split(',') if x.strip()]


def strategy_plan(rows, race_laps):
    """Stints for components.stint_bars: one bar per (label, compounds, stop laps), with Driver set to the label."""
    out = []
    for name, compounds, stop_laps in rows:
        comps, stops = strategy_compounds(compounds), strategy_stops(stop_laps)
        edges = [0] + stops + [race_laps]
        out += [{'Driver': name, 'Stint': i + 1, 'StartLap': a + 1, 'EndLap': b, 'Compound': c}
                for i, (c, a, b) in enumerate(zip(comps, edges[:-1], edges[1:]))]
    return pd.DataFrame(out, columns=['Driver', 'Stint', 'StartLap', 'EndLap', 'Compound'])


def describe_strategy(compounds, stop_laps):
    """Plain words: 'one stop on lap 25, Medium then Hard'."""
    comps, stops = strategy_compounds(compounds), strategy_stops(stop_laps)
    joined = lambda xs, last: ', '.join(xs[:-1]) + f' {last} ' + xs[-1] if len(xs) > 1 else ''.join(xs)
    names = joined([c.title() for c in comps], 'then')
    if not stops:
        return f'no stop, {names}'
    return f'{STOP_WORDS.get(len(stops), f"{len(stops)} stops")} on lap{"s" if len(stops) > 1 else ""} {joined([str(x) for x in stops], "and")}, {names}'


def room_verdict(table):
    """What the page says about one driver's search: the best and real rows, the real strategy's rank and tier,
    how many strategies were raced and how many are near ties with the best."""
    best = table.iloc[0]
    real = table[table['Real']].iloc[0] if table['Real'].any() else None
    return {'best': best, 'real': real, 'raced': len(table), 'ties': int((table['Tier'] == 'near tie').sum()),
            'real_rank': None if real is None else int(real['Rank']),
            'real_tier': None if real is None else real['Tier'],
            'gain': None if real is None else float(real['MeanPlace'] - best['MeanPlace'])}


# Overtaking: tables from boxbox.overtaking_view (scripts/build_overtaking.py)

def pass_ratio(circuits, a, b):
    """How many times likelier a pass is at circuit a than at b at the page's reference pace gap (ProbRef)."""
    p = circuits.set_index('Circuit')['ProbRef']
    return float(p[a] / p[b])


def odds_change(coefs, feature, step=1.0):
    """Change in the odds of a pass for step units of a feature (0.1 = 10% higher odds)."""
    c = coefs.set_index('Feature')['Coef']
    return float(np.exp(c[feature] * step) - 1)


def spread_labels(values, min_gap, top=None):
    """Label positions next to line ends, in the same order as values and at least min_gap apart: each label stays at
    its value unless the one below pushes it up; with top, labels over it are pushed back down from there."""
    v = np.asarray(values, float)
    order = np.argsort(v, kind='stable')
    y = v[order].copy()
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + min_gap)
    if top is not None and len(y):
        y[-1] = min(y[-1], top)
        for i in range(len(y) - 2, -1, -1):
            y[i] = min(y[i], y[i + 1] - min_gap)
    out = np.empty_like(y)
    out[order] = y
    return out


def tyre_guide(stints, events, circuit, years, caps, wear, min_stints=5):
    """How long each dry compound lasts at a circuit, from the dry races of years: the tyre's age at the end of each
    stint (median and the middle half), the longest teams ran, the search's stint cap (Cap) and the wear after 10 laps (s
    per lap, from the simulator's tyre model; wear(compound, age)). A compound with fewer than min_stints there uses
    every circuit (All True)."""
    s = stints[stints['Year'].isin(years)].merge(events[['Year', 'Round', 'Circuit']], on=['Year', 'Round'])
    wet = s.loc[s['Compound'].isin(['INTERMEDIATE', 'WET']), ['Year', 'Round']].drop_duplicates()
    s = s.merge(wet.assign(W=1), on=['Year', 'Round'], how='left')
    s = s[s['W'].isna() & s['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]
    s = s.assign(Age=s['TyreLifeStart'].fillna(1.0) + s['Laps'] - 1)
    rows = []
    for k in ['SOFT', 'MEDIUM', 'HARD']:
        here = s[(s['Circuit'] == circuit) & (s['Compound'] == k)]['Age']
        use = here if len(here) >= min_stints else s[s['Compound'] == k]['Age']
        rows.append({'Compound': k, 'Typical': float(use.median()) if len(use) else float('nan'),
                     'Low': float(use.quantile(0.25)) if len(use) else float('nan'), 'High': float(use.quantile(0.75)) if len(use) else float('nan'),
                     'Longest': float(use.max()) if len(use) else float('nan'), 'Cap': float(caps.get(k, float('nan'))),
                     'Wear10': float(wear(k, 11) - wear(k, 1)),
                     'Stints': int(len(here)), 'All': len(here) < min_stints})
    return pd.DataFrame(rows)
