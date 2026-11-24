"""The 2026 shift study and its 2022 rehearsal (notes/shift_study.md, modelling D3). Every measure is the same
code run on a reference split (no regulation change) and a shift split (across a change); the shift is read as
the change between them. Splits come from config training_seasons (shift_rehearsal_*, shift_study_*) through
config.seasons_for.

Measures: 1 degradation (anchored forecast, wear rates), 2 in season recovery (frozen, season scaled, rolling
refit), 3 fresh tyre gaps and pit loss, 4 finishing predictor, 6 safety cars. Measure 5 (the simulator) is run
by scripts/backtest_simulator.py.
"""
import numpy as np
import pandas as pd

from boxbox import evaluate, outcome
from boxbox.config import seasons_for
from boxbox.degradation import adjust
from boxbox.degradation.compare import ladder
from boxbox.degradation.compound import CompoundPace
from boxbox.degradation.monotone import MonotoneWear
from boxbox.degradation.task import COMPOUNDS, KEY, RACE, STINT, anchored_forecast, model_laps

STUDIES = {'rehearsal': ('shift_rehearsal_train', 'shift_rehearsal_test'),
           'study': ('shift_study_train', 'shift_study_test')}
BLOCKS = [(1, 6), (7, 12), (13, 99)]
TABLES = ['laps', 'events', 'results', 'qualifying', 'sprint_results', 'sprint_laps', 'practice_laps', 'pit_loss_stops']


def splits(cfg, study):
    """{'reference': (train, test), 'shift': (train, test)}: the shift split from config, and the reference that
    drops its last training season and scores it."""
    tr, te = (seasons_for(cfg, c) for c in STUDIES[study])
    return {'reference': (list(tr[:-1]), tr[-1]), 'shift': (list(tr), te[0])}


def component_train(cfg, component, split_train, test):
    """Training seasons for a component on one split: its D2 seasons and the split's, before the test season."""
    return [y for y in sorted(set(seasons_for(cfg, component)) | set(split_train)) if y < test]


def excluded_races(events, circuits):
    """(Year, Round) pairs of the events held at the given circuits."""
    if not circuits:
        return set()
    e = events[events['Circuit'].isin(circuits)]
    return set(zip(e['Year'], e['Round']))


def drop_rounds(df, races):
    """df without rows of the given (Year, Round) races; tables without those columns are returned as they are."""
    if not races or not {'Year', 'Round'} <= set(df.columns):
        return df
    return df[[k not in races for k in zip(df['Year'], df['Round'])]]


def load_tables(cfg, holdout=False, names=TABLES):
    """Processed tables, with data/processed/holdout appended when the study reads a holdout season. Holdout
    races at circuits in shift_study.exclude_circuits are dropped from every table (notes/shift_study.md)."""
    P = cfg['paths']['processed']
    skip = set()
    if holdout and (P / 'holdout' / 'events.parquet').exists():
        skip = excluded_races(pd.read_parquet(P / 'holdout' / 'events.parquet'),
                              (cfg.get('shift_study') or {}).get('exclude_circuits') or [])
    out = {}
    for n in names:
        parts = [pd.read_parquet(P / f'{n}.parquet')]
        if holdout and (P / 'holdout' / f'{n}.parquet').exists():
            parts.append(drop_rounds(pd.read_parquet(P / 'holdout' / f'{n}.parquet'), skip))
        out[n] = pd.concat(parts, ignore_index=True)
    return out


# ---------------------------------------------------------------- measure 1: degradation

def wear_models(cfg):
    """The models of measure 1, as factories: constant pace, linear, and the simulator's wear model."""
    f = ladder(cfg)
    s = cfg['simulator']
    sim = lambda: MonotoneWear(f[s['wear_model']](), s.get('max_age', 60)) if s.get('monotone_wear', True) else f[s['wear_model']]()
    return {'constant pace': f['constant pace'], 'linear': f['linear'], 'simulator wear': sim}


def score_table(preds, by=None, base='constant pace'):
    """MAE, bias and N per model (and group), with the gain over the base model's MAE on the same laps."""
    t = pd.concat({n: evaluate.error_table(p, by) for n, p in preds.items()}, names=['Model'])
    if base in preds:
        ref = t.xs(base, level='Model')['MAE_s']
        t['Gain'] = 1 - t['MAE_s'] / ref.reindex(t.index.droplevel('Model')).to_numpy()
    return t if by else t.droplevel(1)


def degradation(laps, cfg, train, test):
    """Anchored forecasts on the test season from each model fitted on the training seasons. Also returns the
    fitted models (by name) for reuse."""
    tr, te = model_laps(laps, train, cfg, train=True), model_laps(laps, [test], cfg)
    fitted = {n: make().fit(tr) for n, make in wear_models(cfg).items()}
    return {n: anchored_forecast(m, te, cfg) for n, m in fitted.items()}, fitted


def wear_rates(laps, cfg, train, test, fitted=None):
    """Compound wear slopes (s per lap of tyre age, linear model) from the training fit and from the test season
    alone: pooled over all circuits, and the stint weighted mean change over circuit and compound cells in both."""
    lin = fitted or ladder(cfg)['linear']().fit(model_laps(laps, train, cfg, train=True))
    alone = ladder(cfg)['linear']().fit(model_laps(laps, [test], cfg, train=True))
    both = alone.cells[['Raw', 'Stints']].join(lin.cells['Slope'].rename('TrainSlope'), how='inner')
    d = (both['Raw'] - both['TrainSlope']).groupby(level='Compound')
    w = both['Stints'].groupby(level='Compound')
    matched = (both['Stints'] * (both['Raw'] - both['TrainSlope'])).groupby(level='Compound').sum() / w.sum()
    return pd.DataFrame({'Train': lin.pooled, 'TestAlone': alone.pooled, 'Change': alone.pooled - lin.pooled,
                         'MatchedChange': matched, 'MatchedCells': d.size()}).reindex(COMPOUNDS)


def allocation_flags(compounds_wide, year):
    """Circuits of `year` whose Hard, Medium or Soft C number differs from the previous season's at the same
    circuit (modelling D23, the D1 hedge). Returns a frame by Year and Round: Circuit, Flagged, New (no previous
    season row). Empty when the previous season is not in the table."""
    c = compounds_wide
    cur, prev = c[c['Year'] == year], c[c['Year'] == year - 1]
    if prev.empty:
        return pd.DataFrame(columns=RACE + ['Circuit', 'Flagged', 'New'])
    p = prev.drop_duplicates('Circuit', keep='last').set_index('Circuit')[['HARD', 'MEDIUM', 'SOFT']]
    rows = []
    for _, r in cur.iterrows():
        new = r['Circuit'] not in p.index
        known = all(pd.notna(r[k]) for k in ['HARD', 'MEDIUM', 'SOFT'])
        flag = (not new) and known and any(int(r[k]) != int(p.loc[r['Circuit'], k]) for k in ['HARD', 'MEDIUM', 'SOFT'])
        rows.append({'Year': year, 'Round': r['Round'], 'Circuit': r['Circuit'], 'Flagged': flag, 'New': new})
    return pd.DataFrame(rows)


def drop_races(preds, races):
    """preds without the given (Year, Round) races."""
    races = set(map(tuple, races))
    return {n: p[[k not in races for k in zip(p['Year'], p['Round'])]] for n, p in preds.items()}


def new_circuits(laps, cfg, train, test):
    """Circuits of the test season with no training laps in the degradation task."""
    tr = set(model_laps(laps, train, cfg)['Circuit'])
    return sorted(set(model_laps(laps, [test], cfg)['Circuit']) - tr)


def practice_fallback(base, t, cfg, train, test):
    """Anchored forecast of the test season with the practice fallback shift (weekend_adjust: only circuits with at
    most practice_max_history training stints are shifted, by their long runs shrunk with practice_shrink)."""
    w = cfg['degradation']['weekend_adjust']
    laps, prac = t['laps'], t['practice_laps']
    te = model_laps(laps, [test], cfg)
    runs = adjust.model_runs(prac, laps, list(train) + [test], cfg, cfg['long_run']['min_laps'])
    offset = adjust.practice_offset(base, runs[runs['Year'].isin(train)])
    tab = adjust.weekend_table(base, te, model_laps(t['sprint_laps'], [test], cfg), runs[runs['Year'] == test], offset, cfg)
    k = w['practice_shrink']
    shift = adjust.weekend_variants(tab, [k], w['practice_max_history'])[f'practice fallback, k {k}']
    return adjust.forecast_with_shifts(base, te, shift, cfg), tab


# ---------------------------------------------------------------- measure 2: in season recovery

def rolling_refit(laps, cfg, train, test, factory):
    """Race by race through the test season: a model fitted on the training seasons plus the test season's
    earlier rounds forecasts each round."""
    tr, te = model_laps(laps, train, cfg, train=True), model_laps(laps, [test], cfg)
    te_fit = model_laps(laps, [test], cfg, train=True)
    out = []
    for r in sorted(te['Round'].unique()):
        m = factory().fit(pd.concat([tr, te_fit[te_fit['Round'] < r]]))
        out.append(anchored_forecast(m, te[te['Round'] == r], cfg))
    return pd.concat(out)


def season_grid(base, laps, cfg, test):
    """SeasonScaled over the degradation.season_adjust grid on one test season: {(scope, shrink): preds}."""
    te = model_laps(laps, [test], cfg)
    c = cfg['degradation']['season_adjust']
    return {(s, k): adjust.season_adjusted_forecast(base, te, cfg, s, k)[0] for s in c['scopes'] for k in c['shrink_grid']}


def choose_season_adjust(grid):
    """The (scope, shrink) with the lowest pooled MAE."""
    return min(grid, key=lambda key: grid[key]['Err_s'].abs().mean())


def recovery(laps, cfg, train, test, scope, shrink, base=None):
    """The three versions of measure 2 on the test season, linear base: frozen, season scaled, rolling refit."""
    make = ladder(cfg)['linear']
    base = base or make().fit(model_laps(laps, train, cfg, train=True))
    te = model_laps(laps, [test], cfg)
    return {'frozen': anchored_forecast(base, te, cfg),
            'season scaled': adjust.season_adjusted_forecast(base, te, cfg, scope, shrink)[0],
            'rolling refit': rolling_refit(laps, cfg, train, test, make)}


def block_table(preds, blocks=BLOCKS):
    """MAE per version over round blocks (round numbers, inclusive), over all rounds, and the bias over all rounds."""
    rows = {}
    for n, p in preds.items():
        rows[n] = {(f'{lo} to {hi}' if hi < 99 else f'{lo} on'): p.loc[p['Round'].between(lo, hi), 'Err_s'].abs().mean()
                   for lo, hi in blocks}
        rows[n]['all'] = p['Err_s'].abs().mean()
        rows[n]['bias'] = p['Err_s'].mean()
    return pd.DataFrame(rows).T


def round_mae(preds):
    """MAE per round (columns: versions), and the cumulative MAE through each round."""
    per = pd.DataFrame({n: p.groupby('Round')['Err_s'].apply(lambda e: e.abs().mean()) for n, p in preds.items()})
    s = pd.DataFrame({n: p.groupby('Round')['Err_s'].apply(lambda e: e.abs().sum()) for n, p in preds.items()})
    c = pd.DataFrame({n: p.groupby('Round').size() for n, p in preds.items()})
    return per, s.cumsum() / c.cumsum()


def stays_ahead(cum, version, ref='frozen'):
    """First round from which the version's cumulative MAE stays below the reference's to the last round (None if
    it is not ahead at the last round)."""
    ahead = (cum[version] < cum[ref]).to_numpy()
    if not ahead[-1]:
        return None
    i = len(ahead) - 1
    while i > 0 and ahead[i - 1]:
        i -= 1
    return int(cum.index[i])


# ---------------------------------------------------------------- measure 3: fresh tyre gaps and pit loss

def compound_gaps(laps, cfg, train, test, wear=None):
    """All circuit fresh tyre gaps to Medium (s per lap), CompoundPace at the simulator's settings, from the
    training seasons and from the test season alone, both on the simulator's wear model fitted on training."""
    c = cfg['simulator']['compound_pace']
    frac = {'frac': 'start'} if c.get('evolution', 'lap') == 'stint_start' else {}
    wear = wear or wear_models(cfg)['simulator wear']().fit(model_laps(laps, train, cfg, train=True))
    fit = lambda years: CompoundPace(c['shrink'], c['min_laps'], **frac).fit(wear, model_laps(laps, years, cfg))
    a, b = fit(train), fit([test])
    return pd.DataFrame({'Train': a.gaps, 'TestAlone': b.gaps, 'Change': b.gaps - a.gaps,
                         'TestStints': b.n_stints.reindex(a.gaps.index)})


def pit_loss(stops, cfg, train, test, facts=None):
    """Race median pit loss on the test season: per circuit median of the training stops (D26 limit correction to
    the test season when facts are given) and the global median. Races with target_min_stops stops; MAE over
    circuits seen in training. Returns (table, per race frame)."""
    from boxbox import reference
    from boxbox.pitloss import GlobalPitLoss, PitLoss, race_medians
    c = cfg['simulator']['pit_loss']
    tr, te = stops[stops['Year'].isin(train)], stops[stops['Year'] == test]
    if facts is not None and c.get('limit_correction', False):
        tr = reference.limit_corrected_stops(tr, facts, test)
    target = race_medians(te, c['target_min_stops'])
    target = target.assign(Seen=target['Circuit'].isin(tr['Circuit']))
    rows = {}
    for m in [PitLoss(c['recent'], c['min_stops']), GlobalPitLoss()]:
        err = m.fit(tr).predict(target['Circuit']) - target['Median_s']
        target[f'Err_{m.name}'] = err.to_numpy()
        s = target['Seen']
        rows[m.name] = {'MAE_s': err[s].abs().mean(), 'Bias_s': err[s].mean(), 'Races': int(s.sum()),
                        'MAE_all_s': err.abs().mean()}
    return pd.DataFrame(rows).T, target


# ---------------------------------------------------------------- measure 4: finishing predictor

def finishing(df, cfg, train, test, name):
    """The named outcome candidate and the grid baseline, trained on `train` and scored on `test` (classified
    finishers ranked; outcome.summary)."""
    pick = [m for m in outcome.candidates(cfg) if m.name in (name, 'grid')]
    preds = {m.name: outcome.ranked(df[df['Year'] == test], m.fit(df[df['Year'].isin(train)]).score(df[df['Year'] == test]))
             for m in pick}
    return pd.DataFrame({n: outcome.summary(p) for n, p in preds.items()}).T[['MAE', 'Spearman', 'Races', 'N']], preds


# ---------------------------------------------------------------- measure 6: safety cars

def safety_cars(laps, cfg, train, test):
    """NeutralModel fitted on the training seasons' dry races, scored on the test season (neutral.metric_table)."""
    from boxbox import neutral
    n = cfg['neutral']
    tables = neutral.deployment_tables(laps[laps['Year'].isin(list(train) + [test])], cfg)
    m = neutral.NeutralModel(cfg).fit_tables(tables, train)
    scored = neutral.score_races(m, tables, test, n['draws'], n['seed'])
    out = neutral.metric_table(scored)
    out.update({'Rate_SC_pred': scored['P_SC'].mean(), 'Rate_SC_obs': scored['Act_SC'].mean(),
                'Rate_VSC_pred': scored['P_VSC'].mean(), 'Rate_VSC_obs': scored['Act_VSC'].mean()})
    return out


# ---------------------------------------------------------------- figures for the report

SERIES = ['#2a78d6', '#eb6834', '#1baf7a']   # categorical slots 1 to 3 (fixed order: frozen, season scaled, rolling refit)
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _axes(ax):
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save_figures(out, sp, study, folder):
    """PNG figures: MAE per round for the three versions of measure 2 (shift split), and compound wear slopes,
    training fit against each test season alone. Returns the paths written."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=True)
    te = sp['shift'][1]
    paths = []

    per, _ = round_mae(out['shift']['recovery'])
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for col, colour in zip(per.columns, SERIES):
        ax.plot(per.index, per[col], color=colour, linewidth=2, marker='o', markersize=4, label=col)
    _axes(ax)
    ax.set_xlabel('Round', color=INK)
    ax.set_ylabel('MAE, s per lap', color=INK)
    ax.set_title(f'Degradation forecast error by round, {te}', color=INK, loc='left')
    ax.set_xticks(per.index)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    paths.append(folder / f'{study}_error_by_round_{te}.png')
    fig.savefig(paths[-1])
    plt.close(fig)

    ref_te = sp['reference'][1]
    r = pd.DataFrame({f'training fit ({sp["shift"][0][0]} to {sp["shift"][0][-1]})': out['shift']['rates']['Train'],
                      f'{ref_te} alone': out['reference']['rates']['TestAlone'], f'{te} alone': out['shift']['rates']['TestAlone']})
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    x, w = np.arange(len(r)), 0.26
    for i, (col, colour) in enumerate(zip(r.columns, SERIES)):
        ax.bar(x + (i - 1) * w, r[col], width=w - 0.02, color=colour, label=col)
    _axes(ax)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, [c.title() for c in r.index], color=INK)
    ax.set_ylabel('Wear, s per lap of tyre age', color=INK)
    ax.set_title('Compound wear slopes (linear model, all circuits pooled)', color=INK, loc='left')
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    paths.append(folder / f'{study}_wear_slopes_{ref_te}_{te}.png')
    fig.savefig(paths[-1])
    plt.close(fig)
    return paths
