"""Finishing position predictor (modelling D6, O6): the order of the classified finishers, from what is known
before the start: grid, qualifying gap, sprint result, form from earlier races, and how well the grid order
holds at the circuit.

Every model gives each starter a score. A race's classified finishers are ranked by it, so a prediction is a
place 1 to N among those who finished. Knowing who finished flatters every model the same way (evaluation.md).
Splits are by whole season, rolling origin (evaluate.season_splits).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from boxbox import evaluate, features

RACE = ['Year', 'Round']
FEATURES_V1 = ['Grid', 'QualiGap', 'SprintPos', 'DriverFinishForm', 'DriverGainForm', 'TeamFinishForm',
               'TeamGainForm', 'CircuitGridHold', 'Starters']
# Race pace form added 2026-10-18 (experiments.md); needs entries(..., laps)
FEATURES = FEATURES_V1 + ['TeamPaceForm', 'DriverPaceForm']


def entries(results, qualifying, sprint_results, events, cfg, laps=None):
    """One row per starter: the target Finish (place among the classified finishers, NaN otherwise) and the
    pre race features. Non starters and excluded races are dropped. A driver with no qualifying time gets
    the teammate's gap, or the capped gap if neither has one. With laps, also race pace form and the circuit
    passing rate (add_pace_form, add_overtaking)."""
    c = cfg['outcome']
    skip = {tuple(x) for x in c['exclude_races']}
    r = results[results['GridPosition'].notna() & (results['Status'] != 'Did not start')]
    r = r[[k not in skip for k in zip(r['Year'], r['Round'])]]
    df = r[RACE + ['Driver', 'DriverId', 'Team']].assign(
        Lineage=features.team_lineage(r['Team'], cfg), Grid=features.grid_slot(r), Finish=features.classified_finish(r))
    q = qualifying.assign(QualiGap=features.quali_gap(qualifying), QualiPos=qualifying['Position'])
    df = df.merge(q[RACE + ['DriverId', 'QualiGap', 'QualiPos']], on=RACE + ['DriverId'], how='left')
    if c.get('quali_gap_teammate_fill'):
        df['QualiGap'] = df['QualiGap'].fillna(df.groupby(['Lineage'] + RACE)['QualiGap'].transform('min'))
    df['QualiGap'] = df['QualiGap'].clip(upper=c['quali_gap_clip_pct']).fillna(c['quali_gap_clip_pct'])
    sp = sprint_results.rename(columns={'Position': 'SprintPos'})[RACE + ['DriverId', 'SprintPos']]
    df = df.merge(sp, on=RACE + ['DriverId'], how='left').merge(events[RACE + ['Circuit']], on=RACE, how='left')
    df['Starters'] = df.groupby(RACE)['DriverId'].transform('size')
    df['GridPenalty'] = grid_penalty(df, c['sprint_grid_until'])
    df = add_circuit_hold(add_form(df, c['form_races']), c['circuit_shrink_races'])
    # Features that are the same for every car in a race cannot change the order, only their products can
    df['FormShare'] = form_share(df, c['form_races'])
    for who in ['Driver', 'Team']:
        df[f'Stale{who}Form'] = df[f'{who}FinishForm'] * (1 - df['FormShare'])
    if laps is not None:
        df = add_overtaking(add_pace_form(df, features.race_pace(laps), c['form_races']),
                            features.race_overtakes(laps), c['circuit_shrink_races'])
        df['GridXPassing'] = df['Grid'] * df['CircuitPassing']
    return df


def grid_penalty(df, sprint_grid_until):
    """Places a driver starts behind where the session that set the grid put them (qualifying, or the sprint
    up to sprint_grid_until when the sprint set the grid): penalties, and pit lane starts. 0 if unknown."""
    set_by = df['QualiPos'].where(~((df['Year'] <= sprint_grid_until) & df['SprintPos'].notna()), df['SprintPos'])
    return (df['Grid'] - set_by).clip(lower=0).fillna(0.0)


def form_share(df, n):
    """Share of the n races behind a driver's form that were in the current season (0 at the first race of a
    season, 1 from race n + 1): last year's car says less about this year's."""
    started = df.sort_values(RACE).groupby(['DriverId', 'Year']).cumcount()
    return (started.clip(upper=n) / n).reindex(df.index)


def add_pace_form(df, pace, n):
    """Race pace form from earlier dry races only (features.race_pace, percent off the fastest driver): the
    team's best car, mean over its previous n started races with a pace (wet races skipped), and the driver's
    mean gap to that team best, so car and driver pace are separate."""
    df = df.merge(pace, on=RACE + ['Driver'], how='left')
    team = df.groupby(['Lineage'] + RACE)['Pace'].min().rename('TeamPace').reset_index().sort_values(RACE)
    roll = lambda s: s.shift(1).rolling(n, min_periods=1).mean()
    team['TeamPaceForm'] = team.groupby('Lineage')['TeamPace'].transform(roll)
    df = df.merge(team, on=['Lineage'] + RACE, how='left').sort_values(RACE + ['Grid']).reset_index(drop=True)
    df['DriverPaceForm'] = (df['Pace'] - df['TeamPace']).groupby(df['DriverId']).transform(roll)
    return df.drop(columns=['Pace', 'TeamPace'])


def add_overtaking(df, overtakes, k):
    """Circuit passing rate (features.race_overtakes) from earlier seasons, pulled toward the mean of all
    earlier races with weight k races, and OutOfPace: grid slot minus the car's rank in the race by pace form
    (positive: starting behind its pace), times the passing rate, so a fast car out of position recovers more
    where passing is easy."""
    o = overtakes.reset_index().merge(df[RACE + ['Circuit']].drop_duplicates(), on=RACE)
    rate = {}
    for y in df['Year'].unique():
        past = o[o['Year'] < y]
        if len(past):
            g, mean = past.groupby('Circuit')['Overtakes'].agg(['sum', 'size']), past['Overtakes'].mean()
            rate[y] = ((g['sum'] + k * mean) / (g['size'] + k), mean)
    df['CircuitPassing'] = [rate[y][0].get(c, rate[y][1]) if y in rate else np.nan for y, c in zip(df['Year'], df['Circuit'])]
    pace_rank = (df['TeamPaceForm'] + df['DriverPaceForm'].fillna(0)).groupby([df['Year'], df['Round']]).rank(method='first')
    df['OutOfPace'] = (df['Grid'] - pace_rank) * df['CircuitPassing']
    return df


def add_form(df, n):
    """Form from the n previous races started, never the current one: mean classified finish and mean places
    gained from the grid, for the driver and for the team (classified cars only; races where none finished
    are skipped). Crosses seasons; teams by lineage."""
    df = df.sort_values(RACE + ['Grid']).reset_index(drop=True)
    df['Gain'] = df['Grid'] - df['Finish']
    roll = lambda s: s.shift(1).rolling(n, min_periods=1).mean()
    for col in ['Finish', 'Gain']:
        df[f'Driver{col}Form'] = df.groupby('DriverId')[col].transform(roll)
    team = df.groupby(['Lineage'] + RACE)[['Finish', 'Gain']].mean().reset_index().sort_values(RACE)
    for col in ['Finish', 'Gain']:
        team[f'Team{col}Form'] = team.groupby('Lineage')[col].transform(roll)
    df = df.merge(team[['Lineage'] + RACE + ['TeamFinishForm', 'TeamGainForm']], on=['Lineage'] + RACE, how='left')
    return df.drop(columns='Gain')


def add_circuit_hold(df, k):
    """How well the grid order holds at the circuit: the per race Spearman of grid slot and finish (classified
    cars), averaged over the circuit's races in earlier seasons and pulled toward the mean of all earlier races
    with weight k races. A new circuit gets that mean; the first season has none (NaN)."""
    fin = df[df['Finish'].notna()]
    rho = fin.groupby(RACE + ['Circuit'])[['Grid', 'Finish']].apply(
        lambda g: g['Grid'].corr(g['Finish'], method='spearman')).rename('Rho').reset_index()
    hold = {}
    for y in df['Year'].unique():
        past = rho[rho['Year'] < y]
        if len(past):
            g = past.groupby('Circuit')['Rho'].agg(['sum', 'size'])
            mean = past['Rho'].mean()
            hold[y] = ((g['sum'] + k * mean) / (g['size'] + k), mean)
    df['CircuitGridHold'] = [hold[y][0].get(c, hold[y][1]) if y in hold else np.nan
                             for y, c in zip(df['Year'], df['Circuit'])]
    return df


class GridOnly:
    """Baseline: the classified finishers finish in grid order."""
    name = 'grid'

    def fit(self, df):
        return self

    def score(self, df):
        return df['Grid'].astype(float)


class RidgeModel:
    """Linear regression of the finishing place on the features. Missing form is filled with the training
    mean, a missing sprint result with the grid slot."""

    def __init__(self, alpha, feats=FEATURES):
        self.alpha, self.feats = alpha, list(feats)
        self.name = f'ridge, alpha {alpha}' + ('' if self.feats == FEATURES else f", {'+'.join(self.feats)}")

    def _x(self, df):
        x = df[self.feats].copy()
        if 'SprintPos' in x:
            x['SprintPos'] = x['SprintPos'].fillna(df['Grid'])
        return x

    def fit(self, df):
        d = df[df['Finish'].notna()]
        self.m = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(), Ridge(alpha=self.alpha)).fit(self._x(d), d['Finish'])
        return self

    def score(self, df):
        return pd.Series(self.m.predict(self._x(df)), index=df.index)

    def with_features(self, feats):
        return RidgeModel(self.alpha, feats)


class LGBMModel:
    """LightGBM on the features (missing values handled by the trees). regression predicts the finishing place;
    lambdarank learns the order within each race (label: finishers minus place, so the winner is highest)."""

    def __init__(self, objective, num_leaves, n_estimators, min_child_samples, learning_rate, feats=FEATURES, seed=0):
        self.objective, self.feats = objective, list(feats)
        self.params = dict(objective=objective, num_leaves=num_leaves, n_estimators=n_estimators,
                           min_child_samples=min_child_samples, learning_rate=learning_rate, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1)
        self.name = (f'lgbm {objective}, leaves {num_leaves}, trees {n_estimators}, min leaf {min_child_samples}'
                     + ('' if self.feats == FEATURES else f", {'+'.join(self.feats)}"))

    def fit(self, df):
        d = df[df['Finish'].notna()].sort_values(RACE)
        if self.objective == 'lambdarank':
            label = (d.groupby(RACE)['Finish'].transform('max') - d['Finish']).astype(int)
            self.m = lgb.LGBMRanker(**self.params).fit(d[self.feats], label, group=d.groupby(RACE, sort=False).size().to_numpy())
        else:
            self.m = lgb.LGBMRegressor(**self.params).fit(d[self.feats], d['Finish'])
        return self

    def score(self, df):
        s = self.m.predict(df[self.feats])
        return pd.Series(-s if self.objective == 'lambdarank' else s, index=df.index)

    def with_features(self, feats):
        p = self.params
        return LGBMModel(self.objective, p['num_leaves'], p['n_estimators'], p['min_child_samples'], p['learning_rate'],
                         feats, p['random_state'])


class PairwiseLogit:
    """Pairwise ranking (Bradley Terry style): logistic regression of 'i finished ahead of j' on the difference of
    standardised features, over every pair of classified finishers in a training race. The score is the linear
    utility, negated so lower is better. Tested 2026-10-18: no better on places than ridge, picks the winner
    more often (experiments.md)."""

    def __init__(self, C=0.01, feats=FEATURES):
        self.C, self.feats = C, list(feats)
        self.name = f'pairwise logit, C {C}' + ('' if self.feats == FEATURES else f", {'+'.join(self.feats)}")

    def _x(self, df):
        x = df[self.feats].copy()
        if 'SprintPos' in x:
            x['SprintPos'] = x['SprintPos'].fillna(df['Grid'])
        return x

    def fit(self, df):
        from sklearn.linear_model import LogisticRegression
        d = df[df['Finish'].notna()].reset_index(drop=True)
        self.prep = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler()).fit(self._x(d))
        x, finish = self.prep.transform(self._x(d)), d['Finish'].to_numpy()
        diffs, ahead = [], []
        for rows in d.groupby(RACE).indices.values():
            i, j = np.triu_indices(len(rows), 1)
            diffs.append(x[rows[i]] - x[rows[j]])
            ahead.append(finish[rows[i]] < finish[rows[j]])
        a, y = np.vstack(diffs), np.concatenate(ahead).astype(int)
        self.m = LogisticRegression(C=self.C, fit_intercept=False, max_iter=2000).fit(np.vstack([a, -a]), np.r_[y, 1 - y])
        return self

    def score(self, df):
        return pd.Series(-(self.prep.transform(self._x(df)) @ self.m.coef_[0]), index=df.index)

    def with_features(self, feats):
        return PairwiseLogit(self.C, feats)


class Blend:
    """Mean of a ridge and a LightGBM regression score (both predict the finishing place, same scale)."""

    def __init__(self, ridge, lgbm):
        self.ridge, self.lgbm = ridge, lgbm
        self.name = f'blend: {ridge.name} + {lgbm.name}'

    def fit(self, df):
        self.ridge.fit(df), self.lgbm.fit(df)
        return self

    def score(self, df):
        return (self.ridge.score(df) + self.lgbm.score(df)) / 2


def candidates(cfg):
    c, g = cfg['outcome'], cfg['outcome']['lgbm']
    models = [GridOnly()] + [RidgeModel(a) for a in c['ridge_alpha_grid']]
    for o in g['objectives']:
        for nl in g['num_leaves']:
            for ne in g['n_estimators']:
                models += [LGBMModel(o, nl, ne, mc, g['learning_rate']) for mc in g['min_child_samples']]
    for a in c.get('blend_ridge_alphas', []):
        models += [Blend(RidgeModel(a), LGBMModel('regression', nl, ne, mc, g['learning_rate']))
                   for nl in g['num_leaves'] for ne in g['n_estimators'] for mc in g['min_child_samples']]
    return models


def ranked(df, score):
    """Predicted place of each classified finisher: rank of the score within its race (lower is better),
    ties broken by grid slot. Err is predicted minus actual place."""
    fin = df.assign(Score=score)
    fin = fin[fin['Finish'].notna()].sort_values(RACE + ['Score', 'Grid'])
    fin['Pred'] = fin.groupby(RACE).cumcount() + 1.0
    fin['Err'] = fin['Pred'] - fin['Finish']
    return fin


def race_metrics(p):
    """Per race: MAE in places, Spearman of predicted and actual order, share of the podium and of the top 10
    found, and whether the predicted winner won."""
    def one(g):
        top = lambda col, n: set(g.nsmallest(n, col).index)
        return pd.Series({'MAE': g['Err'].abs().mean(), 'Spearman': g['Pred'].corr(g['Finish'], method='spearman'),
                          'Podium': len(top('Pred', 3) & top('Finish', 3)) / 3,
                          'Top10': len(top('Pred', 10) & top('Finish', 10)) / min(10, len(g)),
                          'Winner': float(g.loc[g['Pred'] == 1, 'Finish'].iloc[0] == 1), 'Finishers': len(g)})
    return p.groupby(RACE)[['Pred', 'Finish', 'Err']].apply(one)


def summary(p):
    """MAE pooled over finishers (as in the grid baseline in experiments.md); the rest are per race means."""
    r = race_metrics(p)
    return {'MAE': p['Err'].abs().mean(), 'Spearman': r['Spearman'].mean(), 'Podium': r['Podium'].mean(),
            'Top10': r['Top10'].mean(), 'Winner': r['Winner'].mean(), 'Races': len(r), 'N': len(p)}


def score_folds(df, folds, factories):
    """Ranked predictions per model name over the given (train seasons, test season) folds."""
    preds = {}
    for train_years, year in folds:
        train, test = df[df['Year'].isin(train_years)], df[df['Year'] == year]
        for make in factories:
            m = make()
            preds.setdefault(m.name, []).append(ranked(test, m.fit(train).score(test)))
    return {n: pd.concat(p) for n, p in preds.items()}


def table(preds):
    t = pd.DataFrame({n: summary(p) for n, p in preds.items()}).T
    for y in sorted(next(iter(preds.values()))['Year'].unique()):
        t[f'MAE_{y}'] = pd.Series({n: p.loc[p['Year'] == y, 'Err'].abs().mean() for n, p in preds.items()})
    return t


def validate(df, cfg, seasons):
    """Every candidate on the validation folds. Returns (validation table, validation predictions, chosen name)."""
    factories = [lambda i=i: candidates(cfg)[i] for i in range(len(candidates(cfg)))]
    val_preds = score_folds(df, evaluate.season_splits(seasons)['validation'], factories)
    val = table(val_preds)
    return val, val_preds, val.drop(index='grid')['MAE'].idxmin()


def run(df, cfg, seasons):
    """Choose a model on the validation folds, then score only it and the grid baseline on the test season,
    and refit it on all seasons. Returns (validation table, validation predictions, chosen name, test
    predictions, test table, final model)."""
    split = evaluate.season_splits(seasons)
    factories = [lambda i=i: candidates(cfg)[i] for i in range(len(candidates(cfg)))]
    val, val_preds, best = validate(df, cfg, seasons)
    pick = [i for i, m in enumerate(candidates(cfg)) if m.name in ('grid', best)]
    test_preds = score_folds(df, [split['test']], [factories[i] for i in pick])
    final = candidates(cfg)[[m.name for m in candidates(cfg)].index(best)].fit(df)
    return val, val_preds, best, test_preds, table(test_preds), final
