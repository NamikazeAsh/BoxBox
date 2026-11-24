"""Overtaking for the strategy simulator v1: a rule, not a learned model. At the start of a lap car B runs gap
seconds behind car A on the same lap. delta = A's free air lap time less B's (positive: B is faster this lap).

    B passes if delta - beta x gap > threshold[circuit]
    B is held (crosses follow_gap_s behind A) if it would catch A but does not pass
    otherwise B runs its own lap (with dirty air)

Free air lap times come from the simulator's lap model (simulator.laptime.LapModel) with each car's real stint
level, so the rule is calibrated on the same quantity the engine uses. Calibration pairs: B within window_s of A
at the line, both laps green for both cars, neither pitting on either lap, on lap 2 or later; the label is B
ahead at the next line. A circuit's threshold is the (1 - p) quantile of delta - beta x gap over its pairs, p its
pass rate pulled toward the all circuit rate with weight k pairs, so the rule reproduces the circuit's pass rate.

v2 (simulator.overtaking.model: logit, modelling D17): PassLogit, a learned pass probability per close pair
(future.md step 3): logistic regression of Pass on the pace advantage, the gap, the circuit's pass rate (as a
logit, shrunk toward the all circuit rate), the tyre age difference and the compound difference. The engine draws
each close pair's pass against it, so passing is a Monte Carlo event instead of a hard line. A held car's gap at
the next line is drawn from held_gaps (real held pairs) instead of one fixed follow gap.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

RACE = ['Year', 'Round']
DRIVER_RACE = RACE + ['Driver']
HARDNESS = {'SOFT': 0.0, 'MEDIUM': 1.0, 'HARD': 2.0}   # CompDiff: A's hardness less B's (A on a harder tyre: positive)
PASS_FEATURES = ['Delta', 'Gap', 'CircuitLogit', 'AgeDiff', 'CompDiff', 'StuckRun']
STUCK_CAP = 10


def free_air_laps(laps, lap_model, years):
    """Free air lap time (no noise) of every dry race lap of years, from each driver's real stint level: the
    stint's own level when it has enough clean laps, else the driver's race base plus the stint's compound gap
    and evolution. NaN for drivers with no level. Index aligned to the laps passed."""
    from boxbox.degradation.task import COMPOUNDS
    l = laps[laps['Year'].isin(years) & ~laps['WetRace'] & laps['Compound'].isin(COMPOUNDS) & laps['TyreLife'].notna()]
    lv = lap_model.stint_levels(laps, years)
    bases = lap_model.driver_bases(lv)
    st = l.groupby(DRIVER_RACE + ['Stint']).agg(Circuit=('Circuit', 'first'), Compound=('Compound', 'first'),
                                                 Mid=('LapNumber', 'median')).reset_index()
    st['RaceLaps'] = st.join(l.groupby(RACE)['LapNumber'].max().rename('RL'), on=RACE)['RL']
    st['Frac'] = st['Mid'] / st['RaceLaps']
    st = st.merge(lv[DRIVER_RACE + ['Stint', 'Level']], on=DRIVER_RACE + ['Stint'], how='left').merge(bases, on=DRIVER_RACE, how='left')
    st['Level'] = st['Level'].fillna(st['Base'] + lap_model.pace.offset(st))
    d = l.join(st.set_index(DRIVER_RACE + ['Stint'])['Level'], on=DRIVER_RACE + ['Stint'])
    race_laps = d.groupby(RACE)['LapNumber'].transform('max')
    w = lap_model.wear_model.wear(d.assign(RaceLap=d['LapNumber'], StintLaps=20, GapAhead_s=np.nan))
    fuel = lap_model.fuel(d['LapNumber'], race_laps)
    return (d['Level'] + w + fuel).rename('Free_s')


def pass_pairs(laps, free, window_s):
    """One row per green flag lap where B started within window_s of the car ahead (A) on the same lap:
    Circuit, LapNumber (the lap raced), Gap (s, at the previous line), Delta (A's free air time less B's on the
    lap), Pass (B ahead at the line), GapNext (B behind A at the line, s, negative after a pass), AgeDiff (A's tyre
    age less B's on the lap, laps), CompDiff (A's compound hardness less B's, HARDNESS) and StuckRun (add_stuck)."""
    l = laps.assign(Free_s=free).dropna(subset=['Time_s'])
    l = l.sort_values(RACE + ['LapNumber', 'Time_s'])
    l['Ahead'] = l.groupby(RACE + ['LapNumber'])['Driver'].shift(1)
    l['Gap'] = l.groupby(RACE + ['LapNumber'])['Time_s'].diff()
    ok = (l['TrackStatus'] == '1') & ~l['PitIn'] & ~l['PitOut']
    now = l[RACE + ['Driver', 'LapNumber', 'Time_s', 'Free_s']].assign(Ok=ok)
    every = l[RACE + ['Circuit', 'Driver', 'Ahead', 'LapNumber', 'Gap']].assign(OkPrev=ok, LapNumber=l['LapNumber'] + 1)
    before = every[every['Gap'] < window_s]
    b = before.merge(now.rename(columns={'Time_s': 'TB', 'Free_s': 'FreeB', 'Ok': 'OkB'}), on=DRIVER_RACE + ['LapNumber'])
    a = now.rename(columns={'Driver': 'Ahead', 'Time_s': 'TA', 'Free_s': 'FreeA', 'Ok': 'OkA'})
    p = b.merge(a, on=RACE + ['Ahead', 'LapNumber'])
    # the car ahead's previous lap must be clean too (it did not just leave the pits)
    a_prev = every[['Year', 'Round', 'Driver', 'LapNumber', 'OkPrev']].rename(columns={'Driver': 'Ahead', 'OkPrev': 'OkAPrev'})
    p = p.merge(a_prev, on=RACE + ['Ahead', 'LapNumber'], how='left')
    p = p[p['OkPrev'] & p['OkB'] & p['OkA'] & p['OkAPrev'].fillna(False).astype(bool) & (p['LapNumber'] >= 2)]
    p = p.dropna(subset=['FreeA', 'FreeB'])
    p = p.assign(Delta=p['FreeA'] - p['FreeB'], Pass=p['TB'] < p['TA'], GapNext=p['TB'] - p['TA'])
    tyre = l[DRIVER_RACE + ['LapNumber', 'TyreLife', 'Compound']]
    p = p.merge(tyre.rename(columns={'TyreLife': 'AgeB', 'Compound': 'CompB'}), on=DRIVER_RACE + ['LapNumber'], how='left')
    p = p.merge(tyre.rename(columns={'Driver': 'Ahead', 'TyreLife': 'AgeA', 'Compound': 'CompA'}), on=RACE + ['Ahead', 'LapNumber'], how='left')
    p['AgeDiff'] = (p['AgeA'] - p['AgeB']).fillna(0.0)
    p['CompDiff'] = (p['CompA'].map(HARDNESS) - p['CompB'].map(HARDNESS)).fillna(0.0)
    p = add_stuck(p)
    return p[RACE + ['Circuit', 'LapNumber', 'Driver', 'Ahead', 'Gap', 'Delta', 'Pass', 'GapNext', 'AgeDiff', 'CompDiff',
              'StuckRun']].reset_index(drop=True)


def add_stuck(p, cap=10):
    """StuckRun: laps in a row, up to this one, that B started within the window of the same car A and did not get
    by (0 on a first attempt, capped at cap). A car that stays stuck is less likely to pass on the next lap than
    one arriving for the first time, so the engine tracks the same count."""
    key = RACE + ['Driver', 'Ahead']
    p = p.sort_values(key + ['LapNumber'])
    prev = p[key + ['LapNumber', 'Pass']].assign(LapNumber=p['LapNumber'] + 1).rename(columns={'Pass': 'PrevPass'})
    p = p.merge(prev, on=key + ['LapNumber'], how='left')
    stuck = (p['PrevPass'] == False).astype(int)   # noqa: E712 (NaN: no pair on the lap before)
    run = stuck.groupby([p[k] for k in key] + [(stuck == 0).groupby([p[k] for k in key]).cumsum()]).cumsum()
    return p.assign(StuckRun=run.clip(upper=cap).astype(float)).drop(columns='PrevPass')


class PassThreshold:
    """The circuit threshold rule (module docstring). k=None: one threshold for every circuit."""

    def __init__(self, k=None, beta=1.0):
        self.k, self.beta = k, beta
        self.name = ('global threshold' if k is None else f'circuit threshold, k {k}') + f', beta {beta}'

    def score(self, p):
        return p['Delta'] - self.beta * p['Gap']

    def fit(self, p):
        s = self.score(p)
        self.rate = p['Pass'].mean()
        self.all = float(np.quantile(s, 1 - self.rate))
        self.thresholds = pd.Series(dtype=float)
        if self.k is not None:
            out = {}
            for circ, g in p.groupby('Circuit'):
                r = (g['Pass'].sum() + self.k * self.rate) / (len(g) + self.k)
                out[circ] = float(np.quantile(s.loc[g.index], 1 - r))
            self.thresholds = pd.Series(out)
        return self

    def threshold(self, circuit):
        return float(self.thresholds.get(circuit, self.all))

    def thresholds_for(self, circuits):
        return pd.Series(circuits).map(self.thresholds).fillna(self.all).to_numpy()

    def predict(self, p):
        return (self.score(p).to_numpy() > self.thresholds_for(p['Circuit'])).astype(bool)


class FreePassing:
    """Baseline: no rule, cars order by time (B passes whenever it would cross the line first)."""
    name = 'free passing'
    beta = 1.0

    def fit(self, p):
        return self

    def threshold(self, circuit):
        return 0.0

    def predict(self, p):
        return (p['Delta'] - p['Gap'] > 0).to_numpy()


class NoPassing:
    """Baseline: nobody passes on track."""
    name = 'no passing'
    beta = 0.0

    def fit(self, p):
        return self

    def threshold(self, circuit):
        return np.inf

    def predict(self, p):
        return np.zeros(len(p), bool)


class PassLogit:
    """Learned pass probability per close pair (module docstring, v2). The circuit enters as the logit of its pass
    rate, pulled toward the all circuit rate with weight k pairs, so a circuit with few pairs leans on the rest.
    Features are used raw (seconds, laps, compound steps); C is the inverse ridge penalty."""

    def __init__(self, C=1.0, k=200, feats=PASS_FEATURES, engine_shift=0.0):
        self.C, self.k, self.feats = C, k, list(feats)
        self.engine_shift = engine_shift   # added to the logit in the engine only (calibration to real pass counts)
        extra = '' if self.feats == PASS_FEATURES else ', ' + '+'.join(self.feats)
        self.name = f'pass logit, C {C}, k {k}{extra}'
        self.beta = 1.0   # unused by the engine; kept for code that reads rule.beta

    def _logit(self, circuits):
        return pd.Series(circuits).map(self.circuit_logit).fillna(self.all_logit).to_numpy(float)

    def _x(self, p):
        cols = {'Delta': p['Delta'].to_numpy(float), 'Gap': p['Gap'].to_numpy(float), 'CircuitLogit': self._logit(p['Circuit']),
                'AgeDiff': p['AgeDiff'].to_numpy(float).clip(-30, 30), 'CompDiff': p['CompDiff'].to_numpy(float),
                'StuckRun': p['StuckRun'].to_numpy(float) if 'StuckRun' in p else np.zeros(len(p))}
        return np.column_stack([cols[f] for f in self.feats])

    def fit(self, p):
        self.rate = float(p['Pass'].mean())
        g = p.groupby('Circuit')['Pass'].agg(['sum', 'size'])
        r = (g['sum'] + self.k * self.rate) / (g['size'] + self.k)
        self.circuit_logit = np.log(r / (1 - r))
        self.all_logit = float(np.log(self.rate / (1 - self.rate)))
        m = LogisticRegression(C=self.C, max_iter=2000).fit(self._x(p), p['Pass'].to_numpy(bool))
        self.coef = dict(zip(self.feats, m.coef_[0]))
        self.intercept = float(m.intercept_[0])
        return self

    def prob(self, p):
        """Pass probability of each pair row (pass_pairs columns)."""
        z = self.intercept + self._x(p) @ np.array([self.coef[f] for f in self.feats])
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    predict = prob   # pass_metrics reads probabilities as expected passes

    def pass_prob(self, circuit, delta, gap, age_diff, comp_diff, stuck_run=0.0):
        """The engine's call: arrays of one shape (draws), one circuit."""
        c = self.coef
        z = self.intercept + getattr(self, 'engine_shift', 0.0) + c.get('CircuitLogit', 0.0) * float(self._logit([circuit])[0])
        z = z + c.get('Delta', 0.0) * delta + c.get('Gap', 0.0) * gap
        z = z + c.get('AgeDiff', 0.0) * np.clip(age_diff, -30, 30) + c.get('CompDiff', 0.0) * comp_diff
        z = z + c.get('StuckRun', 0.0) * np.minimum(stuck_run, STUCK_CAP)
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def threshold(self, circuit):
        return np.nan


def close_air(laps, lap_model, years, edges, free_gap_s=5.0):
    """Lap time a car loses running close behind another, s, per band of the gap at the start of the lap (edges,
    up to the passing window): the median of its lap time less its free air time (free_air_laps) on green laps,
    over laps where it is slower than the car ahead in free air (so it is not being held), less the same median
    in free air (more than free_gap_s behind). The wear model's dirty air offsets start at the window (the clean
    lap filter drops closer laps), so this fills the gap the engine needs: running close is what pushes a car
    that cannot pass back toward the edge of the window."""
    free = free_air_laps(laps, lap_model, years)
    d = laps.loc[free.index].assign(Free=free).sort_values(DRIVER_RACE + ['LapNumber'])
    by = d.groupby(DRIVER_RACE)
    d['GapStart'] = by['GapAhead_s'].shift(1)
    line = laps[RACE + ['LapNumber', 'Driver', 'Time_s']].dropna().sort_values(RACE + ['LapNumber', 'Time_s'])
    line['Ahead'] = line.groupby(RACE + ['LapNumber'])['Driver'].shift(1)
    ahead = line.assign(LapNumber=line['LapNumber'] + 1)[RACE + ['LapNumber', 'Driver', 'Ahead']]
    d = d.merge(ahead, on=RACE + ['LapNumber', 'Driver'], how='left')
    fa = d[RACE + ['LapNumber', 'Driver', 'Free']].rename(columns={'Driver': 'Ahead', 'Free': 'FreeAhead'})
    d = d.merge(fa, on=RACE + ['LapNumber', 'Ahead'], how='left')
    ok = (d['TrackStatus'] == '1') & ~d['PitIn'] & ~d['PitOut'] & (d['LapNumber'] >= 3) & d['LapTime_s'].notna()
    d = d[ok].assign(Res=lambda x: x['LapTime_s'] - x['Free'])
    d = d[d['Res'].abs() < 5]
    base = d.loc[d['GapStart'] > free_gap_s, 'Res'].median()
    slow = d[d['FreeAhead'] - d['Free'] < 0]
    band = pd.cut(slow['GapStart'], edges)
    out = slow.groupby(band, observed=False)['Res'].median() - base
    return np.asarray(edges, float), out.fillna(0.0).clip(lower=0.0).to_numpy(float)


def held_gaps(p):
    """Gaps at the line, s, of every held pair (B faster than the gap, so it would have caught A, but no pass): the
    pool a held car's gap is drawn from in the engine (simulator.overtaking.held_gap: draw)."""
    held = p[~p['Pass'] & (p['Delta'] > p['Gap'])]
    return held['GapNext'].clip(lower=0.05).to_numpy(float)


def follow_gap(p):
    """Gap at the line, s, of pairs where B was faster than the gap (it would have caught A) but did not pass:
    how close a held car runs."""
    held = p[~p['Pass'] & (p['Delta'] > p['Gap'])]
    return float(held['GapNext'].median())


def pass_metrics(p, pred):
    """Pair level precision, recall and balanced accuracy, the pass rates, and the per race error in the number
    of passes. pred: booleans (a rule) or probabilities (PassLogit: expected passes, plus log loss and Brier; a
    pair counts as a predicted pass for precision and recall when its probability is over 0.5)."""
    y = p['Pass'].to_numpy(bool)
    pred = np.asarray(pred)
    if pred.dtype != bool:
        q = np.clip(pred.astype(float), 1e-6, 1 - 1e-6)
        out = pass_metrics(p, q > 0.5)
        races = pd.DataFrame({'Real': y, 'Pred': q}).groupby([p['Year'].to_numpy(), p['Round'].to_numpy()]).sum()
        out.update({'Rate, pred': q.mean(), 'Passes per race, pred': races['Pred'].mean(),
                    'Race MAE, passes': (races['Pred'] - races['Real']).abs().mean(),
                    'Circuit rate MAE': (pd.DataFrame({'y': y, 'p': q}).groupby(p['Circuit'].to_numpy()).mean()
                                         .pipe(lambda g: (g['y'] - g['p']).abs().mean())),
                    'Log loss': log_loss(y, q, labels=[False, True]), 'Brier': brier_score_loss(y, q)})
        return out
    tp = (y & pred).sum()
    races = pd.DataFrame({'Real': y, 'Pred': pred}).groupby([p['Year'].to_numpy(), p['Round'].to_numpy()]).sum()
    tpr = tp / max(y.sum(), 1)
    tnr = (~y & ~pred).sum() / max((~y).sum(), 1)
    return {'Pairs': len(y), 'Rate, real': y.mean(), 'Rate, pred': pred.mean(),
            'Precision': tp / pred.sum() if pred.sum() else np.nan, 'Recall': tpr, 'Balanced acc': (tpr + tnr) / 2,
            'Passes per race, real': races['Real'].mean(), 'Passes per race, pred': races['Pred'].mean(),
            'Race MAE, passes': (races['Pred'] - races['Real']).abs().mean(),
            'Circuit rate MAE': (pd.DataFrame({'y': y, 'p': pred}).groupby(p['Circuit'].to_numpy()).mean()
                                 .pipe(lambda g: (g['y'] - g['p']).abs().mean()))}


def candidates(cfg):
    """{name: factory}: the baselines, the global threshold and the circuit thresholds over the beta and k grids."""
    o = cfg['simulator']['overtaking']
    f = {FreePassing.name: FreePassing, NoPassing.name: NoPassing}
    for beta in o['beta']:
        f[PassThreshold(None, beta).name] = lambda beta=beta: PassThreshold(None, beta)
        for k in o['shrink']:
            f[PassThreshold(k, beta).name] = lambda k=k, beta=beta: PassThreshold(k, beta)
    lg = o.get('logit', {})
    sh = lg.get('engine_shift', 0.0)
    for C in lg.get('C', []):
        for k in lg.get('k', []):
            f[PassLogit(C, k).name] = lambda C=C, k=k: PassLogit(C, k, engine_shift=sh)
        for feats in lg.get('feature_sets', []):
            f[PassLogit(C, lg['k'][0], feats).name] = lambda C=C, feats=feats: PassLogit(C, lg['k'][0], feats)
    return f
