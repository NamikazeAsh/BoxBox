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
"""
import numpy as np
import pandas as pd

RACE = ['Year', 'Round']
DRIVER_RACE = RACE + ['Driver']


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
    lap), Pass (B ahead at the line) and GapNext (B behind A at the line, s, negative after a pass)."""
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
    return p[RACE + ['Circuit', 'LapNumber', 'Driver', 'Ahead', 'Gap', 'Delta', 'Pass', 'GapNext']].reset_index(drop=True)


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


def follow_gap(p):
    """Gap at the line, s, of pairs where B was faster than the gap (it would have caught A) but did not pass:
    how close a held car runs."""
    held = p[~p['Pass'] & (p['Delta'] > p['Gap'])]
    return float(held['GapNext'].median())


def pass_metrics(p, pred):
    """Pair level precision, recall and balanced accuracy, the pass rates, and the per race error in the number
    of passes."""
    y = p['Pass'].to_numpy(bool)
    pred = np.asarray(pred, bool)
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
    return f
