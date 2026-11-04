"""How fast each compound is on a fresh tyre (modelling O12). The anchored task gives every stint its own pace
level, so wear models never learn the gap between a new Soft and a new Hard; the strategy simulator needs it.

A stint's level is the median of its clean laps less the wear model's wear and dirty air (fuel corrected, so
the level is what that car would do on that tyre at age 0 with no fuel). Within one driver's race, levels
differ by compound and by when in the race the stint ran (track evolution, and any error in the fuel
assumption):

    level = driver race effect + gap[circuit, compound] + evolution x race fraction (+ length x (stint laps - ref))

Medium is the reference (gap 0). The driver race effect is removed by demeaning within each driver's race, so
only drivers who ran two or more stints inform the fit. The circuit gaps are pulled toward the all circuit
gaps by a ridge penalty of weight `shrink` (in stints), so a circuit with few comparisons leans on the average.

The length term (off by default, modelling D12) uses the stint's realised length: fine when replaying a known
strategy, look ahead when forecasting a race.
"""
import numpy as np
import pandas as pd

from boxbox.degradation.task import COMPOUNDS, RACE, STINT

REF = 'MEDIUM'
OTHERS = [c for c in COMPOUNDS if c != REF]
DRIVER_RACE = RACE + ['Driver']


def stint_levels(wear, laps, min_laps=5):
    """One row per stint with at least min_laps clean laps: its level, compound, circuit, race fraction at the
    stint's middle lap and its length."""
    y = laps['LapTimeFC_s'] - wear.wear(laps) - wear.dirty_air(laps)
    g = laps.assign(Lvl=y).groupby(STINT)
    s = g.agg(Level=('Lvl', 'median'), Compound=('Compound', 'first'), Circuit=('Circuit', 'first'),
              Mid=('LapNumber', 'median'), RaceLaps=('RaceLaps', 'first'), StintLaps=('StintLaps', 'first'),
              N=('Lvl', 'size')).reset_index()
    s = s[s['N'] >= min_laps]
    return s.assign(Frac=s['Mid'] / s['RaceLaps'])


def _design(s, length_ref, length):
    x = pd.DataFrame({c: (s['Compound'] == c).astype(float) for c in OTHERS}, index=s.index)
    x['Frac'] = s['Frac']
    if length:
        x['Length'] = s['StintLaps'] - length_ref
    return x


def _demean(df, keys):
    return df - df.groupby(keys).transform('mean')


class CompoundPace:
    def __init__(self, shrink=20, min_laps=5, length=False, length_ref=20, by_circuit=True):
        self.k, self.min_laps, self.length, self.length_ref, self.by_circuit = shrink, min_laps, length, length_ref, by_circuit
        self.name = ('compound pace, ' + (f'shrink {shrink}' if by_circuit else 'all circuits')
                     + (', stint length' if length else ''))

    def fit(self, wear, laps):
        s = stint_levels(wear, laps, self.min_laps)
        s = s[s.groupby(DRIVER_RACE)['Stint'].transform('size') >= 2]
        keys = [s[k] for k in DRIVER_RACE]
        x, y = _demean(_design(s, self.length_ref, self.length), keys), _demean(s['Level'], keys)
        b = np.linalg.lstsq(x.to_numpy(), y.to_numpy(), rcond=None)[0]
        self.coef = pd.Series(b, index=x.columns)
        self.gaps = self.coef[OTHERS]
        self.n_stints = s['Compound'].value_counts()
        # Circuit gaps: ridge toward the all circuit gaps, other terms held at their global values
        rest = [c for c in x.columns if c not in OTHERS]
        r = y - x[rest] @ self.coef[rest]
        rows = {}
        for circ, idx in s.groupby('Circuit').groups.items():
            xc, rc = x.loc[idx, OTHERS].to_numpy(), r.loc[idx].to_numpy()
            k = self.k if self.by_circuit else 1e12
            d = np.linalg.lstsq(xc.T @ xc + k * np.eye(len(OTHERS)), xc.T @ rc + k * self.gaps.to_numpy(), rcond=None)[0]
            rows[circ] = dict(zip(OTHERS, d), **{f'N_{c}': int((s.loc[idx, 'Compound'] == c).sum()) for c in COMPOUNDS})
        self.circuit_gaps = pd.DataFrame(rows).T.rename_axis('Circuit')
        return self

    def gap(self, circuit, compound):
        """Seconds per lap a fresh compound is slower than a fresh Medium at a circuit (all circuit gap if the
        circuit is new). Vectorised over Series."""
        circuit, compound = pd.Series(circuit).reset_index(drop=True), pd.Series(compound).reset_index(drop=True)
        g = pd.Series(0.0, index=circuit.index)
        for c in OTHERS:
            on = compound == c
            g[on] = circuit[on].map(self.circuit_gaps[c]).fillna(self.gaps[c]).to_numpy()
        return g

    def offset(self, s):
        """Level offset of each stint in a stint_levels frame against a fresh Medium at race fraction 0."""
        o = self.gap(s['Circuit'], s['Compound']).to_numpy() + self.coef['Frac'] * s['Frac'].to_numpy()
        if self.length:
            o = o + self.coef['Length'] * (s['StintLaps'] - self.length_ref).to_numpy()
        return pd.Series(o, index=s.index)

    def export(self):
        g = self.circuit_gaps.reset_index()
        glob = pd.DataFrame([{'Circuit': None, **self.gaps.to_dict()}])
        return {'gaps': pd.concat([glob, g], ignore_index=True),
                'terms': self.coef.drop(OTHERS).rename('Coef').rename_axis('Term').reset_index()}


class SameLevel:
    """Baseline: every later stint at the first stint's level (no compound gap, no track evolution)."""
    name = 'same level'

    def fit(self, wear, laps):
        return self

    def offset(self, s):
        return pd.Series(0.0, index=s.index)


def level_forecast(model, wear, laps, min_laps=5):
    """The evaluation task: each driver's first scored stint of a race sets the level; every later stint's level
    is predicted from it with the model's offsets. Err_s is predicted minus actual level (s per lap)."""
    s = stint_levels(wear, laps, min_laps).sort_values(STINT)
    s['Offset'] = model.offset(s)
    first = s.groupby(DRIVER_RACE).head(1).set_index(DRIVER_RACE)
    later = s[~s.index.isin(s.groupby(DRIVER_RACE).head(1).index)].join(
        first[['Level', 'Offset', 'Compound']].rename(columns=lambda c: 'First' + c), on=DRIVER_RACE)
    later['Pred_s'] = later['FirstLevel'] - later['FirstOffset'] + later['Offset']
    later['Err_s'] = later['Pred_s'] - later['Level']
    later['Pair'] = later['FirstCompound'] + ' to ' + later['Compound']
    return later
