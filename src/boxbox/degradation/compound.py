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

Race fraction (frac): 'mid' (v1) reads it at the stint's middle lap; 'start' at the lap the stint started. A wear
slope is fitted within stints, where tyre age and race lap move together, so it already carries the track
evolution and fuel error that happen during a stint. A level then only moves with the lap the stint started on,
and 'start' is the definition that matches a lap model applying evolution once per stint
(simulator.compound_pace.evolution: stint_start, notes/sim_v2_search.md).

Basis (simulator.compound_pace.basis): 'label' (v1) fits one gap per Soft and Hard label; the C number bases read
the Pirelli compound behind each label at each race (data/reference/pirelli_compounds.csv, c_allocation): 'c_step'
fits a cost per C step softer and per C step harder than that race's Medium, 'c_index' a fresh pace per C number.
A label hides different tyres (Soft is C5 at Monaco, C3 at Sakhir), so pooling by label mixes tyres across circuits
and seasons. The circuit gaps then become deviations from the C number gap, pulled toward 0 with weight shrink.

race_gap_spread: how far one race's gaps depart from the fitted circuit gaps beyond sampling error, by the method
of moments on each training race's own within driver gaps. The strategy search draws one such shift per Monte
Carlo draw (simulator.compound_pace.race_spread).
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
    start = (laps['LapNumber'] - laps['TyreLife']).groupby([laps[k] for k in STINT]).min().clip(lower=0) + 1
    s['Start'] = start.to_numpy()
    s = s[s['N'] >= min_laps]
    return s.assign(Frac=s['Mid'] / s['RaceLaps'], FracStart=(s['Start'] - 1) / s['RaceLaps'])


def _frac(s, at='mid'):
    """Race fraction of each stint: at its middle lap (v1) or its start lap (falls back to the middle when the
    frame has no FracStart, as overtaking.free_air_laps builds it)."""
    return s['FracStart'] if at == 'start' and 'FracStart' in s else s['Frac']


def _design(s, length_ref, length, frac='mid'):
    x = pd.DataFrame({c: (s['Compound'] == c).astype(float) for c in OTHERS}, index=s.index)
    x['Frac'] = _frac(s, frac)
    if length:
        x['Length'] = s['StintLaps'] - length_ref
    return x


def _demean(df, keys):
    return df - df.groupby(keys).transform('mean')


def c_allocation(cfg):
    """The Pirelli compound behind each label at each race on one scale (features.compound_index: 2022's C1 is the
    later C0): Year, Round, Circuit, Compound, CIdx."""
    from boxbox import features, reference
    d = reference.compounds_long(reference.load_compounds(cfg))
    d['CIdx'] = features.compound_index(d['Year'], d['CNumber'], cfg)
    return d.dropna(subset=['CIdx'])[RACE + ['Circuit', 'Compound', 'CIdx']].reset_index(drop=True)


def c_columns(s, alloc):
    """Each stint's C index and its race's Medium's (Year, Round, Compound in s). A race or label missing from the
    table falls back to one step from the Medium (Soft +1, Hard -1); a race with no Medium at all to C3."""
    a = alloc.set_index(RACE + ['Compound'])['CIdx']
    yr, rd = s['Year'].astype(int).to_numpy(), s['Round'].astype(int).to_numpy()
    key = pd.MultiIndex.from_arrays([yr, rd, s['Compound'].to_numpy()])
    med_key = pd.MultiIndex.from_arrays([yr, rd, np.full(len(s), 'MEDIUM', dtype=object)])
    med = pd.Series(a.reindex(med_key).to_numpy(), index=s.index).fillna(3.0)
    step = s['Compound'].map({'SOFT': 1.0, 'MEDIUM': 0.0, 'HARD': -1.0})
    c = pd.Series(a.reindex(key).to_numpy(), index=s.index).fillna(med + step)
    return c, med


class CompoundPace:
    def __init__(self, shrink=20, min_laps=5, length=False, length_ref=20, by_circuit=True, frac='mid', basis='label',
                 alloc=None):
        self.k, self.min_laps, self.length, self.length_ref, self.by_circuit = shrink, min_laps, length, length_ref, by_circuit
        self.frac, self.basis, self.alloc = frac, basis, alloc
        if basis != 'label' and alloc is None:
            raise ValueError('a C number basis needs the allocation table (c_allocation)')
        self.name = ('compound pace, ' + (f'shrink {shrink}' if by_circuit else 'all circuits')
                     + (', stint length' if length else '') + (', fraction at stint start' if frac == 'start' else '')
                     + ('' if basis == 'label' else ', ' + basis.replace('_', ' ')))

    def _cdesign(self, s):
        """C number terms of the gap to the race's Medium (c_step: steps softer, steps harder; c_index: one hot C)."""
        c, med = c_columns(s, self.alloc)
        if self.basis == 'c_step':
            d = c - med
            return pd.DataFrame({'CSofter': d.clip(lower=0), 'CHarder': (-d).clip(lower=0)}, index=s.index)
        return pd.DataFrame({f'C{i}': (c == i).astype(float) - (med == i).astype(float) for i in range(7)}, index=s.index)

    def _c_gap(self, s):
        """The C number part of each stint's gap to its race's fresh Medium, s per lap."""
        x = self._cdesign(s).reindex(columns=self.ccols, fill_value=0.0)
        return x @ self.coef[self.ccols]

    def fit(self, wear, laps):
        s = stint_levels(wear, laps, self.min_laps)
        s = s[s.groupby(DRIVER_RACE)['Stint'].transform('size') >= 2]
        keys = [s[k] for k in DRIVER_RACE]
        if getattr(self, 'basis', 'label') != 'label':
            return self._fit_c(s, keys)
        x, y = _demean(_design(s, self.length_ref, self.length, self.frac), keys), _demean(s['Level'], keys)
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

    def _fit_c(self, s, keys):
        """C number basis: gap = C number terms + a circuit deviation per label (ridge toward 0, weight shrink)."""
        xc = self._cdesign(s)
        xc = xc.loc[:, (xc != 0).any()]
        x = pd.concat([xc, pd.DataFrame({'Frac': _frac(s, self.frac)}, index=s.index)], axis=1)
        if self.length:
            x['Length'] = s['StintLaps'] - self.length_ref
        xd, y = _demean(x, keys), _demean(s['Level'], keys)
        b = np.linalg.lstsq(xd.to_numpy(), y.to_numpy(), rcond=None)[0]
        self.coef = pd.Series(b, index=x.columns)
        self.ccols = list(xc.columns)
        r = y - xd @ self.coef
        lab = _demean(pd.DataFrame({c: (s['Compound'] == c).astype(float) for c in OTHERS}, index=s.index), keys)
        k = self.k if self.by_circuit else 1e12
        rows = {}
        for circ, idx in s.groupby('Circuit').groups.items():
            X = lab.loc[idx].to_numpy()
            d = np.linalg.lstsq(X.T @ X + k * np.eye(len(OTHERS)), X.T @ r.loc[idx].to_numpy(), rcond=None)[0]
            rows[circ] = dict(zip(OTHERS, d), **{f'N_{c}': int((s.loc[idx, 'Compound'] == c).sum()) for c in COMPOUNDS})
        self.circuit_dev = pd.DataFrame(rows).T.rename_axis('Circuit')
        self.n_stints = s['Compound'].value_counts()
        # label gaps at each circuit's latest known allocation (the simulator's per circuit gaps and the export)
        latest = self.alloc.sort_values(RACE).groupby(['Circuit', 'Compound']).last().reset_index()
        latest = latest[latest['Compound'].isin(OTHERS)].reset_index(drop=True)
        latest['Gap'] = self._c_gap(latest).to_numpy()
        cg = latest.pivot(index='Circuit', columns='Compound', values='Gap').reindex(columns=OTHERS)
        for c in OTHERS:
            cg[c] = cg[c] + self.circuit_dev[c].reindex(cg.index).fillna(0.0)
        self.circuit_gaps = cg
        one = pd.DataFrame({'Year': [0, 0], 'Round': [0, 0], 'Compound': OTHERS})
        self.gaps = pd.Series(self._c_gap(one).to_numpy(), index=OTHERS)   # one step from a C3 Medium: new circuit fallback
        return self

    def gap(self, circuit, compound):
        """Seconds per lap a fresh compound is slower than a fresh Medium at a circuit (all circuit gap if the
        circuit is new). Vectorised over Series. C number bases: at the circuit's latest known allocation."""
        circuit, compound = pd.Series(circuit).reset_index(drop=True), pd.Series(compound).reset_index(drop=True)
        g = pd.Series(0.0, index=circuit.index)
        for c in OTHERS:
            on = compound == c
            g[on] = circuit[on].map(self.circuit_gaps[c]).fillna(self.gaps[c]).to_numpy()
        return g

    def offset(self, s):
        """Level offset of each stint in a stint_levels frame against a fresh Medium at race fraction 0. C number
        bases use the stint's own race allocation."""
        if getattr(self, 'basis', 'label') != 'label':
            dev = pd.Series(0.0, index=s.index)
            for c in OTHERS:
                on = s['Compound'] == c
                dev[on] = s.loc[on, 'Circuit'].map(self.circuit_dev[c]).fillna(0.0).to_numpy()
            o = self._c_gap(s).to_numpy() + dev.to_numpy() + self.coef['Frac'] * _frac(s, self.frac).to_numpy()
        else:
            o = self.gap(s['Circuit'], s['Compound']).to_numpy() + self.coef['Frac'] * _frac(s, getattr(self, 'frac', 'mid')).to_numpy()
        if self.length:
            o = o + self.coef['Length'] * (s['StintLaps'] - self.length_ref).to_numpy()
        return pd.Series(o, index=s.index)

    def export(self):
        g = self.circuit_gaps.reset_index()
        glob = pd.DataFrame([{'Circuit': None, **self.gaps.to_dict()}])
        return {'gaps': pd.concat([glob, g], ignore_index=True),
                'terms': self.coef.drop(OTHERS, errors='ignore').rename('Coef').rename_axis('Term').reset_index()}


def race_gap_spread(pace, wear, laps, min_laps=5, min_info=1.0):
    """Race to race spread of the compound gaps, s per lap, per compound other than Medium: each race's own gaps
    (within driver regression of the stint levels' residual on the fitted model, race fraction held at the
    fitted value) depart from the circuit gaps by their sampling error plus a real shift; the spread is
    sqrt(mean dev^2 - mean SE^2), floored at 0 (method of moments, as degradation.noise does for wear slopes).
    A race informs a compound only if its demeaned dummy has at least min_info sum of squares (about four
    drivers who ran it beside another compound). Returns ({compound: spread}, per race table)."""
    s = stint_levels(wear, laps, min_laps)
    s = s[s.groupby(DRIVER_RACE)['Stint'].transform('size') >= 2]
    s = s.assign(R=s['Level'] - pace.offset(s).to_numpy())
    rows = []
    for (y, r), g in s.groupby(RACE):
        keys = [g[k] for k in DRIVER_RACE]
        x = _demean(pd.DataFrame({c: (g['Compound'] == c).astype(float) for c in OTHERS}, index=g.index), keys)
        yd = _demean(g['R'], keys).to_numpy()
        cols = [c for c in OTHERS if (x[c] ** 2).sum() >= min_info]
        if not cols:
            continue
        X = x[cols].to_numpy()
        if np.linalg.matrix_rank(X) < len(cols):
            continue
        b = np.linalg.lstsq(X, yd, rcond=None)[0]
        e = yd - X @ b
        dof = len(g) - len(cols) - g.groupby(DRIVER_RACE).ngroups
        if dof < 3:
            continue
        cov = (e @ e / dof) * np.linalg.inv(X.T @ X)
        rows += [dict(Year=y, Round=r, Compound=c, Dev=b[i], SE=np.sqrt(cov[i, i])) for i, c in enumerate(cols)]
    t = pd.DataFrame(rows, columns=['Year', 'Round', 'Compound', 'Dev', 'SE'])
    spread = {}
    for c in OTHERS:
        g = t[t['Compound'] == c]
        spread[c] = float(np.sqrt(max((g['Dev'] ** 2).mean() - (g['SE'] ** 2).mean(), 0.0))) if len(g) >= 3 else 0.0
    return spread, t


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
