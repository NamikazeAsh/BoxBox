"""Safety cars and virtual safety cars for the strategy simulator (simulator v2, notes/sim_v2_neutral.md).

Per lap codes come from features.neutral_laps (0 green, 1 safety car or red flag, 2 VSC). A neutral period is a run
of laps with the same nonzero code.

Deployment (NeutralModel.sample): a discrete time process over the race laps. On a lap whose previous lap was
green, a safety car period starts with probability h_sc(circuit, lap) and a VSC period with h_vsc(circuit, lap)
(competing: at most one starts); its length is drawn from that kind's durations, so no new period can start
while one runs, and the lap after a period is green. A VSC can turn into a safety car (probability p_convert per
VSC), which then starts on the lap after the VSC ends. Each kind has its own Hazard:

    lap 1:    its own probability per race (start crashes), circuit pulled toward the all circuit one;
    laps 2+:  circuit base rate (starts over laps at risk, pulled toward the all circuit rate with weight `shrink`
              races) times multipliers shared by all circuits: the opening laps 2 to open_laps, the first
              restart_laps green laps after a period, and race fraction bins.

Effects the engine needs (NeutralModel.fit_effects, same training seasons, dry races, red flag periods left out):
neutral lap time over the driver's green reference lap, the gaps of a formed safety car train and at the restart,
the laps the field takes to close up after a deployment, and a stop's loss under a safety car or VSC relative to
a green stop, each against the cars that stayed out on the same laps.
"""
import numpy as np
import pandas as pd
from scipy import stats

from boxbox import features

RACE = ['Year', 'Round']
SC, VSC = 1, 2
KINDS = {SC: 'sc', VSC: 'vsc'}
COUNTS = ['SCLaps', 'VSCLaps', 'SCPeriods', 'VSCPeriods']


# Tables

def dry_races(laps, n):
    """Dry race laps (WetRace False) without neutral.exclude_races."""
    l = laps[~laps['WetRace'].astype(bool)]
    bad = pd.MultiIndex.from_tuples([tuple(x) for x in n.get('exclude_races') or []], names=RACE)
    return l[~pd.MultiIndex.from_frame(l[RACE]).isin(bad)] if len(bad) else l


def race_codes(laps, share=0.75):
    """{(Year, Round): int8 status array, index = lap number (0 unused)} from features.neutral_laps."""
    n = features.neutral_laps(laps, share)
    out = {}
    for (y, r), g in n.groupby(RACE):
        lap = g['LapNumber'].astype(int).to_numpy()
        a = np.zeros(lap.max() + 1, np.int8)
        a[lap] = g['Neutral'].to_numpy()
        out[(int(y), int(r))] = a
    return out


def race_status(laps, year, rnd, share=0.75):
    """The real race's status array (race laps + 1,), index = lap number, for replays."""
    return race_codes(laps[(laps['Year'] == year) & (laps['Round'] == rnd)], share)[(year, rnd)]


def periods(codes):
    """One row per neutral period: Code, Start, End, Laps, RaceLaps, Censored (still running on the last lap, so its
    length is unknown), Converted (a VSC followed straight away by a safety car)."""
    rows = []
    for (y, r), a in codes.items():
        L, t = len(a) - 1, 1
        while t <= L:
            c = a[t]
            if c == 0:
                t += 1
                continue
            e = t
            while e < L and a[e + 1] == c:
                e += 1
            rows.append((y, r, int(c), t, e, e - t + 1, L, e == L, e < L and c == VSC and a[e + 1] == SC))
            t = e + 1
    cols = RACE + ['Code', 'Start', 'End', 'Laps', 'RaceLaps', 'Censored', 'Converted']
    return pd.DataFrame(rows, columns=cols).astype({c: int for c in cols[:-2]} | {'Censored': bool, 'Converted': bool})


def risk_table(codes):
    """One row per lap at risk of a new period (lap 1, or the lap before is green): Lap, RaceLaps, Event (the code
    of the period starting on it, 0 for none) and Since (laps since the last period ended, minus the green lap
    after it: 1 on the first lap a new one can start; NaN before any period)."""
    rows = []
    for (y, r), a in codes.items():
        L, last = len(a) - 1, None
        for t in range(1, L + 1):
            if t == 1 or a[t - 1] == 0:
                rows.append((y, r, t, L, int(a[t]), t - last - 1 if last else np.nan))
            if a[t] > 0:
                last = t
    return pd.DataFrame(rows, columns=RACE + ['Lap', 'RaceLaps', 'Event', 'Since'])


def deployment_tables(laps, cfg):
    """Status arrays, periods and laps at risk of every dry race in laps (exclude_races dropped), with its Circuit:
    the inputs of NeutralModel.fit_tables, built once for studies that refit many times."""
    n = cfg['neutral']
    l = dry_races(laps, n)
    codes = race_codes(l, n['lap_share'])
    circ = l.groupby(RACE)['Circuit'].first()
    circ.index = pd.MultiIndex.from_tuples([(int(y), int(r)) for y, r in circ.index], names=RACE)
    return {'codes': codes, 'circuit': circ, 'periods': periods(codes).join(circ, on=RACE),
            'risk': risk_table(codes).join(circ, on=RACE)}


# Deployment model

def _shrunk(events, exposure, prior, k, unit=1.0):
    """(events + k unit prior) / (exposure + k unit): k None gives every group the prior, 0 the raw rate."""
    if k is None:
        return pd.Series(prior, index=events.index, dtype=float)
    return (events + k * unit * prior) / (exposure + k * unit)


class Hazard:
    """Per lap start probability of one kind of period at a circuit (module docstring). shrink and lap1_shrink are
    in races (None: one all circuit value; 0: the raw circuit value). Multipliers are fitted jointly with the
    circuit rates by alternating updates and pulled toward 1 with multiplier_prior expected starts. Seasons are
    weighted season_decay ** (seasons before the latest)."""

    def __init__(self, shrink=None, lap1=True, lap1_shrink=None, open_laps=1, restart_laps=0, frac_bins=(),
                 season_decay=1.0, multiplier_prior=1.0, iters=30):
        self.shrink, self.lap1, self.lap1_shrink = shrink, lap1, lap1_shrink
        self.open_laps, self.restart_laps, self.frac_bins = open_laps, restart_laps, np.asarray(frac_bins or [], float)
        self.season_decay, self.prior, self.iters = season_decay, multiplier_prior, iters

    def _levels(self, lap, race_laps, since):
        opening = (lap >= 2) & (lap <= self.open_laps)
        restart = (since >= 1) & (since <= self.restart_laps)
        return opening, restart, np.digitize((lap - 1) / race_laps, self.frac_bins)

    def fit(self, risk, kind):
        w = self.season_decay ** (risk['Year'].max() - risk['Year']).to_numpy(float)
        y = (risk['Event'] == kind).to_numpy(float) * w
        first = (risk['Lap'] == 1).to_numpy() if self.lap1 else np.zeros(len(risk), bool)
        self.circuits = sorted(risk['Circuit'].unique())
        if self.lap1:
            g = pd.DataFrame({'C': risk['Circuit'].to_numpy()[first], 'E': y[first], 'N': w[first]}).groupby('C').sum()
            self.p1_global = g['E'].sum() / g['N'].sum()
            self.p1 = _shrunk(g['E'], g['N'], self.p1_global, self.lap1_shrink)
        r = risk[~first]
        y, w = y[~first], w[~first]
        opening, restart, fbin = self._levels(r['Lap'].to_numpy(), r['RaceLaps'].to_numpy(), r['Since'].fillna(0).to_numpy())
        circ = r['Circuit'].to_numpy()
        unit = r.groupby(RACE).size().mean()                    # laps at risk per race: shrink counts races
        self.m_open, self.m_restart, self.m_frac = 1.0, 1.0, np.ones(len(self.frac_bins) + 1)
        for _ in range(self.iters):
            mult = np.where(opening, self.m_open, 1.0) * np.where(restart, self.m_restart, 1.0) * self.m_frac[fbin]
            g = pd.DataFrame({'C': circ, 'E': y, 'X': w * mult}).groupby('C').sum()
            self.h_global = g['E'].sum() / g['X'].sum()
            self.base = _shrunk(g['E'], g['X'], self.h_global, self.shrink, unit)
            b = w * self.base.reindex(circ).to_numpy()
            if self.open_laps > 1 and opening.any():
                other = b * mult / self.m_open
                self.m_open = (y[opening].sum() + self.prior) / (other[opening].sum() + self.prior)
            if self.restart_laps > 0 and restart.any():
                mult = np.where(opening, self.m_open, 1.0) * np.where(restart, self.m_restart, 1.0) * self.m_frac[fbin]
                other = b * mult / self.m_restart
                self.m_restart = (y[restart].sum() + self.prior) / (other[restart].sum() + self.prior)
            if len(self.frac_bins):
                other = b * np.where(opening, self.m_open, 1.0) * np.where(restart, self.m_restart, 1.0)
                e = np.bincount(fbin, y, len(self.m_frac))
                x = np.bincount(fbin, other, len(self.m_frac))
                m = (e + self.prior) / (x + self.prior)
                self.m_frac = m / ((x * m).sum() / x.sum())     # exposure weighted mean 1, the base keeps the level
        return self

    def lap_rates(self, circuit, race_laps):
        """Start probability per lap (race_laps + 1,), index = lap number, outside the restart window."""
        lap = np.arange(1, race_laps + 1)
        opening, _, fbin = self._levels(lap, race_laps, np.zeros(race_laps))
        h = np.zeros(race_laps + 1)
        h[1:] = self.base.get(circuit, self.h_global) * np.where(opening, self.m_open, 1.0) * self.m_frac[fbin]
        if self.lap1:
            h[1] = self.p1.get(circuit, self.p1_global)
        return np.clip(h, 0.0, 1.0)


class PoolDuration:
    """Period length drawn from the training lengths of its kind (periods cut by the finish left out)."""
    name = 'pool'

    def __init__(self, max_laps):
        self.max_laps = max_laps

    def fit(self, lengths):
        v = np.sort(np.asarray(lengths, int))
        self.values = v if len(v) else np.array([1])
        return self

    def mean(self):
        return float(self.values.mean())

    def draw(self, u):
        return self.values[np.minimum((u * len(self.values)).astype(int), len(self.values) - 1)]

    def logpmf(self, lengths, smooth=0.5):
        """Log probability of held out lengths, counts smoothed by `smooth` on 1 to max_laps."""
        c = np.bincount(np.minimum(self.values, self.max_laps), minlength=self.max_laps + 1)[1:] + smooth
        p = c / c.sum()
        return np.log(p[np.clip(np.asarray(lengths, int), 1, self.max_laps) - 1])


class NegBinDuration:
    """Period length one plus a negative binomial count, fitted by moments on the length less one (Poisson when not overdispersed)."""
    name = 'parametric'

    def __init__(self, max_laps):
        self.max_laps = max_laps

    def fit(self, lengths):
        x = np.asarray(lengths, float) - 1
        m = x.mean() if len(x) else 0.0
        v = x.var(ddof=1) if len(x) > 1 else m
        k = np.arange(self.max_laps)
        pmf = stats.nbinom.pmf(k, m * m / (v - m), m / v) if v > m > 0 else stats.poisson.pmf(k, max(m, 1e-9))
        self.pmf = pmf / pmf.sum()
        self.cdf = np.cumsum(self.pmf)
        return self

    def mean(self):
        return float(1 + np.arange(len(self.pmf)).dot(self.pmf))

    def draw(self, u):
        return 1 + np.minimum(np.searchsorted(self.cdf, u, side='right'), self.max_laps - 1)

    def logpmf(self, lengths):
        return np.log(np.maximum(self.pmf[np.clip(np.asarray(lengths, int), 1, self.max_laps) - 1], 1e-12))


DURATIONS = {'pool': PoolDuration, 'parametric': NegBinDuration}


class NeutralModel:
    """Safety car and VSC model (module docstring), configured by cfg['neutral']. fit(laps, years) fits the
    deployment model and the effects on the dry races of the training seasons; sample draws status arrays for the
    engine. Effects (set by fit_effects):
        sc_lap_ratio, vsc_lap_ratio: median full neutral lap over the driver's green reference lap (median clean
            lap of the race); sc_leader_ratio: the same for the first car across the line (the train's pace);
            lap_ratio_table: medians by code and lap of the period (the first and last laps are partly green).
        sc_train_gaps: gaps (s) to the car ahead on the same lap once the train has formed (laps sc_close_laps on,
            not the last lap); sc_restart_gaps: the same at the line at the end of the last safety car lap;
            sc_close_laps: first lap of a period whose gaps a KS test cannot tell from the later laps' (train_gaps).
        pit_loss_ratio: {1: x, 2: y}, median stop loss with the in lap under a safety car or VSC over the circuit's
            median green stop loss, both against the stay out cars nearest it on the same two laps;
            pit_loss_ratio_pool: the per stop ratios; pit_stops: every measured stop.
        vsc_gap_ratio: median gap to the same car ahead after a VSC over before it."""

    def __init__(self, cfg):
        self.cfg, self.n = cfg, cfg['neutral']
        self.draws, self.seed = self.n['draws'], self.n['seed']

    def fit(self, laps, years, effects=True):
        """laps: data/processed/laps.parquet rows (any seasons; only years are used, dry races only)."""
        l = laps[laps['Year'].isin(years)]
        self.fit_tables(deployment_tables(l, self.cfg), years)
        if effects:
            self.fit_effects(l)
        return self

    def fit_tables(self, tables, years):
        n = self.n
        per = tables['periods'][tables['periods']['Year'].isin(years)]
        risk = tables['risk'][tables['risk']['Year'].isin(years)]
        self.years = sorted(years)
        self.hazard = {k: Hazard(multiplier_prior=n['multiplier_prior'], **n['hazard'][name]).fit(risk, k)
                       for k, name in KINDS.items()}
        done = per[~per['Censored']]
        dur = n['duration'] if isinstance(n['duration'], dict) else {name: n['duration'] for name in KINDS.values()}
        self.duration = {k: DURATIONS[dur[name]](n['max_period_laps']).fit(done.loc[done['Code'] == k, 'Laps'])
                         for k, name in KINDS.items()}
        v = per[per['Code'] == VSC]
        self.p_convert = float(v['Converted'].mean()) if n['conversion'] and len(v) else 0.0
        # circuit groups for calibration: terciles of the raw training safety car periods per race (model free)
        races = risk[risk['Lap'] == 1].groupby('Circuit').size()
        rate = (per[per['Code'] == SC].groupby('Circuit').size().reindex(races.index, fill_value=0) / races)
        lo, hi = rate.quantile([1 / 3, 2 / 3])
        self.groups = pd.Series(np.where(rate <= lo, 'low', np.where(rate <= hi, 'mid', 'high')), index=rate.index)
        return self

    def circuit_group(self, circuit):
        return self.groups.get(circuit, 'new')

    def sample(self, circuit, race_laps, draws, rng):
        """int8 status arrays (draws, race_laps + 1), index = lap number (index 0 unused), codes 0, 1, 2. The
        random numbers drawn do not depend on the outcome, so two models compared on one seed share them."""
        L, D = int(race_laps), int(draws)
        out = np.zeros((D, L + 1), np.int8)
        hs, hv = self.hazard[SC].lap_rates(circuit, L), self.hazard[VSC].lap_rates(circuit, L)
        ms, mv = self.hazard[SC].m_restart, self.hazard[VSC].m_restart
        R = max(self.hazard[SC].restart_laps, self.hazard[VSC].restart_laps)
        code, end, pend = np.zeros(D, np.int8), np.full(D, -1), np.zeros(D, int)
        for t in range(1, L + 1):
            u, us, uv, uc, us2 = rng.random((5, D))
            conv = (pend > 0) & (end == t - 1)                 # a VSC that turns into a safety car
            code[conv], end[conv], pend[conv] = SC, t + pend[conv] - 1, 0
            free = t >= end + 2                              # the lap after a period is green
            rs = (end >= 1) & (t - end - 1 <= R) if R else np.zeros(D, bool)
            a = hs[t] * np.where(rs & (t - end - 1 <= self.hazard[SC].restart_laps), ms, 1.0)
            b = hv[t] * np.where(rs & (t - end - 1 <= self.hazard[VSC].restart_laps), mv, 1.0)
            s = free & (u < a)
            v = free & ~s & (u < a + b)
            code[s], end[s] = SC, t + self.duration[SC].draw(us[s]) - 1
            code[v], end[v] = VSC, t + self.duration[VSC].draw(uv[v]) - 1
            if self.p_convert:
                c = v & (uc < self.p_convert)
                pend[c] = self.duration[SC].draw(us2[c])
            out[:, t] = np.where(t <= end, code, 0)
        return out

    def race_probs(self, circuit, race_laps, draws=None, seed=None):
        """P(any safety car), P(any VSC), expected safety car and VSC laps and periods, by sampling."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        s = race_draw_stats(self.sample(circuit, race_laps, draws or self.draws, rng))
        return {'p_sc': s['AnySC'].mean(), 'p_vsc': s['AnyVSC'].mean(), 'sc_laps': s['SCLaps'].mean(),
                'vsc_laps': s['VSCLaps'].mean(), 'sc_periods': s['SCPeriods'].mean(), 'vsc_periods': s['VSCPeriods'].mean()}

    def fit_effects(self, laps):
        n, e = self.n, self.n['effects']
        l = dry_races(laps, n)
        a = annotate(l, race_codes(l, n['lap_share']))
        r = lap_ratios(a)
        self.sc_lap_ratio, self.vsc_lap_ratio, self.sc_leader_ratio = r['sc'], r['vsc'], r['sc_leader']
        self.lap_ratio_table = r['table']
        g = train_gaps(a, e)
        self.sc_close_laps, self.sc_train_gaps, self.sc_restart_gaps = g['close_laps'], g['train'], g['restart']
        self.close_table = g['table']
        self.pit_stops = neutral_pit_loss(a, e, self.cfg['pit_loss'])
        neutral = self.pit_stops[self.pit_stops['Use'] & (self.pit_stops['Kind'] > 0)]
        self.pit_loss_ratio_pool = {k: neutral.loc[neutral['Kind'] == k, 'Ratio'].to_numpy() for k in KINDS}
        self.pit_loss_ratio = {k: float(np.median(p)) if len(p) else np.nan for k, p in self.pit_loss_ratio_pool.items()}
        v = vsc_gaps(a, e)
        self.vsc_gap_ratio = float(v['Ratio'].median()) if len(v) else np.nan
        self.vsc_gap_table = v
        return self


# Effects for the engine

def annotate(laps, codes):
    """Lap rows with the race code (Code), the period they are in (Period, K = lap of the period from 1, KBack = laps
    to its end from 1, PeriodLaps, Censored, Red: a red flag on any car's lap of the period), Status (the car's own
    status string), Ref_s (the driver's median clean lap of the race), Ratio (lap time over Ref_s), Leader (first
    across the line on that lap), Lapped (crossed after the leader finished the next lap), AheadPit (the car ahead on
    the same lap is on an in or out lap), AheadDriver."""
    cols = ['Year', 'Round', 'Driver', 'LapNumber', 'Stint', 'TrackStatus', 'LapTime_s', 'Time_s', 'PitIn', 'PitOut',
            'IsClean', 'GapAhead_s', 'Circuit']
    l = laps[cols].copy()
    l['Status'] = l['TrackStatus'].fillna('').astype(str)
    code = pd.DataFrame([(y, r, t, int(a[t])) for (y, r), a in codes.items() for t in range(1, len(a))],
                        columns=RACE + ['LapNumber', 'Code'])
    per = periods(codes).reset_index(names='Period')
    pl = per.loc[per.index.repeat(per['Laps'])].copy()
    pl['LapNumber'] = pl['Start'] + pl.groupby('Period').cumcount()
    pl['K'], pl['KBack'] = pl['LapNumber'] - pl['Start'] + 1, pl['End'] - pl['LapNumber'] + 1
    pl = pl.rename(columns={'Laps': 'PeriodLaps'})[RACE + ['LapNumber', 'Period', 'K', 'KBack', 'PeriodLaps', 'Censored']]
    for t in (code, pl):
        t['LapNumber'] = t['LapNumber'].astype(l['LapNumber'].dtype)
    l = l.merge(code, on=RACE + ['LapNumber'], how='left').merge(pl, on=RACE + ['LapNumber'], how='left')
    l['Code'] = l['Code'].fillna(0).astype(int)
    red = l.groupby('Period')['Status'].agg(lambda s: s.str.contains('5').any())
    l['Red'] = l['Period'].map(red).eq(True)
    l['Censored'] = l['Censored'].eq(True)
    ref = l[l['IsClean']].groupby(RACE + ['Driver'])['LapTime_s'].median().rename('Ref_s')
    l = l.join(ref, on=RACE + ['Driver'])
    l['Ratio'] = l['LapTime_s'] / l['Ref_s']
    lead = l.groupby(RACE + ['LapNumber'])['Time_s'].min().rename('LeadTime')
    l = l.join(lead, on=RACE + ['LapNumber']).join(lead.groupby(level=RACE).shift(-1).rename('LeadNext'), on=RACE + ['LapNumber'])
    l['Leader'] = l['Time_s'] == l['LeadTime']
    l['Lapped'] = l['Time_s'] > l['LeadNext']
    l = l.sort_values(RACE + ['LapNumber', 'Time_s'])
    g = l.groupby(RACE + ['LapNumber'])
    l['AheadPit'] = g['PitIn'].shift(1, fill_value=False).astype(bool) | g['PitOut'].shift(1, fill_value=False).astype(bool)
    l['AheadDriver'] = g['Driver'].shift(1)
    return l.sort_values(RACE + ['Driver', 'LapNumber']).reset_index(drop=True)


def _q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = pd.Series(x, dtype=float).dropna()
    return {**{f'Q{int(q * 100)}': x.quantile(q) if len(x) else np.nan for q in qs}, 'N': len(x)}


def lap_ratios(a):
    """Neutral lap time over the driver's green reference lap. Full laps: the car's own status is only 4 (safety
    car, not its first lap) or only 6 (VSC), no pit, not lapped, no red flag in the period."""
    base = a['Ratio'].notna() & ~a['PitIn'] & ~a['PitOut'] & ~a['Lapped'] & ~a['Red']
    sc_full = base & (a['Code'] == SC) & a['Status'].str.fullmatch('4+') & (a['K'] >= 2)
    vsc_full = base & (a['Code'] == VSC) & a['Status'].str.fullmatch('6+')
    rows = [{'Code': SC, 'Laps': 'full', **_q(a.loc[sc_full, 'Ratio'])},
            {'Code': SC, 'Laps': 'full, leader', **_q(a.loc[sc_full & a['Leader'], 'Ratio'])},
            {'Code': VSC, 'Laps': 'full', **_q(a.loc[vsc_full, 'Ratio'])}]
    for c in KINDS:
        m = base & (a['Code'] == c)
        single = a['PeriodLaps'] == 1
        for name, sel in [('every coded lap', m), ('first lap', m & (a['K'] == 1) & ~single),
                          ('middle laps', m & (a['K'] > 1) & (a['KBack'] > 1)), ('last lap', m & (a['KBack'] == 1) & ~single),
                          ('one lap period', m & single)]:
            rows.append({'Code': c, 'Laps': name, **_q(a.loc[sel, 'Ratio'])})
    t = pd.DataFrame(rows)
    med = lambda c, laps: float(t.loc[(t['Code'] == c) & (t['Laps'] == laps), 'Q50'].iloc[0])
    return {'sc': med(SC, 'full'), 'vsc': med(VSC, 'full'), 'sc_leader': med(SC, 'full, leader'), 'table': t}


def ks_distance(x, y):
    """Two sample Kolmogorov Smirnov statistic."""
    x, y = np.sort(np.asarray(x, float)), np.sort(np.asarray(y, float))
    z = np.concatenate([x, y])
    return float(np.abs(np.searchsorted(x, z, 'right') / len(x) - np.searchsorted(y, z, 'right') / len(y)).max())


def train_gaps(a, e):
    """Gaps to the car ahead on the same lap at the line on safety car laps (no red flag, not cut by the finish, both
    cars on track, the car not lapped). The train has formed on the first lap k of a period whose gaps (laps
    before the last) a two sample KS test at level close_alpha cannot tell from the gaps on the later laps (the
    lap with the lowest KS distance over its critical value if none passes); the train pool is the gaps from that
    lap on, not the last lap, and the restart pool those at the end of the last safety car lap."""
    ok = ((a['Code'] == SC) & ~a['Red'] & ~a['Censored'] & ~a['Lapped'] & ~a['PitIn'] & ~a['PitOut'] & ~a['AheadPit']
          & a['GapAhead_s'].notna())
    g = a[ok]
    body = g[g['KBack'] >= 2]
    c_alpha = np.sqrt(-0.5 * np.log(e['close_alpha'] / 2))
    close, rows = None, []
    for k in range(1, e['close_max'] + 1):
        x, later = body.loc[body['K'] == k, 'GapAhead_s'], body.loc[body['K'] > k, 'GapAhead_s']
        if len(x) < e['min_gaps'] or len(later) < e['min_gaps']:
            break
        d, crit = ks_distance(x, later), c_alpha * np.sqrt((len(x) + len(later)) / (len(x) * len(later)))
        rows.append({'K': k, 'KS': d, 'Critical': crit, 'Share3s': (x > 3).mean(), **_q(x)})
        if close is None and d <= crit:
            close = k
    t = pd.DataFrame(rows)
    close = close or (int(t.loc[(t['KS'] / t['Critical']).idxmin(), 'K']) if len(t) else 1)
    return {'close_laps': int(close), 'train': body.loc[body['K'] >= close, 'GapAhead_s'].to_numpy(),
            'restart': g.loc[(g['KBack'] == 1) & (g['PeriodLaps'] >= 2), 'GapAhead_s'].to_numpy(), 'table': t}


def neutral_pit_loss(a, e, pit_cfg):
    """Every tyre stop's loss against the stay out cars nearest it: its in lap plus out lap minus the mean of the
    same two laps of up to `neighbours` stay out cars on each side of it in the order at the start of the in lap
    (on the lead lap, no stop on either lap; a lap needs min_stayout of them). Green stops (both laps fully green
    for the car and the field) are corrected for car pace (twice the gap between the driver's reference lap and
    the neighbours'), kept within pit_loss min_s to max_s, and set each circuit's median green loss; a neutral stop
    (in lap coded 1 or 2, no red flag) is not corrected, since the field runs at the neutral pace. Ratio: loss over
    the circuit's median green loss (all circuit median where a circuit has none). Kind: in lap code."""
    a = a.copy()
    g = a.groupby(RACE + ['Driver'])
    a['Two_s'] = a['LapTime_s'] + g['LapTime_s'].shift(-1)
    a['NextIn'], a['NextOut'] = (g[c].shift(-1, fill_value=False).astype(bool) for c in ('PitIn', 'PitOut'))
    a['NextStint'], a['NextCode'], a['NextStatus'] = g['Stint'].shift(-1), g['Code'].shift(-1), g['Status'].shift(-1)
    a['StartTime'] = g['Time_s'].shift(1)
    common = a['Two_s'].notna() & ~a['Lapped'] & a['StartTime'].notna()
    stop = common & a['PitIn'] & a['NextOut'] & (a['NextStint'] > a['Stint'])
    stay = common & ~a['PitIn'] & ~a['PitOut'] & ~a['NextIn'] & ~a['NextOut']
    keys = set(map(tuple, a.loc[stop, RACE + ['LapNumber']].to_numpy().tolist()))
    a['Stop'], a['Stay'] = stop, stay
    nn, rows = e['neighbours'], []
    for key, q in a[stop | stay].groupby(RACE + ['LapNumber']):
        if key not in keys:
            continue
        q = q.sort_values('StartTime')
        st = q[q['Stay']]
        if len(st) < e['min_stayout']:
            continue
        times = st['StartTime'].to_numpy()
        for _, p in q[q['Stop']].iterrows():
            i = np.searchsorted(times, p['StartTime'])
            nb = st.iloc[max(0, i - nn): i + nn]
            rows.append({'Year': key[0], 'Round': key[1], 'InLap': key[2], 'Driver': p['Driver'], 'Circuit': p['Circuit'],
                         'Kind': p['Code'], 'OutKind': p['NextCode'], 'K': p['K'], 'Red': p['Red'],
                         'Green': p['Code'] == 0 and p['NextCode'] == 0 and p['Status'] == '1' and p['NextStatus'] == '1',
                         'Raw_s': p['Two_s'] - nb['Two_s'].mean(), 'Pace_s': 2 * (p['Ref_s'] - nb['Ref_s'].mean())})
    s = pd.DataFrame(rows)
    if not len(s):
        return pd.DataFrame(columns=['Kind', 'Ratio', 'Loss_s', 'Use'])
    s['Loss_s'] = np.where(s['Green'], s['Raw_s'] - s['Pace_s'], s['Raw_s'])
    green = s['Green'] & s['Loss_s'].between(pit_cfg['min_s'], pit_cfg['max_s'])
    med = s[green].groupby('Circuit')['Loss_s'].median()
    s['CircuitGreen_s'] = s['Circuit'].map(med).fillna(s.loc[green, 'Loss_s'].median())
    s['Ratio'] = s['Loss_s'] / s['CircuitGreen_s']
    s['Use'] = green | ((s['Kind'] > 0) & ~s['Red'].astype(bool))
    return s


def vsc_gaps(a, e):
    """Gap to the same car ahead at the end of the last VSC lap over the gap at the lap before the VSC, for pairs
    within vsc_gap_max_s before it, neither car on an in or out lap at either line (VSC periods not cut by the
    finish, no red flag)."""
    ok = ~a['PitIn'] & ~a['PitOut'] & ~a['AheadPit'] & ~a['Lapped'] & a['GapAhead_s'].notna()
    per = a[(a['Code'] == VSC) & ~a['Censored'] & ~a['Red']].groupby('Period').agg(
        Year=('Year', 'first'), Round=('Round', 'first'), Start=('LapNumber', 'min'), End=('LapNumber', 'max'))
    lap = a[ok].set_index(RACE + ['LapNumber', 'Driver'])[['GapAhead_s', 'AheadDriver']].sort_index()
    rows = []
    for _, p in per.iterrows():
        try:
            before = lap.loc[(p['Year'], p['Round'], p['Start'] - 1)]
            after = lap.loc[(p['Year'], p['Round'], p['End'])]
        except KeyError:
            continue
        j = before.join(after, lsuffix='_b', rsuffix='_a', how='inner')
        j = j[(j['AheadDriver_b'] == j['AheadDriver_a']) & (j['GapAhead_s_b'] <= e['vsc_gap_max_s'])]
        rows += [{'Year': p['Year'], 'Round': p['Round'], 'Before_s': b, 'After_s': x}
                 for b, x in zip(j['GapAhead_s_b'], j['GapAhead_s_a'])]
    v = pd.DataFrame(rows, columns=RACE + ['Before_s', 'After_s'])
    v['Ratio'] = v['After_s'] / v['Before_s']
    return v


# Scoring (scripts/tune_neutral.py)

def race_draw_stats(status):
    """Per draw (rows of a status array, index 0 unused): any safety car or VSC lap, their laps and periods."""
    s = np.atleast_2d(status)[:, 1:]
    sc, vsc = s == SC, s == VSC
    runs = lambda m: (m & ~np.pad(m, ((0, 0), (1, 0)))[:, :-1]).sum(1)
    return {'AnySC': sc.any(1), 'AnyVSC': vsc.any(1), 'SCLaps': sc.sum(1), 'VSCLaps': vsc.sum(1),
            'SCPeriods': runs(sc), 'VSCPeriods': runs(vsc)}


def crps(draws, y):
    """Continuous ranked probability score of the sample draws for the outcome y."""
    x = np.sort(np.asarray(draws, float))
    n = len(x)
    return float(np.abs(x - y).mean() - (2 * np.arange(1, n + 1) - n - 1).dot(x) / n ** 2)


def score_races(model, tables, year, draws, seed=0):
    """One row per dry race of the season: predicted and actual any SC, any VSC, laps and periods, with log loss,
    Brier and CRPS. Each race draws from its own seed, so models compared on one seed share random numbers."""
    rows, eps = [], 0.5 / draws
    for (y, r), a in tables['codes'].items():
        if y != year:
            continue
        circ = tables['circuit'][(y, r)]
        sim = race_draw_stats(model.sample(circ, len(a) - 1, draws, np.random.default_rng([seed, y, r])))
        act = {k: v[0] for k, v in race_draw_stats(a).items()}
        row = {'Year': y, 'Round': r, 'Circuit': circ, 'RaceLaps': len(a) - 1, 'Group': model.circuit_group(circ)}
        for k, name in [('AnySC', 'SC'), ('AnyVSC', 'VSC')]:
            p = float(np.clip(sim[k].mean(), eps, 1 - eps))
            o = float(act[k])
            row.update({f'P_{name}': p, f'Act_{name}': o, f'LogLoss_{name}': -(o * np.log(p) + (1 - o) * np.log(1 - p)),
                        f'Brier_{name}': (p - o) ** 2})
        for k in COUNTS:
            row.update({f'E_{k}': sim[k].mean(), f'Act_{k}': act[k], f'CRPS_{k}': crps(sim[k], act[k])})
        rows.append(row)
    return pd.DataFrame(rows)


def metric_table(scored):
    """Means over races: log loss and Brier of any SC / any VSC, MAE and bias of laps and periods, CRPS of laps."""
    out = {'Races': len(scored)}
    for name in ('SC', 'VSC'):
        out[f'LogLoss_{name}'] = scored[f'LogLoss_{name}'].mean()
        out[f'Brier_{name}'] = scored[f'Brier_{name}'].mean()
    for k in COUNTS:
        err = scored[f'E_{k}'] - scored[f'Act_{k}']
        out[f'MAE_{k}'], out[f'Bias_{k}'] = err.abs().mean(), err.mean()
    for k in ('SCLaps', 'VSCLaps'):
        out[f'CRPS_{k}'] = scored[f'CRPS_{k}'].mean()
    return out


def calibration(scored, by):
    """Mean predicted probability against the observed share, per group."""
    g = scored.groupby(by, observed=True)
    return pd.DataFrame({'Races': g.size(), 'P_SC': g['P_SC'].mean(), 'Obs_SC': g['Act_SC'].mean(),
                         'P_VSC': g['P_VSC'].mean(), 'Obs_VSC': g['Act_VSC'].mean(),
                         'E_SCLaps': g['E_SCLaps'].mean(), 'Act_SCLaps': g['Act_SCLaps'].mean()})


def paired_diff(a, b, col, n_boot=2000, seed=0):
    """Mean of a[col] less b[col] over the same races with a bootstrap 95% interval, and races better and worse."""
    d = a[col].to_numpy() - b[col].to_numpy()
    rng = np.random.default_rng(seed)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {'Diff': d.mean(), 'Lo': np.percentile(boot, 2.5), 'Hi': np.percentile(boot, 97.5),
            'Better': int((d < 0).sum()), 'Worse': int((d > 0).sum())}


def lap_field_ratio(laps, share, exclude=()):
    """Per race lap: its code under lap_share `share` and Field, the field's median lap over each driver's green
    reference lap (median clean lap of the race), on dry races without `exclude`."""
    l = dry_races(laps, {'exclude_races': exclude})
    ref = l[l['IsClean']].groupby(RACE + ['Driver'])['LapTime_s'].median().rename('Ref_s')
    l = l.join(ref, on=RACE + ['Driver'])
    field = (l['LapTime_s'] / l['Ref_s']).groupby([l['Year'], l['Round'], l['LapNumber']]).median().rename('Field')
    n = features.neutral_laps(l, share).join(field, on=RACE + ['LapNumber'])
    later = n['LapNumber'] > 1
    n['GreenButSlow'] = (n['Neutral'] == 0) & (n['Field'] > 1.15) & later
    n['NeutralButFast'] = (n['Neutral'] > 0) & (n['Field'] < 1.05) & later
    return n


def definition_check(laps, shares, exclude=()):
    """How each lap_share codes the laps against the field's pace: neutral laps, their median field ratio, green
    laps the field ran slow (median ratio over 1.15, lap 2 on) and neutral laps it ran near green pace (under 1.05)."""
    rows = []
    for s in shares:
        n = lap_field_ratio(laps, s, exclude)
        p = periods(race_codes(dry_races(laps, {'exclude_races': exclude}), s))
        rows.append({'Share': s, 'SCLaps': (n['Neutral'] == SC).sum(), 'VSCLaps': (n['Neutral'] == VSC).sum(),
                     'FieldRatio_SC': n.loc[n['Neutral'] == SC, 'Field'].median(),
                     'FieldRatio_VSC': n.loc[n['Neutral'] == VSC, 'Field'].median(),
                     'GreenButSlow': n['GreenButSlow'].sum(), 'NeutralButFast': n['NeutralButFast'].sum(),
                     'Miscoded': n['GreenButSlow'].sum() + n['NeutralButFast'].sum(),
                     'SCPeriods': (p['Code'] == SC).sum(), 'VSCPeriods': (p['Code'] == VSC).sum(),
                     'VSCRaces': p[p['Code'] == VSC].groupby(RACE).ngroups, 'VSCMedianLaps': p.loc[p['Code'] == VSC, 'Laps'].median()})
    return pd.DataFrame(rows)


def duration_logscore(model, tables, year):
    """Log probability of each finished period of the season under the model's durations of its kind."""
    p = tables['periods']
    p = p[(p['Year'] == year) & ~p['Censored']]
    return pd.concat([p[p['Code'] == k].assign(LogScore=model.duration[k].logpmf(p.loc[p['Code'] == k, 'Laps']))
                      for k in KINDS])
