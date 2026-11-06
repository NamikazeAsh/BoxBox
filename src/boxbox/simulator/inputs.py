"""Every fitted part the simulator needs, in one object fitted on the training seasons (SimInputs): the lap
model, the passing rule and follow gap, pit loss, the lap 1 start model, the car pace model and the circuit
reference times. Saved to data/models/simulator_inputs.joblib by scripts/train_simulator.py.
"""
import numpy as np
import pandas as pd

from boxbox import overtaking as ot
from boxbox.pitloss import PitLoss
from boxbox.simulator import pace as pc
from boxbox.simulator.laptime import LapModel
from boxbox import features
from boxbox.simulator.strategy import CODE, stint_caps

RACE = ['Year', 'Round']


def lap1_terms(laps, lap_model, bases, years):
    """Lap 1 from the data: the median gap to the leader at the end of lap 1 by running place, s, and per
    circuit the median over races of the leader's lap 1 time less its free air time (standing start, first
    lap traffic)."""
    one = laps[laps['Year'].isin(years) & ~laps['WetRace'] & (laps['LapNumber'] == 1) & laps['Time_s'].notna()]
    one = one.sort_values(RACE + ['Time_s'])
    one = one.assign(Place=one.groupby(RACE).cumcount() + 1, Gap=one['Time_s'] - one.groupby(RACE)['Time_s'].transform('min'))
    gaps = one.groupby('Place')['Gap'].median().sort_index().to_numpy()
    lead = one[one['Place'] == 1].merge(bases, on=RACE + ['Driver']).dropna(subset=['LapTime_s', 'Base', 'Compound'])
    lead = lead[lead['Compound'].isin(CODE)]
    race_laps = laps.groupby(RACE)['LapNumber'].max().rename('RL')
    lead = lead.join(race_laps, on=RACE)
    free = np.array([lap_model.free_air(r.Base, r.Circuit, CODE[r.Compound], r.TyreLife if r.TyreLife == r.TyreLife else 1.0, 1, r.RL)
                     for r in lead.itertuples()])
    lead['Extra'] = lead['LapTime_s'].to_numpy() - free
    return gaps, lead.groupby('Circuit')['Extra'].median(), float(lead['Extra'].median())


class SimInputs:
    def __init__(self, cfg):
        o = cfg['simulator']['overtaking']
        self.window_s = o['window_s']
        self._cfg = cfg

    def fit(self, t, pre, years, rule_factory, pace_factory):
        """t: the processed tables (laps, events, pit_loss_stops, stints). pre: pitdecision.pre_race rows. years: the
        training seasons. rule_factory, pace_factory: the chosen passing rule and pace model (make one)."""
        cfg = self._cfg
        laps = t['laps']
        u = cfg['simulator'].get('uncertainty', {})
        self.lap = LapModel(cfg).fit(laps, years, laps['Circuit'].unique(), u.get('bootstrap', 0), u.get('seed', 0))
        free = ot.free_air_laps(laps, self.lap, years)
        pairs = ot.pass_pairs(laps.loc[free.index], free, self.window_s)
        self.rule = rule_factory().fit(pairs)
        self.train_pairs = pairs   # for comparison rules in backtests; dropped by release
        self.follow_gap_s = ot.follow_gap(pairs)
        self.pass_rate = float(pairs['Pass'].mean())
        p = cfg['simulator']['pit_loss']
        stops = t['pit_loss_stops'].merge(t['events'][RACE + ['Circuit']], on=RACE, how='left')
        self.pit = PitLoss(p['recent'], p['min_stops']).fit(stops[stops['Year'].isin(years)])
        bases = self.lap.driver_bases(self.lap.stint_levels(laps, years))
        self.gaps, self.extra, self.extra_all = lap1_terms(laps, self.lap, bases, years)
        targets = pc.pace_targets(bases, pre, t['events'])
        self.pace = pace_factory().fit(targets)
        self.refs = pc.reference_times(targets)
        rl = laps[laps['Year'].isin(years)].groupby(RACE + ['Circuit'])['LapNumber'].max().reset_index()
        self.race_laps = rl.sort_values('Year').groupby('Circuit')['LapNumber'].last()
        self.caps = stint_caps(t['stints'], t['events'], features.wet_races(laps), years, cfg)
        self._cfg = None
        return self

    def release(self):
        """Make the inputs picklable for saving (drops the lap model's wear model, see LapModel.release)."""
        self.lap.release()
        self.train_pairs = None
        return self

    def start_gap(self, n):
        """Lap 1 gaps to the leader for n cars, s (places past the data's last extrapolated by its last step)."""
        g = self.gaps
        if n <= len(g):
            return g[:n]
        step = g[-1] - g[-2] if len(g) > 1 else 1.0
        return np.r_[g, g[-1] + step * np.arange(1, n - len(g) + 1)]

    def lap1_extra(self, circuit):
        return float(self.extra.get(circuit, self.extra_all))

    def ref(self, circuit):
        if circuit not in self.refs:
            raise KeyError(f'no reference time for {circuit}: v1 needs a circuit with training races')
        return float(self.refs[circuit])

    def sample_bases(self, cars, circuit, draws, rng, known=None, ref=None):
        """Base pace draws, s (draws x cars), from the pace model: reference x (1 + (prediction + error) / 100).
        cars: pre race rows (pace.FEATURES). known: optional array of bases, s, used as given where not NaN
        (oracle pace in backtests). ref: reference time, s, in place of the circuit's (a backtest at a circuit
        with no training race uses the race's real median base: places only depend on relative pace)."""
        mu = self.pace.predict(cars)
        pct = mu[None, :] + self.pace.sd * rng.standard_normal((draws, len(cars)))
        b = (self.ref(circuit) if ref is None else ref) * (1 + pct / 100)
        if known is not None:
            k = np.asarray(known, float)
            b = np.where(np.isnan(k)[None, :], b, k[None, :])
        return b
