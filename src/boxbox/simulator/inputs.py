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


def lap1_shuffle(laps, results, bases, years, sigmas=np.arange(0.0, 4.01, 0.1), seed=0):
    """Lap 1 place changes from the data (simulator v2, engine option lap1: shuffle): among classified finishers of the
    dry races of years, a car's place at the end of lap 1 less its grid place (both ranked among those cars), regressed
    on its grid place less its race pace rank (bases), so cars starting behind their pace gain. Returns (beta, sigma):
    the engine orders lap 1 by grid + beta x (pace rank - grid) + sigma x a normal draw, sigma chosen so the mean
    absolute change matches the data."""
    from boxbox import features
    r = results[results['Year'].isin(years)]
    r = r.assign(Grid=features.grid_slot(r), Finish=features.classified_finish(r))[RACE + ['Driver', 'Grid', 'Finish']]
    one = laps[laps['Year'].isin(years) & ~laps['WetRace'] & (laps['LapNumber'] == 1)].dropna(subset=['Time_s'])
    d = one[RACE + ['Driver', 'Time_s']].merge(r, on=RACE + ['Driver']).merge(bases, on=RACE + ['Driver'])
    d = d[d['Finish'].notna()]
    g = d.groupby(RACE)
    d = d.assign(GridR=g['Grid'].rank(method='first'), LapR=g['Time_s'].rank(method='first'), PaceR=g['Base'].rank(method='first'))
    chg, out = (d['LapR'] - d['GridR']).to_numpy(), (d['PaceR'] - d['GridR']).to_numpy()
    beta = float(np.polyfit(out, chg, 1)[0])
    target = np.abs(chg).mean()
    rng = np.random.default_rng(seed)
    best, err = 0.0, np.inf
    races = [x for _, x in d.groupby(RACE)]
    z = [rng.standard_normal((50, len(x))) for x in races]
    for sg in sigmas:
        m = np.mean([np.abs(np.argsort(np.argsort(x['GridR'].to_numpy() + beta * (x['PaceR'] - x['GridR']).to_numpy() + sg * zz, axis=1), axis=1)
                            + 1 - x['GridR'].rank(method='first').to_numpy()).mean() for x, zz in zip(races, z)])
        if abs(m - target) < err:
            best, err = float(sg), abs(m - target)
    return beta, best


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
        self.held_gaps = ot.held_gaps(pairs)
        self.engine = dict(cfg['simulator'].get('engine', {}))
        e = cfg['simulator'].get('engine', {})
        self.close_air = ot.close_air(laps, self.lap, years, e.get('close_air_edges', [0.0, 0.3, 0.5, 0.7, 1.0]))
        self.pass_rate = float(pairs['Pass'].mean())
        p = cfg['simulator']['pit_loss']
        stops = t['pit_loss_stops'].merge(t['events'][RACE + ['Circuit']], on=RACE, how='left')
        stops = stops[stops['Year'].isin(years)]
        if p.get('limit_correction', False) and 'reference' in cfg:
            # D26: past stops moved to the next season's pit lane speed limit (lane length / limit), so a changed limit
            # (Marina Bay and Zandvoort 2025: 60 to 80 km/h) moves the circuit's median in its first season
            from boxbox import reference
            stops = reference.limit_corrected_stops(stops, reference.load_circuits(cfg), max(years) + 1)
        self.pit = PitLoss(p['recent'], p['min_stops']).fit(stops)
        bases = self.lap.driver_bases(self.lap.stint_levels(laps, years))
        self.gaps, self.extra, self.extra_all = lap1_terms(laps, self.lap, bases, years)
        if 'results' in t:
            self.lap1 = lap1_shuffle(laps, t['results'], bases, years)
        targets = pc.pace_targets(bases, pre, t['events'])
        self.pace = pace_factory().fit(targets)
        self.refs = pc.reference_times(targets)
        pp = cfg['simulator']['pace']
        self.reference = pp.get('reference', 'last_season')
        self.team_pace_error = bool(pp.get('team_error', False))
        self.quali_ref = pc.QualiReference(pp.get('reference_shrink', 1.0)).fit(targets, t['qualifying']) if 'qualifying' in t else None
        rl = laps[laps['Year'].isin(years)].groupby(RACE + ['Circuit'])['LapNumber'].max().reset_index()
        self.race_laps = rl.sort_values('Year').groupby('Circuit')['LapNumber'].last()
        self.caps = stint_caps(t['stints'], t['events'], features.wet_races(laps), years, cfg)
        self.neutral = None
        if cfg.get('neutral') and cfg['simulator'].get('neutral', True):
            # safety cars and VSCs (simulator v2): fitted on the safety car seasons up to the last simulator one (D2)
            from boxbox.config import seasons_for
            from boxbox.neutral import NeutralModel
            sc_years = [y for y in seasons_for(cfg, 'safety_car') if y <= max(years)]
            self.neutral = NeutralModel(cfg).fit(laps, sc_years)
            self.neutral.cfg = None   # keep the fitted model free of paths
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
        with no training race uses the race's real median base: places only depend on relative pace). With
        simulator.pace.reference quali (v2) and no ref given, the race's own qualifying sets it (pace.QualiReference).
        Note: in a backtest the known (oracle) bases are used as given, so the reference only moves model pace."""
        mu = self.pace.predict(cars)
        z = rng.standard_normal((draws, len(cars)))
        share = getattr(self.pace, 'team_share', 0.0) if getattr(self, 'team_pace_error', False) else 0.0
        if share > 0 and 'Lineage' in cars:
            team = pd.factorize(cars['Lineage'].fillna(cars['Driver']))[0]
            zt = rng.standard_normal((draws, team.max() + 1))[:, team]   # one draw per team, shared by its cars
            z = np.sqrt(share) * zt + np.sqrt(1 - share) * z
        pct = mu[None, :] + self.pace.sd * z
        if ref is None and getattr(self, 'reference', 'last_season') == 'quali' and getattr(self, 'quali_ref', None) is not None                 and {'Year', 'Round'} <= set(cars.columns) and len(cars):
            ref = self.quali_ref.ref(int(cars['Year'].iloc[0]), int(cars['Round'].iloc[0]), circuit)
        b = (self.ref(circuit) if ref is None else ref) * (1 + pct / 100)
        if known is not None:
            k = np.asarray(known, float)
            b = np.where(np.isnan(k)[None, :], b, k[None, :])
        return b
