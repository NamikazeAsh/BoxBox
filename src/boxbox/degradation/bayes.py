"""Hierarchical Bayesian wear rates (NumPyro, CPU). The linear model's shrinkage weight is hand tuned and counts
stints as if they were independent, but stints in the same race share that race's conditions (O7: each race
departs from its circuit's history by a real 0.027 s per lap). Here those pieces are explicit and their sizes
are learned:

    slope of stint s = mu[compound] + lam[compound] x (stint length - ref)
                       + circuit and compound effect (spread tau_a[compound])
                       + race effect (tau_r) + team in season effect (tau_g) + stint scatter (tau_u)

Each stint enters through its within stint least squares slope and that slope's sampling variance (sigma^2 /
Sxx, sigma the lap noise), so the sampler sees about 3,800 stints, not 57,000 laps (a lap level Gaussian model
reduces exactly to this). Dirty air and race start offsets come from the linear model fitted first.

The forecast is the posterior mean slope; a new race, team season and stint draw fresh effects, so their spread
widens the predictive interval along with the anchor's own noise and the lap noise.
"""
import numpy as np
import pandas as pd
from scipy import stats

from boxbox.degradation.linear import LinearDegradation, lap_terms_kw
from boxbox.degradation.task import AGE_TERMS, COMPOUNDS, KEY, RACE, STINT, anchored_forecast

SEASON_TEAM = ['Year', 'Team']


def stint_stats(laps, y):
    """Per stint: within stint slope of y on tyre age, Sxx, laps, and residual sum of squares."""
    keys = [laps[k] for k in STINT]
    x = laps['TyreLife'].astype(float)
    dx, dy = x - x.groupby(keys).transform('mean'), y - y.groupby(keys).transform('mean')
    g = pd.DataFrame({'xy': dx * dy, 'xx': dx * dx, 'yy': dy * dy}).groupby(keys).sum()
    first = laps.drop_duplicates(STINT).set_index(STINT)
    g = g.join(first[['Circuit', 'Compound', 'StintLaps'] + (['Team'] if 'Team' in first else [])])
    g['n'] = laps.groupby(STINT).size()
    g = g[(g['xx'] > 0) & (g['n'] >= 3)]
    g['b'] = g['xy'] / g['xx']
    g['rss'] = g['yy'] - g['xy'] ** 2 / g['xx']
    return g


def _model(k, cell, cell_k, race, team, z, se2, n_cell, n_race, n_team, b=None):
    import numpyro
    import numpyro.distributions as dist
    mu = numpyro.sample('mu', dist.Normal(0.05, 0.1).expand([3]).to_event(1))
    lam = numpyro.sample('lam', dist.Normal(0.0, 0.01).expand([3]).to_event(1))
    tau_a = numpyro.sample('tau_a', dist.HalfNormal(0.05).expand([3]).to_event(1))
    tau_r = numpyro.sample('tau_r', dist.HalfNormal(0.05))
    tau_g = numpyro.sample('tau_g', dist.HalfNormal(0.05))
    tau_u = numpyro.sample('tau_u', dist.HalfNormal(0.1))
    a = numpyro.sample('a_raw', dist.Normal(0, 1).expand([n_cell]).to_event(1)) * tau_a[cell_k]
    r = numpyro.sample('r_raw', dist.Normal(0, 1).expand([n_race]).to_event(1)) * tau_r
    g = numpyro.sample('g_raw', dist.Normal(0, 1).expand([n_team]).to_event(1)) * tau_g
    numpyro.deterministic('a', a)
    m = mu[k] + lam[k] * z + a[cell] + r[race] + g[team]
    numpyro.sample('b', dist.Normal(m, (se2 + tau_u ** 2) ** 0.5), obs=b)


class BayesDegradation:
    """See the module docstring. cfg supplies the dirty air and race start terms and the stint length reference."""
    age_term = 'linear'

    def __init__(self, cfg, warmup=500, samples=500, chains=2, seed=0):
        self.cfg, self.warmup, self.samples, self.chains, self.seed = cfg, warmup, samples, chains, seed
        self.length_ref = cfg['degradation']['stint_length_ref']
        self.f = AGE_TERMS['linear']
        self.name = 'bayes hierarchical, stint length, dirty air, race start'

    def fit(self, laps):
        import jax
        from numpyro.infer import MCMC, NUTS
        self.lin = LinearDegradation(10, stint_length=True, length_ref=self.length_ref, **lap_terms_kw(self.cfg)).fit(laps)
        y = laps['LapTimeFC_s'] - self.lin.dirty_air(laps) - self.lin._offset(laps, ['DirtyAir'], exclude=True)
        s = stint_stats(laps, y)
        self.sigma2 = s['rss'].sum() / (s['n'] - 2).clip(lower=1).sum()
        s = s.reset_index()
        self.cells = pd.MultiIndex.from_frame(s[KEY].drop_duplicates())
        self.races = pd.MultiIndex.from_frame(s[RACE].drop_duplicates())
        team = s[['Year', 'Team']] if 'Team' in s else s[['Year']].assign(Team='all')
        self.teams = pd.MultiIndex.from_frame(team.drop_duplicates())
        cell = self.cells.get_indexer(pd.MultiIndex.from_frame(s[KEY]))
        data = dict(k=s['Compound'].map(COMPOUNDS.index).to_numpy(), cell=cell,
                    cell_k=self.cells.get_level_values('Compound').map(COMPOUNDS.index).to_numpy(),
                    race=self.races.get_indexer(pd.MultiIndex.from_frame(s[RACE])),
                    team=self.teams.get_indexer(pd.MultiIndex.from_frame(team)),
                    z=(s['StintLaps'] - self.length_ref).to_numpy(float), se2=(self.sigma2 / s['xx']).to_numpy(),
                    n_cell=len(self.cells), n_race=len(self.races), n_team=len(self.teams))
        mcmc = MCMC(NUTS(_model), num_warmup=self.warmup, num_samples=self.samples, num_chains=self.chains,
                    chain_method='sequential', progress_bar=False)
        mcmc.run(jax.random.PRNGKey(self.seed), b=s['b'].to_numpy(), **data)
        post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        self.post = post
        k = self.cells.get_level_values('Compound').map(COMPOUNDS.index).to_numpy()
        cell_draws = post['mu'][:, k] + post['a']                     # draws x cells
        self.cell_mean = pd.Series(cell_draws.mean(0), index=self.cells)
        self.cell_var = pd.Series(cell_draws.var(0), index=self.cells)
        self.mu = pd.Series(post['mu'].mean(0), index=COMPOUNDS)
        self.mu_var = pd.Series(post['mu'].var(0), index=COMPOUNDS)
        self.lam = pd.Series(post['lam'].mean(0), index=COMPOUNDS)
        self.lam_var = pd.Series(post['lam'].var(0), index=COMPOUNDS)
        self.tau = {n: float((post[n] ** 2).mean()) ** 0.5 for n in ['tau_r', 'tau_g', 'tau_u']}
        self.tau_a = pd.Series((post['tau_a'] ** 2).mean(0) ** 0.5, index=COMPOUNDS)
        self.n_divergent = int(np.asarray(mcmc.get_extra_fields()['diverging']).sum())
        return self

    def _z(self, laps):
        return laps['StintLaps'] - self.length_ref

    def slopes_for(self, laps):
        m = laps.join(self.cell_mean.rename('_m'), on=KEY)['_m'].fillna(laps['Compound'].map(self.mu)).fillna(0.0)
        return m + laps['Compound'].map(self.lam).fillna(0.0) * self._z(laps)

    def slope_var(self, laps):
        """Predictive variance of a new stint's slope: the cell estimate's posterior variance (a new circuit: the
        compound mean's plus the circuit spread), the stint length coefficient's, and fresh race, team season and
        stint effects."""
        v = laps.join(self.cell_var.rename('_v'), on=KEY)['_v']
        new = laps['Compound'].map(self.mu_var) + laps['Compound'].map(self.tau_a ** 2)
        v = v.fillna(new)
        return (v + laps['Compound'].map(self.lam_var) * self._z(laps) ** 2
                + self.tau['tau_r'] ** 2 + self.tau['tau_g'] ** 2 + self.tau['tau_u'] ** 2)

    def wear(self, laps):
        return self.slopes_for(laps) * laps['TyreLife'].clip(lower=1) + self.lin._offset(laps, ['DirtyAir'], exclude=True)

    def dirty_air(self, laps):
        return self.lin.dirty_air(laps)

    def forecast(self, laps, cfg, levels=None):
        """Anchored forecast with Gaussian predictive quantiles (levels from degradation.intervals)."""
        levels = levels or cfg['degradation']['intervals']['levels']
        p = anchored_forecast(self, laps, cfg)
        i = laps.groupby(STINT).cumcount()
        a = laps[i < cfg['degradation']['anchor_laps']]
        xbar = a.groupby(STINT)['TyreLife'].mean().rename('_xbar')
        n_a = a.groupby(STINT).size().rename('_na')
        p = p.join(xbar, on=STINT).join(n_a, on=STINT)
        sd = np.sqrt(self.sigma2 * (1 + 1 / p['_na']) + (p['TyreLife'] - p['_xbar']) ** 2 * self.slope_var(p))
        for q in levels:
            p[f'Q{round(100 * q)}'] = p['Pred_s'] + stats.norm.ppf(q) * sd
        p['PredSD_s'] = sd
        return p.drop(columns=['_xbar', '_na'])

    def export(self):
        """Posterior summaries: compound level terms and spreads, slope mean and variance per circuit and
        compound, and the linear model's lap level offsets it uses."""
        cells = pd.DataFrame({'SlopeMean': self.cell_mean, 'SlopeVar': self.cell_var}).reset_index()
        return {'summary': self.summary().rename('Value').rename_axis('Term').reset_index(), 'cells': cells,
                'offsets': self.lin.export()['offsets']}

    def summary(self):
        return pd.Series({**{f'mu {c}': v for c, v in self.mu.items()}, **{f'lam {c}': v for c, v in self.lam.items()},
                          **{f'tau_a {c}': v for c, v in self.tau_a.items()}, **self.tau,
                          'sigma (lap noise)': self.sigma2 ** 0.5, 'divergent': self.n_divergent})
