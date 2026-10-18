"""Tyre degradation models (modelling D6, D8), one module per family on a shared task:

* task: clean race laps and the anchored forecast every model is scored on.
* linear: per circuit and compound wear rates (the chosen model) and its candidate grid.
* adjust: updates from the same season, the same weekend and the same race.
* trees: LightGBM variants.
* gam: smooth wear curves (P splines) on top of the linear rates.
* bayes: hierarchical Bayesian wear rates with race and driver effects.
* intervals: conformal intervals and interval scores for any model.
* compare: every model family on the same folds.

Everything public is re exported here, so `from boxbox import degradation as dg` reaches all of it.
"""
from boxbox.degradation.task import *          # noqa: F401,F403
from boxbox.degradation.linear import *        # noqa: F401,F403
from boxbox.degradation.adjust import *        # noqa: F401,F403
