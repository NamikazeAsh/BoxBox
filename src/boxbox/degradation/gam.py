"""Curved wear on top of the linear model. SplineDegradation: a penalised spline (P spline) curve in tyre age
per compound, a generalised additive model on the same within stint fit. QuadraticDegradation: the quadratic
form of Heilmeier et al. (2018), one age squared term per compound. The spline answers "is wear linear in tyre age?" without
fixing the shape in advance (the square root and log tests in O8 fixed it).

wear = circuit and compound slope x tyre age + curve_compound(tyre age) + race start offsets.
The curve has its straight line part removed, so the circuit slopes keep the linear trend and the curve only
carries bends (warm up, easing off, a cliff). Its wiggliness is penalised by second differences of the spline
coefficients, weight lam, tuned on validation. Ages beyond max_age keep the curve's value at max_age (the
slope still applies), so forecasts extend in a straight line.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from boxbox.degradation.linear import LinearDegradation, flat_frame
from boxbox.degradation.task import COMPOUNDS


def spline_basis(age, knot_step, max_age, degree=3):
    """B spline basis in tyre age (columns), clamped to 1 to max_age, with the constant and straight line part
    of each column removed (projected out on a grid of whole laps)."""
    inner = np.unique(np.r_[np.arange(1, max_age, knot_step), max_age]).astype(float)
    t = np.r_[[inner[0]] * degree, inner, [inner[-1]] * degree]
    make = lambda a: BSpline.design_matrix(np.clip(np.asarray(a, dtype=float), inner[0], inner[-1]), t, degree).toarray()
    grid = np.arange(1, inner[-1] + 1, dtype=float)
    line = np.column_stack([np.ones_like(grid), grid])
    coef = np.linalg.lstsq(line, make(grid), rcond=None)[0]
    a = np.clip(np.asarray(age, dtype=float), inner[0], inner[-1])
    return make(a) - np.column_stack([np.ones_like(a), a]) @ coef


class SplineDegradation(LinearDegradation):
    """LinearDegradation plus a smooth curve in tyre age per compound (see module docstring)."""

    def __init__(self, shrink_stints=10, lam=100.0, knot_step=5, max_age=50, **kw):
        super().__init__(shrink_stints, **kw)
        self.lam, self.knot_step, self.max_age = lam, knot_step, max_age
        self.name += f', spline curve lam {lam}'

    def _design(self, laps):
        d = super()._design(laps)
        b = spline_basis(laps['TyreLife'].fillna(1), self.knot_step, self.max_age)
        cols = {}
        for comp in COMPOUNDS:
            on = (laps['Compound'] == comp).to_numpy()[:, None]
            cols.update({('Curve', comp, j): v for j, v in enumerate((b * on).T)})
        return flat_frame({**dict(zip(d.columns, d.to_numpy().T)), **cols}, laps.index)

    def _penalty(self, columns):
        rows = []
        for i, c in enumerate(columns):
            if c[0] == 'Curve' and i + 2 < len(columns) and columns[i + 2][:2] == c[:2]:
                r = np.zeros(len(columns))
                r[[i, i + 1, i + 2]] = np.sqrt(self.lam) * np.array([1.0, -2.0, 1.0])
                rows.append(r)
        return np.array(rows).reshape(-1, len(columns))

    def export(self):
        return {**super().export(), 'curve': self.curve().rename_axis('TyreLife').reset_index()}

    def curve(self, ages=None):
        """The fitted bend per compound at the given tyre ages (seconds, straight line part removed)."""
        ages = np.arange(1, self.max_age + 1) if ages is None else np.asarray(ages)
        b = spline_basis(ages, self.knot_step, self.max_age)
        return pd.DataFrame({c: b @ self.offsets[[k for k in self.offsets.index if k[:2] == ('Curve', c)]].to_numpy()
                             for c in COMPOUNDS if any(k[:2] == ('Curve', c) for k in self.offsets.index)}, index=ages)


class QuadraticDegradation(LinearDegradation):
    """LinearDegradation plus one age squared term per compound, the quadratic tyre model of Heilmeier et al.
    (2018): wear = slope x age + c x age^2. The age squared column has its straight line part removed on ages 1
    to max_age (so the circuit slopes keep the linear trend) and ages are clipped at max_age. Unpenalised."""

    def __init__(self, shrink_stints=10, max_age=50, **kw):
        super().__init__(shrink_stints, **kw)
        self.max_age = max_age
        self.name += ', quadratic'

    def _quad(self, age):
        grid = np.arange(1, self.max_age + 1, dtype=float)
        line = np.polyfit(grid, grid ** 2, 1)
        a = np.clip(np.asarray(age, dtype=float), 1, self.max_age)
        return a ** 2 - np.polyval(line, a)

    def _design(self, laps):
        d = super()._design(laps)
        q = self._quad(laps['TyreLife'].fillna(1))
        cols = {('Curve', comp, 'quad'): q * (laps['Compound'] == comp).to_numpy() for comp in COMPOUNDS}
        return flat_frame({**dict(zip(d.columns, d.to_numpy().T)), **cols}, laps.index)

    def curve(self, ages=None):
        """The fitted bend per compound at the given tyre ages (seconds, straight line part removed)."""
        ages = np.arange(1, self.max_age + 1) if ages is None else np.asarray(ages)
        return pd.DataFrame({c: self.offsets[('Curve', c, 'quad')] * self._quad(ages)
                             for c in COMPOUNDS if ('Curve', c, 'quad') in self.offsets.index}, index=ages)

    def export(self):
        return {**super().export(), 'curve': self.curve().rename_axis('TyreLife').reset_index()}
