"""Cold-start growth curves.

A new listing does not sell at its steady-state rate on day one: it has no
reviews, no ranking and no sales history, so demand ramps. The heuristic handles
this with a single constant (month 1 = 45% of mature velocity). These models
replace that constant with a *fitted curve*, decomposing the forecast into

    units(t) = asymptote  x  shape(t)

where ``shape`` is estimated by pooling every historical launch cohort in the
catalogue, and ``asymptote`` (the mature ceiling the variant is heading for) is
predicted from colour/size/family attributes.

Three shapes are offered because they encode different adoption mechanics:

* **Gompertz**  — fastest early growth, long right tail. Fits listings whose
  ranking improves steadily.
* **Logistic**  — symmetric S-curve, slower start.
* **Bass**      — separates *innovation* (p, customers who find the listing on
  their own) from *imitation* (q, demand pulled in by reviews and rank). The
  fitted p/q split is diagnostic: a high q means reviews are doing the work, so
  early stock-outs are especially costly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.base import Forecaster

# Historical launch cohorts must have at least this many days of post-launch
# observation to contribute to the shape fit.
MIN_CURVE_DAYS = 60


def gompertz_cum(t: np.ndarray, b: float, c: float) -> np.ndarray:
    """Cumulative share of asymptote at age t. G(t) = exp(-b*exp(-c*t))."""
    return np.exp(-b * np.exp(-c * np.clip(t, 0, None)))


def logistic_cum(t: np.ndarray, b: float, c: float) -> np.ndarray:
    return 1.0 / (1.0 + b * np.exp(-c * np.clip(t, 0, None)))


def bass_cum(t: np.ndarray, p: float, q: float) -> np.ndarray:
    """Bass cumulative adoption fraction F(t)."""
    t = np.clip(t, 0, None)
    e = np.exp(-(p + q) * t)
    return np.clip((1 - e) / (1 + (q / max(p, 1e-6)) * e), 0, 1)


CURVES = {"gompertz": gompertz_cum, "logistic": logistic_cum, "bass": bass_cum}
CURVE_X0 = {"gompertz": [3.0, 0.03], "logistic": [20.0, 0.05], "bass": [0.004, 0.03]}
CURVE_BOUNDS = {
    "gompertz": ([1e-3, 1e-4], [50.0, 1.0]),
    "logistic": ([1e-3, 1e-4], [500.0, 1.0]),
    "bass": ([1e-5, 1e-4], [0.5, 1.0]),
}


class _GrowthModel(Forecaster):
    """Shared machinery: pooled shape fit + attribute model for the asymptote."""

    curve = "gompertz"
    name = "growth"

    def __init__(self, panel: pd.DataFrame | None = None, **kw):
        super().__init__(**kw)
        self.panel = panel
        self.shape_params_ = CURVE_X0[self.curve]

    # -- shape -------------------------------------------------------------
    def _fit_shape(self, asof: pd.Timestamp) -> None:
        """Fit one pooled ramp shape across all historical launch cohorts.

        Each cohort is normalised by its own day-90..180 plateau so that the fit
        is about *shape* only and is not dominated by high-volume launches.
        """
        if self.panel is None:
            return
        p = self.panel[(self.panel.date <= asof) & self.panel.is_organic_day]
        curves = []
        for asin, g in p.groupby("child_asin"):
            g = g.sort_values("days_since_launch")
            if g.days_since_launch.max() < MIN_CURVE_DAYS:
                continue
            plateau = g.loc[g.days_since_launch.between(90, 180), "units_ordered"].mean()
            if not np.isfinite(plateau) or plateau <= 0.05:
                continue
            w = g[g.days_since_launch <= 180].copy()
            w["age"] = w.days_since_launch
            w["norm_cum"] = w.units_ordered.cumsum() / (plateau * (w.age + 1))
            curves.append(w[["age", "norm_cum"]])
        if not curves:
            return
        allc = pd.concat(curves)
        # Median normalised cumulative-share by age = robust pooled shape.
        prof = allc.groupby("age").norm_cum.median()
        prof = prof / prof.loc[prof.index >= 120].median() if (prof.index >= 120).any() else prof
        t, yv = prof.index.to_numpy(float), np.clip(prof.to_numpy(float), 0, 1.5)

        fn = CURVES[self.curve]
        lo, hi = CURVE_BOUNDS[self.curve]
        try:
            fit = least_squares(lambda th: fn(t, *th) - yv, CURVE_X0[self.curve],
                                bounds=(lo, hi), max_nfev=5000)
            self.shape_params_ = list(fit.x)
        except Exception:
            pass
        self.shape_profile_ = prof

    def window_share(self, age_start: np.ndarray, age_end: np.ndarray) -> np.ndarray:
        """Average fraction of asymptotic velocity realised between two ages.

        This is what converts a mature-velocity asymptote into expected units
        over a specific window of the listing's life.
        """
        fn = CURVES[self.curve]
        th = self.shape_params_
        out = np.zeros(len(age_start), float)
        for i, (a, b) in enumerate(zip(age_start, age_end)):
            grid = np.arange(max(a, 0), max(b, a + 1) + 1)
            out[i] = float(np.mean(fn(grid, *th)))
        return np.clip(out, 1e-3, 1.5)

    # -- asymptote ---------------------------------------------------------
    def _fit(self, rows, X, y) -> None:
        asof = pd.Timestamp(rows["asof"].iloc[0])
        self._fit_shape(asof)

        age0 = (pd.to_datetime(rows.target_start) - rows.launch_date).dt.days.to_numpy(float)
        age1 = (pd.to_datetime(rows.target_end) - rows.launch_date).dt.days.to_numpy(float)
        share = self.window_share(age0, age1)
        days = rows.target_days.to_numpy(float)

        # Back out each training listing's implied mature velocity, then learn
        # that from attributes. Deconvolving the ramp first means the attribute
        # model is not trying to explain launch timing as well as colour appeal.
        implied = np.clip(y, 0, None) / np.clip(share * days, 1e-6, None)

        self.cols_ = list(X.columns)
        self.scaler_ = StandardScaler().fit(X.to_numpy(float))
        self.asym_model_ = Ridge(alpha=3.0).fit(
            self.scaler_.transform(X.to_numpy(float)), np.log1p(implied)
        )

    def _predict_mean(self, rows, X) -> np.ndarray:
        Z = X.reindex(columns=self.cols_, fill_value=0.0).to_numpy(float)
        asym = np.expm1(self.asym_model_.predict(self.scaler_.transform(Z)))
        age0 = (pd.to_datetime(rows.target_start) - rows.launch_date).dt.days.to_numpy(float)
        age1 = (pd.to_datetime(rows.target_end) - rows.launch_date).dt.days.to_numpy(float)
        share = self.window_share(age0, age1)
        return np.clip(asym, 0, None) * share * rows.target_days.to_numpy(float)

    def ramp_table(self) -> pd.DataFrame:
        """Realised share of mature velocity by month of life — the fitted
        replacement for the heuristic's single 0.45 constant."""
        rows = []
        for m in range(6):
            a, b = m * 30, (m + 1) * 30 - 1
            rows.append({"month": m,
                         "share_of_mature": float(self.window_share(
                             np.array([a]), np.array([b]))[0])})
        return pd.DataFrame(rows)


class GompertzRamp(_GrowthModel):
    name, curve = "gompertz", "gompertz"


class LogisticRamp(_GrowthModel):
    name, curve = "logistic", "logistic"


class BassDiffusion(_GrowthModel):
    name, curve = "bass", "bass"

    def coefficients(self) -> dict[str, float]:
        p, q = self.shape_params_
        return {"p_innovation": float(p), "q_imitation": float(q),
                "q_over_p": float(q / max(p, 1e-9))}
