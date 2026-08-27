"""Conformal calibration of predictive intervals.

Measured problem: across seven launch events, *every* model's nominal 80%
interval under-covered — from 0.18 (ridge) to 0.71 (size_ratio) against a 0.80
target. Model-derived intervals are too narrow here because they account for
parameter and residual uncertainty but not for the dominant term in a cold
start: the fact that a colour's appeal is genuinely unknown until it ships.

That matters commercially, not just statistically. Safety stock is set from the
upper quantile of the forecast, so an interval that covers 40% of the time
understates the risk of a month-3 stock-out by roughly half.

Split-conformal calibration fixes it without touching the models. Given
out-of-sample residuals on the *log* scale,

    r = log1p(y) - log1p(yhat)

the calibrated interval is ``expm1(log1p(yhat) + quantile(r, tau))``. Working in
log space makes the correction multiplicative, which is right for count demand:
a SKU forecast at 200 units needs a wider absolute band than one forecast at 20.

Calibration residuals come only from launch events *strictly earlier* than the
one being predicted (``leave-one-origin-out``), so the coverage guarantee is not
bought with information from the test window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ConformalCalibrator:
    """Multiplicative interval calibration from historical residual quantiles."""

    def __init__(self, lo_tau: float = 0.10, hi_tau: float = 0.90):
        self.lo_tau, self.hi_tau = lo_tau, hi_tau
        self.q_lo_: float = 0.0
        self.q_hi_: float = 0.0
        self.n_: int = 0

    def fit(self, y: np.ndarray, yhat: np.ndarray) -> "ConformalCalibrator":
        y, yhat = np.asarray(y, float), np.asarray(yhat, float)
        r = np.log1p(np.clip(y, 0, None)) - np.log1p(np.clip(yhat, 0, None))
        r = r[np.isfinite(r)]
        self.n_ = len(r)
        if self.n_ < 5:
            # Not enough history to calibrate; fall back to a wide, honest prior
            # rather than pretending to a tight interval.
            self.q_lo_, self.q_hi_ = -1.2, 1.2
            return self
        self.q_lo_ = float(np.quantile(r, self.lo_tau))
        self.q_hi_ = float(np.quantile(r, self.hi_tau))
        return self

    def apply(self, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base = np.log1p(np.clip(np.asarray(yhat, float), 0, None))
        lo = np.clip(np.expm1(base + self.q_lo_), 0, None)
        hi = np.clip(np.expm1(base + self.q_hi_), 0, None)
        return lo, hi

    def quantile(self, yhat: np.ndarray, tau: float) -> np.ndarray:
        """Arbitrary calibrated quantile — this is what the order rule uses to
        pick a service level rather than a fixed 80% band."""
        if self.n_ < 5:
            z = {0.5: 0.0, 0.9: 1.2, 0.95: 1.6}.get(tau, 0.0)
            return np.clip(np.expm1(np.log1p(np.clip(yhat, 0, None)) + z), 0, None)
        q = float(np.quantile(self._resid, tau))
        return np.clip(np.expm1(np.log1p(np.clip(yhat, 0, None)) + q), 0, None)

    def fit_store(self, y: np.ndarray, yhat: np.ndarray) -> "ConformalCalibrator":
        """Same as ``fit`` but retains the residual sample for arbitrary quantiles."""
        self.fit(y, yhat)
        r = np.log1p(np.clip(np.asarray(y, float), 0, None)) - np.log1p(
            np.clip(np.asarray(yhat, float), 0, None))
        self._resid = r[np.isfinite(r)]
        return self

    def __repr__(self) -> str:
        return (f"ConformalCalibrator(n={self.n_}, "
                f"lo_mult={np.expm1(self.q_lo_):+.2f}, hi_mult={np.expm1(self.q_hi_):+.2f})")


class WaveAwareCalibrator(ConformalCalibrator):
    """Two-component calibration: SKU-specific noise *plus* a wave-level shock.

    Plain conformal calibration lifted mean coverage only from 0.50 to 0.58,
    because it pools all residuals into one distribution and so implicitly
    assumes each listing's error is independent. It is not. Decomposing the
    log-residuals across seven launch events gives

        between-event SD  0.689     (the whole wave over- or under-performs)
        within-event  SD  0.694     (SKU-specific)

    i.e. roughly half the forecast error is a shock common to every SKU in the
    launch — the colour wave lands well or it does not. A SKU-level interval
    that models only the within-event term is therefore too narrow by about
    sqrt(2), which is exactly the observed shortfall.

    Total predictive SD is taken as ``sqrt(within^2 + between^2)``. The two
    components are reported separately because they have different operational
    remedies: within-event spread is diversified away by ordering a *portfolio*
    of colours, whereas the between-event shock is not and must be carried as
    safety stock (or hedged by holding back part of the buy).
    """

    def fit_store(self, y, yhat, origins=None):  # type: ignore[override]
        y, yhat = np.asarray(y, float), np.asarray(yhat, float)
        r = np.log1p(np.clip(y, 0, None)) - np.log1p(np.clip(yhat, 0, None))
        ok = np.isfinite(r)
        r = r[ok]
        self._resid = r
        self.n_ = len(r)
        if self.n_ < 5:
            self.q_lo_, self.q_hi_ = -1.2, 1.2
            self.sd_within_ = self.sd_between_ = 0.9
            return self

        if origins is not None:
            o = np.asarray(origins)[ok]
            grp_means = pd.Series(r).groupby(pd.Series(o)).mean()
            grp_sds = pd.Series(r).groupby(pd.Series(o)).std()
            self.sd_between_ = float(grp_means.std(ddof=1)) if grp_means.notna().sum() > 1 else 0.0
            self.sd_within_ = float(grp_sds.mean()) if grp_sds.notna().any() else float(r.std())
            self.centre_ = float(grp_means.mean())
        else:
            self.sd_between_, self.sd_within_ = 0.0, float(r.std())
            self.centre_ = float(r.mean())

        sd_total = float(np.sqrt(self.sd_within_ ** 2 + self.sd_between_ ** 2))
        z = 1.2816  # 80% central
        self.sd_total_ = sd_total
        self.q_lo_ = self.centre_ - z * sd_total
        self.q_hi_ = self.centre_ + z * sd_total
        return self

    def quantile(self, yhat, tau: float):  # type: ignore[override]
        from scipy.stats import norm

        sd = getattr(self, "sd_total_", 0.9)
        centre = getattr(self, "centre_", 0.0)
        shift = centre + norm.ppf(tau) * sd
        return np.clip(np.expm1(np.log1p(np.clip(yhat, 0, None)) + shift), 0, None)

    def components(self) -> dict[str, float]:
        return {"sd_within": getattr(self, "sd_within_", np.nan),
                "sd_between": getattr(self, "sd_between_", np.nan),
                "sd_total": getattr(self, "sd_total_", np.nan),
                "centre": getattr(self, "centre_", np.nan),
                "n": self.n_}


def calibrators_from_rolling(rolling_detail: pd.DataFrame, model: str,
                             before_origin, wave_aware: bool = True):
    """Build a calibrator for ``model`` from origins strictly before ``before_origin``.

    Using only earlier events means the coverage guarantee is never bought with
    information from the window being scored.
    """
    d = rolling_detail[(rolling_detail.model == model)
                       & (pd.to_datetime(rolling_detail.origin) < pd.Timestamp(before_origin))]
    c = WaveAwareCalibrator() if wave_aware else ConformalCalibrator()
    if d.empty:
        return c
    if wave_aware:
        return c.fit_store(d.y_units.to_numpy(float), d.pred.to_numpy(float),
                           origins=d.origin.to_numpy())
    return c.fit_store(d.y_units.to_numpy(float), d.pred.to_numpy(float))
