"""SKU-level predictive intervals calibrated on focal-programme residuals.

The problem this replaces
------------------------
Model-derived bands were unusable for ordering. Silver Queen's July band came
back as **8 to 173 units** against an actual of 13. A band that wide is
technically well-covered and operationally worthless: it does not constrain a
factory order at all.

Two causes. First, ``hier_bayes`` propagates parameter uncertainty *and* count
noise, which is correct but generous at these volumes. Second, and larger, the
residual spread it was calibrated against pooled both programmes, so it carried
400 TC's volatility into DCS forecasts.

The approach
------------
Split-conformal on the log scale, fitted on **out-of-fold residuals from the
focal programme only**:

    r = log1p(actual) - log1p(predicted)
    lo = expm1(log1p(yhat) + quantile(r, (1-nominal)/2))
    hi = expm1(log1p(yhat) + quantile(r, (1+nominal)/2))

This is the narrowest band that achieves nominal coverage *on the data it was
fitted to*. It is not a free lunch: a tighter band on the same forecast means
lower coverage whenever the residual distribution shifts. Achieved coverage is
therefore always reported next to the width, and ``floor_ratio``/``cap_ratio``
stop the band collapsing to something falsely precise on a near-zero forecast.

Ratio bounds rather than additive ones, because demand here is multiplicative:
a SKU forecast at 100 units needs a wider absolute band than one at 10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ConformalInterval:
    """Log-scale conformal band, optionally fitted per group (e.g. per size)."""

    def __init__(self, nominal: float = 0.80, floor_ratio: float = 0.35,
                 cap_ratio: float = 3.0, min_obs: int = 8):
        self.nominal = nominal
        self.floor_ratio, self.cap_ratio = floor_ratio, cap_ratio
        self.min_obs = min_obs
        self.q_lo_ = self.q_hi_ = 0.0
        self.n_ = 0
        self.by_group_: dict[str, tuple[float, float]] = {}

    def fit(self, actual: np.ndarray, predicted: np.ndarray,
            groups: np.ndarray | None = None) -> "ConformalInterval":
        a, p = np.asarray(actual, float), np.asarray(predicted, float)
        r = np.log1p(np.clip(a, 0, None)) - np.log1p(np.clip(p, 0, None))
        ok = np.isfinite(r)
        r = r[ok]
        self.n_ = len(r)
        lo_t, hi_t = (1 - self.nominal) / 2, (1 + self.nominal) / 2
        if self.n_ >= self.min_obs:
            self.q_lo_, self.q_hi_ = float(np.quantile(r, lo_t)), float(np.quantile(r, hi_t))
        else:
            # Too little history to calibrate; stay wide rather than pretend.
            self.q_lo_, self.q_hi_ = -1.0, 1.0

        if groups is not None:
            g = np.asarray(groups)[ok]
            for key in pd.unique(g):
                sub = r[g == key]
                if len(sub) >= self.min_obs:
                    self.by_group_[str(key)] = (float(np.quantile(sub, lo_t)),
                                                float(np.quantile(sub, hi_t)))
        return self

    def apply(self, yhat: np.ndarray,
              groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        y = np.clip(np.asarray(yhat, float), 0, None)
        base = np.log1p(y)
        q_lo = np.full(len(y), self.q_lo_)
        q_hi = np.full(len(y), self.q_hi_)
        if groups is not None and self.by_group_:
            g = np.asarray(groups).astype(str)
            for key, (a, b) in self.by_group_.items():
                m = g == key
                q_lo[m], q_hi[m] = a, b
        lo = np.clip(np.expm1(base + q_lo), 0, None)
        hi = np.clip(np.expm1(base + q_hi), 0, None)
        # Ratio guards: keep the band from collapsing or exploding.
        lo = np.maximum(lo, y * self.floor_ratio)
        # The cap scales with the nominal level. A fixed cap silently collapsed
        # p80 and p90 onto the same number when both exceeded it, which made the
        # two order quantities identical and the choice between them meaningless.
        cap = self.cap_ratio * (1.0 + 2.0 * max(self.nominal - 0.80, 0.0))
        hi = np.minimum(np.maximum(hi, y * 1.05), y * cap)
        return np.minimum(lo, y), np.maximum(hi, y)

    def width_summary(self, yhat: np.ndarray, groups=None) -> dict[str, float]:
        lo, hi = self.apply(yhat, groups)
        y = np.clip(np.asarray(yhat, float), 1e-9, None)
        return {"median_lo_ratio": float(np.median(lo / y)),
                "median_hi_ratio": float(np.median(hi / y)),
                "median_width_units": float(np.median(hi - lo)),
                "n_calibration": self.n_,
                "n_groups": len(self.by_group_)}

    @property
    def effective_cap(self) -> float:
        return self.cap_ratio * (1.0 + 2.0 * max(self.nominal - 0.80, 0.0))

    def __repr__(self) -> str:
        raw_hi = float(np.expm1(self.q_hi_) + 1)
        binding = " CAP-BOUND" if raw_hi > self.effective_cap else ""
        return (f"ConformalInterval(p{int(self.nominal * 100)}, n={self.n_}, "
                f"lo=x{np.expm1(self.q_lo_) + 1:.2f}, hi=x{raw_hi:.2f}, "
                f"cap=x{self.effective_cap:.2f}{binding}, groups={len(self.by_group_)})")
