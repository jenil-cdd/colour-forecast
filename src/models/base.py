"""Common interface for every forecaster.

Each model maps an ASIN-level feature frame to *cumulative units over the
target window*, plus an 80% predictive interval. Models that have no natural
notion of uncertainty inherit an empirical interval derived from their own
training residuals, so every entry on the leaderboard can be scored on interval
quality as well as point accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Prediction:
    mean: np.ndarray
    lo: np.ndarray          # 10th percentile
    hi: np.ndarray          # 90th percentile
    meta: dict = field(default_factory=dict)


class Forecaster:
    """Base class. Subclasses implement ``_fit`` and ``_predict_mean``."""

    name = "base"
    #: models that only use pre-launch information
    regime = "cold"

    def __init__(self, **kw):
        self.params = kw
        self._resid_log_sd: float | None = None
        self.fitted_ = False

    # -- to implement ------------------------------------------------------
    def _fit(self, rows: pd.DataFrame, X: pd.DataFrame, y: np.ndarray) -> None:
        raise NotImplementedError

    def _predict_mean(self, rows: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    # -- public API --------------------------------------------------------
    def fit(self, rows: pd.DataFrame, X: pd.DataFrame, y: np.ndarray) -> "Forecaster":
        self._fit(rows, X, y)
        self.fitted_ = True
        # Empirical spread of log-residuals, used for the default interval.
        try:
            insample = np.clip(self._predict_mean(rows, X), 1e-6, None)
            r = np.log1p(np.clip(y, 0, None)) - np.log1p(insample)
            r = r[np.isfinite(r)]
            self._resid_log_sd = float(np.std(r)) if len(r) > 2 else 0.6
        except Exception:
            self._resid_log_sd = 0.6
        return self

    def predict(self, rows: pd.DataFrame, X: pd.DataFrame) -> Prediction:
        mu = np.clip(self._predict_mean(rows, X), 0, None)
        return Prediction(mean=mu, **self._interval(mu))

    def _interval(self, mu: np.ndarray) -> dict[str, np.ndarray]:
        """Log-normal interval from training residual spread, widened by Poisson
        counting noise so that low-volume SKUs get honestly wide intervals."""
        sd = self._resid_log_sd if self._resid_log_sd else 0.6
        # Poisson CV = 1/sqrt(mu): dominates the residual term for tiny SKUs.
        cv = 1.0 / np.sqrt(np.clip(mu, 1.0, None))
        tot = np.sqrt(sd**2 + cv**2)
        z = 1.2816  # 80% central interval
        return {
            "lo": np.clip(np.expm1(np.log1p(mu) - z * tot), 0, None),
            "hi": np.clip(np.expm1(np.log1p(mu) + z * tot), 0, None),
        }
