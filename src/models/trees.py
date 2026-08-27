"""Tree ensembles for the interaction structure.

The mechanic the linear models cannot express is *interaction*: the penalty for
being the 4th grey listing is not the same as the penalty for being the 4th
listing in an unentered family, and family depth matters more for Queen than for
Oversized King. Trees pick that up without being told the functional form.

All three fit on log1p(units) with the window length carried as a feature, and
report gain-based importances so the learned mechanics can be audited against
the domain assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.base import Forecaster


class _TreeBase(Forecaster):
    def _fit(self, rows, X, y) -> None:
        self.cols_ = list(X.columns)
        self.model_ = self._make()
        self.model_.fit(X.to_numpy(float), np.log1p(np.clip(y, 0, None)))

    def _predict_mean(self, rows, X) -> np.ndarray:
        Z = X.reindex(columns=self.cols_, fill_value=0.0).to_numpy(float)
        return np.expm1(self.model_.predict(Z))

    def importances(self) -> pd.Series:
        imp = getattr(self.model_, "feature_importances_", None)
        if imp is None:
            return pd.Series(dtype=float)
        return pd.Series(imp, index=self.cols_).sort_values(ascending=False)


class RandomForestForecaster(_TreeBase):
    name = "random_forest"

    def _make(self):
        return RandomForestRegressor(
            n_estimators=600, min_samples_leaf=2, max_features=0.5,
            random_state=self.params.get("seed", 42), n_jobs=-1,
        )

    def predict(self, rows, X):
        """Interval from the spread across trees — a real ensemble interval
        rather than a residual-based approximation."""
        Z = X.reindex(columns=self.cols_, fill_value=0.0).to_numpy(float)
        per_tree = np.array([t.predict(Z) for t in self.model_.estimators_])
        mu = np.clip(np.expm1(per_tree.mean(0)), 0, None)
        lo = np.clip(np.expm1(np.quantile(per_tree, 0.10, axis=0)), 0, None)
        hi = np.clip(np.expm1(np.quantile(per_tree, 0.90, axis=0)), 0, None)
        # Widen by Poisson noise: tree spread captures model uncertainty only.
        cv = 1.0 / np.sqrt(np.clip(mu, 1.0, None))
        from src.models.base import Prediction
        return Prediction(mean=mu,
                          lo=np.clip(lo * (1 - cv), 0, None),
                          hi=hi * (1 + cv))


class _QuantileTreeBase(_TreeBase):
    """Adds genuine quantile regressors alongside the mean model.

    The base class interval is derived from pooled log-residual spread, which is
    the same width for every listing. Boosted quantile models let the band vary
    with the features, so a crowded family gets a wider band than a clean one.
    """

    def _fit(self, rows, X, y) -> None:
        super()._fit(rows, X, y)
        yl = np.log1p(np.clip(y, 0, None))
        self.q_models_ = {}
        for tau in (0.10, 0.90):
            try:
                m = self._make_quantile(tau)
                m.fit(X.to_numpy(float), yl)
                self.q_models_[tau] = m
            except Exception:
                pass

    def predict(self, rows, X):
        from src.models.base import Prediction

        Z = X.reindex(columns=self.cols_, fill_value=0.0).to_numpy(float)
        mu = np.clip(np.expm1(self.model_.predict(Z)), 0, None)
        if len(self.q_models_) < 2:
            return Prediction(mean=mu, **self._interval(mu))
        lo = np.clip(np.expm1(self.q_models_[0.10].predict(Z)), 0, None)
        hi = np.clip(np.expm1(self.q_models_[0.90].predict(Z)), 0, None)
        # Widen by Poisson counting noise: quantile trees model conditional
        # spread of the fitted surface, not the integer arrival process on top.
        cv = 1.0 / np.sqrt(np.clip(mu, 1.0, None))
        lo, hi = np.minimum(lo, mu) * (1 - cv), np.maximum(hi, mu) * (1 + cv)
        return Prediction(mean=mu, lo=np.clip(lo, 0, None), hi=hi)


class XGBoostForecaster(_QuantileTreeBase):
    name = "xgboost"

    def _make_quantile(self, tau: float):
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            random_state=self.params.get("seed", 42), n_jobs=-1,
            objective="reg:quantileerror", quantile_alpha=tau,
        )

    def _make(self):
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, min_child_weight=3,
            random_state=self.params.get("seed", 42), n_jobs=-1,
            objective="reg:squarederror",
        )


class LightGBMForecaster(_QuantileTreeBase):
    name = "lightgbm"

    def _make_quantile(self, tau: float):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=7,
            min_child_samples=5, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=2.0,
            random_state=self.params.get("seed", 42), n_jobs=-1, verbose=-1,
            objective="quantile", alpha=tau,
        )

    def _make(self):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=7,
            min_child_samples=5, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=2.0,
            random_state=self.params.get("seed", 42), n_jobs=-1, verbose=-1,
        )
