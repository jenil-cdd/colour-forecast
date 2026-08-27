"""Benchmarks: the incumbent heuristic, plus regularised linear velocity models.

The heuristic is included as a *model* rather than prose so that the 2,180-unit
recommendation is scored on the same held-out window as everything else. If the
ML suite cannot beat it, that is a finding worth reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

from src.models.base import Forecaster


class HeuristicOrderModel(Forecaster):
    """Reproduces the incumbent spreadsheet logic, faithfully.

    Mature non-white organic Queen velocity measured on clean days, scaled by a
    fixed size ratio (King 0.76, Twin 0.60) and a fixed month-1 ramp factor
    (0.45). Family depth and colour identity are ignored entirely — every new
    variant of a given size gets the same number, which is exactly the weakness
    the ML suite targets.

    The anchor is deliberately taken from *mature* listings in the daily panel
    rather than from the training labels. The labels are themselves first-61-day
    windows and so already contain the ramp; anchoring on them and then applying
    the 0.45 factor again would discount twice and understate the incumbent.
    """

    name = "heuristic"

    def __init__(self, panel: pd.DataFrame | None = None, ramp: float = 0.45,
                 king: float = 0.76, twin: float = 0.60, mature_days: int = 180,
                 lookback: int = 90, **kw):
        super().__init__(**kw)
        self.panel = panel
        self.ramp = ramp
        self.mature_days, self.lookback = mature_days, lookback
        self.ratios = {"Queen": 1.0, "King": king, "Twin": twin, "Oversized King": 0.25}
        self.queen_velocity_ = 1.0

    def _fit(self, rows, X, y) -> None:
        asof = pd.Timestamp(rows["asof"].max())
        if self.panel is not None:
            p = self.panel
            m = (
                (p.date <= asof)
                & (p.date > asof - pd.Timedelta(days=self.lookback))
                & p.is_organic_day
                & (p.days_since_launch >= self.mature_days)
                & p["size"].eq("Queen")
                & ~p.is_core_white.astype(bool)
            )
            sub = p.loc[m, "units_ordered"]
            if len(sub) >= 30:
                self.queen_velocity_ = float(sub.mean())
                return
        # Fallback: de-ramp the training labels so the 0.45 factor is applied once.
        m = rows["size"].eq("Queen") & ~rows["is_core_white"].astype(bool)
        days = rows.loc[m, "target_days"].clip(lower=1)
        v = (rows.loc[m, "y_units"] / days).replace([np.inf, -np.inf], np.nan).dropna()
        self.queen_velocity_ = float(v.mean() / self.ramp) if len(v) else 1.0

    def _predict_mean(self, rows, X) -> np.ndarray:
        ratio = rows["size"].map(self.ratios).fillna(0.5).to_numpy(float)
        days = rows["target_days"].to_numpy(float)
        return self.queen_velocity_ * ratio * self.ramp * days


class NaiveFamilyMean(Forecaster):
    """Velocity of the shade family the variant enters, at the same size.

    Falls back up the hierarchy (family+size -> family -> program) when a level
    has no history. This is the honest "just use the siblings" baseline and it is
    surprisingly hard to beat.
    """

    name = "naive_family"

    def _fit(self, rows, X, y) -> None:
        self.global_ = float(np.mean(y / rows["target_days"].clip(lower=1))) if len(y) else 1.0

    def _predict_mean(self, rows, X) -> np.ndarray:
        v = (rows["sibling_velocity_same_size"]
             .fillna(rows["sibling_velocity"])
             .fillna(rows["family_pool_velocity"])
             .fillna(rows["program_velocity"])
             .fillna(self.global_))
        return v.to_numpy(float) * rows["target_days"].to_numpy(float)


class SizeRatioBaseline(Forecaster):
    """Queen anchor times an *empirically fitted* size ratio.

    Same shape as the heuristic but the ratios are estimated from the training
    data instead of hard-coded, which isolates how much of the heuristic's error
    comes from its size assumptions specifically.
    """

    name = "size_ratio"

    def _fit(self, rows, X, y) -> None:
        d = rows.assign(vel=y / rows["target_days"].clip(lower=1))
        anchor = d[d["size"] == "Queen"].vel.median()
        self.anchor_ = float(anchor) if np.isfinite(anchor) and anchor > 0 else 1.0
        self.ratios_ = {}
        for sz, g in d.groupby("size"):
            r = g.vel.median() / self.anchor_
            self.ratios_[sz] = float(r) if np.isfinite(r) else 0.5

    def _predict_mean(self, rows, X) -> np.ndarray:
        # Prefer the family's own observed velocity where it exists, otherwise
        # fall back to the fitted global anchor.
        base = rows["sibling_velocity"].fillna(self.anchor_).to_numpy(float)
        ratio = rows["size"].map(self.ratios_).fillna(0.5).to_numpy(float)
        return base * ratio * rows["target_days"].to_numpy(float)


class _LinearVelocity(Forecaster):
    """Shared plumbing: regress log1p(units) on the design matrix.

    Modelling in log space keeps predictions positive and makes the
    multiplicative structure of the problem (family effect x size effect x ramp)
    additive, which is what a linear model can actually represent.
    """

    estimator_cls = Ridge
    default_kw: dict = {"alpha": 1.0}

    def _fit(self, rows, X, y) -> None:
        self.cols_ = list(X.columns)
        self.scaler_ = StandardScaler().fit(X.to_numpy(float))
        kw = {**self.default_kw, **{k: v for k, v in self.params.items() if k != "seed"}}
        self.model_ = self.estimator_cls(**kw).fit(
            self.scaler_.transform(X.to_numpy(float)), np.log1p(np.clip(y, 0, None))
        )

    def _align(self, X: pd.DataFrame) -> np.ndarray:
        Z = X.reindex(columns=self.cols_, fill_value=0.0).to_numpy(float)
        return self.scaler_.transform(Z)

    def _predict_mean(self, rows, X) -> np.ndarray:
        return np.expm1(self.model_.predict(self._align(X)))


class RidgeVelocity(_LinearVelocity):
    name = "ridge"
    estimator_cls = Ridge
    default_kw = {"alpha": 3.0}


class LassoVelocity(_LinearVelocity):
    name = "lasso"
    estimator_cls = Lasso
    default_kw = {"alpha": 0.05, "max_iter": 20000}


class ElasticNetVelocity(_LinearVelocity):
    name = "elasticnet"
    estimator_cls = ElasticNet
    default_kw = {"alpha": 0.05, "l1_ratio": 0.5, "max_iter": 20000}
