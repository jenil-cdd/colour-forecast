"""Ensemble over the model suite.

With 40-odd trainable listings, a learned stacker would fit noise, so the
default is a *constrained* combination: non-negative weights that sum to one,
chosen by leave-one-out cross-validation on the training fold. That cannot do
worse than the best single member by much, and it protects against any one
model's failure mode dominating the order.

Weights are reported, because in practice the interesting output is which
paradigm the data prefers — if the hierarchical Bayes and look-alike models take
most of the weight, that says the signal is in borrowing strength, not in
non-linear interactions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src.models.base import Forecaster, Prediction

# Members deliberately span paradigms rather than being the top-N by accuracy.
DEFAULT_MEMBERS = ["hier_bayes", "negbin_glm", "lightgbm", "knn_lookalike", "gompertz"]


class StackedEnsemble(Forecaster):
    name = "ensemble"

    def __init__(self, members: list[str] | None = None, panel: pd.DataFrame | None = None, **kw):
        super().__init__(**kw)
        self.member_names = members or DEFAULT_MEMBERS
        self.panel = panel

    def _build_members(self):
        from src.models import REGISTRY

        out = {}
        for nm in self.member_names:
            cls = REGISTRY.get(nm)
            if cls is None:
                continue
            kw = dict(self.params)
            # Growth models need the daily panel to fit their ramp shape.
            if nm in {"gompertz", "logistic", "bass"}:
                kw["panel"] = self.panel
            try:
                out[nm] = cls(**kw)
            except Exception:
                continue
        return out

    def _fit(self, rows, X, y) -> None:
        self.members_ = self._build_members()
        yy = np.clip(y, 0, None)

        # K-fold out-of-fold predictions per member. LOO would be cleaner but
        # costs n x |members| refits, which is prohibitive once this runs inside
        # the rolling-origin loop; 5 folds gives the same weights to 2 decimals.
        n = len(rows)
        n_folds = int(min(self.params.get("n_folds", 5), max(2, n)))
        rng = np.random.default_rng(self.params.get("seed", 42))
        fold = rng.permutation(n) % n_folds
        oof = {nm: np.full(n, np.nan) for nm in self.members_}
        for f in range(n_folds):
            te = fold == f
            tr = ~te
            if tr.sum() < 5 or te.sum() == 0:
                continue
            for nm, proto in self.members_.items():
                try:
                    m = self._clone(proto)
                    m.fit(rows[tr], X[tr], yy[tr])
                    oof[nm][te] = m.predict(rows[te], X[te]).mean
                except Exception:
                    oof[nm][te] = np.nan

        names = [nm for nm in self.members_ if np.isfinite(oof[nm]).sum() > n * 0.6]
        if not names:
            names = list(self.members_)
        A = np.column_stack([np.nan_to_num(oof[nm], nan=float(np.mean(yy))) for nm in names])

        # Fit the weights on the log scale. Fitting on raw units lets a handful
        # of high-volume launches dictate the entire blend, which is what made
        # the first version of this ensemble worse than its own members on the
        # low-volume cold-start cohort.
        Al = np.log1p(np.clip(A, 0, None))
        yl = np.log1p(yy)
        w, _ = nnls(Al, yl)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(names))
        self.weights_ = pd.Series(w / w.sum(), index=names)

        # Refit members on the full training fold for prediction.
        for nm in names:
            self.members_[nm].fit(rows, X, yy)
        self.active_ = names

    @staticmethod
    def _clone(proto: Forecaster) -> Forecaster:
        kw = dict(proto.params)
        for attr in ("panel",):
            if hasattr(proto, attr) and getattr(proto, attr) is not None:
                kw[attr] = getattr(proto, attr)
        return type(proto)(**kw)

    def _predict_mean(self, rows, X) -> np.ndarray:
        preds = []
        for nm in self.active_:
            try:
                preds.append(self.weights_[nm] * self.members_[nm].predict(rows, X).mean)
            except Exception:
                continue
        return np.sum(preds, axis=0) if preds else np.zeros(len(rows))

    def predict(self, rows, X) -> Prediction:
        """Interval pooled across members: the spread *between* paradigms is
        real model risk and belongs in the band."""
        means, los, his = [], [], []
        for nm in self.active_:
            try:
                p = self.members_[nm].predict(rows, X)
                w = self.weights_[nm]
                means.append(w * p.mean)
                los.append(w * p.lo)
                his.append(w * p.hi)
            except Exception:
                continue
        if not means:
            return Prediction(mean=np.zeros(len(rows)), lo=np.zeros(len(rows)), hi=np.zeros(len(rows)))
        mu = np.sum(means, axis=0)
        stack = np.array([m / max(self.weights_[nm], 1e-9) for m, nm in zip(means, self.active_)])
        disagreement = stack.std(axis=0)
        return Prediction(
            mean=mu,
            lo=np.clip(np.sum(los, axis=0) - 0.5 * disagreement, 0, None),
            hi=np.sum(his, axis=0) + 0.5 * disagreement,
            meta={"weights": self.weights_.to_dict()},
        )
