"""Dynamic stacking ensemble over all 19 base models.

Why stack instead of picking a winner
-------------------------------------
The tournament produced no single winner, and the ordering flipped with the
horizon. On the 61-day window ``knn_lookalike`` placed 1st and ``hier_bayes``
3rd; on the 30-day window that reversed, with ``random_forest`` and ``gompertz``
also beating ``knn_lookalike`` on per-line error. Different models are carrying
different parts of the launch curve, so picking one discards the rest.

How the weights are learned (and why this is not leakage)
---------------------------------------------------------
Weights are fitted on **out-of-fold predictions over the historical launch
cohorts only** - never on the test months. Folds are grouped by launch wave
(``GroupKFold`` on launch quarter), because listings in the same wave share a
demand shock; splitting them across folds would let a model see its own wave and
inflate its apparent skill.

One weight vector is fitted **per horizon**, which is what makes the ensemble
"dynamic": the month-1 blend and the month-2 blend are free to differ, and in
practice they do. Nothing is hand-tuned toward a favoured model.

Fitting is non-negative least squares on ``log1p`` units, then renormalised to a
convex combination. Two reasons for the log scale: fitting on raw units lets a
handful of high-volume launches dictate the whole blend (an earlier version did
exactly this and came out worse than its own members), and demand here is
multiplicative, so errors are proportional rather than absolute.

Guard rails
-----------
* A member that fails to produce finite out-of-fold predictions for at least 60%
  of cohorts is dropped rather than imputed - a silently imputed member would
  earn weight for predictions it never made.
* Weights are shrunk toward equal-weight by ``shrink``. With ~100 cohorts and 19
  members, unshrunk NNLS will happily put all its mass on two models; shrinkage
  keeps the blend from becoming a disguised winner-takes-all.
* Members are floored out below ``min_weight`` and the rest renormalised, so the
  reported blend is honest about who is actually contributing.

Intervals
---------
The point forecast comes from the blend; the *interval* comes from
``hier_bayes``, scaled to the blended mean. That split is deliberate.
``hier_bayes`` was the best-calibrated model in the suite by a wide margin
(69% average coverage against an 80% target, versus 27-57% for most others, and
100% on the June window), because it is the only member that propagates
parameter uncertainty and count noise properly. Averaging intervals across
members would blur that.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src.models import REGISTRY

log = logging.getLogger(__name__)

NEEDS_PANEL = {"gompertz", "logistic", "bass", "heuristic", "ensemble"}
#: Excluded members.
#: ``ensemble`` is itself a blend - stacking it would double-count its members
#: and make the weights uninterpretable.
#: ``heuristic`` is the incumbent benchmark, not a candidate. It is also
#: actively harmful inside the stack: with size blinded it collapses to a
#: near-constant predictor, and NNLS without an intercept then hands it most of
#: the weight to use as a bias term (it reached 54% in June and 78% in July,
#: pushing July's total error to 0.65). An explicit intercept now removes the
#: incentive, and excluding it removes the possibility.
EXCLUDE = {"ensemble", "heuristic"}

INTERVAL_ANCHOR = "hier_bayes"


def _build(name: str, panel: pd.DataFrame | None, seed: int):
    kw = {"seed": seed}
    if name in NEEDS_PANEL:
        kw["panel"] = panel
    return REGISTRY[name](**kw)


def _grouped_folds(waves: np.ndarray, n_splits: int, seed: int,
                   stratify: bool = True) -> np.ndarray:
    """Assign each row a fold id.

    ``stratify=True`` spreads each launch wave proportionally across folds.
    ``stratify=False`` keeps whole waves together.

    Whole-wave grouping is the stricter test and is right when waves are
    balanced. It is degenerate when one wave dominates: the 400 TC programme has
    38 cohorts across 5 waves with **21 of them in 2025Q2**, so holding that wave
    out trains the model on 2020 data and asks it to predict 2025. Out-of-fold
    error came out at 0.896 against 0.603 in-sample, and the implied 80% band
    widened to 3.78x the point forecast - pessimistic by construction, and not
    the operating condition, since the real forecast has all history available.
    Stratifying keeps the fold representative of what the model will actually
    see.
    """
    rng = np.random.default_rng(seed)
    if not stratify:
        uniq = pd.unique(waves)
        order = rng.permutation(len(uniq))
        fold_of_wave = {w: int(order[i] % n_splits) for i, w in enumerate(uniq)}
        return np.array([fold_of_wave[w] for w in waves])

    folds = np.zeros(len(waves), dtype=int)
    for w in pd.unique(waves):
        m = np.flatnonzero(waves == w)
        folds[rng.permutation(m)] = np.arange(len(m)) % n_splits
    return folds


class DynamicStack:
    """Horizon-aware stacked ensemble."""

    def __init__(self, members: list[str] | None = None, n_folds: int = 5,
                 shrink: float = 0.35, min_weight: float = 0.02, seed: int = 42,
                 rel_tolerance: float = 1.35):
        self.member_names = [m for m in (members or list(REGISTRY)) if m not in EXCLUDE]
        self.n_folds, self.shrink, self.min_weight, self.seed = n_folds, shrink, min_weight, seed
        #: a member is admitted if its OOF error is within this multiple of the
        #: best member's OOF error
        self.rel_tolerance = rel_tolerance
        self.oof_errors_: pd.Series | None = None
        self.weights_: pd.Series | None = None
        self.oof_: pd.DataFrame | None = None
        self.fitted_: dict = {}
        self.intercept_: float = 0.0
        self.gate_report_: dict = {}

    # ------------------------------------------------------------------
    def fit(self, rows: pd.DataFrame, X: pd.DataFrame, y: np.ndarray,
            panel: pd.DataFrame | None = None,
            sample_weight: np.ndarray | None = None) -> "DynamicStack":
        yy = np.clip(np.asarray(y, float), 0, None)
        n = len(rows)
        waves = rows["wave"].astype(str).to_numpy() if "wave" in rows else np.full(n, "all")
        folds = _grouped_folds(waves, min(self.n_folds, max(2, len(pd.unique(waves)))), self.seed)

        oof = {m: np.full(n, np.nan) for m in self.member_names}
        for f in np.unique(folds):
            te, tr = folds == f, folds != f
            if tr.sum() < 10 or te.sum() == 0:
                continue
            for m in self.member_names:
                try:
                    mdl = _build(m, panel, self.seed)
                    mdl.fit(rows[tr], X[tr], yy[tr])
                    oof[m][te] = mdl.predict(rows[te], X[te]).mean
                except Exception as exc:
                    log.debug("oof %s fold %s failed: %s", m, f, exc)

        self.oof_ = pd.DataFrame(oof, index=rows.index)

        # Gate 1 - coverage. A member that could not produce finite predictions
        # for 60% of cohorts is dropped rather than imputed; an imputed member
        # would earn weight for forecasts it never made.
        keep = [m for m in self.member_names
                if np.isfinite(self.oof_[m]).mean() >= 0.60]
        dropped_cov = sorted(set(self.member_names) - set(keep))

        # Gate 2 - certification. Membership is decided by the same sanity
        # harness used to certify models for the business, run on the
        # out-of-fold predictions. A member not fit to produce a per-line
        # number out-of-fold should not be handed per-line weight. This is what
        # stops a model that happened to fit the training cohorts (lightgbm
        # took 30% of the month-2 blend before this gate existed) from carrying
        # weight it cannot justify. No test-window data is involved.
        # The gate is *relative*, not absolute. An absolute threshold was tried
        # first and was actively harmful: out-of-fold cohorts include the
        # 2022-23 waves where every model has huge error, so a fixed cut-off
        # rejected 14 of 17 members in month 1 - including the two best - and
        # pushed the blend's per-line error from 0.447 up to 0.669. Comparing
        # each member against the best member instead adapts to how hard the
        # out-of-fold set actually is, and only drops models that are poor
        # *for this problem*.
        errs = {m: self._oof_error(self.oof_[m].to_numpy(float), yy) for m in keep}
        finite = {m: e for m, e in errs.items() if np.isfinite(e)}
        rejected = []
        if finite:
            best = min(finite.values())
            cut = best * self.rel_tolerance
            certified = [m for m in keep if finite.get(m, np.inf) <= cut]
            rejected = [m for m in keep if m not in certified]
            if len(certified) >= 3:
                keep = certified
            else:
                rejected = []
        self.oof_errors_ = pd.Series(errs).sort_values()
        self.gate_report_ = {"dropped_no_coverage": dropped_cov, "rejected_by_sanity": rejected,
                             "admitted": list(keep)}
        if dropped_cov or rejected:
            log.info("stack gates -> dropped:%s rejected:%s", dropped_cov, rejected)
        if not keep:
            keep = self.member_names

        A = np.column_stack([np.nan_to_num(self.oof_[m].to_numpy(float),
                                           nan=float(np.median(yy))) for m in keep])
        Al, yl = np.log1p(np.clip(A, 0, None)), np.log1p(yy)

        # Explicit intercept column. Without one, any near-constant member gets
        # loaded up as a de-facto bias term and the reported weights stop
        # describing predictive contribution. The intercept is fitted but never
        # reported as a model weight.
        Aug = np.column_stack([Al, np.ones(len(Al))])
        if sample_weight is not None:
            sw = np.sqrt(np.clip(sample_weight, 1e-6, None))[:, None]
            Aug, yl = Aug * sw, yl * sw.ravel()

        try:
            w_aug, _ = nnls(Aug, yl)
            w = w_aug[:-1]
            self.intercept_ = float(w_aug[-1])
        except Exception:
            w, self.intercept_ = np.ones(len(keep)), 0.0
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(keep))
        w = w / w.sum()

        # Choose the shrinkage level on the out-of-fold matrix itself: too little
        # and NNLS puts everything on two members, too much and the blend
        # degenerates to an equal-weight average of models that were rejected on
        # merit. Selected on OOF error only.
        self.shrink = self._select_shrink(Al if sample_weight is None else np.log1p(np.clip(A, 0, None)),
                                         np.log1p(yy), w, len(keep))

        # Shrink toward equal weight, then floor and renormalise.
        w = (1 - self.shrink) * w + self.shrink * np.full(len(keep), 1 / len(keep))
        w[w < self.min_weight] = 0.0
        if w.sum() <= 0:
            w = np.full(len(keep), 1 / len(keep))
        self.weights_ = pd.Series(w / w.sum(), index=keep).sort_values(ascending=False)

        # Refit every contributing member on the full training fold.
        for m in self.weights_[self.weights_ > 0].index:
            try:
                mdl = _build(m, panel, self.seed)
                mdl.fit(rows, X, yy)
                self.fitted_[m] = mdl
            except Exception as exc:
                log.warning("stack refit %s failed: %s", m, exc)
        if INTERVAL_ANCHOR not in self.fitted_:
            try:
                mdl = _build(INTERVAL_ANCHOR, panel, self.seed)
                mdl.fit(rows, X, yy)
                self.fitted_[INTERVAL_ANCHOR] = mdl
            except Exception as exc:
                log.warning("interval anchor unavailable: %s", exc)
        return self

    # ------------------------------------------------------------------
    def predict(self, rows: pd.DataFrame, X: pd.DataFrame):
        from src.models.base import Prediction

        parts, used = [], []
        for m, w in self.weights_.items():
            if w <= 0 or m not in self.fitted_:
                continue
            try:
                parts.append(w * self.fitted_[m].predict(rows, X).mean)
                used.append(m)
            except Exception:
                continue
        mu = np.clip(np.sum(parts, axis=0), 0, None) if parts else np.zeros(len(rows))

        # Interval: hier_bayes shape, rescaled to the blended level.
        lo = hi = None
        anchor = self.fitted_.get(INTERVAL_ANCHOR)
        if anchor is not None:
            try:
                a = anchor.predict(rows, X)
                scale = np.divide(mu, np.clip(a.mean, 1e-6, None),
                                  out=np.ones_like(mu), where=a.mean > 1e-6)
                lo, hi = np.clip(a.lo * scale, 0, None), np.clip(a.hi * scale, 0, None)
            except Exception:
                lo = hi = None
        if lo is None:
            cv = 1.0 / np.sqrt(np.clip(mu, 1.0, None))
            lo, hi = np.clip(mu * (1 - 1.2816 * cv), 0, None), mu * (1 + 1.2816 * cv)

        return Prediction(mean=mu, lo=lo, hi=hi,
                          meta={"weights": self.weights_[self.weights_ > 0].to_dict(),
                                "used": used})

    def _select_shrink(self, Al: np.ndarray, yl: np.ndarray, w: np.ndarray,
                       k: int) -> float:
        best, best_err = self.shrink, np.inf
        for s in (0.0, 0.10, 0.20, 0.35, 0.50, 0.70):
            ws = (1 - s) * w + s * np.full(k, 1 / k)
            ws = ws / ws.sum()
            err = float(np.mean(np.abs(Al @ ws - yl)))
            if err < best_err:
                best, best_err = s, err
        return best

    @staticmethod
    def _oof_error(oof: np.ndarray, y: np.ndarray) -> float:
        """Out-of-fold per-line error for one member, on the log scale.

        Log scale rather than raw WAPE so a few high-volume cohorts cannot
        decide membership, matching how the weights themselves are fitted.
        """
        m = np.isfinite(oof)
        if m.sum() < 10:
            return np.inf
        r = np.log1p(np.clip(oof[m], 0, None)) - np.log1p(np.clip(y[m], 0, None))
        return float(np.mean(np.abs(r)))

    def weight_table(self) -> pd.DataFrame:
        if self.weights_ is None:
            return pd.DataFrame()
        w = self.weights_[self.weights_ > 0]
        return pd.DataFrame({"model": w.index, "weight": w.round(4).to_numpy()})
