"""Discrete count GLMs for unit/day velocity.

Units per SKU-day here run 0-4, so the demand process is genuinely a count
process, not a smooth continuous one. Two consequences:

* The variance is tied to the mean. Poisson assumes variance = mean; the data
  are over-dispersed (bursty demand), so Negative Binomial is the better fit and
  the dispersion parameter it estimates is itself useful — it is what sets the
  safety stock.
* The window length must enter as an exposure *offset*, not a feature, so that
  a 61-day and a 120-day window are modelled on the same per-day rate scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.models.base import Forecaster


class _CountGLM(Forecaster):
    family_name = "poisson"

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        Z = X.reindex(columns=self.cols_, fill_value=0.0).to_numpy(float)
        Z = (Z - self.mu_) / self.sd_
        return np.column_stack([np.ones(len(Z)), Z])

    #: L2 penalty. With ~100 launch cohorts and ~60 one-hot columns an
    #: unpenalised exponential-link GLM overfits catastrophically — it produced
    #: 2-4x over-forecasts on the held-out window before this was added.
    l2_alpha = 0.05

    def _fit(self, rows, X, y) -> None:
        # Drop near-constant columns: a GLM with a singular design will not
        # converge, and one-hot blocks are frequently degenerate on small folds.
        keep = [c for c in X.columns if X[c].std() > 1e-8]
        # Cap dimensionality relative to sample size; keep the columns with the
        # strongest marginal association with the (log) target.
        max_cols = max(4, len(y) // 8)
        if len(keep) > max_cols:
            yl = np.log1p(np.clip(y, 0, None))
            score = {}
            for c in keep:
                v = X[c].to_numpy(float)
                sd = v.std()
                score[c] = abs(np.corrcoef(v, yl)[0, 1]) if sd > 1e-8 else 0.0
            keep = sorted(keep, key=lambda c: -(score[c] if np.isfinite(score[c]) else 0.0))[:max_cols]
        self.cols_ = keep
        Z = X[keep].to_numpy(float)
        self.mu_, self.sd_ = Z.mean(0), np.where(Z.std(0) > 1e-8, Z.std(0), 1.0)
        D = self._design(X[keep])

        exposure = rows["target_days"].clip(lower=1).to_numpy(float)
        yy = np.clip(y, 0, None)

        if self.family_name == "poisson":
            fam = sm.families.Poisson()
        else:
            # Estimate dispersion from a first-pass Poisson fit, then refit as
            # NB2 with that alpha. This is the standard two-step approach and
            # avoids the optimiser wandering when alpha is free.
            p0 = sm.GLM(yy, D, family=sm.families.Poisson(), offset=np.log(exposure))
            pen0 = np.full(D.shape[1], self.l2_alpha)
            pen0[0] = 0.0
            try:
                r0 = p0.fit_regularized(alpha=pen0, L1_wt=0.0, maxiter=500)
                mu0 = np.clip(np.exp(D @ np.asarray(r0.params, float) + np.log(exposure)), 1e-6, None)
            except Exception:
                r0 = p0.fit(maxiter=200)
                mu0 = np.clip(r0.fittedvalues, 1e-6, None)
            dof = max(len(yy) - D.shape[1], 1)
            chi2 = float(np.sum((yy - mu0) ** 2 / mu0) / dof)
            alpha = max((chi2 - 1.0) / np.mean(mu0), 1e-3)
            self.alpha_ = float(min(alpha, 10.0))
            fam = sm.families.NegativeBinomial(alpha=self.alpha_)

        glm = sm.GLM(yy, D, family=fam, offset=np.log(exposure))
        try:
            # Elastic-net with L1_wt=0 gives a pure ridge penalty. The intercept
            # is left unpenalised so the overall level stays free.
            pen = np.full(D.shape[1], self.l2_alpha)
            pen[0] = 0.0
            self.res_ = glm.fit_regularized(alpha=pen, L1_wt=0.0, maxiter=500)
        except Exception:
            self.res_ = glm.fit(maxiter=300)

    def _predict_mean(self, rows, X) -> np.ndarray:
        D = self._design(X)
        exposure = rows["target_days"].clip(lower=1).to_numpy(float)
        eta = D @ np.asarray(self.res_.params, float) + np.log(exposure)
        return np.exp(np.clip(eta, -20, 12))


class PoissonGLM(_CountGLM):
    name = "poisson_glm"
    family_name = "poisson"


class NegativeBinomialGLM(_CountGLM):
    """NB2 count model. The fitted dispersion ``alpha_`` is reported because it
    drives the width of the predictive interval and hence the safety stock."""

    name = "negbin_glm"
    family_name = "negbin"

    def _interval(self, mu: np.ndarray) -> dict[str, np.ndarray]:
        # Use the model's own NB variance rather than the generic log-residual
        # interval: var = mu + alpha*mu^2.
        alpha = getattr(self, "alpha_", 0.5)
        var = mu + alpha * mu**2
        sd = np.sqrt(var)
        z = 1.2816
        return {"lo": np.clip(mu - z * sd, 0, None), "hi": mu + z * sd}
