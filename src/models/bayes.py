"""Hierarchical Bayesian velocity model with partial pooling.

This is the model that earns its place on the Twin problem. The focal program
has *no* mature Twin listings, and across the sibling programs the Twin/Queen
ratio ranges from 0.27 to 0.98 on ten families — a spread wide enough that any
single point estimate is a guess. Pooling fully (one global Twin ratio) throws
away real family-level signal; pooling not at all (per-family ratios) fits
noise. Partial pooling is the correct answer, and it also delivers the thing the
order decision actually needs: a *posterior* over velocity, so safety stock can
be set from a quantile instead of a gut multiplier.

Structure, on log velocity:

    log v = mu + a[program] + b[shade_family] + c[size] + d[family x size] + e

Each effect gets its own variance component, so the data decide how much to
shrink each level. Sparse cells (Twin) shrink hard toward their parent; dense
cells (Queen, White) barely move. Variance components are sampled rather than
plugged in, which is what keeps the intervals honest for small cells.

Fitted by Gibbs sampling — all conditionals are conjugate (Normal for effects,
Inverse-Gamma for variances), so no external sampler is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Forecaster, Prediction

GROUPS = ["program", "shade_family", "size", "family_size"]


class HierarchicalBayesVelocity(Forecaster):
    name = "hier_bayes"

    def __init__(self, n_iter: int = 4000, burn: int = 1000, thin: int = 3, **kw):
        super().__init__(**kw)
        self.n_iter, self.burn, self.thin = n_iter, burn, thin

    # ------------------------------------------------------------------
    def _encode(self, rows: pd.DataFrame) -> dict[str, np.ndarray]:
        r = rows.copy()
        r["family_size"] = r["shade_family"].astype(str) + "|" + r["size"].astype(str)
        idx = {}
        for g in GROUPS:
            levels = getattr(self, f"levels_{g}", None)
            if levels is None:
                levels = sorted(r[g].astype(str).unique())
                setattr(self, f"levels_{g}", levels)
            lut = {v: i for i, v in enumerate(levels)}
            # Unseen level -> its own slot, which draws purely from the prior
            # (i.e. shrinks all the way to the parent mean). This is exactly the
            # behaviour wanted for a size the focal program has never sold.
            idx[g] = r[g].astype(str).map(lut).fillna(len(levels)).astype(int).to_numpy()
        return idx

    def _fit(self, rows, X, y) -> None:
        rng = np.random.default_rng(self.params.get("seed", 42))
        days = rows["target_days"].clip(lower=1).to_numpy(float)
        # Work on log velocity. The +0.5 continuity correction keeps zero-sale
        # listings in the sample instead of dropping them (they are informative).
        obs = np.log((np.clip(y, 0, None) + 0.5) / days)
        idx = self._encode(rows)
        n = len(obs)

        sizes = {g: len(getattr(self, f"levels_{g}")) + 1 for g in GROUPS}
        eff = {g: np.zeros(sizes[g]) for g in GROUPS}
        tau2 = {g: 0.5 for g in GROUPS}          # effect variances
        sigma2 = 0.5                              # residual variance
        mu = float(obs.mean())

        a0, b0 = 2.0, 0.5                         # weakly informative IG prior
        draws = {g: [] for g in GROUPS}
        draws["mu"], draws["sigma2"] = [], []

        for it in range(self.n_iter):
            # --- grand mean -------------------------------------------------
            resid = obs - sum(eff[g][idx[g]] for g in GROUPS)
            prec = n / sigma2 + 1 / 100.0
            mean = (resid.sum() / sigma2) / prec
            mu = rng.normal(mean, np.sqrt(1 / prec))

            # --- group effects (conjugate Normal, one block at a time) -------
            for g in GROUPS:
                partial = obs - mu - sum(eff[h][idx[h]] for h in GROUPS if h != g)
                new = np.zeros(sizes[g])
                # Vectorised sufficient statistics per level.
                cnt = np.bincount(idx[g], minlength=sizes[g]).astype(float)
                ssum = np.bincount(idx[g], weights=partial, minlength=sizes[g])
                prec_l = cnt / sigma2 + 1 / tau2[g]
                mean_l = (ssum / sigma2) / prec_l
                new = rng.normal(mean_l, np.sqrt(1 / prec_l))
                # Sum-to-zero centring keeps the decomposition identified.
                observed = cnt > 0
                if observed.any():
                    new[observed] -= new[observed].mean()
                eff[g] = new

                # --- effect variance -------------------------------------
                k = max(int(observed.sum()), 1)
                ss = float(np.sum(eff[g][observed] ** 2))
                tau2[g] = 1.0 / rng.gamma(a0 + k / 2, 1.0 / (b0 + ss / 2))

            # --- residual variance ---------------------------------------
            r = obs - mu - sum(eff[g][idx[g]] for g in GROUPS)
            sigma2 = 1.0 / rng.gamma(a0 + n / 2, 1.0 / (b0 + float(np.sum(r**2)) / 2))

            if it >= self.burn and (it - self.burn) % self.thin == 0:
                for g in GROUPS:
                    draws[g].append(eff[g].copy())
                draws["mu"].append(mu)
                draws["sigma2"].append(sigma2)

        self.post_ = {g: np.array(v) for g, v in draws.items() if g in GROUPS}
        self.post_["mu"] = np.array(draws["mu"])
        self.post_["sigma2"] = np.array(draws["sigma2"])
        self.n_draws_ = len(self.post_["mu"])

    # ------------------------------------------------------------------
    def _posterior_log_velocity(self, rows: pd.DataFrame) -> np.ndarray:
        """(n_draws, n_rows) posterior draws of log velocity."""
        idx = self._encode(rows)
        out = self.post_["mu"][:, None] * np.ones((1, len(rows)))
        for g in GROUPS:
            out = out + self.post_[g][:, idx[g]]
        return out

    def _predict_mean(self, rows, X) -> np.ndarray:
        lv = self._posterior_log_velocity(rows)
        days = rows["target_days"].to_numpy(float)
        # Posterior predictive mean on the unit scale (log-normal correction).
        sig2 = self.post_["sigma2"][:, None]
        return (np.exp(lv + sig2 / 2).mean(axis=0)) * days

    def predict(self, rows, X) -> Prediction:
        lv = self._posterior_log_velocity(rows)
        days = rows["target_days"].to_numpy(float)
        sig2 = self.post_["sigma2"][:, None]
        rng = np.random.default_rng(self.params.get("seed", 42) + 1)
        # Full posterior predictive: parameter uncertainty + residual scatter +
        # Poisson counting noise, which is what makes the interval usable as a
        # safety-stock input rather than just a confidence band.
        lam = np.exp(lv + rng.normal(0, np.sqrt(sig2), size=lv.shape)) * days
        sim = rng.poisson(np.clip(lam, 0, 1e6))
        return Prediction(
            mean=np.clip(np.exp(lv + sig2 / 2).mean(axis=0) * days, 0, None),
            lo=np.quantile(sim, 0.10, axis=0),
            hi=np.quantile(sim, 0.90, axis=0),
            meta={"n_draws": self.n_draws_},
        )

    # ------------------------------------------------------------------
    def size_ratio_posterior(self, ref: str = "Queen") -> pd.DataFrame:
        """Posterior distribution of each size's velocity ratio vs the anchor.

        This is the direct, defensible answer to "what is the Twin ratio?" —
        a posterior with a credible interval, rather than the heuristic's 0.60
        with an unquantified 0.35-1.14 range attached by hand.
        """
        levels = self.levels_size
        if ref not in levels:
            return pd.DataFrame()
        j_ref = levels.index(ref)
        draws = self.post_["size"]
        rows = []
        for i, sz in enumerate(levels):
            r = np.exp(draws[:, i] - draws[:, j_ref])
            rows.append({
                "size": sz,
                "ratio_median": float(np.median(r)),
                "ratio_mean": float(np.mean(r)),
                "q05": float(np.quantile(r, 0.05)),
                "q25": float(np.quantile(r, 0.25)),
                "q75": float(np.quantile(r, 0.75)),
                "q95": float(np.quantile(r, 0.95)),
            })
        return pd.DataFrame(rows).sort_values("ratio_median", ascending=False)

    def variance_components(self) -> pd.DataFrame:
        """How much of the variation each level explains — i.e. where the signal
        actually lives (colour family vs size vs program vs interaction)."""
        rows = []
        for g in GROUPS:
            v = float(np.var(self.post_[g], axis=0).mean() + np.mean(self.post_[g].var(axis=1)))
            rows.append({"level": g, "effect_sd": float(np.std(self.post_[g]))})
        rows.append({"level": "residual", "effect_sd": float(np.sqrt(self.post_["sigma2"].mean()))})
        return pd.DataFrame(rows)
