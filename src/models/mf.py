"""Matrix completion over the Size x Colour grid.

The catalogue grid is sparse and *structurally* so: the focal program has never
listed a Twin in most colour families, and several colours exist in King only.
Those gaps are exactly the cells a launch decision needs filled.

Treating the grid as a low-rank matrix says something specific and testable:
that colour appeal and size appeal are largely separable, with a small number of
latent factors capturing the interaction (e.g. "pale neutrals over-index in
Twin"). Fitting by ALS on log velocity with a global + row + column bias term
means rank-0 already reproduces the multiplicative size-ratio model, and the
latent factors only earn their keep to the extent real interaction exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Forecaster


class MatrixFactorisation(Forecaster):
    name = "matrix_factorisation"

    def __init__(self, rank: int = 2, n_iter: int = 200, reg: float = 0.5, **kw):
        super().__init__(**kw)
        self.rank, self.n_iter, self.reg = rank, n_iter, reg

    def _fit(self, rows, X, y) -> None:
        days = rows["target_days"].clip(lower=1).to_numpy(float)
        v = np.log((np.clip(y, 0, None) + 0.5) / days)

        # Rows of the matrix = colour identity (family x pattern), cols = size.
        rkey = (rows["shade_family"].astype(str) + "|" + rows["pattern_type"].astype(str))
        ckey = rows["size"].astype(str)
        self.rlev_, self.clev_ = sorted(rkey.unique()), sorted(ckey.unique())
        ri = rkey.map({k: i for i, k in enumerate(self.rlev_)}).to_numpy()
        ci = ckey.map({k: i for i, k in enumerate(self.clev_)}).to_numpy()
        nr, nc = len(self.rlev_), len(self.clev_)

        # Average duplicate observations into a single cell value + weight.
        M = np.zeros((nr, nc))
        W = np.zeros((nr, nc))
        np.add.at(M, (ri, ci), v)
        np.add.at(W, (ri, ci), 1.0)
        obs = W > 0
        M[obs] /= W[obs]

        self.mu_ = float(M[obs].mean()) if obs.any() else 0.0
        rng = np.random.default_rng(self.params.get("seed", 42))
        br, bc = np.zeros(nr), np.zeros(nc)
        P = 0.01 * rng.standard_normal((nr, self.rank))
        Q = 0.01 * rng.standard_normal((nc, self.rank))

        for _ in range(self.n_iter):
            # Bias terms first: these carry the separable size/colour effects.
            for i in range(nr):
                m = obs[i]
                if m.any():
                    br[i] = ((M[i, m] - self.mu_ - bc[m] - P[i] @ Q[m].T).sum()
                             / (m.sum() + self.reg))
            for j in range(nc):
                m = obs[:, j]
                if m.any():
                    bc[j] = ((M[m, j] - self.mu_ - br[m] - (P[m] * Q[j]).sum(1)).sum()
                             / (m.sum() + self.reg))
            # ALS on the residual interaction.
            R = M - self.mu_ - br[:, None] - bc[None, :]
            for i in range(nr):
                m = obs[i]
                if m.sum() >= 1:
                    A = Q[m].T @ Q[m] + self.reg * np.eye(self.rank)
                    P[i] = np.linalg.solve(A, Q[m].T @ R[i, m])
            for j in range(nc):
                m = obs[:, j]
                if m.sum() >= 1:
                    A = P[m].T @ P[m] + self.reg * np.eye(self.rank)
                    Q[j] = np.linalg.solve(A, P[m].T @ R[m, j])

        self.br_, self.bc_, self.P_, self.Q_ = br, bc, P, Q

    def _predict_mean(self, rows, X) -> np.ndarray:
        rkey = (rows["shade_family"].astype(str) + "|" + rows["pattern_type"].astype(str))
        ckey = rows["size"].astype(str)
        rmap = {k: i for i, k in enumerate(self.rlev_)}
        cmap = {k: i for i, k in enumerate(self.clev_)}
        out = np.empty(len(rows))
        for n, (rk, ck) in enumerate(zip(rkey, ckey)):
            i, j = rmap.get(rk), cmap.get(ck)
            val = self.mu_
            if i is not None:
                val += self.br_[i]
            if j is not None:
                val += self.bc_[j]
            if i is not None and j is not None:
                val += float(self.P_[i] @ self.Q_[j])
            out[n] = val
        return np.exp(out) * rows["target_days"].to_numpy(float)

    def completed_grid(self) -> pd.DataFrame:
        """The imputed Size x Colour velocity grid, including never-listed cells."""
        G = (self.mu_ + self.br_[:, None] + self.bc_[None, :] + self.P_ @ self.Q_.T)
        return pd.DataFrame(np.exp(G), index=self.rlev_, columns=self.clev_)
