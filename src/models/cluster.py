"""Demand-pool segmentation by clustering.

The brief's language about "saturated families" versus "high-demand pools"
presumes the pools are known. They are not — the mapper's own colour families
are missing for more than half the catalogue and are inconsistent where present
("Multi" and "Navy Dot" for the same ASIN). Clustering derives the pools from
data: colours are grouped in Lab space plus realised velocity, and the cluster
mean becomes the forecast for anything landing in that cluster.

Its value is less as a forecaster than as a *diagnostic*: it shows which colour
regions are crowded and which are empty, which is the input to the colour
selection half of the decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.models.base import Forecaster

FEATS = ["lab_L", "lab_a", "lab_b", "dist_to_white", "lab_chroma"]


class ClusterPoolMean(Forecaster):
    name = "cluster_pool"

    def __init__(self, k: int = 6, **kw):
        super().__init__(**kw)
        self.k = k

    def _fit(self, rows, X, y) -> None:
        days = rows["target_days"].clip(lower=1).to_numpy(float)
        vel = np.clip(y, 0, None) / days
        Z = rows[FEATS].astype(float).fillna(0).to_numpy()
        self.scaler_ = StandardScaler().fit(Z)
        k = int(min(self.k, max(2, len(rows) // 4)))
        self.km_ = KMeans(n_clusters=k, n_init=10,
                          random_state=self.params.get("seed", 42)).fit(self.scaler_.transform(Z))

        lab = self.km_.labels_
        # Velocity is estimated per (cluster, size) because size dominates level.
        df = pd.DataFrame({"cluster": lab, "size": rows["size"].to_numpy(), "vel": vel})
        self.cluster_size_vel_ = df.groupby(["cluster", "size"]).vel.median()
        self.cluster_vel_ = df.groupby("cluster").vel.median()
        self.global_vel_ = float(np.median(vel)) if len(vel) else 0.5

    def _predict_mean(self, rows, X) -> np.ndarray:
        Z = self.scaler_.transform(rows[FEATS].astype(float).fillna(0).to_numpy())
        lab = self.km_.predict(Z)
        out = []
        for c, sz in zip(lab, rows["size"].astype(str)):
            v = self.cluster_size_vel_.get((c, sz))
            if v is None or not np.isfinite(v):
                v = self.cluster_vel_.get(c, self.global_vel_)
            out.append(float(v))
        return np.array(out) * rows["target_days"].to_numpy(float)

    def profile(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Which colours landed in which pool, and how crowded each pool is."""
        Z = self.scaler_.transform(rows[FEATS].astype(float).fillna(0).to_numpy())
        lab = self.km_.predict(Z)
        d = rows.assign(cluster=lab)
        return (d.groupby("cluster")
                 .agg(n_listings=("child_asin", "nunique"),
                      colours=("colour", lambda s: ", ".join(sorted(set(s))[:6])),
                      mean_L=("lab_L", "mean"),
                      mean_dist_white=("dist_to_white", "mean"))
                 .reset_index())
