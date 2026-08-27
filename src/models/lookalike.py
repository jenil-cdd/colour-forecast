"""Instance-based cold-start forecasting: match the new colour to past launches.

The premise is that the best predictor of how "Silver Queen" will launch is how
the most similar previously-launched variants actually launched. No functional
form is assumed at all — the forecast is a weighted average of realised launch
curves from neighbours.

Similarity is computed in a blended space:

* **Perceptual** — CIE Lab distance, so "Silver" sits next to "Light Gray"
  because the colours genuinely are next to each other, not because someone
  wrote them into the same bucket.
* **Structural** — same size, same program, comparable family depth.

This is the model that handles a genuinely novel colour (an unentered family
like Green or Terracotta) most gracefully, because it never needs that family to
have existed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Forecaster, Prediction

LAB_COLS = ["lab_L", "lab_a", "lab_b"]
STRUCT_COLS = ["family_rank", "family_live_count", "dist_to_white", "lab_chroma"]


class KNNLookalike(Forecaster):
    name = "knn_lookalike"

    def __init__(self, k: int = 5, lab_weight: float = 0.6, **kw):
        super().__init__(**kw)
        self.k, self.lab_weight = k, lab_weight

    def _features(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lab = rows[LAB_COLS].astype(float).to_numpy()
        st = rows[[c for c in STRUCT_COLS if c in rows.columns]].astype(float).fillna(0).to_numpy()
        return lab, st

    def _fit(self, rows, X, y) -> None:
        self.train_rows_ = rows.reset_index(drop=True).copy()
        days = rows["target_days"].clip(lower=1).to_numpy(float)
        self.train_velocity_ = np.clip(y, 0, None) / days

        lab, st = self._features(self.train_rows_)
        self.lab_mu_, self.lab_sd_ = lab.mean(0), np.where(lab.std(0) > 1e-8, lab.std(0), 1.0)
        self.st_mu_, self.st_sd_ = st.mean(0), np.where(st.std(0) > 1e-8, st.std(0), 1.0)
        self.train_lab_ = (lab - self.lab_mu_) / self.lab_sd_
        self.train_st_ = (st - self.st_mu_) / self.st_sd_
        self.global_velocity_ = float(np.mean(self.train_velocity_)) if len(y) else 0.5

    def _neighbours(self, rows: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        lab, st = self._features(rows)
        lab = (lab - self.lab_mu_) / self.lab_sd_
        st = (st - self.st_mu_) / self.st_sd_

        out = []
        for i in range(len(rows)):
            d_lab = np.linalg.norm(self.train_lab_ - lab[i], axis=1)
            d_st = np.linalg.norm(self.train_st_ - st[i], axis=1)
            d = self.lab_weight * d_lab + (1 - self.lab_weight) * d_st

            # Hard-prefer same size and program: a Twin launch curve is not
            # interchangeable with a King one, however close the colour is.
            same_size = (self.train_rows_["size"].to_numpy() == rows["size"].iloc[i])
            same_prog = (self.train_rows_["program"].to_numpy() == rows["program"].iloc[i])
            d = d + 3.0 * (~same_size) + 1.0 * (~same_prog)

            # Never match a variant to itself.
            d = np.where(self.train_rows_["child_asin"].to_numpy()
                         == rows["child_asin"].iloc[i], np.inf, d)

            order = np.argsort(d)[: self.k]
            w = 1.0 / (d[order] + 0.25)
            out.append((order, w / w.sum() if w.sum() > 0 else np.ones(len(order)) / max(len(order), 1)))
        return out

    def _predict_mean(self, rows, X) -> np.ndarray:
        nb = self._neighbours(rows)
        days = rows["target_days"].to_numpy(float)
        v = np.array([
            float(np.dot(w, self.train_velocity_[o])) if len(o) else self.global_velocity_
            for o, w in nb
        ])
        return v * days

    def predict(self, rows, X) -> Prediction:
        """Interval taken from the *dispersion among neighbours* — if the five
        closest historical launches disagree wildly, the forecast says so."""
        nb = self._neighbours(rows)
        days = rows["target_days"].to_numpy(float)
        mean, lo, hi = [], [], []
        for o, w in nb:
            v = self.train_velocity_[o] if len(o) else np.array([self.global_velocity_])
            m = float(np.dot(w, v)) if len(o) else self.global_velocity_
            mean.append(m)
            lo.append(float(np.quantile(v, 0.10)))
            hi.append(float(np.quantile(v, 0.90)))
        mean, lo, hi = np.array(mean), np.array(lo), np.array(hi)
        cv = 1.0 / np.sqrt(np.clip(mean * days, 1.0, None))
        return Prediction(
            mean=np.clip(mean * days, 0, None),
            lo=np.clip(lo * days * (1 - cv), 0, None),
            hi=hi * days * (1 + cv),
        )

    def explain(self, rows: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """Which historical launches drove each forecast — the audit trail a
        planner needs before signing a purchase order."""
        nb = self._neighbours(rows)
        recs = []
        for i, (o, w) in enumerate(nb):
            for j, wt in zip(o, w):
                t = self.train_rows_.iloc[j]
                recs.append({
                    "target_asin": rows["child_asin"].iloc[i],
                    "target": f"{rows['colour'].iloc[i]} {rows['size'].iloc[i]}",
                    "neighbour": f"{t['colour']} {t['size']}",
                    "neighbour_asin": t["child_asin"],
                    "weight": round(float(wt), 3),
                    "neighbour_velocity": round(float(self.train_velocity_[j]), 3),
                })
        return pd.DataFrame(recs)
