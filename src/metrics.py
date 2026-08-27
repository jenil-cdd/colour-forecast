"""Forecast accuracy and decision-quality metrics.

Point-accuracy metrics are reported on cumulative window units. MAPE is
deliberately *not* used as a headline: several test SKUs sell single-digit units
over the window, and MAPE explodes on small denominators. WAPE (volume-weighted
absolute error) is the primary metric because it maps directly onto units of
inventory misplaced, which is the actual business cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted absolute percentage error = sum|e| / sum(y). Scale-free, robust
    to zeros, and reads as "we were off by X% of the volume we shipped"."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom > 0 else np.nan


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def bias(y, yhat) -> float:
    """Signed mean error as a share of actual volume. Positive = over-forecast."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y).sum()
    return float((yhat - y).sum() / denom) if denom > 0 else np.nan


def smape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    d = (np.abs(y) + np.abs(yhat)) / 2
    m = d > 0
    return float(np.mean(np.abs(y - yhat)[m] / d[m])) if m.any() else np.nan


def rmsle(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((np.log1p(np.clip(yhat, 0, None)) - np.log1p(np.clip(y, 0, None))) ** 2)))


def pinball(y, q_pred, tau: float) -> float:
    """Quantile (pinball) loss — scores the interval forecast, not just the point."""
    y, q_pred = np.asarray(y, float), np.asarray(q_pred, float)
    d = y - q_pred
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def coverage(y, lo, hi) -> float:
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_score(y, lo, hi, alpha: float = 0.2) -> float:
    """Winkler interval score: width penalised, misses penalised harder.
    Lower is better; rewards intervals that are tight *and* honest."""
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    return float(np.mean((hi - lo)
                         + (2 / alpha) * (lo - y) * (y < lo)
                         + (2 / alpha) * (y - hi) * (y > hi)))


def newsvendor_cost(y, order, unit_cost: float = 1.0, holding: float = 0.25,
                    stockout: float = 1.0) -> float:
    """Asymmetric inventory cost of an order decision.

    Over-ordering costs carrying/markdown (``holding`` per unit); under-ordering
    costs lost margin plus the search-rank damage a month-3 stock-out does
    (``stockout`` per unit). Defaults make a stock-out 4x worse than an
    equivalent overstock, which is why the recommended order sits above the
    median forecast.
    """
    y, order = np.asarray(y, float), np.asarray(order, float)
    over = np.clip(order - y, 0, None) * holding * unit_cost
    under = np.clip(y - order, 0, None) * stockout * unit_cost
    return float((over + under).sum())


def spearman(y, yhat) -> float:
    """Rank correlation between forecast and actual.

    For the *colour selection* half of the decision, getting the ordering right
    matters more than getting the level right: the buyer needs to know that
    Antique White will outsell Silver, even if both levels are off. A model can
    have mediocre WAPE and still be the right tool for choosing colours.
    """
    from scipy.stats import spearmanr

    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if len(y) < 3 or np.all(yhat == yhat[0]):
        return np.nan
    r = spearmanr(y, yhat).statistic
    return float(r) if np.isfinite(r) else np.nan


def topk_hit_rate(y, yhat, k: int = 6) -> float:
    """Share of the truly best-k variants that the model also ranked in its
    top k. This is the metric that maps onto "did we pick the right colours"."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    k = min(k, len(y))
    if k == 0:
        return np.nan
    true_top = set(np.argsort(-y)[:k])
    pred_top = set(np.argsort(-yhat)[:k])
    return len(true_top & pred_top) / k


def evaluate(y, yhat, lo=None, hi=None, prefix: str = "") -> dict[str, float]:
    out = {
        f"{prefix}wape": wape(y, yhat),
        f"{prefix}mae": mae(y, yhat),
        f"{prefix}rmse": rmse(y, yhat),
        f"{prefix}rmsle": rmsle(y, yhat),
        f"{prefix}bias": bias(y, yhat),
        f"{prefix}smape": smape(y, yhat),
        f"{prefix}spearman": spearman(y, yhat),
        f"{prefix}top6_hit": topk_hit_rate(y, yhat, k=6),
    }
    if lo is not None and hi is not None:
        out[f"{prefix}coverage80"] = coverage(y, lo, hi)
        out[f"{prefix}interval_score"] = interval_score(y, lo, hi)
        out[f"{prefix}pinball10"] = pinball(y, lo, 0.10)
        out[f"{prefix}pinball90"] = pinball(y, hi, 0.90)
    return out


def leaderboard(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    sort_col = "wape" if "wape" in df.columns else df.columns[0]
    return df.sort_values(sort_col)
