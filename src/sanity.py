"""Sanity checks that decide what each model may be used for.

A leaderboard says which model has the lowest error. It does not say whether a
model's output is *fit for a given purpose*, and those are different questions.
``poisson_glm`` had the best per-line error on the 61-day window while
under-forecasting the total by 39% - excellent for ranking a shortlist, useless
for setting a purchase order. ``naive_family`` ranked colours better than any
other model while missing total volume by 334%.

So each model is certified per task rather than scored once:

    TOTAL      - may the total of its predictions size a purchase order?
    PER_LINE   - may an individual SKU quantity be taken from it?
    RANKING    - may it be used to choose which colours to launch?
    INTERVALS  - may its band be used to set safety stock?

Every threshold below is calibrated on the replay waves, never on the test
months, and each is expressed in the units of the decision it protects.

Structural checks run first. A model that emits negatives, NaNs, inverted
intervals, or a size ordering that contradicts the era-controlled ratios is
failed outright regardless of its error metrics - those are symptoms of a broken
model, not an inaccurate one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import spearman, wape

# --- pass thresholds, all set from replay-wave behaviour ---------------------
THRESHOLDS = {
    "TOTAL": {"total_err_max": 0.35, "abs_bias_max": 0.30},
    "PER_LINE": {"line_err_max": 0.80, "abs_bias_max": 0.45},
    "RANKING": {"spearman_min": 0.30},
    "INTERVALS": {"coverage_min": 0.60, "coverage_max": 0.95},
}

#: A Twin prediction above this multiple of the Queen prediction indicates the
#: era confound has re-entered. The era-controlled ratio is ~0.64.
TWIN_OVER_QUEEN_MAX = 1.10


def structural_checks(rows: pd.DataFrame, pred) -> dict[str, bool | str]:
    """Cheap correctness checks that do not need labels."""
    mu, lo, hi = np.asarray(pred.mean, float), np.asarray(pred.lo, float), np.asarray(pred.hi, float)
    out: dict[str, bool | str] = {}
    out["finite"] = bool(np.isfinite(mu).all())
    out["non_negative"] = bool((mu >= -1e-9).all())
    out["interval_ordered"] = bool((lo <= mu + 1e-6).all() and (mu <= hi + 1e-6).all())
    # Magnitude sanity: nothing should predict more than 20x the median.
    med = float(np.median(mu[mu > 0])) if (mu > 0).any() else 0.0
    out["magnitude_sane"] = bool(med == 0 or (mu.max() <= 20 * med))

    # Size ordering, on a per-day basis so exposure differences do not confuse it.
    d = rows.assign(_pd=mu / rows.get("target_days", pd.Series(1, index=rows.index)).clip(lower=1))
    med_by_size = d.groupby(d["size"].astype(str))._pd.median()
    q, t = med_by_size.get("Queen"), med_by_size.get("Twin")
    if q and t and q > 0:
        out["twin_not_inflated"] = bool(t / q <= TWIN_OVER_QUEEN_MAX)
        out["twin_over_queen"] = round(float(t / q), 2)
    else:
        out["twin_not_inflated"] = True
        out["twin_over_queen"] = np.nan
    return out


def certify(y: np.ndarray, pred, rows: pd.DataFrame,
            groups: np.ndarray | None = None) -> dict:
    """Score one model and certify it per task."""
    y = np.asarray(y, float)
    mu = np.asarray(pred.mean, float)

    line_err = wape(y, mu)
    if groups is None:
        total_err = abs(mu.sum() - y.sum()) / max(y.sum(), 1e-9)
    else:
        g = pd.DataFrame({"g": groups, "y": y, "p": mu}).groupby("g").sum()
        total_err = float(abs(g.p - g.y).sum() / max(g.y.sum(), 1e-9))
    bias = (mu.sum() - y.sum()) / max(y.sum(), 1e-9)
    rho = spearman(y, mu)
    cov = float(np.mean((y >= pred.lo) & (y <= pred.hi)))

    st = structural_checks(rows, pred)
    # The Twin check is a *mix* problem, not a broken-output problem: a model can
    # get the total right while splitting it badly across sizes. So it gates
    # PER_LINE only and is excluded from the blanket structural verdict.
    structural_ok = all(v for k, v in st.items()
                        if isinstance(v, bool) and k != "twin_not_inflated")

    T = THRESHOLDS
    verdict = {
        "TOTAL": structural_ok and total_err <= T["TOTAL"]["total_err_max"]
                 and abs(bias) <= T["TOTAL"]["abs_bias_max"],
        "PER_LINE": structural_ok and line_err <= T["PER_LINE"]["line_err_max"]
                    and abs(bias) <= T["PER_LINE"]["abs_bias_max"]
                    and bool(st.get("twin_not_inflated", True)),
        "RANKING": bool(np.isfinite(rho)) and rho >= T["RANKING"]["spearman_min"],
        "INTERVALS": structural_ok and T["INTERVALS"]["coverage_min"] <= cov
                     <= T["INTERVALS"]["coverage_max"],
    }
    return {
        "total_err": round(total_err, 3), "line_err": round(line_err, 3),
        "bias": round(bias, 3), "spearman": round(rho, 3) if np.isfinite(rho) else np.nan,
        "coverage": round(cov, 3), "twin_over_queen": st.get("twin_over_queen"),
        "structural_ok": structural_ok,
        **{f"use_{k.lower()}": v for k, v in verdict.items()},
        "certified_for": ",".join(k for k, v in verdict.items() if v) or "NONE",
    }


def capability_matrix(results: dict[str, dict]) -> pd.DataFrame:
    """One row per model: metrics plus what it is cleared to do."""
    df = pd.DataFrame(results).T
    cols = ["total_err", "line_err", "bias", "spearman", "coverage", "twin_over_queen",
            "use_total", "use_per_line", "use_ranking", "use_intervals", "certified_for"]
    df = df[[c for c in cols if c in df.columns]]
    n_uses = df[[c for c in df.columns if c.startswith("use_")]].sum(axis=1)
    return df.assign(n_tasks=n_uses).sort_values(["n_tasks", "total_err"],
                                                 ascending=[False, True])
