"""Era-controlled size-ratio estimation.

Why this module exists
----------------------
The launch-cohort training design has a confound that produced a visibly wrong
answer. Twin listings only began launching **2024-03-26**; Queen and King
cohorts span 2019-2026. The 2024+ era is roughly 2.1x stronger than pre-2024
(mean first-120-day organic units: Queen 274 vs 129). So any Twin/Queen ratio
estimated by pooling launch cohorts across eras compares recent Twins against a
Queen average dragged down by 2019-2023 failures.

The damage was not subtle. Pooled across eras, the hierarchical model put
Twin at **1.54x Queen**, which drove Twin to 53% of a draft order sheet. Every
era-controlled estimate disagrees:

    within-cohort, same colour & launch (n=5, recent)   Twin 0.594   King 0.749
    mature organic, within family (n=10 / n=18)         Twin 0.558   King 0.764
    incumbent heuristic assumption                      Twin 0.600   King 0.760

Three independent identification strategies agreeing to two decimals is much
stronger evidence than one pooled regression, so the size split is estimated
here rather than left to the demand model.

Identification
--------------
Ratios are taken **within launch cohort** — same colour, same programme, same
launch quarter — so era and colour both cancel. Cohorts are then pooled with
shrinkage toward the global ratio, because individual cohorts are noisy (Pink
came in at 0.325, Black at 1.531 on single listings).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Cohorts launched before this date are excluded from ratio estimation: no Twin
#: exists before it, so including them is what creates the confound.
TWIN_ERA_START = pd.Timestamp("2024-01-01")

#: Effective prior weight (in cohorts) pulling each ratio toward the pooled mean.
SHRINK_COHORTS = 2.0


def within_cohort_ratios(panel: pd.DataFrame, asof: pd.Timestamp,
                         anchor: str = "Queen", horizon: int = 120,
                         era_start: pd.Timestamp = TWIN_ERA_START) -> pd.DataFrame:
    """Size ratios vs ``anchor``, estimated within launch cohort.

    Only cohorts whose full ``horizon`` of life is observed by ``asof`` are used,
    so the ratio is a like-for-like comparison over equal exposure.
    """
    p = panel[panel.date <= asof]
    meta = p.drop_duplicates("child_asin")[
        ["child_asin", "colour", "size", "program", "launch_date"]]

    unit_col = "organic_units" if "organic_units" in p.columns else "units_ordered"
    observed = p.groupby("child_asin").days_since_launch.max()
    first = (p[p.days_since_launch <= horizon - 1]
             .groupby("child_asin")[unit_col].sum().rename("u"))

    d = meta.merge(first, on="child_asin", how="inner")
    d = d[d.child_asin.map(observed) >= horizon - 1]
    d = d[d.launch_date >= era_start]

    d["cohort"] = (d.colour.astype(str) + "|" + d.program.astype(str) + "|"
                   + d.launch_date.dt.to_period("Q").astype(str))

    recs = []
    for _, g in d.groupby("cohort"):
        tot = g.groupby("size").u.sum()
        if anchor not in tot or tot[anchor] <= 0:
            continue
        for sz, v in tot.items():
            if sz != anchor:
                recs.append({"size": sz, "ratio": v / tot[anchor]})
    obs = pd.DataFrame(recs)
    if obs.empty:
        return pd.DataFrame(columns=["size", "ratio", "n_cohorts", "shrunk_ratio"])

    # Pool on the log scale — ratios are multiplicative and right-skewed.
    obs["log_ratio"] = np.log(obs.ratio.clip(lower=1e-3))
    grand = float(obs.log_ratio.median())

    out = []
    for sz, g in obs.groupby("size"):
        n = len(g)
        med = float(g.log_ratio.median())
        shrunk = (n * med + SHRINK_COHORTS * grand) / (n + SHRINK_COHORTS)
        out.append({"size": sz, "ratio_raw": float(np.exp(med)), "n_cohorts": n,
                    "ratio": float(np.exp(shrunk)),
                    "q25": float(np.exp(g.log_ratio.quantile(0.25))),
                    "q75": float(np.exp(g.log_ratio.quantile(0.75)))})
    res = pd.DataFrame(out)
    res = pd.concat([res, pd.DataFrame([{"size": anchor, "ratio_raw": 1.0,
                                         "n_cohorts": int(obs.groupby("size").size().max()),
                                         "ratio": 1.0, "q25": 1.0, "q75": 1.0}])],
                    ignore_index=True)
    return res.sort_values("ratio", ascending=False).reset_index(drop=True)


def mature_ratios(panel: pd.DataFrame, asof: pd.Timestamp,
                  anchor: str = "Queen", min_age: int = 180) -> pd.DataFrame:
    """Size ratios from mature listings, computed within shade family.

    Independent corroboration of ``within_cohort_ratios``: a different sample
    (mature rather than launching) and a different control (family rather than
    cohort). Agreement between the two is the reason for confidence in the split.
    """
    p = panel[(panel.date <= asof) & panel.is_organic_day
              & (panel.days_since_launch >= min_age)]
    unit_col = "organic_units" if "organic_units" in p.columns else "units_ordered"
    g = p.groupby(["program", "shade_family", "size"]).agg(
        u=(unit_col, "sum"), d=(unit_col, "size"))
    vel = (g.u / g.d.clip(lower=1)).rename("v").reset_index()
    piv = vel.pivot_table(index=["program", "shade_family"], columns="size", values="v")
    if anchor not in piv.columns:
        return pd.DataFrame()
    rat = piv.div(piv[anchor], axis=0)
    out = []
    for sz in rat.columns:
        v = rat[sz].replace([np.inf, -np.inf], np.nan).dropna()
        if len(v):
            out.append({"size": sz, "ratio": float(v.median()), "n_families": len(v)})
    return pd.DataFrame(out).sort_values("ratio", ascending=False).reset_index(drop=True)


def size_weights(panel: pd.DataFrame, asof: pd.Timestamp, sizes: list[str],
                 anchor: str = "Queen", horizon: int = 120) -> pd.Series:
    """Normalised allocation weights across ``sizes``, summing to 1.

    Blends the two independent estimators. Where they disagree the within-cohort
    figure is trusted for the level (it measures launch behaviour over the same
    horizon as the buy) but is pulled toward the mature figure, since the mature
    sample is an order of magnitude larger.
    """
    wc = within_cohort_ratios(panel, asof, anchor=anchor, horizon=horizon)
    mt = mature_ratios(panel, asof, anchor=anchor)

    ratios = {}
    for sz in sizes:
        cands, weights = [], []
        if len(wc) and sz in set(wc["size"]):
            row = wc[wc["size"] == sz].iloc[0]
            cands.append(np.log(max(row.ratio, 1e-3)))
            weights.append(float(row.n_cohorts))
        if len(mt) and sz in set(mt["size"]):
            row = mt[mt["size"] == sz].iloc[0]
            cands.append(np.log(max(row.ratio, 1e-3)))
            weights.append(float(row.n_families))
        if not cands:
            ratios[sz] = 1.0 if sz == anchor else 0.6
        else:
            ratios[sz] = float(np.exp(np.average(cands, weights=weights)))

    s = pd.Series(ratios)
    return s / s.sum()


def report(panel: pd.DataFrame, asof: pd.Timestamp, sizes: list[str]) -> pd.DataFrame:
    """Side-by-side of both estimators plus the blended weights, for the audit trail."""
    wc = within_cohort_ratios(panel, asof).set_index("size")
    mt = mature_ratios(panel, asof).set_index("size")
    w = size_weights(panel, asof, sizes)
    rows = []
    for sz in sizes:
        rows.append({
            "size": sz,
            "within_cohort_ratio": round(float(wc.ratio.get(sz, np.nan)), 3),
            "within_cohort_n": int(wc.n_cohorts.get(sz, 0)) if sz in wc.index else 0,
            "mature_ratio": round(float(mt.ratio.get(sz, np.nan)), 3),
            "mature_n_families": int(mt.n_families.get(sz, 0)) if sz in mt.index else 0,
            "blended_weight": round(float(w.get(sz, np.nan)), 3),
        })
    return pd.DataFrame(rows)
