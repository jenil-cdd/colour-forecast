"""Test every assumption in the heuristic order model against the data.

The heuristic recommended 2,180 units off six hard-coded constants. Each one is
re-estimated here on the training window only (<= train_end), so the numbers are
what a planner could actually have known at order time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402

pd.set_option("display.width", 200)


def main() -> None:
    cfg = load_config()
    tr_end = pd.Timestamp(cfg.split.train_end)
    p = pd.read_parquet("data/processed/panel.parquet")
    focal = cfg.focal_program
    train = p[p.date <= tr_end]

    print("=" * 78)
    print(f"ASSUMPTION AUDIT — training window {train.date.min().date()} .. {tr_end.date()}")
    print("=" * 78)

    # -- 1. White share of volume ------------------------------------------
    f = train[train.program == focal]
    last90 = f[f.date > tr_end - pd.Timedelta(days=90)]
    for label, d in [("all history", f), ("last 90d", last90)]:
        tot = d.units_ordered.sum()
        wht = d.loc[d.is_core_white, "units_ordered"].sum()
        print(f"\n[1] Core-White share of {focal} volume ({label}): "
              f"{wht/tot:6.1%}  (white={wht:,.0f} / total={tot:,.0f})   heuristic claim: 62.6%")

    # -- 2/3. Size ratios vs Queen anchor ----------------------------------
    print("\n[2/3] Size ratios vs Queen — organic days only, mature listings, non-white")
    org = train[train.is_organic_day & ~train.is_core_white & (train.days_since_launch >= 180)]
    vel = (org.groupby(["program", "shade_family", "size"])
              .agg(units=("units_ordered", "sum"), days=("units_ordered", "size")))
    vel["per_day"] = vel.units / vel.days
    piv = vel["per_day"].unstack("size")
    if "Queen" in piv.columns:
        ratios = piv.div(piv["Queen"], axis=0).dropna(subset=["Queen"])
        print("\n  per-family ratio to Queen:")
        print(ratios[[c for c in ["Queen", "King", "Twin", "Oversized King"] if c in ratios]]
              .round(3).to_string())
        print("\n  pooled ratio summary (families with a Queen anchor):")
        for sz, prior in [("King", 0.76), ("Twin", 0.60)]:
            if sz not in ratios:
                continue
            r = ratios[sz].dropna()
            r = r[np.isfinite(r)]
            if len(r) == 0:
                continue
            print(f"    {sz:15s} n={len(r):2d}  median={r.median():.3f}  mean={r.mean():.3f}  "
                  f"IQR=[{r.quantile(.25):.3f},{r.quantile(.75):.3f}]   heuristic: {prior}")

    # -- 4. Family depth decay ---------------------------------------------
    print("\n[4] Family depth decay — organic per-day velocity by entry rank")
    dd = train[train.is_organic_day & (train.days_since_launch.between(30, 365))]
    dd = dd[dd["size"].isin(["Queen", "King"])]
    tab = (dd.groupby("family_rank")
             .agg(asins=("child_asin", "nunique"), days=("units_ordered", "size"),
                  per_day=("units_ordered", "mean")))
    print(tab.round(3).to_string())
    print("   heuristic claim (Grey): 1st=1.42, 2nd=1.00, 3rd=0.58, 4th=0.00 units/day")
    g = dd[dd.shade_family.isin(["Light Grey", "Mid Grey", "Dark Grey"])]
    if len(g):
        print("\n   grey families only:")
        print(g.groupby("family_rank").units_ordered.agg(["size", "mean"]).round(3).to_string())

    # -- 5. Ramp curve -----------------------------------------------------
    print("\n[5] Ramp-up curve — velocity by month since launch, relative to months 7-12")
    hist = train[train.is_organic_day]
    launched = hist[hist.groupby("child_asin").date.transform("min") >= pd.Timestamp("2021-01-01")]
    mo = (launched[launched.month_since_launch <= 12]
          .groupby("month_since_launch").units_ordered.agg(["size", "mean"]))
    steady = mo.loc[6:11, "mean"].mean() if len(mo) > 7 else np.nan
    mo["vs_steady"] = mo["mean"] / steady
    print(mo.round(3).to_string())
    print(f"   steady-state (months 7-12) = {steady:.3f} units/day")
    print("   heuristic claim: month 1 = 45% of steady state")

    # -- 6. Promo contamination in the test window -------------------------
    print("\n[6] Promo-day incidence, focal program")
    per_yr = (p[p.program == focal]
              .assign(yr=lambda d: d.date.dt.year)
              .groupby("yr")[["is_promo_day", "is_organic_day", "cal_no_deal"]].mean())
    print(per_yr.round(3).to_string())


if __name__ == "__main__":
    main()
