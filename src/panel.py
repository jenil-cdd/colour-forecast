"""Build the modelling panel: one row per (child_asin, date).

Responsibilities, in order:

1. Densify the calendar per ASIN so zero-sales days are explicit zeros rather
   than missing rows (a Poisson/NegBin model must see the zeros).
2. Flag promo days. The curated no-deal calendar is too sparse to use alone
   (0 clean days in Jan-May 2026), so promo detection combines the ASIN-level
   deal table, a realised-ASP discount screen, and a robust velocity-spike
   screen. See ``add_promo_flags``.
3. Derive launch age and the listing's maturity state (cold-start features).
4. Derive family depth / cannibalisation state *as of each date*, which is the
   mechanic that makes the Nth colour in a family sell worse than the first.

Everything here is causal in time: a feature dated ``t`` only ever uses data
from ``<= t``. That is enforced by construction (expanding/shifted windows) and
checked in ``tests/test_leakage.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import PROCESSED, Config
from src.taxonomy import annotate

log = logging.getLogger(__name__)

# A listing is treated as "live" from its first appearance in the sales/traffic
# report. Amazon reports a row once the listing is buyable, including zero-sale
# days, so first appearance is a good launch-date proxy.
MATURITY_DAYS = 180


def densify(panel: pd.DataFrame) -> pd.DataFrame:
    """Expand to a complete date grid per ASIN, between its own first and last
    observed date. Missing days become explicit zero-demand days."""
    panel = panel.sort_values(["child_asin", "date"])
    spans = panel.groupby("child_asin")["date"].agg(["min", "max"])
    frames = []
    for asin, (lo, hi) in spans.iterrows():
        frames.append(pd.DataFrame({"child_asin": asin, "date": pd.date_range(lo, hi, freq="D")}))
    grid = pd.concat(frames, ignore_index=True)
    out = grid.merge(panel, on=["child_asin", "date"], how="left", indicator=True)

    # Amazon already reports zero-unit days for a live listing (36% of raw rows
    # have units_ordered == 0), so "reported" must come from the merge itself.
    # Deriving it from asp/units instead would silently classify every genuine
    # zero-demand day as unobserved and bias organic velocity upward.
    out["was_reported"] = out["_merge"] == "both"
    out = out.drop(columns=["_merge"])

    # Demand-side counters are genuinely zero on a day with no reported row.
    for col in ["units_ordered", "units_ordered_b2b", "sessions", "sessions_total",
                "page_views", "ordered_product_sales", "total_order_items"]:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)
    return out


def add_promo_flags(panel: pd.DataFrame, deals: pd.DataFrame, clean_days: pd.DataFrame,
                    cfg: Config) -> pd.DataFrame:
    """Flag days whose velocity is contaminated by promotion.

    Three independent signals, unioned:

    * ``deal_flag_src``  — ASIN-level BEST_DEAL calendar. Authoritative but
      stops at 2025-12-02, so it cannot cover the test window at all.
    * ``asp_discount``   — realised ASP below the ASIN's own trailing median.
      Validated against the deal table over the training period: recall ~0.69
      at a 0.93 threshold. This is the only promo signal that spans 2026.
    * ``velocity_spike`` — units far above the ASIN's trailing median on a
      robust (MAD) scale, which catches Prime Day style demand events that are
      not price-driven.

    Promo days are *flagged, not dropped*. Organic velocity is computed on the
    clean subset, but the flag is also exposed as a feature so models can
    condition on it, and the forecast is issued as a no-promo counterfactual.
    """
    out = panel.copy()

    deals = deals.copy()
    deals["deal_flag_src"] = True
    out = out.merge(deals[["child_asin", "date", "deal_flag_src"]],
                    on=["child_asin", "date"], how="left")
    out["deal_flag_src"] = out["deal_flag_src"].notna()

    cal = clean_days.copy()
    cal["cal_no_deal"] = True
    out = out.merge(cal[["date", "cal_no_deal"]], on="date", how="left")
    out["cal_no_deal"] = out["cal_no_deal"].notna()

    out = out.sort_values(["child_asin", "date"])
    g = out.groupby("child_asin", sort=False)

    # ASP discount screen. shift(1) keeps the reference median strictly in the
    # past so a deep discount cannot mask itself by moving its own baseline.
    asp_med = g["asp"].transform(lambda x: x.shift(1).rolling(56, min_periods=14).median())
    out["asp_ratio"] = out["asp"] / asp_med
    out["asp_discount"] = out["asp_ratio"] < 0.93

    # Robust velocity spike screen (MAD-scaled, trailing only).
    u = out["units_ordered"]
    med = g["units_ordered"].transform(lambda x: x.shift(1).rolling(56, min_periods=14).median())
    mad = g["units_ordered"].transform(
        lambda x: x.shift(1).rolling(56, min_periods=14).apply(
            lambda w: np.median(np.abs(w - np.median(w))), raw=True
        )
    )
    scale = (1.4826 * mad).replace(0, np.nan)
    out["velocity_z"] = (u - med) / scale
    k = float(cfg.raw["clean_days"]["spike_mad_threshold"])
    out["velocity_spike"] = out["velocity_z"] > k

    out["is_promo_day"] = (
        out["deal_flag_src"]
        | out["asp_discount"].fillna(False).astype(bool)
        | out["velocity_spike"].fillna(False).astype(bool)
    )
    # An organic day is a non-promo day on which the listing was actually live.
    out["is_organic_day"] = ~out["is_promo_day"] & out["was_reported"]
    return out


def add_launch_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Days-since-launch and maturity state."""
    out = panel.sort_values(["child_asin", "date"]).copy()
    first = out.groupby("child_asin")["date"].transform("min")
    out["launch_date"] = first
    out["days_since_launch"] = (out["date"] - first).dt.days
    out["weeks_since_launch"] = out["days_since_launch"] // 7
    out["month_since_launch"] = out["days_since_launch"] // 30
    out["is_mature"] = out["days_since_launch"] >= MATURITY_DAYS
    out["log1p_dsl"] = np.log1p(out["days_since_launch"])
    return out


def add_family_depth(panel: pd.DataFrame, sku: pd.DataFrame) -> pd.DataFrame:
    """Family depth *as of each date*: how many sibling listings in the same
    (program, shade_family, size) were already live.

    This is the cannibalisation / diminishing-returns mechanic. ``family_rank``
    is the listing's own entry order into its family (1 = first in), and
    ``family_live_count`` is how many siblings are competing on that date.
    Both are computed from launch dates only, so they are known at order time
    for a planned new variant.
    """
    keys = ["program", "shade_family", "size"]
    meta = sku[["child_asin", *keys]].drop_duplicates("child_asin")
    out = panel.merge(meta, on="child_asin", how="left")

    launches = (
        out.groupby(["child_asin", *keys], as_index=False)["launch_date"].min()
        .sort_values([*keys, "launch_date"])
    )
    launches["family_rank"] = launches.groupby(keys).cumcount() + 1
    out = out.merge(launches[["child_asin", "family_rank"]], on="child_asin", how="left")

    # How many siblings were live on each date.
    live = launches[[*keys, "child_asin", "launch_date"]]
    counts = []
    for key_vals, grp in out.groupby(keys, sort=False):
        sib = live
        for k, v in zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)):
            sib = sib[sib[k] == v]
        launch_dates = np.sort(sib["launch_date"].values)
        n_live = np.searchsorted(launch_dates, grp["date"].values, side="right")
        counts.append(pd.Series(n_live, index=grp.index))
    out["family_live_count"] = pd.concat(counts).sort_index()

    # Same logic one level up: depth of the whole shade family across sizes.
    prog_keys = ["program", "shade_family"]
    fam_launch = (
        out.groupby(["child_asin", *prog_keys], as_index=False)["launch_date"].min()
        .sort_values([*prog_keys, "launch_date"])
    )
    fam_launch["family_rank_any_size"] = fam_launch.groupby(prog_keys).cumcount() + 1
    out = out.merge(fam_launch[["child_asin", "family_rank_any_size"]],
                    on="child_asin", how="left")
    return out


def build(raw: dict[str, pd.DataFrame], cfg: Config, save: bool = True) -> pd.DataFrame:
    sku = annotate(raw["sku_dim"])
    panel = raw["daily_panel"].copy()
    panel["date"] = pd.to_datetime(panel["date"])

    panel = densify(panel)
    panel = add_promo_flags(panel, raw["asin_deal_days"], raw["clean_days"], cfg)
    panel = add_launch_features(panel)
    panel = add_family_depth(panel, sku)

    # Returns, aligned on the return date (used for net demand, not to shift demand).
    ret = raw["returns"].copy()
    ret["date"] = pd.to_datetime(ret["date"])
    panel = panel.merge(ret, on=["child_asin", "date"], how="left")
    panel[["returned_units", "colour_related_returns"]] = panel[
        ["returned_units", "colour_related_returns"]
    ].fillna(0.0)

    # Attach the SKU attributes needed downstream.
    attr_cols = [
        "child_asin", "program", "colour", "size", "status", "shade_family",
        "pattern_type", "base_colour", "lab_L", "lab_a", "lab_b", "dist_to_white",
        "lab_chroma", "lab_hue", "is_core_white", "is_near_white", "is_solid",
        "is_neutral", "n_colour_words",
    ]
    panel = panel.merge(
        sku[attr_cols].drop_duplicates("child_asin"),
        on="child_asin", how="left", suffixes=("", "_sku"),
    )
    for col in ["program", "shade_family", "size"]:
        dup = f"{col}_sku"
        if dup in panel.columns:
            panel[col] = panel[col].fillna(panel[dup])
            panel = panel.drop(columns=[dup])

    panel = panel.sort_values(["child_asin", "date"]).reset_index(drop=True)
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(PROCESSED / "panel.parquet", index=False)
        sku.to_parquet(PROCESSED / "sku_annotated.parquet", index=False)
        log.info("panel: %d rows -> %s", len(panel), PROCESSED / "panel.parquet")
    return panel


def main() -> None:
    from src.config import load_config
    from src.extract import extract

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    panel = build(extract(cfg), cfg)
    print(panel[["date", "child_asin", "units_ordered", "is_promo_day", "is_organic_day",
                 "days_since_launch", "family_rank", "family_live_count",
                 "shade_family", "size"]].tail(8).to_string(index=False))
    print(f"\nrows={len(panel):,}  asins={panel.child_asin.nunique()}  "
          f"dates={panel.date.min().date()}..{panel.date.max().date()}")
    print(f"organic days: {panel.is_organic_day.mean():.1%}  promo days: {panel.is_promo_day.mean():.1%}")


if __name__ == "__main__":
    main()
