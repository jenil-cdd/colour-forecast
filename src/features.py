"""ASIN-level design matrices for cold-start forecasting.

The decision being modelled is: *given a colour and size we have never sold
before, how many units will it move over the next 120 days?* That decision is
taken at order time, so the default feature regime ("cold") may use only what is
knowable before the listing goes live:

* colour attributes (taxonomy + Lab coordinates)
* size
* program
* the state of the shade family the variant enters (depth, sibling velocity)
* calendar position of the launch (seasonality)

A second regime ("warm") additionally uses the first *k* observed days of the
listing's own sales. That is the post-launch re-forecast — useful for the
in-season correction, and it also quantifies how much the cold model gives up.

Targets are cumulative units over a window, because daily SKU volumes here are
0-4 units and daily point forecasts are dominated by Poisson noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Features available before a listing goes live.
COLD_NUMERIC = [
    "lab_L", "lab_a", "lab_b", "lab_chroma", "lab_hue", "dist_to_white",
    "n_colour_words", "family_rank", "family_rank_any_size", "family_live_count",
    "sibling_velocity", "sibling_velocity_same_size", "family_pool_velocity",
    "program_velocity", "size_share_prior", "launch_month_sin", "launch_month_cos",
    "n_siblings_ever", "days_since_family_first_launch",
]
COLD_BOOL = ["is_core_white", "is_near_white", "is_solid", "is_neutral"]
COLD_CATEGORICAL = ["size", "program", "shade_family", "pattern_type"]

WARM_NUMERIC = [
    "obs_units_total", "obs_units_per_day", "obs_sessions_per_day",
    "obs_conversion", "obs_trend_slope", "obs_zero_share",
]


def _seasonal(dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    ang = 2 * np.pi * (dates.dt.dayofyear / 365.25)
    return np.sin(ang), np.cos(ang)


def sibling_state(panel: pd.DataFrame, asof: pd.Timestamp,
                  lookback_days: int = 90) -> pd.DataFrame:
    """Organic velocity of already-live listings, as of a cut-off date.

    Everything here is computed strictly from ``date <= asof`` so it is legal to
    use as a feature for a listing launching after ``asof``.
    """
    hist = panel[(panel.date <= asof) & (panel.date > asof - pd.Timedelta(days=lookback_days))]
    org = hist[hist.is_organic_day]

    def _vel(keys: list[str], name: str) -> pd.DataFrame:
        g = org.groupby(keys).agg(u=("units_ordered", "sum"), d=("units_ordered", "size"))
        return (g["u"] / g["d"].clip(lower=1)).rename(name).reset_index()

    return {
        "family_size": _vel(["program", "shade_family", "size"], "sibling_velocity_same_size"),
        "family": _vel(["program", "shade_family"], "sibling_velocity"),
        "pool": _vel(["shade_family"], "family_pool_velocity"),
        "program": _vel(["program"], "program_velocity"),
        "size": _vel(["program", "size"], "size_velocity"),
    }


def size_share_prior(panel: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Share of a program's organic volume taken by each size, as of ``asof``.

    Used as a prior for sizes with no history in the focal program (the Twin
    problem: the focal program has no mature Twin listings at all, so its Twin
    share has to be borrowed from sibling programs).
    """
    hist = panel[(panel.date <= asof) & panel.is_organic_day]
    g = hist.groupby(["program", "size"]).units_ordered.sum().rename("u").reset_index()
    tot = g.groupby("program").u.transform("sum")
    g["size_share_prior"] = g.u / tot.clip(lower=1)

    # Cross-program fallback for sizes a program has never sold.
    glob = hist.groupby("size").units_ordered.sum()
    glob = (glob / glob.sum()).rename("size_share_global").reset_index()
    return g[["program", "size", "size_share_prior"]].merge(glob, on="size", how="outer")


def build_asin_features(panel: pd.DataFrame, sku: pd.DataFrame, asof: pd.Timestamp,
                        target_start: pd.Timestamp, target_end: pd.Timestamp,
                        warm_days: int = 0, target: str = "organic") -> pd.DataFrame:
    """One row per ASIN whose launch is known by ``asof``.

    ``asof`` is the information cut-off. ``target_start``..``target_end`` is the
    window whose cumulative units become the label. ``warm_days`` > 0 admits the
    listing's own first *k* days of sales as features (post-launch regime).
    """
    launches = panel.groupby("child_asin").agg(
        launch_date=("launch_date", "min"),
        last_date=("date", "max"),
    ).reset_index()

    attrs = sku.drop_duplicates("child_asin").set_index("child_asin")
    state = sibling_state(panel, asof)
    shares = size_share_prior(panel, asof)

    rows = launches.merge(
        attrs[[c for c in [
            "program", "colour", "size", "shade_family", "pattern_type", "base_colour",
            "lab_L", "lab_a", "lab_b", "lab_chroma", "lab_hue", "dist_to_white",
            "is_core_white", "is_near_white", "is_solid", "is_neutral", "n_colour_words",
        ] if c in attrs.columns]].reset_index(),
        on="child_asin", how="inner",
    )

    # Family state carried on the panel (depth counters as of each date).
    depth = (panel.sort_values("date").groupby("child_asin")
             .agg(family_rank=("family_rank", "first"),
                  family_rank_any_size=("family_rank_any_size", "first")).reset_index())
    rows = rows.merge(depth, on="child_asin", how="left")

    # Family depth as it stands at the information cut-off, not at launch: for a
    # variant we are about to order, this is the competitive field it enters.
    live = launches[launches.launch_date <= asof].merge(
        attrs[["program", "shade_family", "size"]].reset_index(), on="child_asin", how="left")
    live_cnt = (live.groupby(["program", "shade_family", "size"]).size()
                .rename("family_live_count").reset_index())
    sib_ever = (live.groupby(["program", "shade_family"]).size()
                .rename("n_siblings_ever").reset_index())
    fam_first = (live.groupby(["program", "shade_family"]).launch_date.min()
                 .rename("family_first_launch").reset_index())

    rows = (rows.merge(live_cnt, on=["program", "shade_family", "size"], how="left")
                .merge(sib_ever, on=["program", "shade_family"], how="left")
                .merge(fam_first, on=["program", "shade_family"], how="left"))
    rows["family_live_count"] = rows.family_live_count.fillna(0)
    rows["n_siblings_ever"] = rows.n_siblings_ever.fillna(0)
    rows["days_since_family_first_launch"] = (
        (asof - rows.family_first_launch).dt.days.fillna(0).clip(lower=0)
    )

    for df, keys in [
        (state["family_size"], ["program", "shade_family", "size"]),
        (state["family"], ["program", "shade_family"]),
        (state["pool"], ["shade_family"]),
        (state["program"], ["program"]),
    ]:
        rows = rows.merge(df, on=keys, how="left")

    rows = rows.merge(shares, on=["program", "size"], how="left")
    rows["size_share_prior"] = rows.size_share_prior.fillna(rows.size_share_global)

    # Launch seasonality.
    s, c = _seasonal(rows["launch_date"])
    rows["launch_month_sin"], rows["launch_month_cos"] = s, c

    # ---- label: cumulative units in the target window ----------------------
    # ``target`` selects the demand definition. "organic" strips PPC-attributed
    # units so the fitted baseline reflects organic colour appeal rather than
    # advertising support; "total" keeps gross demand.
    tgt = panel[(panel.date >= target_start) & (panel.date <= target_end)]
    unit_col = "organic_units" if (target == "organic" and "organic_units" in tgt.columns) else "units_ordered"
    lab = tgt.groupby("child_asin").agg(
        y_units=(unit_col, "sum"),
        y_units_total=("units_ordered", "sum"),
        y_days_live=("units_ordered", "size"),
        y_promo_days=("is_promo_day", "sum"),
    ).reset_index()
    if "ad_units_7d" in tgt.columns:
        adl = tgt.groupby("child_asin").agg(y_ad_units=("ad_units_7d", "sum"),
                                            y_ad_spend=("ad_spend", "sum")).reset_index()
        lab = lab.merge(adl, on="child_asin", how="left")
    rows = rows.merge(lab, on="child_asin", how="left")

    # ---- warm-start features: the listing's own first k days ---------------
    if warm_days > 0:
        obs = panel.merge(rows[["child_asin", "launch_date"]], on="child_asin", how="inner")
        obs = obs[(obs.date <= asof) &
                  ((obs.date - obs.launch_date_y if "launch_date_y" in obs else
                    obs.date - obs.launch_date).dt.days < warm_days)]
        key = "launch_date_y" if "launch_date_y" in obs.columns else "launch_date"
        obs = obs[(obs.date - obs[key]).dt.days >= 0]
        w = obs.groupby("child_asin").agg(
            obs_units_total=("units_ordered", "sum"),
            obs_days=("units_ordered", "size"),
            obs_sessions=("sessions", "sum"),
            obs_zero_share=("units_ordered", lambda s: float((s == 0).mean())),
        ).reset_index()
        w["obs_units_per_day"] = w.obs_units_total / w.obs_days.clip(lower=1)
        w["obs_sessions_per_day"] = w.obs_sessions / w.obs_days.clip(lower=1)
        w["obs_conversion"] = w.obs_units_total / w.obs_sessions.replace(0, np.nan)

        def _slope(g: pd.DataFrame) -> float:
            if len(g) < 7:
                return 0.0
            x = np.arange(len(g), dtype=float)
            y = g.sort_values("date").units_ordered.to_numpy(dtype=float)
            return float(np.polyfit(x, y, 1)[0])

        sl = obs.groupby("child_asin").apply(_slope, include_groups=False).rename("obs_trend_slope")
        w = w.merge(sl.reset_index(), on="child_asin", how="left")
        rows = rows.merge(w, on="child_asin", how="left")

    rows["asof"] = asof
    rows["target_start"], rows["target_end"] = target_start, target_end
    rows["window_days"] = (target_end - target_start).days + 1

    # Exposure, not window length. A listing that goes live part-way through the
    # window can only sell for the days it is actually live, so that overlap is
    # what every model must be given as its offset. Using the full window length
    # instead attributes 61 days of selling to a listing that had 9, which shows
    # up as a large spurious over-forecast. The launch date is planned in
    # advance, so this is legitimately known at order time.
    eff_start = rows["launch_date"].clip(lower=target_start)
    rows["exposure_days"] = ((target_end - eff_start).dt.days + 1).clip(lower=0)
    rows["target_days"] = rows["exposure_days"]
    return rows


def design_matrix(rows: pd.DataFrame, regime: str = "cold") -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode and assemble the numeric design matrix."""
    num = list(COLD_NUMERIC)
    if regime == "warm":
        num += [c for c in WARM_NUMERIC if c in rows.columns]
    num = [c for c in num if c in rows.columns]

    X = rows[num].astype(float).copy()
    for c in COLD_BOOL:
        if c in rows.columns:
            X[c] = rows[c].astype(float)
    for c in COLD_CATEGORICAL:
        if c in rows.columns:
            d = pd.get_dummies(rows[c].astype(str), prefix=c, dtype=float)
            X = pd.concat([X, d], axis=1)

    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(0.0), list(X.columns)
