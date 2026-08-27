"""Turn forecasts into an order recommendation.

Three things happen here that a point forecast alone cannot do:

1. **Horizon conversion.** Models are validated on a 61-day window; the buy
   covers 120 days from launch. Because a new listing is still climbing its ramp
   throughout, average daily velocity over 120 days is *higher* than over the
   first 61. The fitted Gompertz ramp supplies the conversion factor, so the
   growth model earns its keep here even though it was not the most accurate
   point forecaster.

2. **Net demand.** Returns are subtracted at family-specific rates measured from
   the data, not the brief's assumed levels. This matters for the headline pick:
   Near White is both the fastest-selling family *and* the second-highest return
   family (17.5%), so its gross and net rankings differ.

3. **Asymmetric order sizing.** Stocking out in month 3 damages search rank,
   which is worth more than the carrying cost of an equivalent overstock. The
   order therefore sits at a newsvendor critical fractile above the median
   forecast, drawn from the wave-aware calibrated predictive distribution rather
   than from a flat multiplier.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.calibrate import WaveAwareCalibrator, calibrators_from_rolling
from src.size_structure import size_weights
from src.config import OUTPUTS, Config, load_config
from src.features import build_asin_features, design_matrix
from src.models import REGISTRY
from src.models.growth import GompertzRamp

log = logging.getLogger(__name__)


def ramp_conversion(panel: pd.DataFrame, asof: pd.Timestamp,
                    from_days: int, to_days: int) -> float:
    """Ratio of mean realised velocity over ``to_days`` vs ``from_days`` of life."""
    g = GompertzRamp(panel=panel)
    g._fit_shape(asof)
    a = g.window_share(np.array([0.0]), np.array([float(from_days - 1)]))[0]
    b = g.window_share(np.array([0.0]), np.array([float(to_days - 1)]))[0]
    return float(b / a) if a > 0 else 1.0


def return_rates(panel: pd.DataFrame, asof: pd.Timestamp,
                 lookback_days: int = 365) -> pd.DataFrame:
    """Return rate by (shade_family, size), with hierarchical fallbacks.

    Shrunk toward the family mean and then the global mean using a simple
    Beta-Binomial posterior mean, so a family with 40 units of history does not
    get a 30% point estimate off two returns.
    """
    h = panel[(panel.date <= asof) & (panel.date > asof - pd.Timedelta(days=lookback_days))]
    glob = h.returned_units.sum() / max(h.units_ordered.sum(), 1)

    def _rate(keys):
        g = h.groupby(keys).agg(u=("units_ordered", "sum"), r=("returned_units", "sum"))
        # Beta-Binomial shrinkage with an effective prior sample of 200 units.
        k = 200.0
        g["rate"] = (g.r + k * glob) / (g.u + k)
        return g["rate"]

    fam_size = _rate(["shade_family", "size"]).rename("rr_family_size").reset_index()
    fam = _rate(["shade_family"]).rename("rr_family").reset_index()
    pat = _rate(["pattern_type"]).rename("rr_pattern").reset_index()
    fam_size["rr_global"] = glob
    return fam_size.merge(fam, on="shade_family", how="outer"), pat, float(glob)


def recommend(cfg: Config | None = None, model_name: str = "knn_lookalike",
              candidates: pd.DataFrame | None = None,
              resaleable_share: float = 0.80,
              scope: str = "duvet_all",
              size_structure: str = "era_controlled",
              fractiles: list[float] | None = None) -> pd.DataFrame:
    """Produce the order sheet.

    ``candidates`` is a frame of planned variants with columns ``colour``,
    ``size`` and optionally ``program``. When omitted, the 2026 launch cohort is
    used, which makes the output directly checkable against what actually sold.
    """
    cfg = cfg or load_config()
    panel = pd.read_parquet("data/processed/panel.parquet")
    sku = pd.read_parquet("data/processed/sku_annotated.parquet")
    biz = cfg.business
    asof = pd.Timestamp(cfg.split.train_end)
    horizon = int(biz["horizon_days"])
    window = cfg.split.horizon_days

    # ---- fit the champion on all launch cohorts up to the cut-off -----------
    from src.backtest import build_training_examples, build_test_examples

    train = build_training_examples(panel, sku, cfg, window)
    train = train[train.y_days_live.fillna(0) >= 14]
    Xtr, cols = design_matrix(train, regime="cold")
    kw = {"seed": cfg.seed}
    if model_name in {"gompertz", "logistic", "bass", "ensemble", "heuristic"}:
        kw["panel"] = panel
    model = REGISTRY[model_name](**kw).fit(train, Xtr, train.y_units.to_numpy(float))

    # ---- candidate variants ------------------------------------------------
    test = build_test_examples(panel, sku, cfg)
    rows = test[test.is_cold_start].copy() if candidates is None else _synthesise(
        candidates, panel, sku, cfg, asof)
    if scope == "focal":
        # The order decision is for the focal program in the three sizes that
        # are actually being bought; Oversized King is carried in the panel for
        # ratio estimation only.
        rows = rows[(rows.program == cfg.focal_program)
                    & rows["size"].isin(cfg.sizes["recommend"])]
    elif scope == "duvet_all":
        rows = rows[rows["size"].isin(cfg.sizes["recommend"])]
    if rows.empty:
        raise ValueError("no candidate variants to score")

    X, _ = design_matrix(rows, regime="cold")
    X = X.reindex(columns=cols, fill_value=0.0)
    pred_window = model.predict(rows, X).mean

    # ---- window -> 120-day horizon ----------------------------------------
    conv = ramp_conversion(panel, asof, from_days=window, to_days=horizon)
    # Rows were scored on their own exposure; put everything on a per-day basis
    # first, then extend to the full horizon.
    per_day = pred_window / rows.exposure_days.clip(lower=1).to_numpy(float)
    gross_120 = per_day * horizon * conv

    # ---- impose an era-controlled size split -------------------------------
    # The demand model is trusted for *colour* (its validated strength) but not
    # for the split across sizes. Twin listings only exist from 2024-03-26 while
    # Queen/King cohorts run back to 2019, and the 2024+ era is ~2.1x stronger,
    # so a model fitted on pooled launch cohorts reads that era gap as a Twin
    # size effect. Left uncorrected it put Twin at 53% of the order sheet and
    # ranked Sage Twin above Sage Queen, which Sage Green's own launch
    # (Queen 453 / King 394 / Twin 263) directly contradicts.
    # Colour-level totals are preserved; only the within-colour split changes.
    if size_structure == "era_controlled":
        w = size_weights(panel, asof, sorted(set(rows["size"].astype(str))),
                         horizon=horizon)
        df = pd.DataFrame({"colour": rows.colour.to_numpy(),
                           "size": rows["size"].astype(str).to_numpy(),
                           "g": gross_120})
        df["w"] = df["size"].map(w).fillna(w.mean())
        colour_total = df.groupby("colour").g.transform("sum")
        w_total = df.groupby("colour").w.transform("sum")
        gross_120 = (colour_total * df.w / w_total.clip(lower=1e-9)).to_numpy(float)

    # ---- calibrated predictive quantiles -----------------------------------
    try:
        det = pd.read_csv(OUTPUTS / "rolling_detail.csv")
        cal = calibrators_from_rolling(det, model_name, cfg.split.test_start, wave_aware=True)
    except FileNotFoundError:
        cal = WaveAwareCalibrator()
        log.warning("rolling_detail.csv not found; intervals fall back to a wide prior")

    # ---- order fractile from the cost structure, not a chosen "service level"
    # Newsvendor critical fractile = Cu / (Cu + Co). With a month-3 stock-out
    # costing 4x an equivalent overstock, that is 0.80 -- not the 0.95 the
    # config nominally asks for. At the measured log-residual spread (SD ~1.05)
    # a 0.95 fractile implies ordering 4.4x the point forecast, which is not a
    # real policy; 0.80 implies 1.58x, which is.
    holding = float(biz.get("holding_cost_ratio", 0.25))
    stockout = float(biz.get("stockout_cost_ratio", 1.0))
    fractile = stockout / (stockout + holding)

    q_med = cal.quantile(gross_120, 0.50)
    q_lo = cal.quantile(gross_120, 0.10)
    q_hi = cal.quantile(gross_120, 0.90)
    q_sku = cal.quantile(gross_120, fractile)

    # ---- returns -> net demand --------------------------------------------
    fam_size, pat, glob = return_rates(panel, asof)
    rows = rows.merge(fam_size, on=["shade_family", "size"], how="left")
    rows = rows.merge(pat, on="pattern_type", how="left")
    rr = (rows.rr_family_size.fillna(rows.rr_family)
              .fillna(rows.rr_pattern).fillna(glob)).to_numpy(float)

    # Returned units that come back saleable re-enter supply, so only the
    # unsaleable fraction has to be replaced by the order.
    net_multiplier = 1.0 + rr * (1.0 - resaleable_share)

    # ---- two-tier sizing ---------------------------------------------------
    # Measured on seven launch events, cohort-TOTAL demand is far more
    # predictable than any individual SKU (WAPE 0.20 vs 0.69; log-residual SD
    # 0.60 vs 1.05). Buying each SKU at its own p80 therefore over-buys the
    # portfolio badly, because it pays for diversifiable SKU-level risk 19
    # times over. So the *total* is sized from the total's own distribution and
    # then allocated pro-rata on the point forecasts, with the difference
    # between the two held back as a reallocation reserve.
    total_point = float(gross_120.sum())
    tot_cal = _total_calibrator(model_name, before_origin=cfg.split.test_start)
    total_order = float(tot_cal.quantile(np.array([total_point]), fractile)[0])
    weights = gross_120 / max(gross_120.sum(), 1e-9)
    allocated = total_order * weights * net_multiplier
    sku_independent = q_sku * net_multiplier

    # Order quantities at each requested fractile, so the capital trade-off can
    # be weighed directly rather than inferred from an interval.
    ladder = {}
    for tau in sorted(set(fractiles or [fractile])):
        tot = float(tot_cal.quantile(np.array([total_point]), tau)[0])
        ladder[tau] = tot * weights * net_multiplier

    out = pd.DataFrame({
        "colour": rows.colour.to_numpy(),
        "size": rows["size"].to_numpy(),
        "program": rows.program.to_numpy(),
        "shade_family": rows.shade_family.to_numpy(),
        "family_depth_entered": rows.family_live_count.to_numpy(),
        "forecast_120d": np.round(gross_120, 0),
        "p10": np.round(q_lo, 0),
        "p50": np.round(q_med, 0),
        "p90": np.round(q_hi, 0),
        "return_rate": np.round(rr, 3),
        "recommended_order": np.round(allocated, 0),
        "order_if_sized_per_sku": np.round(sku_independent, 0),
    })
    for tau, vals in ladder.items():
        out[f"order_p{int(round(tau * 100))}"] = np.round(vals, 0)
    if "y_units" in rows.columns:
        out["actual_units_in_test_window"] = rows.y_units.to_numpy()
        out["actual_exposure_days"] = rows.exposure_days.to_numpy()

    out = out.sort_values("recommended_order", ascending=False).reset_index(drop=True)
    out.attrs["fractile"] = fractile
    out.attrs["total_point"] = total_point
    out.attrs["total_order"] = float(allocated.sum())
    out.attrs["total_if_per_sku"] = float(sku_independent.sum())
    out.attrs["ramp_conversion"] = conv
    out.attrs["reserve_units"] = float(allocated.sum()) - float(gross_120.sum())
    out.attrs["size_structure"] = size_structure
    out.attrs["ladder_totals"] = {t_: float(v.sum()) for t_, v in ladder.items()}
    return out


def _total_calibrator(model_name: str, before_origin=None) -> WaveAwareCalibrator:
    """Calibrator fitted on *cohort-total* residuals rather than SKU residuals.

    ``before_origin`` must be supplied so calibration uses only launch events
    that had already happened at decision time. Without it the calibration would
    include the very window being forecast.
    """
    c = WaveAwareCalibrator()
    try:
        det = pd.read_csv(OUTPUTS / "rolling_detail.csv")
    except FileNotFoundError:
        return c
    d = det[det.model == model_name]
    if before_origin is not None:
        d = d[pd.to_datetime(d.origin) < pd.Timestamp(before_origin)]
    if d.empty:
        return c
    agg = d.groupby("origin").agg(y=("y_units", "sum"), p=("pred", "sum")).reset_index()
    return c.fit_store(agg.y.to_numpy(float), agg.p.to_numpy(float),
                       origins=agg.origin.to_numpy())


def _synthesise(candidates: pd.DataFrame, panel: pd.DataFrame, sku: pd.DataFrame,
                cfg: Config, asof: pd.Timestamp) -> pd.DataFrame:
    """Build feature rows for variants that do not exist in the catalogue yet.

    A hypothetical variant has no ASIN and no panel history, so it is injected
    into the SKU dimension with a synthetic id, given the planned launch date,
    and then run through the same feature builder as a real listing. This is how
    a genuinely new colour (an unentered family such as Terracotta) gets scored.
    """
    from src.taxonomy import annotate

    c = candidates.copy()
    c["program"] = c.get("program", cfg.focal_program)
    c["child_asin"] = ["CAND_" + str(i) for i in range(len(c))]
    c = annotate(c)

    launch = pd.Timestamp(cfg.split.test_start)
    stub = pd.DataFrame({
        "child_asin": c.child_asin, "date": launch,
        "units_ordered": 0.0, "sessions": 0.0, "is_organic_day": True,
        "is_promo_day": False, "launch_date": launch, "days_since_launch": 0,
        "returned_units": 0.0, "colour_related_returns": 0.0,
        "family_rank": np.nan, "family_rank_any_size": np.nan,
    })
    for col in panel.columns:
        if col not in stub.columns:
            stub[col] = np.nan
    stub["program"] = c.program.to_numpy()
    stub["shade_family"] = c.shade_family.to_numpy()
    stub["size"] = c["size"].to_numpy()

    big_panel = pd.concat([panel, stub[panel.columns]], ignore_index=True)
    big_sku = pd.concat([sku, c.reindex(columns=sku.columns)], ignore_index=True)

    rows = build_asin_features(
        panel=big_panel, sku=big_sku, asof=asof,
        target_start=launch,
        target_end=launch + pd.Timedelta(days=cfg.split.horizon_days - 1),
    )
    rows = rows[rows.child_asin.isin(c.child_asin)].copy()
    rows["exposure_days"] = cfg.split.horizon_days
    rows["target_days"] = cfg.split.horizon_days

    # Within-cohort family depth. The feature builder counts siblings that were
    # already live, which misses the fact that candidates in the same shade
    # family are launching *against each other*. With two Near-Whites in this
    # wave, the second entrant must see the first ahead of it or the
    # diminishing-returns mechanic is silently switched off for exactly the case
    # it was built for. Order within a family follows the approved list order.
    rows = rows.merge(c[["child_asin"]].assign(cohort_order=range(len(c))),
                      on="child_asin", how="left").sort_values("cohort_order")
    within = rows.groupby(["program", "shade_family", "size"]).cumcount()
    rows["cohort_family_rank"] = within + 1
    rows["family_live_count"] = rows["family_live_count"] + within
    rows["family_rank"] = rows["family_rank"].fillna(0) + rows["family_live_count"]
    rows["family_rank_any_size"] = rows["family_rank_any_size"].fillna(
        rows["family_live_count"])
    return rows.drop(columns=["cohort_order"])
