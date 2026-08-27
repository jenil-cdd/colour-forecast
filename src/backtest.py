"""Backtest harness: train on <= 2026-05-31, score on 2026-06-01..2026-07-31.

Framing
-------
The decision is taken *before* a listing exists, so training examples are
**launch cohorts**, not calendar snapshots. For each historical listing we build
one example:

    as-of      = launch_date - 1 day        (information cut-off)
    features   = colour, size, program, family state as of that date
    label      = units sold in the first ``window`` days of the listing's life

The test examples are the 2026 launches, with the information cut-off pinned at
``train_end`` and the label taken from the actual Jun-Jul 2026 window. Because
features are rebuilt at each example's own as-of date, a model can never see a
sibling launch that had not happened yet.

Two evaluation cuts are reported:

* **cold-start cohort** — listings launched at/after the train/test boundary.
  This is the real question: an entirely new colour with no sales history.
* **all active listings** — every listing live in the window, which shows
  whether a model that wins on cold-start is also sane on the mature book.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.config import OUTPUTS, Config, load_config
from src.features import build_asin_features, design_matrix
from src.metrics import evaluate
from src.models import REGISTRY

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Models that need the daily panel to fit a ramp shape.
NEEDS_PANEL = {"gompertz", "logistic", "bass", "ensemble", "heuristic"}


def build_training_examples(panel: pd.DataFrame, sku: pd.DataFrame, cfg: Config,
                            window: int) -> pd.DataFrame:
    """One example per historical launch, features as of the day before launch."""
    tr_end = pd.Timestamp(cfg.split.train_end)
    launches = (panel.groupby("child_asin").launch_date.min()
                .rename("launch_date").reset_index())
    # Label must lie wholly inside the training period.
    launches = launches[launches.launch_date + pd.Timedelta(days=window - 1) <= tr_end]

    frames = []
    for launch_date, grp in launches.groupby("launch_date"):
        asof = launch_date - pd.Timedelta(days=1)
        hist = panel[panel.date <= asof]
        if hist.empty:
            continue
        rows = build_asin_features(
            panel=panel, sku=sku, asof=asof,
            target_start=launch_date,
            target_end=launch_date + pd.Timedelta(days=window - 1),
        )
        rows = rows[rows.child_asin.isin(grp.child_asin)]
        if len(rows):
            frames.append(rows)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["y_units"] = out.y_units.fillna(0.0)
    return out


def build_test_examples(panel: pd.DataFrame, sku: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """All listings scored over the held-out Jun-Jul 2026 window."""
    s = cfg.split
    rows = build_asin_features(
        panel=panel, sku=sku,
        asof=pd.Timestamp(s.train_end),
        target_start=pd.Timestamp(s.test_start),
        target_end=pd.Timestamp(s.test_end),
    )
    # Keep listings that were actually live at some point in the window.
    live = panel[(panel.date >= pd.Timestamp(s.test_start))
                 & (panel.date <= pd.Timestamp(s.test_end))].child_asin.unique()
    rows = rows[rows.child_asin.isin(live)].copy()
    rows["y_units"] = rows.y_units.fillna(0.0)
    rows["is_cold_start"] = rows.launch_date >= pd.Timestamp(s.train_end) - pd.Timedelta(days=7)
    return rows


def run(cfg: Config | None = None, models: list[str] | None = None,
        save: bool = True) -> dict:
    cfg = cfg or load_config()
    panel = pd.read_parquet("data/processed/panel.parquet")
    sku = pd.read_parquet("data/processed/sku_annotated.parquet")
    window = cfg.split.horizon_days

    log.info("building training examples (launch cohorts, %d-day window)...", window)
    train = build_training_examples(panel, sku, cfg, window)
    test = build_test_examples(panel, sku, cfg)
    log.info("train examples=%d  test listings=%d (cold-start=%d)",
             len(train), len(test), int(test.is_cold_start.sum()))

    # Require a minimum of observation before a listing can be a training label.
    train = train[train.y_days_live.fillna(0) >= 14]
    log.info("train examples after minimum-observation filter: %d", len(train))

    Xtr, cols = design_matrix(train, regime="cold")
    Xte, _ = design_matrix(test, regime="cold")
    Xte = Xte.reindex(columns=cols, fill_value=0.0)
    ytr = train.y_units.to_numpy(float)

    names = models or list(REGISTRY)
    results, preds, fitted = {}, {}, {}

    for nm in names:
        cls = REGISTRY[nm]
        kw = {"seed": cfg.seed}
        if nm in NEEDS_PANEL:
            kw["panel"] = panel
        try:
            model = cls(**kw).fit(train, Xtr, ytr)
            p = model.predict(test, Xte)
            preds[nm] = p
            fitted[nm] = model
        except Exception as exc:  # keep the suite running; report the failure
            log.warning("%-22s FAILED: %s", nm, exc)
            results[nm] = {"error": str(exc)}
            continue

        y = test.y_units.to_numpy(float)
        cold = test.is_cold_start.to_numpy(bool)
        row = evaluate(y, p.mean, p.lo, p.hi, prefix="all_")
        if cold.sum() > 1:
            row.update(evaluate(y[cold], p.mean[cold], p.lo[cold], p.hi[cold], prefix="cold_"))
        row["cold_n"] = int(cold.sum())
        row["all_n"] = int(len(y))
        row["pred_total_cold"] = float(p.mean[cold].sum())
        row["actual_total_cold"] = float(y[cold].sum())
        results[nm] = row
        log.info("%-22s cold WAPE=%.3f  all WAPE=%.3f  cold pred=%.0f vs actual=%.0f",
                 nm, row.get("cold_wape", np.nan), row["all_wape"],
                 row["pred_total_cold"], row["actual_total_cold"])

    lb = pd.DataFrame(results).T
    out = {"leaderboard": lb, "train": train, "test": test,
           "predictions": preds, "models": fitted}

    if save:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        lb.to_csv(OUTPUTS / "leaderboard.csv")
        detail = test[["child_asin", "program", "colour", "size", "shade_family",
                       "launch_date", "is_cold_start", "y_units"]].copy()
        for nm, p in preds.items():
            detail[f"pred_{nm}"] = p.mean
        detail.to_csv(OUTPUTS / "predictions.csv", index=False)
        log.info("wrote %s and %s", OUTPUTS / "leaderboard.csv", OUTPUTS / "predictions.csv")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = run()
    lb = res["leaderboard"]
    cols = [c for c in ["cold_wape", "cold_mae", "cold_bias", "cold_coverage80",
                        "all_wape", "all_bias", "pred_total_cold", "actual_total_cold"]
            if c in lb.columns]
    print("\n" + "=" * 96)
    print("LEADERBOARD — held-out 2026-06-01..2026-07-31 (sorted by cold-start WAPE)")
    print("=" * 96)
    print(lb[cols].sort_values("cold_wape" if "cold_wape" in cols else cols[0]).round(3).to_string())


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Rolling-origin validation
# ---------------------------------------------------------------------------
def find_launch_events(panel: pd.DataFrame, min_size: int = 3,
                       gap_days: int = 14) -> pd.DataFrame:
    """Group launches into *events*: batches of listings that went live together.

    This is how the business actually launches — a colour wave drops as a block,
    not one SKU at a time. The 2026 test cohort is one such event (17 listings
    over 2026-05-26..06-13), and the catalogue contains several earlier ones,
    including a near-identical 17-listing wave on 2025-05-26.

    Using events as validation origins keeps the rolling protocol analogous to
    the real test. Using calendar quarter-starts instead does not: it produced
    cohorts whose listings launched near the *end* of the window, leaving 2 days
    of exposure and 1 unit of demand, on which no accuracy metric is meaningful.
    """
    L = panel.groupby("child_asin").launch_date.min().rename("d").reset_index().sort_values("d")
    grp = (L.d.diff().dt.days.fillna(10**6) > gap_days).cumsum()
    ev = L.groupby(grp).agg(start=("d", "min"), end=("d", "max"), n=("child_asin", "size"))
    return ev[ev.n >= min_size].reset_index(drop=True)


def run_rolling(cfg: Config | None = None, models: list[str] | None = None,
                min_train: int = 20, min_test: int = 3, min_exposure: int = 40,
                min_cohort_units: int = 25, save: bool = True) -> pd.DataFrame:
    """Repeat the launch-cohort experiment at earlier launch events.

    The Jun-Jul 2026 holdout is a single window with 19 listings, which is thin
    evidence for choosing between 19 models — the ordering could easily be noise.
    This re-runs the identical protocol at earlier launch events: train on every
    launch that completed before the event, score the listings the event added.

    Cohorts are skipped when they cannot support a meaningful metric: fewer than
    ``min_exposure`` mean days live in the window, or fewer than
    ``min_cohort_units`` total units sold. Skipped cohorts are logged rather than
    silently dropped, since "we could not test here" is itself a finding.
    """
    cfg = cfg or load_config()
    panel = pd.read_parquet("data/processed/panel.parquet")
    sku = pd.read_parquet("data/processed/sku_annotated.parquet")
    window = cfg.split.horizon_days

    launches = panel.groupby("child_asin").launch_date.min().rename("launch_date").reset_index()
    events = find_launch_events(panel)
    names = models or list(REGISTRY)

    recs, skipped, detail = [], [], []
    for _, ev in events.iterrows():
        origin = pd.Timestamp(ev["start"])
        if origin > pd.Timestamp(cfg.split.test_start):
            continue
        tr_end = origin - pd.Timedelta(days=1)
        n_train_avail = int((launches.launch_date + pd.Timedelta(days=window - 1) < origin).sum())
        if n_train_avail < min_train:
            skipped.append((origin.date(), "too few completed prior launches", n_train_avail))
            continue

        sub = Config(raw={**cfg.raw, "split": {
            **cfg.raw["split"],
            "train_end": str(tr_end.date()),
            "test_start": str(origin.date()),
            "test_end": str((origin + pd.Timedelta(days=window - 1)).date()),
        }})
        train = build_training_examples(panel, sku, sub, window)
        train = train[train.y_days_live.fillna(0) >= 14]
        test = build_test_examples(panel, sku, sub)
        test = test[test.is_cold_start]

        if len(train) < min_train or len(test) < min_test:
            skipped.append((origin.date(), "cohort too small", len(test)))
            continue
        if test.exposure_days.mean() < min_exposure:
            skipped.append((origin.date(), f"mean exposure {test.exposure_days.mean():.0f}d", len(test)))
            continue
        if test.y_units.sum() < min_cohort_units:
            skipped.append((origin.date(), f"cohort sold only {test.y_units.sum():.0f} units", len(test)))
            continue

        Xtr, cols = design_matrix(train, regime="cold")
        Xte, _ = design_matrix(test, regime="cold")
        Xte = Xte.reindex(columns=cols, fill_value=0.0)
        ytr, yte = train.y_units.to_numpy(float), test.y_units.to_numpy(float)

        for nm in names:
            kw = {"seed": cfg.seed}
            if nm in NEEDS_PANEL:
                kw["panel"] = panel[panel.date <= tr_end]
            try:
                m = REGISTRY[nm](**kw).fit(train, Xtr, ytr)
                p = m.predict(test, Xte)
                row = evaluate(yte, p.mean, p.lo, p.hi)
                row.update({"model": nm, "origin": origin.date(), "n_test": len(test),
                            "n_train": len(train), "cohort_units": float(yte.sum()),
                            "pred_units": float(p.mean.sum())})
                recs.append(row)
                detail.append(pd.DataFrame({
                    "model": nm, "origin": origin.date(),
                    "child_asin": test.child_asin.to_numpy(),
                    "colour": test.colour.to_numpy(), "size": test["size"].to_numpy(),
                    "y_units": yte, "pred": p.mean, "lo": p.lo, "hi": p.hi,
                }))
            except Exception as exc:
                log.debug("%s @ %s failed: %s", nm, origin.date(), exc)
        log.info("origin %s  train=%d  test=%d  cohort_units=%.0f  mean_exposure=%.0fd",
                 origin.date(), len(train), len(test), yte.sum(), test.exposure_days.mean())

    for o, why, n in skipped:
        log.info("skipped origin %s: %s (n=%s)", o, why, n)

    df = pd.DataFrame(recs)
    det = pd.concat(detail, ignore_index=True) if detail else pd.DataFrame()
    if save and len(df):
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUTS / "rolling_origin.csv", index=False)
        det.to_csv(OUTPUTS / "rolling_detail.csv", index=False)
        log.info("wrote %s and %s", OUTPUTS / "rolling_origin.csv", OUTPUTS / "rolling_detail.csv")
    return df


def summarise_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Median WAPE and rank stability across origins."""
    if df.empty:
        return df
    df = df.copy()
    df["rank"] = df.groupby("origin").wape.rank()
    return (df.groupby("model")
              .agg(origins=("origin", "nunique"),
                   wape_median=("wape", "median"),
                   wape_mean=("wape", "mean"),
                   wape_iqr=("wape", lambda s: s.quantile(.75) - s.quantile(.25)),
                   mean_rank=("rank", "mean"),
                   worst_rank=("rank", "max"),
                   spearman_median=("spearman", "median"),
                   top6_hit=("top6_hit", "mean"),
                   coverage80=("coverage80", "mean"),
                   bias_median=("bias", "median"))
              .sort_values("mean_rank"))
