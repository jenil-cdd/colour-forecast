"""Re-run the full evaluation for an arbitrary test window.

Usage: python3 scripts/run_window.py <test_start> <test_end> <label>

The window length drives the *training* label length too: a 30-day test is
scored against models trained on 30-day launch cohorts, so the per-day rate the
model learns is on the same footing as the rate it is asked to predict. Simply
truncating a 61-day forecast to 30 days would flatter or punish models
arbitrarily depending on how each handles the ramp.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.backtest import (build_test_examples, build_training_examples,  # noqa: E402
                          find_launch_events, summarise_rolling)
from src.config import OUTPUTS, Config, load_config  # noqa: E402
from src.features import design_matrix  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.models import REGISTRY  # noqa: E402

NEEDS_PANEL = {"gompertz", "logistic", "bass", "ensemble", "heuristic"}


def sub_config(cfg: Config, start: str, end: str, train_end: str | None = None) -> Config:
    s = dict(cfg.raw["split"])
    s["test_start"], s["test_end"] = start, end
    if train_end:
        s["train_end"] = train_end
    return Config(raw={**cfg.raw, "split": s})


def fit_and_score(panel, sku, cfg, min_obs=10):
    window = cfg.split.horizon_days
    train = build_training_examples(panel, sku, cfg, window)
    # Early origins can yield no completed prior launches at all, in which case
    # build_training_examples returns a bare empty frame with no columns.
    if train.empty or "y_days_live" not in train.columns:
        return None, None, None, None
    train = train[train.y_days_live.fillna(0) >= min_obs]
    test = build_test_examples(panel, sku, cfg)
    test = test[test.is_cold_start]
    if len(train) < 15 or len(test) < 3:
        return None, None, None, None

    Xtr, cols = design_matrix(train, regime="cold")
    Xte, _ = design_matrix(test, regime="cold")
    Xte = Xte.reindex(columns=cols, fill_value=0.0)
    ytr, yte = train.y_units.to_numpy(float), test.y_units.to_numpy(float)

    rows, preds = {}, {}
    for nm in REGISTRY:
        kw = {"seed": cfg.seed}
        if nm in NEEDS_PANEL:
            kw["panel"] = panel[panel.date <= pd.Timestamp(cfg.split.train_end)]
        try:
            m = REGISTRY[nm](**kw).fit(train, Xtr, ytr)
            p = m.predict(test, Xte)
            r = evaluate(yte, p.mean, p.lo, p.hi)
            r["pred_total"] = float(p.mean.sum())
            r["actual_total"] = float(yte.sum())
            rows[nm] = r
            preds[nm] = p
        except Exception as exc:
            logging.warning("%s failed: %s", nm, exc)
    return pd.DataFrame(rows).T, train, test, preds


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 260)
    start, end, label = sys.argv[1], sys.argv[2], sys.argv[3]

    cfg0 = load_config()
    panel = pd.read_parquet("data/processed/panel.parquet")
    sku = pd.read_parquet("data/processed/sku_annotated.parquet")
    cfg = sub_config(cfg0, start, end)
    window = cfg.split.horizon_days

    print(f"train <= {cfg.split.train_end}  |  test {start} .. {end}  ({window} days)")
    lb, train, test, preds = fit_and_score(panel, sku, cfg)
    print(f"training cohorts={len(train)}  test listings={len(test)}  "
          f"actual units={test.y_units.sum():.0f}\n")

    # ---- rolling replay on the same window length -------------------------
    events = find_launch_events(panel)
    recs, det = [], []
    for _, ev in events.iterrows():
        origin = pd.Timestamp(ev["start"])
        if origin > pd.Timestamp(start):
            continue
        c = sub_config(cfg0, str(origin.date()),
                       str((origin + pd.Timedelta(days=window - 1)).date()),
                       train_end=str((origin - pd.Timedelta(days=1)).date()))
        l2, tr2, te2, p2 = fit_and_score(panel, sku, c)
        if l2 is None or te2.exposure_days.mean() < window * 0.65 or te2.y_units.sum() < 15:
            continue
        for nm in l2.index:
            recs.append({**l2.loc[nm].to_dict(), "model": nm, "origin": origin.date(),
                         "n_test": len(te2)})
        print(f"  replay origin {origin.date()}: test={len(te2)} units={te2.y_units.sum():.0f}")

    roll = pd.DataFrame(recs)
    if len(roll):
        summ = summarise_rolling(roll)
        roll.to_csv(OUTPUTS / f"rolling_{label}.csv", index=False)
    else:
        summ = pd.DataFrame()

    lb.to_csv(OUTPUTS / f"leaderboard_{label}.csv")
    detail = test[["child_asin", "program", "colour", "size", "launch_date",
                   "exposure_days", "y_units"]].copy()
    for nm, p in preds.items():
        detail[f"pred_{nm}"] = p.mean.round(1)
    detail.to_csv(OUTPUTS / f"predictions_{label}.csv", index=False)

    print(f"\n=== {label.upper()} HELD-OUT: all models ===")
    cols = ["wape", "mae", "bias", "spearman", "coverage80", "pred_total", "actual_total"]
    print(lb[cols].sort_values("wape").round(3).to_string())
    if len(summ):
        print(f"\n=== {label.upper()} ROLLING REPLAY ({roll.origin.nunique()} waves) ===")
        print(summ[["origins", "wape_median", "mean_rank", "worst_rank",
                    "spearman_median", "coverage80", "bias_median"]].round(3).to_string())


if __name__ == "__main__":
    main()
