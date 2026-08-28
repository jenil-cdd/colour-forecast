"""Orchestrator: strict May-train / June-test / July-test pipeline.

Flow
----
1. Estimate the era-controlled size index as of ``train_end`` (data prep).
2. For each horizon (month 1, month 2):
   a. Build launch-cohort training examples labelled over that slice of life.
   b. Neutralise the label by size -> Queen-equivalent scale.
   c. Fit all 19 base models plus the dynamic stack on the same rows.
   d. Score on the isolated test month, de-normalising predictions by size.
3. Certify every model per task and emit the capability matrix.

Everything is fitted once, on data ending 31 May 2026, and scored twice. The
information cut-off for both test months is the same date, so month 2 is a
genuine two-months-ahead forecast rather than a one-month-ahead forecast with a
later label.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import OUTPUTS, Config, load_config
from src.features import design_matrix
from src.horizons import HORIZONS, scoring_examples, training_examples
from src.models import REGISTRY
from src.prep import (DEFAULT_HALFLIFE_DAYS, blind_size, neutralise,
                      recency_weight, size_index)
from src.sanity import capability_matrix, certify
from src.stacking import NEEDS_PANEL, DynamicStack

log = logging.getLogger(__name__)

TEST_MONTHS = [("June", "2026-06-01", "2026-06-30"), ("July", "2026-07-01", "2026-07-31")]


def _fit_base(name, rows, X, y, panel, seed):
    kw = {"seed": seed}
    if name in NEEDS_PANEL:
        kw["panel"] = panel
    return REGISTRY[name](**kw).fit(rows, X, y)


def run(cfg: Config | None = None, halflife: float = DEFAULT_HALFLIFE_DAYS,
        neutralise_size: bool = True, save: bool = True) -> dict:
    cfg = cfg or load_config()
    panel = pd.read_parquet("data/processed/panel.parquet")
    sku = pd.read_parquet("data/processed/sku_annotated.parquet")
    asof = pd.Timestamp(cfg.split.train_end)
    sizes = sorted(panel["size"].dropna().astype(str).unique())

    # One size index per horizon: Twin ramps ~3.0x from month 1 to month 2
    # against ~2.2x for Queen and King, so a single fixed ratio cannot serve
    # both months.
    idx_by_hz = {}
    for hz, lo, hi in HORIZONS:
        idx_by_hz[hz] = size_index(panel, asof, sizes, age_lo=lo, age_hi=hi)
        log.info("size index (%s): %s", hz,
                 {k: round(v, 3) for k, v in idx_by_hz[hz].items()})
    idx = idx_by_hz[HORIZONS[0][0]]

    results, preds, weights, tables = {}, {}, {}, {}

    for hz, lo, hi in HORIZONS:
        train = training_examples(panel, sku, cfg, hz)
        if train.empty:
            log.warning("no training examples for %s", hz)
            continue
        train = train[train.y_days_live.fillna(0) >= 7]
        idx_h = idx_by_hz[hz]

        if neutralise_size:
            train, ytr = neutralise(train, idx_h)
        else:
            train["size_index"] = 1.0
            ytr = train.y_units.to_numpy(float)
        sw = recency_weight(train.launch_date, asof, halflife)

        # Size-blind copies for fitting/predicting. The originals keep the true
        # size so certification can still check the size mix.
        train_m = blind_size(train) if neutralise_size else train
        Xtr, cols = design_matrix(train_m, regime="cold", drop_size=neutralise_size)
        log.info("%s: %d cohorts, label mean=%.1f (queen-equiv)", hz, len(train), ytr.mean())

        # Which calendar month corresponds to this horizon for the test cohort.
        month = "June" if hz == "m1" else "July"
        name, ts, te = next(m for m in TEST_MONTHS if m[0] == month)
        test = scoring_examples(panel, sku, cfg, ts, te)
        test = test[test.is_cold_start].copy()
        if neutralise_size:
            test, _ = neutralise(test, idx_h)
        else:
            test["size_index"] = 1.0
        test_m = blind_size(test) if neutralise_size else test
        Xte, _ = design_matrix(test_m, regime="cold", drop_size=neutralise_size)
        Xte = Xte.reindex(columns=cols, fill_value=0.0)
        yte = test.y_units.to_numpy(float)
        si_te = (test.size_index.to_numpy(float) if neutralise_size
                 else np.ones(len(test)))

        # ---- base models -------------------------------------------------
        for nm in REGISTRY:
            try:
                if nm == "heuristic":
                    # The incumbent is a benchmark, not a candidate: run it
                    # exactly as the business runs it, with its own hard-coded
                    # size ratios and an un-neutralised target. Blinding it
                    # would flatter it by handing over our size fix.
                    Xh, ch = design_matrix(train, regime="cold")
                    Xhe, _ = design_matrix(test, regime="cold")
                    Xhe = Xhe.reindex(columns=ch, fill_value=0.0)
                    m = _fit_base(nm, train, Xh, train.y_units.to_numpy(float),
                                  panel, cfg.seed)
                    p = m.predict(test, Xhe)
                else:
                    m = _fit_base(nm, train_m, Xtr, ytr, panel, cfg.seed)
                    p = m.predict(test_m, Xte)
                    p.mean, p.lo, p.hi = p.mean * si_te, p.lo * si_te, p.hi * si_te
                results[(name, nm)] = certify(yte, p, test,
                                              groups=np.full(len(test), name))
                preds[(name, nm)] = p.mean
            except Exception as exc:
                log.warning("%s / %s failed: %s", name, nm, exc)

        # ---- dynamic stack ----------------------------------------------
        stack = DynamicStack(seed=cfg.seed).fit(train_m, Xtr, ytr, panel=panel,
                                                sample_weight=sw)
        p = stack.predict(test_m, Xte)
        p.mean, p.lo, p.hi = p.mean * si_te, p.lo * si_te, p.hi * si_te
        results[(name, "STACK")] = certify(yte, p, test, groups=np.full(len(test), name))
        preds[(name, "STACK")] = p.mean
        weights[name] = stack.weight_table()

        det = test[["child_asin", "program", "colour", "size", "launch_date",
                    "exposure_days", "size_index", "y_units"]].copy()
        for nm in REGISTRY:
            if (name, nm) in preds:
                det[f"pred_{nm}"] = np.round(preds[(name, nm)], 1)
        det["pred_STACK"] = np.round(preds[(name, "STACK")], 1)
        det["stack_lo"] = np.round(p.lo, 1)
        det["stack_hi"] = np.round(p.hi, 1)
        tables[name] = det
        log.info("%s: actual=%.0f stack=%.0f", name, yte.sum(), p.mean.sum())

    res = pd.DataFrame(results).T
    res.index.names = ["month", "model"]
    if save:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        res.to_csv(OUTPUTS / "pipeline_results.csv")
        for k, v in tables.items():
            v.to_csv(OUTPUTS / f"pipeline_predictions_{k.lower()}.csv", index=False)
        for k, v in weights.items():
            v.to_csv(OUTPUTS / f"stack_weights_{k.lower()}.csv", index=False)
        pd.DataFrame([{"horizon": h, "size": k, "size_index": round(v, 4)}
                      for h, s in idx_by_hz.items() for k, v in s.items()]).to_csv(
            OUTPUTS / "size_index.csv", index=False)
    return {"results": res, "tables": tables, "weights": weights,
            "size_index": idx_by_hz}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 260)
    out = run()
    res = out["results"]
    for month in ["June", "July"]:
        sub = res.xs(month, level="month") if month in res.index.get_level_values(0) else None
        if sub is None:
            continue
        print(f"\n{'='*130}\n{month.upper()} 2026 — all models (trained to 2026-05-31)\n{'='*130}")
        cols = ["total_err", "line_err", "bias", "spearman", "coverage",
                "twin_over_queen", "certified_for"]
        print(sub[cols].sort_values("line_err").to_string())
        print(f"\nstack weights ({month}):")
        print(out["weights"][month].to_string(index=False))


if __name__ == "__main__":
    main()
