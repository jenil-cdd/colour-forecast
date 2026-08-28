"""120-day order sheet for the 5 approved colours, Duvet Cover Set only.

Colour velocity, ramp shape and interval width are all estimated strictly from
DCS history. Size ratios are the one exception and are flagged in the output:
DCS has two usable size-ratio cohorts and neither contains a Twin, so a
DCS-only ratio would collapse to a fixed prior rather than an estimate.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUTS, load_config  # noqa: E402
from src.features import design_matrix  # noqa: E402
from src.horizons import training_examples  # noqa: E402
from src.intervals import ConformalInterval  # noqa: E402
from src.models import REGISTRY  # noqa: E402
from src.prep import blind_size, neutralise, recency_weight, size_index  # noqa: E402
from src.recommend import _synthesise, ramp_conversion, return_rates  # noqa: E402
from src.stacking import DynamicStack  # noqa: E402

#: Model choice needs BOTH per-line accuracy and colour resolution.
#: size_ratio had the best DCS-only worst-month error (0.592) but returns only
#: 3 distinct values across the 5 candidate colours - Cream and Greige come out
#: identical, and so do Olive and Terracotta - because it resolves no finer than
#: shade family. hier_bayes and matrix_factorisation have the same defect.
#: Among models that separate all 5 colours, `ensemble` has the best DCS-only
#: worst-month per-line error (0.718). knn_lookalike is reported alongside as a
#: sensitivity: it spreads colours more widely (3.0x vs 2.4x max/min).
MODEL = "ensemble"
ALT_MODEL = "knn_lookalike"


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 250)
    cfg = load_config()
    asof = pd.Timestamp(cfg.split.train_end)
    horizon = int(cfg.business["horizon_days"])
    focal = cfg.focal_program

    panel_all = pd.read_parquet("data/processed/panel.parquet")
    sku_all = pd.read_parquet("data/processed/sku_annotated.parquet")
    panel = panel_all[panel_all.program == focal].copy()
    sku = sku_all[sku_all.program == focal].copy()
    sizes = cfg.sizes["recommend"]

    # Size ratios: structural, borrowed across programmes (see module docstring).
    idx = size_index(panel_all, asof, sorted(panel_all["size"].astype(str).unique()),
                     age_lo=30, age_hi=59)
    idx_dcs = size_index(panel, asof, sorted(panel["size"].astype(str).unique()),
                         age_lo=30, age_hi=59)

    # ---- fit on DCS month-2 cohorts ---------------------------------------
    tr = training_examples(panel, sku, cfg, "m2")
    tr = tr[tr.y_days_live.fillna(0) >= 7]
    tr, ytr = neutralise(tr, idx)
    trm = blind_size(tr)
    Xtr, cols = design_matrix(trm, drop_size=True)
    sw = recency_weight(tr.launch_date, asof)

    def _fit(name):
        kw = {"seed": cfg.seed}
        if name in {"gompertz", "logistic", "bass", "ensemble", "heuristic"}:
            kw["panel"] = panel
        return REGISTRY[name](**kw).fit(trm, Xtr, ytr)

    model = _fit(MODEL)
    alt = _fit(ALT_MODEL)

    # ---- interval calibrated on DCS out-of-fold residuals ------------------
    stack = DynamicStack(seed=cfg.seed).fit(trm, Xtr, ytr, panel=panel, sample_weight=sw)
    w = stack.weights_[stack.weights_ > 0]
    oof = sum(stack.oof_[m].fillna(0.0) * wt for m, wt in w.items()).to_numpy(float)
    icfg = cfg.raw.get("intervals", {})
    ci = ConformalInterval(nominal=0.80,
                           floor_ratio=float(icfg.get("floor_ratio", 0.35)),
                           cap_ratio=float(icfg.get("cap_ratio", 3.0)))
    ci.fit(tr.y_units.to_numpy(float), oof * tr.size_index.to_numpy(float),
           groups=tr["size"].astype(str).to_numpy())
    ci90 = ConformalInterval(nominal=0.90, floor_ratio=float(icfg.get("floor_ratio", 0.35)),
                             cap_ratio=float(icfg.get("cap_ratio", 3.0)))
    ci90.fit(tr.y_units.to_numpy(float), oof * tr.size_index.to_numpy(float),
             groups=tr["size"].astype(str).to_numpy())

    # ---- candidates -------------------------------------------------------
    cand = pd.read_csv("config/candidates.csv")
    rows = _synthesise(cand[["colour", "size"]], panel, sku, cfg, asof)
    rows = rows[rows["size"].isin(sizes)].copy()
    rows, _ = neutralise(rows, idx)
    rowsm = blind_size(rows)
    X, _ = design_matrix(rowsm, drop_size=True)
    X = X.reindex(columns=cols, fill_value=0.0)

    si = rows.size_index.to_numpy(float)
    conv = ramp_conversion(panel, asof, from_days=cfg.split.horizon_days, to_days=horizon)
    expo = rows.exposure_days.clip(lower=1).to_numpy(float)
    point = (model.predict(rowsm, X).mean * si / expo) * horizon * conv
    point_alt = (alt.predict(rowsm, X).mean * si / expo) * horizon * conv

    grp = rows["size"].astype(str).to_numpy()
    lo80, hi80 = ci.apply(point, groups=grp)
    lo90, hi90 = ci90.apply(point, groups=grp)

    # ---- returns ----------------------------------------------------------
    fam_size, pat, glob = return_rates(panel, asof)
    rows = rows.merge(fam_size, on=["shade_family", "size"], how="left").merge(
        pat, on="pattern_type", how="left")
    rr = (rows.rr_family_size.fillna(rows.rr_family)
              .fillna(rows.rr_pattern).fillna(glob)).to_numpy(float)
    net = 1.0 + rr * 0.20

    merch = cand.drop_duplicates("colour").set_index("colour")["merch_family"]
    out = pd.DataFrame({
        "colour": rows.colour.to_numpy(), "size": rows["size"].to_numpy(),
        "merch_family": rows.colour.map(merch).to_numpy(),
        "family_depth": rows.family_live_count.to_numpy(),
        "forecast_120d": np.round(point, 0),
        "p80_lo": np.round(lo80, 0), "p80_hi": np.round(hi80, 0),
        "p90_hi": np.round(hi90, 0),
        "return_rate": np.round(rr, 3),
        "order_p80": np.round(hi80 * net, 0),
        "order_p90": np.round(hi90 * net, 0),
        f"forecast_{ALT_MODEL}": np.round(point_alt, 0),
    })

    print("=" * 140)
    print(f"120-DAY ORDER SHEET — {focal} ONLY  |  model={MODEL} "
          f"(alt={ALT_MODEL})  |  DCS training cohorts={len(tr)}")
    print(f"ramp conversion {cfg.split.horizon_days}d -> {horizon}d = {conv:.3f}x  |  "
          f"interval: DCS conformal, {ci}")
    print("=" * 140)
    print("\nSIZE RATIOS (Queen = 1.00)")
    cmp = pd.DataFrame({"all_programmes_used": idx.reindex(sizes).round(3),
                        "dcs_only_would_give": idx_dcs.reindex(sizes).round(3)})
    print(cmp.to_string())
    print("  DCS alone has 2 usable size-ratio cohorts and no Twin, so the")
    print("  all-programme estimate is used for the size split only.\n")
    print(out.sort_values("order_p90", ascending=False).to_string(index=False))

    for tag, col in [("p80", "order_p80"), ("p90", "order_p90")]:
        pv = out.pivot_table(index=["colour", "merch_family"], columns="size",
                             values=col, aggfunc="sum").reindex(columns=sizes)
        pv["TOTAL"] = pv.sum(axis=1)
        print(f"\n--- ORDER BY COLOUR ({tag}) ---")
        print(pv.sort_values("TOTAL", ascending=False).astype(int).to_string())

    print(f"\n{'='*140}")
    print(f"expected 120-day organic demand : {out.forecast_120d.sum():>8,.0f} units")
    print(f"ORDER at p80                    : {out.order_p80.sum():>8,.0f} units "
          f"(+{out.order_p80.sum()/out.forecast_120d.sum()-1:.0%})")
    print(f"ORDER at p90                    : {out.order_p90.sum():>8,.0f} units "
          f"(+{out.order_p90.sum()/out.forecast_120d.sum()-1:.0%})")
    print(f"{'='*140}")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUTS / "FINAL_order_sheet_dcs.csv", index=False)
    print(f"wrote {OUTPUTS / 'FINAL_order_sheet_dcs.csv'}")


if __name__ == "__main__":
    main()
