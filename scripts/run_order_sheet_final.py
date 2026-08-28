"""Final 120-day order sheet for the focal programme (config: programs.focal).

Model choice, interval calibration and colour baselines all come from the focal
programme only. Size ratios are the single borrowed parameter and are printed
alongside a focal-only comparison so the borrowing is visible.
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
from src.prep import blind_size, neutralise, size_index  # noqa: E402
from src.recommend import _synthesise, ramp_conversion, return_rates  # noqa: E402
from src.stacking import _grouped_folds  # noqa: E402

#: Best worst-month per-line error on the focal programme (0.423 June /
#: 0.555 July), and it resolves all five candidate colours distinctly.
MODEL = "knn_lookalike"
NEEDS_PANEL = {"gompertz", "logistic", "bass", "ensemble", "heuristic"}


def _fit(name, rows, X, y, panel, seed):
    kw = {"seed": seed}
    if name in NEEDS_PANEL:
        kw["panel"] = panel
    return REGISTRY[name](**kw).fit(rows, X, y)


def oof_predictions(name, rows, X, y, panel, seed, n_folds=5):
    """Out-of-fold predictions, folds grouped by launch wave."""
    waves = rows["wave"].astype(str).to_numpy()
    folds = _grouped_folds(waves, min(n_folds, max(2, len(pd.unique(waves)))), seed)
    out = np.full(len(rows), np.nan)
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        if tr.sum() < 8 or te.sum() == 0:
            continue
        try:
            m = _fit(name, rows[tr], X[tr], y[tr], panel, seed)
            out[te] = m.predict(rows[te], X[te]).mean
        except Exception:
            pass
    return out


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 250)
    cfg = load_config()
    asof = pd.Timestamp(cfg.split.train_end)
    horizon = int(cfg.business["horizon_days"])
    focal = cfg.focal_program
    sizes = cfg.sizes["recommend"]

    panel_all = pd.read_parquet("data/processed/panel.parquet")
    sku_all = pd.read_parquet("data/processed/sku_annotated.parquet")
    panel = panel_all[panel_all.program == focal].copy()
    sku = sku_all[sku_all.program == focal].copy()

    all_sizes = sorted(panel_all["size"].astype(str).unique())
    idx = size_index(panel_all, asof, all_sizes, age_lo=30, age_hi=59)
    idx_focal = size_index(panel, asof, sorted(panel["size"].astype(str).unique()),
                           age_lo=30, age_hi=59)

    # ---- fit on focal month-2 cohorts -------------------------------------
    tr = training_examples(panel, sku, cfg, "m2")
    tr = tr[tr.y_days_live.fillna(0) >= 7]
    tr, ytr = neutralise(tr, idx)
    trm = blind_size(tr)
    Xtr, cols = design_matrix(trm, drop_size=True)
    model = _fit(MODEL, trm, Xtr, ytr, panel, cfg.seed)

    # ---- intervals from THIS model's own out-of-fold residuals -------------
    oof = oof_predictions(MODEL, trm, Xtr, ytr, panel, cfg.seed)
    si_tr = tr.size_index.to_numpy(float)
    icfg = cfg.raw.get("intervals", {})
    cis = {}
    for nom in (0.80, 0.90):
        ci = ConformalInterval(nominal=nom,
                               floor_ratio=float(icfg.get("floor_ratio", 0.35)),
                               cap_ratio=float(icfg.get("cap_ratio", 3.0)))
        # Pooled across sizes deliberately, NOT per size. The size level is
        # already supplied by the era-controlled size index; fitting a separate
        # residual band per size re-imports the wave confound, because 400 TC's
        # 12 Twin cohorts come disproportionately from the strong May-2025 wave.
        # With per-size bands the Twin band came out at 4.61x against Queen's
        # 1.46x, which made Twin the *largest* line in the order sheet even
        # though its point forecast correctly sits below Queen.
        ci.fit(tr.y_units.to_numpy(float), np.nan_to_num(oof) * si_tr)
        cis[nom] = ci

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

    grp = rows["size"].astype(str).to_numpy()
    lo80, hi80 = cis[0.80].apply(point, groups=grp)
    lo90, hi90 = cis[0.90].apply(point, groups=grp)

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
        "family_cohorts_in_focal": rows.shade_family.map(
            tr.groupby("shade_family").size()).fillna(0).astype(int).to_numpy(),
        "forecast_120d": np.round(point, 0),
        "band_lo": np.round(lo80, 0),
        "order_p80": np.round(hi80 * net, 0),
        "order_p90": np.round(hi90 * net, 0),
        "return_rate": np.round(rr, 3),
    })

    print("=" * 136)
    print(f"FINAL 120-DAY ORDER SHEET — {focal} ONLY  |  model={MODEL}  |  "
          f"cohorts={len(tr)}  |  ramp {cfg.split.horizon_days}d->{horizon}d = {conv:.3f}x")
    print("=" * 136)
    print("\nSIZE RATIOS (Queen = 1.00)")
    print(pd.DataFrame({"used_all_programmes": idx.reindex(sizes).round(3),
                        "focal_only_would_give": idx_focal.reindex(sizes).round(3)}).to_string())
    print(f"\nINTERVAL (conformal on {MODEL} out-of-fold residuals, focal only)")
    for nom, ci in cis.items():
        print(f"  p{int(nom*100)}: {ci}")
    print()
    print(out.sort_values("order_p90", ascending=False).to_string(index=False))

    for tag, col in [("p80", "order_p80"), ("p90", "order_p90")]:
        pv = out.pivot_table(index=["colour", "merch_family"], columns="size",
                             values=col, aggfunc="sum").reindex(columns=sizes)
        pv["TOTAL"] = pv.sum(axis=1)
        print(f"\n--- ORDER BY COLOUR AND SIZE ({tag}) ---")
        print(pv.sort_values("TOTAL", ascending=False).astype(int).to_string())
        print(f"    size totals: " + "  ".join(f"{s}={pv[s].sum():,.0f}" for s in sizes)
              + f"  |  TOTAL={pv['TOTAL'].sum():,.0f}")

    print(f"\n{'='*136}")
    print(f"expected 120-day organic demand : {out.forecast_120d.sum():>8,.0f} units")
    print(f"ORDER at p80                    : {out.order_p80.sum():>8,.0f} units "
          f"(+{out.order_p80.sum()/out.forecast_120d.sum()-1:.0%})")
    print(f"ORDER at p90                    : {out.order_p90.sum():>8,.0f} units "
          f"(+{out.order_p90.sum()/out.forecast_120d.sum()-1:.0%})")
    print("=" * 136)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUTS / "FINAL_order_sheet_400tc.csv", index=False)
    print(f"wrote {OUTPUTS / 'FINAL_order_sheet_400tc.csv'}")


if __name__ == "__main__":
    main()
