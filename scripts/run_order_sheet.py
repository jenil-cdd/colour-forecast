"""Final order sheet for the approved candidate colours.

Runs the champion model over the approved candidate list, converts the validated
61-day window forecast to the 120-day buy horizon via the fitted ramp, strips
PPC-attributed demand from the baseline, and sizes the order at the newsvendor
critical fractile implied by the approved 9:1 stockout:holding ratio (p90).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUTS, load_config  # noqa: E402
from src.recommend import recommend  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 250)
    cfg = load_config()
    model = sys.argv[1] if len(sys.argv) > 1 else "knn_lookalike"

    cand = pd.read_csv("config/candidates.csv")
    out = recommend(cfg, model_name=model, candidates=cand[["colour", "size"]], scope="focal")
    a = out.attrs

    merch = cand.drop_duplicates("colour").set_index("colour")["merch_family"]
    out["merch_family"] = out.colour.map(merch)

    fractile = a["fractile"]
    print("=" * 132)
    print(f"FINAL 120-DAY ORDER SHEET — {cfg.focal_program}")
    print(f"model={model} (champion, 7-event rolling mean rank 4.0) | "
          f"demand basis={cfg.raw['target']['demand']} (PPC-attributed units stripped)")
    print(f"cost ratio stockout:holding = "
          f"{cfg.business['stockout_cost_ratio']/cfg.business['holding_cost_ratio']:.0f}:1  "
          f"-> newsvendor critical fractile p{fractile*100:.0f}")
    print(f"ramp conversion 61d -> 120d velocity = {a['ramp_conversion']:.3f}x | "
          f"size split = {a['size_structure']}")
    print("=" * 132)

    from src.size_structure import report as size_report
    panel = pd.read_parquet("data/processed/panel.parquet")
    print("\nSIZE SPLIT (era-controlled; Twin listings only exist from 2024-03-26,")
    print("and the 2024+ launch era is ~2.1x stronger, so pooled ratios are confounded)")
    print(size_report(panel, pd.Timestamp(cfg.split.train_end),
                      cfg.sizes["recommend"]).to_string(index=False))
    print()

    cols = ["colour", "merch_family", "size", "shade_family", "family_depth_entered",
            "forecast_120d", "p10", "p50", "p90", "return_rate", "recommended_order"]
    print(out[cols].to_string(index=False))

    print("\n" + "-" * 132)
    print("BREAKDOWN BY COLOUR")
    print("-" * 132)
    by_c = (out.pivot_table(index=["colour", "merch_family"], columns="size",
                            values="recommended_order", aggfunc="sum")
               .reindex(columns=["Queen", "King", "Twin"]))
    by_c["TOTAL"] = by_c.sum(axis=1)
    by_c = by_c.sort_values("TOTAL", ascending=False)
    print(by_c.astype(int).to_string())

    print("\n" + "-" * 132)
    print("BREAKDOWN BY SIZE")
    print("-" * 132)
    by_s = (out.groupby("size").recommended_order.sum()
               .reindex(["Queen", "King", "Twin"]))
    tot = by_s.sum()
    for s, v in by_s.items():
        print(f"  {s:6s} {v:>7,.0f} units   ({v/tot:>5.1%})")
    print(f"  {'TOTAL':6s} {tot:>7,.0f} units")

    print("\n" + "=" * 132)
    print(f"expected (point) 120-day organic demand : {a['total_point']:>8,.0f} units")
    print(f"TOTAL RECOMMENDED ORDER at p{fractile*100:.0f}         : {a['total_order']:>8,.0f} units")
    print(f"  safety buffer over expected demand    : {a['reserve_units']:>8,.0f} units "
          f"(+{a['reserve_units']/a['total_point']:.0%})")
    print(f"incumbent heuristic recommendation      : {2180:>8,d} units")
    print("=" * 132)

    # ---- fractile ladder + reality check -----------------------------------
    import numpy as np

    from src.metrics import newsvendor_cost
    from src.recommend import _total_calibrator

    cal = _total_calibrator(model, before_origin=cfg.split.test_start)
    point = out.forecast_120d.to_numpy(float)
    h = float(cfg.business["holding_cost_ratio"])
    so = float(cfg.business["stockout_cost_ratio"])

    print("\n" + "-" * 132)
    print("FRACTILE LADDER — the buffer is a choice; this is what each level buys")
    print("-" * 132)
    print(f"{'fractile':>9} {'total order':>12} {'buffer vs expected':>20} {'per colour':>12}")
    for tau in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        tot = float(cal.quantile(np.array([point.sum()]), tau)[0])
        mark = "  <-- APPROVED (9:1)" if abs(tau - fractile) < 1e-9 else ""
        print(f"{tau:>9.2f} {tot:>12,.0f} {tot/point.sum()-1:>19.0%} "
              f"{tot/out.colour.nunique():>12,.0f}{mark}")

    print("\n" + "-" * 132)
    print("REALITY CHECK — against the closest real analogue we have")
    print("-" * 132)
    panel2 = pd.read_parquet("data/processed/panel.parquet")
    sg = panel2[(panel2.colour == "Sage Green") & (panel2.days_since_launch <= 119)
                & panel2["size"].isin(["Queen", "King", "Twin"])]
    sg_actual = float(sg.groupby("child_asin").organic_units.sum().sum())
    n_col = out.colour.nunique()
    print(f"Sage Green actual first 120 days (Queen+King+Twin, launched 2025-05-26): "
          f"{sg_actual:,.0f} units")
    print(f"  if all {n_col} candidates matched that                              : "
          f"{sg_actual * n_col:,.0f} units")
    print(f"  model point forecast for all {n_col}                                : "
          f"{point.sum():,.0f} units  ({point.sum()/(sg_actual*n_col):.2f}x the Sage-Green-for-all case)")
    print(f"  approved p{fractile*100:.0f} order                                            : "
          f"{a['total_order']:,.0f} units  ({a['total_order']/(sg_actual*n_col):.2f}x)")
    print("\nThe p90 order deliberately sits above every demand scenario: at 9:1, one")
    print("stocked-out unit costs as much as nine carried units, so over-buying is correct.")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUTS / "FINAL_order_sheet.csv", index=False)
    by_c.to_csv(OUTPUTS / "FINAL_order_by_colour.csv")
    print(f"\nwrote {OUTPUTS / 'FINAL_order_sheet.csv'}")


if __name__ == "__main__":
    main()
