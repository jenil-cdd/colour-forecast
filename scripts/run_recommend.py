"""Generate the order recommendation sheet."""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUTS, load_config  # noqa: E402
from src.recommend import recommend  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 250)
    cfg = load_config()
    model = sys.argv[1] if len(sys.argv) > 1 else "knn_lookalike"
    scope = sys.argv[2] if len(sys.argv) > 2 else "duvet_all"
    out = recommend(cfg, model_name=model, scope=scope)
    a = out.attrs
    print(f"\nORDER RECOMMENDATION — {cfg.business['horizon_days']}-day horizon, "
          f"model={model}, scope={scope}")
    print(f"newsvendor critical fractile = {a['fractile']:.2f} "
          f"(stockout:holding = {cfg.business['stockout_cost_ratio']}:{cfg.business['holding_cost_ratio']})")
    print(f"ramp conversion {cfg.split.horizon_days}d -> {cfg.business['horizon_days']}d "
          f"velocity = {a['ramp_conversion']:.3f}x")
    print("=" * 165)
    print(out.round(2).to_string(index=False))
    print("=" * 165)
    print(f"expected (point) demand over horizon : {a['total_point']:>8,.0f} units")
    print(f"TOTAL RECOMMENDED ORDER              : {a['total_order']:>8,.0f} units"
          f"   <- portfolio-sized at p{a['fractile']*100:.0f}, allocated pro-rata")
    print(f"  of which reallocation reserve       : {a['reserve_units']:>8,.0f} units")
    print(f"if each SKU were sized independently : {a['total_if_per_sku']:>8,.0f} units"
          f"   <- over-buys diversifiable SKU risk {len(out)}x")
    print(f"incumbent heuristic recommendation   : {2180:>8,d} units")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUTS / f"order_recommendation_{model}.csv", index=False)
    print(f"\nwrote {OUTPUTS / f'order_recommendation_{model}.csv'}")
