"""Assemble the findings report from saved backtest / rolling / recommendation outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest import summarise_rolling  # noqa: E402
from src.config import OUTPUTS, REPORTS, load_config  # noqa: E402
from src.metrics import newsvendor_cost  # noqa: E402


def fractile_sensitivity(rec: pd.DataFrame, cfg) -> pd.DataFrame:
    """What each order fractile would have cost, against what actually sold.

    This is the decision-quality check that accuracy metrics cannot give: an
    order is good or bad only relative to the asymmetric cost of being wrong.
    """
    from src.recommend import _total_calibrator

    horizon = int(cfg.business["horizon_days"])
    conv = 1.249  # fitted ramp conversion, 61d -> 120d (see recommend.ramp_conversion)
    actual_120 = ((rec.actual_units_in_test_window / rec.actual_exposure_days)
                  * horizon * conv).to_numpy(float)
    point = rec.forecast_120d.to_numpy(float)
    cal = _total_calibrator("knn_lookalike", before_origin=cfg.split.test_start)
    weights = point / point.sum()

    holding = float(cfg.business["holding_cost_ratio"])
    stockout = float(cfg.business["stockout_cost_ratio"])

    rows = []
    for tau in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        total = float(cal.quantile(np.array([point.sum()]), tau)[0])
        order = total * weights
        rows.append({
            "fractile": tau,
            "total_order": round(total),
            "vs_actual": round(total / actual_120.sum(), 2),
            "units_short": round(float(np.clip(actual_120 - order, 0, None).sum())),
            "units_excess": round(float(np.clip(order - actual_120, 0, None).sum())),
            "cost_index": round(newsvendor_cost(actual_120, order, holding=holding,
                                                stockout=stockout)),
        })
    df = pd.DataFrame(rows)
    # Also score the perfect-hindsight order and the incumbent heuristic.
    df.attrs["actual_total"] = float(actual_120.sum())
    df.attrs["perfect_cost"] = newsvendor_cost(actual_120, actual_120,
                                               holding=holding, stockout=stockout)
    heur = np.full_like(actual_120, 2180 / len(actual_120))
    df.attrs["heuristic_cost"] = newsvendor_cost(actual_120, heur,
                                                 holding=holding, stockout=stockout)
    return df


def main() -> None:
    cfg = load_config()
    pd.set_option("display.width", 220)
    REPORTS.mkdir(parents=True, exist_ok=True)

    lb = pd.read_csv(OUTPUTS / "leaderboard.csv", index_col=0)
    roll = pd.read_csv(OUTPUTS / "rolling_origin.csv")
    rec = pd.read_csv(OUTPUTS / "order_recommendation_knn_lookalike.csv")
    summary = summarise_rolling(roll)

    sens = fractile_sensitivity(rec, cfg)

    lines: list[str] = []
    A = lines.append
    A("# Findings — duvet cover set colour/size cold-start forecasting\n")
    A(f"Training data through **{cfg.split.train_end}**; held out "
      f"**{cfg.split.test_start} .. {cfg.split.test_end}**.\n")

    A("\n## 1. Model leaderboard, single held-out window\n")
    cols = [c for c in ["cold_wape", "cold_mae", "cold_bias", "cold_spearman",
                        "cold_top6_hit", "cold_coverage80"] if c in lb.columns]
    A("```")
    A(lb[cols].sort_values(cols[0]).round(3).to_string())
    A("```")

    A("\n## 2. Rolling-origin validation across 7 launch events\n")
    A("```")
    A(summary.round(3).to_string())
    A("```")

    A("\n## 3. Order fractile sensitivity vs realised demand\n")
    A(f"Actual 120-day-equivalent cohort demand: **{sens.attrs['actual_total']:,.0f} units**. "
      f"Cost index uses stockout:holding = "
      f"{cfg.business['stockout_cost_ratio']}:{cfg.business['holding_cost_ratio']}.\n")
    A("```")
    A(sens.to_string(index=False))
    A("```")
    A(f"\n- perfect-hindsight cost index: {sens.attrs['perfect_cost']:,.0f}")
    A(f"- incumbent heuristic (2,180 units, flat split): {sens.attrs['heuristic_cost']:,.0f}")

    A("\n## 4. Recommended order sheet\n")
    A("```")
    A(rec.round(1).to_string(index=False))
    A("```")

    out = REPORTS / "FINDINGS.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    print("\n=== fractile sensitivity ===")
    print(sens.to_string(index=False))
    print(f"\nactual 120d-equivalent demand : {sens.attrs['actual_total']:,.0f}")
    print(f"perfect-hindsight cost index  : {sens.attrs['perfect_cost']:,.0f}")
    print(f"heuristic (2,180u) cost index : {sens.attrs['heuristic_cost']:,.0f}")


if __name__ == "__main__":
    main()
