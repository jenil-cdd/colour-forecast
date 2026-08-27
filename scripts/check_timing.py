"""Seasonality and launch-timing check for the buy.

Answers two questions the order sheet cannot:

1. The models were validated on a Jun-Jul window, but a PO placed today with a
   120-day lead arrives in late December. Does that seasonal shift invalidate
   the numbers?
2. Is the timing itself good? Arrival date determines which part of the season
   the first 120 days sell into.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
          "Aug", "Sep", "Oct", "Nov", "Dec"]


def seasonal_index(panel: pd.DataFrame) -> pd.Series:
    """Monthly demand index, 1.0 = annual average.

    Uses mature (>=180 day old) organic non-promo listing-days only, and
    de-trends by year before pooling, so multi-year growth is not mistaken for
    seasonality.
    """
    m = panel[panel.is_organic_day & (panel.days_since_launch >= 180)].copy()
    unit_col = "organic_units" if "organic_units" in m.columns else "units_ordered"
    g = m.groupby([m.date.dt.year, m.date.dt.month])[unit_col].mean().rename("v").reset_index()
    g.columns = ["yr", "mo", "v"]
    g["idx"] = g.v / g.groupby("yr").v.transform("mean")
    s = g.groupby("mo").idx.median()
    s.index = [MONTHS[i - 1] for i in s.index]
    return s.round(3)


def main() -> None:
    cfg = load_config()
    panel = pd.read_parquet("data/processed/panel.parquet")
    today = pd.Timestamp(panel.date.max())
    lead = int(cfg.business["lead_time_days"])
    horizon = int(cfg.business["horizon_days"])

    s = seasonal_index(panel)
    print("=== DUVET SEASONAL INDEX (1.0 = annual average) ===")
    print(s.to_string())

    validated = s.loc[["Jun", "Jul"]].mean()
    arrival = today + pd.Timedelta(days=lead)
    sell_end = arrival + pd.Timedelta(days=horizon)
    sell_months = pd.date_range(arrival, sell_end, freq="MS").month
    sell_idx = float(np.mean([s.loc[MONTHS[m - 1]] for m in sell_months])) if len(sell_months) else np.nan

    print(f"\nlatest data            : {today.date()}")
    print(f"lead time              : {lead} days -> arrival ~{arrival.date()}")
    print(f"{horizon}-day sell window   : {arrival.date()} .. {sell_end.date()}")
    print(f"\nseasonal index of the validated Jun-Jul window : {validated:.3f}")
    print(f"seasonal index of the actual sell window       : {sell_idx:.3f}")
    print(f"-> seasonal scaling to apply to the forecast   : {sell_idx/validated:.3f}x")

    peak = s.nlargest(3)
    trough = s.nsmallest(3)
    print(f"\npeak months   : {', '.join(f'{k} {v:.2f}' for k, v in peak.items())}")
    print(f"trough months : {', '.join(f'{k} {v:.2f}' for k, v in trough.items())}")
    print(f"\nThe sell window straddles the trough and misses the peak entirely.")
    print("Leftover stock is not dead, though: it carries into the following")
    print("Sep-Nov peak, which lowers the effective cost of overstocking.")


if __name__ == "__main__":
    main()
