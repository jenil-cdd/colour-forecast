"""Rolling-origin validation across historical launch cohorts."""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest import run_rolling, summarise_rolling  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 200)
    df = run_rolling()
    s = summarise_rolling(df)
    print("\n" + "=" * 110)
    print("ROLLING-ORIGIN VALIDATION — same protocol at earlier origins (sorted by mean rank)")
    print("=" * 110)
    print(s.round(3).to_string())
    s.to_csv("data/outputs/rolling_summary.csv")
