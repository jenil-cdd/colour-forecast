"""Strict temporal split with isolated per-month horizons.

The pipeline forecasts two horizons separately rather than one 61-day block:

    Month 1  = days 0-29 of a listing's life
    Month 2  = days 30-59 of a listing's life

There is a real reason to separate them beyond tidiness. Month 2 runs at roughly
2.2-2.3x month 1 for Queen and King, but **3.0x for Twin** - the ramp shape is
not the same across sizes, so a single blended window hides it. Splitting also
lets the ensemble weight models differently by horizon, which is the whole point
of the dynamic stack: growth curves and the Bayesian hierarchy carry the cold
start, look-alike matching takes over as the listing establishes itself.

Training labels are built from the same historical launch cohorts, measured over
the matching slice of each cohort's life, so a month-2 model is trained on
month-2 behaviour rather than on a rescaled month-1 number.

Temporal safety: a cohort qualifies for training only if its whole label window
closes on or before ``train_end``. Features are always built as of the day
before that cohort launched. Both are asserted in tests/test_pipeline.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import Config
from src.features import build_asin_features

log = logging.getLogger(__name__)

#: (name, first day of life, last day of life)
HORIZONS: list[tuple[str, int, int]] = [("m1", 0, 29), ("m2", 30, 59)]


def horizon_bounds(name: str) -> tuple[int, int]:
    for n, a, b in HORIZONS:
        if n == name:
            return a, b
    raise KeyError(name)


def training_examples(panel: pd.DataFrame, sku: pd.DataFrame, cfg: Config,
                      horizon: str) -> pd.DataFrame:
    """Launch cohorts labelled over the requested slice of listing life."""
    lo, hi = horizon_bounds(horizon)
    tr_end = pd.Timestamp(cfg.split.train_end)
    span = hi - lo + 1

    launches = (panel.groupby("child_asin").launch_date.min()
                .rename("launch_date").reset_index())
    # The whole label window must close inside the training period.
    launches = launches[launches.launch_date + pd.Timedelta(days=hi) <= tr_end]

    frames = []
    for launch_date, grp in launches.groupby("launch_date"):
        asof = launch_date - pd.Timedelta(days=1)
        if panel[panel.date <= asof].empty:
            continue
        rows = build_asin_features(
            panel=panel, sku=sku, asof=asof,
            target_start=launch_date + pd.Timedelta(days=lo),
            target_end=launch_date + pd.Timedelta(days=hi),
        )
        rows = rows[rows.child_asin.isin(grp.child_asin)]
        if len(rows):
            frames.append(rows)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["y_units"] = out.y_units.fillna(0.0)
    out["horizon"] = horizon
    out["horizon_span"] = span
    # Launch wave = grouping unit for cross-validation, so same-wave listings
    # never straddle a fold boundary.
    out["wave"] = out.launch_date.dt.to_period("Q").astype(str)
    return out


def scoring_examples(panel: pd.DataFrame, sku: pd.DataFrame, cfg: Config,
                     test_start: str, test_end: str) -> pd.DataFrame:
    """Listings scored over one isolated calendar month.

    (Named ``scoring_examples`` rather than ``test_examples`` because pytest
    collects any importable callable whose name starts with ``test_``.)

    The information cut-off is pinned at ``train_end`` for both months: a buyer
    deciding in May has no more knowledge about July than about June. This is
    what makes month 2 a genuine 2-month-ahead forecast rather than a
    one-month-ahead forecast in disguise.
    """
    ts, te = pd.Timestamp(test_start), pd.Timestamp(test_end)
    rows = build_asin_features(
        panel=panel, sku=sku, asof=pd.Timestamp(cfg.split.train_end),
        target_start=ts, target_end=te,
    )
    live = panel[(panel.date >= ts) & (panel.date <= te)].child_asin.unique()
    rows = rows[rows.child_asin.isin(live)].copy()
    rows["y_units"] = rows.y_units.fillna(0.0)
    rows["is_cold_start"] = rows.launch_date >= pd.Timestamp(cfg.split.train_end) - pd.Timedelta(days=7)
    rows["wave"] = "test"

    # Which slice of life the window actually covers, used to pick the horizon
    # model and to sanity-check the ramp.
    age_at_start = (ts - rows.launch_date).dt.days.clip(lower=0)
    rows["age_at_window_start"] = age_at_start
    rows["horizon"] = np.where(age_at_start < 30, "m1", "m2")
    return rows
