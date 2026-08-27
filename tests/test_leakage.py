"""Temporal-leakage guards.

The whole result rests on features at time t using only data from <= t. These
tests fail loudly if that ever stops being true.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.features import build_asin_features, sibling_state


@pytest.fixture(scope="module")
def data():
    p = Path("data/processed/panel.parquet")
    if not p.exists():
        pytest.skip("panel not built")
    return (pd.read_parquet(p),
            pd.read_parquet("data/processed/sku_annotated.parquet"))


def test_sibling_state_ignores_future(data):
    """Sibling velocity as of a date must not move when future rows are added."""
    panel, _ = data
    asof = pd.Timestamp("2025-01-01")
    past_only = panel[panel.date <= asof]
    a = sibling_state(panel, asof)["family"]
    b = sibling_state(past_only, asof)["family"]
    merged = a.merge(b, on=["program", "shade_family"], suffixes=("_full", "_past"))
    assert (merged.sibling_velocity_full - merged.sibling_velocity_past).abs().max() < 1e-9


def test_family_depth_counts_only_prior_launches(data):
    """family_live_count must equal the number of siblings live at the cut-off,
    never including launches that had not happened yet."""
    panel, sku = data
    asof = pd.Timestamp("2026-05-31")
    rows = build_asin_features(panel, sku, asof=asof,
                              target_start=pd.Timestamp("2026-06-01"),
                              target_end=pd.Timestamp("2026-07-31"))

    launches = panel.groupby("child_asin").launch_date.min()
    meta = sku.drop_duplicates("child_asin").set_index("child_asin")
    live = [a for a in launches[launches <= asof].index if a in meta.index]
    truth = (meta.loc[live].groupby(["program", "shade_family", "size"]).size())

    for key, grp in rows.groupby(["program", "shade_family", "size"]):
        expected = int(truth.get(key, 0))
        assert (grp.family_live_count == expected).all(), (
            f"{key}: got {sorted(grp.family_live_count.unique())}, expected {expected}")


def test_exposure_never_exceeds_window(data):
    """Exposure must be the live overlap, capped by the window length."""
    panel, sku = data
    start, end = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-31")
    rows = build_asin_features(panel, sku, asof=pd.Timestamp("2026-05-31"),
                              target_start=start, target_end=end)
    window = (end - start).days + 1
    assert (rows.exposure_days <= window).all()
    assert (rows.exposure_days >= 0).all()
    late = rows[rows.launch_date > start]
    if len(late):
        # Listings that launch after the window closes get zero exposure, which
        # is why the expectation is clipped rather than allowed to go negative.
        expected = ((end - late.launch_date).dt.days + 1).clip(lower=0)
        assert (late.exposure_days == expected).all()
        assert (rows.loc[rows.launch_date <= start, "exposure_days"] == window).all()


def test_training_labels_end_before_test_window(data):
    """No training label may overlap the held-out window."""
    from src.backtest import build_training_examples

    panel, sku = data
    cfg = load_config()
    train = build_training_examples(panel, sku, cfg, cfg.split.horizon_days)
    # NB: bracket access is required for "asof" -- DataFrame.asof is a method,
    # so train.asof silently returns the method rather than the column.
    assert (pd.to_datetime(train["target_end"]) <= pd.Timestamp(cfg.split.train_end)).all()
    assert (pd.to_datetime(train["asof"]) < pd.Timestamp(cfg.split.test_start)).all()
    assert (pd.to_datetime(train["asof"]) < pd.to_datetime(train["target_start"])).all()
