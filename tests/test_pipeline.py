"""Guards for the strict-split pipeline, size fix, and stacking.

These exist because each one caught a real bug during the build.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.features import SIZE_CARRYING, design_matrix
from src.horizons import HORIZONS, scoring_examples, training_examples
from src.prep import blind_size, denormalise, neutralise, recency_weight, size_index


@pytest.fixture(scope="module")
def data():
    p = Path("data/processed/panel.parquet")
    if not p.exists():
        pytest.skip("panel not built")
    return (pd.read_parquet(p),
            pd.read_parquet("data/processed/sku_annotated.parquet"),
            load_config())


# --- temporal safety -------------------------------------------------------
def test_training_labels_close_before_train_end(data):
    panel, sku, cfg = data
    for hz, lo, hi in HORIZONS:
        tr = training_examples(panel, sku, cfg, hz)
        if tr.empty:
            continue
        assert (pd.to_datetime(tr["target_end"]) <= pd.Timestamp(cfg.split.train_end)).all(), hz
        # Features must be built strictly before the cohort launched.
        assert (pd.to_datetime(tr["asof"]) < tr["launch_date"]).all(), hz


def test_both_test_months_share_one_cutoff(data):
    """Month 2 must be a genuine 2-months-ahead forecast, not a re-forecast."""
    panel, sku, cfg = data
    jun = scoring_examples(panel, sku, cfg, "2026-06-01", "2026-06-30")
    jul = scoring_examples(panel, sku, cfg, "2026-07-01", "2026-07-31")
    assert (pd.to_datetime(jun["asof"]) == pd.Timestamp(cfg.split.train_end)).all()
    assert (pd.to_datetime(jul["asof"]) == pd.Timestamp(cfg.split.train_end)).all()


def test_horizon_windows_do_not_overlap(data):
    panel, sku, cfg = data
    bounds = {n: (a, b) for n, a, b in HORIZONS}
    assert bounds["m1"][1] < bounds["m2"][0]


# --- the size fix ----------------------------------------------------------
def test_size_index_orders_correctly(data):
    panel, _, cfg = data
    for hz, lo, hi in HORIZONS:
        idx = size_index(panel, pd.Timestamp(cfg.split.train_end),
                         ["Queen", "King", "Twin"], age_lo=lo, age_hi=hi)
        assert idx["Queen"] == pytest.approx(1.0)
        assert 0.4 < idx["Twin"] < 1.05, (hz, idx.to_dict())
        assert 0.6 < idx["King"] < 1.05, (hz, idx.to_dict())


def test_neutralise_then_denormalise_roundtrips(data):
    panel, sku, cfg = data
    tr = training_examples(panel, sku, cfg, "m1")
    idx = size_index(panel, pd.Timestamp(cfg.split.train_end),
                     sorted(tr["size"].astype(str).unique()))
    rows, y = neutralise(tr, idx)
    back = denormalise(y, rows)
    assert np.allclose(back, rows.y_units.to_numpy(float), rtol=1e-9)


def test_drop_size_removes_every_size_carrying_feature(data):
    """The bug this guards: leaving size in the design matrix let each model
    re-learn the size effect and silently undo the neutralisation."""
    panel, sku, cfg = data
    tr = training_examples(panel, sku, cfg, "m1")
    X_full, _ = design_matrix(tr, drop_size=False)
    X_drop, _ = design_matrix(tr, drop_size=True)
    for c in SIZE_CARRYING:
        assert not any(col == c or col.startswith(f"{c}_") for col in X_drop.columns), c
    assert len(X_drop.columns) < len(X_full.columns)


def test_blind_size_hides_size_but_keeps_truth(data):
    """Four models read rows["size"] directly and bypass the design matrix."""
    panel, sku, cfg = data
    tr = training_examples(panel, sku, cfg, "m1")
    b = blind_size(tr)
    assert set(b["size"].unique()) == {"ALL"}
    assert (b["size_true"] == tr["size"].astype(str)).all()


# --- recency weighting -----------------------------------------------------
def test_recency_weight_decays_monotonically(data):
    _, _, cfg = data
    asof = pd.Timestamp(cfg.split.train_end)
    dates = pd.Series(pd.to_datetime(["2019-06-01", "2022-06-01", "2025-05-26", asof]))
    w = recency_weight(dates, asof, halflife_days=540)
    assert np.all(np.diff(w) > 0), w
    assert w[-1] == pytest.approx(1.0)
    assert w[0] < 0.05


# --- stacking --------------------------------------------------------------
def test_stack_weights_are_a_convex_combination(data):
    panel, sku, cfg = data
    from src.stacking import EXCLUDE, DynamicStack

    tr = training_examples(panel, sku, cfg, "m1")
    tr = tr[tr.y_days_live.fillna(0) >= 7]
    idx = size_index(panel, pd.Timestamp(cfg.split.train_end),
                     sorted(tr["size"].astype(str).unique()))
    rows, y = neutralise(tr, idx)
    rows = blind_size(rows)
    X, _ = design_matrix(rows, drop_size=True)
    s = DynamicStack(n_folds=3, seed=1).fit(rows, X, y, panel=panel)
    w = s.weights_
    assert w is not None and len(w) > 0
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    # The incumbent benchmark must never be a stack member.
    assert not set(w.index) & EXCLUDE


def test_stack_gate_is_relative_not_absolute(data):
    """An absolute OOF cut-off rejected 14 of 17 members - including the two
    best - because out-of-fold cohorts include the very weak 2022-23 waves."""
    from src.stacking import DynamicStack

    s = DynamicStack()
    assert s.rel_tolerance > 1.0
    y = np.array([10.0] * 20)
    good = np.array([11.0] * 20)
    bad = np.array([100.0] * 20)
    assert s._oof_error(good, y) < s._oof_error(bad, y)
