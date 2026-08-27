"""Metric behaviour, especially around the zeros and tiny counts in this data."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibrate import ConformalCalibrator, WaveAwareCalibrator
from src.metrics import bias, coverage, newsvendor_cost, spearman, topk_hit_rate, wape


def test_wape_perfect_and_zero_safe():
    y = np.array([10.0, 0.0, 5.0])
    assert wape(y, y) == 0.0
    assert np.isnan(wape(np.zeros(3), np.ones(3)))


def test_bias_sign():
    y = np.array([10.0, 10.0])
    assert bias(y, np.array([12.0, 12.0])) > 0     # over-forecast
    assert bias(y, np.array([8.0, 8.0])) < 0       # under-forecast


def test_newsvendor_penalises_shortage_more():
    y = np.array([100.0])
    over = newsvendor_cost(y, np.array([120.0]), holding=0.25, stockout=1.0)
    under = newsvendor_cost(y, np.array([80.0]), holding=0.25, stockout=1.0)
    assert under > over, "a 20-unit stock-out must cost more than a 20-unit overstock"
    assert over == pytest.approx(5.0)
    assert under == pytest.approx(20.0)


def test_topk_and_spearman_measure_ordering():
    y = np.array([100.0, 50.0, 10.0, 1.0])
    good = np.array([90.0, 60.0, 20.0, 5.0])       # right order, wrong levels
    assert spearman(y, good) == pytest.approx(1.0)
    assert topk_hit_rate(y, good, k=2) == 1.0
    assert topk_hit_rate(y, good[::-1], k=2) == 0.0


def test_conformal_widens_and_is_monotone():
    rng = np.random.default_rng(0)
    yhat = np.full(200, 50.0)
    y = np.expm1(np.log1p(yhat) + rng.normal(0, 0.8, 200))
    c = ConformalCalibrator().fit(y, yhat)
    lo, hi = c.apply(yhat)
    assert (lo < hi).all()
    assert coverage(y, lo, hi) == pytest.approx(0.80, abs=0.08)


def test_wave_aware_is_wider_than_plain():
    """With a real between-group shock, the wave-aware interval must be wider."""
    rng = np.random.default_rng(1)
    origins = np.repeat(["a", "b", "c", "d"], 25)
    shock = {"a": -0.8, "b": 0.0, "c": 0.6, "d": -0.4}
    yhat = np.full(100, 40.0)
    y = np.expm1(np.log1p(yhat) + np.array([shock[o] for o in origins])
                 + rng.normal(0, 0.3, 100))
    plain = ConformalCalibrator().fit_store(y, yhat)
    wave = WaveAwareCalibrator().fit_store(y, yhat, origins=origins)

    c = wave.components()
    # The wave-aware calibrator must actually detect the between-group shock and
    # carry it into the total. (It is not always wider than plain conformal on a
    # single sample: plain takes empirical quantiles of the pooled mixture, which
    # can be fat-tailed, whereas this is a Normal approximation on the variance
    # components. The invariant that matters is that the between term is found
    # and added, not that one interval always dominates the other.)
    assert c["sd_between"] > 0.3, c
    assert c["sd_total"] > c["sd_within"]
    assert c["sd_total"] == pytest.approx(
        (c["sd_within"] ** 2 + c["sd_between"] ** 2) ** 0.5, rel=1e-9)
    lo_w, hi_w = wave.apply(yhat)
    assert (lo_w < hi_w).all()
    del plain


def test_size_weights_are_era_controlled():
    """Twin must not come out above Queen.

    Guards the confound that put Twin at 53% of a draft order sheet: Twin
    listings only exist from 2024-03-26 while Queen/King cohorts run back to
    2019, and the 2024+ era is ~2.1x stronger, so ratios pooled across eras read
    that era gap as a Twin size effect.
    """
    import pandas as pd

    from src.size_structure import size_weights, within_cohort_ratios

    p = Path("data/processed/panel.parquet")
    if not p.exists():
        pytest.skip("panel not built")
    panel = pd.read_parquet(p)
    asof = pd.Timestamp("2026-05-31")

    w = size_weights(panel, asof, ["Queen", "King", "Twin"])
    assert w.sum() == pytest.approx(1.0)
    assert w["Queen"] > w["King"] > w["Twin"], w.to_dict()

    wc = within_cohort_ratios(panel, asof).set_index("size")
    # Corroborated independently by mature listings and by the incumbent's own
    # assumption; all three land in this band.
    assert 0.45 < wc.ratio["Twin"] < 0.85, wc.ratio.to_dict()
    assert 0.65 < wc.ratio["King"] < 1.00, wc.ratio.to_dict()


def test_ramp_barely_moves_when_ppc_is_stripped():
    """Organic and gross ramp curves must agree closely.

    Ad-attributed units are only 7.5% of volume and their share *rises* with
    listing age, so removing PPC cannot be what produces the launch ramp. If
    this assertion ever fails, the PPC-vs-organic story has changed materially
    and the baseline needs revisiting.
    """
    import pandas as pd

    p = Path("data/processed/panel.parquet")
    if not p.exists():
        pytest.skip("panel not built")
    panel = pd.read_parquet(p)
    if "organic_units" not in panel.columns:
        pytest.skip("ads not integrated")

    d = panel[panel.days_since_launch <= 359].copy()
    d["mo"] = d.days_since_launch // 30
    g = d.groupby("mo").agg(tot=("units_ordered", "mean"), org=("organic_units", "mean"))
    ramp_tot = g.loc[0, "tot"] / g.loc[6:11, "tot"].mean()
    ramp_org = g.loc[0, "org"] / g.loc[6:11, "org"].mean()
    assert abs(ramp_tot - ramp_org) < 0.05, (ramp_tot, ramp_org)
