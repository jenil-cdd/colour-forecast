"""Data-prep transforms applied before any model sees a label.

Two transforms live here. Both exist because of measured failure modes, and both
are applied to *every* model so no individual forecaster has to rediscover them.

--------------------------------------------------------------------------------
1. SIZE NEUTRALISATION  (the permanent Twin fix)
--------------------------------------------------------------------------------
Measured problem: every model over-forecast Twin by 1.77x-2.01x, in both the
June and the June-July windows. Root cause is a data artifact, not a modelling
error. Twin listings first appear on 2024-03-26; Queen and King cohorts run back
to 2019. Median month-1 organic units by era and size:

                King   Queen   Twin
    2019-21     12.0    18.0     --
    2022-23      3.5     4.0     --
    2024+       29.5    27.0   28.5

Twin only ever appears in the strongest era, so any model pooling cohorts across
eras reads "is a Twin" as a proxy for "launched recently" and inflates it.

The fix is to remove size from the target before fitting. Each label is divided
by an era-controlled size weight, giving a *Queen-equivalent* label:

    y_queen_equiv = y_units / size_index[size]

Models learn colour and family effects on one common scale where the Twin
artifact cannot enter. Predictions are multiplied back by the same weight. Size
is still passed as a feature, so a model can learn a residual size effect if a
real one exists - but it starts from the era-controlled ratio rather than from
the confounded one.

The size weights come from src.size_structure, which estimates them within
launch cohort (same colour, same programme, same quarter) so era and colour both
cancel. Three independent estimators agree: within-cohort 2024+ King 0.81 /
Twin 0.64, mature-within-family King 0.76 / Twin 0.69, and Sage Green's own
launch King 0.87 / Twin 0.58.

--------------------------------------------------------------------------------
2. RECENCY WEIGHTING
--------------------------------------------------------------------------------
The 2022-23 cohorts sold at 0.125x the 2024+ level - an eightfold gap. It is
tempting to normalise it away with a market-level index, but that was tested and
rejected: the market in 2022-23 ran at 0.67x the 2024+ level, nowhere near
enough to explain a 0.125x outcome, and normalising by it made the label spread
slightly *worse* (CV 1.284 -> 1.295). The gap is product quality, not market
conditions: those waves were Textured and Two-Tone variants that genuinely did
not sell.

The honest treatment is therefore not to rescale old cohorts as if they were
comparable, but to down-weight them as less representative of the current
assortment. An exponential half-life on cohort age does that, and the half-life
is chosen on replay waves, never on the test window.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.size_structure import size_weights

log = logging.getLogger(__name__)

#: Fallback ratios if a size cannot be estimated from data at all.
FALLBACK_SIZE_RATIO = {"Queen": 1.0, "King": 0.80, "Twin": 0.64, "Oversized King": 0.30}

#: Cohort-age half-life in days for recency weighting. 540 days (~18 months)
#: keeps the May-2025 wave near full weight while cutting 2019-2021 to ~0.1.
DEFAULT_HALFLIFE_DAYS = 540.0


def size_index(panel: pd.DataFrame, asof: pd.Timestamp, sizes: list[str],
               horizon: int = 120, age_lo: int = 0,
               age_hi: int | None = None) -> pd.Series:
    """Era-controlled size ratios, normalised so Queen = 1.0.

    Uses only data at or before ``asof``, so it is legal at decision time.
    ``age_lo``/``age_hi`` make the index horizon-specific, which is necessary
    because Twin ramps faster than Queen: the ratio is ~0.61 in month 1 and
    ~0.88 in month 2.
    """
    try:
        w = size_weights(panel, asof, sizes, anchor="Queen", horizon=horizon,
                         age_lo=age_lo, age_hi=age_hi)
        # size_weights returns allocation shares; convert back to Queen-relative.
        if "Queen" in w.index and w["Queen"] > 0:
            r = w / w["Queen"]
        else:
            r = w / w.max()
        r = r.replace([np.inf, -np.inf], np.nan)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("size_index estimation failed (%s); using fallback", exc)
        r = pd.Series(dtype=float)

    out = {}
    for s in sizes:
        v = r.get(s, np.nan)
        if not np.isfinite(v) or v <= 0:
            v = FALLBACK_SIZE_RATIO.get(s, 0.6)
        out[s] = float(v)
    return pd.Series(out, name="size_index")


def recency_weight(launch_dates: pd.Series, asof: pd.Timestamp,
                   halflife_days: float = DEFAULT_HALFLIFE_DAYS) -> np.ndarray:
    """Exponential decay on cohort age. 1.0 for a launch on ``asof``."""
    age = (pd.Timestamp(asof) - pd.to_datetime(launch_dates)).dt.days.clip(lower=0)
    return np.power(0.5, age.to_numpy(float) / max(halflife_days, 1.0))


def neutralise(rows: pd.DataFrame, idx: pd.Series,
               label_col: str = "y_units") -> tuple[pd.DataFrame, np.ndarray]:
    """Convert labels to Queen-equivalent and attach the divisor used.

    Returns the frame with ``size_index`` / ``y_queen_equiv`` columns added, and
    the Queen-equivalent label array ready to hand to a model.
    """
    out = rows.copy()
    out["size_index"] = out["size"].astype(str).map(idx).fillna(
        float(np.nanmedian(list(idx.values))) if len(idx) else 0.7)
    y = out[label_col].to_numpy(float) / out["size_index"].to_numpy(float).clip(min=1e-6)
    out["y_queen_equiv"] = y
    return out, y


def denormalise(pred_queen_equiv: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    """Undo :func:`neutralise` on the prediction side."""
    si = rows["size_index"].to_numpy(float)
    return np.clip(pred_queen_equiv, 0, None) * si


def blind_size(rows: pd.DataFrame, keep_col: str = "size_true") -> pd.DataFrame:
    """Hide size from the models, preserving the true value for scoring.

    Dropping size-carrying *features* is not enough. Three models reach for
    ``rows["size"]`` directly and so bypass the design matrix entirely:

    * ``knn_lookalike`` applies a hard same-size penalty when choosing
      neighbours, making it a within-size estimator - so a Twin is matched only
      to the 2025 Twin wave, the strongest cohort in the data.
    * ``matrix_factorisation`` uses size as a literal matrix dimension.
    * ``cluster_pool`` estimates velocity per (cluster, size).
    * ``hier_bayes`` carries a size random effect.

    With size visible, those four re-introduced the artifact after
    neutralisation: twin_over_queen stayed at 2.3, 3.0, 5.2 and 2.4
    respectively. Collapsing size to a single level forces every model to pool
    across sizes, so the size effect enters exactly once - through the
    era-controlled index - and cannot be double-counted.
    """
    out = rows.copy()
    if keep_col not in out.columns:
        out[keep_col] = out["size"].astype(str)
    out["size"] = "ALL"
    return out


def report(panel: pd.DataFrame, asof: pd.Timestamp, sizes: list[str]) -> pd.DataFrame:
    """Audit trail: the size weights in force and where they came from."""
    from src.size_structure import mature_ratios, within_cohort_ratios

    wc = within_cohort_ratios(panel, asof).set_index("size")
    mt = mature_ratios(panel, asof).set_index("size")
    idx = size_index(panel, asof, sizes)
    rows = []
    for s in sizes:
        rows.append({
            "size": s,
            "within_cohort_2024plus": round(float(wc.ratio.get(s, np.nan)), 3),
            "mature_within_family": round(float(mt.ratio.get(s, np.nan)), 3),
            "size_index_applied": round(float(idx.get(s, np.nan)), 3),
        })
    return pd.DataFrame(rows)
