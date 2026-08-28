# How this pipeline works, and why

Written to be reviewable in ten minutes. Every design choice below exists
because something measurably broke without it.

---

## The shape of the problem

Forecast how many units a **colour that does not exist yet** will sell in its
first months on Amazon. No sales history, no reviews, no ranking. Volumes are
tiny — 0 to 4 units a day per SKU — so this is a small-count problem, not a
smooth time series.

Training data is 101 historical launch cohorts. That is a small sample, and it
is the single fact that explains most of what follows: sophisticated models
overfit it, and simple structural corrections beat clever functional forms.

---

## The pipeline in one pass

```
BigQuery ──► panel ──► data prep ──► 19 models ──► stack ──► certify
             (daily)   ┌─ size-neutralise         ┌─ per-horizon    ┌─ per-task
                       └─ recency-weight          └─ OOF weights    └─ pass/fail
```

**Strict split.** Everything is fitted on data ending **31 May 2026**. Then two
isolated scores: **June 2026** (month 1) and **July 2026** (month 2). Both use
the same information cut-off, so July is a genuine two-months-ahead forecast
rather than a one-month-ahead forecast wearing a later label.

**Training examples are launch cohorts, not calendar snapshots.** For each past
launch: features as of the day *before* it went live, label = units over the
matching slice of its life. A cohort only qualifies if its whole label window
closes on or before the cut-off.

---

## The two data-prep fixes

### 1. Size neutralisation — the permanent Twin fix

**Symptom.** Every model over-called Twin by 1.8x–2.0x, in both months.

**Cause.** Twin listings first appear **2024-03-26**; Queen and King run back to
2019. Median month-1 organic units:

| era | King | Queen | Twin |
|---|---:|---:|---:|
| 2019–21 | 12.0 | 18.0 | — |
| 2022–23 | 3.5 | 4.0 | — |
| 2024+ | 29.5 | 27.0 | 28.5 |

Twin exists only in the strongest era, so "is a Twin" became a proxy for
"launched recently". The hierarchical model read the ratio as **Twin = 1.54x
Queen**. Every era-controlled estimate says ~0.6–0.9.

**Fix.** Divide each label by an era-controlled size index before fitting
(a *Queen-equivalent* label), and multiply back on prediction. Models learn
colour and family effects on one scale where the artifact cannot enter.

**Two bugs found while implementing this — both worth knowing:**

- Neutralising the label while leaving `size` in the design matrix does
  *nothing*. Every model simply re-learns the size effect through the one-hot
  and undoes the correction. `twin_over_queen` stayed at 1.3–2.3. Size-carrying
  features must be dropped (`drop_size=True`).
- Dropping features is still not enough. Four models reach for `rows["size"]`
  directly: `knn_lookalike` (hard same-size neighbour penalty, making it a
  within-size estimator), `matrix_factorisation` (size is a matrix dimension),
  `cluster_pool` (velocity per cluster×size), `hier_bayes` (size random effect).
  Those four re-introduced the artifact at 2.3, 3.0, 5.2 and 2.4. Size is now
  collapsed to a single level (`blind_size`) before any model sees the rows.

**Result:** every model's `twin_over_queen` fell into 0.5–1.3, against actuals
of 0.61 (June) and 0.88 (July).

**The index is per-horizon.** Twin ramps ~3.0x from month 1 to month 2 against
~2.2x for Queen and King, so one fixed ratio cannot serve both months.

### 2. Recency weighting — and the normalisation that was rejected

2022–23 cohorts sold at **0.125x** the 2024+ level, an eightfold gap. The
obvious move is to normalise it away with a market-level index. **That was tried
and rejected:** the market then ran at 0.67x, nowhere near enough to explain a
0.125x outcome, and normalising by it made the label spread slightly *worse*
(CV 1.284 → 1.295). The gap is product quality, not market conditions — those
waves were Textured and Two-Tone variants that genuinely did not sell.

So old cohorts are **down-weighted** (exponential, 540-day half-life) rather
than rescaled as if comparable.

---

## The dynamic stack

**Why stack.** No model won twice. On the 61-day window `knn_lookalike` placed
1st and `hier_bayes` 3rd; on the 30-day window that reversed. Different models
carry different parts of the launch curve.

**How the weights are learned.** Non-negative least squares on `log1p` units over
**out-of-fold predictions on historical cohorts only** — never the test months.
Folds are grouped by launch wave, because listings in one wave share a demand
shock and splitting them would let a model see its own wave.

One weight vector **per horizon**. That is what makes it dynamic, and the fitted
weights differ exactly as expected: `hier_bayes` takes the largest month-1 share,
`knn_lookalike` and the tree models take more of month 2.

**Three guard rails, each added after a specific failure:**

| Guard | The failure it fixes |
|---|---|
| Explicit intercept | With size blinded, `heuristic` collapses to a near-constant. NNLS without an intercept used it as a bias term — 54% of the June blend, 78% of July, pushing July's total error to 0.65. It is now excluded outright as a benchmark, and the intercept removes the incentive. |
| Log-scale fitting | Fitting on raw units let a few high-volume launches dictate the whole blend. An earlier version came out worse than its own members. |
| **Relative** membership gate | An absolute out-of-fold cut-off rejected **14 of 17 members including the two best**, because out-of-fold cohorts include the brutal 2022–23 waves where everything looks bad. June per-line error rose 0.447 → 0.669. Members are now admitted within 1.35x of the *best* member's out-of-fold error, which adapts to how hard the fold actually is. |

**Intervals come from `hier_bayes`, not the blend.** It was the best-calibrated
model in the suite by a wide margin (100% and 94% coverage on the two months
against an 80% target, versus 47–95% elsewhere) because it is the only member
that propagates parameter uncertainty and count noise properly. Averaging
intervals across members would blur that. The blend supplies the level,
`hier_bayes` supplies the shape.

---

## The sanity harness — what each model may be *used for*

A leaderboard says which model has the lowest error. It does not say whether the
output is fit for a purpose, and those differ sharply here:

- `poisson_glm` had the best per-line error on the 61-day window while missing
  the total by 39% — good for ranking a shortlist, useless for a purchase order.
- `naive_family` ranks colours better than anything else (ρ 0.57–0.61) and misses
  total volume by 334%.

So every model is certified per task, with thresholds in the units of the
decision each protects:

| Task | Passes if | Protects |
|---|---|---|
| `TOTAL` | total error ≤ 0.35, abs bias ≤ 0.30 | the purchase order |
| `PER_LINE` | line error ≤ 0.80, abs bias ≤ 0.45, Twin ratio ≤ 1.10 | per-SKU quantities |
| `RANKING` | Spearman ≥ 0.30 | which colours to launch |
| `INTERVALS` | coverage in [0.60, 0.95] | safety stock |

Structural checks run first — finite, non-negative, ordered intervals, sane
magnitude. A model failing those is failed outright regardless of its metrics.
The Twin check gates `PER_LINE` only: getting the total right while splitting it
badly across sizes is a mix problem, not broken output.

---

## What the results actually say

**Month 1 (June, true cold start):** the stack is **best of 20** — per-line error
0.473, total error 0.125, ranking 0.680, coverage 0.941, certified for all four
tasks. Stacking works where it matters most.

**Month 2 (July):** the stack nails the total (**2.4% error**) but ranks 12th of
20 on per-line. Its out-of-fold month-2 weights favoured `lightgbm` and
`hier_bayes`, which then performed poorly on the actual July window, while
`size_ratio` and `negbin_glm` did well and were under-weighted. **Month-2
out-of-fold skill on historical cohorts does not transfer to the test window.**
That is reported rather than tuned away — fixing it by adjusting weights until
July looks good would be fitting the test set.

**On swing error**, which motivated the refactor: the stack's worst total error
across the two months is **0.125** against a median individual model's 0.283, and
the growth curves are actually the most stable of all on totals (0.050–0.080)
while being the worst on per-line. No single approach dominates.

**Recommended routing**, straight out of the certification matrix:

| Decision | Use |
|---|---|
| Total buy quantity, month 1 | stack, or `gompertz` for maximum stability |
| Total buy quantity, month 2 | `gompertz` / `logistic` / `bass` (total error 0.02–0.04) |
| Per-SKU split, month 1 | stack (0.473) |
| Per-SKU split, month 2 | `size_ratio` (0.553) or `negbin_glm` (0.563) |
| Which colours to pick | `hier_bayes` (ρ 0.83 month 1), `naive_family` (ρ 0.57 month 2) |
| Safety stock band | `hier_bayes` |

---

---

## DCS-only mode (current default)

`programs.restrict_to_focal: true` drops all 400 TC and Organic listings before
feature engineering. Rationale: Silver was over-called **4.3x in DCS in both test
months** because 400 TC's Silver sold well — a cross-programme transfer error.

**It fixed what it targeted.** Every over-forecast colour improved:

| Listing | Ratio, both programmes | Ratio, DCS only |
|---|---:|---:|
| Silver Queen (Jul) | 6.5x | 3.6x |
| Silver King (Jul) | 3.0x | 1.8x |
| Taupe Queen (Jun) | 5.3x | 1.9x |
| Taupe King (Jun) | 2.1x | 0.8x |

**It cost more than it fixed.** Median worst-month per-line error went **0.758 →
0.792**, and only 8 of 20 models improved. The under-forecast colours got much
worse — Antique White fell from 0.6x to **0.2x** in June and 0.4x to 0.2x in July.

The cause is visible in the training data: DCS's own month-1 label mean is 25.3
units against 33.6 across both programmes. DCS historically sold *less* than
400 TC, but the 2026 DCS launches performed at 400-TC-like levels. Dropping
400 TC removed the data that captured the stronger modern launch behaviour, so
every model now runs systematically low.

**One thing cannot be estimated DCS-only.** Size ratios. DCS has 2 usable
size-ratio cohorts and neither contains a Twin; a DCS-only estimate returns
**Twin = 2.088x Queen** off a single observation, against 0.785 from the pooled
estimate and observed actuals of 0.61 / 0.88. `programs.size_ratio_source`
therefore defaults to `all_programmes` for the size split only — colour velocity
stays strictly DCS. Set it to `focal` to forbid borrowing and accept the prior.

### Interval tightening

`src/intervals.py` replaces model-derived bands with split-conformal bands fitted
on DCS out-of-fold residuals, per size where there is enough history.

| | Both programmes | DCS conformal |
|---|---:|---:|
| Median band width, June | 57 units | **20 units** |
| Median band width, July | 87 units | **42 units** |
| Coverage, June | 94% | **50%** |
| Coverage, July | 79% | **50%** |

The bands are ~2.5x tighter and cover half as often. That is the whole trade and
it is not avoidable by choosing a better estimator: with 59 cohorts the residual
spread is genuinely that wide, and a narrower band does not reduce risk, it only
stops reporting it.

### A defect worth remembering

`size_ratio` had the best DCS-only worst-month per-line error (0.592) and was
briefly used for the order sheet. It produced **identical numbers for Cream and
Greige, and for Olive and Terracotta**, because it resolves no finer than shade
family. `hier_bayes` and `matrix_factorisation` share the defect (3 distinct
values across 5 colours). Model choice for an order sheet needs **colour
resolution as well as accuracy**; the sheet now uses `ensemble`, the most
accurate model that separates all five.

---

## The ASIN-day grain defect (fixed)

`sales_and_traffic_..._filtered_final` is **not** one row per ASIN-day. It returns
split records — one row carrying orders, another carrying traffic:

```
B07D2FRRFR  2020-02-09  sessions=0   units=14  sales=745.84
B07D2FRRFR  2020-02-09  sessions=72  units=0   sales=0.00
```

3,164 rows across **1,584 ASIN-days and 57 of 130 ASINs**, 2020–2026. Unit
*totals* were unharmed (391,438 either way) but every per-row statistic was: a
split day looked like two days, one with zero units, so split rows averaged
**0.80 units against 2.79** for clean rows and read as **91% organic against
67%**. That fed the rolling median, the MAD spike screen and the zero-share
features — i.e. the promo detection that every velocity estimate rests on.

`sql/02_daily_panel.sql` now aggregates to one row per ASIN-day: counters
`SUM`med, rates recomputed from the summed counters (ASP as
`SUM(sales)/SUM(units)`, buy-box weighted by sessions), and a `source_rows`
column records how many raw rows collapsed. 108,444 days came from 1 row,
1,578 from 2, and 2 from 4.

## Fetch grain, for reference

Programme filtering happens **once**, against the mapper. Every fact table is
pulled at ASIN×day and filtered by the resulting explicit ASIN list. There is no
programme-level aggregation in SQL — programme is an attribute column and all
rollups happen in pandas.

| Query | Grain | Filter |
|---|---|---|
| `01_sku_dim` | one row per ASIN | `parent IN UNNEST(@programs)` |
| `02_daily_panel` | ASIN × day | `child_asin IN UNNEST(@asins)` |
| `04_asin_deal_days` | ASIN × day | ASIN list |
| `05_returns` | ASIN × day | ASIN list |
| `06_ads_sponsored` | ASIN × day | ASIN list |
| `03_clean_days` | day | date only |

## Two interval-calibration traps

**Whole-wave CV folds are degenerate when one wave dominates.** 400 TC has 38
cohorts across 5 waves with **21 in 2025Q2**. Holding that wave out trains on
2020 data and asks it to predict 2025: out-of-fold error came out at 0.896
against 0.603 in-sample, and the implied 80% band widened to 3.78x. Stratifying
each wave proportionally across folds gives 0.635 — close to in-sample, and
representative of the real forecast, which has all history available.

**Per-size residual bands re-import the wave confound.** The size level is
already supplied by the era-controlled index. Fitting a separate band per size
gave Twin 4.61x against Queen 1.46x — because 400 TC's 12 Twin cohorts come
disproportionately from the strong May-2025 wave — which made Twin the *largest*
line in the order sheet while its point forecast correctly sat below Queen. The
interval is now pooled across sizes.

**A related caution on reading size ratios.** Marginal medians by size in 400 TC
suggest Twin outsells Queen (43 vs 22.5 in month 1). That is the colour-mix
confound again: the Twin cohorts are different colours from the Queen cohorts.
Within-cohort, 400 TC Twin/Queen is **1.048** (n=5), and the 2026 actuals are
0.84 and 0.80. Always read these within cohort.

## Known limits

- **101 training cohorts, 17–19 test listings.** Differences of one or two
  places in any ranking are noise.
- **Month-2 weights do not transfer.** The single clearest open problem.
- **A 370-day gap** separates the newest training launch (27 May 2025) from the
  test window; there were no launches in between.
- **Twin rests on 5 era-controlled cohorts.** Corroborated by 11 mature
  families, but it is the thinnest link.
- **The stack is not universally better.** It wins month 1 outright and gets
  month-2 totals right; it does not win month-2 per-line. Route by task.

## Reproducing

```bash
make panel        # rebuild from data/raw
make pipeline     # strict May-train / June + July test, all 19 models + stack
make test         # 33 tests, including leakage and size-fix guards
```
