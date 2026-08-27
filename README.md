# Duvet Cover Set — colour & size cold-start demand forecasting

A demand-forecasting framework for deciding **which new colour variants to launch, and how many units to buy**, for a 120-day initial order with a 120-day lead time.

It replaces a heuristic that recommended **2,180 units** off six hard-coded constants. Every one of those constants is re-estimated here, and the resulting recommendation is validated on a genuinely held-out window.

---

## The validation setup

Training data runs to **2026-05-31**. Scoring runs on **2026-06-01 → 2026-07-31**.

That split is not arbitrary: the catalogue contains a real launch wave of **17 listings across six new colours** (Antique White, Silver, Silver Gray, Taupe, Indigo Dusty Blue, Light Grey) that went live between 2026-05-26 and 2026-06-13. So the held-out window is an actual cold-start colour launch, not a synthetic holdout — the exact decision the framework exists to support.

Because one window with 19 listings is thin evidence for ranking 19 models, the identical protocol is also replayed at **seven earlier launch events**, including a near-identical 17-listing wave on 2025-05-26.

---

## Headline results

**1. The incumbent heuristic under-orders badly, and cannot rank colours at all.**

It assigns every Queen variant the same number regardless of colour. Actual Jun–Jul demand across the cohort ranged from **20 units (Silver Queen) to 318 units (Antique White Queen)** — a 16x spread the heuristic is structurally blind to. On the held-out window it forecast 572 units against 1,804 actual (bias −68%).

**2. Aggregate demand is far more predictable than any individual SKU.**

| level | WAPE (pooled over 7 launch events) |
|---|---|
| individual SKU | 0.69 |
| cohort total | **0.20** |

Decomposing forecast error across events: between-event SD 0.689, within-event SD 0.694. **Roughly half the error is a shock common to the whole launch wave** — the colour wave lands well or it does not. This is the single most actionable finding: commit the *total* buy with confidence, keep the colour/size *allocation* flexible.

**3. Order-cost outcome vs the incumbent** (stockout:holding = 4:1, actual 120-day-equivalent demand 4,435 units):

| policy | order | cost index |
|---|---|---|
| perfect hindsight | 4,435 | 0 |
| **framework, p60–p70** | **5,300–6,700** | **1,557** |
| framework, p80 (newsvendor optimum) | 8,639 | 1,588 |
| incumbent heuristic | 2,180 | 2,557 |

The framework reduces expected inventory cost by **~39%**. The cost curve is flat between p60 and p80, so the recommendation is robust to fractile choice within that band.

**4. Interval coverage was broken, and is now partly fixed.**

Every model's nominal 80% interval under-covered (0.18–0.71). Plain conformal calibration lifted mean coverage to 0.61; adding the wave-level variance component reached **0.67**. It still falls short of 0.80 — reported as a limitation, not papered over.

---

## What the audit found about the heuristic's constants

| assumption | heuristic | re-estimated | verdict |
|---|---|---|---|
| White share of volume | 62.6% | 59.0% all-history, **67.2% last 90d** | directionally right; non-white pool is *smaller* than assumed recently |
| King / Queen ratio | 0.76 | **0.764** (n=18, IQR 0.71–0.84) | **validated** |
| Twin / Queen ratio | 0.60 | **0.558** (n=10, IQR 0.32–0.85) | median close, but the spread is the real story — see below |
| Month-1 ramp | 45% of mature | **47.5%** (month 0); plateau by month 5–6 | **validated** |
| Family depth decay | 1.42 → 1.00 → 0.58 → 0.00 | grey: 1.03 → 0.51 | real for grey, but the naive rank table is confounded by family and era |
| Return rate, solids | 11.4% | **8.5%** | overstated |
| Return rate, patterns | 17% | **13.3%** | overstated |
| Return rate, Beige Queen | 28% | **13.6%** | substantially overstated |
| — | not flagged | **Textured 23.9%, Near White 17.5%** | the two real outliers were missed |

The Near White finding matters commercially: **Antique White was both the best-selling new colour and the second-highest return family**, so its gross and net rankings differ.

### On Twin: the honest answer is "we cannot know yet"

The focal program has **no mature Twin listings at all**, so its Twin ratio has to be borrowed from sibling programs, where it ranges 0.27–0.98 across ten families. Worse, the bias is not stable: pre-2026 Twin residuals averaged −0.375, but the 2026 wave came in at −0.827. **A size-specific correction fitted on history would have pointed the wrong way.** Twin uncertainty is therefore treated as irreducible and handled by staging the Twin buy, not by picking a better point ratio.

---

## Final order sheet (rev 2 — approved 2026-08-27)

Five candidates, 120-day horizon, demand basis **organic** (PPC-attributed units stripped), era-controlled size split. p80 and p90 shown side by side.

| colour | merch family | p80 Queen | p80 King | p80 Twin | **p80 total** | p90 Queen | p90 King | p90 Twin | **p90 total** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dusty Blue | Blue 2 | 536 | 416 | 361 | **1,313** | 762 | 591 | 512 | **1,865** |
| Cream | Near-White 1 | 498 | 389 | 332 | **1,219** | 707 | 552 | 471 | **1,730** |
| Olive | Green | 477 | 371 | 320 | **1,168** | 677 | 527 | 454 | **1,658** |
| Greige | Near-White 2 | 416 | 325 | 277 | **1,018** | 591 | 461 | 394 | **1,446** |
| Terracotta | Warm Earth | 351 | 273 | 236 | **860** | 499 | 387 | 335 | **1,221** |
| **total** | | **2,278** | **1,774** | **1,526** | **5,577** | **3,236** | **2,518** | **2,166** | **7,921** |

Expected (point) 120-day organic demand **3,123 units**.

| | order | buffer | vs "all five match Sage Green" (5,445u) |
|---|---:|---:|---:|
| p80 | 5,577 | +79% | 1.02x |
| **p90** (9:1) | **7,921** | **+154%** | 1.45x |
| capital delta | +2,344 | | +42% more units |

**Ads are an argument for p90, not against it.** The baseline is organic-only, so the PPC lift being planned is *not* in these numbers. Historically the best-funded launch in the catalogue reached **22.8%** ad-attributed share (at \$70/day); the top decile of launches sits near 10.4% (\$8.90/day) against a median of 0.1%. Funding to top-decile levels plausibly adds ~10–25% on top of the organic forecast — which lands inside the p80→p90 gap. Two caveats: the spend→units correlation across 79 launches is only **0.217**, so this is a plausible range and not a prediction; and attributed units include clicks from customers who would have converted anyway, so true incrementality is lower than the attributed share.

### "Truly net-new" — only two of the five qualify

| candidate | closest existing listing | verdict |
|---|---|---|
| **Greige** | nothing matches | genuinely net-new |
| **Terracotta** | nothing matches (Warm Earth unentered) | genuinely net-new |
| Cream | `Textured - Cream Boho Tufted` — 1,339 units, wound down ~Sep–Oct 2025 | new as a *solid*; shade has been in market |
| Dusty Blue | `Indigo Dusty Blue` Twin — **live**, 128 units since 2026-05-26 | near-duplicate of a current listing |
| Olive | `Olive Green` Queen/King/Twin listed in *Organic Duvet Cover Set* — **zero sales ever** | net-new to this programme, but see below |

**Olive carries the same zero-sales signature that disqualified Soft Lavender.** Olive Green exists as three listings in the Organic Duvet Cover Set programme with no sales on record. That may mean the listings were never activated rather than that they failed — the data cannot distinguish the two — but the pattern is identical to the Lavender Grey case, so the reasoning that dropped Soft Lavender applies here too.

Olive also ranks below Sage (1,658 vs 2,089 at p90) on colour attributes alone: it is much darker and more saturated (Lab L 52 vs 69, chroma 26 vs 15, distance-to-white 54.5 vs 33.8), and this catalogue is dominated by light neutrals. Note too that **Sage Green's success sits in the 400 TC programme, not this one**, so its depth does not penalise Olive here — and cross-programme cannibalisation is not modelled.

### Two corrections found while building this

**PPC is not what drives the launch ramp.** Ad-attributed units are 7.5% of volume overall, and the share is *lowest* at launch (1.9% in month 0) rising to 7.7% by month 7. The 2026 launch cohort was **1.3% ad-attributed** on \$1,748 of total spend. Stripping PPC moves the month-0 ramp factor from 0.438 to **0.456** — a 1.8pp change in the *opposite* direction to the concern.

**The size split had a confound worth naming.** A draft put **Twin at 53% of the order** and ranked Sage Twin above Sage Queen. Cause: **Twin listings only exist from 2024-03-26**, while Queen/King cohorts run back to 2019, and the 2024+ launch era is ~**2.1x** stronger (mean first-120-day organic units: Queen 274 vs 129). A model fitted on cohorts pooled across eras reads that era gap as a Twin size effect — the hierarchical model returned Twin = **1.54x** Queen.

Estimating ratios **within launch cohort** (same colour, programme, quarter) cancels era and colour. Three independent strategies then agree:

| method | control | King | Twin |
|---|---|---:|---:|
| within-cohort, 2024+ (n=7 / n=5) | era + colour | 0.813 | 0.644 |
| mature listings, within family (n=18 / n=11) | family | 0.762 | 0.686 |
| Sage Green's own launch | single cohort | 0.870 | 0.581 |
| incumbent heuristic | assumed | 0.760 | 0.600 |
| **blended weights used** | | **0.317** | **0.275** |

`src/size_structure.py` owns this and `tests/test_metrics.py` guards it. The demand model is trusted for *colour*; the size split is imposed. This revises an earlier conclusion of mine that the Twin bias was irreducible — that was the confound, not noise.

p90 itself is validated on the held-out window: theory says p90, the empirically cost-minimising fractile was p85 (cost index 768 vs p90's 795, a 3.4% gap), and the curve is flat p80–p90. Against the incumbent's flat 2,180 units (2,271), p90 cuts expected cost **65%**.

## Model suite

19 models across the paradigms, all scored on the same held-out window and the same rolling events.

| paradigm | models | rolling mean rank |
|---|---|---|
| instance-based | `knn_lookalike` (Lab-space colour matching to past launch curves) | **4.3 — best** |
| count GLM | `negbin_glm`, `poisson_glm` | 6.3, 6.7 |
| hierarchical Bayes | `hier_bayes` (Gibbs, partial pooling, posterior size ratios) | 7.6 |
| matrix completion | `matrix_factorisation` (Size × Colour ALS) | 7.7 |
| trees | `random_forest`, `xgboost`, `lightgbm` | 8.3, 11.0, 11.7 |
| **incumbent** | `heuristic` | 8.9 |
| regularised linear | `lasso`, `elasticnet`, `ridge` | 9.0, 9.1, 13.7 |
| combination | `ensemble` (NNLS on log scale, K-fold OOF) | 9.3 |
| clustering | `cluster_pool` (KMeans demand pools) | 10.3 |
| growth curves | `gompertz`, `logistic`, `bass` | 12.0, 12.4, 12.4 |
| naive | `naive_family`, `size_ratio` | 16.3, 13.0 |

**`knn_lookalike` wins** — matching a new colour to perceptually similar past launches beats every parametric alternative, with near-zero bias (0.04) and the best cohort-total accuracy (WAPE 0.198). It also gives a natural audit trail: `.explain()` returns which historical launches drove each forecast.

Two cautions worth stating plainly:

- **Single-window rankings are unreliable.** On the Jun–Jul window alone, `poisson_glm` and `naive_family` looked best. Across seven events, `naive_family` ranks **last**. The single-window result was partly luck.
- **The growth curves lose on point accuracy but earn their place elsewhere** — the fitted Gompertz ramp supplies the 61-day → 120-day horizon conversion (1.249x) that the order sizing depends on.

---

## Design decisions worth knowing

**`title` goes NULL from 2026-05 onward** in the sales/traffic table. Filtering the panel by `title LIKE '%duvet%'` — the obvious approach — silently truncates it at **2026-04-19** and destroys the entire test window. Membership is resolved through the SKU mapper instead.

**The curated no-deal calendar is unusable for this period**: 0 clean days in Jan–May 2026 and 16 in the test window. Promo days are detected instead from realised-ASP discounts (validated at recall 0.69 against the ASIN-level deal table) plus a robust MAD velocity screen. Promo days are *flagged, not dropped*, so models can condition on them.

**The mapper's own `color_family` is missing for 114 of 193 ASINs**, including every 2026 launch, and is inconsistent where present (the same ASIN filed under both "Multi" and "Navy Dot"). A complete taxonomy is derived from the colour string, and colours are mapped into **CIE Lab space** so `dist_to_white` is a continuous perceptual feature rather than a hand-drawn "near-white" list.

**Exposure, not window length.** A listing going live part-way through the window can only sell for the days it is live. Two Oversized King ASINs launched 9 days before the window closed; giving them 61 days of exposure produced large spurious over-forecasts.

**Zero-sale days are real observations.** Amazon reports zero-unit rows for live listings (36% of raw rows). Treating them as unobserved biased organic velocity upward and cut usable observations from 67.5% to 40.6% of the panel.

---

## Usage

```bash
make install
make extract      # BigQuery -> data/raw (cached; FORCE=1 to re-query)
make panel        # build the modelling panel
make diagnose     # re-estimate the heuristic's constants
make backtest     # held-out window, all 19 models
make rolling      # replay at 7 earlier launch events
make recommend    # order sheet
make report       # reports/FINDINGS.md
make order-sheet  # final order sheet for config/candidates.csv
make test
```

Or the whole pipeline: `make all`

Scoring a hypothetical colour that does not exist in the catalogue yet (e.g. an unentered family such as Terracotta):

```python
import pandas as pd
from src.recommend import recommend

candidates = pd.DataFrame({
    "colour": ["Terracotta", "Terracotta", "Sage Green"],
    "size":   ["Queen", "King", "Queen"],
})
print(recommend(candidates=candidates, model_name="knn_lookalike"))
```

Configuration lives in `config/config.yaml` — split dates, cost ratios, sizes, and programs.

---

## Layout

```
config/config.yaml     split dates, cost ratios, business constants
sql/                   BigQuery extraction (parameterised, no string interpolation of user data)
src/
  extract.py           BigQuery -> parquet, cached
  taxonomy.py          colour -> family / pattern / CIE Lab
  panel.py             daily panel: promo flags, launch age, family depth
  features.py          ASIN-level design matrices (cold and warm regimes)
  models/              19 forecasters behind one interface
  calibrate.py         conformal + wave-aware interval calibration
  backtest.py          held-out window + rolling-origin protocol
  recommend.py         horizon conversion, net demand, newsvendor sizing
  metrics.py           WAPE, rank metrics, interval scores, newsvendor cost
scripts/               runnable entry points
tests/                 21 tests, including temporal-leakage guards
```

## Limitations

- **19 listings in the held-out window, 101 training launch cohorts.** Small-sample. The rolling replay mitigates but does not eliminate this.
- **Interval coverage reaches ~0.67, not 0.80.** Launch-wave demand shifts faster than six historical waves can calibrate.
- **No ad-spend or competitor-price features.** Sponsored-product data exists in the warehouse and is the most obvious next lift, since launch support plausibly drives much of the wave-level shock.
- **Return rates are measured at family level**, not SKU level, which is too coarse for a colour with a genuine fit or dye-lot problem.
- **Twin ratios rest on 5 era-controlled cohorts.** Corroborated by 11 mature families, but this is the thinnest link in the size split.
- **The p90 buffer is large in absolute terms** (+153%). It follows correctly from the approved 9:1 ratio and the measured uncertainty, but it is sensitive to that ratio: at 4:1 the optimum drops to roughly p70–p80. The fractile ladder is printed with the sheet so the trade-off stays visible.
- **`advertising_deals_dates_asin_level_new` stops at 2025-12-02**, so ASIN-level deal flags cannot cover the test window; the ASP screen carries that load.
