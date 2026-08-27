-- Sponsored Products spend and ad-attributed units, aggregated to ASIN x day.
--
-- Attribution column choice matters here:
--   * `_1_day_advertised_sku_units` is the cleanest same-day measure but is
--     NULL for every row before 2024 and for ~20% of 2024, so it cannot cover
--     the historical launch cohorts the ramp curve is fitted on.
--   * `_7_day_advertised_sku_units` is populated across the whole history.
--     Amazon attributes it to the *click* date, not the purchase date, so
--     summing it per day does not double-count: each click belongs to exactly
--     one date. It is carried as the primary measure for that reason.
-- Both are returned so the organic split can be sensitivity-checked on the
-- 2025+ period where they overlap.
SELECT
  DATE(date)                                   AS date,
  advertised_asin                              AS child_asin,
  SUM(spend)                                   AS ad_spend,
  SUM(clicks)                                  AS ad_clicks,
  SUM(impressions)                             AS ad_impressions,
  SUM(COALESCE(_7_day_advertised_sku_units, 0)) AS ad_units_7d,
  SUM(COALESCE(_1_day_advertised_sku_units, 0)) AS ad_units_1d,
  -- Non-null only from 2024 onward; used to decide when the 1-day series is usable.
  COUNTIF(_1_day_advertised_sku_units IS NOT NULL) AS n_rows_with_1d
FROM `{project}.{dataset}.ads_sponsored_products_advertised_product_frau_filtered`
WHERE DATE(date) BETWEEN @history_start AND @history_end
  AND advertised_asin IN UNNEST(@asins)
GROUP BY date, child_asin
