-- Daily child-ASIN sales & traffic panel for the duvet programs.
--
-- NOTE 1: `title` and `program` in the sales table go NULL from 2026-05 onward,
-- which is why membership is resolved through the mapper (sql/01_sku_dim.sql)
-- rather than a title LIKE filter. Filtering on title would silently truncate
-- the panel at 2026-04-19 and destroy the entire test window.
--
-- NOTE 2: the `_filtered_final` view is NOT one row per ASIN-day. It returns
-- split records -- one row carrying orders, another carrying traffic for the
-- same ASIN and date:
--
--     B07D2FRRFR  2020-02-09  sessions=0   units=14  sales=745.84
--     B07D2FRRFR  2020-02-09  sessions=72  units=0   sales=0.00
--
-- 3,164 rows across 1,584 ASIN-days and 57 ASINs were affected. Unit *totals*
-- were unharmed, but every per-row statistic was: a split day looked like two
-- days, one with zero units, so split rows averaged 0.80 units against 2.79 for
-- clean rows and read as 91% organic against 67%. That fed the rolling median,
-- the MAD spike screen and the zero-share features, i.e. the promo detection.
--
-- Fix: aggregate to exactly one row per ASIN-day. Counters are SUMmed; rates are
-- recomputed from the summed counters rather than averaged, because an average
-- of two ratios (one of which has a zero denominator) is meaningless.
SELECT
  DATE(s.date)                                  AS date,
  s.child_asin,
  ANY_VALUE(s.parent_asin)                      AS parent_asin,
  SUM(s.sessions)                               AS sessions,
  SUM(s.sessions_total)                         AS sessions_total,
  SUM(s.page_views)                             AS page_views,
  -- Buy-box share is a rate: weight it by sessions so a zero-traffic split row
  -- cannot drag it down. Falls back to a plain mean when there is no traffic.
  COALESCE(
    SAFE_DIVIDE(SUM(s.buy_box_percentage * s.sessions), SUM(s.sessions)),
    AVG(s.buy_box_percentage)
  )                                             AS buy_box_percentage,
  SUM(s.units_ordered)                          AS units_ordered,
  SUM(s.units_ordered_b2b)                      AS units_ordered_b2b,
  -- Conversion recomputed from summed counters, not averaged.
  SAFE_DIVIDE(SUM(s.units_ordered), SUM(s.sessions)) AS unit_session_percentage,
  SUM(s.ordered_product_sales)                  AS ordered_product_sales,
  SUM(s.total_order_items)                      AS total_order_items,
  -- Realised ASP from summed sales over summed units.
  SAFE_DIVIDE(SUM(s.ordered_product_sales), SUM(s.units_ordered)) AS asp,
  COUNT(*)                                      AS source_rows
FROM `{project}.{dataset}.sales_and_traffic_detail_sales_traffic_by_child_item_frau_filtered_final` AS s
INNER JOIN UNNEST(@asins) AS a ON a = s.child_asin
WHERE DATE(s.date) BETWEEN @history_start AND @history_end
GROUP BY date, s.child_asin
