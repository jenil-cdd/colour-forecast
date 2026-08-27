-- Daily child-ASIN sales & traffic panel for the duvet programs.
--
-- NOTE: `title` and `program` in the sales table go NULL from 2026-05 onward,
-- which is why membership is resolved through the mapper (sql/01_sku_dim.sql)
-- rather than a title LIKE filter. Filtering on title would silently truncate
-- the panel at 2026-04-19 and destroy the entire test window.
SELECT
  DATE(s.date)                        AS date,
  s.child_asin,
  s.parent_asin,
  s.sessions,
  s.sessions_total,
  s.page_views,
  s.buy_box_percentage,
  s.units_ordered,
  s.units_ordered_b2b,
  s.unit_session_percentage,
  s.ordered_product_sales,
  s.total_order_items,
  -- realised average selling price; the only price signal available here
  SAFE_DIVIDE(s.ordered_product_sales, s.units_ordered) AS asp
FROM `{project}.{dataset}.sales_and_traffic_detail_sales_traffic_by_child_item_frau_filtered_final` AS s
INNER JOIN UNNEST(@asins) AS a ON a = s.child_asin
WHERE DATE(s.date) BETWEEN @history_start AND @history_end
