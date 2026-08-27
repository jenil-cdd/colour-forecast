-- Customer returns aggregated to ASIN x day, for net-demand and margin work.
SELECT
  DATE(return_date) AS date,
  asin              AS child_asin,
  SUM(quantity)     AS returned_units,
  COUNTIF(LOWER(COALESCE(reason,''))  LIKE '%color%'
       OR LOWER(COALESCE(customer_comments,'')) LIKE '%color%'
       OR LOWER(COALESCE(customer_comments,'')) LIKE '%colour%') AS colour_related_returns
FROM `{project}.{dataset}.returns_customer_returns_frau_filtered`
WHERE DATE(return_date) BETWEEN @history_start AND @history_end
  AND asin IN UNNEST(@asins)
GROUP BY date, child_asin
