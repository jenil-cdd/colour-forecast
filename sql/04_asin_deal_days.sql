-- ASIN-level deal days (training period only; source stops 2025-12-02).
SELECT asin AS child_asin, deal_date AS date, ANY_VALUE(deal_type) AS deal_type
FROM `{project}.{dataset}.advertising_deals_dates_asin_level_new`
WHERE deal_date BETWEEN @history_start AND @history_end
  AND asin IN UNNEST(@asins)
GROUP BY child_asin, date
