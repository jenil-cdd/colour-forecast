-- Global promo calendar.
-- `no_deals_dates` is a curated list of clean (no-deal) trading days and is
-- maintained through the present, so it is the primary clean-day filter.
-- `advertising_deals_dates_asin_level_new` gives ASIN-level deal days but stops
-- at 2025-12-02, so it is used only to enrich the training period.
SELECT date, TRUE AS is_no_deal_day
FROM `{project}.{dataset}.no_deals_dates`
WHERE date BETWEEN @history_start AND @history_end
