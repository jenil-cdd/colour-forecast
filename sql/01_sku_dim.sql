-- SKU dimension for duvet programs.
-- The mapper has duplicate child_asin rows (the same ASIN can be listed under
-- more than one color_family, e.g. B089DHDSWN under both 'Multi' and 'Navy Dot'),
-- so we deduplicate deterministically to one row per ASIN.
WITH ranked AS (
  SELECT
    child_asin,
    parent_asin,
    parent            AS program,
    category,
    sku,
    thread_count,
    colour,
    size,
    color_family      AS color_family_src,
    status,
    f1_analysis,
    ROW_NUMBER() OVER (
      PARTITION BY child_asin
      ORDER BY
        -- prefer rows that actually carry a color_family, then stable tiebreak
        CASE WHEN color_family IS NOT NULL THEN 0 ELSE 1 END,
        color_family,
        sku
    ) AS rn
  FROM `{project}.{mapper}`
  WHERE parent IN UNNEST(@programs)
)
SELECT * EXCEPT (rn)
FROM ranked
WHERE rn = 1
