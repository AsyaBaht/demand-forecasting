-- Weekly bottles-sold demand per (store, liquor_type), top-25 stores by
-- volume, 2019-onward. See README "Why top-25 stores, why weekly" for the
-- reasoning behind the scope cuts baked into this query.
WITH categorized AS (
  SELECT
    store_number,
    date,
    bottles_sold,
    CASE
      WHEN STARTS_WITH(category, '101') THEN 'whiskey'
      WHEN STARTS_WITH(category, '102') THEN 'tequila'
      WHEN STARTS_WITH(category, '103') THEN 'vodka'
      WHEN STARTS_WITH(category, '104') THEN 'gin'
      WHEN STARTS_WITH(category, '105') THEN 'brandy'
      WHEN STARTS_WITH(category, '106') THEN 'rum'
      WHEN STARTS_WITH(category, '108') THEN 'cordial_liqueur'
      ELSE 'other'
    END AS liquor_type
  FROM `bigquery-public-data.iowa_liquor_sales.sales`
  WHERE date >= '2019-01-01'
    AND store_number IS NOT NULL
    AND category IS NOT NULL
),
top_stores AS (
  SELECT store_number
  FROM categorized
  GROUP BY store_number
  ORDER BY SUM(bottles_sold) DESC
  LIMIT 25
)
SELECT
  DATE_TRUNC(c.date, WEEK(MONDAY)) AS week_start,
  c.store_number,
  c.liquor_type,
  SUM(c.bottles_sold) AS bottles_sold
FROM categorized c
JOIN top_stores t USING (store_number)
GROUP BY week_start, store_number, liquor_type
ORDER BY store_number, liquor_type, week_start
