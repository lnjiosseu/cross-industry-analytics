-- sql/churn_segmentation.sql
-- Churn rate per industry

SELECT
    industry,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct
FROM industry_modeling_data
GROUP BY industry
ORDER BY churn_rate_pct DESC;