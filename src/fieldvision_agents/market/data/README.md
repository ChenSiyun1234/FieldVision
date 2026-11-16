# corn_market_history.csv

USDA NASS Quick Stats API, national level, pulled 2026-11-15 (survey, reference period YEAR unless noted):

| column | Quick Stats series |
|---|---|
| prod | CORN, GRAIN - PRODUCTION, MEASURED IN BU |
| grain_ac | CORN, GRAIN - ACRES HARVESTED |
| silage_ac | CORN, SILAGE - ACRES HARVESTED |
| yield | CORN, GRAIN - YIELD, MEASURED IN BU / ACRE |
| price_my | CORN, GRAIN - PRICE RECEIVED, MEASURED IN $ / BU, MARKETING YEAR (Sep Y to Aug Y+1) |
| p_sep ... p_jan | same price, monthly, calendar months of year Y |
| prod_aug, prod_nov | production forecasts, YEAR - AUG FORECAST and YEAR - NOV FORECAST |
| sep1_stocks | CORN, GRAIN - STOCKS, MEASURED IN BU, FIRST OF SEP (ending stocks of the previous marketing year) |

Stocks-to-use for marketing year Y: ending stocks = sep1_stocks of Y+1; use = sep1_stocks of Y + prod of Y - ending stocks (imports ignored, under 1 % of supply).

2026 is in progress (production is the November forecast; no marketing-year price yet).
