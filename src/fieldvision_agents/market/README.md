# market: corn price and silage acres

Two models of the marketing-year corn price from USDA Quick Stats (`data/corn_market_history.csv`,
national, 2000-2026), and what an error in silage acres would mean for grain supply and price.

| File | What it is |
|---|---|
| `price_impact.py` | Four ordinary least squares models (no change, supply, stocks, stocks + last price), leave-one-year-out 2001-2024, 2025 held out |
| `stock_model_prediction.py` | The "regional embedding" ridge model (PCA of regional silage shares, convex stocks-to-use, yield trend) and a 2026-2028 projection |
| `loyo_clean_check.py` | The same models with the PCA, yield trend and embedding fill-in refitted inside each held-out year, next to last year's price |

## What the numbers say (clean check, 2001-2024 held out one year at a time)

| Model | Mean abs. error | Direction right | 2025 error |
|---|---|---|---|
| Last year's price | 15.9% | n/a | **+1.9%** |
| Linear stocks-to-use (this season's ending stocks) | 14.9% | 70.8% | -4.8% |
| Convex stocks-to-use (this season's ending stocks) | **14.6%** | 83.3% | -7.8% |
| Regional embedding (this season's ending stocks), refitted per fold | 14.7% | 83.3% | -8.4% |
| Convex stocks-to-use, last season's stocks | 19.6% | 54.2% | +1.9% |
| Regional embedding, last season's stocks | 19.1% | 62.5% | -0.8% |

- **These models explain a season's price; they do not forecast it.** They use the season's ending
  stocks, which are known only after the season. With only what is known when the season starts
  (last season's stocks), both do worse than last year's price. A forecast would need USDA's projected
  stocks at forecast time (WASDE), which is not in this data.
- **Barely ahead of "no change".** The best explaining model is 1.3 points better on average, and
  last year's price beat every model on 2025.
- **The leak was real but small.** Fitting the PCA, trend and fill-in on all years before the loop
  moved the embedding model's error from 14.71% to 14.68%; refitting per fold does not change the picture.

## The silage what-if

Acres cut for silage are not harvested for grain, so if the true silage area differed from USDA's by
D acres, grain production and ending stocks would differ by about D x yield. Each model turns that
into a price effect. **+1 point of national silage share in 2025** (971,350 acres, -181 million bushels):

| Model | Price effect |
|---|---|
| Supply (`price_impact.py`) | +0.85% |
| Linear / convex / embedding stocks-to-use (clean fits) | +4.2% / +5.2% / +5.0% |
| Stocks + last price (`price_impact.py`) | +5.0% |
| Stocks only (`price_impact.py`) | +9.4% |

The answer depends on the model, from under 1% to over 9% per point.

`project_3_years` in `stock_model_prediction.py` gives +$0.05 to +$0.08 per bushel (+1.9% to +2.5%)
for 2026-2028. That comes from the embedding model alone, applied to typed-in acres, yields and use
that also drive the baseline price down to $2.77, with a **+0.8 point silage shift that is an assumed
input** (`ml_silage_loss_pct=0.8`), not a model output. Our own 2025 four-state run differs from USDA
by +301,000 acres net (about +0.3 points nationally), and that net hides large errors both ways:
Pennsylvania 82,000 vs 315,000 acres, Wisconsin 1,139,000 vs 885,000, Iowa 661,000 vs 250,000,
Nebraska 89,000 vs 220,000. Until the silage estimate beats USDA's, present any price effect as a
what-if ("if silage were 1 point higher, then ..."), never as a finding.

Research estimates of supply-price relationships from public data; not trading advice.

```bash
python -m fieldvision_agents.market.loyo_clean_check src/fieldvision_agents/market/data/corn_market_history.csv rag-pipeline/dataset/quickstats/corn_grain_silage_acres_2012_2025.csv
```
