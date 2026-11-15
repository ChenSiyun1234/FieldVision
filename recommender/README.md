# Silage Match (recommender)

Matches dairies that are short on silage with nearby standing corn fields that are predicted grain-bound and could still be chopped.

## Run it (workspace terminal)

```bash
cd /workspace/FieldVision
export QUICKSTATS_CSV=/workspace/02_QuickStats/corn_grain_silage_acres_2012_2025.csv
python3 recommender/fetch_dairies.py                      # once: MDE permitted dairies -> data/dairies_md.csv
python3 recommender/silage_match.py 24021 2022 --as-of 2022-09-01
python3 recommender/silage_match.py 24021 2022 --as-of 2022-09-15 --shortfall 0.5 --corn-price 6.5 --llm
python3 recommender/map.py                                # builds out/map_*.html for every result
python3 recommender/silage_match.py 24021 2022 --dairies /workspace/data/usa_dairy_farms.csv   # team's EPA FRS file
python3 recommender/tests/test_silage_match.py            # 7 offline tests
```

From Python or the supervisor notebook:

```python
from silage_match import SilageMatchToolbox, TOOL_SPEC
tools = SilageMatchToolbox(CriticToolbox(Toolbox()))     # still has scout/classify/aggregate/run_critic
tools.recommend_sources("24021", 2022, as_of="2022-09-15", shortfall_frac=0.5)
```

**Inside the agent (wired in):** `recommend_sources` is in `agents/react/prompts.py` and `supervisor.py`. The supervisor may call it once, only after a `run_critic` result that is not `fail`; it uses the last policy classified with and never changes the county estimate. Backups of both files: `*.bak_before_recommender`. Full run: `cd /workspace && python3 FieldVision/recommender/run_agent_demo.py 24021 2022`. Tests: `python3 FieldVision/recommender/tests/test_supervisor_recommend.py` (plus the team's `python3 -m agents.react.test_supervisor`, still 10/10).

## How it works

| Step | What | Where the number comes from |
|---|---|---|
| 1. Prevalence fix | Shift p_silage from a 50/50 prior to the county's silage share in years **before** the target year | QuickStats via rag-pipeline |
| 2. Candidates as of a date | Not abstained, not harvested yet, grain-bound, 5+ acres, canopy still green (NDVI 0.5+, only data up to the date) | predictions + Sentinel-2 time series |
| 3. Requests | Each dairy asks for a share of its annual need (head x 10 t/yr). Or pass `--requests file.csv` | MDE herd sizes |
| 4. Matching | Min-cost max-flow: field capacity = (1 - p_adj) x acres x 20 t/ac; cost = haul $ + 8 $/t x p_adj; only edges within 20 road miles | networkx |
| 5. Critic | Code decides: requests covered, long hauls, uncertain sources, partial view, dairy coverage, hindsight window | deterministic |
| 6. Explain | Optional, Nemotron Ultra writes a short note from the result JSON | `--llm` |

Road miles = straight line x 1.3 until we have a road graph. All assumptions are in `Params` with sources in `SOURCES`.

## First results, Frederick 2022 (composite_v1, standard policy)

- **Prevalence fix:** county silage share goes from 36.5% raw to **20.0%** using only 2012 and 2017 history. Census 2022 says **19.6%**.
- **Sep 1, 15% shortfall:** 1,504 of 2,233 fields still choppable; 8 dairies in reach asked for 13,065 t, all matched from 48 fields, median haul 1.8 road miles. All checks pass, hindsight 100%.
- **Sep 15, 50% shortfall (dry year), corn at $6.50:** 1,151 fields left; 43,550 t matched from 142 fields at about $46 to $56 per ton delivered. The critic **warns**: only 78.2% of matched tons were on fields that really stayed standing two more weeks. The harvest window is closing, and the check catches it.

## Known limits

- MDE lists permitted operations only (14 dairies statewide, 6 in Frederick). Smaller dairies are missing, so real demand is higher.
- Field predictions use the full season. A production version would use an as-of classifier.
- Only fields inside the classified county are counted; dairies near the line get a partial view.
- Dairy IDs only, no operator names or addresses in outputs.
- The team file `/workspace/data/usa_dairy_farms.csv` (EPA FRS, 472 facilities) loads fine but has no herd size and no Maryland rows (mostly CA, AR, CO, IA, ID). For Frederick it finds no dairy in reach, so MDE stays the default. Missing herd sizes fall back to 300 head and are flagged.

## Files

- `fetch_dairies.py` downloads the MDE layer and keeps id, county, lat, lon, head
- `silage_match.py` does the core, critic, tool wrapper and CLI
- `map.py` builds a one-file Leaflet map (open it from the file browser, or copy it out)
- `tests/test_silage_match.py` runs on a fake county
- `out/` holds results (json, geojson, html)
