# Using the real toolbox in the agent

`fieldvision.toolbox.Toolbox` has the same method names and return keys as the agent's
`StubToolbox` (`agents/react/tools_stub.py`), without the `stub` flag. Every number comes from
the field polygons, the Sentinel-2 series and the frozen models in `registry/`.

## Swap it in

In the workspace the repo is at `/workspace/FieldVision/`. Where `supervisor.py` builds the stub:

```python
import sys
sys.path[:0] = ["/workspace/FieldVision/.deps", "/workspace/FieldVision/src"]
from fieldvision.toolbox import Toolbox

# tools = StubToolbox()
tools = Toolbox()          # data/counties, registry and results, all inside the repo
```

The workspace's shared environment has no Parquet engine, so pyarrow is installed privately
into the repo's `.deps/` folder (ignored by git), leaving the shared environment untouched:

```
pip install --target /workspace/FieldVision/.deps pyarrow
```

Without it `Toolbox()` stops with a one-line message naming this command. There is no
`lightgbm` in the workspace either, so `eu_gbm_v1` and `ts_model_v1` cannot be loaded there;
every other registry model (including `composite_v1`, which `standard` uses) can.

`Toolbox(data_root="data/counties", registry="registry", out_root="results")` are the defaults.
Relative paths are taken from the repo root, so it does not matter where the agent is started.
Pass absolute paths to use other folders.

Python packages needed at run time: `numpy`, `pandas`, `pyarrow`, `scikit-learn` (it brings
`scipy`, `joblib`, `threadpoolctl`). Not needed: `lightgbm` (only if you load `eu_gbm_v1` /
`ts_model_v1`), `matplotlib`, `geopandas`, `shapely` (only for field files without
`centroid_lon`, `centroid_lat` and `area_ha`). Checked by running every method with those
packages blocked.

## Try it in one command

```
python3 /workspace/FieldVision/scripts/toolbox_demo.py 24033 2022
```

It calls `scout`, `route_options`, `extract_features`, `classify` and `aggregate` on the demo
county and prints each return value and the first two per-field results. The arguments are
`fips year policy`; the script sets the two paths itself.

## Data layout

```
data/counties/<fips>/<year>/fields.geojson      field polygons, EPSG:4326, one feature per field;
                                                properties field_id, year, area_ha (labels ignored)
data/counties/<fips>/<year>/timeseries.parquet  the extractor's long table: field_id, date,
                                                clear_fraction, B02..B12 (0-1 reflectance)
```

`data/` is not in git. The demo county is a label-free copy of the Beltsville fields under FIPS
24033 (Prince George's County, MD), seasons 2021-2023, built with
`fieldvision.toolbox.make_pseudo_county`. Frederick County (24021, 2022) will use the same layout.

## The six methods, with real output (24033, 2022, 7 fields)

`scout(fips, year)`: what data there is.

```python
{"fips": "24033", "year": 2022, "n_fields": 7, "fields_with_clear_series": 5,
 "fields_with_cloud_gap_radar_ok": 0, "fields_with_little_data": 0,
 "median_clear_obs_jul_nov": 27.0, "largest_optical_gap_days_p90": 25.0,
 "notes": "no radar data or radar model in this toolbox, so no field is counted as rescued by radar; 'clear series' = not low evidence under the optical rule (>= 6 usable dates Jul-Nov, harvest bracket <= 20 days, drop confirmed)"}
```

`route_options(fips, year)`: what each policy would do.

```python
{"fips": "24033", "year": 2022,
 "routing_table": "standard: every field to composite_v1; abstain on 2 of 7 fields\nradar_first: identical to 'standard': no radar model exists, so no field is routed to one; abstain on 2 of 7 fields\nregional_threshold: eu_timing_rule_v1 with its drop-date threshold moved by -2.5 days: this county's median drop day 250.0 minus the training median 252.5; abstain on 2 of 7 fields",
 "options": {"standard": {"models_used": {"composite_v1": 7}, "n_abstain": 2, "note": "..."},
             "radar_first": {"models_used": {"composite_v1": 7}, "n_abstain": 2, "note": "..."},
             "regional_threshold": {"models_used": {"eu_timing_rule_v1+regional_shift": 7}, "n_abstain": 2, "note": "..."}},
 "notes": "radar_first cannot differ from standard here (no radar model)"}
```

`extract_features(fips, year)`: writes `results/features/<fips>_<year>.parquet`.

```python
{"fips": "24033", "year": 2022, "rows": 7, "path": ".../results/features/24033_2022.parquet",
 "quality": {"rows_with_gap_filled": 7}}
```

Nothing is imputed into the table; missing values stay missing. `rows_with_gap_filled` counts
fields whose greenness integrals interpolate across a gap longer than 15 days (the only values
computed across a gap).

`classify(fips, year, policy)`: writes one result per field.

```python
{"fips": "24033", "year": 2022, "policy": "standard", "n_fields": 7,
 "models_used": {"composite_v1": 7}, "n_abstain": 2,
 "results_path": ".../results/MD_2022_24033_standard.parquet"}
```

`aggregate(fips, year)`: county totals from the last `classify` of that county-year
(runs `classify(..., "standard")` first if there was none).

```python
{"fips": "24033", "year": 2022, "silage_acres": 47.7, "silage_acres_low": 16.1,
 "silage_acres_high": 73.5, "grain_acres": 25.8, "abstained_acres": 36.8, "n_fields": 7,
 "models_used": {"composite_v1": 7}}
```

Silage acres = sum over non-abstained fields of area x p_silage; grain acres the same with
1 - p_silage; the range is the 5th-95th percentile of 2,000 seeded draws where each field is
silage with its own probability. Abstained fields are only in `abstained_acres`, never in grain.
Area comes from the polygons (hectares x 2.47105). The probabilities assume a 50/50 class
balance (how the training data was sampled), so in a county where silage is rare the expected
silage acres will read high; compare with USDA's county figures before trusting a total.

`abstain_region(fips, year, reason)`: records that the whole county-year is not called.

```python
{"fips": "24033", "year": 2022, "abstained": True, "reason": "...", "n_fields": 7,
 "abstained_acres": 110.3}
```

The stub's `run_critic` is not here; it belongs to the agent.

## What the policies really do

| policy | model | abstains when |
|---|---|---|
| `standard` | `composite_v1`: mean of the calibrated probabilities of the timing rule, the timing+moisture rule and the six-feature logistic, all trained on Austrian registry labels | low evidence (harvest bracket over 20 days, fewer than 6 usable dates, no or unconfirmed drop) or the three members split their votes and are at least 0.5 apart |
| `radar_first` | the same as `standard`. There is no radar model behind it; the policy exists so the agent's routing logic runs, and its note says so | as `standard` |
| `regional_threshold` | `eu_timing_rule_v1` with its drop-date threshold moved by (this county's median drop day minus Austria's median drop day, 252.5) | low evidence |
| `in_site` | `eu_barc_logistic_v1`, fitted on the Beltsville fields themselves plus Austria 2024 (equal weight). Only for FIPS 24033; elsewhere `route_options` lists it as not available and `classify` refuses it. On 24033 its calls are in-sample; its out-of-fold accuracy (43 of 44) is in its model card | low evidence |

## Result files

`results/<STATE>_<year>_<fips>_<policy>.parquet` and `.jsonl`, one object per field:

```json
{"field_id": "BARC-231", "year": 2022, "source_field_id": "barc-231-2022", "model": "composite_v1",
 "p_silage": 0.7168, "label": "silage", "abstain": false,
 "evidence": {"drop_date": "2022-09-15", "last_standing_date": "2022-08-26", "days_to_drop": 20.0,
              "ndvi_before": 0.826, "ndvi_after": 0.276, "ndmi_before": 0.373,
              "greenness_before_share_of_peak": 0.923, "days_peak_to_drop": 48.0,
              "residue_ndti_after": 0.11, "n_usable_dates": 30, "largest_gap_days": 20.0,
              "abstain_reason": null,
              "members_p_silage": {"timing_rule": 0.568, "rule": 0.745, "logistic": 0.838}}}
```

`results/abstained_regions.jsonl` collects `abstain_region` calls. These outputs are ignored by
git; the committed `results/MD_<year>.jsonl` files are a fixed worked example (see
`results/README.md`).

## Limits to keep in mind

- `composite_v1` is trained on Austria 2024 only. It has not been scored on the Beltsville
  labels (it is a candidate for the second, final evaluation).
- Fields without a clear observation before and after the cut are flagged, not guessed.
- The toolbox never reads labels; any `label` column in a field file is dropped on load.
