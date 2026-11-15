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
every other registry model (including `eu_logistic_v1`, which `standard` uses, and `composite_v1`) can.

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

The outputs below were recorded when `standard` still meant `composite_v1` (now the `module_v1`
policy); under module_v2 the model name is `eu_logistic_v1`, the members list is gone and each
field's evidence also carries `evidence_tier`. The keys are the same.

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
 "models_used": {"composite_v1": 7},
 "silage_acres_adjusted": ..., "silage_acres_adjusted_low": ..., "silage_acres_adjusted_high": ...,
 "grain_acres_adjusted": ..., "silage_share_counted": ..., "silage_share_adjusted": ...,
 "adjustment_method": "adjusted classify-and-count, European held-out TPR/FPR",
 "adjustment_clipped": false, "adjusted_share_by_rates": {...}, "notes": "..."}
```

The stub's keys come first and keep their meaning; the adjusted keys are explained in
"Reading the county total" below. Silage acres = sum over non-abstained fields of area x p_silage; grain acres the same with
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

## Reading the county total

**Under module_v2 (the `standard` policy now)** the headline is the local-threshold share of the
clear evidence tier, against a same-season reference county from `<data root>/references.json`;
without one it is the counted share of the clear tier. `headline_method` says which. The
`clear_*` keys hold the four clear-tier estimates side by side, with `clear_tier_share_of_area`,
`wide_tier_acres` and `wide_tier_share_called_silage` beside them. Rules, validation and limits:
`docs/module.md`. The rest of this section describes the rule that came before (still used by the
other policies).

**Headline rule (fixed at 16:12 on 14 Nov, before any full-county total existed, and logged in
the evaluation access log):** the headline county estimate is the **counted share**,
`silage_share_headline` = the share of the non-abstained corn area that the policy's model calls
silage. Beside it, with their ranges: the adjusted share and its spread across European
validation designs, the raw sum of probabilities, and the prior-shift (EM) re-estimate. A wide
spread of the adjusted share is itself the warning that error rates measured in Europe do not
carry over to the county.

Why the counted share: it needs no error rates from Europe, which did not hold on the partial
Frederick snapshot, and it does not assume half the fields are silage. Its known bias, stated in
advance: in expectation it equals TPR x true share + FPR x (1 - true share), so it reads high
where the true silage share is below the model's false-positive rate (about 0.16 in Europe) and
low where silage is common. The rule was chosen after seeing the partial Frederick snapshot, so
Frederick is no longer a clean test of it; Queen Anne's County (24035, Census silage share 1.3 %)
was declared the untouched check and is run only after the rule was logged.

`aggregate` keeps the stub's keys with their meaning and adds these estimates beside them.

- `silage_acres` (and its range) sums area x `p_silage`. The probabilities are calibrated for the
  training data, which is half silage, so this sum assumes half the fields are silage and reads
  high wherever silage is the minority (most US counties).
- `silage_acres_adjusted` (with `_low`, `_high`) counts the area called silage and corrects it
  with the model's true- and false-positive rates measured on held-out European fields only
  (no US labels): adjusted share = (counted share - FPR) / (TPR - FPR). The range resamples the
  fields and draws both rates from their uncertainty. `grain_acres_adjusted` is the rest of the
  area that was not abstained.
- `adjusted_share_by_rates` repeats the adjustment with the rates from each European validation
  design (spatial CV, each Austrian year held out) so the spread is visible. Rates measured in
  Austria may not hold in a US county; when they differ a lot, the total is not trustworthy.
- A share that would fall below 0 or above 1 is clipped and `adjustment_clipped` says so.
- `silage_share_prior_shift` (with `_low`, `_high`, `prior_shift_converged`,
  `prior_shift_iterations`) re-estimates the share from the county's own probabilities by the
  expectation-maximisation prior-shift correction, started from 0.5. It needs no error rates,
  but it trusts the probabilities to be calibrated; where they are not, it runs to 0 or 1.
- Abstained acres are in neither total. `notes` says all of this in one paragraph.

Example, Frederick County MD (24021), 2022, **partial snapshot: 248 of 2,233 fields, 11 % of the
area, 7 neighbouring extraction blocks**, `standard` policy (composite_v1):

| | silage share | silage acres |
|---|---|---|
| raw sum of p (assumes half silage) | 33 % | 1,397 [1,136, 1,661] |
| **counted (area called silage): headline** | 18.9 % | |
| prior-shift (EM) re-estimate | 0 % (converged at the boundary) | |
| adjusted with European rates (TPR 0.70, FPR 0.16) | 4.9 % | 205 [0, 642] |
| adjusted, other European designs | 0 % (clipped) to 5.3 % | |
| USDA Census 2022, whole county | 19.6 % | 7,748 of 39,500 |

On this snapshot the adjustment overcorrects: Austria's false-positive rate (0.16) is much
higher than what the European logistic showed on the Beltsville fields (0.03), so subtracting it
removes real silage. Read the counted share, the adjusted share and their spread together, and
compare with the county's Census figure before quoting either. To be replaced with the full-county
numbers when the extraction finishes.

## What the policies really do

| policy | model | abstains when |
|---|---|---|
| `standard` (= `module_v2`) | `eu_logistic_v1`, the six-feature logistic trained on Austrian registry labels. Each field gets an evidence tier in `evidence.evidence_tier`: `clear` (harvest bracket <= 12 days), `wide` (12-20 days) or `abstain`; the county headline uses the clear tier only (see `docs/module.md`) | low evidence (harvest bracket over 20 days, fewer than 6 usable dates, no or unconfirmed drop) |
| `module_v1` | `composite_v1`: mean of the calibrated probabilities of the timing rule, the timing+moisture rule and the six-feature logistic; the standard policy before module_v2 | low evidence, or the three members split their votes and are at least 0.5 apart |
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

- `eu_logistic_v1` and `composite_v1` are trained on Austria 2024 only. `eu_logistic_v1` was
  scored once on the Beltsville labels in Evaluation 1 (42 of 44, secondary); `composite_v1` has
  not been (it is a candidate for the second, final evaluation).
- The county headline rests on the clear tier, which may not be a fair sample of the county;
  read `clear_tier_share_of_area` next to it.
- Fields without a clear observation before and after the cut are flagged, not guessed.
- The toolbox never reads labels; any `label` column in a field file is dropped on load.
