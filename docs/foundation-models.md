# Pretrained satellite models: does one help?

A comparison experiment next to the main line (physical features + rule /
logistic regression / LightGBM in `src/fieldvision/model/`), not a
replacement for it. Code: `src/fieldvision/foundation/`, tests:
`tests/test_foundation.py`. Outputs (not in git):
`C:\t\fv\data\foundation\runs\<run>\` (REPORT.md, summary.json,
predictions.parquet, embeddings).

## Conclusion

RESULTS_PENDING

## Which model and why

YOLO and DINO are trained on natural RGB photos; our input is a 10-band
time series per field at 10-20 m, so they do not fit. Candidates checked:

| model | input it expects | fit to our data | weights / licence |
|---|---|---|---|
| **Presto** (Tseng et al. 2023, NASA Harvest) | per-pixel monthly series of S1, S2 (10 bands), ERA5, SRTM, Dynamic World, lat/lon; every channel group can be masked | close: we have S2 per field per date, everything else can be masked | 3.3 MB file in the public GitHub repo, MIT, no login |
| Galileo (2025, same group) | Presto-like plus image patches, more modalities | similar, but larger and built around space-time patches | public, MIT, but more setup |
| Prithvi-EO-2.0 (NASA/IBM) | HLS image chips (6 bands, 224x224 px, 3-4 dates) | poor: needs image chips, few dates, HLS bands | Hugging Face, Apache-2.0, 300M+ params |
| AnySat | multi-resolution image patches | poor for per-field medians | public, needs patches |

**Picked: Presto.** It is the only one whose native unit is a pixel (or
field) time series of Sentinel-2 bands, it masks absent inputs rather than
asking us to invent them, it is small (404k encoder parameters, runs on a
laptop CPU in seconds for thousands of fields; the GPU was not needed), and
its weights download without a login. What we gave up: image context
(Galileo/AnySat/Prithvi see neighbouring pixels; Presto sees our per-field
median only), radar and weather inputs that Presto was trained with, and
newer models' possible gains.

Downloaded: the Presto GitHub repo (shallow clone, 84 MB incl. sample data;
the weights file `data/default_model.pt` is 3.3 MB, SHA-256 in each run's
summary.json), the openmapflow 0.2.4 source archive (84 kB, only read to
check Presto's Sentinel-2 product), and a CPU-only PyTorch 2.15 wheel plus
LightGBM/scikit-learn into `C:\t\fv\venv-fm` (about 0.5 GB on disk).

## Method

**Encoder.** `presto_encoder.py` is a trimmed copy of Presto's encoder
(the original package imports Earth Engine and Google Cloud at import
time). On 120 real Austria 2024 fields it gives exactly the same
embeddings as the original code (max absolute difference 0.0).

**Input adapter** (`adapter.py`, unit-tested on a synthetic table):

- one time step per calendar month, April to November of the label year
  (our extraction window), so 8 steps; Presto's month input says the first
  step is April;
- per month, the median of each band over the field's usable acquisitions,
  with "usable" defined exactly as in the feature code
  (`features.season.usable_mask`: clear fraction >= 0.6, all bands finite
  and in range); a month with none is masked;
- **scaling:** Presto divides Sentinel-2 digital numbers by 10,000, so the
  model sees reflectance on a 0-1 scale; our table is already 0-1, so values
  go in unchanged (a test checks this). Two differences we cannot remove:
  Presto was pre-trained on Level-1C top-of-atmosphere reflectance
  (`COPERNICUS/S2` in openmapflow) while ours is Level-2A surface
  reflectance, and its monthly value was a least-cloudy mosaic per 30-day
  window, ours is a median;
- NDVI computed from the composite as Presto does;
- Sentinel-1, ERA5, SRTM and Dynamic World are masked (a test checks that
  values under a mask do not change the output);
- the field centroid as Presto's lat/lon input.

**Embedding.** Frozen encoder, one 128-number vector per field-season,
pooled the way Presto's authors do downstream: the mean of all non-masked
output tokens (one per channel group per observed month, plus the location
token), then the final LayerNorm. Fields with no usable month get no
embedding and are dropped from every model.

**Evaluation.** Same code as the main line: `model.splits.spatial_folds`
(5 folds, grouped by `block_id`, seed 0, computed on all labelled fields
of the run, exactly as `model.train` does), `model.models` heads
(logistic regression with in-fold median imputation and scaling;
LightGBM with the main line's fixed parameters; both with in-fold Platt
calibration grouped by block), `model.harness.run_splits` and
`summarize_predictions`, positive class silage. Fields that any input could
not handle were removed from the train and test parts of every model, so
all rows of a table are scored on the same fields. Inputs compared on
identical folds:

- `phys`: the main line's physical features, variant "relative" (rule,
  logistic regression on its 6 features, LightGBM on all signal features);
- `emb`: Presto embeddings alone;
- `both`: physical features + embeddings;
- controls added to tell "pretraining helps" from "more data helps":
  `raw` = the same monthly composites Presto sees (8 months x 11 values),
  given straight to the heads; `rand` = the same network with random
  weights; `physraw` = physical features + raw composites;
- `majority`: always the training majority class.

"Better" or "worse" is only claimed when the mean per-fold difference in
ROC-AUC is larger than its fold-to-fold standard deviation and the paired
block-bootstrap 95 % interval excludes zero; otherwise "no clear
difference".

## Results

RESULTS_PENDING

## Field plus surroundings

The idea raised in the team: let a model look at a field and at the fields
around it (several heads, cross-attention between the field and its
neighbours), because silage fields cluster around dairies and neighbouring
farms harvest in step.

**What a context-aware model would need as input.** For every target field:
its own time series; the time series (or embeddings) of every corn field
within some radius, say 2-5 km, with their distances and directions; and,
for training, labels that reflect the real mix of silage and grain in that
neighbourhood. It would also need many neighbourhoods per region so it
learns "how this field differs from its neighbours" rather than "what this
region looks like".

**Why our current label files cannot train it directly.** Every label file
was built by sampling the same number of silage and grain fields inside
each 10 km block (60 + 60 in Austria). The neighbourhood a model would see
is therefore not the real one: the share of silage around any field is
about 50 % by construction, whatever it is on the ground (in Austria
silage maize is very roughly a quarter to a third of maize area, in the US
about 6-8 % of corn, and both vary a lot from county to county). A model trained on these neighbourhoods would learn a relationship
between a field and its neighbours that does not exist in Maryland. On
top of that, we only fetched the sampled fields, not all corn fields
around them, so most real neighbours are missing from our time series.
Fixing this means fetching every corn parcel in a set of blocks
(unbalanced, as declared) and extracting time series for all of them:
several times the current extraction, and still a different landscape
from Beltsville.

**The cheap version.** Keep the per-field model and add a few features
computed relative to the neighbourhood, from fields that are in the data
anyway, with no labels needed:

- harvest date minus the median harvest date of other corn fields within
  a few km in the same year (an early outlier is more likely silage than
  an early field in an early region);
- pre-harvest moisture (NDMI) minus the neighbourhood median, same idea;
- for US fields, distance to the nearest dairy (public lists of licensed
  dairies by state, or USDA census counts of milk cows by county as a
  coarser prior).

The first two use only satellite data and the unlabelled neighbours, so
the balanced sampling of labels does not bias them; they need the
time series of nearby corn fields, which for the US come from the
Cropland Data Layer corn mask. They should be added to the main
feature table and evaluated with the same spatial folds, like any other
feature. Distance to a dairy is a prior about the farm, not about the
field's harvest, and should be reported separately so the satellite
evidence stays explainable on its own.
