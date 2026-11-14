# Pretrained satellite models: does one help?

A comparison experiment next to the main line (physical features + rule /
logistic regression / LightGBM in `src/fieldvision/model/`), not a
replacement for it. Code: `src/fieldvision/foundation/`, tests:
`tests/test_foundation.py`. Outputs (not in git):
`C:\t\fv\data\foundation\runs\<run>\` (REPORT.md, summary.json,
predictions.parquet, embeddings).

## Conclusion

**In plain words:** adding the pretrained satellite model (Presto) to our
feature model makes it slightly better, but giving the model the same
monthly satellite values directly, with no pretrained network, helps just
as much, so the gain comes from using the whole season's colours and not
from pretraining. Presto on its own is no better than our interpretable
features, so it stays a side experiment and the main line does not change.

Data: Austria 2024 only so far (2,400 fields, 20 blocks); the
leave-one-year-out check is below once Austria 2022 is extracted.

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
LightGBM/scikit-learn into `C:\t\fv\venv-fm` (1.3 GB on disk after
installation; the CUDA build of PyTorch, about 3 GB, was not needed).

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

All numbers: Austria 2024, `C:\t\fv\data\timeseries\at_2024\timeseries.parquet`
(listed in READY.txt at 12:02), labels `at_2024\fields.geojson`, 2,400
fields, 20 blocks, run `C:\t\fv\data\foundation\runs\at2024_v2`. Every
field had at least 5 usable months (1,831 had all 8), so no field was
dropped by any model; all rows below are scored on the same 2,400 fields.
The `phys-gbm` row reproduces the main line's own run on this data
exactly (recall 0.771, FPR 0.175, ROC-AUC 0.857), which confirms the folds
and features are the same. Threshold 0.5 on in-fold calibrated
probabilities; classes are balanced 50/50 by design.

Cell: pooled over the five held-out folds (fold mean ± fold SD).

| model | input | precision | recall | false pos. rate | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| majority | none | 0.500 | 1.000 | 1.000 | 0.667 | 0.500 |
| phys-rule | physical | 0.688 (0.69 ± 0.05) | 0.708 (0.71 ± 0.03) | 0.321 (0.32 ± 0.07) | 0.698 (0.70 ± 0.03) | 0.750 (0.75 ± 0.04) |
| phys-logistic | physical | 0.754 (0.75 ± 0.03) | 0.715 (0.72 ± 0.03) | 0.233 (0.23 ± 0.04) | 0.734 (0.73 ± 0.03) | 0.783 (0.78 ± 0.03) |
| **phys-gbm** (main line) | physical | 0.815 (0.82 ± 0.03) | 0.771 (0.77 ± 0.05) | 0.175 (0.17 ± 0.04) | 0.792 (0.79 ± 0.03) | 0.857 (0.86 ± 0.02) |
| emb-logistic | Presto | 0.801 (0.80 ± 0.02) | 0.790 (0.79 ± 0.03) | 0.197 (0.20 ± 0.02) | 0.795 (0.80 ± 0.02) | 0.858 (0.86 ± 0.02) |
| emb-gbm | Presto | 0.777 (0.78 ± 0.02) | 0.772 (0.77 ± 0.03) | 0.222 (0.22 ± 0.03) | 0.774 (0.77 ± 0.02) | 0.839 (0.84 ± 0.03) |
| both-logistic | physical + Presto | 0.816 (0.82 ± 0.03) | 0.785 (0.78 ± 0.03) | 0.177 (0.18 ± 0.03) | 0.800 (0.80 ± 0.03) | 0.867 (0.87 ± 0.02) |
| **both-gbm** | physical + Presto | 0.820 (0.82 ± 0.02) | 0.788 (0.79 ± 0.04) | 0.172 (0.17 ± 0.02) | 0.804 (0.80 ± 0.02) | 0.871 (0.87 ± 0.02) |
| raw-logistic | monthly composites | 0.812 (0.81 ± 0.03) | 0.764 (0.76 ± 0.05) | 0.177 (0.18 ± 0.03) | 0.787 (0.79 ± 0.03) | 0.856 (0.86 ± 0.03) |
| raw-gbm | monthly composites | 0.809 (0.81 ± 0.03) | 0.790 (0.79 ± 0.02) | 0.187 (0.19 ± 0.03) | 0.799 (0.80 ± 0.02) | 0.869 (0.87 ± 0.03) |
| physraw-gbm | physical + composites | 0.834 (0.83 ± 0.03) | 0.795 (0.80 ± 0.04) | 0.158 (0.16 ± 0.03) | 0.814 (0.81 ± 0.03) | 0.878 (0.88 ± 0.02) |
| rand-logistic | untrained network | 0.781 (0.78 ± 0.03) | 0.733 (0.73 ± 0.04) | 0.206 (0.21 ± 0.03) | 0.756 (0.76 ± 0.03) | 0.826 (0.83 ± 0.03) |
| rand-gbm | untrained network | 0.698 (0.70 ± 0.03) | 0.679 (0.68 ± 0.02) | 0.294 (0.29 ± 0.05) | 0.688 (0.69 ± 0.01) | 0.755 (0.76 ± 0.01) |
| ft-presto | Presto fine-tuned (below) | 0.769 (0.77 ± 0.02) | 0.844 (0.84 ± 0.03) | 0.253 (0.25 ± 0.02) | 0.805 (0.80 ± 0.02) | 0.870 (0.87 ± 0.02) |

Per-fold ROC-AUC: phys-gbm 0.891, 0.845, 0.849, 0.869, 0.841; both-gbm
0.909, 0.858, 0.860, 0.885, 0.849.

Paired ROC-AUC differences on the same fields (block-bootstrap 95 % CI,
1,000 resamples of whole blocks, plus the per-fold differences):

| comparison | pooled diff | 95 % CI | per fold | verdict |
|---|---:|---|---|---|
| both-gbm vs phys-gbm (the question asked) | +0.014 | [+0.007, +0.023] | +0.018, +0.014, +0.010, +0.017, +0.009 | better, small |
| both-logistic vs phys-logistic | +0.084 | [+0.066, +0.106] | +0.088, +0.043, +0.085, +0.103, +0.090 | better |
| both-gbm vs physraw-gbm | -0.007 | [-0.015, +0.002] | -0.008, -0.010, -0.008, +0.000, -0.007 | no clear difference |
| both-logistic vs physraw-logistic | +0.004 | [-0.005, +0.014] | -0.007, +0.016, +0.013, -0.002, -0.007 | no clear difference |
| emb-gbm vs phys-gbm | -0.018 | [-0.030, -0.005] | -0.005, +0.000, -0.025, -0.041, -0.020 | worse |
| emb-gbm vs raw-gbm | -0.029 | [-0.038, -0.020] | -0.027, -0.021, -0.021, -0.039, -0.032 | worse |
| emb-logistic vs raw-logistic | +0.002 | [-0.010, +0.012] | -0.003, +0.017, +0.012, -0.019, -0.011 | no clear difference |
| emb-gbm vs rand-gbm | +0.084 | [+0.060, +0.107] | +0.125, +0.081, +0.092, +0.066, +0.059 | better |
| emb-logistic vs rand-logistic | +0.032 | [+0.022, +0.042] | +0.024, +0.013, +0.039, +0.052, +0.026 | better |

**Reading.** (c) beats (b) by more than the fold-to-fold spread: adding
Presto to the physical features raises ROC-AUC by 0.014 for LightGBM
(every fold positive) and by 0.08 for logistic regression (whose six
physical features leave more on the table). The controls show where the
gain comes from. Adding the raw monthly composites instead of the Presto
embedding gives the same or a slightly larger gain (0.878 vs 0.871, no
clear difference), and Presto alone is no better than the composites alone
(logistic) or worse (LightGBM). Pretraining does matter compared with the
same network with random weights (+0.03 to +0.08), so Presto learned
something; it just does not beat a decent classifier on the composites it
is given. The useful finding for the main line is that whole-season
spectra carry about 0.02 ROC-AUC beyond the current physical features in
this region and year.

**Caveats.** One country, one year so far. The Presto and composite inputs
are tied to calendar months, while the main line's relative features are
measured from each field's own peak, so the calendar inputs may transfer
worse to another year or to Maryland. Presto was pre-trained on L1C
top-of-atmosphere data with least-cloudy mosaics; ours is L2A medians, a
domain shift that may cost it something. Monthly steps are coarse for a
question about a 2-4 week difference in harvest date.

**Response time.** For 2,400 fields: monthly composites 0.1 s, Presto
embeddings 9 s on the laptop CPU (about 4 s per 1,000 fields), physical
features 13 s; the heads predict in milliseconds.

### Light fine-tuning (step 6)

One run, kept separate from the frozen comparison. For each of the same
five folds, a fresh copy of the pretrained encoder had its last
transformer block and final LayerNorm unfrozen and a linear head added,
and was trained on that fold's training fields only (15 epochs, AdamW,
learning rate 1e-4 for the encoder and 1e-3 for the head, class-weighted
loss; all fixed before looking at results, no early stopping on the test
fold), then scored once on the test fold. About 40 s per fold on CPU.
ROC-AUC 0.870 (fold SD 0.02): better than the frozen embedding with a
logistic head (+0.012, CI [+0.006, +0.018]), no clear difference from the
composites with LightGBM (+0.001, CI [-0.008, +0.008]) or from the main
line (+0.013, CI [-0.003, +0.029]). Its threshold is a plain 0.5 without
in-fold calibration, which is why it sits at a different operating point
(recall 0.84, false positive rate 0.25); compare it on ROC-AUC.

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
