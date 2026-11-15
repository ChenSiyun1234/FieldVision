# Pretrained satellite models: does one help?

A comparison experiment next to the main line (physical features + rule /
logistic regression / LightGBM in `src/fieldvision/model/`), not a
replacement for it. Code: `src/fieldvision/foundation/`, tests:
`tests/test_foundation.py`. Outputs (not in git):
`<data>\foundation\runs\<run>\` (REPORT.md, summary.json,
predictions.parquet, embeddings).

## Conclusion

**In two plain sentences:** a pretrained satellite model (Presto) did no
better than simply handing our model the same monthly satellite values,
so the small gain it adds inside Austria comes from the extra season-long
data, not from pretraining. When the country changes, the richer models
lose far more than the 6-feature logistic (with three European countries
it loses at most 0.015 ROC-AUC, they lose 0.02-0.19), so across countries
the ordering depends on the pair, from the simple model winning outright
(trained on Austria, tested on France, as in Maryland) to the richer ones
keeping a small lead (tested on Austria), and in the Netherlands every
model is equally poor.

What that means for the main line: no change. Presto is not worth its
extra complexity here, and it transfers worst of all (it loses 0.10-0.20
ROC-AUC across countries). Richer inputs win within Austria and across
Austrian years, but across countries only the 6-feature logistic keeps its
in-country accuracy (it loses 0.00-0.02 ROC-AUC, the richer models
0.06-0.20), which matches the modelling session's one-time US check (as
reported to us: 6-feature logistic 42 of 44 correct, LightGBM 30 of 44;
not re-checked here, this experiment never read the US data).

Data: Austria 2024 (2,400 fields, 20 blocks) for the main table;
Austria 2022 + 2024 (3,600 fields) for the first leave-one-year-out;
Austria 2022, 2023, 2024 + France 2022 (6,400 fields) for
leave-one-country-out and the country-to-country pairs; the same plus the
Netherlands 2025 (8,000 fields) for the three-country hold-out.

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
LightGBM/scikit-learn into `<venv>-fm` (1.3 GB on disk after
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

All numbers: Austria 2024, `<data>\timeseries\at_2024\timeseries.parquet`
(listed in READY.txt at 12:02), labels `at_2024\fields.geojson`, 2,400
fields, 20 blocks, run `<data>\foundation\runs\at2024_v2`. Every
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

### Transfer to a new year (leave-one-year-out)

Data: Austria 2022 (`at_2022\timeseries.parquet`, listed 13:55 local,
1,200 fields in 10 blocks that are all also in 2024) + Austria 2024
(2,400 fields), run `<data>\foundation\runs\at2022_2024`. Train on
one year, test on the other, same models and settings as above; the
training year's fields in the same blocks are kept (the main line's
default). "AUC in" is the same held-out fields scored by the spatial-CV
models of the same run, whose training did include that year (other
blocks); "loss" = in - out. The last column is paired against the main
line's 6-feature logistic regression on the same fields, block-bootstrap
95 % CI. Note the two years also differ in imagery source (2022 from
Planetary Computer, 2024 from Earth Search Collection 1), so "a new year"
here also means "slightly different processing".

| test year | model | accuracy | recall | FPR | AUC out | AUC in | loss | AUC vs 6-feature logistic |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2022 | phys-logistic (6 features) | 0.629 | 0.605 | 0.347 | 0.690 | 0.715 | 0.025 | reference |
| 2022 | phys-rule | 0.637 | 0.780 | 0.507 | 0.708 | 0.740 | 0.033 | +0.018 [-0.005, +0.039] |
| 2022 | phys-gbm (main line) | 0.706 | 0.802 | 0.390 | 0.780 | 0.810 | 0.030 | +0.090 [+0.058, +0.122] |
| 2022 | emb-logistic (Presto) | 0.701 | 0.668 | 0.267 | 0.771 | 0.786 | 0.016 | +0.081 [+0.042, +0.120] |
| 2022 | raw-gbm | 0.664 | 0.698 | 0.370 | 0.744 | 0.781 | 0.037 | +0.054 [+0.012, +0.103] |
| 2022 | raw-logistic | 0.724 | 0.653 | 0.205 | 0.789 | 0.788 | -0.002 | +0.100 [+0.067, +0.137] |
| 2022 | both-gbm (phys + Presto) | 0.703 | 0.827 | 0.420 | 0.797 | 0.823 | 0.026 | +0.107 [+0.076, +0.140] |
| 2022 | physraw-gbm (phys + raw) | 0.698 | 0.802 | 0.405 | 0.790 | 0.819 | 0.029 | +0.101 [+0.073, +0.129] |
| 2022 | majority | 0.500 | 1.000 | 1.000 | 0.500 | 0.500 | 0 | -0.190 |
| 2024 | phys-logistic (6 features) | 0.676 | 0.769 | 0.417 | 0.743 | 0.780 | 0.038 | reference |
| 2024 | phys-rule | 0.701 | 0.707 | 0.305 | 0.749 | 0.743 | -0.006 | +0.007 [-0.006, +0.020] |
| 2024 | phys-gbm (main line) | 0.764 | 0.695 | 0.167 | 0.826 | 0.862 | 0.036 | +0.083 [+0.049, +0.110] |
| 2024 | emb-logistic (Presto) | 0.759 | 0.700 | 0.182 | 0.829 | 0.862 | 0.033 | +0.087 [+0.055, +0.119] |
| 2024 | raw-gbm | 0.737 | 0.793 | 0.319 | 0.824 | 0.870 | 0.046 | +0.081 [+0.048, +0.114] |
| 2024 | raw-logistic | 0.770 | 0.829 | 0.290 | 0.843 | 0.861 | 0.018 | +0.100 [+0.066, +0.136] |
| 2024 | both-gbm (phys + Presto) | 0.777 | 0.727 | 0.172 | 0.846 | 0.880 | 0.034 | +0.103 [+0.072, +0.125] |
| 2024 | physraw-gbm (phys + raw) | 0.780 | 0.738 | 0.179 | 0.843 | 0.885 | 0.042 | +0.100 [+0.069, +0.123] |
| 2024 | majority | 0.500 | 1.000 | 1.000 | 0.500 | 0.500 | 0 | -0.243 |

(emb-gbm, the logistic combinations and the controls are in the run's
REPORT.md and transfer_table.csv.)

**Do calendar-month inputs lose more when the year changes?** Not
detectably within Austria. Losses are 0.02-0.05 ROC-AUC for nearly
every model in both directions; the calendar inputs (raw-gbm 0.037 /
0.046, Presto emb-gbm 0.014 / 0.053) sit in the same range as the
relative-timing physical features (phys-gbm 0.030 / 0.036, 6-feature
logistic 0.025 / 0.038), and with two test years and differences of a
hundredth or two there is no clear difference. Austria's harvest
calendar does not move much between 2022 and 2024, so this test is too
gentle to expose a calendar dependence; the US is the real test.

**Does any combination beat the 6-feature logistic out of year?** Yes,
within Austria: every richer model (main-line LightGBM, Presto, raw
values, and the combinations) beats it by 0.05-0.11 ROC-AUC in both
directions with intervals that exclude zero; physical + Presto and
physical + raw are the best (about +0.10). This is the opposite of what
the US check found, which says the gap between Austrian years is not a
good stand-in for the gap between Austria and Maryland. A transfer test
that matters needs a second country (Netherlands or France, not yet
extracted), run with the same command (`compare.py` adds
leave-one-country-out automatically when the field files cover two
countries).

Spatial CV on the combined 3,600 fields repeats the Austria 2024 pattern:
both-gbm vs phys-gbm +0.017 [+0.010, +0.024]; both-gbm vs physraw-gbm
-0.002 [-0.006, +0.003], no clear difference; emb-gbm vs raw-gbm -0.023
[-0.032, -0.014], worse.

### Transfer to a new country (leave-one-country-out)

Data: Austria 2022, 2023, 2024 (4,800 fields) and France 2022 (1,600
fields, listed 01:02Z on 15 November), run
`<data>\foundation\runs\at_fr`, 6,400 fields, none dropped. Same
models and settings; "trained on Austria" means all three Austrian years.
Cells: ROC-AUC on the held-out country, then the paired difference to the
6-feature logistic on the same fields with its block-bootstrap 95 % CI;
"loss" = AUC in (spatial CV, that country in training) minus AUC out.

| model | test France (trained on Austria) | loss | test Austria (trained on France) | loss |
|---|---|---:|---|---:|
| phys-logistic (6 features) | 0.822 (reference) | 0.004 | 0.766 (reference) | 0.005 |
| timing rule | 0.692, -0.130 [-0.166, -0.097] | 0.000 | 0.655, -0.110 [-0.139, -0.087] | -0.002 |
| phys-rule | 0.770, -0.052 [-0.065, -0.038] | 0.002 | 0.738, -0.028 [-0.041, -0.017] | 0.012 |
| phys-gbm (main line) | 0.752, -0.070 [-0.103, -0.034] | 0.101 | 0.783, +0.017 [+0.005, +0.031] | 0.066 |
| emb-logistic (Presto) | 0.632, -0.190 [-0.208, -0.149] | 0.196 | 0.727, -0.039 [-0.062, -0.014] | 0.096 |
| raw-gbm | 0.772, -0.050 [-0.074, -0.016] | 0.071 | 0.751, -0.015 [-0.037, +0.010] | 0.082 |
| both-gbm (phys + Presto) | 0.750, -0.072 [-0.104, -0.032] | 0.107 | 0.800, +0.034 [+0.019, +0.049] | 0.062 |
| physraw-gbm (phys + raw) | 0.784, -0.037 [-0.070, +0.001] | 0.076 | 0.790, +0.025 [+0.010, +0.041] | 0.072 |

Country-year to country-year (ROC-AUC on the target; one training
year only, so smaller training sets):

| pair | 6-feature logistic | phys-rule | phys-gbm | emb-logistic | raw-gbm | both-gbm | physraw-gbm |
|---|---:|---:|---:|---:|---:|---:|---:|
| AT 2022 -> FR 2022 (same imagery source) | 0.750 | 0.768 | 0.689 | 0.626 | 0.731 | 0.703 | 0.711 |
| AT 2023 -> FR 2022 | 0.809 | 0.785 | 0.784 | 0.600 | 0.759 | 0.802 | 0.805 |
| AT 2024 -> FR 2022 | 0.829 | 0.783 | 0.755 | 0.610 | 0.730 | 0.760 | 0.763 |
| FR 2022 -> AT 2022 (same imagery source) | 0.693 | 0.727 | 0.728 | 0.719 | 0.729 | 0.753 | 0.742 |
| FR 2022 -> AT 2023 | 0.794 | 0.762 | 0.799 | 0.703 | 0.686 | 0.815 | 0.802 |
| FR 2022 -> AT 2024 | 0.784 | 0.736 | 0.804 | 0.747 | 0.789 | 0.816 | 0.809 |

Paired to the logistic: on AT 2022 -> FR 2022 every richer model is worse
with the interval excluding zero (phys-gbm -0.060 [-0.093, -0.021],
both-gbm -0.047 [-0.074, -0.015], physraw-gbm -0.039 [-0.063, -0.008],
Presto -0.123); only the two-feature rule is level (+0.019 [-0.007,
+0.047]). On FR 2022 -> AT 2022 the richer models are ahead (phys-gbm
+0.036 [+0.005, +0.066], both-gbm +0.060 [+0.020, +0.100]).

**When the country changes, does the ordering look like the Austrian year
hold-out (richer wins) or like Maryland (simpler wins)?** It depends on
the direction, but the mechanism is the same both ways. The 6-feature
logistic keeps its in-country accuracy almost exactly (loss 0.004 and
0.005), while every richer model loses 0.06-0.20. Trained on Austria and
tested on France, that loss is bigger than the richer models' in-country
lead, so the simple model wins outright, like Maryland. Trained on France
and tested on Austria, the richer models' lead shrinks from about 0.09 to
0.02-0.03 but survives. Within Austria the year hold-out cost every model
about the same 0.02-0.05, which is why it ranked the models the other way.
Presto transfers worst in every direction. Note also that the three-year
leave-one-year-out in this run (with France 2022 inside the 2022 group)
already narrows the richer models' lead: on 2022 none of them is clearly
ahead of the logistic (phys-gbm +0.002 [-0.036, +0.040], both-gbm +0.023
[-0.014, +0.059]).

**Three countries, each left out in turn** (the Netherlands 2025 added,
1,600 fields, Earth Search Collection 1; run
`<data>\foundation\runs\at_fr_nl`, 8,000 fields, none dropped).
Trained on the other two countries; ROC-AUC, paired difference to the
6-feature logistic [95 % CI], and the loss against in-country spatial CV:

| model | test Austria | loss | test France | loss | test Netherlands | loss |
|---|---|---:|---|---:|---|---:|
| phys-logistic (6 features) | 0.754 (ref) | 0.015 | 0.826 (ref) | 0.002 | 0.717 (ref) | -0.001 |
| timing rule | 0.654, -0.100 [-0.128, -0.077] | -0.001 | 0.694, -0.132 [-0.169, -0.099] | -0.002 | 0.638, -0.079 [-0.141, -0.010] | 0.000 |
| phys-rule | 0.746, -0.008 [-0.022, +0.005] | 0.000 | 0.772, -0.053 [-0.066, -0.039] | -0.001 | 0.665, -0.052 [-0.077, -0.026] | -0.002 |
| phys-gbm | 0.770, +0.016 [+0.007, +0.025] | 0.078 | 0.810, -0.016 [-0.038, +0.009] | 0.036 | 0.713, -0.004 [-0.028, +0.020] | 0.027 |
| emb-logistic (Presto) | 0.754, +0.001 [-0.019, +0.024] | 0.071 | 0.650, -0.175 [-0.197, -0.122] | 0.188 | 0.714, -0.003 [-0.044, +0.041] | 0.025 |
| raw-gbm | 0.747, -0.007 [-0.024, +0.014] | 0.093 | 0.797, -0.029 [-0.054, -0.001] | 0.050 | 0.715, -0.002 [-0.050, +0.047] | 0.022 |
| both-gbm | 0.798, +0.045 [+0.033, +0.058] | 0.063 | 0.815, -0.011 [-0.030, +0.013] | 0.047 | 0.733, +0.017 [-0.017, +0.049] | 0.023 |
| physraw-gbm | 0.789, +0.035 [+0.023, +0.047] | 0.071 | 0.823, -0.003 [-0.026, +0.023] | 0.040 | 0.727, +0.010 [-0.025, +0.044] | 0.030 |

With two countries to learn from, the richer models lose less out of
country than with one (on France: phys-gbm loses 0.036 trained on
Austria + the Netherlands, 0.101 trained on Austria alone), so their
penalty shrinks to "no clear difference" from the logistic on France and
the Netherlands, while the logistic still loses at most 0.015 everywhere.
The Netherlands is hard for every model alike (ROC-AUC 0.71-0.73, and at
the 0.5 cut-off every model calls about half of its grain silage, FPR
0.49-0.57), a calibration problem rather than a ranking one, the same
kind of failure as in Queen Anne's County. Presto again transfers worst
(0.65 on France). So across three countries: the simple model never
loses much and is never clearly beaten except on Austria, where the
physical + Presto or physical + raw GBMs gain 0.035-0.045.

**Caveat: country and imagery source are partly confounded.** France 2022
and Austria 2022 come from Planetary Computer (Sentinel-2 L2A), Austria
2023 and 2024 and the Netherlands 2025 from Earth Search (Collection 1),
so in the three-country hold-out source and country are mixed in both
directions. The pairs that hold the
source fixed (AT 2022 <-> FR 2022) show the same direction-dependent
pattern as the full country hold-out, so the ordering is not just a
change of processing; but the source-fixed pairs use one training year
(1,200 or 1,600 fields), so they are noisier and their training sets
smaller than in the full hold-out.

### Can augmentation stop the big model memorising? (three-country hold-out)

Question: the 135-feature LightGBM reaches 0.763 against the 6-feature
logistic's 0.765 when each country is held out in turn; does training it
on deliberately perturbed copies of the training fields, or regularising
it, make it lose less out of country? Code:
`src/fieldvision/foundation/augment.py`, `augment_eval.py`, tests
`tests/test_augment.py`; run `<data>\foundation\augment\three_country`.

Augmentation acts on the time series of training fields only, before the
features are computed; each training field gets 3 copies, test fields are
always scored on their real data, and a copy keeps its original's block,
so it never crosses a split. Magnitudes were fixed in advance from the
request, nothing was chosen on a held-out country:
- shift: the whole season moved by a whole number of days, uniform in
  +/-21 days;
- thin: each observation kept with probability 0.7, then one window of
  10-20 days dropped (cloud gaps, different revisit between sources);
- gain: per band, reflectance x U(0.95, 1.05) + U(-0.01, +0.01);
- all: the three together.
Regularisation (gbm|reg): 7 leaves instead of 15, at least 100 fields per
leaf, 30% instead of 50% of features per tree, L2 5 instead of 1. The
135 features are the "relative" variant, which already contains no
absolute-calendar feature (no calendar bins, no day-of-year columns), so
there was nothing further to remove. The same augmentations were applied
to the 6-feature logistic as a control. Splits, paired comparisons and
intervals as in the three-country table above (8,000 fields, none dropped).

ROC-AUC on the held-out country; in brackets the paired difference to the
unaugmented 135-feature GBM on the same fields, block-bootstrap 95 % CI;
"mean loss" = in-country spatial CV minus out of country, averaged over
the three countries:

| arm | test Austria | test France | test Netherlands | mean out | mean loss | mean diff vs 6-feature logistic |
|---|---|---|---|---:|---:|---:|
| gbm, base | 0.770 | 0.810 | 0.713 | 0.764 | 0.047 | -0.001 |
| gbm, regularised | 0.773 (+0.003 [-0.001, +0.008]) | 0.809 (-0.001 [-0.007, +0.007]) | 0.711 (-0.002 [-0.010, +0.007]) | 0.765 | 0.047 | -0.001 |
| gbm, shift | 0.775 (+0.005 [-0.000, +0.010]) | 0.809 (-0.001 [-0.011, +0.008]) | 0.715 (+0.002 [-0.005, +0.010]) | 0.766 | 0.045 | +0.001 |
| gbm, thin | 0.791 (+0.022 [+0.013, +0.030]) | 0.826 (+0.016 [+0.000, +0.031]) | 0.723 (+0.010 [+0.002, +0.017]) | 0.780 | 0.035 | +0.015 |
| gbm, gain | 0.764 (-0.006 [-0.012, -0.001]) | 0.792 (-0.018 [-0.035, -0.001]) | 0.718 (+0.005 [-0.008, +0.018]) | 0.758 | 0.053 | -0.007 |
| gbm, all three | 0.793 (+0.023 [+0.015, +0.031]) | 0.825 (+0.015 [+0.001, +0.028]) | 0.721 (+0.008 [-0.001, +0.017]) | 0.780 | 0.038 | +0.014 |
| gbm, all three + regularised | 0.793 (+0.023 [+0.017, +0.030]) | 0.818 (+0.008 [-0.002, +0.018]) | 0.720 (+0.007 [-0.002, +0.015]) | 0.777 | 0.035 | +0.012 |
| logistic (6 features), base | 0.754 | 0.826 | 0.717 | 0.765 | 0.005 | reference |
| logistic, thin | 0.755 | 0.825 | 0.720 | 0.767 | 0.005 | +0.001 |
| logistic, all three | 0.756 | 0.823 | 0.706 | 0.762 | 0.007 | -0.003 |

Against the 6-feature logistic, the thinned GBM is ahead on Austria
(+0.038 [+0.024, +0.051]) and level on France (+0.000 [-0.021, +0.022])
and the Netherlands (+0.006 [-0.019, +0.032]). At the 0.5 cut-off it
keeps the Netherlands' calibration problem (FPR 0.55, the logistic 0.49).

Reading:
- Dropping observation dates is the one augmentation that helps: it
  raises the big model out of country in all three countries, with every
  interval clear of zero (France only just), and cuts its average loss
  from 0.047 to 0.035. The plausible reason is that the features most
  sensitive to how often and when a field was seen (exact harvest bracket,
  slopes over few points) are the ones that change between countries and
  imagery sources, and thinning teaches the model not to trust them.
- Shifting the season does nothing, as expected: the features are already
  measured from each field's own peak. The per-band gain hurts slightly
  (France -0.018), and stronger regularisation alone changes nothing; it
  is the input perturbation, not a smaller model, that matters.
- The logistic barely moves under any augmentation (its six features are
  already robust), so the control behaves as it should.
- Even augmented, the big model is level with the 6-feature logistic on
  two of the three countries and still loses about seven times as much out
  of country (0.035 against 0.005). Six arms were tried and "thin" is
  picked after seeing them, so the size of its gain should be confirmed
  once on new data before it is relied on. Evaluation 2's primary stays the
  6-feature logistic; the thinned GBM is a candidate second member, to be
  tested once on Beltsville by the modelling session.

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
