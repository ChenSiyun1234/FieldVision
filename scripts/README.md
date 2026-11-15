# scripts

Command-line scripts around the model package. Run from the repository root, e.g. `python scripts/<name>.py --help`.
Most were written for one step of the hackathon and are kept because `docs/log.md` and `handoff/results/REPORT.md` cite them.

## Extraction: getting satellite series for fields

| Script | What it does |
|---|---|
| `probe_s2_catalog.py` | Check public catalog availability and save one item per region/year. |
| `prefetch_s2_catalog.py` | Fill an extraction run's catalog cache for all blocks in parallel. |
| `probe_beltsville.py` | Count Sentinel-2 L2A scenes over the Beltsville (BARC) farm per source and season. |
| `compare_s2_sources.py` | Measure calibration agreement between Sentinel-2 L2A sources on the same acquisitions. |
| `analyze_s2_calibration.py` | Summarise per-band agreement from compare_s2_sources.py output. |
| `range_download.py` | Download a large file with parallel HTTP range requests. |
| `report_s2_run.py` | Check a finished extractor run and print (or append) its READY.txt line. |
| `summarize_s2_smoke.py` | Summarize a completed extractor run and plot three field NDVI series. |
| `fetch_austria_smoke.py` | Fetch 30 declared silage-maize parcels for the extractor smoke test. |
| `landsat_gap_report.py` | Deliver a merged Landsat run next to its county and report what it adds in a window. |
| `deliver_landsat_sample.py` | Put the Landsat rows of a state sample (three parts) next to its optical series, split by stratum. |

## Colab: running extraction on Google Colab

| Script | What it does |
|---|---|
| `build_colab_bundle.py` | Build the upload folder for running one extraction on Google Colab. |
| `build_colab_county.py` | Colab bundle for a county with no field file yet: make the field file on Colab, then extract there. |
| `build_colab_run.py` | Build a Colab upload folder that continues a laptop extraction run. |
| `build_colab_state_frames.py` | Colab notebook that builds statewide corn frames and block draws from CSB (boundaries only, no imagery). |
| `merge_colab_blocks.py` | Merge block results computed on Colab back into the laptop run folder. |
| `export_colab_notebooks.py` | Copy the extraction notebooks and their READMEs into colab/extraction/ for GitHub. |
| `build_view_notebook.py` | Write colab/view_field_images.ipynb: picture strips for any extracted county, in Colab. |
| `colab_to_county.py` | Turn one Colab county extraction into the toolbox's county folder. |

## Fields and samples: choosing what to score

| Script | What it does |
|---|---|
| `csb_subset.py` | Cut one state out of the national USDA NASS Crop Sequence Boundaries archive. |
| `county_corn_check.py` | Sanity check of a county corn-field file before any modelling. |
| `county_snapshot.py` | Copy a county's fields and Sentinel-2 time series into the toolbox layout <root>/<fips>/<year>/. |
| `sample_snapshot.py` | Merge the finished parts of a statewide sample into one toolbox-style folder. |
| `state_sample.py` | Statewide sample of whole read blocks from a corn frame, split into equal parts for parallel extraction. |
| `scope_states.py` | Scoping for county checks outside Maryland: states and two counties per state, by rule. |
| `pick_test_county.py` | Rank candidate test counties by the rule in docs/us-county-fields.md. |
| `unscored_corn.py` | Corn the county step does not score: CSB polygons whose CDL class that year is a small grain followed by corn. |
| `pod_batch_counties.py` | Run the top silage counties through the pipeline in one batch, on the pod (not the Mac). |

## Context checks and priors (boundaries and land use, no model)

| Script | What it does |
|---|---|
| `rotation_check.py` | Crop rotation of called fields, from the CSB class sequence (boundaries only, no imagery, no model). |
| `forage_landscape.py` | Forage landscape around each scored corn field, from CSB polygons (boundaries only, no imagery, no model). |
| `landscape_groups.py` | Distribution of the forage-landscape share by the rotation-check groups of called fields. |
| `landuse_prior.py` | County silage share predicted from land use (CDL forage and pasture shares): a prior, not a field label. |
| `landuse_prior_figure.py` | Figure for docs/landuse-prior.md: county forage share (2019-2023 mean) against the Census 2022 silage share. |
| `fetch_cdl_county_stats.py` | Fetch CropScape county statistics of the Cropland Data Layer (pixel acres by class) for many counties and years. |

## Sequence and stream model

| Script | What it does |
|---|---|
| `stream_subset.py` | Select whole read blocks, or copy field geometries with labels removed. |
| `build_stream_bundle.py` | Build the eight-run pixel-chip upload bundle from prepared subset fields. |
| `build_stream_train_bundle.py` | Build the Colab job that trains the sequence and stream models on a GPU runtime. |
| `stream_unpack.py` | Verify and safely unpack multipart stream_px chip results with bounded I/O memory. |

## Pictures and demos

| Script | What it does |
|---|---|
| `make_field_strips.py` | Picture strips for a few fields: true-colour chips across the season plus the NDVI curve. |
| `pick_strip_fields.py` | Helper for make_field_strips.py: picks which fields to draw. |
| `make_deck_pictures.py` | Slide-size pictures for the deck: county call maps and a before/after-the-cut sheet. |
| `toolbox_demo.py` | Run the toolbox on one county-season and print what each method returns. |
