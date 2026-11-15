# Sentinel-1 radar pilot, Kent County 2022 (`s1_24029_2022`)

Per field and per radar acquisition, April 1 to November 30 2022: mean and median VV
and VH backscatter (gamma0, linear power) over the field's interior pixels, the dB of
the means, VH/VV in dB, and the orbit (direction, relative orbit, absolute orbit,
platform). Same fields file (`us_24029_2022`, 1,363 fields), same 5 m inward buffer,
same block layout (43 blocks), checkpoints, resume and merge as the optical runs.
Kent is an untouched test county: extraction only, no model.

Source: Planetary Computer STAC collection `sentinel-1-rtc` (radiometrically terrain
corrected GRD, 10 m, UTM). **No account needed**: the notebook gets an anonymous SAS
token from `https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc`
and appends it to each file URL (tested on the laptop at 19:00 and in the check below).
The files end in `.tiff`; the extractor allows that extension in GDAL.

Code: new package `fieldvision.s1` (commit in branch `work/s2-timeseries-eafb89`),
frozen copy `C:\t\fv\frozen_src_s1_3fc72639`; fingerprint `3fc72639…` covers
`s1/*.py` plus the three s2 files it reuses (`core.py`, `geometry.py`, `catalog.py`).
The optical extractor and its frozen copy `9cd0447b` are untouched.

## Files (upload all into `/content`)

| File | Size | What |
|---|---:|---|
| `bundle.json` | 1 KB | names, fingerprint, arguments (`--year 2022`), pinned versions |
| `code_3fc72639.zip` | 22 KB | frozen `fieldvision` package (s1 + s2) |
| `fields_s1_24029_2022.geojson` | 2.8 MB | byte-identical to `C:\t\fv\data\labels\us_24029_2022\fields.geojson` |
| `check_blocks.zip` | 16 KB | two Kent blocks the laptop extracted for July 2022 (`32618_US-E380N4330_0`, 2 fields; `32618_US-E400N4320_0`, 5 fields; 4 passes each) with reference rows |
| `extract_s1_24029_2022.ipynb` | 16 KB | open this in Colab |

## Same-result check

Step 4 re-extracts the two check blocks for July 2022 and compares every row, keyed by
field and absolute orbit: scene, date and time, platform, orbit, pixel counts and all
VV/VH values exactly. Any difference stops the notebook. Run on the laptop with the
same script at 19:10: both blocks IDENTICAL (8 and 20 rows), 51 s. To rerun it here:

```bash
PYTHONPATH=C:/t/fv/frozen_src_s1_3fc72639 /c/t/fv/venv/Scripts/python.exe -W ignore C:/t/fv/s1_checktest/check.py C:/t/fv/s1_checktest C:/t/fv/colab/s1_24029_2022/check_blocks.zip 8 C:/t/fv/colab/s1_24029_2022/fields_s1_24029_2022.geojson 32618_US-E380N4330_0 32618_US-E400N4320_0
```

## What Kent 2022 contains (catalog, 19:05)

52 RTC scenes = 37 passes, all Sentinel-1A ascending (Sentinel-1B failed in Dec 2021),
from two relative orbits: 106 (20 scenes) and 4 (32). That is one pass every 6.5 days
on average, alternating orbits; same-orbit repeats are 12 days apart. Per block 20-52
scenes (median 37), 1,385 scene-block reads in total, two polarisations each.
Note for modelling: the two relative orbits look at the fields from different
incidence angles, so compare backscatter within an orbit (or normalise per orbit).

## Expected time and volume

Laptop measurement: ~0.5-1 CPU-s per scene-block read (two float32 windows on 512 px
tiles). Kent: ~1,400 reads, a few GB streamed from Azure.
* 8 vCPU: about 10-20 min (plus ~2-3 min for catalog queries and the check).
* 2 vCPU: about 25-45 min.

Maryland 2025 sample the same way (3 parts, 150 blocks, 4,028 fields): 163 scenes = 64
passes (Sentinel-1A, plus 4 Sentinel-1C scenes, 6 descending), 4,215 scene-block reads,
about 3x Kent: ~30-60 min on 8 vCPU in one runtime, or three ~10-20 min runs (one per
part, run folders `s1_24_2025_s1..s3` made the same way). No account, no cost beyond
Colab time; the result zips are small (block parquet + json only).

## Run order and back on the laptop

Cells top to bottom (settings, Drive, pinned installs, fingerprint + check, prepare,
extraction `done/43`, result parts). Then, as for the optical runs, put the parts and
`s1_24029_2022_SHA256SUMS.txt` in `C:\t\fv\colab_results\s1_24029_2022\` and:

```bash
PYTHONPATH=src /c/t/fv/venv/Scripts/python.exe scripts/merge_colab_blocks.py --parts-dir C:/t/fv/colab_results/s1_24029_2022 --run-dir C:/t/fv/data/timeseries/s1_24029_2022
```
```bash
PYTHONPATH=C:/t/fv/frozen_src_s1_3fc72639 /c/t/fv/venv/Scripts/python.exe -m fieldvision.s1.extract --fields C:/t/fv/data/labels/us_24029_2022/fields.geojson --year 2022 --out C:/t/fv/data/timeseries/s1_24029_2022 --workers 16
```

Output: `timeseries.parquet` (columns `field_id, date, datetime, scene_id, collection,
platform, orbit_state, relative_orbit, absolute_orbit, n_pixels, n_valid,
valid_fraction, vv_mean, vh_mean, vv_median, vh_median, vv_db, vh_db, vh_vv_db`) and
`fields_coverage.csv` (passes per field, largest gap). `report_s2_run.py` is for the
optical runs; a READY line for radar would need its own wording (not written yet).
