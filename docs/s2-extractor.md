# Sentinel-2 field time series

Run from the repository root in PowerShell. No login or API key is needed.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& C:\t\fv\venv\Scripts\python.exe -m fieldvision.s2.extract --fields fields.geojson --year 2025 --start 04-01 --end 11-30 --out C:\t\fv\data\timeseries\nl_2025 --workers 16
& C:\t\fv\venv\Scripts\python.exe -m pytest -q
```

Set `PYTHONPATH` once per shell. The package also supports an ordinary editable
installation in a team's own environment. Runtime dependencies are in
`pyproject.toml`; the supplied environment already has them.

## Input and output

Input is a WGS84 GeoJSON FeatureCollection. Each feature needs a unique string
`field_id`, an integer `year` matching the CLI, and polygon geometry. The other
contract properties (`label`, `country`, `source`, `crop_code`, `area_ha`, optional
`block_id`) are retained in input metadata but not repeated in the time series.
Join labels by `field_id`; extraction never uses them. Spatial blocks are computed
independently of the optional upstream `block_id`.

The output directory contains:

- `timeseries.parquet`: final field/date table with the requested columns; dates
  are ISO `YYYY-MM-DD` strings, bands are dimensionless reflectance in [0, 1].
- `blocks/*.parquet`: independently resumable block tables.
- `blocks/*.json`: scene completion checkpoints, errors and reflectance diagnostics.
- `catalog/*.json`: cached STAC items, including original asset calibration metadata.
- `manifest.json`: exact science options, input/code fingerprint, timing, counts,
  per-block status, scene errors, and calibration checks.
- `fields_coverage.csv`: every input ID, status, maximum observed interior pixel
  count, usable date count, first/last date, and maximum gap in days.

One row per field/calendar date survives, choosing the greatest `n_clear`, then
clear fraction, then Collection 1 on ties. Rows with no clear pixels have NaN in
every band. `n_pixels` is the number of interior pixel centers in the polygon,
including cloudy or missing pixels; `n_clear` requires SCL 4/5/6/7 and valid
reflectance in all ten bands. `clear_fraction = n_clear / n_pixels`, or zero for a
zero-pixel polygon. Pixel counts are on a 10 m grid; resampling does not create
independent 10 m measurements for the native 20 m bands.

A usable date defaults to at least one clear pixel and clear fraction >= 0.2.
Change `--min-clear-pixels` and `--min-clear-fraction` for stricter modeling.
Coverage gaps include start-to-first and last-to-end gaps within the inclusive
requested window. A field with no usable dates has a gap spanning the window.

## Catalog and calibration

The catalog is <https://earth-search.aws.element84.com/v1>. Collection 1 is the
default. Direct probes on 2026-11-14 found the following item counts for April 1
through November 30, with no cloud filter and a 0.1 degree bbox per region:

| Region | 2022 C1 / older | 2024 C1 / older | 2025 C1 / older |
|---|---:|---:|---:|
| Austria (15.7,48.0,15.8,48.1) | 0 / 97 | 95 / 95 | 144 / 144 |
| Netherlands (5.5,52.0,5.6,52.1) | 1 / 94 | 104 / 104 | 142 / 140 |
| Western France (-1.0,47.0,-0.9,47.1) | 1 / 100 | 103 / 103 | 146 / 146 |
| US Midwest (-93.7,41.9,-93.6,42.0) | 1 / 98 | 96 / 96 | 121 / 120 |

Whole-year queries return 10–14 C1 items for 2022 in these locations, which can
misleadingly suggest coverage. The growing season is effectively missing.
For 2022 and earlier, search both collections and add legacy acquisitions missing
from C1 by MGRS tile/date/platform. For newer years, fallback occurs only when the
block has no C1 scenes. `--no-fallback` disables it; `--collection sentinel-2-l2a`
explicitly selects the older collection. All rows name their source collection.

Every asset must declare `raster:bands` scale and offset. For the older collection,
`earthsearch:boa_offset_applied=true` means use that scale with zero additional
offset. False means apply the declared offset; a missing flag with a nonzero offset
is rejected. There are known upstream inconsistencies even in that flag; see
<https://github.com/Element84/earth-search/issues/71>. Nothing guesses a correction.
For blocks with at least 20 SCL vegetation pixels, require median red in (0, 0.35),
NIR > red + 0.02, and NIR < 1 before writing. A failed check records the scene error.
After that check, out-of-range reflectance is marked missing rather than clipped.
This conservative test may reject unusual vegetation/illumination; inspect errors.

## Spatial and operational behavior

Fields are grouped in their centroid's local UTM zone into compact blocks up to
6 km wide. Each block is reprojected to each scene's own CRS and snapped to its
10 m grid. Whole parcels stay together even across a zone or MGRS boundary.
A single parcel wider than 6 km gets its own block; a window exceeding four million
pixels fails explicitly rather than exhausting memory. This is intended for farm
parcels, not regional masks or globally distributed multipart geometries.

The polygon is inset 5 m only if at least four actual pixel centers remain;
otherwise the original polygon is used. Pixel centers must fall inside polygons;
subpixel parcels can have zero centers and are reported as `no_pixels`.
Shared pixels in overlapping polygons are aggregated for each field without
additional imagery reads.

The worker pool reads `(scene, band)` windows through GDAL COG range requests.
Three scenes are in flight by default (`--scene-batch`); blocks run sequentially.
Both spectral bands and SCL use nearest-neighbor resampling, preserving original
values and nodata. SCL classes 4, 5, 6, 7 remain; bare post-harvest soil is retained.
Actual STAC footprints filter candidate scenes; asset nodata handles internal
missing footprints. Each band nodata is excluded before scale/offset conversion.

HTTP and raster reads retry with backoff. A failed scene does not abort a block.
Rerunning the same command skips complete blocks and successful scenes in partial
blocks. Inputs, scientific options and source code must match the saved
fingerprint; use a new output directory after changes. Do not run two processes
against the same output directory. The final table is rebuilt from block files.
Exit code 0 means processing complete, 2 means partial (scene/block/input errors),
and 1 means an invalid invocation or fatal setup error. Complete does not imply
that every field has a usable date; always inspect coverage.

## Indices

`fieldvision.s2.indices` supplies pure `ndvi`, `ndmi` (B8A/B11), `evi`, `nbr` and
`psri` functions accepting a DataFrame or mapping of array-like band columns.
`add_indices(table)` returns a copy with all five. PSRI is the Sentinel-2
approximation `(B04 - B02) / B06`; zero denominators return NaN. These are indices
of per-field band medians, not per-pixel index medians.

## Reproduce the Austrian smoke test

```powershell
& C:\t\fv\venv\Scripts\python.exe scripts/fetch_austria_smoke.py --out C:\t\fv\data\timeseries\smoke_input
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& C:\t\fv\venv\Scripts\python.exe -m fieldvision.s2.extract --fields C:\t\fv\data\timeseries\smoke_input\fields.geojson --year 2024 --start 04-01 --end 11-30 --out C:\t\fv\data\timeseries\austria_2024_smoke --workers 16
```

The fetcher uses the public INVEKOS API with `snar_code='109'` and bbox
15.5,47.9,16.0,48.2. No imagery or result tables belong in the repository.
