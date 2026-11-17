# Data log

Queries and processing choices for the BARC Sentinel-2 field time series. Each run of `src/build_timeseries.py` appends a record below.

## 2026-11-14 13:18 PST — BARC Sentinel-2 field time series

- mode: test
- geopackage: `/Users/hetvi/Documents/Hackathon/FieldVision/notebooks/dataset/BARC_Corn.2021-2023.gpkg`
- catalog: https://planetarycomputer.microsoft.com/api/stac/v1
- collection: sentinel-2-l2a
- date range: 05-01 to 12-15 of each processed year
- cloud threshold: eo:cloud_cover < 60
- bands: B04, B08, B11, B12, SCL
- resampling: reflectance bilinear, SCL nearest
- CRS: EPSG:32618
- resolution: 10 m
- buffer: 15 m inward
- SCL classes kept: [4, 5] (4 vegetation, 5 bare soil)
- offset correction: on dates on or after 2022-01-25, subtract 1000 DN then divide by 10000; DN 0 is nodata and is masked before the offset; clip at 0
- clear_frac cutoff: 0.8 (valid pixels / pixels inside the polygon; valid = inside and SCL in the classes above)
- scene counts: 2021: not run, 2022: 69, 2023: not run
- rows written: 140
- fields dropped by the inward buffer: none
- field-years under 8 valid dates from Aug 1 to Nov 30: none
- skip notes: none
- output: `data/processed/field_timeseries.csv`, `data/processed/ndvi_curves.png`
- error: none

## 2026-11-14 14:23 PST — NASS county selection
- survey file: /Users/hetvi/Documents/Hackathon/FieldVision/notebooks/dataset/corn_grain_silage_acres_2021_2023.csv
- census file: /Users/hetvi/Documents/Hackathon/FieldVision/notebooks/dataset/corn_grain_silage_acres_2012_2025.csv
- filters: COUNTY, domain TOTAL, exact grain/silage harvested descriptions, source SURVEY for 2023 and CENSUS for 2022
- withheld values: (D) and (Z) set to missing; commas stripped
- 2023 survey counties: 1455
- 2023 both classes: 199
- 2022 census counties: 2738
- candidate rule: silage share 0.2-0.6 and total corn >= 10000; 2023 survey when both classes exist, else 2022 Census
- candidates: 139
- outputs: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/nass_county_2023.csv, /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/nass_county_2022.csv, /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/candidate_counties.csv

## 2026-11-14 15:13 PST — Sentinel-2 barc test
- catalog: https://planetarycomputer.microsoft.com/api/stac/v1
- collection: sentinel-2-l2a
- date range: 05-01 to 12-15
- years: [2023]
- cloud threshold: eo:cloud_cover < 60
- bands: B04, B08, B11, B12, SCL
- crs: EPSG:32618
- resolution_m: 10
- buffer_m: 15
- scl classes: [4, 5]
- clear_frac cutoff: 0.8
- tiles: 1
- scene counts: {2023: 0}
- rows written: 0
- output: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/barc/s2_timeseries_test.csv
- skip notes: tile all 2023 failed and will be retried next run: HTTPSConnectionPool(host='planetarycomputer.microsoft.com', port=443): Max retries exceeded with url: /api/stac/v1 (Caused by NameResolutionError("HTTPSConnection(host='planetarycomputer.microsoft.com', port=443): Failed to resolve 'planetarycomputer.microsoft.com' ([Errno 8] nodename nor servname provided, or not known)"))

## 2026-11-14 15:14 PST — Sentinel-2 barc
- catalog: https://planetarycomputer.microsoft.com/api/stac/v1
- collection: sentinel-2-l2a
- date range: 05-01 to 12-15
- years: [2023]
- cloud threshold: eo:cloud_cover < 60
- bands: B04, B08, B11, B12, SCL
- crs: EPSG:32618
- resolution_m: 10
- buffer_m: 15
- scl classes: [4, 5]
- clear_frac cutoff: 0.8
- tiles: 1
- scene counts: {2023: 0}
- rows written: 0
- output: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/barc/s2_timeseries.csv
- skip notes: tile all 2023 failed and will be retried next run: HTTPSConnectionPool(host='planetarycomputer.microsoft.com', port=443): Max retries exceeded with url: /api/stac/v1 (Caused by NameResolutionError("HTTPSConnection(host='planetarycomputer.microsoft.com', port=443): Failed to resolve 'planetarycomputer.microsoft.com' ([Errno 8] nodename nor servname provided, or not known)"))

Removed B05/B8A (NDRE) from Sentinel-2 bands to reduce load; NDVI and NDTI unchanged.

## 2026-11-14 15:14 PST — Sentinel-1 barc
- catalog: https://planetarycomputer.microsoft.com/api/stac/v1
- collection: sentinel-1-rtc
- date range: 05-01 to 12-15
- years: [2023]
- bands: vv, vh
- crs: EPSG:32618
- resolution_m: 10
- buffer_m: 15
- units: median vv and vh in linear power, then 10*log10; vh/vv ratio in dB
- tiles: 1
- scene counts: {2023: 0}
- rows written: 0
- output: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/barc/s1_timeseries.csv
- skip notes: tile all 2023 failed and will be retried next run: HTTPSConnectionPool(host='planetarycomputer.microsoft.com', port=443): Max retries exceeded with url: /api/stac/v1 (Caused by NameResolutionError("HTTPSConnection(host='planetarycomputer.microsoft.com', port=443): Failed to resolve 'planetarycomputer.microsoft.com' ([Errno 8] nodename nor servname provided, or not known)"))

## 2026-11-14 16:17 PST — Sentinel-2 barc test
- catalog: https://planetarycomputer.microsoft.com/api/stac/v1
- collection: sentinel-2-l2a
- date range: 05-01 to 12-15
- years: [2023]
- cloud threshold: eo:cloud_cover < 60
- bands: B04, B08, B11, B12, SCL
- crs: EPSG:32618
- resolution_m: 10
- buffer_m: 15
- scl classes: [4, 5]
- clear_frac cutoff: 0.8
- tiles: 1
- scene counts: {2023: 99}
- rows written: 115
- output: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/barc/s2_timeseries_test.csv
- skip notes: 2023: 99 scenes, 58 dates after dedup, 44 dates kept

## 2026-11-14 16:32 PST — Sentinel-2 barc
- catalog: https://planetarycomputer.microsoft.com/api/stac/v1
- collection: sentinel-2-l2a
- date range: 05-01 to 12-15
- years: [2023]
- cloud threshold: eo:cloud_cover < 60
- bands: B04, B08, B11, B12, SCL
- crs: EPSG:32618
- resolution_m: 10
- buffer_m: 15
- scl classes: [4, 5]
- clear_frac cutoff: 0.8
- tiles: 1
- scene counts: {2023: 99}
- rows written: 512
- output: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/barc/s2_timeseries.csv
- skip notes: 2023: 99 scenes, 58 dates after dedup, 48 dates kept

## 2026-11-14 16:36 PST — Sentinel-1 barc test
- catalog: https://planetarycomputer.microsoft.com/api/stac/v1
- collection: sentinel-1-rtc
- date range: 05-01 to 12-15
- years: [2023]
- bands: vv, vh
- crs: EPSG:32618
- resolution_m: 10
- buffer_m: 15
- units: median vv and vh in linear power, then 10*log10; vh/vv ratio in dB
- tiles: 1
- scene counts: {2023: 30}
- rows written: 72
- output: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/barc/s1_timeseries_test.csv
- skip notes: none

## 2026-11-14 16:39 PST — Sentinel-1 barc
- catalog: https://planetarycomputer.microsoft.com/api/stac/v1
- collection: sentinel-1-rtc
- date range: 05-01 to 12-15
- years: [2023]
- bands: vv, vh
- crs: EPSG:32618
- resolution_m: 10
- buffer_m: 15
- units: median vv and vh in linear power, then 10*log10; vh/vv ratio in dB
- tiles: 1
- scene counts: {2023: 30}
- rows written: 363
- output: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/barc/s1_timeseries.csv
- skip notes: none

Saved median offset-corrected B04, B08, B11, and B12 reflectance, and per-pixel NDMI (B08, B11), in s2_timeseries.csv. Columns: field_id, year, date, doy, ndvi, ndti, ndmi, b04, b08, b11, b12, clear_frac, n_pixels, label. Deleted the BARC 2023 Sentinel-2 file before the rerun so every year uses this schema.

## 2026-11-14 16:57 PST — Sentinel-2 barc
- catalog: https://planetarycomputer.microsoft.com/api/stac/v1
- collection: sentinel-2-l2a
- date range: 05-01 to 12-15
- years: [2023]
- cloud threshold: eo:cloud_cover < 60
- bands: B04, B08, B11, B12, SCL
- indices: per-pixel then median: ndvi (B08, B04), ndti (B11, B12), ndmi (B08, B11); b04, b08, b11, b12 are median reflectance on the same mask
- crs: EPSG:32618
- resolution_m: 10
- buffer_m: 15
- scl classes: [4, 5]
- clear_frac cutoff: 0.8
- tiles: 1
- scene counts: {2023: 99}
- rows written: 512
- output: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/barc/s2_timeseries.csv
- skip notes: 2023: 99 scenes, 58 dates after dedup, 48 dates kept

## 2026-11-14 17:21 PST — Collect transfer counties
- year: 2023
- csb: CSB1623.gdb (2016-2023 rev23)
- cdl: /Users/hetvi/Documents/Hackathon/FieldVision/data/raw/cdl/2023_30m_cdls.tif
- counties: PENNSYLVANIA 42071
- PENNSYLVANIA 42071: grain 38200, silage 30600, share 44.5%; qualifying 3698, kept 3698, kept acres 65097 vs NASS 68800; extract /Users/hetvi/Documents/Hackathon/FieldVision/data/raw/csb/csb_PA_2023.gpkg

## 2026-11-14 17:31 PST — Sentinel-2 pennsylvania_42071 test
- catalog: https://planetarycomputer.microsoft.com/api/stac/v1
- collection: sentinel-2-l2a
- date range: 05-01 to 12-15
- years: [2023]
- cloud threshold: eo:cloud_cover < 60
- bands: B04, B08, B11, B12, SCL
- indices: per-pixel then median: ndvi (B08, B04), ndti (B11, B12), ndmi (B08, B11); b04, b08, b11, b12 are median reflectance on the same mask
- crs: EPSG:32618
- resolution_m: 10
- buffer_m: 15
- scl classes: [4, 5]
- clear_frac cutoff: 0.8
- tiles: 1
- scene counts: {2023: 42}
- rows written: 49
- output: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/pennsylvania_42071/s2_timeseries_test.csv
- skip notes: 2023: 42 scenes, 23 dates after dedup, 14 dates kept

## 2026-11-14 20:07 PST — Sentinel-2 barc
- catalog: https://planetarycomputer.microsoft.com/api/stac/v1
- collection: sentinel-2-l2a
- date range: 05-01 to 12-15
- years: [2021, 2022, 2023]
- cloud threshold: eo:cloud_cover < 60
- bands: B04, B08, B11, B12, SCL
- indices: per-pixel then median: ndvi (B08, B04), ndti (B11, B12), ndmi (B08, B11); b04, b08, b11, b12 are median reflectance on the same mask
- crs: EPSG:32618
- resolution_m: 10
- buffer_m: 15
- scl classes: [4, 5]
- clear_frac cutoff: 0.8
- tiles: 1
- scene counts: {2021: 52, 2022: 69, 2023: 0}
- rows written: 1303
- output: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/barc/s2_timeseries.csv
- skip notes: 2021: 52 scenes, 51 dates after dedup, 50 dates kept; 2022: 69 scenes, 49 dates after dedup, 42 dates kept

## 2026-11-14 20:18 PST — Sentinel-1 barc
- catalog: https://planetarycomputer.microsoft.com/api/stac/v1
- collection: sentinel-1-rtc
- date range: 05-01 to 12-15
- years: [2021, 2022, 2023]
- bands: vv, vh
- crs: EPSG:32618
- resolution_m: 10
- buffer_m: 15
- units: median vv and vh in linear power, then 10*log10; vh/vv ratio in dB
- tiles: 1
- scene counts: {2021: 35, 2022: 35, 2023: 0}
- rows written: 540
- output: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/barc/s1_timeseries.csv
- skip notes: none

The 20:18 Sentinel-1 run above was incomplete and its 2021 and 2022 rows were discarded. Both years loaded at the same time and 29 scene reads were dropped after three timeouts each, leaving 3 full-coverage passes in 2021 and 7 in 2022. The 20:29 run below loaded one year at a time with no dropped reads: 16 full-coverage passes in each of 2021 and 2022. Added BARC 2021 and 2022 (YEARS = [2021, 2022, 2023]); 2023 rows are unchanged.

## 2026-11-14 20:29 PST — Sentinel-1 barc
- catalog: https://planetarycomputer.microsoft.com/api/stac/v1
- collection: sentinel-1-rtc
- date range: 05-01 to 12-15
- years: [2021, 2022, 2023]
- bands: vv, vh
- crs: EPSG:32618
- resolution_m: 10
- buffer_m: 15
- units: median vv and vh in linear power, then 10*log10; vh/vv ratio in dB
- tiles: 1
- scene counts: {2021: 35, 2022: 35, 2023: 0}
- rows written: 837
- output: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/barc/s1_timeseries.csv
- skip notes: none

## 2026-11-14 20:38 PST — Silage model
- training: BARC only, logistic regression on ndmi_pre, ndmi_drop, pre_ndvi, ndvi_drop, n_decline_obs, days_below08_to_harvest, ndti_delta
- barc validation: leave one year out, and leave one field out
- independent estimate: BARC model applied with no county NASS input
- calibrated map: one prior shift per county so the acre-weighted probability matches NASS
- features: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/model/features.csv
- county summary: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/model/county_summary.csv

## 2026-11-14 20:38 PST — Silage model
- training: BARC only, logistic regression on ndmi_pre, ndmi_drop, pre_ndvi, ndvi_drop, n_decline_obs, days_below08_to_harvest, ndti_delta
- barc validation: leave one year out, and leave one field out
- independent estimate: BARC model applied with no county NASS input
- calibrated map: prior shift plus a rank cutoff so the flagged acres match NASS
- features: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/model/features.csv
- county summary: /Users/hetvi/Documents/Hackathon/FieldVision/data/processed/model/county_summary.csv
