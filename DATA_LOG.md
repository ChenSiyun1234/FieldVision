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
