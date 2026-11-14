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
