"""Presto input layout (from presto/dataops/pipelines/s1_s2_era5_srtm.py, MIT License,
Copyright (c) 2024 Presto Authors). Kept free of torch so the adapter can be
imported and tested without it."""
from collections import OrderedDict

# Channel order of the model input after Presto drops B1, B9 and B10
# (NORMED_BANDS in presto/dataops/pipelines/s1_s2_era5_srtm.py).
PRESTO_BANDS = ["VV", "VH", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
                "temperature_2m", "total_precipitation", "elevation", "slope", "NDVI"]
BAND_GROUPS = OrderedDict([
    ("S1", [0, 1]),
    ("S2_RGB", [2, 3, 4]),
    ("S2_Red_Edge", [5, 6, 7]),
    ("S2_NIR_10m", [8]),
    ("S2_NIR_20m", [9]),
    ("S2_SWIR", [10, 11]),
    ("ERA5", [12, 13]),
    ("SRTM", [14, 15]),
    ("NDVI", [16]),
])
DW_MISSING = 9          # Dynamic World class index meaning "no data"
EMBED_DIM = 128
