
# Data

This directory documents the datasets used in FieldVision for corn silage-versus-grain classification, exploratory data analysis, satellite processing, and agricultural intelligence.

Large raw datasets, satellite imagery, and generated model inputs are kept outside Git. Only small reference files, analysis notebooks, and selected processed outputs are version-controlled.

## Dataset Inventory

| Dataset | Location | How it was made | Notes |
|---|---|---|---|
| USDA BARC Corn Labels (2021–2023) | `notebooks/dataset/BARC_Corn.2021-2023.gpkg` | Obtained from the USDA Beltsville Agricultural Research Center challenge data. Explored using `BARC Fields Explorer.ipynb`. | Contains field polygons and 44 labeled field-year observations: 32 grain and 12 silage. |
| USDA Cropland Data Layer (CDL) | `notebooks/dataset/CDL_2023/` and `data/raw/cdl/` | Downloaded the national 2023 CDL raster and performed nationwide and Maryland-level EDA using `CDL Basic EDA.ipynb`. | Identifies corn cultivation (class 1), but does not distinguish silage from grain. Large GeoTIFF files are not committed. |
| USDA Crop Sequence Boundaries (CSB) | `data/raw/csb/` | Downloaded USDA national CSB data and extracted cornfield polygons by state and county. See `src/extract_csb.py` and `src/collect_fields.py`. | Provides field boundaries, crop history, and spatial identifiers for field-level analysis. |
| USDA NASS Quick Stats | `notebooks/dataset/corn_grain_silage_acres_2012_2025.csv` and `corn_grain_silage_acres_2021_2023.csv` | Retrieved and structured corn grain and silage acreage statistics using `notebooks/dataset/download_corn_acres.py`. | Used for county selection, historical acreage analysis, and comparison against official agricultural statistics. |
| Sentinel-2 Level-2A | `data/processed/` and external time-series directories | Retrieved satellite imagery through public STAC catalogs, including Microsoft Planetary Computer and Earth Search. Extracted field-level observations using the project's satellite processing pipelines. | Provides spectral reflectance, NDVI, NDMI, NDTI, and seasonal crop observations. Raw imagery is not stored in Git. |
| Landsat 8/9 Level-2 | External extraction directories | Accessed public Landsat imagery and processed additional field-level observations using `src/fieldvision/ls/`. | Supplements Sentinel-2 observations and helps reduce temporal gaps caused by clouds. |
| Sentinel-1 Radar | `data/processed/` | Extracted field-level radar observations using `src/s1_timeseries.py` and `src/fieldvision/s1/`. | Provides VV/VH radar measurements for supplementary crop and harvest analysis. |
| gridMET Weather | `data/processed/` | Retrieved gridded weather data and processed daily observations using `src/weather.py`. | Includes precipitation, maximum and minimum temperature, and growing degree days. |
| US Dairy Facility Locations | `notebooks/dataset/dairy_farms.kml` and `usa_dairy_farms.csv` | Loaded all KML layers, extracted point geometries, converted coordinates to EPSG:4326, and exported a structured CSV using `dairy_farms.ipynb`. | Provides geographic context for dairy-related agricultural analysis. Facility records should not be treated as a complete dairy-farm census. |
| European Corn Reference Labels | External data directory: `data/labels/` | Collected from public agricultural parcel registries in Austria, France, and the Netherlands using `src/fieldvision/labels/fetch.py`. | Provides the 8,000 labeled European fields used to train the primary classification model. |
| Field-Level Satellite Time Series | `data/processed/` or `data/timeseries/` | Joined field polygons with satellite observations, applied cloud filtering, and calculated per-field spectral measurements. | Stores field IDs, dates, satellite bands, indices, and observation-quality indicators. |
| Engineered Model Features | Generated training and prediction outputs outside Git | Computed seasonal and harvest-related features using `src/fieldvision/features/` and `src/features.py`. | Includes crop moisture, greenness, senescence, harvest timing, and residue indicators. The primary model uses six features. |
| USDA State Reference Targets | `handoff/results/nass_2025_state_targets.csv` | Prepared from USDA NASS published state-level acreage statistics. | Used to compare state-level silage and grain acreage estimates against official figures. |
| Agricultural Market History | `src/fieldvision_agents/market/data/corn_market_history.csv` | Retrieved and organized national corn acreage, production, stocks, and price data from USDA NASS Quick Stats. | Supports agricultural market analysis and scenario-based insights. |

## Exploratory Data Analysis

### BARC Dataset Analysis
Notebook: `notebooks/dataset/BARC Fields Explorer.ipynb`

- Explored field geometries and dataset attributes.
- Examined silage and grain labels across 2021–2023.
- Analyzed label distributions and acreage by year.
- Visualized labeled field polygons.
- Identified class imbalance and considerations for model validation.

### USDA CDL Analysis
Notebook: `notebooks/dataset/CDL Basic EDA.ipynb`

- Explored nationwide crop and land-cover distributions.
- Analyzed Maryland's crop composition and corn coverage.
- Clipped CDL raster data using Maryland state boundaries.
- Generated corn distribution maps and crop-class visualizations.
- Compared crop distributions across available years.

### Dairy Facility Data
Notebook: `notebooks/dataset/dairy_farms.ipynb`

- Loaded and combined KML layers.
- Extracted dairy facility point locations.
- Standardized geographic coordinates.
- Exported structured location records to CSV.
- Prepared geographic data for potential proximity-based analysis.

## Data Processing Pipeline

The satellite data pipeline follows these stages:

1. Collect agricultural field boundaries and reference labels.
2. Identify cornfields using USDA CDL and CSB.
3. Retrieve satellite observations for the relevant fields and seasons.
4. Apply cloud masking, reflectance scaling, and spatial preprocessing.
5. Aggregate satellite measurements at field level.
6. Compute NDVI, NDMI, NDTI, and seasonal harvest features.
7. Combine features into structured field-year datasets.
8. Train or evaluate classification models and compare results with reference data.

