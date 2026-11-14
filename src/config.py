"""Shared settings. Edit TEST_COUNTIES, CDL_PATHS, and CSB_PATH before collect."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# BARC labeled years. Add 2021 or 2022 here later; finished years are not redone.
YEARS = [2023]
# Transfer counties are corn in this CDL year only. Other years are a different crop.
TRANSFER_YEAR = 2023

# "all" keeps every qualifying field. An integer draws a sample with RANDOM_SEED.
N_FIELDS = "all"
# Per-county override. Counties not listed use N_FIELDS.
N_FIELDS_BY_COUNTY = {"42071": "all", "55073": "all"}
RANDOM_SEED = 42

BUFFER_M = 15
SEASON_START = (5, 1)
SEASON_END = (12, 15)
CLOUD_COVER_LT = 60
SCL_CLASSES = (4, 5)  # vegetation, bare soil
CLEAR_FRAC_MIN = 0.8
RESOLUTION_M = 10
TILE_SIZE_M = 5000
N_WORKERS = 4
# Concurrent COG reads inside one tile. Separate from the tile worker pool.
N_LOAD_THREADS = 16
LOAD_CHUNK_XY = 2048
LOAD_PAD_M = 30

OFFSET_DATE = "2022-01-25"
DN_OFFSET = 1000.0
SCALE = 10000.0

CORN_CDL_CLASS = 1
MIN_CORN_FRAC = 0.8
MIN_AREA_HA = 2.0
SILAGE_SHARE_MIN = 0.2
SILAGE_SHARE_MAX = 0.6
MIN_TOTAL_ACRES = 10_000

CATALOG = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
S1_COLLECTION = "sentinel-1-rtc"
S2_BANDS = ("B04", "B08", "B11", "B12", "SCL")
S2_REFLECTANCE = ("B04", "B08", "B11", "B12")
S2_RESAMPLING = {
    "B04": "bilinear",
    "B08": "bilinear",
    "B11": "bilinear",
    "B12": "bilinear",
    "SCL": "nearest",
}
S1_BANDS = ("vv", "vh")
S1_RESAMPLING = {"vv": "bilinear", "vh": "bilinear"}

# Applied once, before GDAL opens any remote COG.
GDAL_CLOUD_DEFAULTS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
}

GRIDMET_VARS = ("tmmx", "tmmn", "pr")
# Direct yearly NetCDF files. {var} is tmmx, tmmn, or pr.
GRIDMET_URL_TEMPLATE = "https://www.reacchpna.org/thredds/fileServer/MET/{var}/{var}_{year}.nc"
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"

NASS_SURVEY_CSV = ROOT / "notebooks/dataset/corn_grain_silage_acres_2021_2023.csv"
NASS_CENSUS_CSV = ROOT / "notebooks/dataset/corn_grain_silage_acres_2012_2025.csv"
PROCESSED = ROOT / "data/processed"
RAW = ROOT / "data/raw"
LOG_PATH = ROOT / "DATA_LOG.md"

# National 2023 Cropland Data Layer (30 m) and Crop Sequence Boundaries 2016-2023 rev23.
CDL_PATHS = {2023: "data/raw/cdl/2023_30m_cdls.tif"}
CSB_PATH = "data/raw/csb/NationalCSB_2016-2023_rev23/CSB1623.gdb"
CSB_LAYER = "national1623"
CSB_RELEASE = "2016-2023 rev23"
# State extracts written by src/extract_csb.py. Collect reads these, not the national file.
CSB_EXTRACTS = {
    "PENNSYLVANIA": "data/raw/csb/csb_PA_2023.gpkg",
    "WISCONSIN": "data/raw/csb/csb_WI_2023.gpkg",
}

# Lancaster, Pennsylvania and Marathon, Wisconsin. 2023 survey acres.
TEST_COUNTIES: list[tuple[str, str]] = [
    ("PENNSYLVANIA", "42071"),  # Lancaster: 38,200 grain + 30,600 silage acres, 44% silage
    ("WISCONSIN", "55073"),  # Marathon: 61,300 grain + 61,000 silage acres, 50% silage
]

BARC_LAYER = "barc_silage_grain_corn20212023"
BARC_LABELS = ("Grain", "Silage")
LABEL_COLORS = {"Silage": "#2f7d32", "Grain": "#d6a81d", "": "#4c78a8"}
FOCUS_STATES = ("PENNSYLVANIA", "WISCONSIN", "CALIFORNIA", "MARYLAND")
TOP_STATES = ("PENNSYLVANIA", "WISCONSIN", "CALIFORNIA")

S2_COLUMNS = [
    "field_id", "year", "date", "doy", "ndvi", "ndti", "clear_frac", "n_pixels", "label",
]
S1_COLUMNS = [
    "field_id", "year", "date", "doy", "vv_db", "vh_db", "vh_vv_db",
    "orbit_state", "relative_orbit", "n_pixels", "label",
]
WEATHER_COLUMNS = ["field_id", "year", "date", "tmax_c", "tmin_c", "pr_mm", "gdd"]
GRAIN_DESC = "CORN, GRAIN - ACRES HARVESTED"
SILAGE_DESC = "CORN, SILAGE - ACRES HARVESTED"


def resolve_data_path(path: str | Path | None) -> Path | None:
    """Resolve a config path. Relative paths are from the repo root."""
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def field_limit(fips: str, test: bool = False) -> int | None:
    """How many fields to keep. None means every qualifying field."""
    if test:
        return 4
    raw = N_FIELDS_BY_COUNTY.get(str(fips).zfill(5), N_FIELDS)
    if raw == "all":
        return None
    return int(raw)
