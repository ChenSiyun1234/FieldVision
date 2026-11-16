"""Paths and keys shared by every fieldvision_agents module. Change them here, nowhere else.

Everything is relative to the repository root, so a fresh clone runs as it is. Environment variables
override single paths (in the Keeboh workspace, set FIELDVISION_COUNTY_DATA and FIELDVISION_DEPS):

    FIELDVISION_HOME          repository root (default: two levels above this file)
    FIELDVISION_COUNTY_DATA   <fips>/<year>/fields.geojson + timeseries.parquet (default: data/counties)
    FIELDVISION_RUNS          audit logs, reports, evaluation and benchmark runs, LLM cache (default: runs)
    FIELDVISION_DEPS          a private pip --target folder, such as the workspace's .deps (optional)
    QUICKSTATS_CSV            corn grain and silage acres, 2012-2025 (default: the copy in rag-pipeline)

Keys (NVIDIA_API_KEY, QUICKSTATS_KEY) are read from the environment or from .env at the repository root.
"""

import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("FIELDVISION_HOME", Path(__file__).resolve().parents[2]))
ENV_FILE = REPO / ".env"


def _load_env(path=ENV_FILE):
    """KEY=value lines of .env into os.environ; values already in the environment win."""
    if path.is_file():
        for line in path.read_text().splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name and not name.startswith("#"):
                os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


_load_env()

# The model module (fieldvision) and its frozen models live in this repository. FIELDVISION and
# FIELDVISION_FINAL were two folders in the workspace (team copy and GitHub main); here they are one.
FIELDVISION = REPO
FIELDVISION_FINAL = REPO
# Landsat 8/9 rows beside Sentinel-2 wherever a county has timeseries_landsat.parquet (fieldvision.ls);
# counties without the file run on Sentinel-2 alone. FIELDVISION_LANDSAT=0 turns it off.
LANDSAT = os.environ.get("FIELDVISION_LANDSAT", "1") != "0"
REGISTRY = REPO / "registry"
COUNTY_DATA = Path(os.environ.get("FIELDVISION_COUNTY_DATA", REPO / "data" / "counties"))
RAG_PIPELINE = REPO / "rag-pipeline"                    # the critic's checks and the NASS report index
RECOMMENDER = REPO / "recommender"                      # silage sources for dairies (recommend_sources)
QUICKSTATS_CSV = Path(os.environ.get(
    "QUICKSTATS_CSV", RAG_PIPELINE / "dataset" / "quickstats" / "corn_grain_silage_acres_2012_2025.csv"))
DATA = REPO / "data"                                    # large run data (four-state samples and time series)

RUNS = Path(os.environ.get("FIELDVISION_RUNS", REPO / "runs"))
LOGS = RUNS / "logs"                                    # one audit_<run_id>.jsonl per agent run
REPORTS = RUNS / "reports"
EVALUATION = RUNS / "evaluation"
BENCHMARK = RUNS / "benchmark"
LLM_CACHE = RUNS / "cache" / "llm"                      # replies keyed by request, so reruns cost nothing

# pandas looks for pyarrow once, when it is first imported. If a private package folder is added later,
# read_parquet fails ("No type extension with name arrow.py_extension_type"). Appended, so the
# environment's own numpy and pandas still win.
FIELDVISION_DEPS = Path(os.environ.get("FIELDVISION_DEPS", REPO / ".deps"))
if FIELDVISION_DEPS.is_dir() and str(FIELDVISION_DEPS) not in sys.path:
    sys.path.append(str(FIELDVISION_DEPS))

# Kept for modules written in the workspace: the folder that holds .env, data/ and runs/.
WORKSPACE = REPO
