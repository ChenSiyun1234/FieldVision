"""Run the toolbox on one county-season and print what each method returns.

Usage, from anywhere:
    python scripts/toolbox_demo.py [fips] [year] [policy]
Defaults: 24033 2022 standard (the demo county built from the Beltsville fields).
Needs data/counties/<fips>/<year>/fields.geojson and timeseries.parquet inside the repo.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".deps"), str(ROOT / "src")]

from fieldvision.toolbox import Toolbox  # noqa: E402


def show(name, value):
    print(f"--- {name}")
    print(json.dumps(value, indent=2, default=str))


def main():
    fips = sys.argv[1] if len(sys.argv) > 1 else "24033"
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2022
    policy = sys.argv[3] if len(sys.argv) > 3 else "standard"
    tools = Toolbox()
    show("scout", tools.scout(fips, year))
    show("route_options", tools.route_options(fips, year))
    show("extract_features", tools.extract_features(fips, year))
    result = tools.classify(fips, year, policy)
    show("classify", result)
    show("aggregate", tools.aggregate(fips, year))
    rows = Path(str(result["results_path"])).with_suffix(".jsonl")
    if rows.exists():
        print("--- first two per-field results in", rows.name)
        with rows.open(encoding="utf-8") as fh:
            for _, line in zip(range(2), fh):
                print(line.rstrip())


if __name__ == "__main__":
    main()
