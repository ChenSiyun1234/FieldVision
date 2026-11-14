"""Check public catalog availability and save one item per region/year."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import requests

REGIONS = {
    "austria": [15.7, 48.0, 15.8, 48.1],
    "netherlands": [5.5, 52.0, 5.6, 52.1],
    "western_france": [-1.0, 47.0, -0.9, 47.1],
    "us_midwest": [-93.7, 41.9, -93.6, 42.0],
}


def probe(task):
    region, bbox, year, collection, start, end = task
    params = {"collections": collection, "bbox": ",".join(map(str, bbox)),
              "datetime": f"{year}-{start}T00:00:00Z/{year}-{end}T23:59:59Z", "limit": 1}
    for attempt in range(4):
        try:
            response = requests.get("https://earth-search.aws.element84.com/v1/search", params=params, timeout=90)
            response.raise_for_status()
            data = response.json()
            result = {"region": region, "bbox": bbox, "year": year, "collection": collection,
                      "start": start, "end": end,
                      "matched": data.get("numberMatched", data.get("context", {}).get("matched")),
                      "available": bool(data.get("features")), "item": next(iter(data.get("features", [])), None)}
            print(json.dumps({k: v for k, v in result.items() if k != "item"}), flush=True)
            return result
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="01-01", help="Inclusive MM-DD")
    parser.add_argument("--end", default="12-31", help="Inclusive MM-DD")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(region, bbox, year, collection, args.start, args.end) for region, bbox in REGIONS.items()
             for year in (2022, 2024, 2025)
             for collection in ("sentinel-2-c1-l2a", "sentinel-2-l2a")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(probe, tasks))
    (args.out / "catalog_probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
