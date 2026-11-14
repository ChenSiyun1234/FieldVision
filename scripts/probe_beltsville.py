"""Count Sentinel-2 L2A scenes over the Beltsville (BARC) farm per source and season."""
import argparse
from collections import Counter
import json
from pathlib import Path

import requests

BBOX = [-76.96, 39.00, -76.84, 39.07]
SOURCES = {
    "earthsearch:sentinel-2-c1-l2a": ("https://earth-search.aws.element84.com/v1/search", "sentinel-2-c1-l2a"),
    "earthsearch:sentinel-2-l2a": ("https://earth-search.aws.element84.com/v1/search", "sentinel-2-l2a"),
    "planetary:sentinel-2-l2a": ("https://planetarycomputer.microsoft.com/api/stac/v1/search", "sentinel-2-l2a"),
}


def search(url, collection, year, start, end):
    body = {"collections": [collection], "bbox": BBOX, "limit": 200,
            "datetime": f"{year}-{start}T00:00:00Z/{year}-{end}T23:59:59Z"}
    items = []
    while True:
        response = requests.post(url, json=body, timeout=90)
        response.raise_for_status()
        data = response.json()
        items += data["features"]
        link = next((l for l in data.get("links", []) if l["rel"] == "next"), None)
        if not link:
            return items
        if link.get("method", "GET") == "POST":
            body = link.get("body", body)
            url = link["href"]
        else:
            response = None
            url, body = link["href"], None
            data = requests.get(url, timeout=90).json()
            items += data["features"]
            if not any(l["rel"] == "next" for l in data.get("links", [])):
                return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024])
    args = parser.parse_args()
    rows = []
    for year in args.years:
        for name, (url, collection) in SOURCES.items():
            items = search(url, collection, year, "04-01", "11-30")
            low = [i for i in items if i["properties"].get("eo:cloud_cover", 100) < 80]
            dates = sorted({i["properties"]["datetime"][:10] for i in low})
            baselines = Counter(str(i["properties"].get("s2:processing_baseline")) for i in low)
            flags = Counter(str(i["properties"].get("earthsearch:boa_offset_applied")) for i in low)
            tiles = Counter(str(i["properties"].get("s2:mgrs_tile") or i["properties"].get("grid:code")) for i in low)
            row = {"year": year, "source": name, "items": len(items), "items_cloud_lt80": len(low),
                   "unique_dates_cloud_lt80": len(dates), "baselines": dict(baselines),
                   "boa_offset_applied": dict(flags), "tiles": dict(tiles)}
            print(json.dumps(row), flush=True)
            rows.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "beltsville_coverage.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
