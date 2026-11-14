"""Fill an extraction run's catalog cache for all blocks in parallel.

The extractor queries STAC once per block, sequentially, which costs about
30 s per block on Planetary Computer. This runs the same query and the same
footprint filter for every block ahead of time and writes the same cache file
(catalog/<block>.json) the extractor reads, so the scene list is identical.
Safe to run while the extractor is running; existing cache files are kept.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from shapely.geometry import shape

from fieldvision.s2.catalog import LEGACY, PLANETARY, PRIMARY, SOURCES, find_scenes
from fieldvision.s2.extract import atomic_json
from fieldvision.s2.geometry import load_fields, make_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start", default="04-01")
    parser.add_argument("--end", default="11-30")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--block-size", type=float, default=6000)
    parser.add_argument("--max-cloud", type=float, default=80)
    parser.add_argument("--collection", choices=SOURCES, default=PRIMARY)
    parser.add_argument("--fallback-source", choices=(LEGACY, PLANETARY), default=PLANETARY)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    start = date.fromisoformat(f"{args.year}-{args.start}").isoformat()
    end = date.fromisoformat(f"{args.year}-{args.end}").isoformat()
    blocks = make_blocks(load_fields(args.fields, args.year), args.block_size)
    (args.out / "catalog").mkdir(parents=True, exist_ok=True)
    fallback = None if args.no_fallback else args.fallback_source

    def fetch(block):
        path = args.out / "catalog" / (block["id"] + ".json")
        if path.exists():
            return block["id"], "cached"
        items, counts = find_scenes(block["bbox"], start, end, args.max_cloud, args.collection, fallback)
        # Same footprint filter as extract.process_block.
        items = [item for item in items if any(shape(item["geometry"]).intersects(f["geometry"]) for f in block["fields"])]
        if not path.exists():
            atomic_json(path, {"items": items, "counts": counts})
        return block["id"], len(items)

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for block_id, result in pool.map(fetch, blocks):
            print(block_id, result, flush=True)


if __name__ == "__main__":
    main()
