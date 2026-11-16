"""Measure calibration agreement between Sentinel-2 L2A sources on the same acquisitions.

For each source (Earth Search Collection 1, Earth Search legacy, Planetary Computer)
this reads raw digital numbers for the same field polygons on the same 10 m grid,
takes per-field medians over clear pixels, and writes them to Parquet. The
analysis step pairs acquisitions (same tile and date) and reports, per band, the
reflectance difference to a reference source under each offset interpretation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.features import rasterize
from shapely.geometry import shape

from fieldvision.s2.catalog import LEGACY, PC_URL, PLANETARY, acquisition_key, search, search_source
from fieldvision.s2.core import BANDS, scl_clear
from fieldvision.s2.extract import read_raw
from fieldvision.s2.geometry import load_fields, make_blocks, scene_grid


def field_medians(block, item, pool):
    grid = scene_grid(block, item)
    futures = {name: pool.submit(read_raw, item, name, grid) for name in (*BANDS, "SCL")}
    arrays = {name: future.result() for name, future in futures.items()}
    _, affine, dims, geometries = grid
    labels = rasterize([(g.buffer(-5), i + 1) for i, g in enumerate(geometries)], out_shape=dims,
                       transform=affine, dtype="int32")
    valid = scl_clear(arrays["SCL"]) & np.all([arrays[b] > 0 for b in BANDS], axis=0)
    veg = arrays["SCL"] == 4
    rows = []
    props = item["properties"]
    for index, field in enumerate(block["fields"]):
        inside = labels == index + 1
        mask = inside & valid
        row = {"field_id": field["field_id"], "date": props["datetime"][:10], "scene_id": item["id"],
               "source": item["collection"], "tile": acquisition_key(item)[0],
               "baseline": props.get("s2:processing_baseline"),
               "offset_flag": props.get("earthsearch:boa_offset_applied"),
               "n_pixels": int(inside.sum()), "n_clear": int(mask.sum()), "n_veg": int((mask & veg).sum())}
        for band in BANDS:
            row[band] = float(np.median(arrays[band][mask])) if mask.any() else np.nan
        rows.append(row)
    return rows


def collect(args):
    fields = load_fields(args.fields, args.year)
    blocks = make_blocks(fields)
    bbox = [min(b["bbox"][0] for b in blocks), min(b["bbox"][1] for b in blocks),
            max(b["bbox"][2] for b in blocks), max(b["bbox"][3] for b in blocks)]
    start, end = f"{args.year}-{args.start}", f"{args.year}-{args.end}"
    by_source = {s: search_source(s, bbox, start, end, args.max_cloud) for s in args.sources}
    if PLANETARY in by_source and args.pc_all_baselines:
        # Keep both the original and the reprocessed product for each acquisition.
        by_source[PLANETARY] = search(LEGACY, bbox, start, end, args.max_cloud, url=PC_URL)
        for item in by_source[PLANETARY]:
            item["collection"] = PLANETARY
    if args.dates:
        by_source = {s: [i for i in items if i["properties"]["datetime"][:10] in args.dates]
                     for s, items in by_source.items()}
    for source, items in by_source.items():
        print(source, len(items), "items", flush=True)
    # Keep acquisitions present in every requested source.
    keys = [{acquisition_key(i)[:2] for i in items} for items in by_source.values()]
    common = sorted(set.intersection(*keys), key=lambda k: k[1])
    if args.max_dates and len(common) > args.max_dates:
        common = [common[int(i)] for i in np.linspace(0, len(common) - 1, args.max_dates)]
    print(len(common), "common acquisitions:", [k[1] for k in common], flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for key in common:
            for source, items in by_source.items():
                for item in (i for i in items if acquisition_key(i)[:2] == key):
                    for block in blocks:
                        try:
                            rows += field_medians(block, item, pool)
                        except Exception as exc:  # report and keep going
                            print("FAILED", item["id"], type(exc).__name__, exc, flush=True)
            print("done", key, flush=True)
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"raw_dn_{args.year}{args.tag}.parquet"
    table.to_parquet(path, index=False)
    print("wrote", path, len(table), "rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start", default="04-01")
    parser.add_argument("--end", default="11-30")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--max-cloud", type=float, default=20)
    parser.add_argument("--max-dates", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dates", nargs="*", default=[], help="Only these YYYY-MM-DD acquisitions")
    parser.add_argument("--pc-all-baselines", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", type=Path, required=True)
    collect(parser.parse_args())
