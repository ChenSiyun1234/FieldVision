"""Extract dense field time series from public Sentinel-2 COGs.

Run with ``python -m fieldvision.s2.extract --help``.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from shapely.geometry import shape

from .catalog import (LEGACY, PLANETARY, PRIMARY, SOURCES, asset_for, asset_href, calibration, check_vegetation,
                      find_scenes, scale_reflectance)
from .core import BANDS, COLUMNS, coverage_table, interior_geometry, polygon_mask, resolve_duplicates, scl_clear, zonal_medians
from .geometry import load_fields, make_blocks, scene_grid

LOG = logging.getLogger("fieldvision.s2")
GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_CONNECTTIMEOUT": "15",
    "GDAL_HTTP_TIMEOUT": "90",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "5000000",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def replace_with_retry(temporary, path, attempts=8):
    """os.replace, retried: on Windows a reader or virus scanner holding the
    destination open makes the rename fail with access denied for a moment."""
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.25 * 2 ** attempt, 8))


def atomic_json(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
    replace_with_retry(temporary, path)


def empty_table():
    return pd.DataFrame({column: pd.Series(dtype="int64" if column in ("n_pixels", "n_clear") else
                        "float64" if column in BANDS or column == "clear_fraction" else "object") for column in COLUMNS})


def write_parquet(path, table):
    temporary = path.with_suffix(".parquet.tmp")
    table.to_parquet(temporary, index=False)
    replace_with_retry(temporary, path)


def _has_asset(item, name):
    try:
        return "href" in asset_for(item, name)
    except KeyError:
        return False


def read_raw(item, name, grid, retries=3):
    """Raw digital numbers of one band on the block grid; nodata becomes 0."""
    asset = asset_for(item, name)
    nodata = asset.get("raster:bands", [{}])[0].get("nodata", 0)
    crs, affine, dimensions = grid[:3]
    for attempt in range(retries):
        try:
            # Planetary Computer URLs carry a short-lived token; re-sign per attempt.
            with rasterio.Env(**GDAL_OPTIONS):
                with rasterio.open(asset_href(item, name)) as source:
                    with WarpedVRT(source, crs=crs, transform=affine, width=dimensions[1], height=dimensions[0],
                                   src_nodata=nodata, nodata=0, resampling=Resampling.nearest) as vrt:
                        return vrt.read(1, out_dtype="float32")
        except (rasterio.errors.RasterioError, OSError):
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def read_band(item, name, grid, retries=3):
    """One COG read per (scene, band, block), with a private dataset per thread."""
    if name == "SCL":
        return read_raw(item, name, grid, retries)
    asset = asset_for(item, name)
    calibration(item, asset)  # Fail before networking if metadata is ambiguous.
    return scale_reflectance(read_raw(item, name, grid, retries), item, asset)


def aggregate_scene(block, item, grid, arrays):
    _, affine, dimensions, geometries = grid
    selected = [interior_geometry(geom, affine, dimensions) for geom in geometries]
    labels = rasterize([(geom, index + 1) for index, (geom, _) in enumerate(selected)],
                       out_shape=dimensions, transform=affine, dtype="int32")
    scl = arrays["SCL"]
    bands = {band: arrays[band] for band in BANDS}
    sanity = check_vegetation(bands, scl, labels)
    # Values outside the physical output range are missing, never clipped to zero.
    # The vegetation check above sees unmodified values to catch offset mistakes.
    for band in bands.values():
        band[(band < 0) | (band > 1)] = np.nan
    clear = scl_clear(scl)
    result = zonal_medians(labels, bands, clear, len(geometries))
    overlap_fields = 0
    for index, (geom, _) in enumerate(selected):
        mask = polygon_mask(geom, affine, dimensions)
        if int(mask.sum()) != int(result.iloc[index].n_pixels):
            # Shared/overlapping parcels retain their pixels without another COG read.
            overlap_fields += 1
            replacement = zonal_medians(mask.astype("int32"), bands, clear, 1)
            result.loc[index, replacement.columns] = replacement.iloc[0].values
    result["field_id"] = [field["field_id"] for field in block["fields"]]
    result["date"] = item["properties"]["datetime"][:10]
    result["scene_id"] = item["id"]
    result["collection"] = item["collection"]
    sanity.update({"inset_fields": sum(inset for _, inset in selected), "overlap_fields": overlap_fields,
                   "zero_coverage_fields": int((result.n_clear == 0).sum())})
    return result[COLUMNS], sanity


def process_block(block, args, pool):
    path = args.out / "blocks" / (block["id"] + ".parquet")
    status_path = path.with_suffix(".json")
    cache_path = args.out / "catalog" / (block["id"] + ".json")
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous.get("status") == "complete" and path.exists():
        LOG.info("Skipping complete block %s", block["id"])
        return previous
    started = time.monotonic()
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        items, counts = cached["items"], cached["counts"]
    else:
        items, counts = find_scenes(block["bbox"], args.start_date, args.end_date, args.max_cloud,
                                   args.collection, None if args.no_fallback else args.fallback_source)
        # STAC query intersects the block bbox; only actual polygon footprints count.
        items = [item for item in items if any(shape(item["geometry"]).intersects(f["geometry"]) for f in block["fields"])]
        atomic_json(cache_path, {"items": items, "counts": counts})
    # Some upstream items lack band assets altogether (seen: a 2024 Collection 1
    # item with no 10 m bands). No retry can read them; record and skip them.
    skipped = {}
    for item in items:
        missing = [name for name in (*BANDS, "SCL") if not _has_asset(item, name)]
        if missing:
            skipped[item["id"]] = "missing assets: " + ",".join(missing)
    items = [item for item in items if item["id"] not in skipped]
    completed = set(previous.get("completed_scene_ids", [])) if path.exists() else set()
    frames = [pd.read_parquet(path)] if path.exists() else []
    diagnostics = previous.get("diagnostics", {}) if path.exists() else {}
    errors = {}
    pending = [item for item in items if item["id"] not in completed]
    LOG.info("Block %s: %d fields, %d scenes, %d pending", block["id"], len(block["fields"]), len(items), len(pending))
    # Keep a bounded number of scenes in memory while the common thread pool reads
    # (scene, band) tasks. Sequential blocks avoid repeated large memory peaks.
    for offset in range(0, len(pending), args.scene_batch):
        batch = []
        for item in pending[offset:offset + args.scene_batch]:
            try:
                grid = scene_grid(block, item)
                futures = {name: pool.submit(read_band, item, name, grid) for name in (*BANDS, "SCL")}
                batch.append((item, grid, futures))
            except Exception as exc:
                errors[item["id"]] = f"{type(exc).__name__}: {exc}"
        for item, grid, futures in batch:
            arrays, failure = {}, None
            # Drain every future even if one fails; no hidden worker failures.
            for name, future in futures.items():
                try:
                    arrays[name] = future.result()
                except Exception as exc:
                    failure = exc
            try:
                if failure is not None:
                    raise failure
                rows, diagnostic = aggregate_scene(block, item, grid, arrays)
                frames.append(rows)
                diagnostics[item["id"]] = diagnostic
                completed.add(item["id"])
            except Exception as exc:
                errors[item["id"]] = f"{type(exc).__name__}: {exc}"
                LOG.warning("Scene %s failed: %s", item["id"], exc)
            del arrays
        LOG.info("Block %s: processed %d/%d scenes (%d failed)", block["id"],
                 min(offset + args.scene_batch, len(pending)), len(pending), len(errors))
        table = resolve_duplicates(pd.concat(frames, ignore_index=True)) if frames else empty_table()
        write_parquet(path, table)
        frames = [table]
        # Data precedes the checkpoint: interruption can repeat work but cannot
        # cause a completed scene to be omitted from the saved data.
        atomic_json(status_path, {"status": "running", "completed_scene_ids": sorted(completed),
                                  "diagnostics": diagnostics, "scene_errors": errors})
    table = resolve_duplicates(pd.concat(frames, ignore_index=True)) if frames else empty_table()
    write_parquet(path, table)
    status = {"status": "partial" if errors else "complete", "field_count": len(block["fields"]),
              "field_ids": [f["field_id"] for f in block["fields"]], "bbox": block["bbox"],
              "scenes_found": len(items), **counts, "scenes_succeeded": len(completed),
              "completed_scene_ids": sorted(completed), "scene_errors": errors, "skipped_items": skipped,
              "diagnostics": diagnostics,
              "rows": len(table), "wall_seconds": previous.get("wall_seconds", 0) + time.monotonic() - started}
    atomic_json(status_path, status)
    return status


def run(args):
    start = time.monotonic()
    args.start_date = date.fromisoformat(f"{args.year}-{args.start}").isoformat()
    args.end_date = date.fromisoformat(f"{args.year}-{args.end}").isoformat()
    if args.start_date > args.end_date:
        raise ValueError("Start must precede end within the selected year")
    if args.workers < 1 or args.scene_batch < 1 or not 0 < args.block_size <= 6000:
        raise ValueError("Workers/batch must be positive; block size must be in (0, 6000]")
    if not 0 <= args.max_cloud <= 100 or not 0 <= args.min_clear_fraction <= 1 or args.min_clear_pixels < 1:
        raise ValueError("Invalid cloud/clear thresholds")
    args.out.mkdir(parents=True, exist_ok=True)
    for subdir in ("blocks", "catalog"):
        (args.out / subdir).mkdir(exist_ok=True)
    fields = load_fields(args.fields, args.year)
    parameters = {key: value for key, value in vars(args).items() if key not in ("fields", "out", "workers", "scene_batch")}
    # Resume requires identical inputs, science options and extractor source.
    source_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()
    signature = hashlib.sha256(args.fields.read_bytes() + json.dumps(parameters, sort_keys=True).encode() + source_hash.encode()).hexdigest()
    manifest_path = args.out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous.get("signature") != signature:
        raise ValueError("Output belongs to different inputs/options/code; use a new --out directory")
    blocks = make_blocks(fields, args.block_size)
    manifest = {"signature": signature, "source_sha256": source_hash, "parameters": parameters,
                "fields_file": str(args.fields.resolve()), "workers": args.workers, "scene_batch": args.scene_batch,
                "started_at": previous["started_at"] if previous else utc_now(), "attempt_started_at": utc_now(),
                "ended_at": None, "status": "running", "input_fields": len(fields), "block_count": len(blocks),
                "invalid_fields": {f["field_id"]: f["error"] for f in fields if f.get("error")}, "blocks": {}}
    atomic_json(manifest_path, manifest)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for block in blocks:
            try:
                status = process_block(block, args, pool)
            except Exception as exc:
                LOG.exception("Block %s failed", block["id"])
                status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                          "field_ids": [f["field_id"] for f in block["fields"]]}
            manifest["blocks"][block["id"]] = status
            atomic_json(manifest_path, manifest)
    tables = [pd.read_parquet(args.out / "blocks" / (block["id"] + ".parquet")) for block in blocks
              if (args.out / "blocks" / (block["id"] + ".parquet")).exists()]
    table = resolve_duplicates(pd.concat(tables, ignore_index=True)) if tables else empty_table()
    write_parquet(args.out / "timeseries.parquet", table)
    coverage = coverage_table(fields, table, args.start_date, args.end_date, args.min_clear_pixels, args.min_clear_fraction)
    for block in blocks:
        if manifest["blocks"][block["id"]]["status"] == "failed":
            coverage.loc[coverage.field_id.isin([f["field_id"] for f in block["fields"]]), "status"] = "block_failed"
    coverage.to_csv(args.out / "fields_coverage.csv", index=False)
    incomplete = bool(manifest["invalid_fields"]) or any(b["status"] != "complete" for b in manifest["blocks"].values())
    manifest.update({"status": "partial" if incomplete else "complete", "ended_at": utc_now(),
                     "attempt_wall_seconds": time.monotonic() - start, "rows": len(table),
                     "unique_scenes": int(table.scene_id.nunique()),
                     "fields_with_usable_dates": int((coverage.usable_dates > 0).sum()),
                     "scene_failures": sum(len(b.get("scene_errors", {})) for b in manifest["blocks"].values())})
    atomic_json(manifest_path, manifest)
    LOG.info("%s: %d rows, %d/%d fields usable, %.1f seconds", manifest["status"], len(table),
             manifest["fields_with_usable_dates"], len(fields), manifest["attempt_wall_seconds"])
    return 2 if incomplete else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start", default="04-01", help="Inclusive MM-DD")
    parser.add_argument("--end", default="11-30", help="Inclusive MM-DD")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--scene-batch", type=int, default=3)
    parser.add_argument("--block-size", type=float, default=6000)
    parser.add_argument("--max-cloud", type=float, default=80)
    parser.add_argument("--collection", choices=SOURCES, default=PRIMARY)
    parser.add_argument("--fallback-source", choices=(LEGACY, PLANETARY), default=PLANETARY,
                        help="Source for acquisitions missing from Collection 1 (years <= 2022)")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--min-clear-pixels", type=int, default=1)
    parser.add_argument("--min-clear-fraction", type=float, default=0.2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
