"""Per-field Landsat 8/9 surface reflectance in the Sentinel-2 table layout.

Run with ``python -m fieldvision.ls.extract --help``.

Source: Planetary Computer ``landsat-c2-l2`` (Collection 2 Level-2), Landsat 8 and 9,
tier T1 only (T2 has poorer geolocation; skipped items are recorded). Rows use the
fieldvision.s2 columns so features can read both: blue->B02, green->B03, red->B04,
nir08->B08 and B8A, swir16->B11, swir22->B12; B05-B07 are NaN. Reflectance =
DN * scale + offset from each asset's raster:bands (Collection 2: 0.0000275, -0.2),
DN 0 is nodata. Clear pixels: QA_PIXEL bits 0-5 all unset (fill, dilated cloud,
cirrus, cloud, cloud shadow, snow); water is kept, as for Sentinel-2.

Same blocks (from the same fields file and block size), 5 m inward buffer counted on
the 30 m grid, per-field medians, checkpoints, resume and manifest as the Sentinel-2
extractor. A plausibility check (median red and NIR over clear field pixels with
NDVI > 0.6) rejects and logs a scene; nothing is corrected. Default window 1 Aug to
30 Nov: this product fills harvest-season cloud gaps.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from shapely.geometry import shape

from fieldvision.s2.catalog import http_session
from fieldvision.s2.core import BANDS, COLUMNS, coverage_table, interior_geometry, polygon_mask, resolve_duplicates, zonal_medians
from fieldvision.s2.geometry import load_fields, make_blocks, project

LOG = logging.getLogger("fieldvision.ls")
PC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "landsat-c2-l2"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/" + COLLECTION
PLATFORMS = ("landsat-8", "landsat-9")
TIERS = ("T1",)
# Sentinel-2 column -> Landsat asset
BAND_ASSETS = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir08", "B8A": "nir08", "B11": "swir16",
               "B12": "swir22"}
READ_ASSETS = ("blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel")
QA_NOT_CLEAR = 0b111111            # bits 0-5: fill, dilated cloud, cirrus, cloud, cloud shadow, snow
GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_CONNECTTIMEOUT": "15",
    "GDAL_HTTP_TIMEOUT": "90",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "5000000",
}
FINGERPRINT_FILES = ("ls/*.py", "s2/core.py", "s2/geometry.py", "s2/catalog.py")


def fingerprint(package_dir):
    files = sorted({p for pattern in FINGERPRINT_FILES for p in Path(package_dir).glob(pattern)},
                   key=lambda p: p.relative_to(package_dir).as_posix())
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def replace_with_retry(temporary, path, attempts=8):
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


def write_parquet(path, table):
    temporary = path.with_suffix(".parquet.tmp")
    table.to_parquet(temporary, index=False)
    replace_with_retry(temporary, path)


def empty_table():
    return pd.DataFrame({c: pd.Series(dtype="int64" if c in ("n_pixels", "n_clear") else
                         "float64" if c in BANDS or c == "clear_fraction" else "object") for c in COLUMNS})


# ---------------------------------------------------------------- catalog and token

def search(bbox, start, end, max_cloud, session=None):
    session = session or http_session()
    params = {"collections": COLLECTION, "bbox": ",".join(map(str, bbox)),
              "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 200}
    url, method, body, items, seen = PC_URL + "/search", "GET", None, {}, set()
    while url:
        response = session.request(method, url, params=params, json=body, timeout=(15, 90))
        response.raise_for_status()
        data = response.json()
        for item in data.get("features", []):
            item["collection"] = COLLECTION
            items[item["id"]] = item
        link = next((l for l in data.get("links", []) if l["rel"] == "next"), None)
        if not link or str(link) in seen:
            break
        seen.add(str(link))
        url, method, params, body = link["href"], link.get("method", "GET"), None, link.get("body")
    return sorted((i for i in items.values() if float(i["properties"].get("eo:cloud_cover", 100)) < max_cloud),
                  key=lambda i: (i["properties"]["datetime"], i["id"]))


def unusable_reason(item):
    p = item["properties"]
    if p.get("platform") not in PLATFORMS:
        return f"platform {p.get('platform')}"
    if p.get("landsat:collection_category") not in TIERS:
        return f"tier {p.get('landsat:collection_category')}"
    missing = [a for a in READ_ASSETS if a not in item.get("assets", {})]
    if missing:
        return "missing assets: " + ",".join(missing)
    return None


_TOKEN = {"token": None, "expires": 0.0}
_LOCK = threading.Lock()


def token():
    with _LOCK:
        if _TOKEN["token"] is None or time.time() > _TOKEN["expires"] - 300:
            with http_session() as session:
                response = session.get(TOKEN_URL, timeout=(15, 60))
                response.raise_for_status()
                data = response.json()
            try:
                expires = datetime.fromisoformat(data["msft:expiry"].replace("Z", "+00:00")).timestamp()
            except (KeyError, ValueError):
                expires = time.time() + 1800
            _TOKEN.update(token=data["token"], expires=expires)
        return _TOKEN["token"]


# ---------------------------------------------------------------- grid, reads, calibration

def scene_grid(block, item):
    """Block window on the scene's own 30 m grid (aligned to the red asset's transform)."""
    props, red = item["properties"], item["assets"]["red"]
    crs = props.get("proj:epsg") or red.get("proj:epsg")
    affine = red.get("proj:transform") or props.get("proj:transform")
    if crs is None or affine is None:
        raise ValueError("Item lacks proj:epsg/proj:transform")
    geometries = [project(f["geometry"], 4326, crs) for f in block["fields"]]
    bounds = np.array([g.bounds for g in geometries])
    west, south = bounds[:, :2].min(axis=0)
    east, north = bounds[:, 2:].max(axis=0)
    res, x0, y0 = affine[0], affine[2], affine[5]
    west = x0 + math.floor((west - x0) / res) * res
    north = y0 + math.ceil((north - y0) / res) * res
    width, height = max(1, math.ceil((east - west) / res)), max(1, math.ceil((north - south) / res))
    if width * height > 4_000_000:
        raise ValueError("Block window exceeds 4 million pixels")
    return crs, from_origin(west, north, res, res), (height, width), geometries


def calibration(asset):
    meta = asset.get("raster:bands", [{}])[0]
    if "scale" not in meta or "offset" not in meta:
        raise ValueError("Reflectance asset lacks raster:bands scale/offset")
    scale, offset = float(meta["scale"]), float(meta["offset"])
    if not (np.isfinite(scale) and scale > 0 and np.isfinite(offset)):
        raise ValueError("Invalid scale/offset")
    return scale, offset, meta.get("nodata", 0)


def read_asset(item, name, grid, retries=3):
    crs, affine, dims = grid[:3]
    asset = item["assets"][name]
    nodata = asset.get("raster:bands", [{}])[0].get("nodata", 0)
    href = asset["href"].split("?")[0]
    for attempt in range(retries):
        try:
            with rasterio.Env(**GDAL_OPTIONS):
                with rasterio.open(href + "?" + token()) as source:
                    with WarpedVRT(source, crs=crs, transform=affine, width=dims[1], height=dims[0],
                                   src_nodata=nodata, nodata=nodata, resampling=Resampling.nearest) as vrt:
                        return vrt.read(1, out_dtype="float64" if name == "qa_pixel" else "float32")
        except (rasterio.errors.RasterioError, OSError):
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def reflectance(raw, asset):
    scale, offset, nodata = calibration(asset)
    value = raw.astype("float32") * scale + offset
    value[(raw == nodata) | ~np.isfinite(raw)] = np.nan
    return value


def qa_clear(qa):
    """True where none of fill, dilated cloud, cirrus, cloud, cloud shadow, snow is set."""
    qa = np.nan_to_num(qa, nan=1).astype("uint16")
    return (qa & QA_NOT_CLEAR) == 0


def check_plausible(bands, clear, labels):
    """Reject a scene whose vegetated clear field pixels have implausible red/NIR."""
    red, nir = bands["B04"], bands["B08"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
    mask = clear & (labels > 0) & np.isfinite(ndvi) & (ndvi > 0.6)
    count = int(mask.sum())
    if count < 20:
        return {"vegetation_pixels": count, "checked": False}
    r, n = float(np.median(red[mask])), float(np.median(nir[mask]))
    if not (0 < r < 0.35 and n > r + 0.02 and n < 1):
        raise ValueError(f"Vegetation reflectance sanity check failed: red={r:.4f}, nir={n:.4f}, pixels={count}")
    return {"vegetation_pixels": count, "checked": True, "median_red": r, "median_nir": n}


def aggregate_scene(block, item, grid, arrays):
    _, affine, dims, geometries = grid
    selected = [interior_geometry(g, affine, dims) for g in geometries]
    labels = rasterize([(g, i + 1) for i, (g, _) in enumerate(selected)], out_shape=dims, transform=affine,
                       dtype="int32")
    bands = {col: reflectance(arrays[asset], item["assets"][asset]) for col, asset in BAND_ASSETS.items()}
    clear = qa_clear(arrays["qa_pixel"])
    sanity = check_plausible(bands, clear, labels)
    for band in bands.values():
        band[(band < 0) | (band > 1)] = np.nan
    result = zonal_medians(labels, bands, clear, len(geometries))
    for index, (geom, _) in enumerate(selected):
        mask = polygon_mask(geom, affine, dims)
        if int(mask.sum()) != int(result.iloc[index].n_pixels):
            replacement = zonal_medians(mask.astype("int32"), bands, clear, 1)
            result.loc[index, replacement.columns] = replacement.iloc[0].values
    for col in BANDS:
        if col not in result:
            result[col] = np.nan            # B05-B07: no Landsat equivalent
    result["field_id"] = [f["field_id"] for f in block["fields"]]
    result["date"] = item["properties"]["datetime"][:10]
    result["scene_id"] = item["id"]
    result["collection"] = item["collection"]
    sanity.update({"inset_fields": sum(inset for _, inset in selected),
                   "zero_coverage_fields": int((result.n_clear == 0).sum())})
    return result[COLUMNS], sanity


# ---------------------------------------------------------------- blocks and run

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
        items = json.loads(cache_path.read_text())["items"]
    else:
        items = search(block["bbox"], args.start_date, args.end_date, args.max_cloud)
        items = [i for i in items if any(shape(i["geometry"]).intersects(f["geometry"]) for f in block["fields"])]
        atomic_json(cache_path, {"items": items})
    skipped = {i["id"]: r for i in items if (r := unusable_reason(i))}
    items = [i for i in items if i["id"] not in skipped]
    completed = set(previous.get("completed_scene_ids", [])) if path.exists() else set()
    frames = [pd.read_parquet(path)] if path.exists() else []
    diagnostics = previous.get("diagnostics", {}) if path.exists() else {}
    errors = {}
    pending = [i for i in items if i["id"] not in completed]
    LOG.info("Block %s: %d fields, %d scenes, %d pending", block["id"], len(block["fields"]), len(items), len(pending))
    for offset in range(0, len(pending), args.scene_batch):
        batch = []
        for item in pending[offset:offset + args.scene_batch]:
            try:
                for asset in BAND_ASSETS.values():
                    calibration(item["assets"][asset])     # fail before networking
                grid = scene_grid(block, item)
                batch.append((item, grid, {a: pool.submit(read_asset, item, a, grid) for a in READ_ASSETS}))
            except Exception as exc:
                errors[item["id"]] = f"{type(exc).__name__}: {exc}"
        for item, grid, futures in batch:
            arrays, failure = {}, None
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
        LOG.info("Block %s: processed %d/%d scenes (%d failed)", block["id"],
                 min(offset + args.scene_batch, len(pending)), len(pending), len(errors))
        table = resolve_duplicates(pd.concat(frames, ignore_index=True)) if frames else empty_table()
        write_parquet(path, table)
        frames = [table]
        atomic_json(status_path, {"status": "running", "completed_scene_ids": sorted(completed),
                                  "diagnostics": diagnostics, "scene_errors": errors})
    table = resolve_duplicates(pd.concat(frames, ignore_index=True)) if frames else empty_table()
    write_parquet(path, table)
    status = {"status": "partial" if errors else "complete", "field_count": len(block["fields"]),
              "field_ids": [f["field_id"] for f in block["fields"]], "bbox": block["bbox"],
              "scenes_found": len(items), "scenes_succeeded": len(completed),
              "completed_scene_ids": sorted(completed), "scene_errors": errors, "skipped_items": skipped,
              "diagnostics": diagnostics, "rows": len(table),
              "wall_seconds": previous.get("wall_seconds", 0) + time.monotonic() - started}
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
    for sub in ("blocks", "catalog"):
        (args.out / sub).mkdir(exist_ok=True)
    fields = load_fields(args.fields, args.year)
    # --blocks only limits which blocks this invocation processes; it never changes a row.
    parameters = {k: v for k, v in vars(args).items() if k not in ("fields", "out", "workers", "scene_batch", "blocks")}
    source_hash = fingerprint(Path(__file__).parent.parent)
    signature = hashlib.sha256(args.fields.read_bytes() + json.dumps(parameters, sort_keys=True).encode()
                               + source_hash.encode()).hexdigest()
    manifest_path = args.out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous.get("signature") != signature:
        raise ValueError("Output belongs to different inputs/options/code; use a new --out directory")
    blocks = make_blocks(fields, args.block_size)
    selected = [b for b in blocks if not args.blocks or b["id"] in args.blocks]
    manifest = {"signature": signature, "source_sha256": source_hash, "parameters": parameters,
                "product": "landsat-c2-l2 (Landsat 8/9, T1) per-field surface reflectance in Sentinel-2 columns",
                "fields_file": str(args.fields.resolve()), "workers": args.workers, "scene_batch": args.scene_batch,
                "started_at": previous["started_at"] if previous and previous.get("started_at") else utc_now(),
                "attempt_started_at": utc_now(), "ended_at": None, "status": "running", "input_fields": len(fields),
                "block_count": len(blocks), "invalid_fields": {f["field_id"]: f["error"] for f in fields if f.get("error")},
                "blocks": dict(previous.get("blocks", {})) if previous else {}}
    atomic_json(manifest_path, manifest)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for block in selected:
            try:
                status = process_block(block, args, pool)
            except Exception as exc:
                LOG.exception("Block %s failed", block["id"])
                status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                          "field_ids": [f["field_id"] for f in block["fields"]]}
            manifest["blocks"][block["id"]] = status
            atomic_json(manifest_path, manifest)
    tables = [pd.read_parquet(args.out / "blocks" / (b["id"] + ".parquet")) for b in blocks
              if (args.out / "blocks" / (b["id"] + ".parquet")).exists()]
    table = resolve_duplicates(pd.concat(tables, ignore_index=True)) if tables else empty_table()
    write_parquet(args.out / "timeseries.parquet", table)
    in_scope = fields if not args.blocks else [f for b in selected for f in b["fields"]]
    coverage = coverage_table(in_scope, table, args.start_date, args.end_date, args.min_clear_pixels,
                              args.min_clear_fraction)
    coverage.to_csv(args.out / "fields_coverage.csv", index=False)
    done = [manifest["blocks"].get(b["id"], {}).get("status") for b in blocks]
    incomplete = bool(manifest["invalid_fields"]) or any(s != "complete" for s in done)
    manifest.update({"status": "partial" if incomplete else "complete", "ended_at": utc_now(),
                     "attempt_wall_seconds": time.monotonic() - start, "rows": len(table),
                     "unique_scenes": int(table.scene_id.nunique()),
                     "fields_with_usable_dates": int((coverage.usable_dates > 0).sum()),
                     "scene_failures": sum(len(b.get("scene_errors", {})) for b in manifest["blocks"].values())})
    atomic_json(manifest_path, manifest)
    LOG.info("%s: %d rows, %d/%d fields usable, %.1f seconds", manifest["status"], len(table),
             manifest["fields_with_usable_dates"], len(coverage), manifest["attempt_wall_seconds"])
    return 2 if incomplete else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start", default="08-01", help="Inclusive MM-DD")
    parser.add_argument("--end", default="11-30", help="Inclusive MM-DD")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--scene-batch", type=int, default=6)
    parser.add_argument("--block-size", type=float, default=6000)
    parser.add_argument("--max-cloud", type=float, default=80)
    parser.add_argument("--min-clear-pixels", type=int, default=1)
    parser.add_argument("--min-clear-fraction", type=float, default=0.2)
    parser.add_argument("--blocks", nargs="*", default=[], help="Only these block ids (rows unaffected)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
