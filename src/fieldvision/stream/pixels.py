"""Per-date field chips using the frozen Sentinel-2 reading and aggregation code."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
import json
import logging
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd
from affine import Affine
from shapely.geometry import shape

from fieldvision.s2 import extract
from fieldvision.s2.catalog import LEGACY, PLANETARY, PRIMARY, SOURCES, find_scenes
from fieldvision.s2.core import BANDS, interior_geometry, polygon_mask, resolve_duplicates
from fieldvision.s2.geometry import load_fields, make_blocks, scene_grid

LOG = logging.getLogger(__name__)
SCALE, MISSING = 10000, 65535
KEYS = ("field_id", "date", "scene_id", "bands", "scl", "inside")
read_band = extract.read_band


def encode_bands(bands):
    bands = np.asarray(bands)
    result = np.full(bands.shape, MISSING, dtype=np.uint16)
    valid = np.isfinite(bands) & (bands >= 0) & (bands <= 1)
    result[valid] = np.rint(bands[valid] * SCALE).astype(np.uint16)
    return result


def decode_bands(bands):
    result = np.asarray(bands).astype(np.float32) / SCALE
    result[np.asarray(bands) == MISSING] = np.nan
    return result


def chip_window(geometry, affine, dimensions, chip_size):
    """Return the unbounded window origin, chip transform and selected interior."""
    if chip_size < 1:
        raise ValueError("Chip size must be positive")
    selected = interior_geometry(geometry, affine, dimensions)[0]
    center = selected.centroid
    if not selected.contains(center):
        center = selected.representative_point()
    column = int(np.floor((center.x - affine.c) / 10)) - chip_size // 2
    row = int(np.floor((affine.f - center.y) / 10)) - chip_size // 2
    return row, column, affine * Affine.translation(column, row), selected


def cut_chip(geometry, grid, arrays, chip_size=12):
    """Cut calibrated arrays after aggregate_scene has applied its range mask."""
    _, affine, dimensions = grid[:3]
    row, column, transform, selected = chip_window(geometry, affine, dimensions, chip_size)
    bands = np.full((len(BANDS), chip_size, chip_size), MISSING, dtype=np.uint16)
    scl = np.zeros((chip_size, chip_size), dtype=np.uint8)
    r0, c0 = max(0, row), max(0, column)
    r1, c1 = min(dimensions[0], row + chip_size), min(dimensions[1], column + chip_size)
    if r1 > r0 and c1 > c0:
        source = np.s_[r0:r1, c0:c1]
        target = np.s_[r0 - row:r1 - row, c0 - column:c1 - column]
        for i, band in enumerate(BANDS):
            bands[i][target] = encode_bands(arrays[band][source])
        scl[target] = arrays["SCL"][source].astype(np.uint8)
    return {"bands": bands, "scl": scl,
            "inside": polygon_mask(selected, transform, (chip_size, chip_size))}


def select_chips(table, chips, min_clear_fraction=0.6, chip_size=12):
    table = resolve_duplicates(table)
    usable = table[(table.n_clear >= 1) & (table.clear_fraction >= min_clear_fraction)]
    keys = [tuple(str(value) for value in row) for row in
            usable[["field_id", "date", "scene_id"]].itertuples(index=False, name=None)]
    keys = [key for key in keys if key in chips]
    result = {name: np.asarray([key[i] for key in keys], dtype=str)
              for i, name in enumerate(KEYS[:3])}
    for name, dims, dtype in (("bands", (len(BANDS), chip_size, chip_size), np.uint16),
                              ("scl", (chip_size, chip_size), np.uint8),
                              ("inside", (chip_size, chip_size), bool)):
        result[name] = np.stack([chips[key][name] for key in keys]) if keys else np.empty((0, *dims), dtype=dtype)
    return result


def load_chips(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def write_chips(path, arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    extract.replace_with_retry(temporary, path)


def process_block(block, args, pool):
    path = args.out / "blocks" / (block["id"] + ".parquet")
    status_path, chip_path = path.with_suffix(".json"), path.with_suffix(".npz")
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous.get("status") == "complete" and path.exists() and chip_path.exists():
        return previous
    started = time.monotonic()
    # Invalidate an old checkpoint before rebuilding a partial or interrupted block.
    extract.atomic_json(status_path, {"status": "running"})
    cache = args.out / "catalog" / (block["id"] + ".json")
    if cache.exists():
        cached = json.loads(cache.read_text())
        items, counts = cached["items"], cached["counts"]
    else:
        items, counts = find_scenes(block["bbox"], args.start_date, args.end_date, args.max_cloud,
                                   args.collection, None if args.no_fallback else args.fallback_source)
        items = [item for item in items if any(shape(item["geometry"]).intersects(f["geometry"])
                                               for f in block["fields"])]
        extract.atomic_json(cache, {"items": items, "counts": counts})
    skipped = {}
    for item in items:
        missing = [name for name in (*BANDS, "SCL") if not extract._has_asset(item, name)]
        if missing:
            skipped[item["id"]] = "missing assets: " + ",".join(missing)
    items = [item for item in items if item["id"] not in skipped]
    frames, chips, errors, diagnostics = [], {}, {}, {}
    for offset in range(0, len(items), args.scene_batch):
        batch = []
        for item in items[offset:offset + args.scene_batch]:
            try:
                grid = scene_grid(block, item)
                futures = {name: pool.submit(read_band, item, name, grid) for name in (*BANDS, "SCL")}
                batch.append((item, grid, futures))
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
                # the frozen extractor has no bright-row guard; pass it only when asked for
                extra = () if args.max_blue is None else (args.max_blue,)
                rows, diagnostic = extract.aggregate_scene(block, item, grid, arrays, *extra)
                scene_chips = {}
                for i, row in enumerate(rows.itertuples(index=False)):
                    if row.n_clear >= 1 and row.clear_fraction >= args.min_clear_fraction:
                        key = (str(row.field_id), str(row.date), str(row.scene_id))
                        scene_chips[key] = cut_chip(grid[3][i], grid, arrays, args.chip)
                frames.append(rows)
                chips.update(scene_chips)
                diagnostics[item["id"]] = diagnostic
            except Exception as exc:
                errors[item["id"]] = f"{type(exc).__name__}: {exc}"
                LOG.warning("Scene %s failed: %s", item["id"], exc)
            del arrays
            futures.clear()
        batch.clear()
        LOG.info("Block %s: processed %d/%d scenes (%d failed)", block["id"],
                 min(offset + args.scene_batch, len(items)), len(items), len(errors))
    table = resolve_duplicates(pd.concat(frames, ignore_index=True)) if frames else extract.empty_table()
    selected = select_chips(table, chips, args.min_clear_fraction, args.chip)
    extract.write_parquet(path, table)
    write_chips(chip_path, selected)
    status = {"status": "partial" if errors else "complete", "field_count": len(block["fields"]),
              "field_ids": [f["field_id"] for f in block["fields"]], "scenes_found": len(items), **counts,
              "scenes_succeeded": len(diagnostics), "scene_errors": errors, "skipped_items": skipped,
              "diagnostics": diagnostics, "rows": len(table), "n_obs": len(selected["field_id"]),
              "wall_seconds": time.monotonic() - started}
    extract.atomic_json(status_path, status)
    return status


def merge_chips(paths, out, chip_size):
    """Merge one block at a time into disk-backed arrays in field/date order."""
    metadata = {key: [] for key in KEYS[:3]}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for key in metadata:
                metadata[key].append(data[key])
    merged = {key: np.concatenate(values) if values else np.array([], dtype=str)
              for key, values in metadata.items()}
    order = np.lexsort((merged["date"], merged["field_id"]))
    positions = np.empty(len(order), dtype=np.int64)
    positions[order] = np.arange(len(order))
    merged = {key: values[order] for key, values in merged.items()}
    counts = Counter(merged["field_id"].tolist())
    with TemporaryDirectory(prefix="merge_", dir=out) as temporary:
        for key, dims, dtype in (("bands", (len(BANDS), chip_size, chip_size), np.uint16),
                                  ("scl", (chip_size, chip_size), np.uint8),
                                  ("inside", (chip_size, chip_size), bool)):
            dimensions = (len(order), *dims)
            merged[key] = (np.memmap(Path(temporary) / key, mode="w+", dtype=dtype, shape=dimensions)
                           if len(order) else np.empty(dimensions, dtype=dtype))
        offset = 0
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                count = len(data["field_id"])
                for key in KEYS[3:]:
                    merged[key][positions[offset:offset + count]] = data[key]
                offset += count
        merged.update(band_names=np.asarray(BANDS), chip_size=np.int64(chip_size),
                      scale=np.int64(SCALE), missing=np.int64(MISSING))
        try:
            write_chips(out / "chips.npz", merged)
        finally:
            for key in KEYS[3:]:
                array = merged.pop(key)
                if isinstance(array, np.memmap):
                    array._mmap.close()
    return counts


def run(args):
    started = time.monotonic()
    args.start_date = date.fromisoformat(f"{args.year}-{args.start}").isoformat()
    args.end_date = date.fromisoformat(f"{args.year}-{args.end}").isoformat()
    if args.start_date > args.end_date:
        raise ValueError("Start must precede end within the selected year")
    if args.workers < 1 or args.scene_batch < 1 or not 0 < args.block_size <= 6000 or args.chip < 1:
        raise ValueError("Workers, batch and chip must be positive; block size must be in (0, 6000]")
    if not 0 <= args.max_cloud <= 100 or not 0 <= args.min_clear_fraction <= 1:
        raise ValueError("Invalid cloud/clear thresholds")
    if args.max_blue is not None and not 0 <= args.max_blue <= 1:
        raise ValueError("Max blue must be in [0, 1]")
    fields = load_fields(args.fields, args.year)
    parameters = {k: v for k, v in vars(args).items() if k not in ("fields", "out", "workers", "scene_batch")}
    s2_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(extract.__file__).parent.glob("*.py")))).hexdigest()
    pixels_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    signature = hashlib.sha256(args.fields.read_bytes() + json.dumps(parameters, sort_keys=True).encode()
                               + s2_hash.encode() + pixels_hash.encode()).hexdigest()
    manifest_path = args.out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous.get("signature") != signature:
        raise ValueError("Output belongs to different inputs/options/code; use a new --out directory")
    args.out.mkdir(parents=True, exist_ok=True)
    for subdir in ("blocks", "catalog"):
        (args.out / subdir).mkdir(exist_ok=True)
    blocks = make_blocks(fields, args.block_size)
    manifest = {"status": "running", "signature": signature, "parameters": parameters,
                "s2_source_sha256": s2_hash, "pixels_sha256": pixels_hash,
                "fields": len(fields), "input_fields": len(fields), "block_count": len(blocks), "blocks": {},
                "invalid_fields": {f["field_id"]: f["error"] for f in fields if f.get("error")}}
    extract.atomic_json(manifest_path, manifest)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for block in blocks:
            try:
                status = process_block(block, args, pool)
            except Exception as exc:
                LOG.exception("Block %s failed", block["id"])
                status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                extract.atomic_json(args.out / "blocks" / (block["id"] + ".json"), status)
            manifest["blocks"][block["id"]] = status
            extract.atomic_json(manifest_path, manifest)
    valid = [b for b in blocks if manifest["blocks"][b["id"]]["status"] in ("complete", "partial")]
    paths = [args.out / "blocks" / (b["id"] + ".parquet") for b in valid]
    tables = [pd.read_parquet(path) for path in paths]
    table = resolve_duplicates(pd.concat(tables, ignore_index=True)) if tables else extract.empty_table()
    extract.write_parquet(args.out / "timeseries.parquet", table)
    counts = merge_chips([p.with_suffix(".npz") for p in paths], args.out, args.chip)
    failures = Counter(error.split(":", 1)[0] for status in manifest["blocks"].values()
                       for error in status.get("scene_errors", {}).values())
    incomplete = bool(manifest["invalid_fields"]) or any(s["status"] != "complete" for s in manifest["blocks"].values())
    manifest.update(status="partial" if incomplete else "complete", rows=len(table), fields_with_chips=len(counts),
                    n_obs=sum(counts.values()), median_chips_per_field=float(np.median([counts.get(f["field_id"], 0)
                                                                                    for f in fields])),
                    scene_failures=sum(failures.values()), scene_failures_by_kind=dict(failures),
                    wall_seconds=time.monotonic() - started)
    extract.atomic_json(manifest_path, manifest)
    LOG.info("%s: %d chips, %d/%d fields, %.1f seconds", manifest["status"], manifest["n_obs"],
             len(counts), len(fields), manifest["wall_seconds"])
    return 2 if incomplete else 0


def check_against(new_table, reference_table, min_clear_fraction=0.6, tolerance=1e-6):
    """Compare usable common field/dates; empty comparisons never pass."""
    tables = []
    for table in (new_table, reference_table):
        table = pd.read_parquet(table) if isinstance(table, (str, Path)) else table.copy()
        table["field_id"] = table.field_id.astype(str)
        table["date"] = pd.to_datetime(table.date).dt.strftime("%Y-%m-%d")
        tables.append(resolve_duplicates(table))
    new, reference = tables
    new_fields, reference_fields = set(new.field_id), set(reference.field_id)
    usable = [t[(t.n_clear >= 1) & (t.clear_fraction >= min_clear_fraction)] for t in tables]
    common = usable[0].merge(usable[1], on=["field_id", "date"], suffixes=("_new", "_reference"))
    agree = common.n_clear_new.to_numpy() == common.n_clear_reference.to_numpy()
    for band in BANDS:
        agree &= np.isclose(common[band + "_new"], common[band + "_reference"], atol=tolerance, rtol=0, equal_nan=True)
    return {"rows_compared": len(common), "same_scene_id": int((common.scene_id_new == common.scene_id_reference).sum()),
            "rows_agree": int(agree.sum()), "fields_compared": int(common.field_id.nunique()),
            "fields_only_new": len(new_fields - reference_fields), "fields_only_reference": len(reference_fields - new_fields),
            "fields_only_one": len(new_fields ^ reference_fields),
            "agreement_fraction": float(agree.mean()) if len(common) else 0.0}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--fields", type=Path, required=True)
    result.add_argument("--year", type=int, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--start", default="04-01")
    result.add_argument("--end", default="11-30")
    result.add_argument("--block-size", type=float, default=6000)
    result.add_argument("--max-cloud", type=float, default=80)
    result.add_argument("--collection", choices=SOURCES, default=PRIMARY)
    result.add_argument("--fallback-source", choices=(LEGACY, PLANETARY), default=PLANETARY)
    result.add_argument("--no-fallback", action="store_true")
    result.add_argument("--max-blue", type=float, default=None)
    result.add_argument("--workers", type=int, default=16)
    result.add_argument("--scene-batch", type=int, default=3)
    result.add_argument("--chip", type=int, default=12)
    result.add_argument("--min-clear-fraction", type=float, default=0.6)
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if argv and argv[0] == "check":
            check = argparse.ArgumentParser(description=check_against.__doc__)
            check.add_argument("--run", type=Path, required=True)
            check.add_argument("--reference", type=Path, required=True)
            check.add_argument("--min-agree", type=float, default=0.99)
            args = check.parse_args(argv[1:])
            if not 0 <= args.min_agree <= 1:
                raise ValueError("Min agree must be in [0, 1]")
            counts = check_against(args.run / "timeseries.parquet", args.reference)
            print(json.dumps(counts, indent=2))
            return int(counts["rows_compared"] == 0 or counts["agreement_fraction"] < args.min_agree)
        return run(parser().parse_args(argv))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
