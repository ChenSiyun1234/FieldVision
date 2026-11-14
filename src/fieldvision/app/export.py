"""Turn fields, time series and predictions into the small JSON files the demo loads.

    python -m fieldvision.app.export --fields fields.geojson --timeseries ts_dir_or.parquet \
        --predictions predictions.parquet --out app/data/at-2024 --title "Austria 2024"

Writes into --out:
    meta.json        region info, headline metrics, thresholds used for display
    fields.geojson   field outlines with prediction and label properties
    series.json      per-field NDVI / NDMI observations, each tagged usable / excluded / no clear pixels
    counties.json    (US only, with --counties) predicted vs USDA silage share per county
    counties.geojson (with --county-shapes) simplified county outlines for the map
and adds or updates the region's entry in <out>/../regions.json.

Predictions are optional: without them the region still shows fields and curves,
and the app says that no predictions exist yet. Metrics are computed here from the
declared labels and the predictions passed in, so pass out-of-fold (cross-validation)
predictions for labelled regions, otherwise the numbers are training-set numbers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

# Display-only thresholds for the plain-English evidence sentences in the app.
# They describe what the curves show; they are not the model's decision rule.
DEFAULT_CUES = {
    "green_ndvi": 0.6,   # at or above: "still green"
    "dry_ndvi": 0.45,    # below: "after dry-down"
    "wet_ndmi": 0.2,     # at or above: "still wet"
    "dry_ndmi": 0.05,    # below: "dry"
}

PRED_COLUMNS = [
    "field_id", "p_silage", "harvest_start", "harvest_end",
    "pre_ndvi", "pre_ndmi", "low_evidence",
]
PRED_OPTIONAL = ["low_evidence_reason", "pred_label"]

TS_COLUMNS = ["field_id", "date", "n_pixels", "n_clear", "clear_fraction", "B04", "B08", "B8A", "B11"]

# Observation quality codes stored in series.json ("q")
Q_NONE = 0       # acquisition exists but no clear pixel (bands NaN)
Q_EXCLUDED = 1   # some clear pixels, but clear_fraction below the threshold
Q_USABLE = 2


# ---------------------------------------------------------------- helpers

def _num(value, digits=None):
    """Python float or None; NaN and inf become None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits) if digits is not None else value


def _bool(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y", "t"):
            return True
        if v in ("false", "0", "no", "n", "f", ""):
            return False
        raise ValueError(f"cannot read {value!r} as a boolean")
    return bool(value)


def _date(value):
    """ISO date string or None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return pd.Timestamp(value).date().isoformat()


def _label(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    v = str(value).strip().lower()
    if v in ("silage", "grain"):
        return v
    if v in ("", "none", "null", "nan"):
        return None
    raise ValueError(f"unexpected label {value!r}; expected 'silage', 'grain' or null")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")


def _round_coords(coords, digits=6):
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        return [round(float(c), digits) for c in coords]
    return [_round_coords(c, digits) for c in coords]


# ---------------------------------------------------------------- loading

def load_fields(path, year=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"{path}: expected a GeoJSON FeatureCollection")
    features = []
    seen = set()
    for feat in data["features"]:
        props = dict(feat.get("properties") or {})
        fid = props.get("field_id")
        if fid is None:
            raise ValueError(f"{path}: a feature has no field_id")
        fid = str(fid)
        if fid in seen:
            raise ValueError(f"{path}: duplicate field_id {fid}")
        if year is not None and props.get("year") is not None and int(props["year"]) != int(year):
            continue
        seen.add(fid)
        props["field_id"] = fid
        features.append({"geometry": feat.get("geometry"), "properties": props})
    return features


def load_timeseries(path, field_ids=None) -> pd.DataFrame:
    path = Path(path)
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no parquet files under {path}")
    frames = []
    for f in files:
        t = pd.read_parquet(f)
        missing = [c for c in TS_COLUMNS if c not in t.columns]
        if missing:
            raise ValueError(f"{f}: time series is missing columns {missing}")
        t = t[TS_COLUMNS]
        t["field_id"] = t["field_id"].astype(str)
        if field_ids is not None:
            t = t[t["field_id"].isin(field_ids)]
        frames.append(t)
    ts = pd.concat(frames, ignore_index=True)
    ts["date"] = pd.to_datetime(ts["date"]).dt.tz_localize(None).dt.normalize()
    return ts


def load_predictions(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        p = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        p = pd.read_csv(path)
    else:
        raise ValueError(f"{path}: predictions must be .parquet or .csv")
    missing = [c for c in PRED_COLUMNS if c not in p.columns]
    if missing:
        raise ValueError(
            f"{path}: predictions are missing columns {missing}; "
            f"required: {PRED_COLUMNS} (values may be null), optional: {PRED_OPTIONAL}"
        )
    p = p.copy()
    p["field_id"] = p["field_id"].astype(str)
    if p["field_id"].duplicated().any():
        dup = p.loc[p["field_id"].duplicated(), "field_id"].iloc[0]
        raise ValueError(f"{path}: duplicate field_id {dup} in predictions")
    prob = pd.to_numeric(p["p_silage"], errors="coerce")
    bad = prob.notna() & ((prob < 0) | (prob > 1))
    if bad.any():
        raise ValueError(f"{path}: p_silage must be between 0 and 1")
    p["p_silage"] = prob
    for col in ("harvest_start", "harvest_end"):
        p[col] = pd.to_datetime(p[col], errors="coerce")
    for col in ("pre_ndvi", "pre_ndmi"):
        p[col] = pd.to_numeric(p[col], errors="coerce")
    return p.set_index("field_id")


# ---------------------------------------------------------------- series

def build_series(ts: pd.DataFrame, year: int, min_clear: float):
    """Per-field observation arrays plus simple coverage numbers.

    Same-day duplicates (overlapping tiles) keep the row with the most clear pixels.
    """
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    ts = ts.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        den = ts["B08"] + ts["B04"]
        ts["ndvi"] = np.where(np.abs(den) > 1e-12, (ts["B08"] - ts["B04"]) / den, np.nan)
        den = ts["B8A"] + ts["B11"]
        ts["ndmi"] = np.where(np.abs(den) > 1e-12, (ts["B8A"] - ts["B11"]) / den, np.nan)
    ts["_rank"] = ts["n_clear"].fillna(-1)
    ts = ts.sort_values(["field_id", "date", "_rank"]).drop_duplicates(["field_id", "date"], keep="last")

    series, coverage = {}, {}
    for fid, g in ts.groupby("field_id", sort=False):
        t, ndvi, ndmi, cf, q = [], [], [], [], []
        usable_days = []
        for row in g.itertuples(index=False):
            day = int((row.date - jan1).days)
            v, m, c = _num(row.ndvi, 3), _num(row.ndmi, 3), _num(row.clear_fraction, 2)
            if v is None and m is None:
                code = Q_NONE
            elif c is not None and c >= min_clear and v is not None:
                code = Q_USABLE
                usable_days.append(day)
            else:
                code = Q_EXCLUDED
            t.append(day)
            ndvi.append(v)
            ndmi.append(m)
            cf.append(c)
            q.append(code)
        series[fid] = {"t": t, "ndvi": ndvi, "ndmi": ndmi, "cf": cf, "q": q}
        gaps = np.diff(usable_days) if len(usable_days) > 1 else []
        coverage[fid] = {
            "n_obs": len(t),
            "n_usable": len(usable_days),
            "max_gap": int(max(gaps)) if len(gaps) else None,
        }
    return series, coverage


# ---------------------------------------------------------------- metrics

def confusion(y_true, y_pred):
    """Binary metrics with silage as the positive class. Undefined ratios are None."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == "silage" and b == "silage")
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == "grain" and b == "silage")
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == "grain" and b == "grain")
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == "silage" and b == "grain")

    def ratio(a, b):
        return round(a / b, 4) if b else None

    return {
        "n": tp + fp + tn + fn,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "fpr": ratio(fp, fp + tn),
        "accuracy": ratio(tp + tn, tp + fp + tn + fn),
    }


def region_metrics(props: list[dict]):
    n_fields = len(props)
    predicted = [p for p in props if p["pred"] is not None]
    labelled = [p for p in props if p["label"] is not None]
    both = [p for p in predicted if p["label"] is not None]
    confident = [p for p in both if p["low"] is not True]
    n_low = sum(1 for p in predicted if p["low"] is True)
    out = {
        "n_fields": n_fields,
        "n_predicted": len(predicted),
        "n_labelled": len(labelled),
        "n_low_evidence": n_low,
        "low_evidence_share": round(n_low / len(predicted), 4) if predicted else None,
        "predicted_silage_share": (
            round(sum(1 for p in predicted if p["pred"] == "silage") / len(predicted), 4) if predicted else None
        ),
        "all": confusion([p["label"] for p in both], [p["pred"] for p in both]) if both else None,
        "confident": (
            confusion([p["label"] for p in confident], [p["pred"] for p in confident]) if confident else None
        ),
    }
    return out


# ---------------------------------------------------------------- counties

def county_table(props, geoms, counties_csv, county_shapes=None):
    """Predicted silage share per county (area weighted) next to the USDA share."""
    usda = pd.read_csv(counties_csv, dtype={"county_fips": str})
    need = ["county_fips", "county_name", "state", "silage_acres", "grain_acres"]
    missing = [c for c in need if c not in usda.columns]
    if missing:
        raise ValueError(f"{counties_csv}: missing columns {missing}; expected {need}")
    usda["county_fips"] = usda["county_fips"].str.zfill(5)

    # field -> county: a county_fips property wins, else centroid in county shape
    shapes = None
    if county_shapes is not None:
        import geopandas as gpd

        shapes = gpd.read_file(county_shapes)
        key = next((c for c in ("county_fips", "GEOID", "geoid", "fips") if c in shapes.columns), None)
        if key is None:
            raise ValueError(f"{county_shapes}: needs a county_fips or GEOID column")
        shapes = shapes.rename(columns={key: "county_fips"})[["county_fips", "geometry"]]
        shapes["county_fips"] = shapes["county_fips"].astype(str).str.zfill(5)
        shapes = shapes.to_crs(4326)

    fips_of = {}
    for p in props:
        if p.get("county_fips"):
            fips_of[p["field_id"]] = str(p["county_fips"]).zfill(5)
    if shapes is not None:
        import geopandas as gpd
        from shapely.geometry import Point

        todo = [p for p in props if p["field_id"] not in fips_of]
        if todo:
            pts = gpd.GeoDataFrame(
                {"field_id": [p["field_id"] for p in todo]},
                geometry=[Point(geoms[p["field_id"]].representative_point()) for p in todo],
                crs=4326,
            )
            joined = gpd.sjoin(pts, shapes, how="left", predicate="within")
            for row in joined.itertuples():
                if isinstance(row.county_fips, str):
                    fips_of.setdefault(row.field_id, row.county_fips)

    rows = []
    for c in usda.itertuples(index=False):
        mine = [p for p in props if fips_of.get(p["field_id"]) == c.county_fips]
        pred = [p for p in mine if p["pred"] is not None]
        area_all = sum(p["area_ha"] or 0 for p in pred)
        area_sil = sum(p["area_ha"] or 0 for p in pred if p["pred"] == "silage")
        sil, grn = _num(c.silage_acres), _num(c.grain_acres)
        usda_share = round(sil / (sil + grn), 4) if sil is not None and grn is not None and sil + grn > 0 else None
        rows.append({
            "fips": c.county_fips,
            "name": str(c.county_name),
            "state": str(c.state),
            "n_fields": len(mine),
            "n_predicted": len(pred),
            "n_low_evidence": sum(1 for p in pred if p["low"] is True),
            "pred_silage_share": round(area_sil / area_all, 4) if area_all > 0 else None,
            "usda_silage_acres": sil,
            "usda_grain_acres": grn,
            "usda_silage_share": usda_share,
        })
    for p in props:
        p["county_fips"] = fips_of.get(p["field_id"])

    pairs = [(r["pred_silage_share"], r["usda_silage_share"]) for r in rows
             if r["pred_silage_share"] is not None and r["usda_silage_share"] is not None]
    summary = {"n_counties": len(rows), "n_compared": len(pairs), "mae": None, "r": None}
    if pairs:
        a = np.array(pairs)
        summary["mae"] = round(float(np.mean(np.abs(a[:, 0] - a[:, 1]))), 4)
        if len(pairs) >= 3 and a[:, 0].std() > 0 and a[:, 1].std() > 0:
            summary["r"] = round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 4)

    outlines = None
    if shapes is not None:
        keep = shapes[shapes["county_fips"].isin({r["fips"] for r in rows})].copy()
        keep["geometry"] = keep.geometry.simplify(0.001, preserve_topology=True)
        outlines = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"fips": r.county_fips},
                 "geometry": _round_coords_geom(r.geometry.__geo_interface__, 5)}
                for r in keep.itertuples()
            ],
        }
    return {"rows": rows, "summary": summary}, outlines


def _round_coords_geom(geom, digits):
    return {"type": geom["type"], "coordinates": _round_coords(geom["coordinates"], digits)}


# ---------------------------------------------------------------- main export

def export_region(
    fields_path,
    out_dir,
    timeseries_path=None,
    predictions_path=None,
    *,
    title=None,
    group=None,
    region_id=None,
    year=None,
    synthetic=False,
    min_clear=0.7,
    threshold=0.5,
    gap_days=20,
    max_fields=None,
    note=None,
    counties=None,
    county_shapes=None,
    seed=0,
):
    from shapely.geometry import mapping, shape

    out_dir = Path(out_dir)
    region_id = region_id or out_dir.name
    features = load_fields(fields_path, year)
    if not features:
        raise ValueError(f"{fields_path}: no fields{' for year ' + str(year) if year else ''}")

    years = sorted({int(f["properties"]["year"]) for f in features if f["properties"].get("year") is not None})
    if year is None:
        if len(years) != 1:
            raise ValueError(f"fields span years {years}; pass --year")
        year = years[0]
    countries = sorted({str(f["properties"].get("country")) for f in features if f["properties"].get("country")})
    field_ids = {f["properties"]["field_id"] for f in features}

    series, coverage = {}, {}
    if timeseries_path is not None:
        ts = load_timeseries(timeseries_path, field_ids)
        series, coverage = build_series(ts, year, min_clear)

    preds = None
    if predictions_path is not None:
        preds = load_predictions(predictions_path)
        unknown = set(preds.index) - field_ids
        if unknown:
            print(f"note: {len(unknown)} predictions have no matching field and are ignored", file=sys.stderr)

    geoms, props = {}, []
    for f in features:
        src = f["properties"]
        fid = src["field_id"]
        geom = shape(f["geometry"]) if f["geometry"] else None
        geoms[fid] = geom
        cov = coverage.get(fid, {"n_obs": 0, "n_usable": 0, "max_gap": None})
        p = {
            "field_id": fid,
            "label": _label(src.get("label")),
            "area_ha": _num(src.get("area_ha"), 2),
            "block_id": None if src.get("block_id") is None else str(src.get("block_id")),
            "county_fips": src.get("county_fips"),
            "n_obs": cov["n_obs"],
            "n_usable": cov["n_usable"],
            "max_gap": cov["max_gap"],
            "p": None, "pred": None, "low": None, "reason": None,
            "hs": None, "he": None, "pre_ndvi": None, "pre_ndmi": None, "correct": None,
        }
        if p["area_ha"] is None and geom is not None:
            p["area_ha"] = _num(_area_ha(geom), 2)
        if preds is not None and fid in preds.index:
            r = preds.loc[fid]
            prob = _num(r["p_silage"], 3)
            pred_label = None
            if "pred_label" in preds.columns and not pd.isna(r["pred_label"]):
                pred_label = _label(r["pred_label"])
            elif prob is not None:
                pred_label = "silage" if prob >= threshold else "grain"
            p.update({
                "p": prob,
                "pred": pred_label,
                "low": _bool(r["low_evidence"]),
                "reason": (None if "low_evidence_reason" not in preds.columns or pd.isna(r["low_evidence_reason"])
                           else str(r["low_evidence_reason"])),
                "hs": _date(r["harvest_start"]),
                "he": _date(r["harvest_end"]),
                "pre_ndvi": _num(r["pre_ndvi"], 3),
                "pre_ndmi": _num(r["pre_ndmi"], 3),
            })
            if p["label"] is not None and p["pred"] is not None:
                p["correct"] = p["label"] == p["pred"]
        props.append(p)

    metrics = region_metrics(props)

    counties_out = outlines = None
    if counties is not None:
        counties_out, outlines = county_table(props, geoms, counties, county_shapes)

    # thin out very large regions for the browser; metrics above use every field
    shown = props
    if max_fields is not None and len(props) > max_fields:
        rng = np.random.default_rng(seed)
        keep = set(rng.choice([p["field_id"] for p in props], size=max_fields, replace=False))
        shown = [p for p in props if p["field_id"] in keep]

    all_days = [d for fid in (p["field_id"] for p in shown) for d in series.get(fid, {}).get("t", [])]
    season = {"start": min(all_days), "end": max(all_days)} if all_days else None

    geo = {"type": "FeatureCollection", "features": []}
    for p in shown:
        g = geoms[p["field_id"]]
        if g is not None:
            g = g.simplify(0.00001, preserve_topology=True)
            c = g.representative_point()
            p["lon"], p["lat"] = round(c.x, 6), round(c.y, 6)
            gj = mapping(g)
            gj = {"type": gj["type"], "coordinates": _round_coords(gj["coordinates"], 6)}
        else:
            p["lon"] = p["lat"] = None
            gj = None
        geo["features"].append({"type": "Feature", "properties": {k: v for k, v in p.items()}, "geometry": gj})

    meta = {
        "schema": SCHEMA_VERSION,
        "region_id": region_id,
        "title": title or region_id,
        "group": group or title or region_id,
        "year": year,
        "countries": countries,
        "synthetic": bool(synthetic),
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": {
            "fields": Path(fields_path).name,
            "timeseries": None if timeseries_path is None else Path(timeseries_path).name,
            "predictions": None if predictions_path is None else Path(predictions_path).name,
        },
        "has_series": bool(series),
        "has_predictions": preds is not None,
        "has_labels": metrics["n_labelled"] > 0,
        "has_counties": counties_out is not None,
        "has_county_shapes": outlines is not None,
        "n_fields_shown": len(shown),
        "min_clear": min_clear,
        "threshold": threshold,
        "gap_days": gap_days,
        "season": season,
        "cues": DEFAULT_CUES,
        "metrics": metrics,
        "note": note,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    shown_ids = {p["field_id"] for p in shown}
    write_json(out_dir / "meta.json", meta)
    write_json(out_dir / "fields.geojson", geo)
    write_json(out_dir / "series.json", {k: v for k, v in series.items() if k in shown_ids})
    for name, obj in (("counties.json", counties_out), ("counties.geojson", outlines)):
        if obj is not None:
            write_json(out_dir / name, obj)
        elif (out_dir / name).exists():
            (out_dir / name).unlink()
    update_index(out_dir.parent / "regions.json", out_dir.name, meta)
    return meta


def _area_ha(geom):
    """Rough area for lon/lat polygons (local equirectangular), only used when area_ha is missing."""
    import shapely

    lat0 = math.radians(geom.representative_point().y)
    kx, ky = 111320.0 * math.cos(lat0), 110540.0
    return shapely.transform(geom, lambda xy: xy * np.array([kx, ky])).area / 1e4


def update_index(index_path: Path, folder: str, meta: dict) -> None:
    index = {"regions": []}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = {
        "id": meta["region_id"],
        "path": folder,
        "title": meta["title"],
        "group": meta["group"],
        "year": meta["year"],
        "synthetic": meta["synthetic"],
        "has_counties": meta["has_counties"],
    }
    regions = index.get("regions", [])
    for i, r in enumerate(regions):
        if r.get("id") == entry["id"] or r.get("path") == folder:
            regions[i] = entry
            break
    else:
        regions.append(entry)
    index["regions"] = regions
    write_json(index_path, index)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fields", required=True, help="GeoJSON FeatureCollection (EPSG:4326)")
    ap.add_argument("--timeseries", help="Parquet file or directory of Parquet files (long format)")
    ap.add_argument("--predictions", help="Per-field predictions, .parquet or .csv (see app/README.md)")
    ap.add_argument("--out", required=True, help="Output folder, e.g. app/data/at-2024")
    ap.add_argument("--title", help='Region name shown in the app, e.g. "Austria 2024"')
    ap.add_argument("--group", help='Region group for the selector, e.g. "BARC Maryland"; defaults to the title')
    ap.add_argument("--region-id", help="Defaults to the output folder name")
    ap.add_argument("--year", type=int, help="Keep only fields of this year (needed if fields span years)")
    ap.add_argument("--min-clear", type=float, default=0.7,
                    help="clear_fraction at or above which an observation counts as usable (default 0.7)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="p_silage cut-off when predictions have no pred_label column (default 0.5)")
    ap.add_argument("--gap-days", type=int, default=20,
                    help="Curves are not joined across gaps longer than this many days (default 20)")
    ap.add_argument("--max-fields", type=int, help="Show a random subset in the app; metrics still use all fields")
    ap.add_argument("--note", help='Short note under the metrics, e.g. "spatial 5-fold CV, out-of-fold"')
    ap.add_argument("--counties", help="USDA county table CSV: county_fips,county_name,state,silage_acres,grain_acres")
    ap.add_argument("--county-shapes", help="County polygons (GeoJSON/GPKG with county_fips or GEOID)")
    ap.add_argument("--synthetic", action="store_true", help="Mark the region as synthetic sample data")
    args = ap.parse_args(argv)

    meta = export_region(
        args.fields, args.out, args.timeseries, args.predictions,
        title=args.title, group=args.group, region_id=args.region_id, year=args.year,
        synthetic=args.synthetic, min_clear=args.min_clear, threshold=args.threshold,
        gap_days=args.gap_days, max_fields=args.max_fields, note=args.note,
        counties=args.counties, county_shapes=args.county_shapes,
    )
    m = meta["metrics"]
    print(f"{meta['region_id']}: {m['n_fields']} fields, {m['n_predicted']} predicted, "
          f"{m['n_labelled']} labelled -> {args.out}")


if __name__ == "__main__":
    main()
