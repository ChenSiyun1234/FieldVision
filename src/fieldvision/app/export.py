"""Turn fields, time series and predictions into the small JSON files the demo loads.

    python -m fieldvision.app.export --fields fields.geojson --timeseries ts_dir_or.parquet \
        --predictions predictions.parquet --asof predictions_by_cutoff.parquet \
        --out app/data/at-2024 --title "Austria 2024"

Writes into --out:
    meta.json        region info, headline metrics, thresholds used for display
    fields.geojson   field outlines with prediction, label and evidence properties
    series.json      per-field NDVI / NDMI / NDTI observations, each tagged usable / excluded / no clear pixels
    asof.json        (with --asof) per-field probability at each cutoff date, metrics per cutoff
    counties.json    (US only, with --counties) predicted vs USDA acres and shares per county and state
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

SCHEMA_VERSION = 2
ACRES_PER_HA = 2.47105

# Display-only thresholds for the plain-English evidence sentences in the app.
# They describe what the curves show; they are not the model's decision rule.
DEFAULT_CUES = {
    "green_ndvi": 0.6,     # at or above: "still green"
    "dry_ndvi": 0.45,      # below: "after dry-down"
    "wet_ndmi": 0.2,       # at or above: "still wet"
    "dry_ndmi": 0.05,      # below: "dry"
    "residue_ndti": 0.10,  # at or above after the cut: stalks left on the ground; below: bare soil
}

PRED_COLUMNS = [
    "field_id", "p_silage", "harvest_start", "harvest_end",
    "pre_ndvi", "pre_ndmi", "low_evidence",
]
PRED_OPTIONAL = ["low_evidence_reason", "pred_label", "explanation", "flip_reason", "post_ndvi", "post_ndti"]
ASOF_COLUMNS = ["field_id", "cutoff_date", "p_silage", "low_evidence"]

TS_COLUMNS = ["field_id", "date", "n_pixels", "n_clear", "clear_fraction", "B04", "B08", "B8A", "B11"]
TS_OPTIONAL = ["B12"]

# Observation quality codes stored in series.json ("q")
Q_NONE = 0       # acquisition exists but no clear pixel (bands NaN)
Q_EXCLUDED = 1   # some clear pixels, but clear_fraction below the threshold
Q_USABLE = 2

POST_WINDOW_DAYS = 30   # residue is read from clear views up to this long after the cut


# ---------------------------------------------------------------- helpers

def _isnull(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _num(value, digits=None):
    """Python float or None; NaN and inf become None."""
    if _isnull(value):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits) if digits is not None else value


def _bool(value):
    if _isnull(value):
        return None
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
    if _isnull(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _text(value):
    if _isnull(value):
        return None
    s = str(value).strip()
    return s or None


def _label(value):
    if _isnull(value):
        return None
    v = str(value).strip().lower()
    if v in ("silage", "grain"):
        return v
    if v in ("", "none", "null", "nan"):
        return None
    raise ValueError(f"unexpected label {value!r}; expected 'silage', 'grain' or null")


def _day(iso, year):
    """Day offset from Jan 1 of `year` for an ISO date string (None stays None)."""
    if iso is None:
        return None
    return int((pd.Timestamp(iso) - pd.Timestamp(year=year, month=1, day=1)).days)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")


def _round_coords(coords, digits=6):
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        return [round(float(c), digits) for c in coords]
    return [_round_coords(c, digits) for c in coords]


def _read_table(path):
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"{path}: expected .parquet or .csv")


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
        t = t[TS_COLUMNS + [c for c in TS_OPTIONAL if c in t.columns]].copy()
        t["field_id"] = t["field_id"].astype(str)
        if field_ids is not None:
            t = t[t["field_id"].isin(field_ids)]
        frames.append(t)
    ts = pd.concat(frames, ignore_index=True)
    for c in TS_OPTIONAL:
        if c not in ts.columns:
            ts[c] = np.nan
    ts["date"] = pd.to_datetime(ts["date"]).dt.tz_localize(None).dt.normalize()
    return ts


def load_predictions(path) -> pd.DataFrame:
    p = _read_table(path)
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
    p["p_silage"] = _check_prob(p["p_silage"], path)
    for col in ("harvest_start", "harvest_end"):
        p[col] = pd.to_datetime(p[col], errors="coerce")
    for col in ("pre_ndvi", "pre_ndmi", "post_ndvi", "post_ndti"):
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce")
    return p.set_index("field_id")


def load_asof(path) -> pd.DataFrame:
    a = _read_table(path)
    missing = [c for c in ASOF_COLUMNS if c not in a.columns]
    if missing:
        raise ValueError(f"{path}: as-of predictions are missing columns {missing}; required: {ASOF_COLUMNS}")
    a = a.copy()
    a["field_id"] = a["field_id"].astype(str)
    a["cutoff_date"] = pd.to_datetime(a["cutoff_date"]).dt.date.astype(str)
    a["p_silage"] = _check_prob(a["p_silage"], path)
    if a.duplicated(["field_id", "cutoff_date"]).any():
        raise ValueError(f"{path}: more than one row for a field and cutoff date")
    return a


def _check_prob(col, path):
    prob = pd.to_numeric(col, errors="coerce")
    if (prob.notna() & ((prob < 0) | (prob > 1))).any():
        raise ValueError(f"{path}: p_silage must be between 0 and 1")
    return prob


# ---------------------------------------------------------------- series

def build_series(ts: pd.DataFrame, year: int, min_clear: float):
    """Per-field observation arrays plus simple coverage numbers.

    Same-day duplicates (overlapping tiles) keep the row with the most clear pixels.
    NDTI = (B11 - B12) / (B11 + B12) is a crop residue index, read after the cut.
    """
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    ts = ts.copy()
    if "B12" not in ts.columns:
        ts["B12"] = np.nan

    def ratio(a, b):
        den = a + b
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(den) > 1e-12, (a - b) / den, np.nan)

    ts["ndvi"] = ratio(ts["B08"], ts["B04"])
    ts["ndmi"] = ratio(ts["B8A"], ts["B11"])
    ts["ndti"] = ratio(ts["B11"], ts["B12"])
    ts["_rank"] = ts["n_clear"].fillna(-1)
    ts = ts.sort_values(["field_id", "date", "_rank"]).drop_duplicates(["field_id", "date"], keep="last")

    series, coverage = {}, {}
    for fid, g in ts.groupby("field_id", sort=False):
        t, ndvi, ndmi, ndti, cf, q = [], [], [], [], [], []
        usable = []
        for row in g.itertuples(index=False):
            day = int((row.date - jan1).days)
            v, m, r, c = _num(row.ndvi, 3), _num(row.ndmi, 3), _num(row.ndti, 3), _num(row.clear_fraction, 2)
            if v is None and m is None:
                code = Q_NONE
            elif c is not None and c >= min_clear and v is not None:
                code = Q_USABLE
                usable.append((day, v))
            else:
                code = Q_EXCLUDED
            t.append(day)
            ndvi.append(v)
            ndmi.append(m)
            ndti.append(r)
            cf.append(c)
            q.append(code)
        series[fid] = {"t": t, "ndvi": ndvi, "ndmi": ndmi, "ndti": ndti, "cf": cf, "q": q}
        days = [d for d, _ in usable]
        gaps = np.diff(days) if len(days) > 1 else []
        peak = max(usable, key=lambda x: x[1]) if usable else None
        coverage[fid] = {
            "n_obs": len(t),
            "n_usable": len(usable),
            "max_gap": int(max(gaps)) if len(gaps) else None,
            "peak_ndvi": peak[1] if peak else None,
            "peak_day": peak[0] if peak else None,
        }
    return series, coverage


def post_harvest(s, he_day, window=POST_WINDOW_DAYS):
    """NDVI at the first clear view after the cut, and median NDTI of the first
    (up to three) clear views within `window` days. Nones when nothing usable."""
    if s is None or he_day is None:
        return None, None
    idx = [i for i, t in enumerate(s["t"]) if s["q"][i] == Q_USABLE and he_day <= t <= he_day + window]
    if not idx:
        return None, None
    post_ndvi = s["ndvi"][idx[0]]
    vals = [s["ndti"][i] for i in idx[:3] if s["ndti"][i] is not None]
    post_ndti = round(float(np.median(vals)), 3) if vals else None
    return post_ndvi, post_ndti


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
    predicted = [p for p in props if p["pred"] is not None]
    labelled = [p for p in props if p["label"] is not None]
    both = [p for p in predicted if p["label"] is not None]
    confident = [p for p in both if p["low"] is not True]
    n_low = sum(1 for p in predicted if p["low"] is True)
    return {
        "n_fields": len(props),
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


def harvest_reference(props, year):
    """Median harvest-window midpoint of confident silage and grain calls (day offsets).

    The app uses it to say "cut N days before the typical grain field here"."""
    mids = {"silage": [], "grain": []}
    for p in props:
        if p["pred"] in mids and p["low"] is not True and p["hs"] and p["he"]:
            mids[p["pred"]].append((_day(p["hs"], year) + _day(p["he"], year)) / 2)
    if len(mids["silage"]) < 3 or len(mids["grain"]) < 3:
        return None
    return {
        "silage_mid": int(round(float(np.median(mids["silage"])))),
        "grain_mid": int(round(float(np.median(mids["grain"])))),
        "n_silage": len(mids["silage"]),
        "n_grain": len(mids["grain"]),
    }


# ---------------------------------------------------------------- as-of (early-season) predictions

def asof_tables(asof: pd.DataFrame, props: list[dict], shown_ids: set, threshold: float):
    """Per-cutoff probabilities for the shown fields and metrics per cutoff over all fields."""
    by_id = {p["field_id"]: p for p in props}
    asof = asof[asof["field_id"].isin(by_id)]
    cutoffs = sorted(asof["cutoff_date"].unique())
    fields = {fid: {"p": [None] * len(cutoffs), "low": [None] * len(cutoffs)} for fid in shown_ids}
    metrics = []
    has_label_col = "pred_label" in asof.columns
    for k, cut in enumerate(cutoffs):
        rows = asof[asof["cutoff_date"] == cut].set_index("field_id")
        at_cut = []
        for fid, base in by_id.items():
            prob = low = pred = None
            if fid in rows.index:
                r = rows.loc[fid]
                prob = _num(r["p_silage"], 3)
                low = _bool(r["low_evidence"])
                if has_label_col and not _isnull(r["pred_label"]):
                    pred = _label(r["pred_label"])
                elif prob is not None:
                    pred = "silage" if prob >= threshold else "grain"
            at_cut.append({"field_id": fid, "label": base["label"], "pred": pred, "low": low})
            if fid in fields:
                fields[fid]["p"][k] = prob
                fields[fid]["low"][k] = low
        m = region_metrics(at_cut)
        m["cutoff"] = cut
        metrics.append(m)
    return {"cutoffs": cutoffs, "metrics": metrics, "fields": fields}


# ---------------------------------------------------------------- counties

def county_table(props, geoms, counties_csv, county_shapes=None):
    """Predicted silage and grain acres per county (and state) next to USDA's reported acres."""
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

        todo = [p for p in props if p["field_id"] not in fips_of and geoms.get(p["field_id"]) is not None]
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

    def acres(fields):
        return round(sum((p["area_ha"] or 0) for p in fields) * ACRES_PER_HA, 1)

    rows = []
    for c in usda.itertuples(index=False):
        mine = [p for p in props if fips_of.get(p["field_id"]) == c.county_fips]
        pred = [p for p in mine if p["pred"] is not None]
        sil = [p for p in pred if p["pred"] == "silage"]
        grn = [p for p in pred if p["pred"] == "grain"]
        area_all = acres(pred)
        usda_sil, usda_grn = _num(c.silage_acres), _num(c.grain_acres)
        rows.append({
            "fips": c.county_fips,
            "name": str(c.county_name),
            "state": str(c.state),
            "n_fields": len(mine),
            "n_predicted": len(pred),
            "n_low_evidence": sum(1 for p in pred if p["low"] is True),
            "mapped_acres": acres(mine),
            "pred_silage_acres": acres(sil),
            "pred_grain_acres": acres(grn),
            "pred_low_acres": acres([p for p in pred if p["low"] is True]),
            "pred_silage_share": round(acres(sil) / area_all, 4) if area_all > 0 else None,
            "usda_silage_acres": usda_sil,
            "usda_grain_acres": usda_grn,
            "usda_silage_share": (
                round(usda_sil / (usda_sil + usda_grn), 4)
                if usda_sil is not None and usda_grn is not None and usda_sil + usda_grn > 0 else None
            ),
        })
    for p in props:
        p["county_fips"] = fips_of.get(p["field_id"])

    states = []
    for st in sorted({r["state"] for r in rows}):
        rs = [r for r in rows if r["state"] == st]

        def total(key):
            vals = [r[key] for r in rs if r[key] is not None]
            return round(sum(vals), 1) if len(vals) == len(rs) else None

        states.append({
            "state": st,
            "n_counties": len(rs),
            "n_predicted": sum(r["n_predicted"] for r in rs),
            "mapped_acres": total("mapped_acres"),
            "pred_silage_acres": total("pred_silage_acres"),
            "pred_grain_acres": total("pred_grain_acres"),
            "pred_low_acres": total("pred_low_acres"),
            "usda_silage_acres": total("usda_silage_acres"),
            "usda_grain_acres": total("usda_grain_acres"),
        })

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
    return {"rows": rows, "states": states, "summary": summary}, outlines


def _round_coords_geom(geom, digits):
    return {"type": geom["type"], "coordinates": _round_coords(geom["coordinates"], digits)}


# ---------------------------------------------------------------- main export

def export_region(
    fields_path,
    out_dir,
    timeseries_path=None,
    predictions_path=None,
    *,
    asof_path=None,
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

    empty_cov = {"n_obs": 0, "n_usable": 0, "max_gap": None, "peak_ndvi": None, "peak_day": None}
    geoms, props = {}, []
    for f in features:
        src = f["properties"]
        fid = src["field_id"]
        geom = shape(f["geometry"]) if f["geometry"] else None
        geoms[fid] = geom
        cov = coverage.get(fid, empty_cov)
        p = {
            "field_id": fid,
            "label": _label(src.get("label")),
            "area_ha": _num(src.get("area_ha"), 2),
            "block_id": None if src.get("block_id") is None else str(src.get("block_id")),
            "county_fips": src.get("county_fips"),
            **cov,
            "p": None, "pred": None, "low": None, "reason": None,
            "hs": None, "he": None, "pre_ndvi": None, "pre_ndmi": None,
            "post_ndvi": None, "post_ndti": None, "explanation": None, "flip": None,
            "correct": None,
        }
        if p["area_ha"] is None and geom is not None:
            p["area_ha"] = _num(_area_ha(geom), 2)
        if preds is not None and fid in preds.index:
            r = preds.loc[fid]
            prob = _num(r["p_silage"], 3)
            pred_label = None
            if "pred_label" in preds.columns and not _isnull(r["pred_label"]):
                pred_label = _label(r["pred_label"])
            elif prob is not None:
                pred_label = "silage" if prob >= threshold else "grain"
            p.update({
                "p": prob,
                "pred": pred_label,
                "low": _bool(r["low_evidence"]),
                "reason": _text(r["low_evidence_reason"]) if "low_evidence_reason" in preds.columns else None,
                "hs": _date(r["harvest_start"]),
                "he": _date(r["harvest_end"]),
                "pre_ndvi": _num(r["pre_ndvi"], 3),
                "pre_ndmi": _num(r["pre_ndmi"], 3),
                "explanation": _text(r["explanation"]) if "explanation" in preds.columns else None,
                "flip": _text(r["flip_reason"]) if "flip_reason" in preds.columns else None,
            })
            # ground after the cut: model's numbers if given, else read from the series
            post_v, post_r = post_harvest(series.get(fid), _day(p["he"], year))
            given_v = _num(r["post_ndvi"], 3) if "post_ndvi" in preds.columns else None
            given_r = _num(r["post_ndti"], 3) if "post_ndti" in preds.columns else None
            p["post_ndvi"] = given_v if given_v is not None else post_v
            p["post_ndti"] = given_r if given_r is not None else post_r
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
    shown_ids = {p["field_id"] for p in shown}

    asof_out = None
    if asof_path is not None:
        asof_out = asof_tables(load_asof(asof_path), props, shown_ids, threshold)

    all_days = [d for fid in shown_ids for d in series.get(fid, {}).get("t", [])]
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
        geo["features"].append({"type": "Feature", "properties": dict(p), "geometry": gj})

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
            "asof": None if asof_path is None else Path(asof_path).name,
        },
        "has_series": bool(series),
        "has_predictions": preds is not None,
        "has_labels": metrics["n_labelled"] > 0,
        "has_asof": asof_out is not None and bool(asof_out["cutoffs"]),
        "has_counties": counties_out is not None,
        "has_county_shapes": outlines is not None,
        "n_fields_shown": len(shown),
        "min_clear": min_clear,
        "threshold": threshold,
        "gap_days": gap_days,
        "post_window_days": POST_WINDOW_DAYS,
        "season": season,
        "cues": DEFAULT_CUES,
        "harvest_ref": harvest_reference(props, year) if preds is not None else None,
        "metrics": metrics,
        "note": note,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "meta.json", meta)
    write_json(out_dir / "fields.geojson", geo)
    write_json(out_dir / "series.json", {k: v for k, v in series.items() if k in shown_ids})
    for name, obj in (("asof.json", asof_out), ("counties.json", counties_out), ("counties.geojson", outlines)):
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
    ap.add_argument("--asof", help="Per-field predictions at several cutoff dates, .parquet or .csv (see app/README.md)")
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
        args.fields, args.out, args.timeseries, args.predictions, asof_path=args.asof,
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
