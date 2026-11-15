"""Per-field results written by FieldVision's classify, joined with field areas."""

import json
from pathlib import Path

import pandas as pd

from .config import ACRES_PER_HECTARE


def load_field_results(results_path, year=None):
    """Rows from results/<STATE>_<year>_<fips>_<policy>.jsonl (preferred) or the .parquet next to it."""
    path = Path(results_path)
    jsonl = path.with_suffix(".jsonl")
    if jsonl.exists():
        df = pd.DataFrame([json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()])
    else:
        df = pd.read_parquet(path)
    if year is not None and "year" in df:
        df = df[df["year"].astype(int) == int(year)]
    if "evidence" in df:
        df["evidence"] = df["evidence"].map(lambda e: json.loads(e) if isinstance(e, str) else (e or {}))
    df["abstain"] = df.get("abstain", False)
    df["abstain"] = df["abstain"].fillna(False).astype(bool)
    return df.reset_index(drop=True)


def field_areas(fields_geojson):
    """Acres per field, keyed by every id the field file carries."""
    areas = {}
    for feature in json.loads(Path(fields_geojson).read_text())["features"]:
        props = feature.get("properties") or {}
        if props.get("area_ha") is None:
            continue
        acres = float(props["area_ha"]) * ACRES_PER_HECTARE
        for key in ("field_id", "source_field_id", "id"):
            if props.get(key) is not None:
                areas[str(props[key])] = acres
    return areas


def attach_area(df, areas):
    """Add area_acres, matching on source_field_id first, then field_id (case-insensitive)."""
    lower = {k.lower(): v for k, v in areas.items()}
    out = df.copy()
    out["area_acres"] = None
    for col in ("source_field_id", "field_id"):
        if col in out:
            missing = out["area_acres"].isna()
            out.loc[missing, "area_acres"] = out.loc[missing, col].astype(str).str.lower().map(lower)
    out["area_acres"] = out["area_acres"].astype(float)
    return out
