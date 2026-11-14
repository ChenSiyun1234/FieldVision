#!/usr/bin/env python3
"""Per-field Sentinel-2 time series for BARC labeled corn fields.

Reads the BARC silage/grain GeoPackage, loads cloud-filtered Sentinel-2 L2A
from the Microsoft Planetary Computer, and writes one median NDVI/NDTI row
per field per clear date. Raw imagery is not saved.

Run from the repo root:

    python src/build_timeseries.py --test
    python src/build_timeseries.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
from matplotlib.lines import Line2D
from odc.stac import load
from pystac_client import Client
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
    from src.common import field_date_medians, normalized_difference, polygonal, scale_reflectance
else:
    from .common import field_date_medians, normalized_difference, polygonal, scale_reflectance

# Every processing choice lives here so a run can be repeated from the log.
CONFIG = {
    "layer": "barc_silage_grain_corn20212023",
    "years": (2021, 2022, 2023),
    "label_columns": {2021: "Corn_2021", 2022: "Corn_2022", 2023: "Corn_2023"},
    "labels": ("Grain", "Silage"),
    "expected_rows": 44,
    "crs": "EPSG:32618",
    "resolution": 10,  # metres; one pixel covers 100 m^2
    "buffer_m": 15,  # inward buffer, drops mixed edge pixels
    "min_pixels": 50,
    "load_pad_m": 30,  # keep field edges inside the load window after pixel snapping
    "catalog": "https://planetarycomputer.microsoft.com/api/stac/v1",
    "collection": "sentinel-2-l2a",
    "month_day_start": (5, 1),
    "month_day_end": (12, 15),
    "cloud_cover_lt": 60,
    "bands": ("B04", "B08", "B11", "B12", "SCL"),
    "refl_bands": ("B04", "B08", "B11", "B12"),
    # SCL is categorical, so it is nearest-neighbor. Reflectance is bilinear.
    "resampling": {
        "B04": "bilinear",
        "B08": "bilinear",
        "B11": "bilinear",
        "B12": "bilinear",
        "SCL": "nearest",
    },
    # 4 = vegetation, 5 = bare soil. Not water, cloud, shadow, or snow.
    "scl_classes": (4, 5),
    # Baseline 04.00 (from 2022-01-25) stores reflectance as DN = value * 10000 + 1000.
    "offset_date": "2022-01-25",
    "dn_offset": 1000.0,
    "scale": 10000.0,
    "clear_frac_min": 0.8,
    "late_start": (8, 1),
    "late_end": (11, 30),
    "min_late_dates": 8,
    "out_dir": "data/processed",
    "csv_name": "field_timeseries.csv",
    "plot_name": "ndvi_curves.png",
    "log_name": "DATA_LOG.md",
    "colors": {"Silage": "#2f7d32", "Grain": "#d6a81d"},
    "test_year": 2022,
    "test_per_class": 2,
}

COLUMNS = [
    "OBJECTID",
    "year",
    "date",
    "doy",
    "ndvi",
    "ndti",
    "clear_frac",
    "n_pixels",
    "label",
]


def find_gpkg() -> Path:
    """Locate the BARC GeoPackage without modifying it."""
    name = "BARC_Corn.2021-2023.gpkg"
    candidates = [
        ROOT / "04_Evaluation" / name,
        ROOT / "notebooks" / "dataset" / name,
        Path.cwd() / "04_Evaluation" / name,
        Path.cwd() / "notebooks" / "dataset" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    for base in (ROOT, Path.cwd()):
        matches = [
            path
            for path in base.rglob(name)
            if ".venv" not in path.parts and "site-packages" not in path.parts
        ]
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {name} under {ROOT} or the working directory.")


def season_dates(year: int) -> tuple[str, str]:
    start_m, start_d = CONFIG["month_day_start"]
    end_m, end_d = CONFIG["month_day_end"]
    return f"{year}-{start_m:02d}-{start_d:02d}", f"{year}-{end_m:02d}-{end_d:02d}"


def load_labels(path: Path) -> gpd.GeoDataFrame:
    """Melt Corn_YYYY into one row per labeled field-year. Blank is not a label."""
    fields = gpd.read_file(path, layer=CONFIG["layer"])
    pieces = []
    for year, column in CONFIG["label_columns"].items():
        labeled = fields.loc[fields[column].isin(CONFIG["labels"]), ["OBJECTID", column, "geometry"]].copy()
        labeled = labeled.rename(columns={column: "label"})
        labeled["year"] = year
        pieces.append(labeled)

    labels = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), geometry="geometry", crs=fields.crs)
    counts = labels.groupby(["year", "label"]).size().unstack(fill_value=0)
    print("Labeled field-years by year and class:")
    print(counts.to_string())
    print(f"Labeled field-years: {len(labels)} on {labels['OBJECTID'].nunique()} polygons")
    assert len(labels) == CONFIG["expected_rows"], (
        f"Expected {CONFIG['expected_rows']} labeled field-years, found {len(labels)}."
    )
    return labels


def prepare_polygons(labels: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Reproject to UTM 18N and buffer inward so field edges do not mix with neighbours."""
    projected = labels.to_crs(CONFIG["crs"]).copy()
    projected["geometry"] = projected.geometry.map(lambda geom: polygonal(geom.buffer(-CONFIG["buffer_m"])))
    empty = projected.geometry.is_empty | projected.geometry.isna() | (projected.geometry.area <= 0)
    dropped = projected.loc[empty, ["OBJECTID", "year", "label"]].copy()
    if len(dropped):
        print("Dropped after the inward buffer (empty geometry):")
        print(dropped.to_string(index=False))
    else:
        print("No fields dropped by the inward buffer.")

    kept = projected.loc[~empty].copy()
    pixel_area = CONFIG["resolution"] ** 2
    kept["n_pixels_expected"] = kept.geometry.area / pixel_area
    small = kept.loc[kept["n_pixels_expected"] < CONFIG["min_pixels"], ["OBJECTID", "year", "label", "n_pixels_expected"]]
    if len(small):
        print(f"Warning: fields under {CONFIG['min_pixels']} expected 10 m pixels:")
        print(small.to_string(index=False))
    return kept, dropped


def select_test_fields(fields: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Four fields from the test year: two Silage and two Grain, largest first."""
    year = CONFIG["test_year"]
    pool = fields.loc[fields["year"] == year]
    chosen = []
    for label in CONFIG["labels"]:
        part = pool.loc[pool["label"] == label].sort_values(
            ["n_pixels_expected", "OBJECTID"], ascending=[False, True]
        )
        take = part.head(CONFIG["test_per_class"])
        if len(take) < CONFIG["test_per_class"]:
            print(f"Warning: only {len(take)} {label} fields left in {year} after buffering.")
        chosen.append(take)
    selected = gpd.GeoDataFrame(pd.concat(chosen, ignore_index=True), geometry="geometry", crs=fields.crs)
    print("Test mode is using these fields:")
    print(selected[["OBJECTID", "year", "label", "n_pixels_expected"]].to_string(index=False))
    return selected


def load_bbox(fields: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """Lon/lat bounding box of the fields, padded so snapping does not clip the edge."""
    minx, miny, maxx, maxy = fields.total_bounds
    pad = CONFIG["load_pad_m"]
    box = Polygon(
        [
            (minx - pad, miny - pad),
            (maxx + pad, miny - pad),
            (maxx + pad, maxy + pad),
            (minx - pad, maxy + pad),
        ]
    )
    lonlat = gpd.GeoSeries([box], crs=fields.crs).to_crs("EPSG:4326").total_bounds
    return tuple(float(value) for value in lonlat)


def search_scenes(catalog: Client, fields: gpd.GeoDataFrame, year: int):
    """One STAC search for this year's fields. Returns signed items, or none."""
    start, end = season_dates(year)
    bbox = load_bbox(fields)
    print(f"{year}: searching {start} to {end}, cloud cover < {CONFIG['cloud_cover_lt']}%")
    search = catalog.search(
        collections=[CONFIG["collection"]],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": CONFIG["cloud_cover_lt"]}},
    )
    items = list(search.items())
    print(f"{year}: {len(items)} scenes")
    for item in items:
        planetary_computer.sign_inplace(item)
    return items, bbox


def summarize_fields(cube, reflectance: dict[str, np.ndarray], fields: gpd.GeoDataFrame) -> tuple[list[dict], list[str]]:
    """Median NDVI and NDTI inside each field, on dates that are mostly clear."""
    ndvi = normalized_difference(reflectance["B08"], reflectance["B04"])
    ndti = normalized_difference(reflectance["B11"], reflectance["B12"])
    clear = np.isin(cube["SCL"].values, CONFIG["scl_classes"])
    rows: list[dict] = []
    notes: list[str] = []
    for field in fields.itertuples(index=False):
        kept, note = field_date_medians(
            cube,
            field.geometry,
            clear,
            {"ndvi": ndvi, "ndti": ndti},
            CONFIG["clear_frac_min"],
            require_finite="ndvi",
        )
        if note:
            message = f"OBJECTID {field.OBJECTID} {field.year}: {note}"
            print(message)
            notes.append(message)
        for item in kept:
            rows.append(
                {
                    "OBJECTID": int(field.OBJECTID),
                    "year": int(field.year),
                    "label": field.label,
                    **item,
                }
            )
    return rows, notes


def _date_counts(table, name: str) -> pd.DataFrame:
    """Field-year counts. An empty table still has the columns the merge expects."""
    if len(table) == 0:
        return pd.DataFrame(columns=["OBJECTID", "year", name])
    return table.groupby(["OBJECTID", "year"]).size().rename(name).reset_index()


def report_coverage(table, attempted: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Print how many kept dates each field-year has, and who is thin in late season."""
    keys = attempted[["OBJECTID", "year", "label"]].drop_duplicates()
    summary = keys.merge(_date_counts(table, "n_dates"), on=["OBJECTID", "year"], how="left")
    summary["n_dates"] = summary["n_dates"].fillna(0).astype(int)

    print(
        "Valid dates per field-year: "
        f"min {summary['n_dates'].min()}, "
        f"median {float(np.median(summary['n_dates'])):.1f}, "
        f"max {summary['n_dates'].max()} "
        f"({len(summary)} field-years)"
    )

    late_start_m, late_start_d = CONFIG["late_start"]
    late_end_m, late_end_d = CONFIG["late_end"]
    if len(table):
        start = table["year"].map(lambda year: f"{year}-{late_start_m:02d}-{late_start_d:02d}")
        end = table["year"].map(lambda year: f"{year}-{late_end_m:02d}-{late_end_d:02d}")
        late = table.loc[(table["date"] >= start) & (table["date"] <= end)]
    else:
        late = table
    summary = summary.merge(_date_counts(late, "late_dates"), on=["OBJECTID", "year"], how="left")
    summary["late_dates"] = summary["late_dates"].fillna(0).astype(int)
    thin = summary.loc[summary["late_dates"] < CONFIG["min_late_dates"]].sort_values(["year", "OBJECTID"])
    window = f"{late_start_m:02d}-{late_start_d:02d} to {late_end_m:02d}-{late_end_d:02d}"
    if len(thin):
        print(f"Field-years with fewer than {CONFIG['min_late_dates']} valid dates between {window}:")
        print(thin[["OBJECTID", "year", "label", "late_dates"]].to_string(index=False))
    else:
        print(f"Every field-year has at least {CONFIG['min_late_dates']} valid dates between {window}.")
    return summary


def plot_curves(table, years: list[int], path: Path) -> None:
    """One NDVI line per field-year, one subplot per year."""
    n_years = max(len(years), 1)
    fig, axes = plt.subplots(1, n_years, figsize=(5.2 * n_years, 4.2), sharey=True, squeeze=False)
    axes = list(axes[0])
    for ax, year in zip(axes, years):
        subset = table.loc[table["year"] == year] if len(table) else table
        for field_key, curve in subset.groupby(["OBJECTID", "label"]):
            curve = curve.sort_values("doy")
            _, label = field_key
            ax.plot(
                curve["doy"],
                curve["ndvi"],
                color=CONFIG["colors"][label],
                marker="o",
                markersize=3.5,
                linewidth=1.0,
                alpha=0.85,
            )
        ax.set_title(str(year))
        ax.set_xlabel("Day of year")
        ax.set_xlim(121, 350)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("NDVI")
    handles = [
        Line2D([0], [0], color=CONFIG["colors"][label], marker="o", label=label)
        for label in ("Silage", "Grain")
    ]
    axes[-1].legend(handles=handles, loc="lower right")
    fig.suptitle("BARC corn fields — cloud-masked median NDVI")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def append_log(
    gpkg: Path,
    test_mode: bool,
    scene_counts: dict,
    n_rows: int,
    dropped: gpd.GeoDataFrame,
    notes: list[str],
    thin: gpd.GeoDataFrame | None,
    error: str | None,
) -> None:
    window = (
        f"{CONFIG['month_day_start'][0]:02d}-{CONFIG['month_day_start'][1]:02d}"
        f" to {CONFIG['month_day_end'][0]:02d}-{CONFIG['month_day_end'][1]:02d}"
    )
    scene_text = ", ".join(f"{year}: {count}" for year, count in scene_counts.items()) or "none"
    dropped_text = "none"
    if len(dropped):
        dropped_text = ", ".join(f"{int(row.OBJECTID)}/{int(row.year)}" for row in dropped.itertuples(index=False))
    thin_text = "none"
    if thin is not None and len(thin):
        thin_text = ", ".join(
            f"{int(row.OBJECTID)}/{int(row.year)} ({int(row.late_dates)})" for row in thin.itertuples(index=False)
        )
    note_text = "; ".join(notes) if notes else "none"
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    entry = f"""
## {stamp} — BARC Sentinel-2 field time series

- mode: {"test" if test_mode else "full"}
- geopackage: `{gpkg}`
- catalog: {CONFIG["catalog"]}
- collection: {CONFIG["collection"]}
- date range: {window} of each processed year
- cloud threshold: eo:cloud_cover < {CONFIG["cloud_cover_lt"]}
- bands: {", ".join(CONFIG["bands"])}
- resampling: reflectance bilinear, SCL nearest
- CRS: {CONFIG["crs"]}
- resolution: {CONFIG["resolution"]} m
- buffer: {CONFIG["buffer_m"]} m inward
- SCL classes kept: {list(CONFIG["scl_classes"])} (4 vegetation, 5 bare soil)
- offset correction: on dates on or after {CONFIG["offset_date"]}, subtract {CONFIG["dn_offset"]:.0f} DN then divide by {CONFIG["scale"]:.0f}; DN 0 is nodata and is masked before the offset; clip at 0
- clear_frac cutoff: {CONFIG["clear_frac_min"]} (valid pixels / pixels inside the polygon; valid = inside and SCL in the classes above)
- scene counts: {scene_text}
- rows written: {n_rows}
- fields dropped by the inward buffer: {dropped_text}
- field-years under {CONFIG["min_late_dates"]} valid dates from Aug 1 to Nov 30: {thin_text}
- skip notes: {note_text}
- output: `{CONFIG["out_dir"]}/{CONFIG["csv_name"]}`, `{CONFIG["out_dir"]}/{CONFIG["plot_name"]}`
- error: {error or "none"}
"""
    log_path = ROOT / CONFIG["log_name"]
    if not log_path.exists():
        log_path.write_text(
            "# Data log\n\nQueries and processing choices for the BARC Sentinel-2 field time series. "
            "Each run of `src/build_timeseries.py` appends a record below.\n",
            encoding="utf-8",
        )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    print(f"Appended a run record to {log_path}")


def process_year(catalog: Client, fields: gpd.GeoDataFrame, year: int) -> tuple[list[dict], list[str], int]:
    year_fields = fields.loc[fields["year"] == year]
    if year_fields.empty:
        print(f"{year}: no fields left after buffering, skipping the search")
        return [], [f"{year}: no fields after buffering"], 0

    items, _bbox = search_scenes(catalog, year_fields, year)
    if not items:
        note = f"{year}: zero scenes, skipped"
        print(note)
        return [], [note], 0

    print(f"{year}: loading {len(items)} scenes into memory (no imagery written to disk)")
    cube = load(
        items,
        bands=list(CONFIG["bands"]),
        bbox=load_bbox(year_fields),
        crs=CONFIG["crs"],
        resolution=CONFIG["resolution"],
        groupby="solar_day",
        resampling=CONFIG["resampling"],
        fail_on_error=False,
    )
    if cube.sizes.get("time", 0) == 0 or cube.sizes.get("x", 0) == 0 or cube.sizes.get("y", 0) == 0:
        note = f"{year}: load returned an empty cube, skipped"
        print(note)
        return [], [note], len(items)

    missing = [band for band in CONFIG["bands"] if band not in cube]
    if missing:
        raise RuntimeError(f"{year}: cube is missing bands {missing}")

    reflectance = scale_reflectance(cube, CONFIG["refl_bands"])
    rows, notes = summarize_fields(cube, reflectance, year_fields)
    print(f"{year}: kept {len(rows)} field-date rows")
    return rows, notes, len(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-field Sentinel-2 time series for BARC corn fields.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process only 4 fields (2 Silage, 2 Grain) from 2022.",
    )
    args = parser.parse_args()

    gpkg = find_gpkg()
    print(f"GeoPackage: {gpkg}")
    print("Mode: test" if args.test else "Mode: full")

    labels = load_labels(gpkg)
    fields, dropped = prepare_polygons(labels)
    if args.test:
        fields = select_test_fields(fields)
    years = sorted(int(year) for year in fields["year"].unique())

    out_dir = ROOT / CONFIG["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / CONFIG["csv_name"]
    plot_path = out_dir / CONFIG["plot_name"]

    scene_counts: dict = {year: "not run" for year in CONFIG["years"]}
    notes: list[str] = []
    rows: list[dict] = []
    error = None
    summary = None
    try:
        catalog = Client.open(CONFIG["catalog"])
        for year in years:
            year_rows, year_notes, n_scenes = process_year(catalog, fields, year)
            scene_counts[year] = n_scenes
            notes.extend(year_notes)
            rows.extend(year_rows)

        table = pd.DataFrame(rows, columns=COLUMNS)
        if len(table):
            table = table.sort_values(["OBJECTID", "year", "date"]).reset_index(drop=True)
        table.to_csv(csv_path, index=False)
        print(f"Wrote {len(table)} rows to {csv_path}")

        summary = report_coverage(table, fields)
        plot_curves(table, years, plot_path)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        thin = None
        if summary is not None:
            thin = summary.loc[summary["late_dates"] < CONFIG["min_late_dates"]]
        append_log(gpkg, args.test, scene_counts, len(rows), dropped, notes, thin, error)


if __name__ == "__main__":
    main()
