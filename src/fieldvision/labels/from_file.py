"""Turn any vector file of field polygons into the label contract (fields.geojson + GeoParquet + summary.json).

Three layouts are understood:

* one label column and one year for the whole file          --label-column L --year 2024
* one label column and a column holding each row's year      --label-column L --year-column Y
* one label column per year ("wide" table)                   --year-columns Corn_2021=2021 Corn_2022=2022
* no labels at all (fields to classify), one year            --unlabelled --year 2022

Rows can be filtered on attributes first (--where COL=V1,V2, repeatable, all must hold).

Example for unlabelled fields (USDA Crop Sequence Boundaries: corn in 2022 in one county):

    python -m fieldvision.labels.from_file md_csb.parquet --where CNTYFIPS=021 --where CDL2022=1 \\
        --unlabelled --year 2022 --id-column CSBID --no-year-in-id --prefix csb --country US \\
        --source nass-csb --metric-crs EPSG:32618 --keep CSBACRES=acres_reported --set county_fips=24021 --out <folder>

Example (the organisers' evaluation file, see docs/data-handoff.md):

    python -m fieldvision.labels.from_file BARC_Corn.2021-2023.gpkg --layer barc_silage_grain_corn20212023 \\
        --id-column OBJECTID --year-columns Corn_2021=2021 Corn_2022=2022 Corn_2023=2023 \\
        --map Silage=silage --map Grain=grain --country US --source barc --prefix barc --out <folder>
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from urllib.parse import quote

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, mapping
from shapely.ops import unary_union
from pyproj import CRS, Transformer

from .core import block_id as grid_block_id

LABELS = ('silage', 'grain')
CONTRACT = ['field_id', 'label', 'year', 'country', 'source', 'crop_code', 'crop_name',
            'area_ha', 'block_id', 'centroid_lon', 'centroid_lat']
SMALL_HA = 0.33          # about a third of a hectare: below this a Sentinel-2 pixel (0.01 ha) is mostly edge
EUROPE = (-32.0, 27.0, 45.0, 72.0)


class InputError(ValueError):
    """The input cannot be converted; the message says what is needed."""


def now():
    return datetime.now(timezone.utc).isoformat()


def polygonal(geom):
    """Valid polygon or multipolygon made of the polygonal parts of `geom`; None if nothing is left."""
    if geom is None or geom.is_empty:
        return None
    geom = shapely.force_2d(geom)
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    parts = []

    def walk(g):
        if g.geom_type == 'Polygon':
            parts.append(g)
        elif g.geom_type == 'MultiPolygon':
            parts.extend(g.geoms)
        elif g.geom_type == 'GeometryCollection':
            for h in g.geoms:
                walk(h)

    walk(geom)
    parts = [p for p in parts if not p.is_empty and p.area > 0]
    if not parts:
        return None
    out = parts[0] if len(parts) == 1 else unary_union(parts)
    if out.geom_type == 'GeometryCollection':
        return polygonal(out)
    if out.geom_type not in ('Polygon', 'MultiPolygon') or not out.is_valid:
        return None
    return out


def clean_geometries(geoms, counters):
    """Repair geometries, keeping only polygonal parts. Counts what happened."""
    result = []
    for g in geoms:
        if g is None or g.is_empty:
            counters['missing_geometry'] += 1
            result.append(None)
            continue
        if g.geom_type in ('Point', 'MultiPoint', 'LineString', 'MultiLineString'):
            counters['non_polygon'] += 1
            result.append(None)
            continue
        if not g.is_valid:
            counters['invalid_input'] += 1
        elif g.geom_type == 'GeometryCollection':
            counters['geometry_collection'] += 1
        fixed = polygonal(g)
        if fixed is None:
            counters['unrepairable'] += 1
        elif g.geom_type == 'GeometryCollection' or not g.is_valid:
            counters['repaired'] += 1
        result.append(fixed)
    return result


def grid_crs_for(frame):
    """Metric equal-area CRS used for areas and the 10 km grid: EPSG:3035 in Europe, else a local LAEA."""
    x0, y0, x1, y1 = frame.total_bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, s, e, n = EUROPE
    if w <= x0 and x1 <= e and s <= y0 and y1 <= n:
        return CRS.from_epsg(3035)
    return CRS.from_proj4(f'+proj=laea +lat_0={round(cy)} +lon_0={round(cx)} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')


def parse_map(items):
    mapping_ = {}
    for item in items or []:
        if '=' not in item:
            raise InputError(f'--map expects RAW=silage or RAW=grain, got "{item}"')
        raw, label = item.rsplit('=', 1)
        label = label.strip().lower()
        if label not in LABELS:
            raise InputError(f'--map target must be silage or grain, got "{label}" in "{item}"')
        mapping_[raw.strip().lower()] = label
    if not mapping_:
        raise InputError('At least one --map RAW=silage / RAW=grain is needed so we know which value means which class')
    return mapping_


def parse_pairs(items, option):
    """['A=1,2', 'B=x'] -> {'A': ['1', '2'], 'B': ['x']} for --where; values stay strings."""
    out = {}
    for item in items or []:
        if '=' not in item:
            raise InputError(f'{option} expects NAME=VALUE, got "{item}"')
        key, value = item.split('=', 1)
        out[key.strip()] = [v.strip() for v in value.split(',')]
    return out


def as_text(value):
    """Attribute value compared as text: 1, 1.0 and '1' all become '1'."""
    if isinstance(value, (float, np.floating)) and not math.isnan(value) and float(value).is_integer():
        return str(int(value))
    return clean_text(value)


def parse_year_columns(items):
    out = []
    for item in items:
        if '=' not in item:
            raise InputError(f'--year-columns expects COLUMN=YEAR, got "{item}"')
        col, year = item.rsplit('=', 1)
        try:
            out.append((col, int(year)))
        except ValueError:
            raise InputError(f'Year in "{item}" is not a number') from None
    return out


def clean_text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        return ''
    return str(value).strip()


def json_value(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value if isinstance(value, (str, int, bool)) else str(value)


def convert(frame, *, label_map, country, source, prefix, id_column=None, label_column=None, year=None,
            year_column=None, year_columns=None, split=None, keep=(), block_const=None, block_column=None,
            crop_code_column=None, min_area_ha=0.0, assume_crs=None, where=None, unlabelled=False,
            year_in_id=True, metric_crs=None, set_props=None, crop_name=None):
    """GeoDataFrame in, (features, report) out. No file access, so it can be tested offline.

    `keep` items are column names or COLUMN=NEW_NAME. `where` maps column -> allowed values (text).
    `unlabelled` writes label null for every row (fields to be classified); needs `year`.
    `metric_crs` replaces the default equal-area CRS for area_ha, centroids and the 10 km grid.
    """
    keep = [k.split('=', 1) if '=' in k else [k, k] for k in (keep or [])]
    set_props = dict(set_props or {})
    modes = [year is not None, year_column is not None, bool(year_columns)]
    if sum(modes) != 1:
        raise InputError('Give exactly one of --year, --year-column or --year-columns')
    if unlabelled:
        if year is None:
            raise InputError('--unlabelled needs one --year for the whole file')
        if label_column or label_map:
            raise InputError('--unlabelled cannot be combined with --label-column or --map')
        needed = []
    elif year_columns:
        needed = [c for c, _ in year_columns]
    else:
        if not label_column:
            raise InputError('--label-column is needed with --year or --year-column')
        needed = [label_column] + ([year_column] if year_column else [])
    clash = [new for _, new in keep if new in CONTRACT + ['split', 'geometry']]
    clash += [k for k in set_props if k in CONTRACT + ['split', 'geometry']]
    if clash:
        raise InputError(f'--keep / --set cannot use contract names: {", ".join(clash)}')
    names = [new for _, new in keep] + list(set_props)
    if len(names) != len(set(names)):
        raise InputError(f'--keep / --set write the same property twice: {names}')
    for col in needed + [old for old, _ in keep] + list(where or {}) + [c for c in (id_column, block_column, crop_code_column) if c]:
        if col not in frame.columns:
            raise InputError(f'Column "{col}" not found. Columns in the file: {", ".join(map(str, frame.columns))}')

    frame = frame.reset_index(drop=True)
    report = {'input_rows': int(len(frame)), 'dropped': Counter(), 'unmapped_values': Counter()}
    if where:
        keep_rows = np.ones(len(frame), dtype=bool)
        for col, allowed in where.items():
            keep_rows &= frame[col].map(as_text).isin(allowed).to_numpy()
        report['where'] = {'conditions': {c: list(v) for c, v in where.items()},
                           'rows_matching': int(keep_rows.sum()), 'rows_not_matching': int((~keep_rows).sum())}
        frame = frame[keep_rows].reset_index(drop=True)
        if frame.empty:
            raise InputError(f'No row matches --where {report["where"]["conditions"]}')
    geom_types = frame.geometry.geom_type.fillna('None')
    report['input_geometry_types'] = dict(Counter(geom_types))
    polygons = geom_types.isin(['Polygon', 'MultiPolygon', 'GeometryCollection']).sum()
    if polygons == 0:
        if geom_types.isin(['Point', 'MultiPoint']).all():
            raise InputError(
                'This file only contains points. We need the outline of each field as a polygon '
                '(a shape with an area), because the satellite signal is averaged over the whole field. '
                'Please export the field boundaries (a Shapefile, GeoPackage or GeoJSON with polygons).')
        raise InputError(f'No polygon geometries found (found: {dict(Counter(geom_types))}). We need field outlines as polygons.')
    if frame.crs is None:
        if assume_crs is None:
            raise InputError('The file has no coordinate system. Pass --crs EPSG:xxxx (for example --crs EPSG:4326).')
        frame = frame.set_crs(assume_crs)
    report['input_crs'] = frame.crs.to_string() if frame.crs.to_epsg() is None else f'EPSG:{frame.crs.to_epsg()}'
    report['input_crs_name'] = frame.crs.name

    counters = report['dropped']
    repair = Counter()
    first = clean_geometries(list(frame.geometry), repair)
    frame = frame.assign(geometry=gpd.GeoSeries(first, index=frame.index, crs=frame.crs))
    frame = frame.to_crs(4326)
    # reprojection can leave self-touching rings or stray lines: repair again in lon/lat
    second_repair = Counter()
    second = clean_geometries(list(frame.geometry), second_repair)
    report['repair'] = {'source_crs': dict(repair), 'after_reprojection': dict(second_repair)}
    frame = frame.assign(geometry=gpd.GeoSeries(second, index=frame.index, crs=4326))
    usable = frame.geometry.notna()
    for reason in ('missing_geometry', 'non_polygon'):
        if repair[reason]:
            counters[reason] = int(repair[reason])
    lost = repair['unrepairable'] + second_repair['unrepairable']
    if lost:
        counters['unrepairable'] = int(lost)
    report['rows_with_polygon'] = int(usable.sum())

    # one row per (polygon, year) with a mapped label
    records = []
    mapped = {}
    plan = [(c, y) for c, y in (year_columns or [])] if year_columns else [(label_column, None)]
    if unlabelled:
        records = [(idx, year, None, crop_name) for idx in frame.index if usable[idx]]
        plan = []
    for col, fixed_year in plan:
        for idx, raw in frame[col].items():
            if not usable[idx]:
                continue
            text = clean_text(raw)
            row_year = fixed_year if year_columns else (year if year is not None else None)
            if year_column:
                try:
                    row_year = int(float(clean_text(frame.at[idx, year_column])))
                except ValueError:
                    counters['bad_year'] += 1
                    continue
            if not text:
                counters[f'blank_label{"_" + str(row_year) if row_year else ""}'] += 1
                continue
            label = label_map.get(text.lower())
            if label is None:
                report['unmapped_values'][text] += 1
                counters[f'unmapped_label{"_" + str(row_year) if row_year else ""}'] += 1
                continue
            records.append((idx, row_year, label, text))
    if not records:
        raise InputError('No row has a label that maps to silage or grain. Unmapped values seen: '
                         f'{dict(report["unmapped_values"]) or "none (all labels blank)"}')

    grid = CRS.from_user_input(metric_crs) if metric_crs else grid_crs_for(frame[usable])
    report['grid_crs'] = grid.to_string() if grid.to_epsg() is None else f'EPSG:{grid.to_epsg()}'
    metric = frame.geometry.to_crs(grid)
    to_lonlat = Transformer.from_crs(grid, 4326, always_xy=True).transform
    features, seen = [], {}
    for idx, row_year, label, text in records:
        geom, mgeom = frame.geometry[idx], metric[idx]
        area = mgeom.area / 10000
        if area < min_area_ha or area <= 0:
            counters['below_min_area'] += 1
            continue
        center = mgeom.centroid
        lon, lat = to_lonlat(center.x, center.y)
        pid = clean_text(frame.at[idx, id_column]) if id_column else str(idx + 1)
        if id_column and not pid:
            counters['missing_id'] += 1
            continue
        if id_column and isinstance(frame.at[idx, id_column], (float, np.floating)) and float(pid).is_integer():
            pid = str(int(float(pid)))
        fid = f'{prefix}-{quote(pid, safe="-._")}' + (f'-{row_year}' if year_in_id else '')
        if fid in seen:
            raise InputError(f'Duplicate field_id "{fid}" (rows {seen[fid]} and {idx}). '
                             f'Every field needs a unique value in --id-column{"" if id_column else " (no id column given: row numbers were used)"}, '
                             'and in a long layout the same field may appear only once per year'
                             f'{"" if year_in_id else " (with --no-year-in-id an id may appear only once)"}.')
        seen[fid] = idx
        block = (block_const if block_const else
                 clean_text(frame.at[idx, block_column]) if block_column else
                 grid_block_id(country, center.x, center.y))
        props = {'field_id': fid, 'label': label, 'year': row_year, 'country': country, 'source': source,
                 'crop_code': json_value(frame.at[idx, crop_code_column]) if crop_code_column else None,
                 'crop_name': text, 'area_ha': area, 'block_id': block,
                 'centroid_lon': lon, 'centroid_lat': lat}
        if split:
            props['split'] = split
        for col, new in keep:
            props[new] = json_value(frame.at[idx, col])
        props.update(set_props)
        features.append({'type': 'Feature', 'id': fid, 'geometry': mapping(geom), 'properties': props})
    if not features:
        raise InputError('Every labelled field was dropped; see the dropped counts: ' + json.dumps(dict(counters)))
    features.sort(key=lambda f: f['properties']['field_id'])
    report['dropped'] = dict(counters)
    report['unmapped_values'] = dict(report['unmapped_values'])
    return features, report


def verify_geojson(path, features):
    """Re-read the written file and check the contract."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    frame = gpd.GeoDataFrame.from_features(data['features'], crs=4326)
    assert len(frame) == len(features) and frame.field_id.is_unique
    assert set(frame.label.dropna()) <= set(LABELS)
    assert frame.label.isna().all() or frame.label.notna().all(), 'mix of labelled and unlabelled fields'
    assert all(c in frame.columns for c in CONTRACT)
    assert frame.geometry.is_valid.all() and frame.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).all()
    assert frame.centroid_lon.between(-180, 180).all() and frame.centroid_lat.between(-90, 90).all()
    assert (frame.area_ha > 0).all()
    return frame


def summarise(frame, report, args_dict, started):
    frame = frame.assign(label=frame.label.fillna('unlabelled'))
    by_year_class = frame.groupby(['year', 'label']).size().unstack(fill_value=0)
    summary = {
        'fetch_started': started, 'fetch_finished': now(),
        'counts_per_class': frame.label.value_counts().to_dict(),
        'counts_per_year': {int(y): {k: int(v) for k, v in row.items()} for y, row in by_year_class.iterrows()},
        'features': int(len(frame)), 'unique_polygons_labelled': int(frame.geometry.to_wkb().nunique()),
        'blocks': int(frame.block_id.nunique()),
        'counts_per_block': {b: {k: int(v) for k, v in row.items()}
                             for b, row in frame.groupby(['block_id', 'label']).size().unstack(fill_value=0).iterrows()},
        'area_ha': frame.groupby('label').area_ha.describe().to_dict(),
        'area_ha_total': float(frame.area_ha.sum()),
        'fields_below_third_hectare': int((frame.area_ha < SMALL_HA).sum()),
        'input': {k: report[k] for k in ('input_rows', 'where', 'input_geometry_types', 'input_crs', 'input_crs_name',
                                         'rows_with_polygon') if k in report},
        'grid_crs': report['grid_crs'], 'grid_size_m': 10000,
        'repair': report['repair'], 'dropped': report['dropped'], 'unmapped_values': report['unmapped_values'],
        'parameters': args_dict,
        'verification': 'GeoJSON re-read: unique ids, labels in {silage, grain} or all null, contract columns, valid polygonal geometry, positive area; GeoParquet re-read: ids and CRS',
    }
    return summary


def write_outputs(out, features, report, args_dict, started, by_year_dirs=False):
    out.mkdir(parents=True, exist_ok=True)
    geojson = out / 'fields.geojson'
    geojson.write_text(json.dumps({'type': 'FeatureCollection', 'features': features}, ensure_ascii=False), encoding='utf-8')
    frame = verify_geojson(geojson, features)
    frame.to_parquet(out / 'fields.parquet', index=False)
    reread = gpd.read_parquet(out / 'fields.parquet')
    assert list(reread.field_id) == list(frame.field_id) and reread.crs.to_epsg() == 4326
    if by_year_dirs:
        years = sorted({f['properties']['year'] for f in features})
        for y in years:
            d = out / str(y)
            d.mkdir(exist_ok=True)
            part = [f for f in features if f['properties']['year'] == y]
            (d / 'fields.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': part}, ensure_ascii=False), encoding='utf-8')
    summary = summarise(frame, report, args_dict, started)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def read_input(path, layer=None):
    path = Path(path)
    if not path.exists():
        raise InputError(f'File not found: {path}')
    if path.suffix.lower() in ('.parquet', '.geoparquet'):
        return gpd.read_parquet(path)
    return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', type=Path, help='GeoPackage, Shapefile, GeoJSON or GeoParquet with field polygons')
    p.add_argument('--layer', help='layer name (GeoPackage with several layers)')
    p.add_argument('--out', type=Path, required=True, help='output folder (must be outside the repository)')
    p.add_argument('--label-column', help='column holding Silage/Grain (not needed with --year-columns)')
    p.add_argument('--map', action='append', metavar='RAW=LABEL', help='e.g. --map Silage=silage --map Grain=grain (case-insensitive; repeat)')
    p.add_argument('--year', type=int, help='one harvest year for the whole file')
    p.add_argument('--year-column', help='column holding the year of each row (long layout)')
    p.add_argument('--year-columns', nargs='+', metavar='COLUMN=YEAR', help='wide layout: one label column per year')
    p.add_argument('--id-column', help='column with a unique field id (default: row number)')
    p.add_argument('--country', required=True, help='country code written to every feature, e.g. US')
    p.add_argument('--source', help='source name written to every feature (default: file name)')
    p.add_argument('--prefix', help='prefix of field_id (default: lower-case country)')
    p.add_argument('--split', help='value of an extra "split" property, e.g. external_eval')
    p.add_argument('--keep', nargs='+', default=[], metavar='COL[=NAME]', help='extra input columns to copy into properties, optionally renamed')
    p.add_argument('--set', nargs='+', default=[], metavar='NAME=VALUE', help='constant text properties written to every feature')
    p.add_argument('--where', action='append', metavar='COL=V1,V2', help='keep only rows whose COL (as text) is one of the values; repeat to AND')
    p.add_argument('--unlabelled', action='store_true', help='no label column: write label null for every field (needs --year)')
    p.add_argument('--crop-name', help='crop_name written with --unlabelled (default: null)')
    p.add_argument('--no-year-in-id', action='store_true', help='field_id is prefix-<id> without the year suffix')
    p.add_argument('--metric-crs', help='CRS for area_ha, centroids and the 10 km grid (default: EPSG:3035 in Europe, else local equal-area)')
    p.add_argument('--block-id', help='give every field this one block_id instead of the 10 km grid')
    p.add_argument('--block-column', help='take block_id from this column instead of the 10 km grid')
    p.add_argument('--crop-code-column', help='column copied to crop_code (default: null)')
    p.add_argument('--min-area-ha', type=float, default=0.0, help='drop fields smaller than this (default: keep all)')
    p.add_argument('--crs', help='coordinate system to assume when the file has none, e.g. EPSG:4326')
    p.add_argument('--by-year-dirs', action='store_true', help='also write <out>/<year>/fields.geojson')
    p.add_argument('--overwrite', action='store_true', help='allow replacing an existing fields.geojson')
    a = p.parse_args(argv)
    repo = Path(__file__).resolve().parents[3]
    out = a.out.resolve()
    if out == repo or repo in out.parents:
        p.error('Data output must be outside the repository')
    if (out / 'fields.geojson').exists() and not a.overwrite:
        p.error(f'{out / "fields.geojson"} already exists; choose another --out or pass --overwrite')
    started = now()
    try:
        label_map = {} if a.unlabelled and not a.map else parse_map(a.map)
        set_props = {k: ','.join(v) for k, v in parse_pairs(a.set, '--set').items()}
        year_columns = parse_year_columns(a.year_columns) if a.year_columns else None
        frame = read_input(a.input, a.layer)
        features, report = convert(
            frame, label_map=label_map, country=a.country, source=a.source or a.input.name,
            prefix=a.prefix or a.country.lower(), id_column=a.id_column, label_column=a.label_column,
            year=a.year, year_column=a.year_column, year_columns=year_columns, split=a.split,
            keep=a.keep, block_const=a.block_id, block_column=a.block_column,
            crop_code_column=a.crop_code_column, min_area_ha=a.min_area_ha, assume_crs=a.crs,
            where=parse_pairs(a.where, '--where'), unlabelled=a.unlabelled, year_in_id=not a.no_year_in_id,
            metric_crs=a.metric_crs, set_props=set_props, crop_name=a.crop_name)
    except InputError as e:
        raise SystemExit(f'Cannot convert: {e}')
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    summary = write_outputs(out, features, report, args_dict, started, a.by_year_dirs)
    print(json.dumps({k: summary[k] for k in ('counts_per_class', 'counts_per_year', 'blocks', 'dropped', 'fields_below_third_hectare')}))
    print(f'Wrote {out / "fields.geojson"}')


if __name__ == '__main__':
    main()
