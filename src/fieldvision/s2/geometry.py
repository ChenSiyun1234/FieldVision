"""Field validation and compact blocks in local UTM coordinates."""
from collections import defaultdict
import json
import math
import numpy as np
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import shape, box
from shapely.ops import transform, unary_union


def project(geometry, source, target):
    return transform(Transformer.from_crs(source, target, always_xy=True).transform, geometry)


def load_fields(path, year):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError("Input must be a GeoJSON FeatureCollection in EPSG:4326")
    crs = data.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    if not ("4326" in crs or "CRS84" in crs):
        raise ValueError("Input CRS must be EPSG:4326")
    fields, seen = [], set()
    for feature in data["features"]:
        props = feature.get("properties") or {}
        field_id = props.get("field_id")
        if not isinstance(field_id, str) or not field_id or field_id in seen:
            raise ValueError(f"field_id must be a unique nonempty string: {field_id!r}")
        seen.add(field_id)
        record = {"field_id": field_id, "properties": props}
        try:
            if props.get("year") != year:
                raise ValueError("year_mismatch")
            if props.get("label") not in ("silage", "grain", None):
                raise ValueError("invalid_label")
            geom = shape(feature["geometry"])
            if geom.geom_type not in ("Polygon", "MultiPolygon") or not geom.is_valid or geom.is_empty:
                raise ValueError("invalid_geometry")
            west, south, east, north = geom.bounds
            if not (-180 <= west <= east <= 180 and -80 <= south <= north <= 84):
                raise ValueError("outside_utm_extent")
            record["geometry"] = geom
        except (ValueError, TypeError, KeyError) as exc:
            record["error"] = str(exc)
        fields.append(record)
    if not fields:
        raise ValueError("Input has no fields")
    return fields


def _split_compact(records, size):
    bounds = np.array([field["local_geometry"].bounds for field in records])
    span = bounds[:, 2:].max(axis=0) - bounds[:, :2].min(axis=0)
    if max(span) <= size or len(records) == 1:
        yield records
        return
    axis = int(np.argmax(span))
    records = sorted(records, key=lambda f: f["local_geometry"].centroid.coords[0][axis])
    middle = len(records) // 2
    yield from _split_compact(records[:middle], size)
    yield from _split_compact(records[middle:], size)


def make_blocks(fields, size=6000):
    groups = defaultdict(list)
    for field in fields:
        if field.get("error"):
            continue
        center = field["geometry"].centroid
        zone = max(1, min(60, int((center.x + 180) // 6) + 1))
        epsg = (32600 if center.y >= 0 else 32700) + zone
        local = project(field["geometry"], 4326, epsg)
        field["local_geometry"] = local
        key = (epsg, math.floor(local.centroid.x / size), math.floor(local.centroid.y / size))
        groups[key].append(field)
    blocks = []
    for key, records in sorted(groups.items()):
        for part, subset in enumerate(_split_compact(sorted(records, key=lambda f: f["field_id"]), size)):
            blocks.append({"id": f"{key[0]}_{key[1]}_{key[2]}_{part}", "epsg": key[0], "fields": subset,
                           "bbox": unary_union([f["geometry"] for f in subset]).bounds})
    return blocks


def scene_grid(block, item):
    props, red = item["properties"], item["assets"]["red"]
    crs = red.get("proj:epsg") or props.get("proj:epsg") or props.get("proj:code")
    if crs is None:
        raise ValueError("Scene lacks a projected CRS")
    geometries = [project(field["geometry"], 4326, crs) for field in block["fields"]]
    bounds = np.array([geom.bounds for geom in geometries])
    west, south = bounds[:, :2].min(axis=0)
    east, north = bounds[:, 2:].max(axis=0)
    # Align to the scene's 10 m grid, including when its UTM zone differs.
    affine = red.get("proj:transform", props.get("proj:transform"))
    if affine is None:
        raise ValueError("Scene lacks asset grid transform")
    x0, y0 = affine[2], affine[5]
    west = x0 + math.floor((west - x0) / 10) * 10
    north = y0 + math.ceil((north - y0) / 10) * 10
    width, height = max(1, math.ceil((east - west) / 10)), max(1, math.ceil((north - south) / 10))
    if width * height > 4_000_000:
        raise ValueError("Single field/block exceeds 4 million pixels; split the input polygon")
    return crs, from_origin(west, north, 10, 10), (height, width), geometries
