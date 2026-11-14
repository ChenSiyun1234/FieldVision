"""Geometry normalization and deterministic spatial sampling."""

from collections import Counter, defaultdict
import math
import random
from urllib.parse import quote

from pyproj import Transformer
from shapely import make_valid
from shapely.geometry import shape, mapping
from shapely.ops import transform, unary_union

TO_METRIC = Transformer.from_crs(4326, 3035, always_xy=True).transform
TO_WGS84 = Transformer.from_crs(3035, 4326, always_xy=True).transform


def field_id(country, year, parcel_id):
    if parcel_id is None or not str(parcel_id).strip():
        raise ValueError("Missing parcel identifier")
    return f"{country.lower()}{int(year)}-{quote(str(parcel_id), safe='-._')}"


def block_id(country, x, y):
    """Lower-left grid coordinates in kilometres, from a metric centroid."""
    return f"{country}-E{math.floor(x / 10000) * 10}N{math.floor(y / 10000) * 10}"


def normalize(feature, adapter, region, counters, boundary):
    counters['received'] += 1
    p = feature['properties']
    code = str(p[adapter.code_field])
    if code not in adapter.codes:
        counters['other_crop'] += 1
        return None
    if adapter.country == 'NL' and int(p['jaar']) != adapter.year:
        raise ValueError(f"Wrong year returned: {p['jaar']}")
    if not feature.get('geometry'):
        counters['missing_geometry'] += 1
        return None
    geom = shape(feature['geometry'])
    if not geom.is_valid:
        counters['invalid_input'] += 1
        geom = make_valid(geom)
        if geom.geom_type == 'GeometryCollection':
            geom = unary_union([g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')])
        if geom.is_valid and not geom.is_empty and geom.geom_type in ('Polygon', 'MultiPolygon'):
            counters['repaired'] += 1
        else:
            counters['invalid_dropped'] += 1
            return None
    if geom.is_empty or geom.geom_type not in ('Polygon', 'MultiPolygon'):
        counters['non_polygon'] += 1
        return None
    metric = transform(TO_METRIC, geom)
    area = metric.area / 10000
    if area < 1:
        counters['below_1ha'] += 1
        return None
    if not boundary.covers(geom):
        counters['outside_country_boundary'] += 1
        return None
    center = metric.centroid
    lon, lat = TO_WGS84(center.x, center.y)
    label, fallback_name = adapter.codes[code]
    return {'type': 'Feature', 'geometry': mapping(geom), 'properties': {
        'field_id': field_id(adapter.country, adapter.year, adapter.parcel_id(feature)),
        'label': label, 'year': adapter.year, 'country': adapter.country,
        'source': adapter.source, 'crop_code': code,
        'crop_name': p.get(adapter.name_field) or fallback_name,
        'area_ha': area, 'block_id': block_id(adapter.country, center.x, center.y),
        'centroid_lon': lon, 'centroid_lat': lat, 'region': region,
    }}


def balance(features, target, seed=42, cap=60, rare_label='silage'):
    """Rank by rare-class count within regions; rotate regions for coverage."""
    grouped = defaultdict(lambda: defaultdict(list))
    seen = set()
    for f in features:
        p = f['properties']
        if p['field_id'] in seen:
            continue
        seen.add(p['field_id'])
        grouped[p['block_id']][p['label']].append(f)
    regions = defaultdict(list)
    for block, classes in grouped.items():
        n = min(cap, len(classes['silage']), len(classes['grain']))
        if n:
            region = classes['silage'][0]['properties']['region']
            regions[region].append((block, classes, n))
    for blocks in regions.values():
        blocks.sort(key=lambda item: (-len(item[1][rare_label]), item[0]))
    rng = random.Random(seed)
    result, count = [], 0
    while count < target and any(regions.values()):
        for region in sorted(regions):
            if not regions[region] or count >= target:
                continue
            _, classes, n = regions[region].pop(0)
            n = min(n, target - count)
            for label in ('silage', 'grain'):
                candidates = sorted(classes[label], key=lambda f: f['properties']['field_id'])
                result.extend(rng.sample(candidates, n))
            count += n
    return sorted(result, key=lambda f: (f['properties']['block_id'], f['properties']['field_id']))


def counts(features, key):
    result = defaultdict(Counter)
    for f in features:
        p = f['properties']
        result[p[key]][p['label']] += 1
    return {k: dict(v) for k, v in sorted(result.items())}
