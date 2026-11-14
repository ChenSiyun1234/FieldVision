"""Fetch and verify balanced maize labels from public parcel registries."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time

import geopandas as gpd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union

from .adapters import get_adapter
from .core import normalize, balance, counts, block_bbox


def now():
    return datetime.now(timezone.utc).isoformat()


class Client:
    """HTTP client with an on-disk page cache, so a stopped run can resume."""

    def __init__(self, cache_dir=None):
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'FieldVision student research label fetcher'
        self.session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504]), pool_maxsize=16))
        self.log = []
        self.lock = threading.Lock()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, url, params=None, adapter=None):
        """GET JSON. With `adapter`, raw properties outside adapter.keep are dropped before caching."""
        prepared = requests.Request('GET', url, params=params).prepare()
        path = None
        if self.cache_dir:
            path = self.cache_dir / (hashlib.sha1(prepared.url.encode()).hexdigest() + '.json')
            if path.exists():
                return json.loads(path.read_text(encoding='utf-8'))
        started = now()
        for attempt in range(3):
            try:
                r = self.session.send(prepared, timeout=(20, 180))
                r.raise_for_status()
                j = r.json()
                break
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        if adapter is not None:
            for f in j.get('features', []):
                f['properties'] = {k: v for k, v in f['properties'].items() if k in adapter.keep}
            j = {'numberMatched': j.get('numberMatched'), 'features': j.get('features', [])}
        with self.lock:
            self.log.append({'url': prepared.url[:400], 'fetched_at': started, 'bytes': len(r.content),
                             'numberMatched': j.get('numberMatched'), 'returned': len(j.get('features', []))})
        if path:
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(j, ensure_ascii=False), encoding='utf-8')
            tmp.replace(path)
        return j

    def fetch_all(self, adapter, bbox, labels):
        """Every parcel of the given classes in a lon/lat box. Pages after the first run in parallel."""
        first = self.get(adapter.url, adapter.query(bbox, labels, 0), adapter=adapter)
        features = list(first['features'])
        matched = first.get('numberMatched')
        size = len(first['features'])
        if not size:
            return features
        if isinstance(matched, int):
            starts = list(range(size, matched, size)) if size >= adapter.page_size else []
            with ThreadPoolExecutor(adapter.workers) as pool:
                for page in pool.map(lambda st: self.get(adapter.url, adapter.query(bbox, labels, st), adapter=adapter), starts):
                    features.extend(page['features'])
            if len(features) != matched:
                raise RuntimeError(f'Expected {matched} parcels, got {len(features)} for {bbox}')
        else:
            start = size
            while size >= adapter.page_size:
                page = self.get(adapter.url, adapter.query(bbox, labels, start), adapter=adapter)
                size = len(page['features'])
                features.extend(page['features'])
                start += size
        return features


def country_boundary(client, country, out):
    url = f'https://gisco-services.ec.europa.eu/distribution/v2/countries/distribution/{country}-region-01m-4326-2024.geojson'
    data = client.get(url)
    (out / 'country_boundary.geojson').write_text(json.dumps(data), encoding='utf-8')
    if data['type'] == 'FeatureCollection':
        geom = unary_union([shape(f['geometry']) for f in data['features']])
    else:
        geom = shape(data['geometry'])
    shapely.prepare(geom)
    return geom


def verify(path, boundary, country, year):
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['type'] == 'FeatureCollection' and data['features']
    frame = gpd.GeoDataFrame.from_features(data['features'], crs=4326)
    assert frame.field_id.is_unique
    assert set(frame.label) == {'silage', 'grain'}
    assert set(frame.country) == {country} and set(frame.year) == {year}
    assert frame.geometry.is_valid.all() and (~frame.geometry.is_empty).all()
    assert frame.geometry.map(boundary.covers).all(), 'Geometry outside country polygon'
    area = frame.to_crs(3035).area / 10000
    assert (area >= 1).all() and np.allclose(area, frame.area_ha)
    table = frame.groupby(['block_id', 'label']).size().unstack(fill_value=0)
    assert (table.silage == table.grain).all()
    assert frame.region.nunique() >= 3, 'Need at least three regions'
    return frame


def quicklook(frame, boundary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    regions = sorted(frame.region.unique())
    fig, axes = plt.subplots(1, len(regions) + 1, figsize=(5 * (len(regions) + 1), 6))
    outline = gpd.GeoSeries([boundary], crs=4326).to_crs(3035)
    colors = {'silage': '#008837', 'grain': '#d95f02'}
    for ax, region in zip(axes, [None] + regions):
        outline.plot(ax=ax, color='#f2f0e5', edgecolor='#777777', linewidth=.5)
        sub = frame if region is None else frame[frame.region == region]
        metric = sub.to_crs(3035)
        metric.plot(ax=ax, color=sub.label.map(colors), linewidth=.15, edgecolor=sub.label.map(colors))
        if region is None:
            centers = metric.geometry.centroid
            ax.scatter(centers.x, centers.y, c=sub.label.map(colors), s=3, alpha=.7)
        else:
            x0, y0, x1, y1 = metric.total_bounds
            ax.set_xlim(x0 - 5000, x1 + 5000)
            ax.set_ylim(y0 - 5000, y1 + 5000)
        ax.set_title(region or f'{frame.country.iloc[0]} {frame.year.iloc[0]}')
        ax.set_facecolor('#dcecf4')
        ax.ticklabel_format(style='plain')
        ax.tick_params(labelsize=7)
    axes[0].legend(handles=[Patch(color=c, label=l) for l, c in colors.items()], loc='lower left')
    fig.suptitle('Declared maize use | EPSG:3035 metres | country outline: Eurostat GISCO 2024')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def normalize_all(raws, adapter, region, counters, boundary, only_block=None):
    out = []
    for raw in raws:
        f = normalize(raw, adapter, region, counters, boundary)
        if f and (only_block is None or f['properties']['block_id'] == only_block):
            out.append(f)
    return out


def rank_blocks(client, adapter, boundary, counters):
    """Window phase: fetch the rarer class of each window and count it per 10 km block."""
    per_region = defaultdict(list)
    seen = set()
    for region, bbox, rare in adapter.windows:
        feats = normalize_all(client.fetch_all(adapter, bbox, [rare]), adapter, region, counters, boundary)
        per_block = Counter(f['properties']['block_id'] for f in feats)
        print(f'{region}: {len(feats)} qualifying {rare} parcels in {len(per_block)} blocks', flush=True)
        for block, n in per_block.items():
            if block not in seen:
                seen.add(block)
                per_region[region].append((-n, block))
    for blocks in per_region.values():
        blocks.sort()
    return per_region


def rotation(per_region):
    """Blocks in the order they should be tried: the best block of each region in turn."""
    queues = {r: list(b) for r, b in sorted(per_region.items())}
    order = []
    while any(queues.values()):
        for region in sorted(queues):
            if queues[region]:
                order.append((region, queues[region].pop(0)[1]))
    return order


def fetch_blocks(client, adapter, blocks, boundary, counters):
    """Block phase: every qualifying parcel of both classes inside each block."""
    def one(item):
        region, block = item
        local = Counter()
        raws = client.fetch_all(adapter, block_bbox(block), ['silage', 'grain'])
        return normalize_all(raws, adapter, region, local, boundary, only_block=block), local
    result = []
    with ThreadPoolExecutor(2) as pool:
        for feats, local in pool.map(one, blocks):
            counters.update(local)
            result.extend(feats)
    return result


def collect(client, adapter, boundary, args, window_counters, block_counters):
    """Return (candidates, block priority or None, number of blocks fetched)."""
    candidates, seen = [], set()

    def add(feats):
        for f in feats:
            fid = f['properties']['field_id']
            if fid in seen:
                block_counters['duplicates'] += 1
            else:
                seen.add(fid)
                candidates.append(f)

    if args.blocks_from:
        prior = json.loads(args.blocks_from.read_text(encoding='utf-8'))['features']
        info = {}
        for f in prior:
            p = f['properties']
            info.setdefault(p['block_id'], [p['region'], 0])[1] += 1
        per_region = defaultdict(list)
        for block, (region, n) in info.items():
            per_region[region].append((-n, block))
        for blocks in per_region.values():
            blocks.sort()
        order = rotation(per_region)
        priority = {block: i for i, (_, block) in enumerate(order)}
        print(f'Reusing {len(order)} blocks from {args.blocks_from}', flush=True)
        add(fetch_blocks(client, adapter, order, boundary, block_counters))
        return candidates, priority, len(order)
    order = rotation(rank_blocks(client, adapter, boundary, window_counters))
    done = 0
    while done < len(order):
        selected = balance(candidates, args.target_per_class, args.seed, args.cap_per_class_block, args.min_per_class_block)
        have = len(selected) // 2
        if have >= args.target_per_class and len({f['properties']['region'] for f in selected}) >= 3:
            break
        batch = max(4, -(-(args.target_per_class - have) // max(args.cap_per_class_block, 1)) + 2)
        chunk = order[done:done + batch]
        done += len(chunk)
        add(fetch_blocks(client, adapter, chunk, boundary, block_counters))
        print(f'{done} blocks fetched, {len(candidates)} candidates', flush=True)
    return candidates, None, done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--country', choices=['AT', 'NL', 'FR'], required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--target-per-class', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cap-per-class-block', type=int, default=60)
    parser.add_argument('--min-per-class-block', type=int, default=5)
    parser.add_argument('--blocks-from', type=Path, default=None,
                        help='fields.geojson of another year: reuse its blocks instead of ranking new ones')
    parser.add_argument('--cache-dir', type=Path, default=None)
    args = parser.parse_args()
    if args.target_per_class < 3 or args.cap_per_class_block < 1:
        parser.error('Target must be at least 3 and cap must be positive')
    adapter = get_adapter(args.country, args.year)
    out = args.out.resolve()
    repo = Path(__file__).resolve().parents[3]
    if out == repo or repo in out.parents:
        parser.error('Data output must be outside the repository')
    out.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or out.parent / '_cache' / out.name
    client, started = Client(cache), now()
    window_counters, block_counters = Counter(), Counter()
    boundary = country_boundary(client, args.country, out)
    candidates, priority, fetched = collect(client, adapter, boundary, args, window_counters, block_counters)
    selected = balance(candidates, args.target_per_class, args.seed, args.cap_per_class_block,
                       args.min_per_class_block, priority)
    (out / 'candidates.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': candidates}, ensure_ascii=False), encoding='utf-8')
    geojson = out / 'fields.geojson'
    geojson.write_text(json.dumps({'type': 'FeatureCollection', 'features': selected}, ensure_ascii=False), encoding='utf-8')
    frame = verify(geojson, boundary, args.country, args.year)
    frame.to_parquet(out / 'fields.parquet', index=False)
    reread = gpd.read_parquet(out / 'fields.parquet')
    assert list(reread.field_id) == list(frame.field_id) and reread.crs.to_epsg() == 4326
    quicklook(frame, boundary, out / 'quicklook.png')
    summary = {'country': args.country, 'year': args.year, 'fetch_started': started, 'fetch_finished': now(),
               'source_release': adapter.info.get('release'), 'source_url': adapter.source,
               'counts_per_class': frame.label.value_counts().to_dict(),
               'blocks': int(frame.block_id.nunique()),
               'counts_per_block': counts(selected, 'block_id'), 'counts_per_region': counts(selected, 'region'),
               'area_ha': frame.groupby('label').area_ha.describe().to_dict(),
               'parameters': {**{k: str(v) for k, v in vars(args).items()}, 'out': str(out), 'grid_crs': 'EPSG:3035',
                              'grid_size_m': 10000, 'minimum_area_ha': 1, 'regions': adapter.windows},
               'ranking_filter_counts': dict(window_counters), 'filter_counts': dict(block_counters),
               'eligible_candidates': len(candidates), 'blocks_fetched': fetched,
               'target_reached': len(selected) >= 2 * args.target_per_class,
               'verification': 'GeoJSON re-read: IDs, labels, balance, valid polygons, metric area, full country containment, three regions; GeoParquet re-read: IDs and CRS',
               'api_requests': client.log}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['counts_per_class', 'blocks', 'counts_per_region', 'target_reached']}), flush=True)


if __name__ == '__main__':
    main()
