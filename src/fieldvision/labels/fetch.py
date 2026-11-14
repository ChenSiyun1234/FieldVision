"""Fetch and verify balanced maize labels from public parcel registries."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urljoin

import geopandas as gpd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from shapely.geometry import shape
from shapely.ops import unary_union

from .adapters import get_adapter
from .core import normalize, balance, counts


def now():
    return datetime.now(timezone.utc).isoformat()


class Client:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'FieldVision student research label fetcher'
        self.session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        self.log = []

    def get(self, url, params=None):
        started = now()
        r = self.session.get(url, params=params, timeout=(20, 90))
        r.raise_for_status()
        j = r.json()
        self.log.append({'url': r.url, 'fetched_at': started, 'bytes': len(r.content),
                         'numberMatched': j.get('numberMatched'), 'returned': len(j.get('features', []))})
        return j

    def pages(self, adapter, bbox):
        url, params, visited = adapter.url, adapter.query(bbox), set()
        while url:
            data = self.get(url, params)
            if self.log[-1]['url'] in visited:
                raise RuntimeError('Repeated pagination URL')
            visited.add(self.log[-1]['url'])
            yield data['features']
            if adapter.country == 'NL':
                if not data['features']:
                    break
                params['startIndex'] += len(data['features'])
                continue
            next_links = [l['href'] for l in data.get('links', []) if l.get('rel') == 'next']
            url = urljoin(url, next_links[0]) if next_links else None
            params = None


def country_boundary(client, country, out):
    url = f'https://gisco-services.ec.europa.eu/distribution/v2/countries/distribution/{country}-region-01m-4326-2024.geojson'
    data = client.get(url)
    (out / 'country_boundary.geojson').write_text(json.dumps(data), encoding='utf-8')
    if data['type'] == 'FeatureCollection':
        return unary_union([shape(f['geometry']) for f in data['features']])
    return shape(data['geometry'])


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--country', choices=['AT', 'NL', 'FR'], required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--target-per-class', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cap-per-class-block', type=int, default=60)
    args = parser.parse_args()
    if args.target_per_class < 3 or args.cap_per_class_block < 1:
        parser.error('Target must be at least 3 and cap must be positive')
    adapter = get_adapter(args.country, args.year)
    out = args.out.resolve()
    repo = Path(__file__).resolve().parents[3]
    if out == repo or repo in out.parents:
        parser.error('Data output must be outside the repository')
    out.mkdir(parents=True, exist_ok=True)
    client, started, counters = Client(), now(), Counter()
    boundary = country_boundary(client, args.country, out)
    candidates, seen = [], set()
    for region, bbox in adapter.regions:
        for page in client.pages(adapter, bbox):
            for raw in page:
                feature = normalize(raw, adapter, region, counters, boundary)
                if feature:
                    fid = feature['properties']['field_id']
                    if fid in seen:
                        counters['duplicates'] += 1
                    else:
                        seen.add(fid)
                        candidates.append(feature)
            if len(client.log) % 10 == 0:
                print(f'{region}: {len(candidates)} eligible candidates', flush=True)
        print(f'{region} complete: {len(candidates)} eligible candidates', flush=True)
        selected = balance(candidates, args.target_per_class, args.seed, args.cap_per_class_block, adapter.rare_label)
        if len(selected) >= 2 * args.target_per_class and len(counts(selected, 'region')) >= 3:
            break
    # Save only whitelisted properties, so the cache is safe to share as well.
    (out / 'candidates.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': candidates}, ensure_ascii=False), encoding='utf-8')
    selected = balance(candidates, args.target_per_class, args.seed, args.cap_per_class_block, adapter.rare_label)
    geojson = out / 'fields.geojson'
    geojson.write_text(json.dumps({'type': 'FeatureCollection', 'features': selected}, ensure_ascii=False), encoding='utf-8')
    frame = verify(geojson, boundary, args.country, args.year)
    frame.to_parquet(out / 'fields.parquet', index=False)
    reread = gpd.read_parquet(out / 'fields.parquet')
    assert list(reread.field_id) == list(frame.field_id) and reread.crs.to_epsg() == 4326
    quicklook(frame, boundary, out / 'quicklook.png')
    summary = {'country': args.country, 'year': args.year, 'fetch_started': started, 'fetch_finished': now(),
               'counts_per_class': frame.label.value_counts().to_dict(),
               'counts_per_block': counts(selected, 'block_id'), 'counts_per_region': counts(selected, 'region'),
               'area_ha': frame.groupby('label').area_ha.describe().to_dict(),
               'parameters': {**vars(args), 'out': str(out), 'grid_crs': 'EPSG:3035', 'grid_size_m': 10000,
                              'minimum_area_ha': 1, 'rare_label': adapter.rare_label, 'regions': adapter.regions},
               'filter_counts': dict(counters), 'eligible_candidates': len(candidates),
               'target_reached': len(selected) >= 2 * args.target_per_class,
               'verification': 'GeoJSON re-read: IDs, labels, balance, valid polygons, metric area, full country containment, three regions; GeoParquet re-read: IDs and CRS',
               'api_requests': client.log}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['counts_per_class', 'counts_per_region', 'target_reached']}), flush=True)


if __name__ == '__main__':
    main()
