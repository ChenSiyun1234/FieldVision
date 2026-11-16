"""Copy a county's fields and Sentinel-2 time series into the toolbox layout <root>/<fips>/<year>/.

    python scripts/county_snapshot.py --fields C:/t/fv/data/labels/us_24021_2022/fields.geojson \\
        --run C:/t/fv/data/timeseries/us_24021_2022 --root C:/t/fv/data/counties --fips 24021 --year 2022

While the extractor is still running this writes a partial snapshot: only fields whose block is
listed as complete in the run's manifest.json, read from those blocks' parquet files (which the
extractor no longer touches), plus PARTIAL.json with the counts. The status files of the block in
progress are never opened. Once the run has finished (manifest status complete, or partial only
because of permanent scene errors) the run's own timeseries.parquet is copied instead and
PARTIAL.json is removed. Files are written to a temporary name and then renamed.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import geopandas as gpd
import pandas as pd

from fieldvision.s2.core import resolve_duplicates

ACRES_PER_HA = 2.471053814671653


def read_json(path, attempts=10):
    """The extractor replaces manifest.json atomically; retry if a read races a replace."""
    for attempt in range(attempts):
        try:
            return json.loads(Path(path).read_bytes())
        except (PermissionError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(0.5)


def replace(tmp, path):
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5 * (attempt + 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fields', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--fips', required=True)
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--final', action='store_true', help='require a finished run and write the final files')
    a = ap.parse_args(argv)

    manifest = read_json(a.run / 'manifest.json')
    blocks = manifest['blocks']
    fields = gpd.read_file(a.fields)
    finished = manifest['status'] in ('complete', 'partial') and manifest.get('ended_at')
    if a.final:
        if not finished or len(blocks) != manifest['block_count'] or any(b['status'] == 'failed' for b in blocks.values()):
            raise SystemExit('run not finished (or has failed blocks); rerun the extractor first')
        table = pd.read_parquet(a.run / 'timeseries.parquet')
        done_ids = set(fields.field_id)
        done_blocks = sorted(blocks)
    else:
        done_blocks = sorted(b for b, s in blocks.items() if s['status'] == 'complete')
        if not done_blocks:
            raise SystemExit('no complete block yet')
        table = resolve_duplicates(pd.concat([pd.read_parquet(a.run / 'blocks' / f'{b}.parquet') for b in done_blocks],
                                             ignore_index=True))
        done_ids = {f for b in done_blocks for f in blocks[b]['field_ids']}
    part = fields[fields.field_id.isin(done_ids)].sort_values('field_id')
    table = table[table.field_id.isin(done_ids)]
    assert set(table.field_id) <= set(part.field_id)

    out = a.root / a.fips / str(a.year)
    out.mkdir(parents=True, exist_ok=True)
    part.to_file(out / 'fields.geojson.tmp', driver='GeoJSON')
    replace(out / 'fields.geojson.tmp', out / 'fields.geojson')
    table.to_parquet(out / 'timeseries.parquet.tmp', index=False)
    replace(out / 'timeseries.parquet.tmp', out / 'timeseries.parquet')
    info = {
        'written_at': datetime.now(timezone.utc).isoformat(), 'run': str(a.run), 'fields_file': str(a.fields),
        'fields_included': int(len(part)), 'fields_total': int(len(fields)),
        'acres_included': round(float(part.area_ha.sum() * ACRES_PER_HA), 1),
        'acres_total': round(float(fields.area_ha.sum() * ACRES_PER_HA), 1),
        'blocks_included': len(done_blocks), 'blocks_total': manifest['block_count'],
        'fields_with_rows': int(table.field_id.nunique()), 'rows': int(len(table)),
        'blocks': done_blocks,
    }
    if a.final:
        coverage = pd.read_csv(a.run / 'fields_coverage.csv')
        assert set(coverage.field_id) == set(fields.field_id), 'coverage does not list every field'
        coverage.to_csv(out / 'fields_coverage.csv.tmp', index=False)
        replace(out / 'fields_coverage.csv.tmp', out / 'fields_coverage.csv')
        (out / 'PARTIAL.json').unlink(missing_ok=True)
        info.pop('blocks')
        info['run_status'] = manifest['status']
    else:
        (out / 'PARTIAL.json.tmp').write_text(json.dumps(info, indent=2), encoding='utf-8')
        replace(out / 'PARTIAL.json.tmp', out / 'PARTIAL.json')
    print(json.dumps({k: v for k, v in info.items() if k != 'blocks'}))


if __name__ == '__main__':
    main()
