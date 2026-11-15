"""Merge the finished parts of a statewide sample into one toolbox-style folder.

    python scripts/sample_snapshot.py --sample-dir C:/t/fv/data/labels/us_24_2025 \\
        --runs C:/t/fv/data/timeseries/us_24_2025_s1 C:/t/fv/data/timeseries/us_24_2025_s2 C:/t/fv/data/timeseries/us_24_2025_s3 \\
        --out C:/t/fv/data/counties/24_2025_sample

Only parts whose run is finished (manifest complete, every block complete, coverage listing every
field) are taken. Writes:

* fields.geojson: every drawn field of the finished parts (both strata), with field_id, block_id
  (the drawn read block, the sampling unit), block10_id, county_fips, stratum, crop_code, area_ha ...;
* timeseries.parquet: rows of the corn stratum only (the fields that are scored);
* timeseries_dblcrop.parquet: rows of the small-grain/corn stratum, kept apart so nothing scores them;
* fields_coverage.csv, draw.json and state_acres.json (copied), and PARTIAL.json listing the parts included while any part
  is missing; PARTIAL.json is removed once all parts are in. sample_info.json always lists the parts and any
  permanent scene errors (plausibility rejections, missing assets) that a part was accepted with.
"""
import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def replace(tmp, path):
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5 * (attempt + 1))


def permanent(message):
    """Scene errors no rerun can fix (same rule as scripts/report_s2_run.py): plausibility rejections, missing assets."""
    return 'sanity check' in message or message.startswith('KeyError')


def finished(run, fields):
    """(ok, reason, permanent scene errors). Partial runs count when every scene error is permanent."""
    m = json.loads((run / 'manifest.json').read_text())
    if m.get('status') not in ('complete', 'partial') or not m.get('ended_at') or len(m.get('blocks', {})) != m.get('block_count'):
        return False, f"status {m.get('status')}, {len(m.get('blocks', {}))}/{m.get('block_count')} blocks", {}
    if m.get('invalid_fields'):
        return False, 'invalid fields', {}
    errors = {}
    for bid, b in m['blocks'].items():
        if b['status'] not in ('complete', 'partial'):
            return False, f'block {bid} is {b["status"]}', {}
        for scene, msg in b.get('scene_errors', {}).items():
            if not permanent(msg):
                return False, f'block {bid} has a retryable scene error: {msg[:80]}', {}
            errors[f'{bid} {scene}'] = msg
    cov = pd.read_csv(run / 'fields_coverage.csv')
    if set(cov.field_id) != {f['properties']['field_id'] for f in fields}:
        return False, 'coverage does not list exactly the part fields', {}
    return True, m['status'], errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample-dir', type=Path, required=True, help='folder with s<k>/fields.geojson and draw.json')
    ap.add_argument('--runs', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    draw = json.loads((a.sample_dir / 'draw.json').read_text())
    feats, tables, covs, used, missing, perm = [], [], [], [], {}, {}
    for run in a.runs:
        k = run.name.rsplit('_s', 1)[1]
        part = json.loads((a.sample_dir / f's{k}' / 'fields.geojson').read_text(encoding='utf-8'))['features']
        ok, why, errors = finished(run, part) if (run / 'manifest.json').exists() else (False, 'no manifest', {})
        if not ok:
            missing[run.name] = why
            continue
        if errors:
            perm[run.name] = errors
        feats += part
        tables.append(pd.read_parquet(run / 'timeseries.parquet'))
        covs.append(pd.read_csv(run / 'fields_coverage.csv'))
        used.append(run.name)
    if not used:
        raise SystemExit(f'no finished part yet: {missing}')
    feats.sort(key=lambda f: f['properties']['field_id'])
    ids = [f['properties']['field_id'] for f in feats]
    assert len(ids) == len(set(ids)), 'a field appears in two parts'
    stratum = {f['properties']['field_id']: f['properties']['stratum'] for f in feats}
    table = pd.concat(tables, ignore_index=True)
    assert set(table.field_id) <= set(stratum), 'time series rows for fields outside the parts'
    assert not table.duplicated(['field_id', 'date']).any()
    s = table.field_id.map(stratum)
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = a.out / 'fields.geojson.tmp'
    tmp.write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}, ensure_ascii=False), encoding='utf-8')
    replace(tmp, a.out / 'fields.geojson')
    for name, rows in (('timeseries.parquet', table[s == 'corn']), ('timeseries_dblcrop.parquet', table[s != 'corn'])):
        rows.to_parquet(a.out / (name + '.tmp'), index=False)
        replace(a.out / (name + '.tmp'), a.out / name)
    pd.concat(covs, ignore_index=True).to_csv(a.out / 'fields_coverage.csv', index=False)
    shutil.copyfile(a.sample_dir / 'draw.json', a.out / 'draw.json')
    shutil.copyfile(a.sample_dir / 'state_acres.json', a.out / 'state_acres.json')
    info = {'written_at': datetime.now(timezone.utc).isoformat(), 'parts_included': used, 'parts_missing': missing,
            'fields': len(feats), 'fields_corn': sum(v == 'corn' for v in stratum.values()),
            'fields_dblcrop': sum(v != 'corn' for v in stratum.values()),
            'read_blocks_included': sum(draw['parts'][u]['read_blocks'] for u in used),
            'read_blocks_drawn': draw['read_blocks_drawn'], 'read_blocks_total': draw['read_blocks_total'],
            'rows_corn': int((s == 'corn').sum()), 'rows_dblcrop': int((s != 'corn').sum()),
            'permanent_scene_errors': perm}
    (a.out / 'sample_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    if missing:
        (a.out / 'PARTIAL.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    else:
        (a.out / 'PARTIAL.json').unlink(missing_ok=True)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
