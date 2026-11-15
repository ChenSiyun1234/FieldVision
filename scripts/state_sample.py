"""Statewide sample of whole read blocks from a corn frame, split into equal parts for parallel extraction.

    python scripts/state_sample.py --frame data/labels/us_24_2025/frame/fields.geojson --state 24 --year 2025 \\
        --blocks 150 --parts 3 --seed 42 --out data/labels/us_24_2025 [--target-csv nass_2025_state_targets.csv]

The frame is a from_file field file of every CSB polygon whose CDL class that year is corn (1) or a
small grain followed by corn (225, 226, 228, 237), with cnty_fips kept. This script

* adds `county_fips` (state + county) and `stratum` ("corn" for class 1, "dblcrop_sg_corn" for the
  double crops) to every field and writes frame_fields.geojson;
* forms the extractor's read blocks (fieldvision.s2.geometry.make_blocks, same block size),
  draws --blocks of them by simple random sampling without replacement (numpy default_rng(seed)),
  and splits the drawn blocks into --parts equal parts in draw order;
* writes s<k>/fields.geojson per part, where each field's block_id is its read block's id (the
  10 km id stays in block10_id), so the extractor forms exactly the drawn blocks;
* writes draw.json (seed, blocks drawn of how many, inclusion probability, acres and fields in the
  frame and in the draw by stratum and county, block ids per part) and state_acres.json (frame acres
  by CDL class and by county, from boundaries alone, beside the NASS target if given).

Every read block has the same inclusion probability n/N, so a state total is estimated as
(N / n) x the sum over drawn blocks; the acre ratio in draw.json is a check, not the weight.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fieldvision.s2.geometry import load_fields, make_blocks  # noqa: E402

ACRES_PER_HA = 2.471053814671653
DOUBLE = {225: 'Dbl Crop WinWht/Corn', 226: 'Dbl Crop Oats/Corn', 228: 'Dbl Crop Triticale/Corn', 237: 'Dbl Crop Barley/Corn'}


def acres_by(features, key):
    out = defaultdict(lambda: {'fields': 0, 'acres': 0.0})
    for f in features:
        p = f['properties']
        k = key(p)
        out[k]['fields'] += 1
        out[k]['acres'] += p['area_ha'] * ACRES_PER_HA
    return {k: {'fields': v['fields'], 'acres': round(v['acres'], 1)} for k, v in sorted(out.items())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frame', type=Path, required=True)
    ap.add_argument('--state', required=True)
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--blocks', type=int, default=150)
    ap.add_argument('--parts', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--block-size', type=float, default=6000)
    ap.add_argument('--target-csv', type=Path)
    ap.add_argument('--state-name')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    if a.blocks % a.parts:
        raise SystemExit('--blocks must divide evenly into --parts')

    frame = json.loads(a.frame.read_text(encoding='utf-8'))
    for f in frame['features']:
        p = f['properties']
        code = int(p['crop_code'])
        if code != 1 and code not in DOUBLE:
            raise SystemExit(f"unexpected CDL class {code} in the frame")
        p['county_fips'] = a.state.zfill(2) + str(p['cnty_fips']).zfill(3)
        p['stratum'] = 'corn' if code == 1 else 'dblcrop_sg_corn'
        p['crop_name'] = 'Corn' if code == 1 else DOUBLE[code]
        p['block10_id'] = p['block_id']
    a.out.mkdir(parents=True, exist_ok=True)
    frame_path = a.out / 'frame_fields.geojson'
    frame_path.write_text(json.dumps(frame, ensure_ascii=False), encoding='utf-8')
    by_id = {f['properties']['field_id']: f for f in frame['features']}

    blocks = make_blocks(load_fields(frame_path, a.year), a.block_size)
    n_total = len(blocks)
    if a.blocks > n_total:
        raise SystemExit(f'only {n_total} read blocks in the frame')
    order = np.random.default_rng(a.seed).permutation(n_total)[:a.blocks]
    drawn = [blocks[i] for i in order]
    parts = [drawn[k::a.parts] for k in range(a.parts)]

    run_base = f'us_{a.state.zfill(2)}_{a.year}'
    part_info = {}
    drawn_features = []
    for k, part in enumerate(parts, 1):
        feats = []
        for blk in part:
            for fld in blk['fields']:
                f = json.loads(json.dumps(by_id[fld['field_id']]))
                f['properties']['block_id'] = blk['id']
                feats.append(f)
        feats.sort(key=lambda f: f['properties']['field_id'])
        d = a.out / f's{k}'
        d.mkdir(exist_ok=True)
        path = d / 'fields.geojson'
        path.write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}, ensure_ascii=False), encoding='utf-8')
        check = make_blocks(load_fields(path, a.year), a.block_size)
        assert sorted(len(b['fields']) for b in check) == sorted(len(b['fields']) for b in part), f'part {k}: blocks re-formed differently'
        assert len(check) == len(part)
        part_info[f'{run_base}_s{k}'] = {'fields_file': str(path), 'read_blocks': len(part), 'fields': len(feats),
                                         'acres': round(sum(f['properties']['area_ha'] for f in feats) * ACRES_PER_HA, 1),
                                         'block_ids': [b['id'] for b in part]}
        drawn_features += feats

    fr, dr = frame['features'], drawn_features
    draw = {
        'frame': str(frame_path), 'year': a.year, 'state_fips': a.state.zfill(2),
        'rule': ('simple random sample without replacement of the extractor read blocks (make_blocks, block size '
                 f'{int(a.block_size)} m, grouped by 10 km block) in numpy default_rng(seed).permutation order; '
                 'parts are drawn blocks k, k+parts, ... in draw order'),
        'seed': a.seed, 'read_blocks_total': n_total, 'read_blocks_drawn': a.blocks,
        'inclusion_probability': round(a.blocks / n_total, 6), 'expansion_factor': round(n_total / a.blocks, 6),
        'frame_by_stratum': acres_by(fr, lambda p: p['stratum']), 'draw_by_stratum': acres_by(dr, lambda p: p['stratum']),
        'frame_by_county': acres_by(fr, lambda p: p['county_fips']), 'draw_by_county': acres_by(dr, lambda p: p['county_fips']),
        'frame_fields': len(fr), 'drawn_fields': len(dr),
        'frame_acres': round(sum(f['properties']['area_ha'] for f in fr) * ACRES_PER_HA, 1),
        'drawn_acres': round(sum(f['properties']['area_ha'] for f in dr) * ACRES_PER_HA, 1),
        'parts': part_info,
    }
    draw['drawn_acre_fraction'] = round(draw['drawn_acres'] / draw['frame_acres'], 4)
    (a.out / 'draw.json').write_text(json.dumps(draw, indent=2), encoding='utf-8')

    names = {}
    for f in fr:
        names[f['properties']['county_fips']] = f['properties'].get('county_name')
    state_acres = {
        'source': ('CSB 2018-2025 rev23 polygons by their CDL class of the year; acres = area_ha of the frame '
                   '(polygon area in the metric CRS given to from_file --metric-crs)'),
        'year': a.year, 'state_fips': a.state.zfill(2),
        'by_class': acres_by(fr, lambda p: f"{p['crop_code']} {p['crop_name']}"),
        'by_county': {k: dict(v, name=names.get(k)) for k, v in acres_by(fr, lambda p: p['county_fips']).items()},
        'by_county_and_stratum': acres_by(fr, lambda p: f"{p['county_fips']} {p['stratum']}"),
        'corn_class1_acres': draw['frame_by_stratum'].get('corn', {}).get('acres'),
        'small_grain_then_corn_acres': draw['frame_by_stratum'].get('dblcrop_sg_corn', {}).get('acres'),
    }
    if a.target_csv and a.state_name:
        for line in a.target_csv.read_text(encoding='utf-8').splitlines():
            parts_ = [x.strip() for x in line.split(',')]
            if parts_ and parts_[0].upper() == a.state_name.upper() and len(parts_) >= 3:
                g, s = float(parts_[1]), float(parts_[2])
                state_acres['nass_target'] = {'grain': g, 'silage': s, 'silage_share': round(s / (g + s), 4)}
                state_acres['class1_over_target_total'] = round(state_acres['corn_class1_acres'] / (g + s), 4)
    (a.out / 'state_acres.json').write_text(json.dumps(state_acres, indent=2), encoding='utf-8')
    print(json.dumps({k: draw[k] for k in ('read_blocks_total', 'read_blocks_drawn', 'frame_fields', 'drawn_fields',
                                           'frame_acres', 'drawn_acres', 'drawn_acre_fraction', 'frame_by_stratum',
                                           'draw_by_stratum')}, indent=1))
    print({k: (v['read_blocks'], v['fields'], v['acres']) for k, v in part_info.items()})
    print('target', state_acres.get('nass_target'), 'class1/target', state_acres.get('class1_over_target_total'))


if __name__ == '__main__':
    main()
