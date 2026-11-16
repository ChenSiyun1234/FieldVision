"""Corn the county step does not score: CSB polygons whose CDL class that year is a small grain followed by corn.

    python scripts/unscored_corn.py --check <labels>/us_24021_2022/county_check.json --out <counties>/24021/2022/unscored_corn.json

Reads the county check written by scripts/county_corn_check.py (run with --csb), which counts the
county's CSB polygons by their CDL class in the field year. Writes, by class, the polygons and acres
of the double crops "small grain then corn" (225 winter wheat/corn, 226 oats/corn, 228 triticale/corn,
237 barley/corn) beside the scored class-1 corn, plus other corn-containing classes (sweet corn,
popcorn, corn then soybeans) listed separately. Acres are polygon areas in the county's UTM zone.
Only boundaries are read; no imagery and no model.
"""
import argparse
import json
from pathlib import Path

SMALL_GRAIN_THEN_CORN = {225: 'Dbl Crop WinWht/Corn', 226: 'Dbl Crop Oats/Corn', 228: 'Dbl Crop Triticale/Corn',
                         237: 'Dbl Crop Barley/Corn'}
OTHER_CORN = {12: 'Sweet Corn', 13: 'Pop or Orn Corn', 241: 'Dbl Crop Corn/Soybeans'}


def summarise(check):
    by_code = {}
    for key, v in (check.get('csb_other_corn_classes') or {}).items():
        by_code[int(key.split()[0])] = v
    rows = lambda table: [{'class': k, 'name': n, 'polygons': int(by_code[k]['fields']), 'acres': round(by_code[k]['acres'], 1)}
                          for k, n in table.items() if k in by_code]
    small = rows(SMALL_GRAIN_THEN_CORN)
    total = round(sum(r['acres'] for r in small), 1)
    scored = round(check['csb_class1_acres_check'], 1)
    u, s = check.get('usda') or {}, check.get('survey') or {}
    return {
        'fips': check['fips'], 'year': check['year'], 'county_name': check.get('county_name'),
        'source': 'USDA NASS Crop Sequence Boundaries 2018-2025 rev23, polygons by CDL class of the field year',
        'scored_class1': {'class': 1, 'name': 'Corn', 'polygons': int(check['fields']), 'acres': scored},
        'small_grain_then_corn': small,
        'small_grain_then_corn_acres': total,
        'small_grain_then_corn_share_of_scored': round(total / scored, 4) if scored else None,
        'other_corn_containing': rows(OTHER_CORN),
        'census': {'year': check.get('census_year', check['year']), 'grain': u.get('grain_harvested_acres'),
                   'silage': u.get('silage_harvested_acres')},
        'survey': {'year': s.get('year'), 'planted': s.get('planted_acres'), 'grain': s.get('grain_harvested_acres'),
                   'silage': s.get('silage_harvested_acres')} if s else None,
        'note': ('Double-crop polygons are fields whose majority class that year was the double crop; how much of '
                 'their corn was cut for silage is unknown (an upper bound for silage outside the scored set). '
                 'Corn inside polygons whose majority is another crop is not counted here.'),
    }


def cdl_pixels(csv_path):
    """CropScape county statistics (GetCDLStat CSV: Value, Category, Count, Acreage): corn-containing classes."""
    out = {}
    for line in Path(csv_path).read_text(encoding='utf-8').splitlines()[1:]:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 4 and parts[0].isdigit():
            code = int(parts[0])
            if code == 1 or code in SMALL_GRAIN_THEN_CORN or code in OTHER_CORN:
                out[code] = {'name': parts[1], 'pixels': int(parts[2]), 'acres': float(parts[3])}
    small = round(sum(v['acres'] for k, v in out.items() if k in SMALL_GRAIN_THEN_CORN), 1)
    return {'source': 'CropScape GetCDLStat county statistics, 30 m CDL pixels', 'file': Path(csv_path).name,
            'corn_class1_acres': out.get(1, {}).get('acres'), 'small_grain_then_corn_acres': small,
            'by_class': {str(k): v for k, v in sorted(out.items())}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', type=Path, required=True)
    ap.add_argument('--cropscape', type=Path, help='CropScape county statistics CSV for the same year (optional)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    check = json.loads(a.check.read_text(encoding='utf-8'))
    if 'csb_other_corn_classes' not in check:
        raise SystemExit('county check was run without --csb; rerun scripts/county_corn_check.py with --csb')
    res = summarise(check)
    if a.cropscape:
        res['cdl_pixel_acres'] = cdl_pixels(a.cropscape)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('fips', 'year', 'scored_class1', 'small_grain_then_corn_acres',
                                          'small_grain_then_corn_share_of_scored')}))


if __name__ == '__main__':
    main()
