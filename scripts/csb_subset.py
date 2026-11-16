"""Cut one state out of the national USDA NASS Crop Sequence Boundaries archive.

    python scripts/csb_subset.py --zip NationalCSB_2018-2025_rev23.zip --state 24 --out csb1825_md.parquet

Reads the file geodatabase straight from the zip (no extraction), checks it against the
published metadata (layer name, attribute names, Albers CRS of the NLCD), optionally checks
the zip CRC of the big table, and writes every polygon of the state with all attributes to
GeoParquet in the source CRS. A JSON report is written next to the output.

Metadata: https://www.nass.usda.gov/research-and-science/crop-sequence-boundaries/metadata-crop-sequence-boundaries-2025
CDL class 1 is corn (data dictionary in the same metadata).
"""
import argparse
import json
import time
import zipfile
from pathlib import Path

import pyogrio
from pyproj import CRS

EXPECTED_FIELDS = ['CSBID', 'CSBACRES', 'CDL2018', 'CDL2019', 'CDL2020', 'CDL2021', 'CDL2022', 'CDL2023',
                   'CDL2024', 'CDL2025', 'STATEFIPS', 'STATEASD', 'ASD', 'CNTY', 'CNTYFIPS', 'INSIDE_X', 'INSIDE_Y']
# "Albers Conical Equal Area as used by mrlc.gov (NLCD)", NAD83, standard parallels 29.5/45.5, origin 23N 96W
NLCD_ALBERS = CRS.from_proj4('+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs')


def check_crc(zip_path, members):
    """Read members completely; zipfile raises BadZipFile on a CRC mismatch."""
    with zipfile.ZipFile(zip_path) as z:
        for name in members:
            with z.open(name) as f:
                while f.read(1 << 24):
                    pass
    return members


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zip', type=Path, required=True)
    ap.add_argument('--gdb', default='CSB1825.gdb')
    ap.add_argument('--layer', default='national1825')
    ap.add_argument('--state', required=True, help='two-digit state FIPS, e.g. 24 for Maryland')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--skip-crc', action='store_true')
    a = ap.parse_args(argv)
    src = f'/vsizip/{a.zip.resolve().as_posix()}/{a.gdb}'
    t0 = time.time()
    layers = [l[0] for l in pyogrio.list_layers(src)]
    if a.layer not in layers:
        raise SystemExit(f'layer {a.layer} not in {layers}')
    info = pyogrio.read_info(src, layer=a.layer)
    fields = dict(zip(info['fields'], info['dtypes']))
    missing = [f for f in EXPECTED_FIELDS if f not in fields]
    if missing:
        raise SystemExit(f'attributes missing compared with the metadata: {missing}; found {list(fields)}')
    crs = CRS.from_user_input(info['crs'])
    if not crs.equals(NLCD_ALBERS, ignore_axis_order=True):
        # compare parameters rather than names: the gdb may call it by an ESRI name
        same = crs.to_proj4() == NLCD_ALBERS.to_proj4()
        if not same:
            raise SystemExit(f'CRS differs from the metadata: {crs.to_wkt()}')
    state = a.state.zfill(2)
    where = f"STATEFIPS = '{state}'" if fields['STATEFIPS'] == 'object' else f'STATEFIPS = {int(state)}'
    frame = pyogrio.read_dataframe(src, layer=a.layer, where=where)
    read_s = time.time() - t0
    if frame.empty:
        raise SystemExit(f'no rows for {where}')
    ids = frame['CSBID'].astype(str)
    assert ids.is_unique, 'CSBID not unique within the state'
    assert ids.str[:2].eq(state).all(), 'CSBID prefix does not match the state FIPS'
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(a.out, index=False)
    crc = [] if a.skip_crc else check_crc(a.zip, [f'{a.gdb}/a00000010.gdbtable', f'{a.gdb}/a00000010.gdbtablx'])
    report = {
        'zip': str(a.zip), 'zip_bytes': a.zip.stat().st_size, 'layer': a.layer, 'layers': layers,
        'national_features': info['features'], 'fields': fields, 'crs_wkt': crs.to_wkt(),
        'crs_matches_metadata_albers': True, 'where': where, 'rows': int(len(frame)),
        'counties': frame.groupby(['CNTYFIPS', 'CNTY']).size().reset_index().values.tolist(),
        'cdl2022_corn_rows': int((frame['CDL2022'].astype(str) == '1').sum()),
        'crc_checked': crc, 'read_seconds': round(read_s, 1), 'out': str(a.out),
    }
    a.out.with_suffix('.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('layer', 'national_features', 'where', 'rows', 'cdl2022_corn_rows', 'read_seconds')}))


if __name__ == '__main__':
    main()
