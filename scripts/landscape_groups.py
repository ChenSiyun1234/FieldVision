"""Distribution of the forage-landscape share by the rotation-check groups of called fields.

    python scripts/landscape_groups.py --results data/diag/dairy/results --counties data/counties \\
        --year 2022 --fips 24021 24035 24029 24013 --out landscape_groups.csv

Groups as in scripts/rotation_check.py (clear tier called silage with drop before day 252; clear tier called
silage later; clear tier called grain; everything else). For each county and group: fields, acres, and
the 25th/50th/75th percentiles and the acre-weighted mean of the 2 km and 5 km forage share (own year),
plus the share of fields with hay or alfalfa in their own history.
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rotation_check import ORDER, groups_for  # noqa: E402

ACRES_PER_HA = 2.471053814671653


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--counties', type=Path, required=True)
    ap.add_argument('--year', type=int, default=2022)
    ap.add_argument('--fips', nargs='+', required=True)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args(argv)
    rows = []
    for fips in a.fips:
        d = groups_for(pd.read_parquet(a.results / f'MD_{a.year}_{fips}_module_v2.parquet'))
        folder = a.counties / fips / str(a.year)
        land = pd.read_parquet(folder / 'landscape.parquet')
        area = gpd.read_file(folder / 'fields.geojson')[['field_id', 'area_ha']]
        d = d[['field_id', 'group']].merge(land, on='field_id', how='left').merge(area, on='field_id', how='left')
        assert d['forage_share_2km_year'].notna().all()
        d['acres'] = d['area_ha'] * ACRES_PER_HA
        for grp in ORDER + ['all fields']:
            s = d if grp == 'all fields' else d[d['group'] == grp]
            if s.empty:
                continue
            r = {'fips': fips, 'group': grp, 'fields': len(s), 'acres': round(s['acres'].sum())}
            for col in ('forage_share_2km_year', 'forage_share_5km_year', 'forage_share_5km_mean8'):
                q = s[col].quantile([.25, .5, .75]).to_numpy()
                r[f'{col}_p25'], r[f'{col}_p50'], r[f'{col}_p75'] = np.round(q, 3)
                r[f'{col}_wmean'] = round(float((s[col] * s['acres']).sum() / s['acres'].sum()), 3)
            r['own_hay_history'] = round(float(s['hay_or_alfalfa_history'].mean()), 3)
            rows.append(r)
    out = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(out[['fips', 'group', 'fields', 'acres', 'forage_share_2km_year_p25', 'forage_share_2km_year_p50',
               'forage_share_2km_year_p75', 'forage_share_5km_year_p25', 'forage_share_5km_year_p50',
               'forage_share_5km_year_p75', 'forage_share_5km_mean8_p50', 'own_hay_history']].to_string(index=False))
    if a.out:
        out.to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
