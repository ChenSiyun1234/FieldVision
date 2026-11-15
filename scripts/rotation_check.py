"""Crop rotation of called fields, from the CSB class sequence (boundaries only, no imagery, no model).

    python scripts/rotation_check.py --results data/diag/dairy/results --csb data/us/csb/csb1825_md.parquet \\
        --labels data/labels --year 2022 --fips 24021 24035 24029 24013 --out rotation_check.csv

For each county, joins the per-field calls (MD_<year>_<fips>_module_v2.parquet: field_id, label, abstain,
evidence with evidence_tier and drop_date) to the CSB polygon's CDL class in every year of the release,
and reports by group (clear tier called silage with drop before day 252; clear tier called silage later
or without a drop date; clear tier called grain; everything else) the share of fields and of acres whose
OTHER years include each class group, and the mean number of the 8 years that are corn (class 1).
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

ACRES_PER_HA = 2.471053814671653
YEARS = range(2018, 2026)
GROUPS = {
    'alfalfa_or_hay': {36, 37},
    'sweet_corn': {12},
    'vegetable': {12, 42, 43, 46, 47, 48, 49, 50, 53, 54, 206, 207, 208, 209, 213, 214, 216, 219, 222, 227, 229,
                  230, 231, 232, 233, 243, 244, 245, 246, 247, 248, 249},
    'soybeans': {5, 26, 239, 240, 241, 254},
    'wheat_or_wheat_double': {22, 23, 24, 26, 225, 236, 238},
    'pasture_grass': {176},
}
ORDER = ['clear, silage, drop < day 252', 'clear, silage, later', 'clear, grain', 'everything else']


def groups_for(d):
    ev = pd.json_normalize(d['evidence'].map(lambda e: e if isinstance(e, dict) else json.loads(e)))
    d = pd.concat([d.drop(columns=['evidence']).reset_index(drop=True), ev], axis=1)
    doy = pd.to_datetime(d['drop_date'], errors='coerce').dt.day_of_year
    clear = (d['evidence_tier'] == 'clear') & ~d['abstain'].astype(bool)
    g = pd.Series(ORDER[3], index=d.index)
    g[clear & (d['label'] == 'grain')] = ORDER[2]
    g[clear & (d['label'] == 'silage')] = ORDER[1]
    g[clear & (d['label'] == 'silage') & (doy < 252)] = ORDER[0]
    return d.assign(group=g)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--csb', type=Path, required=True)
    ap.add_argument('--labels', type=Path, required=True)
    ap.add_argument('--year', type=int, default=2022)
    ap.add_argument('--fips', nargs='+', required=True)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args(argv)
    csb = pd.DataFrame(gpd.read_parquet(a.csb).drop(columns='geometry'))
    csb['field_id'] = 'csb-' + csb['CSBID'].astype(str)
    cols = [f'CDL{y}' for y in YEARS]
    others = [f'CDL{y}' for y in YEARS if y != a.year]
    rows = []
    for fips in a.fips:
        calls = groups_for(pd.read_parquet(a.results / f'MD_{a.year}_{fips}_module_v2.parquet'))
        fields = gpd.read_file(a.labels / f'us_{fips}_{a.year}' / 'fields.geojson')[['field_id', 'area_ha']]
        d = calls.merge(fields, on='field_id', how='left').merge(csb[['field_id'] + cols], on='field_id', how='left')
        assert d['area_ha'].notna().all() and d[cols[0]].notna().all(), f'{fips}: fields missing from the CSB or field file'
        d['acres'] = d['area_ha'] * ACRES_PER_HA
        for name, codes in GROUPS.items():
            d[name] = d[others].isin(codes).any(axis=1)
        d['corn_years'] = (d[cols] == 1).sum(axis=1)
        for grp in ORDER:
            s = d[d['group'] == grp]
            if s.empty:
                continue
            r = {'fips': fips, 'group': grp, 'fields': len(s), 'acres': round(s['acres'].sum())}
            for name in GROUPS:
                r[f'{name}_fields'] = round(s[name].mean(), 3)
                r[f'{name}_acres'] = round((s['acres'] * s[name]).sum() / s['acres'].sum(), 3)
            r['corn_years_mean'] = round(s['corn_years'].mean(), 2)
            r['sweet_corn_any_year_fields'] = int(s['sweet_corn'].sum())
            rows.append(r)
    out = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    show = ['fips', 'group', 'fields', 'acres'] + [f'{n}_acres' for n in GROUPS] + ['corn_years_mean', 'sweet_corn_any_year_fields']
    print(out[show].to_string(index=False))
    if a.out:
        out.to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
