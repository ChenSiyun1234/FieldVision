"""Rank candidate test counties by the rule in docs/us-county-fields.md.

    python scripts/pick_test_county.py --quickstats corn_grain_silage_acres_2012_2025.csv [--state 24 --year 2022]

Census county rows after fieldvision.model.quickstats.select; keeps counties with grain and silage
both published, grain + silage >= --min-acres, silage share >= --min-share and outside
--avoid-share +/- --avoid-band, minus --exclude; sorted by grain + silage acres, largest first.
"""
import argparse

from fieldvision.model import quickstats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quickstats', required=True)
    ap.add_argument('--state', default='24')
    ap.add_argument('--year', type=int, default=2022)
    ap.add_argument('--min-acres', type=float, default=10000)
    ap.add_argument('--min-share', type=float, default=0.03)
    ap.add_argument('--avoid-share', type=float, default=0.196, help="Frederick's share")
    ap.add_argument('--avoid-band', type=float, default=0.02)
    ap.add_argument('--exclude', nargs='*', default=['24021', '24035'])
    a = ap.parse_args(argv)
    sel = quickstats.select(quickstats.load(a.quickstats))
    s = quickstats.shares(sel, 'COUNTY', a.year)
    s = s[s.fips.str[:2] == a.state.zfill(2)].copy()
    s['total'] = s.grain_harvested + s.silage_harvested
    s['published'] = s.grain_harvested.notna() & s.silage_harvested.notna()
    s['passes'] = (s.published & (s.total >= a.min_acres) & (s.silage_share >= a.min_share)
                   & ((s.silage_share - a.avoid_share).abs() > a.avoid_band) & ~s.fips.isin(a.exclude))
    out = s.sort_values(['passes', 'total'], ascending=[False, False])
    cols = ['fips', 'county_name', 'grain_harvested', 'silage_harvested', 'total', 'silage_share', 'published', 'passes']
    print(out[cols].to_string(index=False, float_format=lambda v: f'{v:,.3f}' if v < 1 else f'{v:,.0f}'))


if __name__ == '__main__':
    main()
