"""Scoping for county checks outside Maryland: states and two counties per state, by rule.

    python scripts/scope_states.py --quickstats corn_grain_silage_acres_2012_2025.csv [--year 2022]

State totals for a Census year are sums of the county Census rows, including the
"OTHER (COMBINED) COUNTIES" rows; withheld county values are missing, so totals are lower bounds.
States: Iowa plus the --n-states states with the most silage acres among those with at least
--min-state-grain grain acres. Counties per state: grain and silage both published, grain + silage
>= --min-acres; the eligible counties are split into terciles of silage share; "high" is the
largest-corn county in the top tercile, "low" the largest-corn county in the bottom tercile.
"""
import argparse

import pandas as pd

from fieldvision.model import quickstats


def state_totals(raw, year):
    f = raw[(raw.aggregation_level == 'COUNTY') & (raw.source == 'CENSUS') & (raw.year == year)
            & (raw.domain == 'TOTAL') & (raw.reference_period == 'YEAR')
            & raw.short_description.isin(['CORN, GRAIN - ACRES HARVESTED', 'CORN, SILAGE - ACRES HARVESTED'])]
    g = f.groupby(['state_name', 'crop_use']).agg(acres=('value_num', 'sum'), withheld=('suppressed', 'sum')).unstack('crop_use')
    out = pd.DataFrame({'grain': g[('acres', 'grain')], 'silage': g[('acres', 'silage')],
                        'grain_withheld': g[('withheld', 'grain')], 'silage_withheld': g[('withheld', 'silage')]}).fillna(0)
    out['silage_share'] = out.silage / (out.grain + out.silage)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quickstats', required=True)
    ap.add_argument('--year', type=int, default=2022)
    ap.add_argument('--n-states', type=int, default=5)
    ap.add_argument('--min-state-grain', type=float, default=1_000_000)
    ap.add_argument('--min-acres', type=float, default=10_000)
    a = ap.parse_args(argv)
    raw = quickstats.load(a.quickstats)
    st = state_totals(raw, a.year)
    big = st[(st.grain >= a.min_state_grain) & (st.index != 'IOWA')].sort_values('silage', ascending=False)
    states = ['IOWA'] + list(big.index[:a.n_states])
    pd.set_option('display.width', 200)
    print('States by silage acres (grain >= %d):' % a.min_state_grain)
    print(st.loc[['IOWA'] + list(big.index[:10])].round(3).to_string())
    sel = quickstats.select(raw)
    shares = quickstats.shares(sel, 'COUNTY', a.year)
    rows = []
    for s in states:
        c = shares[(shares.state_name == s) & shares.grain_harvested.notna() & shares.silage_harvested.notna()].copy()
        c['total'] = c.grain_harvested + c.silage_harvested
        c = c[c.total >= a.min_acres]
        if len(c) < 3:
            continue
        lo, hi = c.silage_share.quantile([1 / 3, 2 / 3])
        for kind, part in (('high', c[c.silage_share >= hi]), ('low', c[c.silage_share <= lo])):
            r = part.sort_values('total', ascending=False).iloc[0]
            rows.append({'state': s, 'kind': kind, 'fips': r.fips, 'county': r.county_name, 'grain': r.grain_harvested,
                         'silage': r.silage_harvested, 'total': r.total, 'share': r.silage_share, 'eligible': len(c)})
    print()
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.3f}' if v < 1 else f'{v:,.0f}'))


if __name__ == '__main__':
    main()
