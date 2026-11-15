"""Sanity check of a county corn-field file before any modelling.

    python scripts/county_corn_check.py --fields data/labels/us_24021_2022/fields.geojson \\
        --fips 24021 --year 2022 --counties tl_2022_us_county.zip \\
        --quickstats corn_grain_silage_acres_2012_2025.csv [--csb csb1825_md.parquet] [--out check.json]

Reports field counts, the polygon area distribution, fields under 0.3 ha, how many fields lie
outside the Census TIGER county outline, and the county's USDA corn acres for the year after the
record-selection rule in fieldvision.model.quickstats (one record per geography, year, crop use and
measurement; county numbers come from the Census of Agriculture). The county SURVEY planted figure
is dropped by that rule; it is shown separately and labelled as such. With --csb, CDL classes other
than plain corn that contain corn (sweet/pop corn, double crops) are counted in the county for context.
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from fieldvision.model import quickstats

ACRES_PER_HA = 2.471053814671653
OTHER_CORN = {12: 'Sweet Corn', 13: 'Pop or Orn Corn', 225: 'Dbl Crop WinWht/Corn', 226: 'Dbl Crop Oats/Corn',
              228: 'Dbl Crop Triticale/Corn', 237: 'Dbl Crop Barley/Corn', 241: 'Dbl Crop Corn/Soybeans'}


def county_outline(path, fips):
    counties = gpd.read_file(path, where=f"GEOID = '{fips}'")
    if len(counties) != 1:
        raise SystemExit(f'expected one county {fips} in {path}, found {len(counties)}')
    return counties


def usda_numbers(csv, fips, year):
    raw = quickstats.load(csv)
    sel = quickstats.select(raw)
    rows = sel[(sel.aggregation_level == 'COUNTY') & (sel.fips == fips) & (sel.year == year)]
    out = {'selected_records': rows[['crop_use', 'measurement', 'value', 'value_num', 'suppressed', 'source',
                                     'short_description']].to_dict('records')}
    for use in ('grain', 'silage'):
        r = rows[(rows.crop_use == use) & (rows.measurement == 'harvested')]
        out[f'{use}_harvested_acres'] = None if r.empty or r.suppressed.iloc[0] else float(r.value_num.iloc[0])
        out[f'{use}_suppressed'] = bool(r.empty or r.suppressed.iloc[0])
    g, s = out['grain_harvested_acres'], out['silage_harvested_acres']
    out['grain_plus_silage_acres'] = g + s if g is not None and s is not None else None
    out['silage_share'] = s / (g + s) if out['grain_plus_silage_acres'] else None
    planted = rows[(rows.crop_use == 'all_corn') & (rows.measurement == 'planted')]
    out['planted_in_selection'] = None if planted.empty else planted.value_num.iloc[0]
    st, ct = fips[:2], fips[2:]
    survey = raw[(raw.aggregation_level == 'COUNTY') & (raw.state_ansi.str.zfill(2) == st)
                 & (raw.county_ansi.fillna('').str.zfill(3) == ct) & (raw.year == year)
                 & (raw.source == 'SURVEY') & (raw.domain == 'TOTAL') & (raw.reference_period == 'YEAR')]
    out['county_survey_records_outside_rule'] = survey[['short_description', 'value', 'value_num']].to_dict('records')
    return out


SURVEY_SERIES = {'CORN - ACRES PLANTED': 'planted_acres', 'CORN, GRAIN - ACRES HARVESTED': 'grain_harvested_acres',
                 'CORN, SILAGE - ACRES HARVESTED': 'silage_harvested_acres'}


def survey_numbers(csv, fips, year):
    """County SURVEY estimates (domain TOTAL, reference period YEAR) for one year; one record per series, checked."""
    raw = quickstats.load(csv)
    rows = raw[(raw.aggregation_level == 'COUNTY') & (raw.source == 'SURVEY') & (raw.year == year)
               & (raw.domain == 'TOTAL') & (raw.reference_period == 'YEAR')
               & (raw.state_ansi.str.zfill(2) == fips[:2]) & (raw.county_ansi.fillna('').str.zfill(3) == fips[2:])
               & raw.short_description.isin(SURVEY_SERIES)]
    if rows.short_description.duplicated().any():
        raise SystemExit(f'more than one survey record per series for {fips} {year}')
    out = {'year': year, 'records': rows[['short_description', 'value']].to_dict('records')}
    for desc, key in SURVEY_SERIES.items():
        r = rows[rows.short_description == desc]
        out[key] = None if r.empty or r.suppressed.iloc[0] else float(r.value_num.iloc[0])
    g, s = out['grain_harvested_acres'], out['silage_harvested_acres']
    out['grain_plus_silage_acres'] = g + s if g is not None and s is not None else None
    out['silage_share'] = s / (g + s) if out['grain_plus_silage_acres'] else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fields', type=Path, required=True)
    ap.add_argument('--fips', required=True)
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--counties', required=True, help='TIGER/Line county file (zip or shapefile)')
    ap.add_argument('--quickstats', type=Path, required=True)
    ap.add_argument('--csb', type=Path, help='state CSB GeoParquet, for the other-corn-classes context')
    ap.add_argument('--utm', default='EPSG:32618')
    ap.add_argument('--census-year', type=int, help='Census year for the USDA comparison (default: --year)')
    ap.add_argument('--survey-csv', type=Path, help='Quick Stats table with county SURVEY rows (e.g. 2021-2023)')
    ap.add_argument('--survey-year', type=int, help='survey year to report (default: --year)')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args(argv)

    fields = gpd.read_file(a.fields)
    assert fields.field_id.is_unique and fields.geometry.is_valid.all()
    area = fields.area_ha
    county = county_outline(a.counties, a.fips).to_crs(a.utm)
    shape = county.geometry.iloc[0]
    metric = fields.to_crs(a.utm)
    inside_point = metric.geometry.representative_point().within(shape)
    inside_area = metric.geometry.intersection(shape).area / 10000
    q = area.quantile([0, .05, .25, .5, .75, .95, 1]).round(3)
    res = {
        'fields': int(len(fields)), 'year': a.year, 'fips': a.fips, 'county_name': county.NAMELSAD.iloc[0],
        'area_ha_total': float(area.sum()), 'area_acres_total': float(area.sum() * ACRES_PER_HA),
        'acres_reported_total': float(fields.acres_reported.sum()) if 'acres_reported' in fields else None,
        'area_ha_quantiles': {str(k): float(v) for k, v in q.items()}, 'area_ha_mean': float(area.mean()),
        'fields_under_0_3_ha': int((area < 0.3).sum()), 'fields_under_1_ha': int((area < 1).sum()),
        'fields_with_point_outside_tiger_county': int((~inside_point).sum()),
        'area_ha_outside_tiger_county': float(area.sum() - inside_area.sum()),
        'county_land_ha': float(county.ALAND.iloc[0]) / 10000,
        'census_year': a.census_year or a.year,
        'usda': usda_numbers(a.quickstats, a.fips, a.census_year or a.year),
    }
    u = res['usda']
    if u['grain_plus_silage_acres']:
        res['polygon_acres_over_census_harvested'] = res['area_acres_total'] / u['grain_plus_silage_acres']
    if a.survey_csv:
        res['survey'] = survey_numbers(a.survey_csv, a.fips, a.survey_year or a.year)
        s = res['survey']
        if s['grain_plus_silage_acres']:
            res['polygon_acres_over_survey_harvested'] = res['area_acres_total'] / s['grain_plus_silage_acres']
        if s['planted_acres']:
            res['polygon_acres_over_survey_planted'] = res['area_acres_total'] / s['planted_acres']
    if a.csb:
        csb = gpd.read_parquet(a.csb)
        c = csb[csb.CNTYFIPS.astype(str).str.zfill(3) == a.fips[2:]]
        code = pd.to_numeric(c[f'CDL{a.year}'], errors='coerce')
        acres = c.to_crs(a.utm).geometry.area / 10000 * ACRES_PER_HA
        res['csb_county_polygons'] = int(len(c))
        res['csb_other_corn_classes'] = {f'{k} {v}': {'fields': int((code == k).sum()), 'acres': float(acres[code == k].sum())}
                                         for k, v in OTHER_CORN.items() if (code == k).any()}
        res['csb_class1_acres_check'] = float(acres[code == 1].sum())
    text = json.dumps(res, indent=2, default=lambda v: None if pd.isna(v) else v)
    if a.out:
        a.out.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
