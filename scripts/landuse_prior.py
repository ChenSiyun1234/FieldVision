"""County silage share predicted from land use (CDL forage and pasture shares): a prior, not a field label.

    python scripts/landuse_prior.py table   --cdl-dir C:/t/fv/data/us/cropscape/county --out C:/t/fv/data/us/landuse/county_landuse.csv
    python scripts/landuse_prior.py holdout --landuse county_landuse.csv --census corn_grain_silage_acres_2012_2025.csv \\
                                            --survey corn_grain_silage_acres_2021_2023.csv --out-dir C:/t/fv/data/us/landuse
    python scripts/landuse_prior.py predict --landuse county_landuse.csv --census ... --targets nass_2025_state_targets.csv \\
                                            --model <name> --out-dir C:/t/fv/data/us/landuse

Land use per county and year from CropScape county statistics (30 m CDL pixel acres by class):
  forage   = alfalfa 36 + other hay 37 + clover/wildflowers 58 + vetch 224
  pasture  = grass/pasture 176
  cropland = CDL crop classes 1-61, 66-77, 204-254 (includes forage, excludes pasture)
  forage_share = forage / cropland;  fp_share = (forage + pasture) / (cropland + pasture)
  *_mean1923 = mean of the yearly shares 2019-2023 (before CDL 2024 moved much other hay into pasture)

Target: Census 2022 county silage share = silage / (grain + silage) harvested acres, counties where both are
published (fieldvision.model.quickstats.select). Models, each weighted by the county's grain + silage acres:
  linear: share = a + b x, clipped to [0, 1];  logit: logit(share) = a + b x (fractional logit, binomial deviance)
for x in {forage_share, fp_share, forage_share_mean1923, fp_share_mean1923} of the Census year.
Hold-outs: whole states left out in turn (2022), and the 2023 county survey (counties reporting both classes)
predicted from 2023 land use by models fitted on all 2022 counties. Errors in share points: county
(acre-weighted mean absolute) and pooled to states (|predicted - actual| of the state's pooled share).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fieldvision.model import quickstats  # noqa: E402

FORAGE, PASTURE = {36, 37, 58, 224}, {176}
FEATURES = ['forage_share', 'fp_share', 'forage_share_mean1923', 'fp_share_mean1923']
FORMS = ['linear', 'logit']


def is_crop(c):
    return (1 <= c <= 61) or (66 <= c <= 77) or (204 <= c <= 254)


def build_table(cdl_dir):
    rows = []
    for p in sorted(Path(cdl_dir).glob('CDL_*_*.csv')):
        m = re.match(r'CDL_(\d{4})_(\d{5})\.csv', p.name)
        if not m:
            continue
        year, fips = int(m.group(1)), m.group(2)
        acc = {'corn': 0.0, 'alfalfa': 0.0, 'other_hay': 0.0, 'forage': 0.0, 'pasture': 0.0, 'cropland': 0.0}
        for line in p.read_text(encoding='utf-8', errors='replace').splitlines()[1:]:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            code, acres = int(parts[0]), float(parts[-1])
            if code == 1:
                acc['corn'] += acres
            if code == 36:
                acc['alfalfa'] += acres
            if code == 37:
                acc['other_hay'] += acres
            if code in FORAGE:
                acc['forage'] += acres
            if code in PASTURE:
                acc['pasture'] += acres
            if is_crop(code):
                acc['cropland'] += acres
        rows.append({'fips': fips, 'year': year, **acc})
    t = pd.DataFrame(rows)
    t['forage_share'] = t.forage / t.cropland.where(t.cropland > 0)
    t['fp_share'] = (t.forage + t.pasture) / (t.cropland + t.pasture).where(t.cropland + t.pasture > 0)
    pre = t[t.year.between(2019, 2023)].groupby('fips')[['forage_share', 'fp_share']].mean()
    t = t.join(pre.add_suffix('_mean1923'), on='fips')
    return t


def census_shares(path, year):
    sel = quickstats.select(quickstats.load(path))
    s = quickstats.shares(sel, 'COUNTY', year)
    s = s[s.grain_harvested.notna() & s.silage_harvested.notna()].copy()
    s['total'] = s.grain_harvested + s.silage_harvested
    s = s[s.total > 0]
    s['state'] = s.fips.str[:2]
    return s[['fips', 'state', 'state_name', 'county_name', 'grain_harvested', 'silage_harvested', 'total', 'silage_share']]


def survey_shares(path, year):
    raw = quickstats.load(path)
    r = raw[(raw.aggregation_level == 'COUNTY') & (raw.source == 'SURVEY') & (raw.year == year) & (raw.domain == 'TOTAL')
            & (raw.reference_period == 'YEAR') & raw.county_ansi.notna() & (raw.county_ansi.str.strip() != '')]
    r = r.assign(fips=r.state_ansi.str.zfill(2) + r.county_ansi.str.zfill(3))
    piv = r[r.short_description.isin(['CORN, GRAIN - ACRES HARVESTED', 'CORN, SILAGE - ACRES HARVESTED'])]
    if piv.duplicated(['fips', 'short_description']).any():
        raise SystemExit('survey: more than one record per county and series')
    w = piv.pivot(index='fips', columns='short_description', values='value_num')
    w.columns = ['grain_harvested', 'silage_harvested']
    w = w.dropna()
    w['total'] = w.grain_harvested + w.silage_harvested
    w = w[w.total > 0].reset_index()
    w['silage_share'] = w.silage_harvested / w.total
    w['state'] = w.fips.str[:2]
    return w


def fit(form, x, y, w):
    if form == 'linear':
        X = np.column_stack([np.ones_like(x), x])
        W = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
        return coef
    eps = 1e-9

    def loss(c):
        p = 1 / (1 + np.exp(-(c[0] + c[1] * x)))
        p = np.clip(p, eps, 1 - eps)
        return -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / w.sum()
    return minimize(loss, x0=np.array([-3.0, 5.0]), method='BFGS').x


def predict(form, coef, x):
    if form == 'linear':
        return np.clip(coef[0] + coef[1] * x, 0, 1)
    return 1 / (1 + np.exp(-(coef[0] + coef[1] * x)))


def errors(d, pred):
    d = d.assign(pred=pred)
    county = float(np.sum(d.total * (d.pred - d.silage_share).abs()) / d.total.sum())
    st = d.groupby('state').apply(lambda g: pd.Series({
        'acres': g.total.sum(), 'actual': np.sum(g.silage_harvested) / g.total.sum(),
        'pred': np.sum(g.pred * g.total) / g.total.sum()}), include_groups=False)
    st['abs_err'] = (st.pred - st.actual).abs()
    return county, st


def holdout(a):
    lu = pd.read_csv(a.landuse, dtype={'fips': str})
    cen = census_shares(a.census, 2022).merge(lu[lu.year == 2022], on='fips', how='inner')
    sur = survey_shares(a.survey, 2023).merge(lu[lu.year == 2023], on='fips', how='inner')
    results, coefs = [], {}
    for feat in FEATURES:
        for form in FORMS:
            d = cen.dropna(subset=[feat])
            preds = pd.Series(np.nan, index=d.index)
            for st in sorted(d.state.unique()):
                tr, te = d[d.state != st], d[d.state == st]
                c = fit(form, tr[feat].to_numpy(), tr.silage_share.to_numpy(), tr.total.to_numpy())
                preds[te.index] = predict(form, c, te[feat].to_numpy())
            c_err, st_err = errors(d, preds.to_numpy())
            full = fit(form, d[feat].to_numpy(), d.silage_share.to_numpy(), d.total.to_numpy())
            coefs[f'{feat}|{form}'] = [float(v) for v in full]
            s = sur.dropna(subset=[feat])
            s_c, s_st = errors(s, predict(form, full, s[feat].to_numpy()))
            results.append({'model': f'{feat}|{form}', 'counties_2022': len(d), 'states_2022': len(st_err),
                            'county_mae_2022_lso': round(100 * c_err, 2),
                            'state_mae_2022_lso': round(100 * st_err.abs_err.mean(), 2),
                            'state_median_2022_lso': round(100 * st_err.abs_err.median(), 2),
                            'state_acre_weighted_2022_lso': round(100 * np.sum(st_err.abs_err * st_err.acres) / st_err.acres.sum(), 2),
                            'counties_2023': len(s), 'states_2023': len(s_st),
                            'county_mae_2023': round(100 * s_c, 2), 'state_mae_2023': round(100 * s_st.abs_err.mean(), 2),
                            'state_acre_weighted_2023': round(100 * np.sum(s_st.abs_err * s_st.acres) / s_st.acres.sum(), 2),
                            'coef': coefs[f'{feat}|{form}']})
            if feat == 'fp_share' and form == 'logit':
                st_err.to_csv(Path(a.out_dir) / 'holdout_states_fp_share_logit_2022.csv')
    out = pd.DataFrame(results)
    out.to_csv(Path(a.out_dir) / 'holdout_results.csv', index=False)
    (Path(a.out_dir) / 'coefficients.json').write_text(json.dumps(coefs, indent=1))
    cen.to_csv(Path(a.out_dir) / 'census2022_with_landuse.csv', index=False)
    pd.set_option('display.width', 250)
    print(out.drop(columns='coef').to_string(index=False))


def predict_states(a):
    lu = pd.read_csv(a.landuse, dtype={'fips': str})
    coefs = json.loads((Path(a.out_dir) / 'coefficients.json').read_text())
    feat, form = a.model.split('|')
    coef = coefs[a.model]
    year_rows = lu[lu.year == a.predict_year].dropna(subset=[feat]).copy()
    year_rows['pred'] = predict(form, coef, year_rows[feat].to_numpy())
    year_rows['state'] = year_rows.fips.str[:2]
    names = quickstats.load(a.census)[['state_ansi', 'state_name']].dropna().drop_duplicates()
    names = dict(zip(names.state_ansi.str.zfill(2), names.state_name))
    st = year_rows.groupby('state').apply(lambda g: pd.Series({
        'cdl_corn_acres': g.corn.sum(), 'pred_share': np.sum(g.pred * g.corn) / g.corn.sum()}), include_groups=False)
    st['state_name'] = st.index.map(names)
    st['pred_silage_acres'] = st.pred_share * st.cdl_corn_acres
    tg = []
    for line in Path(a.targets).read_text(encoding='utf-8').splitlines():
        if line.startswith('#') or not line.strip():
            continue
        p = [x.strip() for x in line.split(',')]
        try:
            tg.append({'state_name': p[0].upper(), 'target_grain': float(p[1]), 'target_silage': float(p[2])})
        except (ValueError, IndexError):
            continue
    tg = pd.DataFrame(tg)
    tg['target_share'] = tg.target_silage / (tg.target_grain + tg.target_silage)
    out = tg.merge(st.reset_index(), on='state_name', how='left')
    out['share_err_points'] = 100 * (out.pred_share - out.target_share)
    out.to_csv(Path(a.out_dir) / f'state_prediction_{a.predict_year}.csv', index=False)
    pd.set_option('display.width', 250)
    print(out.round(4).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['table', 'holdout', 'predict'])
    ap.add_argument('--cdl-dir', type=Path)
    ap.add_argument('--landuse', type=Path)
    ap.add_argument('--census', type=Path)
    ap.add_argument('--survey', type=Path)
    ap.add_argument('--targets', type=Path)
    ap.add_argument('--model')
    ap.add_argument('--predict-year', type=int, default=2025)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--out-dir', type=Path)
    a = ap.parse_args(argv)
    if a.stage == 'table':
        t = build_table(a.cdl_dir)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(a.out, index=False)
        print(t.groupby('year').agg(counties=('fips', 'size'), corn=('corn', 'sum'), other_hay=('other_hay', 'sum'),
                                    pasture=('pasture', 'sum')).round(0).to_string())
    elif a.stage == 'holdout':
        a.out_dir.mkdir(parents=True, exist_ok=True)
        holdout(a)
    else:
        predict_states(a)


if __name__ == '__main__':
    main()
