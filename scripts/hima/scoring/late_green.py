"""What separates the places we get right from the ones we read low? One table per place.

Fixed before the first run (15 Nov, 2:35 AM), nothing tuned:
  late window     day of year 298 to 334 (25 Oct to 30 Nov)
  usable look     clear_fraction >= 0.8
  late NDVI       median NDVI of a field's usable looks in the late window
  green late      late NDVI >= 0.45  (a growing crop; standing dry corn and stalk residue sit near 0.2 to 0.3)
Idea: after silage the ground is bare in September and a winter cover or forage crop is usually sown, so
the field is green again by November. After grain the field is brown stalks. This does not need the
harvest to be seen. USDA figures are only compared in the last columns."""
import sys, json, os
sys.path.insert(0, '/workspace/FieldVision/.deps')
import numpy as np, pandas as pd
from fieldvision.toolbox import Toolbox
from fieldvision_agents.gap.gap import gap_table, county_gap_report
NASS = {"PA": 32.8, "WI": 21.6, "IA": 1.9, "NE": 2.1, "NY": 53.2, "MI": 14.7, "MN": 4.1, "SD": 5.8, "CA": 83.1}
ROOT = '/workspace/data/states'
LO, HI, CLEAR, GREEN = 298, 334, 0.8, 0.45

def late_ndvi(ts):
    ts = ts[ts['clear_fraction'] >= CLEAR].copy()
    ts['doy'] = pd.to_datetime(ts['date']).dt.dayofyear
    ts['ndvi'] = (ts['B08'] - ts['B04']) / (ts['B08'] + ts['B04'])
    late = ts[(ts.doy >= LO) & (ts.doy <= HI)].groupby('field_id')['ndvi'].median()
    sep = ts[(ts.doy >= 244) & (ts.doy <= 273)].groupby('field_id')['ndvi'].count()
    return late, sep

def wshare(mask, w):
    return round(100 * float(w[mask].sum() / w.sum()), 1) if w.sum() > 0 else None

rows_out, field_frames = [], {}
def add(name, df, ts, usda):
    late, sep = late_ndvi(ts)
    df = df.copy()
    df['late'] = df['field_id'].map(late)
    df['sep_looks'] = df['field_id'].map(sep).fillna(0)
    w = df['acres'].to_numpy(float)
    has = df['late'].notna().to_numpy()
    green = (df['late'] >= GREEN).to_numpy()
    judged = (~df['abstain']).to_numpy() & (df['kind'] != 'unseen').to_numpy()
    sil, gr = judged & (df['call'] == 'silage').to_numpy(), judged & (df['call'] == 'grain').to_numpy()
    notj = ~judged
    def g(m):
        m2 = m & has
        return wshare(green[m2], w[m2]) if m2.sum() >= 5 else None
    called = wshare(sil[judged], w[judged])
    rows_out.append({"place": name, "n": len(df), "late_cover": wshare(has, w), "sep_looks": round(float(df.sep_looks.median()), 0),
                     "not_judged": wshare(notj, w), "called": called, "usda": usda, "err": None if called is None else round(called - usda, 1),
                     "green_all": g(np.ones(len(df), bool)), "green_sil": g(sil), "green_grain": g(gr), "green_notjudged": g(notj),
                     "grain_called_green_acres": wshare(gr & has & green, w)})
    field_frames[name] = df

for st, nass in NASS.items():
    p = f'{ROOT}/{st}/results/eu2_logistic_v1/{st}_2025.jsonl'
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    for r in rows:
        e = r['evidence']
        if not e.get('evidence_tier'):
            d = e.get('days_to_drop')
            e['evidence_tier'] = 'abstain' if r['abstain'] else ('clear' if d is not None and d <= 12 else 'wide')
    gj = json.load(open(f'{ROOT}/{st}/fields.geojson'))
    area = {f['properties']['field_id']: f['properties']['area_ha'] * f['properties']['weight'] for f in gj['features']}
    ts = pd.read_parquet(f'{ROOT}/{st}/timeseries/timeseries.parquet', columns=['field_id', 'date', 'clear_fraction', 'B04', 'B08'])
    add(st + ' 2025', gap_table(rows, area), ts, nass)
for fips, data, truth in (("24021", '/workspace/FieldVision/data/counties', 19.6), ("36121", '/workspace/hima_data/fieldvision_export/counties', 66.7)):
    tb = Toolbox(data_root=data, registry='registry', out_root='/workspace/hima_build/results/gap/toolbox')
    df, _, _ = county_gap_report(tb, fips, 2022)
    ts = pd.read_parquet(f'{data}/{fips}/2022/timeseries.parquet', columns=['field_id', 'date', 'clear_fraction', 'B04', 'B08'])
    add(fips + ' 2022', df, ts, truth)
t = pd.DataFrame(rows_out)
print(t.to_string(index=False))
ok = t.dropna(subset=['green_all', 'err'])
print('corr(green_all, usda) =', round(float(np.corrcoef(ok.green_all, ok.usda)[0, 1]), 2), ' corr(grain_called_green_acres, -err) =', round(float(np.corrcoef(ok.grain_called_green_acres, -ok.err)[0, 1]), 2), ' n =', len(ok))
t.to_csv('/workspace/hima_build/results/gap/late_green_by_place.csv', index=False)

# labelled Beltsville fields: the same flag against the reference label (scored once, not tuned)
import geopandas as gpd
b = []
for year in (2021, 2022, 2023):
    d = f'/workspace/FieldVision/data/counties/24033/{year}'
    gf = gpd.read_file(d + '/fields.geojson')
    ts = pd.read_parquet(d + '/timeseries.parquet', columns=['field_id', 'date', 'clear_fraction', 'B04', 'B08'])
    late, _ = late_ndvi(ts)
    for r in gf.itertuples():
        b.append({"truth": 'silage' if 'silage' in str(r.crop_name).lower() else 'grain', "late": late.get(r.field_id, np.nan), "year": year})
b = pd.DataFrame(b)
b['green'] = np.where(b.late.isna(), 'no late look', np.where(b.late >= GREEN, 'green late', 'not green'))
print(pd.crosstab(b.truth, b.green).to_string())
print(b.groupby('truth').late.describe()[['count', '25%', '50%', '75%']].round(2).to_string())
b.to_csv('/workspace/hima_build/results/gap/late_green_barc.csv', index=False)
