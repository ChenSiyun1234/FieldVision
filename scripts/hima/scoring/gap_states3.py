"""Blind check 3: the interval formula p = pi*S / (pi*S + (1-pi)*G) for abstained gap fields.
Fixed before the first run (15 Nov, 2:30 AM): judged fields keep the model's call; abstained fields
with a bracket take the interval probability from the place's own harvest curves; unseen fields take pi.
PRIMARY: pi = share of judged fields called silage (no USDA input at all).
SECONDARY: pi = last official share before the season. USDA's figure for the season is only compared."""
import sys, json, os
sys.path.insert(0, '/workspace/FieldVision/.deps')
import numpy as np, pandas as pd
from fieldvision.toolbox import Toolbox
from fieldvision_agents.settings import QUICKSTATS_CSV
from fieldvision_agents.fieldvision_bridge.prevalence import silage_share_prior
from fieldvision_agents.gap.gap import gap_table, gap_summary, interval_share, interval_probability, county_gap_report
NASS = {"PA": 32.8, "WI": 21.6, "IA": 1.9, "NE": 2.1, "NY": 53.2, "MI": 14.7, "MN": 4.1, "SD": 5.8, "CA": 83.1}
FIPS = {"PA": "42", "WI": "55", "IA": "19", "NE": "31", "NY": "36", "MI": "26", "MN": "27", "SD": "46", "CA": "06"}
ROOT = '/workspace/data/states'
out = []
def add(name, df, prior, truth):
    s = gap_summary(df)['silage_share']
    sh, info = interval_share(df)
    sh2, _ = interval_share(df, prior['share'])
    p, _ = interval_probability(df)
    ab = df['abstain'].to_numpy() & (df['kind'] != 'unseen').to_numpy()
    out.append({"place": name, "abst_gap": int(ab.sum()), "used": info['used'],
                "med_sil": info.get('median_silage_harvest_doy'), "med_gr": info.get('median_grain_harvest_doy'),
                "model_sil_in_gap": round(float((df.loc[ab, 'call'] == 'silage').mean()) * 100, 0) if ab.any() else None,
                "interval_p_in_gap": round(float(np.nanmean(p[ab])) * 100, 0) if ab.any() and np.isfinite(p[ab]).any() else None,
                "called": s['called'] * 100, "with_gap": s['with_gap_fields'] * 100,
                "interval": None if sh is None else sh * 100, "interval_prior": None if sh2 is None else sh2 * 100,
                "prior": prior['share'] * 100, "usda": truth})
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
    g = json.load(open(f'{ROOT}/{st}/fields.geojson'))
    area = {f['properties']['field_id']: f['properties']['area_ha'] * f['properties']['weight'] for f in g['features']}
    add(st + ' 2025', gap_table(rows, area), silage_share_prior(QUICKSTATS_CSV, FIPS[st] + '000', 2025), nass)
for fips, data, truth in (("24021", '/workspace/FieldVision/data/counties', 19.6), ("36121", '/workspace/hima_data/fieldvision_export/counties', 66.7),
                          ("06107", '/workspace/hima_data/fieldvision_export/counties', 98.7)):
    tb = Toolbox(data_root=data, registry='registry', out_root='/workspace/hima_build/results/gap/toolbox')
    df, _, _ = county_gap_report(tb, fips, 2022)
    add(fips + ' 2022', df, silage_share_prior(QUICKSTATS_CSV, fips, 2022), truth)
t = pd.DataFrame(out).round(1)
cols = ["called", "with_gap", "interval", "interval_prior", "prior"]
print(t.to_string(index=False))
for c in cols:
    t["e_" + c] = (t[c] - t["usda"]).round(1)
print(t[["place"] + ["e_" + c for c in cols]].to_string(index=False))
for name, m in (("9 states 2025", t.place.str.endswith('2025')), ("3 counties 2022", t.place.str.endswith('2022')),
                ("Frederick + Wyoming", t.place.isin(['24021 2022', '36121 2022']))):
    print(name, "mean abs error:", {c: round(float(t.loc[m, "e_" + c].abs().mean()), 1) for c in cols})
t.to_csv('/workspace/hima_build/results/gap/interval_rule_vs_usda.csv', index=False)
