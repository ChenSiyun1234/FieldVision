"""Blind check 2: prior-shifted share on the 2025 state samples and two 2022 counties.
Rule fixed before the first run (15 Nov): every field with a harvest bracket keeps its probability
moved to the prior; unseen fields take the prior. Prior = silage_share_prior(): county Census before
the season, else the state's survey mean over the 3 years before it. The season's own USDA figure
appears only in the last columns."""
import sys, json, os
sys.path.insert(0, '/workspace/FieldVision/.deps')
import numpy as np, pandas as pd
from fieldvision.toolbox import Toolbox
from fieldvision_agents.settings import QUICKSTATS_CSV
from fieldvision_agents.fieldvision_bridge.prevalence import silage_share_prior
from fieldvision_agents.gap.gap import gap_table, gap_summary, prior_shifted_share, county_gap_report
NASS = {"PA": 32.8, "WI": 21.6, "IA": 1.9, "NE": 2.1, "NY": 53.2, "MI": 14.7, "MN": 4.1, "SD": 5.8, "CA": 83.1}
FIPS = {"PA": "42", "WI": "55", "IA": "19", "NE": "31", "NY": "36", "MI": "26", "MN": "27", "SD": "46", "CA": "06"}
ROOT = '/workspace/data/states'
out = []
def add(name, df, prior, truth, truth_src):
    s = gap_summary(df)['silage_share']
    out.append({"place": name, "fields": len(df), "prior": prior['share'], "prior_level": prior['level'],
                "called": s['called'], "with_gap": s['with_gap_fields'],
                "prior_shifted": prior_shifted_share(df, prior['share']), "usda": truth / 100, "usda_src": truth_src})
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
    add(st + ' 2025', gap_table(rows, area), silage_share_prior(QUICKSTATS_CSV, FIPS[st] + '000', 2025), nass, 'survey 2025')
for fips, data, truth in (("24021", '/workspace/FieldVision/data/counties', 19.6), ("36121", '/workspace/hima_data/fieldvision_export/counties', 66.7)):
    tb = Toolbox(data_root=data, registry='registry', out_root='/workspace/hima_build/results/gap/toolbox')
    df, _, _ = county_gap_report(tb, fips, 2022)
    add(fips + ' 2022', df, silage_share_prior(QUICKSTATS_CSV, fips, 2022), truth, 'census 2022')
t = pd.DataFrame(out)
for c in ["prior", "called", "with_gap", "prior_shifted", "usda"]:
    t[c] = (t[c].astype(float) * 100).round(1)
for c in ["prior", "called", "with_gap", "prior_shifted"]:
    t["e_" + c] = (t[c] - t["usda"]).round(1)
print(t.drop(columns=["usda_src", "prior_level", "fields"]).to_string(index=False))
for name, m in (("9 states 2025", t.place.str.endswith('2025')), ("2 counties 2022", t.place.str.endswith('2022'))):
    print(name, "mean abs error:", {c: round(t.loc[m, "e_" + c].abs().mean(), 1) for c in ["prior", "called", "with_gap", "prior_shifted"]})
t.to_csv('/workspace/hima_build/results/gap/gap_rules_vs_usda.csv', index=False)
