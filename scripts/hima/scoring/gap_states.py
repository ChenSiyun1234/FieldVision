"""Blind check of the gap rule on the 2025 state samples. The rule (count gap fields by the model's
call, abstained ones included) was fixed on 15 Nov before this script was first run.
NASS 2025 survey shares are used only in the last column, never in the estimate."""
import sys, json, os
sys.path.insert(0, '/workspace/FieldVision/.deps')
import numpy as np, pandas as pd
from fieldvision_agents.gap.gap import gap_table, gap_summary
NASS = {"PA": 32.8, "WI": 21.6, "IA": 1.9, "NE": 2.1, "NY": 53.2, "MI": 14.7, "MN": 4.1, "SD": 5.8, "CA": 83.1}
ROOT = '/workspace/data/states'
out = []
for st, nass in NASS.items():
    p = f'{ROOT}/{st}/results/eu2_logistic_v1/{st}_2025.jsonl'
    if not os.path.exists(p):
        print(st, 'no eu2_logistic_v1 results'); continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    for r in rows:
        e = r['evidence']
        if not e.get('evidence_tier'):
            d = e.get('days_to_drop')
            e['evidence_tier'] = 'abstain' if r['abstain'] else ('clear' if d is not None and d <= 12 else 'wide')
    g = json.load(open(f'{ROOT}/{st}/fields.geojson'))
    area = {f['properties']['field_id']: f['properties']['area_ha'] * f['properties']['weight'] for f in g['features']}
    df = gap_table(rows, area)
    s = gap_summary(df)
    sh = s['silage_share']
    out.append({"state": st, "fields": len(df), "abstained": s['abstained_fields'], "abst_with_bracket": s['abstained_with_bracket'],
                "clear_only": sh['clear_only'], "called": sh['called'], "with_gap": sh['with_gap_fields'],
                "lower": sh['lower'], "upper": sh['upper'], "nass_2025": nass / 100})
t = pd.DataFrame(out)
for c in ["clear_only", "called", "with_gap", "lower", "upper", "nass_2025"]:
    t[c] = (t[c].astype(float) * 100).round(1)
t["err_called"] = (t["called"] - t["nass_2025"]).round(1)
t["err_with_gap"] = (t["with_gap"] - t["nass_2025"]).round(1)
print(t.to_string(index=False))
print("mean abs error, called:", round(t.err_called.abs().mean(), 1), " with_gap:", round(t.err_with_gap.abs().mean(), 1), " states:", len(t))
os.makedirs('/workspace/hima_build/results/gap', exist_ok=True)
t.to_csv('/workspace/hima_build/results/gap/states_2025_gap_rule.csv', index=False)
