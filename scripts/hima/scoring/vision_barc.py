"""Score the vision check once on the labelled Beltsville fields (held-out; nothing is tuned here).
For every field with a harvest bracket: chips at the two bracket dates, the vision model's answer,
compared with the reference label. Also the model's own call beside it."""
import sys, json, os
sys.path.insert(0, '/workspace/FieldVision/.deps')
import pandas as pd, geopandas as gpd
from pathlib import Path
from fieldvision.toolbox import Toolbox
from fieldvision_agents.gap.gap import county_gap_report, run_vision
from fieldvision_agents.react.supervisor import VISION_MODEL, NemotronClient
DATA = '/workspace/FieldVision/data/counties'
OUT = Path('/workspace/hima_build/results/gap/barc')
tb = Toolbox(data_root=DATA, registry='registry', out_root=str(OUT / 'toolbox'))
vlm = NemotronClient(model=VISION_MODEL, max_tokens=2048)
allr = []
for year in (2021, 2022, 2023):
    df, s, _ = county_gap_report(tb, '24033', year)
    gf = gpd.read_file(Path(DATA) / '24033' / str(year) / 'fields.geojson').to_crs('EPSG:4326')
    geoms = gf.set_index('field_id').geometry.to_dict()
    truth = {r.field_id: ('silage' if 'silage' in str(r.crop_name).lower() else 'grain') for r in gf.itertuples()}
    for kind in ('clear', 'gap'):
        res = run_vision(df, geoms, kind, 100, 0, OUT / str(year), vlm)
        for r in res:
            r['truth'] = truth.get(r['field_id']); r['year'] = year
            allr.append(r)
t = pd.DataFrame(allr)
t['vision_call'] = t['hint'].map({'silage-like': 'silage', 'grain-like': 'grain'})
t['readable'] = t['vision_call'].notna() & (t['confidence'].astype(float) >= 0.6)
rd = t[t.readable]
out = {"fields_with_bracket": int(len(t)), "truth_counts": t.truth.value_counts().to_dict(),
       "vision_readable": int(len(rd)), "vision_correct": int((rd.vision_call == rd.truth).sum()),
       "vision_says": t.after_ground.value_counts().to_dict() if 'after_ground' in t else {},
       "vision_by_truth": pd.crosstab(rd.truth, rd.vision_call).to_dict(),
       "model_correct_same_fields": int((t.call == t.truth).sum()),
       "errors": int(t['error'].notna().sum()) if 'error' in t else 0}
print(json.dumps(out, indent=1))
t.drop(columns=[c for c in ('reason',) if c in t]).to_csv(OUT / 'vision_barc.csv', index=False)
(OUT / 'vision_barc.json').write_text(json.dumps(out, indent=1))
