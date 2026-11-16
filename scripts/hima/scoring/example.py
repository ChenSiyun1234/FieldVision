import sys
sys.path.insert(0, '/workspace/FieldVision/.deps')
import numpy as np, pandas as pd
from fieldvision.toolbox import Toolbox
from fieldvision_agents.gap.gap import county_gap_report, harvest_curves, interval_probability
for fips, data in (("36121", '/workspace/hima_data/fieldvision_export/counties'), ("24021", '/workspace/FieldVision/data/counties')):
    tb = Toolbox(data_root=data, registry='registry', out_root='/workspace/hima_build/results/gap/toolbox')
    df, s, _ = county_gap_report(tb, fips, 2022)
    days, fs, fg, ns, ng = harvest_curves(df)
    p, info = interval_probability(df)
    print(fips, info)
    g = df[df.abstain & (df.kind == 'gap')]
    top = g.groupby(['last_standing_date', 'drop_date']).size().sort_values(ascending=False).head(2)
    for (a, b), n in top.items():
        da, db = pd.Timestamp(a).dayofyear, pd.Timestamp(b).dayofyear
        sel = (days > da) & (days <= db)
        S, G = fs[sel].sum(), fg[sel].sum(); pi = info['pi']
        print('  gap', a, b, 'fields', n, 'S', round(S, 3), 'G', round(G, 3), 'pi', pi, 'p', round(pi * S / (pi * S + (1 - pi) * G), 3))
