import pandas as pd, numpy as np
t = pd.read_csv('/workspace/hima_build/results/gap/late_green_by_place.csv')
t['if_all_notjudged_silage'] = (t.called * (100 - t.not_judged) / 100 + t.not_judged).round(1)
t['err_if_all'] = (t.if_all_notjudged_silage - t.usda).round(1)
print(t[['place', 'not_judged', 'called', 'usda', 'err', 'if_all_notjudged_silage', 'err_if_all', 'green_notjudged']].to_string(index=False))
print('corr(not_judged, -err), all 11:', round(float(np.corrcoef(t.not_judged, -t.err)[0, 1]), 2))
m = t.place != 'CA 2025'
print('without CA:', round(float(np.corrcoef(t.not_judged[m], -t.err[m])[0, 1]), 2))
hi = t[t.not_judged >= 20]; lo = t[t.not_judged < 20]
print('not judged >= 20%:', list(hi.place), 'mean miss', round(float(hi.err.abs().mean()), 1), '| under 20%:', len(lo), 'places, mean miss', round(float(lo.err.abs().mean()), 1))
t.to_csv('/workspace/hima_build/results/gap/separator_by_place.csv', index=False)
