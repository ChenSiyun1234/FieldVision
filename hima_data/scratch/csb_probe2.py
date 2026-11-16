import pyogrio, time
p = '/vsizip//workspace/data/states/csb/NationalCSB_2018-2025_rev23.zip/CSB1825.gdb'
t = time.time()
d = pyogrio.read_dataframe(p, layer='national1825', columns=['STATEFIPS', 'CNTYFIPS', 'CDL2022'], read_geometry=False)
print('rows', len(d), round(time.time() - t), 's', flush=True)
s = d.STATEFIPS.astype(str)
print('states', sorted(s.unique())[:60], flush=True)
for st, c in [('06', '107'), ('6', '107'), ('36', '121'), ('36', '011'), ('16', '047'), ('06', '031'), ('24', '021')]:
    m = (s == st) & (d.CNTYFIPS.astype(str) == c)
    print(st, c, int(m.sum()), 'corn22', int((m & (d.CDL2022 == 1)).sum()), flush=True)
