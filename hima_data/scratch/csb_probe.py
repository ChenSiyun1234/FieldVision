import pyogrio, time, sys
p = '/vsizip//workspace/data/states/csb/NationalCSB_2018-2025_rev23.zip/CSB1825.gdb'
info = pyogrio.read_info(p, layer='national1825')
dt = dict(zip(info['fields'], info['dtypes']))
print('types', dt['STATEFIPS'], dt['CNTYFIPS'], dt['CDL2022'], flush=True)
for st, c in [('06', '107'), ('36', '121'), ('36', '011'), ('06', '031')]:
    t = time.time()
    d = pyogrio.read_dataframe(p, layer='national1825', where=f"STATEFIPS='{st}' AND CNTYFIPS='{c}'",
                               columns=['CSBID', 'CDL2022', 'CDL2023', 'CSBACRES'], read_geometry=False)
    c22 = d[d.CDL2022.astype(str) == '1']
    print(st, c, 'polys', len(d), 'corn2022', len(c22), 'acres', round(c22.CSBACRES.sum()),
          'corn2023', int((d.CDL2023.astype(str) == '1').sum()), round(time.time() - t, 1), 's', flush=True)
