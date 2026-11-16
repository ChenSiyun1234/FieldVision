import json, glob, collections
for p in sorted(glob.glob('/workspace/hima_build/results/gap/*/*/vision_*.jsonl')):
    r=[json.loads(l) for l in open(p) if l.strip()]
    c=collections.Counter((x.get('call'), x.get('after_ground','ERR'), x['status']) for x in r)
    print(p.split('gap/')[1], len(r), dict(collections.Counter(x['status'] for x in r)))
    for k,v in sorted(c.items(), key=lambda kv:-kv[1])[:8]: print('   ',k,v)
    e=[x['error'][:90] for x in r if x.get('error')][:2]
    if e: print('   errors e.g.', e)
