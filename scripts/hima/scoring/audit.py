import json, sys, glob, os
p=max(glob.glob('/workspace/hima_build/runs/logs/audit_'+sys.argv[1]+'*.jsonl'), key=os.path.getmtime)
print(os.path.basename(p))
for l in open(p):
    j=json.loads(l)
    name=j.get('tool') or j.get('name') or ''
    if not name: continue
    extra=''
    if name in ('inspect_gaps','finish','abstain_region'): extra=' => '+json.dumps(j.get('result',''))[:1500]
    print(j.get('step'), name, json.dumps(j.get('args',''))[:700 if name in ('finish','abstain_region') else 80], extra)
