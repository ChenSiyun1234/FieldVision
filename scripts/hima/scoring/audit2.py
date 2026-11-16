import json, sys, glob, os
p=max(glob.glob('/workspace/hima_build/runs/logs/audit_'+sys.argv[1]+'*.jsonl'), key=os.path.getmtime)
for l in open(p):
    j=json.loads(l)
    name=j.get('tool') or j.get('name') or ''
    if name=='inspect_gaps': print('VISION', json.dumps(j['result'].get('vision')), 'landsat', j['result'].get('landsat_rows_on_disk'))
    if name in ('abstain_region','finish'): print(name.upper(), json.dumps(j.get('args'))[:1100])
