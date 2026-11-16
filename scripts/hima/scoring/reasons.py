import json
r=json.load(open('/workspace/hima_build/results/gap/36121_2022/s2/gap_reasons.json'))
import collections
print(len(r), dict(collections.Counter(x['written_by'] for x in r)), dict(collections.Counter(x.get('rejected') for x in r)))
for x in r[:3]: print('-', x['field_id'], x['written_by'], '|', x['reason'][:420])
