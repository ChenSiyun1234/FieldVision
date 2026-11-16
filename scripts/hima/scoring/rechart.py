import sys, json
sys.path.insert(0, '/workspace/FieldVision/.deps')
from fieldvision_agents.watch.watch import chart
names = {"36121": ("Wyoming County, NY, 2022: what we know, by date", ("Census 2022", 66.7)),
         "24021": ("Frederick County, MD, 2022: what we know, by date", ("Census 2022", 19.6))}
for fips, (title, ref) in names.items():
    d = '/workspace/hima_build/results/watch/' + fips + '_2022/'
    try:
        j = json.load(open(d + 'replay.json'))
    except FileNotFoundError:
        continue
    chart(j['steps'], j['summary'], d + 'replay.png', title=title, reference=ref)
    print(fips, j['summary'])
