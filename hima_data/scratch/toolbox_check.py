import sys, json
sys.path[:0] = ["/workspace/FieldVision/.deps", "/workspace/FieldVision/src"]
from fieldvision.toolbox import Toolbox
tb = Toolbox(data_root="/workspace/hima_data/fieldvision_export/counties", out_root="/workspace/hima_data/results")
for fips in ("36121", "06107"):
    s = tb.scout(fips, 2022)
    print(fips, "scout", json.dumps(s, default=str)[:300], flush=True)
    tb.extract_features(fips, 2022)
    c = tb.classify(fips, 2022, "standard")
    print(fips, "classify", json.dumps(c, default=str)[:300], flush=True)
    a = tb.aggregate(fips, 2022)
    print(fips, "aggregate", {k: a.get(k) for k in ("silage_acres", "grain_acres", "abstained_acres", "n_fields")}, flush=True)
