import json, sys
d=json.load(open(sys.argv[1]))
s=d["summary"]
print({k:(v["fields"],v["acres"]) for k,v in s["kinds"].items()})
print(s["silage_share"])
print("abstained",s["abstained_fields"],"with bracket",s["abstained_with_bracket"],"called silage",s["abstained_with_bracket_called_silage_share"])
for g in s["main_gaps"]: print(g["from"],g["to"],g["fields"],g["abstained_fields"],g["acres"],g["median_ndvi_before"],g["median_ndvi_after"],g["called_silage_share"])
for k in ("vision_gap","vision_clear","reasons_written","reasons_by_nemotron"):
    if k in d: print(k, d[k])
