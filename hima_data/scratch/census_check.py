import sys
sys.path[:0] = ["/workspace/FieldVision/.deps", "/workspace/FieldVision/rag-pipeline"]
import os
os.environ.setdefault("QUICKSTATS_CSV", "/workspace/02_QuickStats/corn_grain_silage_acres_2012_2025.csv")
import rag_pipeline as rp
for st, co in [("California", "Tulare"), ("New York", "Wyoming"), ("New York", "Cayuga"), ("Idaho", "Gooding"),
               ("Pennsylvania", "Lancaster"), ("Wisconsin", "Dodge"), ("Maryland", "Frederick"), ("Maryland", "Queen Anne's")]:
    h = rp.silage_share_history(st, co)
    rows = {r["year"]: (round(100 * r["silage_share"], 1), int(r["silage_acres"]), int(r["grain_acres"]), r["note"]) for r in h["by_year"]}
    print(st, co, {y: rows[y] for y in sorted(rows) if y in (2012, 2017, 2022)}, flush=True)
