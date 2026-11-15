"""Offline smoke test: no GPU, no API key, no downloads.
Builds a tiny index with the 'hash' embedder over fake reports and a fake QuickStats CSV,
then checks state filtering, QuickStats history and the plausibility check.

  python rag-pipeline/tests/smoke_test.py
"""
import csv
import os
import sys
import tempfile
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp())
os.environ.update({"RAG_DATASET": str(tmp / "dataset"), "RAG_INDEX": str(tmp / "idx"),
                   "QUICKSTATS_CSV": str(tmp / "qs.csv"), "EMBED_BACKEND": "hash"})
sys.path.insert(0, str(RAG_DIR))
import rag_pipeline as rp  # noqa: E402

(tmp / "dataset" / "maryland").mkdir(parents=True)
(tmp / "dataset" / "national").mkdir(parents=True)
(tmp / "dataset" / "maryland" / "MD_crop_progress_2023-09-11.txt").write_text(
    "Maryland Crop Progress\n\nCorn silage harvest was 85 percent complete by late September.", encoding="utf-8")
(tmp / "dataset" / "national" / "US_crop_progress_2023_w38.txt").write_text(
    "Crop Progress\n\nIowa Illinois Indiana Nebraska Texas corn harvested 15 percent.", encoding="utf-8")

COLS = ["year", "aggregation_level", "state_name", "state_ansi", "county_name", "county_ansi", "crop_use",
        "measurement", "source", "domain", "reference_period", "short_description", "value", "unit"]
SD = {"silage": "CORN, SILAGE - ACRES HARVESTED", "grain": "CORN, GRAIN - ACRES HARVESTED"}


def row(year, use, val, agg="STATE", county="", source="SURVEY", domain="TOTAL", ref="YEAR", sd=None):
    return {"year": year, "aggregation_level": agg, "state_name": "MARYLAND", "state_ansi": "24",
            "county_name": county, "county_ansi": "", "crop_use": use, "measurement": "harvested",
            "source": source, "domain": domain, "reference_period": ref, "short_description": sd or SD[use],
            "value": val, "unit": "ACRES"}


rows = []
for year, silage, grain in [(2018, 60000, 380000), (2019, 65000, 390000), (2020, 58000, 370000), (2021, 62000, 385000)]:
    rows += [row(year, "silage", f"{silage:,}"), row(year, "grain", f"{grain:,}")]
    # the traps found in the real file: forecasts, organic and irrigated subsets
    rows += [row(year, "grain", f"{grain:,}", ref=r) for r in ("YEAR - AUG FORECAST", "YEAR - SEP FORECAST", "YEAR - NOV FORECAST")]
    rows += [row(year, "grain", "2,316", source="CENSUS", domain="ORGANIC STATUS", sd="CORN, GRAIN, ORGANIC - ACRES HARVESTED")]
# a Census year: county rows only, SURVEY counties incl. OTHER COUNTIES, CENSUS counties with (D)
rows += [row(2022, "grain", "300,000", "COUNTY", "FREDERICK"), row(2022, "grain", "60,000", "COUNTY", "OTHER COUNTIES"),
         row(2022, "silage", "40,000", "COUNTY", "OTHER COUNTIES"),
         row(2022, "grain", "50,000", "COUNTY", "FREDERICK", source="CENSUS"),
         row(2022, "silage", "20,000", "COUNTY", "FREDERICK", source="CENSUS"),
         row(2022, "silage", "(D)", "COUNTY", "KENT", source="CENSUS"),
         row(2022, "grain", "(D)", "COUNTY", "FREDERICK", source="CENSUS", sd="CORN, GRAIN, IRRIGATED - ACRES HARVESTED")]
with open(tmp / "qs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)

n = rp.build_index(rp.DATASET_DIR, rp.INDEX_DIR, "hash")
assert n == 2, n
hits = rp.retrieve_reports("silage harvest", state="Maryland", k=5)
assert {h["state"] for h in hits} <= {"Maryland", "US"}, hits
assert any(h["state"] == "US" for h in hits) and any(h["state"] == "Maryland" for h in hits), hits
hist = rp.silage_share_history("Maryland")
by_year = {r["year"]: r for r in hist["by_year"]}
assert by_year[2021]["grain_acres"] == 385000, by_year[2021]  # forecasts and organic rows not added
assert by_year[2022]["silage_acres"] == 40000 and by_year[2022]["grain_acres"] == 360000, by_year[2022]
assert hist["summary"]["n_years"] == 5, hist
county = rp.silage_share_history("Maryland", county="Frederick")
assert county["by_year"][0]["note"] == "census" and county["by_year"][0]["silage_share"] == round(20000 / 70000, 4), county
assert rp.check_prediction("Maryland", 62000, 385000)["overall"] == "pass"
assert rp.check_prediction("Maryland", 200000, 250000)["overall"] == "fail"
print("smoke test passed:", n, "chunks,", len(hits), "hits")

# state tags inside national documents (two state tables can share a page)
page = ("Oct 15 Dec 1 (NA) Not available. "
        "Usual Planting and Harvesting Dates by Crop \u2013 Louisiana 2009 Corn for grain ..... 500 Mar 1 "
        "Usual Planting and Harvesting Dates by Crop \u2013 Maryland 2009 Corn for grain ..... 390 Apr 20 Sep 12 Nov 6")
segs = rp.state_segments(page, "US")
assert [st for st, _ in segs] == ["US", "Louisiana", "Maryland"], segs
assert "390" in segs[-1][1] and "390" not in segs[1][1]
assert rp.state_segments("Dates by Crop \u2013 Arkansas 2009 Corn for grain 410", "US")[0][0] == "Arkansas"
assert rp.state_segments("Dates by Crop \u2013 West Virginia 2009", "US")[0][0] == "West Virginia"
assert rp.state_segments("Dates by Crop \u2013 New England 2009", "US")[0][0] == "US"
assert rp.state_segments("Corn for Grain Usual Dates \u2013 States Alabama Arizona Arkansas", "US")[0][0] == "US"
assert rp.state_segments("anything", "Maryland")[0][0] == "Maryland"
print("state tagging ok")
