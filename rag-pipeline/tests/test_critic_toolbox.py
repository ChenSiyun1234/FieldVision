"""Offline test of the real critic: fake toolbox, fake QuickStats, no LLM, no GPU.

  python rag-pipeline/tests/test_critic_toolbox.py
"""
import csv
import os
import sys
import tempfile
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp())
os.environ.update({"QUICKSTATS_CSV": str(tmp / "qs.csv"), "RAG_INDEX": str(tmp / "idx"), "EMBED_BACKEND": "hash"})
sys.path.insert(0, str(RAG_DIR))
import critic_toolbox as ct  # noqa: E402

COLS = ["year", "aggregation_level", "state_name", "state_ansi", "county_name", "county_ansi", "crop_use",
        "measurement", "source", "domain", "reference_period", "short_description", "value", "unit"]
SD = {"silage": "CORN, SILAGE - ACRES HARVESTED", "grain": "CORN, GRAIN - ACRES HARVESTED"}
rows = []
for y, s, g in [(2018, 45000, 380000), (2019, 40000, 460000), (2020, 40000, 420000), (2021, 35000, 415000)]:
    for use, v in (("silage", s), ("grain", g)):
        rows.append({"year": y, "aggregation_level": "STATE", "state_name": "MARYLAND", "state_ansi": "24",
                     "county_name": "", "county_ansi": "", "crop_use": use, "measurement": "harvested",
                     "source": "SURVEY", "domain": "TOTAL", "reference_period": "YEAR",
                     "short_description": SD[use], "value": f"{v:,}", "unit": "ACRES"})
for y, s, g in [(2012, 9000, 50000), (2017, 10000, 48000), (2022, 11000, 52000)]:  # Frederick, Census years
    for use, v in (("silage", s), ("grain", g)):
        rows.append({"year": y, "aggregation_level": "COUNTY", "state_name": "MARYLAND", "state_ansi": "24",
                     "county_name": "FREDERICK", "county_ansi": "021", "crop_use": use, "measurement": "harvested",
                     "source": "CENSUS", "domain": "TOTAL", "reference_period": "YEAR",
                     "short_description": SD[use], "value": f"{v:,}", "unit": "ACRES"})
with open(tmp / "qs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)


class FakeToolbox:
    data_root = str(tmp / "counties")

    def __init__(self, agg):
        self.agg, self.calls = agg, []

    def scout(self, fips, year):
        return {"fips": fips, "year": year, "n_fields": self.agg["n_fields"]}

    def classify(self, fips, year, policy):
        self.calls.append(policy)
        return {"fips": fips, "year": year, "policy": policy}

    def aggregate(self, fips, year):
        return dict(self.agg, fips=fips, year=year)


assert ct.fips_to_names("24021") == ("Maryland", "FREDERICK"), ct.fips_to_names("24021")
assert ct.fips_to_names("24033")[0] == "Maryland"

# 1. plausible Frederick result -> pass / accept, compared against the county's own history
tb = ct.CriticToolbox(FakeToolbox({"silage_acres": 9500, "silage_acres_low": 9000, "silage_acres_high": 10000,
                                   "grain_acres": 50000, "abstained_acres": 2000, "n_fields": 2233}), use_llm=False)
tb.classify("24021", 2022, "standard"); tb.aggregate("24021", 2022)
r = tb.run_critic("24021", 2022)
assert r["overall"] == "pass" and r["suggested_action"] == "accept", r
assert r["checks"][0]["history_level"] == "county", r["checks"][0]
c0 = r["checks"][0]
assert c0["expected_range_pct"] == [round(c0["expected_range"][0] * 100, 1), round(c0["expected_range"][1] * 100, 1)]
# the supervisor's number guard reads "19.6%-32.1%" as -32.1; ranges must be written "A% to B%"
import re  # noqa: E402
assert not re.search(r"%-\d", r["summary"]), r["summary"]
assert tb.scout("24021", 2022)["n_fields"] == 2233  # pass-through works

# 2. silage far too high (50%) vs a 16-19% history: the regional rerun pushes toward 50%, so it cannot
#    help; fail goes straight to abstain (real case: Frederick 2022 went 33% -> 51% after the rerun)
tb = ct.CriticToolbox(FakeToolbox({"silage_acres": 30000, "grain_acres": 30000, "abstained_acres": 0,
                                   "n_fields": 2233}), use_llm=False)
tb.classify("24021", 2022, "standard"); tb.aggregate("24021", 2022)
r = tb.run_critic("24021", 2022)
assert r["overall"] == "fail" and r["suggested_action"] == "abstain_region", r

# 3. slightly high (21%, just over the 19.0% max): warn, no rerun, accept with warnings
tb = ct.CriticToolbox(FakeToolbox({"silage_acres": 13300, "grain_acres": 50000, "abstained_acres": 0,
                                   "n_fields": 2233}), use_llm=False)
tb.classify("24021", 2022, "standard"); tb.aggregate("24021", 2022)
r = tb.run_critic("24021", 2022)
assert r["overall"] == "warn" and r["suggested_action"] == "accept_with_warnings", r

# 3b. too LOW (5% vs 16-19%): pushing toward 50% heads into history, so the regional rerun is offered
tb = ct.CriticToolbox(FakeToolbox({"silage_acres": 2600, "grain_acres": 50000, "abstained_acres": 0,
                                   "n_fields": 2233}), use_llm=False)
tb.classify("24021", 2022, "standard"); tb.aggregate("24021", 2022)
r = tb.run_critic("24021", 2022)
assert r["suggested_action"] == "rerun_with_regional_threshold", r
tb.classify("24021", 2022, "regional_threshold"); tb.aggregate("24021", 2022)
r = tb.run_critic("24021", 2022)
assert r["suggested_action"] in ("abstain_region", "accept_with_warnings"), r  # no second rerun

# 4. tiny demo county with many abstentions, partial extraction file present
(tmp / "counties" / "24033" / "2022").mkdir(parents=True)
(tmp / "counties" / "24033" / "2022" / "PARTIAL.json").write_text('{"done": 248, "total": 2233}', encoding="utf-8")
tb = ct.CriticToolbox(FakeToolbox({"silage_acres": 47.7, "silage_acres_low": 16.1, "silage_acres_high": 73.5,
                                   "grain_acres": 25.8, "abstained_acres": 36.8, "n_fields": 7}), use_llm=False)
r = tb.run_critic("24033", 2022)
names = {c["check"]: c["status"] for c in r["checks"]}
assert names["abstained_share"] == "warn" and names["enough_fields"] == "info" and names["coverage"] == "info", names
assert names["silage_share_in_history"] == "warn", names  # capped: only 7 fields
assert r["suggested_action"] == "accept_with_warnings", r  # 65% vs 8-11%: regional rerun cannot help
assert r["summary"] and r["explained_by"] == "template"
print("critic toolbox tests passed")
print("example:", r["overall"], r["suggested_action"], "|", r["summary"][:160])
