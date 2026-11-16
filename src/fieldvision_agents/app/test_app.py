"""Offline tests for the app: the rule agent under the supervisor's guardrails, the offline report and
answers, the service layer and the web API. No network, no key.

    python -m fieldvision_agents.app.test_app
"""

import tempfile
from pathlib import Path

from ..evaluation.evaluate import SCENARIOS
from ..react.supervisor import run_county, ungrounded_numbers
from ..react.tools_stub import StubToolbox
from ..reporter.reporter import _allowed_numbers, build_evidence
from . import service
from .offline import RuleAgent, evidence_answer, template_report


def _run(critic, tmp, faults=()):
    return run_county("24021", 2022, StubToolbox(critic, faults), RuleAgent(), log_dir=tmp)


def test_rule_agent_meets_every_evaluation_scenario():
    """The same five scenarios the Nemotron agent is scored on, same allowed outcomes."""
    with tempfile.TemporaryDirectory() as tmp:
        for sc in SCENARIOS:
            result = _run(sc["critic"], tmp, sc.get("faults", ()))
            assert result["status"] in sc["allowed"], (sc["name"], result)
            assert len(result["policies"]) <= 2 and len(set(result["policies"])) == len(result["policies"])


def test_rule_agent_takes_the_suggested_rerun():
    with tempfile.TemporaryDirectory() as tmp:
        result = _run(("warn_regional", "accept"), tmp)
    assert result["policies"] == ["standard", "regional_threshold"] and result["status"] == "accepted"


def test_offline_report_and_answer_are_grounded():
    with tempfile.TemporaryDirectory() as tmp:
        result = _run(("warn",), tmp)
        rep = template_report(result, out_dir=tmp)
        allowed = _allowed_numbers(build_evidence(result))
        assert not ungrounded_numbers(rep["report_md"], allowed), rep["report_md"]
        assert Path(rep["path"]).is_file() and "not NASS statistics" in rep["report_md"]
        a = evidence_answer(result, "What did the critic flag?")
        assert a["sections_used"][0] == "critic" and "Critic 1" in a["answer"]


def test_service_lists_and_opens_runs():
    with tempfile.TemporaryDirectory() as tmp:
        result = _run(("accept",), tmp)
        listed = service.runs(log_dir=tmp)
        assert listed[0]["run_id"] == result["run_id"] and listed[0]["status"] == "accepted"
        d = service.detail(result["run_id"], log_dir=tmp)
        assert d["evidence"]["estimate"]["silage_acres"] and [s["tool"] for s in d["steps"]][0] == "scout"


def test_recommender_reads_the_final_module_predictions():
    """The agent's "standard" route saves predictions under the final module's name; the recommender must look there."""
    tb = service._with_recommender(type("Inner", (), {"final_policy": "module_v2"})())
    import silage_match
    seen = {}
    real = silage_match.recommend, silage_match.save
    silage_match.recommend = lambda fips, year, as_of, P, policy: seen.setdefault("policy", policy) and {
        "as_of": as_of, "overall": "pass", "funnel": {}, "county_silage_share_pct": None,
        "standing_price_usd_per_ton": None, "checks": [], "summary": "", "dairies": []}
    silage_match.save = lambda r: None
    try:
        tb.recommend_sources("24021", 2022, policy="standard")
        assert seen["policy"] == "module_v2"
        seen.clear()
        tb.recommend_sources("24021", 2022, policy="regional_threshold")
        assert seen["policy"] == "regional_threshold"
    finally:
        silage_match.recommend, silage_match.save = real


def test_dashboard_numbers_come_from_usda_history():
    from . import dashboard
    tiles = {t["id"]: t for t in dashboard.supply()["tiles"]}
    assert tiles["production"]["change_pct"] == -6.3 and tiles["stocks"]["change_pct"] == 35.1
    w = dashboard.what_if(1.0)
    assert w["acres"] > 0 and w["range_pct"][0] <= w["range_pct"][1] and "not a forecast" in w["note"]
    assert any(r["model"].startswith("no change") for r in dashboard.price_models())


class _OneToolLLM:
    """Calls silage_what_if once, then answers with the given text."""

    def __init__(self, answers):
        self.answers, self.n = list(answers), 0

    def chat(self, messages, tools=None):
        self.n += 1
        if self.n == 1:
            return {"content": "", "tool_calls": [{"id": "c1", "name": "silage_what_if", "arguments": '{"points": 1}'}]}
        return {"content": self.answers.pop(0), "tool_calls": []}


def test_chat_checks_every_number_and_retries_once():
    from . import chat
    w = __import__("fieldvision_agents.app.dashboard", fromlist=["what_if"]).what_if(1.0)
    good = f"About {w['acres']:,} acres would move to silage."
    r = chat.chat("usda", [{"role": "user", "content": "what if"}], {"offline": False}, llm=_OneToolLLM([good]))
    assert r["grounded"] and r["cards"][0]["type"] == "whatif" and r["tools"] == ["silage_what_if"]
    r = chat.chat("usda", [{"role": "user", "content": "what if"}], {"offline": False},
                  llm=_OneToolLLM(["Price rises 98765 percent.", "Still 98765."]))
    assert not r["grounded"] and "[unverified]" in r["reply"]


def test_offline_chat_routes_questions_to_tools():
    from . import chat
    ctx = {"offline": True}
    r = chat.chat("usda", [{"role": "user", "content": "What's the corn supply and price picture?"}], ctx)
    assert r["tools"] == ["supply_outlook"] and r["cards"][0]["type"] == "supply"
    r = chat.chat("usda", [{"role": "user", "content": "What if silage were 0.5 points lower?"}], ctx)
    assert r["tools"] == ["silage_what_if"] and r["cards"][0]["data"]["points"] == -0.5
    r = chat.chat("usda", [{"role": "user", "content": "How well has the agent tested?"}], ctx)
    assert r["tools"] == ["agent_quality"]


def test_dairy_finder_on_local_data():
    """Runs only where the Frederick 2022 county data is on disk (it is not in git)."""
    from ..settings import COUNTY_DATA
    if not (COUNTY_DATA / "24021" / "2022" / "timeseries.parquet").is_file():
        return
    from . import dairy
    r = dairy.find_silage("24021", 2022, "2022-09-15", lat=39.45, lon=-77.4, head=400, tons=1000, max_miles=15)
    assert r["dairy"]["matched_tons"] <= 1000 and all(f["road_miles"] <= 15 for f in r["fields"])
    assert len(r["map"]["features"]) == len(r["fields"])


def test_us_map_data_for_states_and_counties():
    from . import geo
    md = geo.state_summary("24")
    assert md["name"] == "Maryland" and md["nass_2025"]["silage"] == 40000 and "24021" in md["counties"]
    assert all(h["source"] == "SURVEY" for h in md["history"])
    fr = geo.county_detail("24021")
    census = {h["year"]: h["silage_share_pct"] for h in fr["history"] if h["source"] == "CENSUS"}
    assert census == {2012: 32.1, 2017: 24.3, 2022: 19.6}           # withheld duplicate rows ignored
    assert geo.county_detail("46059")["seasons"] == []                 # USDA data for any county, ours where we have it
    pa = geo.state_summary("42")
    assert any("Lancaster" in o["what"] and o["usda_pct"] == 27.4 for o in pa["ours"])   # the second state's county result
    from . import dashboard
    assert dashboard.silage()["lancaster_2023"]["census_year"] == 2022


def test_gap_checks_from_saved_results():
    from ..validation import gap_checks as gc
    assert gc.wilson(42, 44) == [84.9, 98.7]
    cb = gc.corn_belt()
    assert {r["state"] for r in cb["states"] if r["corn_belt"]} == {"WI", "IA", "NE"}
    bt = gc.blind_test()
    assert bt["blind_n"] >= 6 and all(r["hidden"].startswith(("yes", "no")) for r in bt["rows"])
    assert "not a forecast" in gc.price()["finding"]


def test_web_api_serves_the_page_and_status():
    try:
        from fastapi.testclient import TestClient
    except ImportError:                       # fastapi[test] needs httpx; skip quietly without it
        return
    from .server import app
    client = TestClient(app)
    assert client.get("/").status_code == 200
    s = client.get("/api/status").json()
    assert {"nemotron", "report_search", "county_data"} <= set(s)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
