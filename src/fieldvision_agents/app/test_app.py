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


def _run(critic, tmp):
    return run_county("24021", 2022, StubToolbox(critic), RuleAgent(), log_dir=tmp)


def test_rule_agent_meets_every_evaluation_scenario():
    """The same five scenarios the Nemotron agent is scored on, same allowed outcomes."""
    with tempfile.TemporaryDirectory() as tmp:
        for sc in SCENARIOS:
            result = _run(sc["critic"], tmp)
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
