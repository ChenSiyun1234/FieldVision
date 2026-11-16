"""Offline tests for the reporter and Q&A. No Nemotron calls, no credits.

Run in the workspace terminal:  python -m fieldvision_agents.reporter.test_reporter
"""

import tempfile

from .reporter import ask, build_evidence, write_report
from ..react.supervisor import run_county
from ..react.test_supervisor import FIPS, YEAR, ScriptedLLM, call, summary_from_aggregate
from ..react.tools_stub import StubToolbox


class FakeLLM:
    """Returns scripted replies: a string is a text answer, a list is tool calls."""

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, tools=None):
        r = self.replies.pop(0)
        if isinstance(r, list):
            return {"content": "", "tool_calls": r, "usage": None, "cached": False}
        return {"content": r, "tool_calls": [], "usage": None, "cached": False}


def finished_run(tmp):
    """A real supervisor run (scripted agent, stub tools) with a warn -> rerun -> accept path."""
    steps = [
        [call("scout")], [call("route_options")], [call("extract_features")],
        [call("classify", policy="standard", reason="Most fields have clear series.")],
        [call("aggregate")], [call("run_critic")],
        [call("classify", policy="radar_first", reason="Critic asked for radar on cloudy fields.")],
        [call("aggregate")], [call("run_critic")],
        lambda msgs: [call("finish", status="accepted", policy_used="radar_first",
                           summary=summary_from_aggregate(msgs))],
    ]
    return run_county(FIPS, YEAR, StubToolbox(("warn", "accept")), ScriptedLLM(steps), log_dir=tmp)


def test_evidence_comes_from_audit_log():
    with tempfile.TemporaryDirectory() as tmp:
        ev = build_evidence(finished_run(tmp))
    assert ev["run"]["policy_sequence"] == ["standard", "radar_first"], ev["run"]
    assert [d["reason"] for d in ev["routing_decisions"]][1] == "Critic asked for radar on cloudy fields."
    assert ev["estimate"]["silage_acres"] > 0 and ev["is_stub"] is True
    assert [c["overall"] for c in ev["critic"]] == ["warn", "pass"]


def test_editor_fixes_and_code_marks_leftovers():
    with tempfile.TemporaryDirectory() as tmp:
        run = finished_run(tmp)
        silage = build_evidence(run)["estimate"]["silage_acres"]
        writer = FakeLLM([f"Silage {silage} acres, maybe 999999 in total."])
        editor = FakeLLM([f"Silage {silage} acres, maybe 999999 in total."])   # editor misses it
        rep = write_report(run, writer=writer, editor=editor, out_dir=tmp)
    assert rep["grounding"]["suspects_in_draft"] == ["999999"], rep["grounding"]
    assert "[unverified]" in rep["report_md"] and str(silage) in rep["report_md"], rep["report_md"]
    assert rep["grounding"]["grounded"] is False


def test_clean_report_is_grounded():
    with tempfile.TemporaryDirectory() as tmp:
        run = finished_run(tmp)
        est = build_evidence(run)["estimate"]
        text = f"Silage {est['silage_acres']} acres ({est['silage_acres_low']} to {est['silage_acres_high']})."
        rep = write_report(run, writer=FakeLLM([text]), editor=FakeLLM([text]), out_dir=tmp)
    assert rep["grounding"]["grounded"] is True and rep["report_md"] == text


def test_qa_fetches_then_answers():
    with tempfile.TemporaryDirectory() as tmp:
        run = finished_run(tmp)
        silage = build_evidence(run)["estimate"]["silage_acres"]
        llm = FakeLLM([[call("get_evidence", section="estimate")],
                       f"Estimated silage is {silage} acres (sources: estimate)."])
        out = ask(run, "How much silage?", llm=llm)
    assert out["grounded"] is True and out["sections_used"] == ["estimate"], out


def test_qa_invented_number_gets_one_retry():
    with tempfile.TemporaryDirectory() as tmp:
        run = finished_run(tmp)
        silage = build_evidence(run)["estimate"]["silage_acres"]
        llm = FakeLLM([[call("get_evidence", section="estimate")],
                       "Silage is 777777 acres.",                                   # invented
                       f"Silage is {silage} acres (sources: estimate)."])
        out = ask(run, "How much silage?", llm=llm)
    assert out["grounded"] is True and "777777" not in out["answer"], out


def test_qa_does_not_call_our_estimate_nass():
    with tempfile.TemporaryDirectory() as tmp:
        run = finished_run(tmp)
        silage = build_evidence(run)["estimate"]["silage_acres"]
        llm = FakeLLM([[call("get_evidence", section="estimate")],
                       f"The NASS estimate is {silage} silage acres.",              # wrong attribution
                       f"Our pipeline estimates {silage} silage acres (sources: estimate)."])
        out = ask(run, "How much silage?", llm=llm)
    assert "NASS" not in out["answer"] and str(silage) in out["answer"], out


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
