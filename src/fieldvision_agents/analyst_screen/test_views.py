"""Offline tests for the analyst screen's views. No Nemotron calls, no widgets.

Run in the workspace terminal:  python -m fieldvision_agents.analyst_screen.test_views
"""

import tempfile

from ..react.supervisor import run_county
from ..react.test_supervisor import FIPS, YEAR, ScriptedLLM, call, summary_from_aggregate
from ..react.tools_stub import StubToolbox
from ..reporter.reporter import build_evidence
from .views import answer_card, steps_timeline, summary_card


def demo_run(tmp):
    """Warn -> rerun with regional thresholds -> accept, including one refused step."""
    steps = [
        [call("scout")], [call("route_options")], [call("extract_features")],
        [call("classify", policy="standard", reason="Most fields have clear series.")],
        [call("aggregate")], [call("run_critic")],
        [call("classify", policy="radar_first", reason="Try radar.")],                 # wrong policy: refused
        [call("classify", policy="regional_threshold", reason="Critic asked for regional thresholds.")],
        [call("aggregate")], [call("run_critic")],
        lambda msgs: [call("finish", status="accepted", policy_used="regional_threshold",
                           summary=summary_from_aggregate(msgs))],
    ]
    return run_county(FIPS, YEAR, StubToolbox(("warn_regional", "accept")), ScriptedLLM(steps), log_dir=tmp)


def test_summary_card_shows_status_and_acres():
    with tempfile.TemporaryDirectory() as tmp:
        run = demo_run(tmp)
        est = build_evidence(run)["estimate"]
        card = summary_card(run, "Frederick County, MD", seconds=12.3)
    assert "Accepted" in card and "Frederick County, MD" in card, card
    assert f"{est['silage_acres']:,}" in card and "standard → regional_threshold" in card
    assert "Placeholder test data" in card and "12 s" in card


def test_timeline_shows_reasons_and_refusals():
    with tempfile.TemporaryDirectory() as tmp:
        html = steps_timeline(demo_run(tmp)["audit_log"])
    assert "Why: Critic asked for regional thresholds." in html
    assert "Refused by guardrail" in html and "rerun_with_regional_threshold" in html


def test_answer_card_badges():
    good = answer_card("How much?", {"answer": "Our estimate is **4,837** acres.", "sections_used": ["estimate"],
                                     "grounded": True})
    bad = answer_card("How much?", {"answer": "[unverified] acres", "sections_used": [], "grounded": False})
    assert "every number checked" in good and "**" not in good
    assert "[unverified]" in bad and "unverified] numbers" in bad


def test_html_is_escaped():
    card = answer_card("<script>x</script>", {"answer": "<b>hi</b>", "sections_used": [], "grounded": True})
    assert "<script>" not in card and "&lt;script&gt;" in card


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
