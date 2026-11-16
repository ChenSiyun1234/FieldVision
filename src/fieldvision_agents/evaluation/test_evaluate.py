"""Offline tests for the evaluation scoring. No Nemotron calls.

Run in the workspace terminal:  python -m fieldvision_agents.evaluation.test_evaluate
"""

import tempfile

from ..analyst_screen.test_views import demo_run
from .evaluate import QA_CASES, SCENARIOS, check_answer, check_run, markdown_summary


def test_good_run_passes():
    with tempfile.TemporaryDirectory() as tmp:
        result = demo_run(tmp)                   # warn_regional -> regional rerun -> accepted, one refusal
        scored = check_run(result, SCENARIOS[1])
    assert scored["passed"], scored
    assert scored["guardrail_refusals"] == 1 and scored["policies"] == ["standard", "regional_threshold"]


def test_wrong_outcome_fails():
    with tempfile.TemporaryDirectory() as tmp:
        result = demo_run(tmp)                   # accepted, but this scenario requires abstaining
        scored = check_run(result, SCENARIOS[4])
    assert not scored["passed"] and not scored["checks"]["correct_outcome"]


def test_answer_checks():
    ok = check_answer(QA_CASES[0], {"answer": "Our model estimates 4,837 acres; the run was accepted.",
                                    "sections_used": ["estimate", "run", "critic"], "grounded": True})
    nass = check_answer(QA_CASES[0], {"answer": "The NASS estimate is 4,837 acres.",
                                      "sections_used": ["estimate", "run"], "grounded": True})
    refuse = check_answer(QA_CASES[2], {"answer": "The run record does not include price data.",
                                        "sections_used": ["run"], "grounded": True})
    invent = check_answer(QA_CASES[2], {"answer": "Corn sold for about four dollars a bushel.",
                                        "sections_used": [], "grounded": True})
    assert ok["passed"] and not nass["passed"] and refuse["passed"] and not invent["passed"]


def test_negated_claims_and_plain_refusals_count_correctly():
    bolt = check_answer(QA_CASES[2], {"answer": "I do not have information about the price of corn in the provided evidence.",
                                      "sections_used": [], "grounded": True})
    negated = check_answer(QA_CASES[0], {"answer": "Our model estimates 4,837 acres. The official NASS figure is not available yet.",
                                         "sections_used": ["estimate", "run"], "grounded": True})
    assert bolt["passed"] and negated["passed"], (bolt, negated)


def test_markdown_has_totals():
    with tempfile.TemporaryDirectory() as tmp:
        runs = [check_run(demo_run(tmp), SCENARIOS[1])]
    md = markdown_summary(runs, [])
    assert "1/1 scenarios correct" in md and "| Warning a rerun can fix |" in md


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
