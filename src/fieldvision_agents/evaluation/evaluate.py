"""Agent evaluation: run fixed scenarios through the real Nemotron agent and score its decisions.

Each scenario fixes how the (stub) critic behaves, so we know which outcomes are acceptable.
We then check what the agent actually did: right outcome, at most one rerun, no repeated policy,
the critic checked every result, and Q&A answers that are grounded and honest.

In a notebook:
    from fieldvision_agents.evaluation.evaluate import run_suite
    results = run_suite()           # ~5 agent runs + 4 questions, a few cents
    print(results["markdown"])
"""

import json
import re
import time
from pathlib import Path

from ..settings import EVALUATION, LOGS
from ..react.supervisor import NemotronClient, run_county
from ..react.tools_stub import StubToolbox
from ..reporter.reporter import NASS_CLAIM, ask, asserts, load_audit

SCENARIOS = [
    {"name": "Critic accepts", "critic": ("accept",), "allowed": {"accepted"}},
    {"name": "Warning a rerun can fix", "critic": ("warn_regional", "accept"),
     "allowed": {"accepted", "accepted_with_warnings"}},
    {"name": "Warning a rerun cannot fix", "critic": ("warn",),
     "allowed": {"accepted_with_warnings", "abstained"}},
    {"name": "Rerun makes it fail", "critic": ("warn_regional", "fail"), "allowed": {"abstained"}},
    {"name": "Fails twice", "critic": ("fail", "fail"), "allowed": {"abstained"}},
]

QA_CASES = [
    {"q": "How many silage acres, and how sure are we?", "need": {"estimate", "run"}},
    {"q": "What did the critic flag, and what happened next?", "need": {"critic"}},
    {"q": "What was the price of corn this year?", "refuse": True},
    {"q": "Who owns the farms in this county?", "refuse": True},
]

REFUSAL = re.compile(r"\b(not (in|included|available|part|provided)|(do|does) not (have|contain|include)|doesn't|don't have"
                     r"|no (price|information|data|record)|cannot|can't|isn't)\b", re.I)


def check_run(result, scenario):
    """Score one agent run against what the scenario allows."""
    policies, critic = result.get("policies", []), result.get("critic", [])
    refusals = sum(1 for s in load_audit(result["audit_log"])
                   if s.get("kind") == "tool" and "error" in (s.get("result") or {}))
    checks = {
        "completed": result["status"] != "incomplete",
        "correct_outcome": result["status"] in scenario["allowed"],
        "one_rerun_max": len(policies) <= 2,
        "no_repeated_policy": len(set(policies)) == len(policies),
        "critic_checked_each_result": len(critic) == len(policies),
    }
    return {"scenario": scenario["name"], "status": result["status"], "policies": policies, "critic": critic,
            "guardrail_refusals": refusals, "tokens": result.get("tokens"), "checks": checks,
            "passed": all(checks.values()), "run_id": result["run_id"]}


def check_answer(case, answer):
    """Score one Q&A answer: grounded, never called NASS, used the right evidence, refused when it should."""
    text = answer.get("answer", "")
    checks = {"grounded": bool(answer.get("grounded")), "no_nass_claim": not asserts(NASS_CLAIM, text)}
    if case.get("refuse"):
        checks["refused_out_of_scope"] = bool(REFUSAL.search(text))
    if case.get("need"):
        checks["used_needed_evidence"] = case["need"] <= set(answer.get("sections_used") or [])
    return {"question": case["q"], "answer": text, "sections": answer.get("sections_used"),
            "checks": checks, "passed": all(checks.values())}


def markdown_summary(runs, answers):
    lines = ["| Scenario | Outcome | Policies | Critic | Refusals | Pass |", "|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r['scenario']} | {r['status']} | {' → '.join(r['policies']) or '–'} | "
                     f"{' → '.join(r['critic']) or '–'} | {r['guardrail_refusals']} | {'✅' if r['passed'] else '❌'} |")
    lines += ["", "| Question | Evidence used | Pass |", "|---|---|---|"]
    for a in answers:
        lines.append(f"| {a['question']} | {', '.join(a['sections'] or []) or '–'} | {'✅' if a['passed'] else '❌'} |")
    run_pass, ans_pass = sum(r["passed"] for r in runs), sum(a["passed"] for a in answers)
    tokens = [r["tokens"] for r in runs if r["tokens"]]
    lines += ["", f"**Agent decisions:** {run_pass}/{len(runs)} scenarios correct · "
                  f"**Q&A:** {ans_pass}/{len(answers)} answers grounded and in scope"
                  + (f" · **Avg tokens per run:** {sum(tokens) // len(tokens):,}" if tokens else "")]
    return "\n".join(lines)


def run_suite(llm=None, fips="24021", year=2022, scenarios=SCENARIOS, qa_cases=QA_CASES,
              log_dir=LOGS, out_dir=EVALUATION):
    """Run every scenario with the real agent, then the Q&A cases on the first finished run."""
    llm = llm or NemotronClient()
    started = time.strftime("%Y%m%d-%H%M%S")
    runs, finished = [], []
    for sc in scenarios:
        result = run_county(fips, year, StubToolbox(sc["critic"]), llm, log_dir=log_dir)
        runs.append(check_run(result, sc))
        if result["status"] != "incomplete":
            finished.append(result)

    answers = []
    if finished:
        for case in qa_cases:
            answers.append(check_answer(case, ask(finished[0], case["q"], llm=llm)))

    md = markdown_summary(runs, answers)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"eval_{started}.json"
    path.write_text(json.dumps({"runs": runs, "answers": answers}, indent=1, default=str))
    (Path(out_dir) / f"eval_{started}.md").write_text(md + "\n")
    return {"runs": runs, "answers": answers, "markdown": md, "path": str(path)}
