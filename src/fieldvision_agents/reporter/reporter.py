"""Reporter and analyst Q&A on top of a finished supervisor run.

Code builds an evidence pack from the audit log. Nemotron writes the report (writer),
a second pass corrects it against the evidence (editor), and code checks every number
at the end, marking any it cannot find as [unverified]. The analyst Q&A agent answers
questions by deciding which evidence sections to fetch.

Usage (in the workspace):
    from fieldvision_agents.reporter.reporter import write_report, ask
    report = write_report(result)          # result = run_county(...)
    print(report["report_md"])
    ask(result, "Why did the agent rerun this county?")
"""

import json
import re
from pathlib import Path

from ..settings import REPORTS
from ..react.supervisor import DEFAULT_MODEL, JUDGE_MODEL, NemotronClient, _numbers_in, ungrounded_numbers

SECTIONS = ("run", "coverage", "routing_decisions", "estimate", "critic", "abstained", "refused_steps")


def load_audit(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def result_from_log(path):
    """Rebuild a run_county() result from a saved audit log, so any past run can be reported on."""
    final = next(s for s in reversed(load_audit(path)) if s.get("kind") == "final")
    return {"run_id": final["run_id"], "status": final["status"], "policy_used": final.get("policy_used"),
            "summary": final.get("summary"), "policies": final.get("policies", []),
            "critic": final.get("critic", []), "audit_log": str(path), "tokens": final.get("tokens")}


def _clean(result):
    """Drop bookkeeping keys that mean nothing to an analyst."""
    return {k: v for k, v in (result or {}).items() if k not in ("stub", "fips", "year")}


def build_evidence(run_result):
    """Everything the reporter may say, taken only from the audit log and run result."""
    steps = load_audit(run_result["audit_log"])
    tools = [s for s in steps if s.get("kind") == "tool"]
    ok = [s for s in tools if "error" not in (s.get("result") or {})]

    def results(name):
        return [s["result"] for s in ok if s["tool"] == name]

    first_args = next((s["args"] for s in tools if s.get("args", {}).get("fips")), {})
    scout, aggregate = results("scout"), results("aggregate")
    abstain = results("abstain_region")
    return {
        "county": {"fips": first_args.get("fips"), "year": first_args.get("year")},
        "run": {"run_id": run_result["run_id"], "status": run_result["status"],
                "policy_sequence": run_result.get("policies", []),
                "critic_sequence": run_result.get("critic", []),
                "agent_summary": run_result.get("summary")},
        "coverage": _clean(scout[-1]) if scout else None,
        "routing_decisions": [{"order": i + 1, "policy": s["args"].get("policy"), "reason": s["args"].get("reason")}
                              for i, s in enumerate(x for x in ok if x["tool"] == "classify")],
        "estimate": _clean(aggregate[-1]) if aggregate else None,
        "critic": [{k: c.get(k) for k in ("overall", "suggested_action", "flags", "summary") if k in c}
                   for c in results("run_critic")],
        "abstained": {"reason": abstain[-1].get("reason")} if abstain else None,
        "refused_steps": [{"tool": s["tool"], "error": s["result"]["error"]} for s in tools
                          if "error" in (s.get("result") or {})],
        "is_stub": any((s.get("result") or {}).get("stub") for s in tools),
    }


def _strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _allowed_numbers(evidence):
    allowed = set()
    _numbers_in(evidence, allowed)
    return allowed


def _mark_unverified(text, bad):
    for raw in sorted(set(bad), key=len, reverse=True):
        text = re.sub(rf"(?<![\w.]){re.escape(raw)}(?![\w])", "[unverified]", text)
    return text


WRITER_PROMPT = """You write a short county-season report for a USDA NASS statistician, using only an evidence JSON.
Rules: use only facts and numbers in the evidence. Copy numbers exactly; never compute, round or add numbers.
Format (markdown, under 220 words):
## County <fips> · <year>: <one-line headline that states run.status>
**Estimate:** silage acres with the low to high range, grain acres, abstained acres.
**Data:** what the scout found.
**How the agent decided:** each routing decision in order, with its reason.
**Critic:** each verdict in order, with the flags and any cited sources. Silage-share history ranges come from
USDA Quick Stats; cite a report passage only as background (harvest timing or pace), never as the source of a number.
**Confidence and limits:** what makes this more or less certain. If is_stub is true, say the numbers are placeholder test data.
"""

EDITOR_PROMPT = """You are the editor. Correct the draft so every statement matches the evidence JSON.
- Remove or correct every number in the suspect list; they are not in the evidence.
- Describe policies in the order of run.policy_sequence and critic verdicts in the order of run.critic_sequence.
- Do not add facts, sections or numbers. Keep the format and length.
Return only the corrected markdown.
"""


def write_report(run_result, writer=None, editor=None, out_dir=REPORTS):
    """Writer -> editor -> code grounding check. Saves the report and its evidence next to each other."""
    writer = writer or NemotronClient(model=JUDGE_MODEL, max_tokens=1500)
    editor = editor or writer
    evidence = build_evidence(run_result)
    allowed = _allowed_numbers(evidence)
    ev_json = json.dumps(evidence, indent=1, default=str)

    draft = _strip_thinking(writer.chat([
        {"role": "system", "content": WRITER_PROMPT},
        {"role": "user", "content": f"Evidence JSON:\n{ev_json}"}])["content"])
    suspects = ungrounded_numbers(draft, allowed)

    edited = _strip_thinking(editor.chat([
        {"role": "system", "content": EDITOR_PROMPT},
        {"role": "user", "content": f"Evidence JSON:\n{ev_json}\n\nSuspect numbers: {suspects}\n\nDraft:\n{draft}"}])["content"])
    edited = edited or draft
    remaining = ungrounded_numbers(edited, allowed)
    final = _mark_unverified(edited, remaining) if remaining else edited

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(out_dir) / f"report_{run_result['run_id']}"
    stem.with_suffix(".md").write_text(final + "\n")
    stem.with_suffix(".evidence.json").write_text(ev_json + "\n")
    return {"report_md": final, "draft_md": draft,
            "grounding": {"suspects_in_draft": suspects, "unverified_in_final": remaining,
                          "grounded": not remaining},
            "path": str(stem.with_suffix(".md"))}


QA_PROMPT = """You answer a USDA NASS analyst's questions about one finished county-season run.
Fetch evidence with get_evidence before answering; choose only the sections you need.
- The acres are this pipeline's model estimate, not official NASS statistics. Never call them NASS figures.
- For any question about confidence or certainty, also fetch "run" and "critic", and state the run status
  (for example abstained, or accepted with warnings) and the critic's verdicts.
- Describe events in the order they happened; run.policy_sequence and run.critic_sequence give the order.
Rules: every number must be copied from fetched evidence. If the answer is not in the run record, say so.
Answer in 2 to 4 sentences and name the sections you used, e.g. (sources: estimate, critic).
"""

# Catches answers that present our model output as an official NASS number.
NASS_CLAIM = re.compile(r"\bNASS(?:'s)?\s+(?:estimate|figure|number|statistic|report)s?\b", re.I)
NEGATION = re.compile(r"\b(not|no|never|none|isn't|aren't|doesn't|don't|cannot|can't|without|unavailable)\b", re.I)


def asserts(pattern, text):
    """True if some sentence matches the pattern without negating it ("not available" is not a claim)."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return any(pattern.search(s) and not NEGATION.search(s) for s in sentences)

QA_TOOLS = [{"type": "function", "function": {
    "name": "get_evidence",
    "description": "Fetch one section of the run's evidence. Sections: run (status, policy and critic order), "
                   "coverage (scout), routing_decisions (policy and reason per classify), estimate (acres and range), "
                   "critic (verdicts, flags, cited sources), abstained, refused_steps (guardrail refusals).",
    "parameters": {"type": "object",
                   "properties": {"section": {"type": "string", "enum": list(SECTIONS)}},
                   "required": ["section"]}}}]


def ask(run_result, question, llm=None, max_steps=5):
    """Agentic Q&A: the model decides which evidence sections to fetch, then answers; numbers are checked."""
    llm = llm or NemotronClient(model=DEFAULT_MODEL, max_tokens=700)
    evidence = build_evidence(run_result)
    messages = [{"role": "system", "content": QA_PROMPT}, {"role": "user", "content": question}]
    used, allowed, retried = [], set(), False

    for _ in range(max_steps):
        reply = llm.chat(messages, QA_TOOLS)
        calls = reply["tool_calls"]
        messages.append({"role": "assistant", "content": reply["content"],
                         **({"tool_calls": [{"id": c["id"], "type": "function",
                                             "function": {"name": c["name"], "arguments": c["arguments"]}}
                                            for c in calls]} if calls else {})})
        if calls:
            for c in calls:
                try:
                    section = json.loads(c["arguments"] or "{}").get("section")
                except json.JSONDecodeError:
                    section = None
                data = evidence.get(section) if section in SECTIONS else {"error": f"section must be one of {list(SECTIONS)}"}
                if section in SECTIONS:
                    used.append(section)
                    _numbers_in(data, allowed)
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": json.dumps(data, default=str)})
            continue

        answer = _strip_thinking(reply["content"])
        _numbers_in(evidence.get("county"), allowed)
        bad = ungrounded_numbers(answer, allowed)
        problems = []
        if bad:
            problems.append(f"These numbers are not in the evidence you fetched: {bad}. "
                            "Fetch the right section or remove them.")
        if asserts(NASS_CLAIM, answer):
            problems.append("These acres are our pipeline's estimate, not NASS statistics. Do not attribute them to NASS.")
        if problems and not retried:
            retried = True
            messages.append({"role": "user", "content": " ".join(problems) + " Answer again."})
            continue
        return {"answer": _mark_unverified(answer, bad) if bad else answer,
                "sections_used": used, "grounded": not bad}

    return {"answer": "I could not answer from the run record.", "sections_used": used, "grounded": True}
