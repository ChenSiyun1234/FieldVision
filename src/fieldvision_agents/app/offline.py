"""Offline mode: the same pipeline with no language model, for machines without an NVIDIA key.

RuleAgent stands in for Nemotron inside the unchanged supervisor loop. It calls the tools in the
documented order, takes the critic's one suggested rerun, abstains on a failed result, and copies
the final numbers from the aggregate. Every guardrail in the supervisor still applies to it.

template_report and evidence_answer replace the writer and the analyst Q&A agent: both only lay out
what build_evidence returns, so every number in them comes from the run record.
"""

import json
from pathlib import Path

from ..react.prompts import RERUN_POLICY
from ..reporter.reporter import build_evidence
from ..settings import REPORTS

OFFLINE_NOTE = "Offline mode: rule-based agent, no language model."


def _calls_and_results(messages):
    """(tool name, arguments, result) for every tool call so far, in order."""
    names, out = {}, []
    for m in messages:
        for tc in m.get("tool_calls") or []:
            names[tc["id"]] = (tc["function"]["name"], json.loads(tc["function"]["arguments"] or "{}"))
        if m.get("role") == "tool" and m.get("tool_call_id") in names:
            name, args = names[m["tool_call_id"]]
            out.append((name, args, json.loads(m["content"])))
    return out


def _fmt(x):
    """A number exactly as the tool returned it, so the supervisor's grounding check finds it."""
    return "n/a" if x is None else str(x)


class RuleAgent:
    """A deterministic agent with NemotronClient's chat() interface."""

    model = "offline-rule-agent"

    def __init__(self):
        self._n = 0

    def _call(self, name, thought, **args):
        """One tool call; `thought` is what Nemotron would write beside it (kept in the audit log)."""
        self._n += 1
        return {"content": thought, "usage": None, "cached": False,
                "tool_calls": [{"id": f"rule-{self._n}", "name": name, "arguments": json.dumps(args)}]}

    def chat(self, messages, tools=None):
        history = _calls_and_results(messages)
        done = [name for name, _, r in history if "error" not in r]
        last_name, _, last = history[-1] if history else (None, {}, {})
        classified = [a.get("policy") for n, a, r in history if n == "classify" and "error" not in r]

        if "scout" not in done:
            return self._call("scout", "Start by seeing what imagery the fields have.")
        if "route_options" not in done:
            return self._call("route_options", "Compare the routing policies.")
        if "extract_features" not in done:
            return self._call("extract_features", "Compute the field features.")
        if not classified:
            options = next(r for n, _, r in history if n == "route_options").get("options") or {}
            usable = [p for p, o in options.items()
                      if isinstance(o, dict) and o.get("available", True) is not False
                      and not str(o.get("note", "")).lower().startswith("not available")]
            policy = "standard" if "standard" in usable or not usable else usable[0]
            return self._call("classify", "Use the default route.", policy=policy,
                              reason=f"Offline rule: the '{policy}' route, the final model module.")
        if last_name == "classify" and "error" not in last:
            return self._call("aggregate", "Add the fields up to county acres.")
        if last_name == "aggregate":
            return self._call("run_critic", "Check the county result.")
        if last_name == "abstain_region" and "error" not in last:
            return self._finish("abstained", classified, history)
        if last_name == "finish" and "error" in last:          # the loop refused the finish (e.g. nothing judged)
            return self._call("abstain_region", "The loop refused the finish.", reason=last["error"])

        critic = next((r for n, _, r in reversed(history) if n == "run_critic" and "error" not in r), None)
        if critic is None:
            return self._call("run_critic", "Check the county result.")
        rerun = RERUN_POLICY.get(critic.get("suggested_action"))
        refused_rerun = last_name == "classify" and "error" in last
        if rerun and len(classified) == 1 and rerun != classified[-1] and not refused_rerun:
            return self._call("classify", "The critic suggested one rerun.", policy=rerun,
                              reason=f"Offline rule: the critic suggested {critic['suggested_action']}.")
        if critic.get("overall") == "fail":
            if last_name == "abstain_region":            # refused: the loop says why; finish incomplete
                return self._finish("abstained", classified, history)
            return self._call("abstain_region", "The critic failed the result.",
                              reason=critic.get("summary") or "The critic failed the county result.")
        status = "accepted" if critic.get("suggested_action") == "accept" and critic.get("overall") == "pass" \
            else "accepted_with_warnings"
        return self._finish(status, classified, history)

    def _finish(self, status, classified, history):
        agg = next((r for n, _, r in reversed(history) if n == "aggregate" and "error" not in r), {})
        summary = (f"Silage {_fmt(agg.get('silage_acres'))} acres (range {_fmt(agg.get('silage_acres_low'))} "
                   f"to {_fmt(agg.get('silage_acres_high'))}), grain {_fmt(agg.get('grain_acres'))} acres, "
                   f"abstained {_fmt(agg.get('abstained_acres'))} acres. {OFFLINE_NOTE}")
        if status == "abstained":
            summary = "Abstained after the critic failed the result. " + OFFLINE_NOTE
        return self._call("finish", "Close the run.", status=status,
                          policy_used=classified[-1] if classified else "none", summary=summary)


def _acres(x):
    return "n/a" if x is None else f"{x:,}"


def template_report(run_result, out_dir=REPORTS):
    """The report's sections, filled from the evidence pack. Same return shape as reporter.write_report."""
    ev = build_evidence(run_result)
    county, run, est, cov = ev["county"], ev["run"], ev.get("estimate") or {}, ev.get("coverage") or {}
    lines = [f"## County {county['fips']} · {county['year']}: {run['status']}", ""]
    if est:
        lines.append(f"**Estimate:** silage acres {_acres(est.get('silage_acres'))} "
                     f"(range {_acres(est.get('silage_acres_low'))} to {_acres(est.get('silage_acres_high'))}), "
                     f"grain acres {_acres(est.get('grain_acres'))}, abstained acres {_acres(est.get('abstained_acres'))}.")
        if est.get("adjustment"):
            lines.append(f"Totals: {est['adjustment']}.")
    if cov:
        lines.append(f"**Data:** {_acres(cov.get('n_fields'))} fields; {_acres(cov.get('fields_with_clear_series'))} "
                     f"with a clear optical series, {_acres(cov.get('fields_with_cloud_gap_radar_ok'))} with a cloud gap, "
                     f"{_acres(cov.get('fields_with_little_data'))} with little data.")
    if ev["routing_decisions"]:
        lines.append("**How the agent decided:** " + " ".join(
            f"{d['order']}. {d['policy']}: {d['reason']}" for d in ev["routing_decisions"]))
    for i, c in enumerate(ev["critic"], 1):
        flags = "; ".join(f"{f.get('check')} ({f.get('status')}): {(f.get('why') or '').rstrip('.')}"
                          for f in c.get("flags") or []) + "."
        lines.append(f"**Critic {i}:** {c.get('overall')}, suggested {c.get('suggested_action')}."
                     + (f" Flags: {flags}" if flags else ""))
    if ev.get("abstained"):
        lines.append(f"**Abstained:** {ev['abstained']['reason']}")
    if ev["refused_steps"]:
        lines.append(f"**Guardrails:** {len(ev['refused_steps'])} step(s) refused: "
                     + "; ".join(s["error"] for s in ev["refused_steps"]))
    lines.append(f"_These are model estimates, not NASS statistics. {OFFLINE_NOTE}_")
    if ev.get("is_stub"):
        lines.append("_Placeholder test data: the tools were stubs._")
    text = "\n\n".join(l for l in lines if l)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(out_dir) / f"report_{run_result['run_id']}"
    stem.with_suffix(".md").write_text(text + "\n")
    stem.with_suffix(".evidence.json").write_text(json.dumps(ev, indent=1, default=str) + "\n")
    return {"report_md": text, "draft_md": text, "path": str(stem.with_suffix(".md")),
            "grounding": {"suspects_in_draft": [], "unverified_in_final": [], "grounded": True}}


# Question words -> evidence sections; the first match wins, "confidence" questions get three.
_TOPICS = [
    (("sure", "confiden", "certain", "trust", "uncertain"), ("run", "estimate", "critic")),
    (("critic", "flag", "warn", "fail", "check"), ("critic", "run")),
    (("rerun", "policy", "route", "routing", "why did", "decid"), ("routing_decisions", "run")),
    (("abstain",), ("abstained", "run")),
    (("cloud", "data", "field", "imagery", "observation", "coverage"), ("coverage",)),
    (("refus", "guardrail", "rule"), ("refused_steps",)),
    (("acre", "silage", "grain", "how many", "estimate", "range"), ("estimate",)),
]


def evidence_answer(run_result, question):
    """Answer by quoting the evidence sections the question is about. Same shape as reporter.ask."""
    ev = build_evidence(run_result)
    q = question.lower()
    sections = next((s for words, s in _TOPICS if any(w in q for w in words)), ("run", "estimate"))
    parts = []
    for s in sections:
        data = ev.get(s)
        if not data:
            continue
        if s == "estimate":
            parts.append(f"Model estimate: silage {_acres(data.get('silage_acres'))} acres (range "
                         f"{_acres(data.get('silage_acres_low'))} to {_acres(data.get('silage_acres_high'))}), "
                         f"grain {_acres(data.get('grain_acres'))}, abstained {_acres(data.get('abstained_acres'))}.")
        elif s == "run":
            parts.append(f"Run status {data['status']}; policies {' → '.join(data['policy_sequence']) or 'none'}; "
                         f"critic {' → '.join(c or '?' for c in data['critic_sequence']) or 'not run'}.")
        elif s == "critic":
            for i, c in enumerate(data, 1):
                flags = "; ".join(f"{f.get('check')}: {(f.get('why') or '').rstrip('.')}" for f in c.get("flags") or []) + "."
                parts.append(f"Critic {i}: {c.get('overall')} ({c.get('suggested_action')})." + (f" {flags}" if flags else ""))
        elif s == "routing_decisions":
            parts += [f"Step {d['order']}: {d['policy']}, because {d['reason']}" for d in data]
        elif s == "coverage":
            parts.append(f"{_acres(data.get('n_fields'))} fields; {_acres(data.get('fields_with_clear_series'))} clear, "
                         f"{_acres(data.get('fields_with_little_data'))} with little data.")
        elif s == "abstained":
            parts.append(f"Abstained: {data.get('reason')}")
        elif s == "refused_steps":
            parts += [f"Refused {r['tool']}: {r['error']}" for r in data] or ["No step was refused."]
    answer = " ".join(parts) or "The run record does not answer that."
    return {"answer": answer + " (Not NASS statistics. " + OFFLINE_NOTE + ")",
            "sections_used": [s for s in sections if ev.get(s)], "grounded": True}
