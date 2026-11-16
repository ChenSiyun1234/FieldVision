"""Field-level analyst Q&A: answers about individual fields with FieldVision's own evidence.

The model decides which tools to call: the run's evidence sections (from the audit log), lists of
fields by kind, or one field's evidence (drop date, greenness before and after, residue, the model's
own one-line reason). Every number in the answer is checked against what the tools returned.
"""

import json

from ..react.supervisor import NemotronClient, _numbers_in, ungrounded_numbers
from ..reporter.reporter import NASS_CLAIM, SECTIONS, asserts, build_evidence

KINDS = ("silage_like", "grain_like", "abstained", "most_uncertain", "largest")

FIELD_QA_PROMPT = """You answer a USDA NASS analyst's questions about the fields in one finished county-season run.
Use the tools: run_evidence for the county-level record, list_fields to find fields of a kind, field_detail for
one field's satellite evidence. Fetch what you need before answering.
- Every number must come from a tool result or the question. These are model estimates, not NASS statistics.
- A field's call (silage or grain) comes from the model's own probability, p_silage_model, at 0.5. Quote that
  number when you explain one field.
- p_silage_county_prior, when present, is shifted to the county's usual silage share and is used only for county
  totals. Never say it supports, confirms or contradicts a field's call.
- Name each satellite value by its own key: ndvi is greenness, ndmi is moisture, ndti is crop residue. Never
  give one index's value under another index's name.
- Explain a field's call in plain words from its evidence (when the green canopy dropped, how green it was
  before, bare soil or residue after). If the run record does not answer the question, say so.
- search_reports (when offered) returns NASS report passages for general context, such as usual harvest dates
  or the season's harvest pace. Cite each passage you use by source and page. They describe the state or
  nation, not these fields, and are never a source for this run's acres.
- Answer in 2 to 5 sentences and name the tools you used.
"""


def _tool(name, desc, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


FIELD_QA_TOOLS = [
    _tool("run_evidence", "One section of the county-level run record.",
          {"section": {"type": "string", "enum": list(SECTIONS)}}, ["section"]),
    _tool("list_fields", "Fields of one kind, largest first: silage_like, grain_like, abstained, most_uncertain, largest.",
          {"kind": {"type": "string", "enum": list(KINDS)}, "limit": {"type": "integer"}}, ["kind"]),
    _tool("field_detail", "One field's call, probabilities, area and satellite evidence.",
          {"field_id": {"type": "string"}}, ["field_id"]),
]


SEARCH_REPORTS = _tool("search_reports", "NASS report passages (Crop Progress, harvest-date handbooks) with source and page.",
                       {"query": {"type": "string"}, "k": {"type": "integer"}}, ["query"])


def list_fields(df, kind, limit=5):
    """Kinds follow each field's call, which is the model's own p_silage at 0.5."""
    decided = df[~df["abstain"]]
    if kind == "silage_like":
        pick = decided[decided["p_silage"] >= 0.5]
    elif kind == "grain_like":
        pick = decided[decided["p_silage"] < 0.5]
    elif kind == "abstained":
        pick = df[df["abstain"]]
    elif kind == "most_uncertain":
        pick = decided.assign(d=(decided["p_silage"] - 0.5).abs()).sort_values("d")
    else:
        pick = df
    if kind != "most_uncertain" and "area_acres" in pick:
        pick = pick.sort_values("area_acres", ascending=False)
    rows = []
    for _, r in pick.head(max(1, min(int(limit), 20))).iterrows():
        ev = r.get("evidence") or {}
        area = r.get("area_acres")
        row = {"field_id": r["field_id"],
               "area_acres": None if area is None or area != area else round(float(area), 1),
               "call": r.get("label"), "p_silage_model": round(float(r["p_silage"]), 3), "abstain": bool(r["abstain"]),
               "abstain_reason": ev.get("abstain_reason"), "drop_date": ev.get("drop_date")}
        if df.attrs.get("prior_applied"):
            row["p_silage_county_prior"] = round(float(r["p_adjusted"]), 3)
        rows.append(row)
    return {"kind": kind, "count": int(len(pick)), "fields": rows}


def ask_fields(toolbox, fips, year, run_result, question, llm=None, max_steps=6):
    """Agentic field Q&A over a FieldVisionToolbox that has classified this county-year."""
    llm = llm or NemotronClient(max_tokens=1500)    # room for reasoning plus the answer after long report passages
    df = toolbox.fields(fips, year)
    evidence = build_evidence(run_result) if run_result else {}
    tools = {
        "run_evidence": lambda a: evidence.get(a.get("section")) if a.get("section") in SECTIONS
        else {"error": f"section must be one of {list(SECTIONS)}"},
        "list_fields": lambda a: list_fields(df, a.get("kind", "largest"), a.get("limit", 5)),
        "field_detail": lambda a: toolbox.field(fips, year, a.get("field_id")) or {"error": "unknown field_id"},
    }
    specs = FIELD_QA_TOOLS
    if getattr(toolbox, "reports", None):
        tools["search_reports"] = lambda a: toolbox.reports(a.get("query", ""), fips, year, a.get("k", 3))
        specs = FIELD_QA_TOOLS + [SEARCH_REPORTS]
    messages = [{"role": "system", "content": FIELD_QA_PROMPT}, {"role": "user", "content": question}]
    allowed, used, retried, nudged, checked = set(), [], False, False, []
    _numbers_in(question, allowed)
    _numbers_in({"fips": fips, "year": year}, allowed)
    for step in range(max_steps):
        last = step == max_steps - 1
        if last:                               # out of tool steps: the final turn may only answer
            messages.append({"role": "user", "content": "Answer the question now in 2 to 5 sentences from the tool "
                                                        "results above. If they do not answer it, say what is missing."})
        reply = llm.chat(messages, None if last else specs)
        calls = reply["tool_calls"]
        messages.append({"role": "assistant", "content": reply["content"],
                         **({"tool_calls": [{"id": c["id"], "type": "function",
                                             "function": {"name": c["name"], "arguments": c["arguments"]}}
                                            for c in calls]} if calls else {})})
        if calls:
            for c in calls:
                try:
                    result = tools[c["name"]](json.loads(c["arguments"] or "{}"))
                    used.append(c["name"])
                    _numbers_in(result, allowed)
                    if c["name"] == "search_reports":
                        checked += [f"{h['source']} p.{h['page']}" for h in result
                                    if f"{h['source']} p.{h['page']}" not in checked]
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, default=str)})
            continue
        answer = (reply["content"] or "").strip()
        if not answer:
            if nudged or last:
                break
            nudged = True                      # the model went quiet after its tool calls: ask once for the answer
            messages.append({"role": "user", "content": "Answer the question now in 2 to 5 sentences from the "
                                                        "tool results above. Do not call more tools."})
            continue
        bad = ungrounded_numbers(answer, allowed)
        problems = ([f"These numbers are not in any tool result: {bad}."] if bad else []) + \
                   (["These are model estimates, not NASS statistics."] if asserts(NASS_CLAIM, answer) else [])
        if problems and not retried:
            retried = True
            messages.append({"role": "user", "content": " ".join(problems) + " Fix this and answer again."})
            continue
        return {"answer": answer, "tools_used": used, "grounded": not bad}
    if checked:
        return {"answer": "The NASS report passages I found do not answer this directly. Checked: "
                          + "; ".join(checked[:6]) + ".", "tools_used": used, "grounded": True}
    return {"answer": "I could not answer from the run record.", "tools_used": used, "grounded": True}
