"""The chat assistant: one conversation, two roles, every pipeline feature as a tool.

    dairy owner   where is standing corn I could buy for silage, at what price, and when is it cut
    USDA analyst  run a county through the agent, read the evidence and report, plan field visits,
                  compare our silage estimates with USDA, corn supply and price, a silage what-if

Nemotron picks the tools and writes the answer; code computes every number, and an answer whose numbers
are not in a tool result, the conversation or the context is sent back once, then marked [unverified].
Offline (no key), a keyword router calls the same tools and lays out their results.

Each tool returns (result for the model, card for the page). Cards are what the page draws: a map of
matched fields, a run summary, a visit plan, report passages, supply tiles, a what-if table.
"""

import json
import re

from ..fieldvision_bridge.field_qa import list_fields
from ..fieldvision_bridge.rag import STATE_FIPS, load_rag
from ..planner.planner import county_status, plan_visits, supply_impact
from ..react.supervisor import DEFAULT_MODEL, NemotronClient, _numbers_in, ungrounded_numbers
from ..reporter.reporter import NASS_CLAIM, SECTIONS, _mark_unverified, _strip_thinking, asserts, build_evidence
from . import dairy, dashboard, service


def _tool(name, desc, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


_SEASON = {"fips": {"type": "string", "description": "5-digit county FIPS; default: the season picked on the page"},
           "year": {"type": "integer", "description": "season year; default: the season picked on the page"}}


def _season(args, ctx):
    fips = str(args.get("fips") or ctx.get("fips") or "")
    year = int(args.get("year") or ctx.get("year") or 0)
    if not fips or not year:
        raise ValueError("Pick a county and season first.")
    return fips, year


# ---------------------------------------------------------------- shared tools

def t_search_reports(args, ctx):
    rag, _ = load_rag()
    fips = str(args.get("fips") or ctx.get("fips") or "24")
    state = STATE_FIPS.get(fips.zfill(5)[:2]) if len(fips) >= 2 else None
    hits = rag.retrieve_reports(args.get("query") or "corn silage harvest dates", state=state, k=10)
    # The handbook covers every crop; passages about corn go first, so a question about silage timing is not
    # answered from the tobacco page.
    hits = sorted(hits, key=lambda h: "corn" not in (h.get("text") or "").lower())[:3]
    rows = [{"source": h.get("source"), "page": h.get("page"), "state": h.get("state"), "year": h.get("year"),
             "text": (h.get("text") or "")[:500]} for h in hits]
    return {"passages": rows, "note": "NASS reports and handbooks: general context, never this season's acres."}, \
        {"type": "passages", "data": rows}


def t_list_seasons(args, ctx):
    rows = service.counties()
    return {"seasons": rows}, None


# ---------------------------------------------------------------- dairy tools

def t_list_dairies(args, ctx):
    rows = dairy.dairies()
    return {"dairies": rows, "note": "Permitted dairies (Maryland Department of the Environment); ids only, no names."}, None


def t_find_silage(args, ctx):
    fips, year = _season(args, ctx)
    prof = ctx.get("dairy") or {}
    kw = {k: args.get(k, prof.get(k)) for k in ("dairy_id", "lat", "lon", "head", "tons", "max_miles")}
    r = dairy.find_silage(fips, year, as_of=args.get("as_of") or ctx.get("as_of"), **kw)
    slim = {k: r[k] for k in ("name", "as_of", "funnel", "standing_price_usd_per_ton", "overall", "hindsight", "in_reach")}
    slim["dairy"] = {k: r["dairy"].get(k) for k in ("dairy_id", "head", "request_tons", "matched_tons", "n_fields",
                                                    "delivered_usd_per_ton", "avg_haul_usd_per_ton")}
    slim["fields"] = r["fields"][:8]
    slim["checks"] = [{k: c.get(k) for k in ("check", "status", "why")} for c in r["checks"]]
    slim["note"] = r["note"]
    return slim, {"type": "silage", "data": r}


# ---------------------------------------------------------------- USDA tools

def t_run_county(args, ctx):
    fips, year = _season(args, ctx)
    result = service.run(fips, year, offline=ctx.get("offline", False))
    d = service.detail(result["run_id"])
    ctx["run_id"] = result["run_id"]
    ev = d["evidence"]
    return {"run_id": result["run_id"], "status": result["status"], "summary": result.get("summary"),
            "estimate": ev.get("estimate"), "critic": ev.get("critic"), "policies": result.get("policies")}, \
        {"type": "run", "data": d}


def t_run_evidence(args, ctx):
    run_id = args.get("run_id") or ctx.get("run_id")
    if not run_id:
        raise ValueError("No run yet: run a county first.")
    section = args.get("section") or "estimate"
    if section not in SECTIONS:
        raise ValueError(f"section must be one of {list(SECTIONS)}")
    return {"run_id": run_id, section: build_evidence(service.load(run_id)).get(section)}, None


def t_county_report(args, ctx):
    run_id = args.get("run_id") or ctx.get("run_id")
    if not run_id:
        raise ValueError("No run yet: run a county first.")
    rep = service.report(run_id, offline=ctx.get("offline", False))
    return {"report_md": rep["report_md"], "grounded": rep["grounding"]["grounded"]}, \
        {"type": "report", "data": {"run_id": run_id, "report_md": rep["report_md"], "path": rep["path"],
                                    "grounded": rep["grounding"]["grounded"]}}


def _classified(fips, year):
    tb = service.toolbox(offline=True)
    try:
        return tb._tb.fields(fips, year), tb
    except RuntimeError:
        tb.scout(fips, year)
        tb.extract_features(fips, year)
        tb.classify(fips, year, "standard")
        return tb._tb.fields(fips, year), tb


def t_list_fields(args, ctx):
    fips, year = _season(args, ctx)
    df, _ = _classified(fips, year)
    out = list_fields(df, args.get("kind") or "most_uncertain", args.get("limit") or 5)
    return out, {"type": "fields", "data": {**out, "fips": fips, "year": year}}


def t_field_detail(args, ctx):
    fips, year = _season(args, ctx)
    _, tb = _classified(fips, year)
    f = tb._tb.field(fips, year, args.get("field_id"))
    if f is None:
        raise ValueError(f"No field {args.get('field_id')} in {fips} {year}")
    return f, None


def t_plan_visits(args, ctx):
    fips, year = _season(args, ctx)
    _, tb = _classified(fips, year)
    frame = tb._tb.planner_frame(fips, year)
    prior = frame.attrs.get("abstain_prior")
    plan = plan_visits(frame, args.get("visits") or 10, abstain_prior=prior)
    status = county_status(frame, abstain_prior=prior)
    out = {"county_status": status, "plan": plan, "supply": supply_impact(status)}
    return out, {"type": "visits", "data": {**out, "fips": fips, "year": year, "name": service.COUNTY_NAMES.get(fips, fips)}}


def t_supply_outlook(args, ctx):
    s = dashboard.supply()
    return {"tiles": s["tiles"]}, {"type": "supply", "data": s}


def t_silage_what_if(args, ctx):
    w = dashboard.what_if(args.get("points") or 1.0)
    return w, {"type": "whatif", "data": w}


def t_silage_vs_usda(args, ctx):
    s = dashboard.silage()
    return s, {"type": "silage_compare", "data": s}


def t_price_model_check(args, ctx):
    rows = dashboard.price_models()
    return {"leave_one_year_out_2001_2024": rows,
            "note": "Explain models use this season's ending stocks (known only after the season); forecast models use last "
                    "season's. Neither is a forecast of next season's price."}, {"type": "price_models", "data": rows}


def t_county_dairy_matching(args, ctx):
    fips, year = _season(args, ctx)
    sm = dairy._silage_match()
    policy = dairy._ensure_predictions(fips, year)
    r = sm.recommend(fips, year, args.get("as_of") or ctx.get("as_of"), sm.Params(), policy)
    slim = {k: r[k] for k in ("as_of", "funnel", "standing_price_usd_per_ton", "overall", "summary")}
    slim["dairies"] = [{k: x[k] for k in ("dairy_id", "request_tons", "matched_tons", "n_fields", "delivered_usd_per_ton")}
                       for x in r["dairies"]]
    return slim, None


def t_agent_quality(args, ctx):
    q = dashboard.agent_quality()
    return q, {"type": "quality", "data": q}


TOOLS = {
    "dairy": {
        "find_silage": (t_find_silage, _tool(
            "find_silage", "Standing corn this dairy could buy for silage as of a date: matched fields, tons, road miles, "
            "delivered price range and the matcher's checks. Uses the dairy profile on the page unless arguments say otherwise.",
            {**_SEASON, "as_of": {"type": "string", "description": "YYYY-MM-DD within the season"},
             "dairy_id": {"type": "string"}, "lat": {"type": "number"}, "lon": {"type": "number"},
             "head": {"type": "number", "description": "herd size"},
             "tons": {"type": "number", "description": "tons of silage wanted"},
             "max_miles": {"type": "number", "description": "longest haul in road miles"}})),
        "list_dairies": (t_list_dairies, _tool("list_dairies", "Permitted dairies with id, county, herd size and location.")),
        "list_seasons": (t_list_seasons, _tool("list_seasons", "County-seasons with field data on this machine.")),
        "search_reports": (t_search_reports, _tool(
            "search_reports", "Passages from NASS crop progress reports and the usual harvest dates handbook, with source and page.",
            {"query": {"type": "string"}, "fips": {"type": "string"}}, ["query"])),
    },
    "usda": {
        "run_county": (t_run_county, _tool(
            "run_county", "Run one county-season through the full agent pipeline (scout, route, classify, aggregate, "
            "critic, at most one rerun). Takes up to a minute. Returns the run id, status, estimate and critic verdicts.", _SEASON)),
        "run_evidence": (t_run_evidence, _tool(
            "run_evidence", "One section of a finished run's evidence.",
            {"run_id": {"type": "string"}, "section": {"type": "string", "enum": list(SECTIONS)}}, ["section"])),
        "county_report": (t_county_report, _tool("county_report", "Write the county report for a finished run.",
                                                 {"run_id": {"type": "string"}})),
        "list_fields": (t_list_fields, _tool(
            "list_fields", "Fields of the season by kind, with the model's probability and evidence.",
            {**_SEASON, "kind": {"type": "string", "enum": ["silage_like", "grain_like", "abstained", "most_uncertain", "largest"]},
             "limit": {"type": "integer"}}, ["kind"])),
        "field_detail": (t_field_detail, _tool("field_detail", "One field's satellite evidence and call.",
                                               {**_SEASON, "field_id": {"type": "string"}}, ["field_id"])),
        "plan_visits": (t_plan_visits, _tool(
            "plan_visits", "The field visits that shrink the county's silage range the most, the range before and after, "
            "and what the estimate means for grain supply in bushels.",
            {**_SEASON, "visits": {"type": "integer", "description": "number of field visits"}})),
        "silage_vs_usda": (t_silage_vs_usda, _tool(
            "silage_vs_usda", "Our silage estimates next to USDA's: four states 2025, Maryland 2025, Frederick 2022, and the "
            "labelled field test.")),
        "supply_outlook": (t_supply_outlook, _tool(
            "supply_outlook", "USDA corn supply and price: production, yield, area, stocks, stocks-to-use, price received, "
            "silage acres, with change against the year before.")),
        "silage_what_if": (t_silage_what_if, _tool(
            "silage_what_if", "If national silage share were this many points higher (or lower), how much grain and what price "
            "effect under each explaining model. A what-if, not a forecast.",
            {"points": {"type": "number", "description": "percentage points of silage share, e.g. 1 or -0.5"}}, ["points"])),
        "price_model_check": (t_price_model_check, _tool(
            "price_model_check", "How well the price models did, held out one year at a time, against last year's price.")),
        "county_dairy_matching": (t_county_dairy_matching, _tool(
            "county_dairy_matching", "All permitted dairies near the county matched to standing grain-bound corn as of a date.",
            {**_SEASON, "as_of": {"type": "string"}})),
        "agent_quality": (t_agent_quality, _tool(
            "agent_quality", "How the agent did on its own tests: scenario checks, Q&A checks, the Nemotron model benchmark "
            "and the limitations found.")),
        "search_reports": (t_search_reports, _tool(
            "search_reports", "Passages from NASS crop progress reports and handbooks, with source and page.",
            {"query": {"type": "string"}, "fips": {"type": "string"}}, ["query"])),
    },
}

PROMPTS = {
    "dairy": """You help a dairy farmer find corn silage. Tools find standing corn fields they could buy and chop, with tons,
haul distance and a delivered price range, and NASS report passages about usual harvest timing.
- Call find_silage for any question about where to get silage, how much, or what it costs. Use the dairy profile and season
  on the page unless the farmer gives other numbers.
- Every number must come from a tool result, the farmer's message or the page context. Prices are a rule of thumb, not quotes.
- The fields are a replay of a past season as of a date; say so once. These are model estimates, not NASS statistics.
- Answer in 2 to 5 short sentences a farmer can act on: how many tons, from how many fields, how far, what price range,
  and the one check to look at first. Cite report passages by source and page.""",
    "usda": """You assist a USDA NASS analyst. Tools run the silage/grain agent pipeline on a county-season, read its evidence and
report, list fields, plan field visits, compare our silage estimates with USDA's, give USDA's corn supply and price figures,
run a silage what-if on price, check the price models, report the agent's own test results, and search NASS reports.
- Pick the tools the question needs. Reuse the run on the page (context run_id) instead of running a county again,
  unless the analyst asks for a new run.
- Every number must come from a tool result, the analyst's message or the page context.
- Our acres are model estimates, never NASS statistics. USDA's own figures are USDA's: say which is which.
- Price: the models explain a finished season's price; they do not forecast it. A silage effect on price is a what-if,
  and the answer depends on the model: give the range.
- Answer in 2 to 6 sentences, plainly, then name the tools you used.""",
}

SUGGESTIONS = {
    "dairy": ["Where can I get 2,000 tons of silage within 15 miles?", "What would it cost me delivered?",
              "When is corn silage usually harvested in Maryland?"],
    "usda": ["Run the selected county and tell me how sure we are", "Where should enumerators go with 12 visits?",
             "How do our silage estimates compare with USDA's?", "What's the corn supply and price picture?",
             "If silage were 1 point higher nationally, what happens to price?", "How well has the agent tested?"],
}


def _call(role, name, args, ctx):
    fn = TOOLS[role][name][0]
    try:
        return fn(args, ctx)
    except Exception as e:  # a failing tool becomes a message the model and the page can show
        return {"error": f"{type(e).__name__}: {e}"}, {"type": "error", "data": {"tool": name, "error": str(e)}}


def _context_note(ctx):
    keep = {k: ctx.get(k) for k in ("fips", "year", "run_id", "as_of") if ctx.get(k)}
    if ctx.get("fips"):
        keep["county"] = service.COUNTY_NAMES.get(str(ctx["fips"]), ctx["fips"])
    if ctx.get("dairy"):
        keep["dairy_profile"] = {k: v for k, v in ctx["dairy"].items() if v not in (None, "")}
    return "Page context: " + json.dumps(keep)


def chat(role, messages, ctx=None, llm=None, max_steps=6):
    """messages: [{"role": "user"|"assistant", "content": str}, ...], last one the user's.
    Returns {"reply", "cards", "tools", "grounded", "mode", "context"}."""
    ctx = dict(ctx or {})
    if role not in TOOLS:
        raise ValueError("role must be 'dairy' or 'usda'")
    offline = ctx.get("offline") or (llm is None and not service.has_key())
    ctx["offline"] = offline
    if offline:
        return offline_chat(role, messages, ctx)

    llm = llm or NemotronClient(model=DEFAULT_MODEL, max_tokens=1500)
    specs = [spec for _, spec in TOOLS[role].values()]
    convo = [{"role": "system", "content": PROMPTS[role] + "\n" + _context_note(ctx)}] + \
            [{"role": m["role"], "content": m["content"]} for m in messages[-10:]]
    allowed, cards, used, retried = set(), [], [], False
    _numbers_in({k: v for k, v in ctx.items() if k != "offline"}, allowed)
    _numbers_in([m["content"] for m in messages if m["role"] == "user"], allowed)

    for _ in range(max_steps):
        reply = llm.chat(convo, specs)
        calls = reply["tool_calls"]
        convo.append({"role": "assistant", "content": reply["content"],
                      **({"tool_calls": [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"], "arguments": c["arguments"]}}
                                         for c in calls]} if calls else {})})
        if calls:
            for c in calls:
                try:
                    args = json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                if c["name"] in TOOLS[role]:
                    result, card = _call(role, c["name"], args, ctx)
                    used.append(c["name"])
                    if card:
                        cards.append(card)
                    if "error" not in result:
                        _numbers_in(result, allowed)
                else:
                    result = {"error": f"Unknown tool {c['name']}"}
                convo.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, default=str)[:12000]})
            continue

        answer = _strip_thinking(reply["content"])
        bad = ungrounded_numbers(answer, allowed)
        problems = []
        if bad:
            problems.append(f"These numbers are not in any tool result or the conversation: {bad}. Fetch them or remove them.")
        if asserts(NASS_CLAIM, answer):
            problems.append("Our acres are model estimates, not NASS statistics. Do not attribute them to NASS.")
        if problems and not retried:
            retried = True
            convo.append({"role": "user", "content": " ".join(problems) + " Answer again."})
            continue
        return {"reply": _mark_unverified(answer, bad) if bad else answer, "cards": cards, "tools": used,
                "grounded": not bad, "mode": "nemotron", "context": _public(ctx)}

    return {"reply": "I could not finish that in the steps I have. Try a narrower question.", "cards": cards,
            "tools": used, "grounded": True, "mode": "nemotron", "context": _public(ctx)}


def _public(ctx):
    return {k: v for k, v in ctx.items() if k in ("fips", "year", "run_id", "as_of")}


# ---------------------------------------------------------------- offline: keyword router

_INTENTS = {
    "dairy": [
        (("when", "harvest date", "usually", "timing", "calendar"), "search_reports"),
        (("dairy list", "which dairies", "permitted"), "list_dairies"),
        (("silage", "tons", "near", "buy", "cost", "price", "where", "find", "field", "haul"), "find_silage"),
    ],
    "usda": [
        (("what if", "what-if", "points higher", "point higher", "points lower", "if silage"), "silage_what_if"),
        (("visit", "enumerator", "where should"), "plan_visits"),
        (("report",), "county_report"),
        (("compare", "vs usda", "against usda", "with usda's", "states", "census"), "silage_vs_usda"),
        (("tested", "evaluation", "benchmark", "limitation", "agent do", "trust the agent"), "agent_quality"),
        (("price model", "baseline", "leave-one", "accuracy of the price"), "price_model_check"),
        (("supply", "production", "stocks", "price", "yield", "outlook"), "supply_outlook"),
        (("dairies", "recommend", "standing corn"), "county_dairy_matching"),
        (("uncertain", "abstain", "silage-like", "grain-like", "which fields", "largest fields"), "list_fields"),
        (("run", "how sure", "estimate", "acres", "county"), "run_county"),
        (("harvest", "crop progress", "nass report", "handbook"), "search_reports"),
    ],
}


def _fmt(x, digits=1):
    """Acres and tons with one decimal at most; pass digits=2 for dollars."""
    if x is None:
        return "n/a"
    if isinstance(x, float) and not x.is_integer():
        return f"{x:,.{digits}f}"
    return f"{int(x):,}"


def _offline_text(role, name, r):
    if "error" in r:
        return f"That did not work: {r['error']}"
    if name == "find_silage":
        d = r["dairy"]
        if not r["in_reach"]:
            return f"As of {r['as_of']} no field of {r['name']} is within reach of this dairy."
        checks = "; ".join(f"{c['check']} {c['status']}" for c in r["checks"] if c["status"] != "pass") or "every check passes"
        price = d.get("delivered_usd_per_ton") or [None, None]
        return (f"As of {r['as_of']} in {r['name']}: {_fmt(d['matched_tons'])} of the {_fmt(d['request_tons'])} tons you need "
                f"can come from {d['n_fields']} standing fields, delivered for about {_fmt(price[0], 2)} to {_fmt(price[1], 2)} dollars "
                f"a ton. Checks: {checks}. This replays a past season; prices are a rule of thumb, not quotes.")
    if name == "search_reports":
        return "From NASS reports: " + " ".join(f"({p['source']}, p.{p['page']}) {p['text'][:220]}…" for p in r["passages"][:2])
    if name == "list_dairies":
        return f"{len(r['dairies'])} permitted dairies are in the file: " + ", ".join(
            f"{x['dairy_id']} ({x.get('county')}, {_fmt(x.get('head'))} head)" for x in r["dairies"][:8]) + "."
    if name == "run_county":
        est = r.get("estimate") or {}
        critic = " → ".join(c.get("overall", "?") for c in r.get("critic") or []) or "not run"
        return (f"Run {r['run_id']} finished {r['status'].replace('_', ' ')}: silage {_fmt(est.get('silage_acres'))} acres "
                f"(range {_fmt(est.get('silage_acres_low'))} to {_fmt(est.get('silage_acres_high'))}), grain "
                f"{_fmt(est.get('grain_acres'))}, abstained {_fmt(est.get('abstained_acres'))}. Critic: {critic}. "
                "Model estimates, not NASS statistics.")
    if name == "county_report":
        return "The county report is below, every number taken from the run record."
    if name == "plan_visits":
        p, s = r["plan"], r["county_status"]
        return (f"With {p['visits']} visits covering {_fmt(p['acres_visited'])} acres, the silage range narrows from "
                f"±{_fmt(p['plus_minus_before'])} to ±{_fmt(p['plus_minus_after'])} acres ({p['range_reduction_pct']}% smaller) "
                f"around {_fmt(s['silage_acres'])} acres. That is about {_fmt(r['supply']['bushels_not_in_grain_supply'])} "
                "bushels not in grain supply. The fields to visit are listed below.")
    if name == "silage_vs_usda":
        st = "; ".join(f"{x['state']} ours {x['ours_pct']}% vs USDA {x['usda_pct']}%" for x in r["states_2025"])
        md = r.get("maryland_2025") or {}
        return (f"2025, ours against USDA: {st}. Maryland: {md.get('ours_pct')}% vs {md.get('usda_pct')}%. Field level, "
                f"{r['field_test']['right']} of {r['field_test']['of']} labelled Beltsville fields were right. County and state "
                "totals are not validated yet.")
    if name == "supply_outlook":
        t = {x["id"]: x for x in r["tiles"]}
        return (f"USDA: 2026 production forecast {_fmt(t['production']['value'] / 1e9)} billion bushels "
                f"({t['production']['change_pct']}% vs 2025); September 1 stocks {_fmt(t['stocks']['value'] / 1e9)} billion "
                f"({t['stocks']['change_pct']}%); 2025/26 price received ${t['price']['value']} a bushel; September 2026 "
                f"${t['price_sep']['value']}. All USDA figures.")
    if name == "silage_what_if":
        lo, hi = r["range_pct"]
        unit = "point" if abs(r["points"]) == 1 else "points"
        return (f"{r['points']:+g} {unit} of national silage share is {_fmt(r['acres'])} acres and {_fmt(-r['grain_bushels'] / 1e6)} "
                f"million fewer bushels of grain, which these models turn into a {lo}% to {hi}% price change. "
                "A what-if, not a forecast.")
    if name == "price_model_check":
        rows = {x["model"]: x for x in r["leave_one_year_out_2001_2024"]}
        base = rows["no change (last year's price)"]
        best = min(r["leave_one_year_out_2001_2024"], key=lambda x: x["mape_pct"])
        return (f"Held out one year at a time, last year's price is off by {base['mape_pct']}% on average and the best model "
                f"({best['model']}) by {best['mape_pct']}%. For 2025 last year's price was {base['error_2025_pct']}% off. "
                "The explaining models use ending stocks, so they explain a season; they do not forecast it.")
    if name == "agent_quality":
        return "The agent's latest evaluation and benchmark are below, as recorded."
    if name == "county_dairy_matching":
        return r["summary"]
    if name == "list_fields":
        return f"{r['count']} fields of kind {r['kind']}; the first {len(r['fields'])} are listed below."
    return json.dumps(r, default=str)[:600]


def offline_chat(role, messages, ctx):
    q = messages[-1]["content"].lower() if messages else ""
    name = next((tool for words, tool in _INTENTS[role] if any(w in q for w in words)), None)
    if name is None:
        return {"reply": "Offline mode answers by matching your question to a tool. Try: " + " · ".join(SUGGESTIONS[role]),
                "cards": [], "tools": [], "grounded": True, "mode": "offline", "context": _public(ctx)}
    args = {}
    if name == "silage_what_if":
        m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:percentage\s+)?points?", q)
        args["points"] = float(m.group(1)) if m else 1.0
        if any(w in q for w in ("lower", "less", "fewer")) and args["points"] > 0:
            args["points"] = -args["points"]
    if name == "plan_visits":
        m = re.search(r"(\d+)\s*visits?", q)
        args["visits"] = int(m.group(1)) if m else 10
    if name == "find_silage":
        m = re.search(r"([\d,]+)\s*tons?", q)
        if m:
            args["tons"] = float(m.group(1).replace(",", ""))
        m = re.search(r"(\d+)\s*(?:road\s*)?miles?", q)
        if m:
            args["max_miles"] = float(m.group(1))
    if name == "search_reports":
        args["query"] = messages[-1]["content"]
    if name == "list_fields":
        args["kind"] = next((k for k in ("abstained", "silage_like", "grain_like", "largest") if k.replace("_", " ")[:6] in q),
                            "most_uncertain")
    if name == "county_report" and not ctx.get("run_id"):
        r1, c1 = _call(role, "run_county", {}, ctx)
        r2, c2 = _call(role, "county_report", {}, ctx)
        return {"reply": _offline_text(role, "run_county", r1) + " " + _offline_text(role, "county_report", r2),
                "cards": [c for c in (c1, c2) if c], "tools": ["run_county", "county_report"], "grounded": True,
                "mode": "offline", "context": _public(ctx)}
    result, card = _call(role, name, args, ctx)
    return {"reply": _offline_text(role, name, result), "cards": [card] if card else [], "tools": [name],
            "grounded": True, "mode": "offline", "context": _public(ctx)}
