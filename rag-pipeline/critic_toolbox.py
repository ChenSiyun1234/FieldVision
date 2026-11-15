"""
Real critic for the supervisor agent.

CriticToolbox wraps the team's real Toolbox (fieldvision.toolbox.Toolbox) and adds the one method
the toolbox leaves to the agent: run_critic(fips, year). It has the same signature and return keys
as StubToolbox.run_critic in agents/react/tools_stub.py, so the supervisor needs no other change:

    import sys
    sys.path[:0] = ["/workspace/FieldVision/.deps", "/workspace/FieldVision/src",
                    "/workspace/FieldVision/rag-pipeline"]
    from fieldvision.toolbox import Toolbox
    from critic_toolbox import CriticToolbox
    tools = CriticToolbox(Toolbox())            # use_llm=False for offline / test runs

How it decides (code, deterministic) and how it explains (Nemotron, optional):
  1. Code checks: the county's predicted silage share vs its QuickStats history (county rows from
     Census years, else the state's survey history), how much of the county abstained, how many
     fields the estimate rests on, and whether the field extraction is still partial.
  2. Code picks overall (pass / warn / fail) and suggested_action. Nemotron cannot change either.
  3. Nemotron writes the summary and each flag's "why", citing NASS report passages from the RAG
     index. If the call fails or returns bad JSON, a plain template summary is used instead.

suggested_action values the supervisor understands (prompts.RERUN_POLICY):
  accept | rerun_with_regional_threshold | rerun_with_radar
plus abstain_region / accept_with_warnings when no rerun is left. We never suggest radar:
the real toolbox has no radar model, so radar_first is identical to standard.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# The workspace keeps pyarrow etc. in FieldVision/.deps. It must be on the path BEFORE pandas is
# imported (rag_pipeline imports pandas), or pandas starts without pyarrow and parquet reads fail.
_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src", _REPO / ".deps"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rag_pipeline as rp  # noqa: E402

JUDGE_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"  # super-120b returns 404 on our key (probed 2026-11-14)

STATE_FIPS = {
    "01": "Alabama", "04": "Arizona", "05": "Arkansas", "06": "California", "08": "Colorado",
    "09": "Connecticut", "10": "Delaware", "12": "Florida", "13": "Georgia", "16": "Idaho",
    "17": "Illinois", "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky",
    "22": "Louisiana", "23": "Maine", "24": "Maryland", "25": "Massachusetts", "26": "Michigan",
    "27": "Minnesota", "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York",
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma", "41": "Oregon",
    "42": "Pennsylvania", "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota",
    "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont", "51": "Virginia",
    "53": "Washington", "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
}

MIN_FIELDS_FOR_COUNTY_CHECK = 30   # below this, a county share says little (BARC demo county has 7)
ABSTAIN_WARN_SHARE = 0.30          # more than 30% of acres abstained -> warn
TOLERANCE = 0.05                   # same tolerance as rag_pipeline.check_prediction


def fips_to_names(fips: str) -> tuple[str | None, str | None]:
    """'24021' -> ('Maryland', 'FREDERICK'), county name as QuickStats spells it."""
    fips = str(fips).zfill(5)
    state = STATE_FIPS.get(fips[:2])
    county = None
    try:
        if rp._QS is None:
            rp._QS = rp.load_quickstats()
        qs = rp._QS
        rows = qs[(qs["aggregation_level"] == "COUNTY")
                  & (qs["state_ansi"].astype(str).str.zfill(2) == fips[:2])
                  & (qs["county_ansi"].astype(str).str.zfill(3) == fips[2:])]
        if not rows.empty:
            county = rows["county_name"].iloc[0]
    except Exception:
        pass
    return state, county


def _history(state: str, county: str | None, year: int | None = None) -> tuple[dict, str]:
    """QuickStats history from years before the target year only (no peeking at the answer).
    County history when it has 2+ usable years (Census-only counties get one every 5 years), else the state's."""
    years = list(range(year - 15, year)) if year else None
    if county:
        h = rp.silage_share_history(state, county, years=years)
        if h["summary"].get("n_years", 0) >= 2:
            return h, "county"
    return rp.silage_share_history(state, years=years), "state"


def critic_checks(fips: str, year: int, agg: dict, policy: str | None,
                  partial: dict | None = None) -> list[dict]:
    state, county = fips_to_names(fips)
    silage, grain = float(agg.get("silage_acres") or 0), float(agg.get("grain_acres") or 0)
    abstained = float(agg.get("abstained_acres") or 0)
    n_fields = int(agg.get("n_fields") or 0)
    checks: list[dict] = []

    if state is None:
        return [{"check": "region_known", "status": "skip", "why": f"FIPS {fips} is not a known state code."}]

    # 1. predicted silage share vs QuickStats history
    if silage + grain <= 0:
        checks.append({"check": "silage_share_in_history", "status": "skip",
                       "why": "No decided acres to compare (all fields abstained)."})
    else:
        share = silage / (silage + grain)
        hist, level = _history(state, county, year)
        s = hist["summary"]
        if s.get("n_years", 0) < 2:
            checks.append({"check": "silage_share_in_history", "status": "skip", "observed": round(share, 4),
                           "why": f"Fewer than 2 usable QuickStats years before {year} for {county or state}."})
        else:
            lo, hi = s["min"], s["max"]
            status = ("pass" if lo <= share <= hi
                      else "warn" if lo - TOLERANCE <= share <= hi + TOLERANCE else "fail")
            if status == "fail" and n_fields < MIN_FIELDS_FOR_COUNTY_CHECK:
                status = "warn"  # a share from a handful of fields cannot fail a whole county
            low = agg.get("silage_acres_low")
            high = agg.get("silage_acres_high")
            range_overlaps = None
            if low is not None and high is not None and silage + grain > 0:
                s_lo, s_hi = float(low) / (float(low) + grain), float(high) / (float(high) + grain)
                range_overlaps = s_lo <= hi and s_hi >= lo
            place = f"{county.title()} County, {state}" if level == "county" else state
            why = ("" if status == "pass" else
                   f"Predicted silage share {share:.1%} is {'above' if share > hi else 'below'} "
                   f"{place}'s {lo:.1%} to {hi:.1%} range in QuickStats.")
            checks.append({"check": "silage_share_in_history", "status": status, "why": why,
                           "observed": round(share, 4), "expected_range": [lo, hi],
                           # percent copies: the supervisor only lets the agent quote numbers found in tool
                           # results, so give it the exact figures the summary uses
                           "observed_pct": round(share * 100, 1),
                           "expected_range_pct": [round(lo * 100, 1), round(hi * 100, 1)],
                           "history_mean_pct": round(s["mean"] * 100, 1),
                           "history_mean": s["mean"], "history_level": level,
                           "history_place": f"{county.title()} County, {state}" if level == "county" else state,
                           "history_years": [r["year"] for r in hist["by_year"]],
                           "estimate_range_overlaps_history": range_overlaps})

    # 2. how much of the county was not called
    total = silage + grain + abstained
    if total > 0:
        a_share = abstained / total
        checks.append({"check": "abstained_share", "status": "warn" if a_share > ABSTAIN_WARN_SHARE else "pass",
                       "observed": round(a_share, 4), "limit": ABSTAIN_WARN_SHARE,
                       "why": f"Abstained share of acres {a_share:.1%} against a {ABSTAIN_WARN_SHARE:.0%} limit."})

    # 3. is the county result resting on enough fields
    if n_fields < MIN_FIELDS_FOR_COUNTY_CHECK:
        checks.append({"check": "enough_fields", "status": "info", "observed": n_fields,
                       "why": f"Only {n_fields} fields; a county-level share from this few fields is weak evidence."})

    # 4. partial extraction (e.g. Frederick: first 248 of 2,233 fields)
    if partial:
        keep = ("fields_included", "fields_total", "acres_included", "acres_total", "note")
        detail = {k: partial[k] for k in keep if k in partial} or partial
        checks.append({"check": "coverage", "status": "info", "detail": detail,
                       "why": "Field extraction is still partial; shares are indicative, totals are not final."})
    return checks


def decide(checks: list[dict], policy: str | None) -> tuple[str, str]:
    statuses = {c["check"]: c["status"] for c in checks}
    worst = ("fail" if "fail" in statuses.values() else "warn" if "warn" in statuses.values() else "pass")
    if worst == "pass":
        return "pass", "accept"
    share = next((c for c in checks if c["check"] == "silage_share_in_history"), {})
    share_bad = share.get("status") in ("warn", "fail")
    if share_bad and policy != "regional_threshold" and _regional_can_help(share):
        # timing thresholds trained in Austria may not fit this county; shift them to its own drop dates
        return worst, "rerun_with_regional_threshold"
    return worst, ("abstain_region" if worst == "fail" else "accept_with_warnings")


def _regional_can_help(share_check: dict) -> bool:
    """regional_threshold moves the drop-date cutoff to the county's MEDIAN drop day, so it pushes the
    silage share toward about 50%. Only suggest it when moving toward 50% heads into the history range:
    share too high with history above it, or share too low with history above it. Found on Frederick 2022:
    33% (history 20-32%) went to 51% after the regional rerun."""
    obs, rng = share_check.get("observed"), share_check.get("expected_range")
    if obs is None or not rng:
        return False
    lo, hi = rng
    if obs > hi:                       # too much silage: only helps if history sits above 50%
        return lo > 0.5
    if obs < lo:                       # too little silage: helps when 50% is in the direction of history
        return obs < 0.5
    return False


def _template_summary(checks: list[dict], overall: str, action: str) -> str:
    parts = []
    for c in checks:
        if c["check"] == "silage_share_in_history" and "observed" in c and "expected_range" in c:
            parts.append(f"Predicted silage share {c['observed']:.1%} vs {c['history_place']} history "
                         f"{c['expected_range'][0]:.1%} to {c['expected_range'][1]:.1%} ({c['status']}).")
        elif c["check"] == "abstained_share":
            parts.append(f"Abstained share of acres {c['observed']:.1%} ({c['status']}).")
        elif c.get("why"):
            parts.append(c["why"])
    parts.append(f"Overall {overall}; suggested action: {action}.")
    return " ".join(parts)


EXPLAIN_PROMPT = """You are the critic in a USDA NASS pipeline that labels corn fields as silage or grain.
Code has already run the checks and decided the verdict. You only explain it.
Rules:
- Do not change "overall" or "suggested_action".
- Every number you write must appear in the CHECKS JSON. Never compute, round or invent numbers.
  For percentages use the *_pct fields exactly as given. Write ranges as "A% to B%", never "A%-B%".
- Use the REPORT PASSAGES only as context (usual harvest windows, harvest pace, weather). Cite each one
  you use by source and page. If none is relevant, use an empty evidence list.
Return ONLY a JSON object:
{"summary": "2 to 4 sentences", "flags": [{"check": "<check name>", "why": "one sentence",
  "evidence": [{"source": "...", "page": 1, "quote": "under 20 words"}]}]}
Include one flag for every check whose status is warn, fail or info."""


def _explain_with_nemotron(fips, year, state, checks, overall, action, model=JUDGE_MODEL):
    from openai import OpenAI
    client = OpenAI(base_url=rp.NV_BASE_URL, api_key=rp._nvidia_key())
    passages = []
    try:
        passages = rp.retrieve_reports(f"corn harvest progress timing silage {state} {year}", state=state, k=4)
    except Exception as exc:  # index missing etc.: explain without passages
        passages = [{"error": str(exc)}]
    user = json.dumps({"fips": fips, "year": year, "state": state, "overall": overall,
                       "suggested_action": action, "CHECKS": checks,
                       "REPORT_PASSAGES": [{k: p.get(k) for k in ("source", "page", "text")} for p in passages]},
                      default=str)[:14000]
    # Reasoning off: the verdict is already decided in code, and with reasoning on the model can spend
    # the whole token budget thinking and return empty content.
    resp = client.chat.completions.create(model=model, temperature=0, max_tokens=1500,
                                          messages=[{"role": "system", "content": EXPLAIN_PROMPT},
                                                    {"role": "user", "content": user}],
                                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"no JSON in reply (finish_reason={resp.choices[0].finish_reason}, chars={len(text)})")
    return json.loads(m.group(0))


class CriticToolbox:
    """Real Toolbox + run_critic. Every other method is passed straight through."""

    def __init__(self, toolbox, use_llm: bool = True, model: str = JUDGE_MODEL):
        self.tb, self.use_llm, self.model = toolbox, use_llm, model
        self._policy: dict = {}
        self._agg: dict = {}

    def __getattr__(self, name):  # scout, route_options, extract_features, abstain_region, ...
        return getattr(self.tb, name)

    def classify(self, fips, year, policy):
        self._policy[(str(fips), int(year))] = policy
        return self.tb.classify(fips, year, policy)

    def aggregate(self, fips, year):
        out = self.tb.aggregate(fips, year)
        self._agg[(str(fips), int(year))] = out
        return out

    def _partial(self, fips, year):
        root = getattr(self.tb, "data_root", None)
        if root is None:
            return None
        path = Path(root) / str(fips) / str(year) / "PARTIAL.json"
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        except Exception:
            return {"note": "PARTIAL.json present but unreadable"}

    def run_critic(self, fips, year):
        key = (str(fips), int(year))
        agg = self._agg.get(key) or self.aggregate(fips, year)
        policy = self._policy.get(key, "standard")
        checks = critic_checks(fips, year, agg, policy, self._partial(fips, year))
        overall, action = decide(checks, policy)
        result = {"overall": overall, "suggested_action": action, "checks": checks,
                  "flags": [{"check": c["check"], "status": c["status"], "why": c.get("why", ""), "evidence": []}
                            for c in checks if c["status"] in ("warn", "fail", "info")],
                  "summary": _template_summary(checks, overall, action), "explained_by": "template"}
        if self.use_llm:
            try:
                state, _ = fips_to_names(fips)
                exp = _explain_with_nemotron(fips, year, state, checks, overall, action, self.model)
                if not (exp and exp.get("summary")):
                    raise ValueError("reply had no summary")
                if exp and exp.get("summary"):
                    by_check = {f.get("check"): f for f in exp.get("flags", [])}
                    for f in result["flags"]:
                        e = by_check.get(f["check"], {})
                        f["why"] = e.get("why") or f["why"]
                        f["evidence"] = e.get("evidence", [])
                    result["summary"] = exp["summary"]
                    result["explained_by"] = self.model
            except Exception as exc:  # never let the explainer break the supervisor loop
                result["explain_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run the real critic on one county-year (needs the real Toolbox).")
    ap.add_argument("fips")
    ap.add_argument("year", type=int)
    ap.add_argument("--policy", default="standard")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--repo", default=str(_REPO))
    args = ap.parse_args()
    if Path(args.repo).resolve() != _REPO:
        sys.path[:0] = [str(Path(args.repo) / ".deps"), str(Path(args.repo) / "src")]
    from fieldvision.toolbox import Toolbox
    tools = CriticToolbox(Toolbox(), use_llm=not args.no_llm)
    tools.classify(args.fips, args.year, args.policy)
    print(json.dumps(tools.aggregate(args.fips, args.year), indent=2))
    print(json.dumps(tools.run_critic(args.fips, args.year), indent=2, default=str))
