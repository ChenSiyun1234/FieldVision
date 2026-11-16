"""Field verification planner and supply impact.

Planner: which fields should an enumerator visit to shrink the county's silage range the most?
Each field's outcome is uncertain: the frozen model gives P(silage) = p. With fields independent,
the county's variance is the sum of area^2 * p * (1 - p), and visiting a field removes its own term
exactly. So the best visits are the fields with the largest area^2 * p * (1 - p), and we can say
precisely how much each visit narrows the 90% range.

Supply impact: production = harvested acres x yield. Every silage acre is an acre that does not
produce grain, so silage acres x grain yield = bushels kept out of grain supply.

In a notebook:
    from fieldvision_agents.planner.planner import load_fields, plan_visits, supply_impact, ask_planner
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..react.supervisor import WORKSPACE, NemotronClient, _numbers_in, read_env, ungrounded_numbers

Z90 = 1.645
# USDA NASS Crop Production 2025 Summary (Jan 2026), from the challenge kickoff deck.
NATIONAL_GRAIN_YIELD = (186.5, "USDA NASS Crop Production 2025 Summary, U.S. grain yield (fallback)")
NATIONAL_SILAGE_YIELD = (21.8, "USDA NASS Crop Production 2025 Summary, U.S. silage yield")


# --- field-level inputs ---------------------------------------------------

def load_fields(path):
    """Per-field results written by the real classify tool (parquet or csv)."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = {"field_id", "area_acres", "p_silage", "abstain"} - set(df)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return df


def demo_fields(n=600, seed=7):
    """Synthetic county for demos before real results exist. Clearly placeholder data."""
    rng = np.random.default_rng(seed)
    kind = rng.choice(["grain", "silage", "unsure"], size=n, p=[0.72, 0.18, 0.10])
    p = np.where(kind == "grain", rng.beta(1, 14, n), np.where(kind == "silage", rng.beta(12, 2, n), rng.beta(2, 2, n)))
    df = pd.DataFrame({"field_id": [f"DEMO-{i:04d}" for i in range(n)],
                       "area_acres": np.round(rng.lognormal(3.1, 0.6, n), 1),
                       "p_silage": np.round(p, 3)})
    df["abstain"] = rng.random(n) < 0.05
    df.loc[df["abstain"], "p_silage"] = np.nan
    df.attrs["demo"] = True
    return df


def _prepared(df, abstain_prior=None):
    """Give abstained fields a prior (the county's area-weighted silage share) and compute variance terms."""
    d = df.copy()
    d["abstain"] = d["abstain"].astype(bool) | d["p_silage"].isna()
    decided = d[~d["abstain"]]
    if abstain_prior is None:
        abstain_prior = df.attrs.get("abstain_prior")      # set by a caller that knows the county prior
    if abstain_prior is None:
        area = decided["area_acres"].sum()
        abstain_prior = float((decided["area_acres"] * decided["p_silage"]).sum() / area) if area else 0.5
    d["p_used"] = d["p_silage"].where(~d["abstain"], abstain_prior)
    d["var_term"] = d["area_acres"] ** 2 * d["p_used"] * (1 - d["p_used"])
    return d, abstain_prior


# --- planner --------------------------------------------------------------

def county_status(df, abstain_prior=None):
    d, prior = _prepared(df, abstain_prior)
    est, sd = float((d["area_acres"] * d["p_used"]).sum()), math.sqrt(d["var_term"].sum())
    return {"silage_acres": round(est), "range_low": round(max(0.0, est - Z90 * sd)),
            "range_high": round(est + Z90 * sd), "plus_minus": round(Z90 * sd),
            "corn_acres": round(float(d["area_acres"].sum())), "n_fields": int(len(d)),
            "n_unsure_fields": int(((d["p_used"] > 0.2) & (d["p_used"] < 0.8)).sum()),
            "n_abstained_fields": int(d["abstain"].sum()), "prior_for_abstained": round(prior, 3),
            "demo_data": bool(df.attrs.get("demo"))}


def _why(row):
    if row["abstain"]:
        return f"no satellite read (abstained); {row['area_acres']:.0f} acres at stake"
    p, acres = row["p_used"], row["area_acres"]
    if 0.2 < p < 0.8:                                   # same "unsure" band as county_status
        return f"model unsure (P(silage) {p:.2f}) on a {acres:.0f}-acre field"
    lean = "silage" if p >= 0.5 else "grain"
    return (f"large field: the call leans {lean} (P(silage) {p:.2f}), but at {acres:.0f} acres "
            f"even that doubt is one of the biggest in the county")


def plan_visits(df, k=10, abstain_prior=None):
    """The k visits that shrink the county's 90% range the most, with the range before and after."""
    d, _ = _prepared(df, abstain_prior)
    k = max(0, min(int(k), len(d)))
    picks = d.sort_values("var_term", ascending=False).head(k)
    before = math.sqrt(d["var_term"].sum())
    after = math.sqrt(max(0.0, d["var_term"].sum() - picks["var_term"].sum()))
    return {"visits": k, "plus_minus_before": round(Z90 * before), "plus_minus_after": round(Z90 * after),
            "range_reduction_pct": round(100 * (1 - after / before)) if before else 0,
            "acres_visited": round(float(picks["area_acres"].sum())),
            "fields": [{"rank": i + 1, "field_id": r["field_id"], "area_acres": round(float(r["area_acres"]), 1),
                        "p_silage": None if r["abstain"] else round(float(r["p_used"]), 2), "why": _why(r)}
                       for i, (_, r) in enumerate(picks.iterrows())]}


def uncertainty_curve(df, max_visits=60, step=5, abstain_prior=None):
    """How the 90% range shrinks as more fields are visited, in the best order."""
    d, _ = _prepared(df, abstain_prior)
    terms = d["var_term"].sort_values(ascending=False).to_numpy()
    total, cum = terms.sum(), np.concatenate([[0.0], np.cumsum(terms)])
    points = []
    for k in range(0, min(max_visits, len(terms)) + 1, step):
        points.append({"visits": k, "plus_minus": round(Z90 * math.sqrt(max(0.0, total - cum[k])))})
    return {"points": points}


def apply_reports(df, reports):
    """Field reports from enumerators, {field_id: "silage" | "grain"}, become certain outcomes."""
    d = df.copy()
    for field_id, label in reports.items():
        mask = d["field_id"] == field_id
        if not mask.any():
            raise ValueError(f"Unknown field {field_id}")
        if label not in ("silage", "grain"):
            raise ValueError(f"Report for {field_id} must be 'silage' or 'grain', got {label!r}")
        d.loc[mask, "p_silage"] = 1.0 if label == "silage" else 0.0
        d.loc[mask, "abstain"] = False
    d.attrs = dict(df.attrs)
    return d


# --- supply impact ----------------------------------------------------------

def fetch_state_grain_yield(state_alpha, year):
    """State corn-for-grain yield (bu/acre) from the Quick Stats API, or (None, reason)."""
    import requests
    key = os.environ.get("QUICKSTATS_KEY") or read_env(WORKSPACE / ".env").get("QUICKSTATS_KEY", "")
    if not key:
        return None, "no QUICKSTATS_KEY in .env"
    params = dict(key=key, source_desc="SURVEY", agg_level_desc="STATE", state_alpha=state_alpha, year=str(year),
                  short_desc="CORN, GRAIN - YIELD, MEASURED IN BU / ACRE", reference_period_desc="YEAR", format="JSON")
    try:
        r = requests.get("https://quickstats.nass.usda.gov/api/api_GET/", params=params, timeout=30)
        rows = r.json().get("data", []) if r.ok else []
        value = float(rows[0]["Value"].replace(",", "")) if rows else None
    except Exception as e:  # network or format problems fall back to the national figure
        return None, f"Quick Stats request failed: {e}"
    return (value, f"USDA NASS Quick Stats, {state_alpha} {year} corn grain yield") if value else (None, "no value returned")


def supply_impact(status, grain_yield=None, silage_yield=NATIONAL_SILAGE_YIELD):
    """Turn the silage estimate and its range into grain bushels not produced and silage tons."""
    gy, gsrc = grain_yield or NATIONAL_GRAIN_YIELD
    sy, ssrc = silage_yield
    return {"grain_yield_bu_per_acre": gy, "grain_yield_source": gsrc,
            "bushels_not_in_grain_supply": round(status["silage_acres"] * gy),
            "bushels_low": round(status["range_low"] * gy), "bushels_high": round(status["range_high"] * gy),
            "silage_yield_tons_per_acre": sy, "silage_yield_source": ssrc,
            "silage_tons": round(status["silage_acres"] * sy),
            "silage_share_of_corn_pct": round(100 * status["silage_acres"] / status["corn_acres"], 1)
            if status.get("corn_acres") else None,
            "note": "Model estimate, not NASS statistics."}


# --- the planner agent ------------------------------------------------------

PLANNER_PROMPT = """You help a USDA NASS supervisor plan field visits for one county-season, using tools.
Tools give the county's current silage estimate and uncertainty, the best fields to visit for a number of visits,
how the uncertainty shrinks as visits increase, and what the estimate means for grain supply.
- Work out the number of visits from what the user says (for example 2 enumerators x 8 visits = 16).
  If no budget is given, call uncertainty_curve and recommend a number where extra visits stop helping much.
- Every number must come from a tool result or the user's question. These are model estimates, not NASS statistics.
- Answer in 3 to 6 sentences, then list the fields to visit as a short bulleted list (field id, acres, why).
"""


def _tool(name, desc, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


PLANNER_TOOLS = [
    _tool("county_status", "Current silage estimate, 90% range, number of unsure and abstained fields."),
    _tool("plan_visits", "Best fields to visit for a given number of visits, and the 90% range before and after.",
          {"visits": {"type": "integer", "description": "Number of field visits available"}}, ["visits"]),
    _tool("uncertainty_curve", "How the 90% range shrinks as visits increase (every 5 visits).",
          {"max_visits": {"type": "integer"}}),
    _tool("supply_impact", "Grain bushels not produced because of silage, and silage tons, with the yield sources."),
]


def ask_planner(df, question, llm=None, grain_yield=None, max_steps=6):
    """Agentic planning: Nemotron picks tools, then answers; every number is checked against tool results."""
    llm = llm or NemotronClient(max_tokens=900)
    status = county_status(df)
    tools = {
        "county_status": lambda a: status,
        "plan_visits": lambda a: plan_visits(df, a.get("visits", 10)),
        "uncertainty_curve": lambda a: uncertainty_curve(df, a.get("max_visits", 60)),
        "supply_impact": lambda a: supply_impact(status, grain_yield),
    }
    messages = [{"role": "system", "content": PLANNER_PROMPT}, {"role": "user", "content": question}]
    allowed, used, retried = {"90"}, [], False   # "90% range" is the method's own wording
    _numbers_in(question, allowed)
    for _ in range(max_steps):
        reply = llm.chat(messages, PLANNER_TOOLS)
        calls = reply["tool_calls"]
        messages.append({"role": "assistant", "content": reply["content"],
                         **({"tool_calls": [{"id": c["id"], "type": "function",
                                             "function": {"name": c["name"], "arguments": c["arguments"]}}
                                            for c in calls]} if calls else {})})
        if calls:
            for c in calls:
                try:
                    args = json.loads(c["arguments"] or "{}")
                    result = tools[c["name"]](args)
                    used.append(c["name"])
                    _numbers_in(result, allowed)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, default=str)})
            continue
        answer = reply["content"].strip()
        bad = ungrounded_numbers(answer, allowed)
        if bad and not retried:
            retried = True
            messages.append({"role": "user", "content": f"These numbers are not in any tool result: {bad}. "
                                                        "Use the tools or remove them, then answer again."})
            continue
        return {"answer": answer, "tools_used": used, "grounded": not bad}
    return {"answer": "I could not finish the plan.", "tools_used": used, "grounded": True}
