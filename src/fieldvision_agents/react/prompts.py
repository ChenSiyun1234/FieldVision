"""System prompt and tool schemas for the supervisor agent."""

POLICIES = ("standard", "radar_first", "regional_threshold")

# Critic suggested_action -> routing policy for the single allowed rerun.
RERUN_POLICY = {
    "rerun_with_radar": "radar_first",
    "rerun_with_regional_threshold": "regional_threshold",
}

SUPERVISOR_PROMPT = """You are the supervisor agent for a USDA NASS estimate of corn silage vs grain acres.
You run ONE county-season at a time, only by calling tools.

Steps:
1. scout: see what imagery the county's fields have this season.
2. route_options: see how each routing policy would assign frozen models to the fields.
   Pick one policy that route_options lists as available, and give a one-sentence reason based on the
   scout and route_options results (read each option's note: it says which model runs and how it was validated).
3. extract_features, then classify with the policy you picked.
4. aggregate, then run_critic.
5. If the critic's suggested_action is a rerun, you may classify ONCE more with the policy it implies
   (rerun_with_radar -> radar_first, rerun_with_regional_threshold -> regional_threshold),
   then aggregate and run_critic again. Never more than one rerun.
   If the implied policy is the one you already used, do not rerun: finish with accepted_with_warnings or abstain.
6. If the critic does not accept after the rerun, or the data is too poor, call abstain_region.
   Optional: after classify, you may call inspect_gaps once. It shows how many fields the model could not
   judge because cloud hid the harvest, the dates of the main cloud gap, and the range the county share
   could take. Set vision_fields (up to 12) to have the vision model look at before and after images of
   some of those fields. It never changes the county estimate. If it reports cloud_gap_at_harvest true,
   add one sentence to the summary with the gap dates and the range, copied exactly.
   If it reports reads_low_warning true, say that the estimate likely reads low and give
   not_judged_share_of_corn_pct and silage_share_if_all_not_judged_were_silage_pct, copied exactly.
   Optional: after a run_critic result that is not "fail", you may call recommend_sources once. It matches
   dairies short on silage with standing grain-bound fields nearby. It never changes the county estimate.
   If you call it, add one sentence to the summary with its matched tons, copied exactly.
7. Always end by calling finish.

Rules:
- Every number you write must be copied exactly from a tool result. Never calculate, round, estimate or invent numbers.
- Do not pick regional_threshold as your first policy. It moves the drop-date cutoff to the county's median
  drop day, which splits fields about 50/50 whatever the true mix; use it only as a rerun the critic suggests.
- You cannot train models, change thresholds or edit features. You only choose among the policies route_options offers.
- Before each tool call, write one short sentence saying why you are calling it.
- finish.summary is 3 to 5 sentences: what data existed, which policy and why, silage acres with the low-high range,
  abstained acres, and the critic's verdict.
"""

FINISH_STATUSES = ("accepted", "accepted_with_warnings", "abstained")


def _tool(name, description, properties=None, required=None):
    return {"type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
    }}


_COUNTY = {
    "fips": {"type": "string", "description": "5-digit county FIPS code as text, e.g. '24021'"},
    "year": {"type": "integer", "description": "Season year, e.g. 2022"},
}

TOOL_SPECS = [
    _tool("scout",
          "Summarise the imagery each field in the county has this season: clear optical observations, "
          "cloud gaps over the harvest window, radar coverage. Call this first.",
          _COUNTY, ["fips", "year"]),
    _tool("route_options",
          "For each routing policy, show how many fields would go to each frozen model and how many would abstain. "
          "Use it to choose a policy.",
          _COUNTY, ["fips", "year"]),
    _tool("extract_features",
          "Compute the per-field feature table for the county-season with the shared feature engine.",
          _COUNTY, ["fips", "year"]),
    _tool("classify",
          "Label every field with frozen models, assigned by the chosen routing policy. "
          "Allowed twice at most: the first run and one critic-requested rerun.",
          {**_COUNTY, "policy": {"type": "string", "description": "One of the policies route_options lists as available"},
           "reason": {"type": "string", "description": "One sentence: why this policy, citing scout or critic results"}},
          ["fips", "year", "policy", "reason"]),
    _tool("aggregate",
          "Sum the latest field results to county silage, grain and abstained acres with an uncertainty range.",
          _COUNTY, ["fips", "year"]),
    _tool("run_critic",
          "Check the latest county result against QuickStats history (excluding this year), planted acres "
          "and dairy density. Returns overall pass/warn/fail and a suggested_action.",
          _COUNTY, ["fips", "year"]),
    _tool("inspect_gaps",
          "Optional, after classify: count the fields the model could not judge because cloud hid the harvest, "
          "name the main cloud gap, and give the range of the county silage share. With vision_fields, the "
          "vision model looks at before and after images of that many of those fields. "
          "Does not change the county estimate.",
          {**_COUNTY,
           "vision_fields": {"type": "integer", "description": "0 to 12 fields to show the vision model; default 0"}},
          ["fips", "year"]),
    _tool("recommend_sources",
          "Optional, after run_critic (not fail): match dairies that are short on silage with nearby standing "
          "corn fields predicted grain-bound, as of a date. Returns matched tons per dairy, prices and checks. "
          "Does not change the county estimate.",
          {**_COUNTY,
           "as_of": {"type": "string", "description": "YYYY-MM-DD; default September 1 of the year"},
           "shortfall_frac": {"type": "number", "description": "Share of each dairy's yearly need it asks for; default 0.15"}},
          ["fips", "year"]),
    _tool("abstain_region",
          "Mark the county-season as abstained instead of guessing. Use when data is too poor or the critic "
          "rejects the result after the one rerun. Afterwards, call finish with status 'abstained'.",
          {**_COUNTY, "reason": {"type": "string"}},
          ["fips", "year", "reason"]),
    _tool("finish",
          "End the run with a status and a short summary whose numbers are copied from tool results.",
          {"status": {"type": "string", "enum": list(FINISH_STATUSES)},
           "policy_used": {"type": "string", "description": "The last policy classified with, or 'none'"},
           "summary": {"type": "string"}},
          ["status", "policy_used", "summary"]),
]
