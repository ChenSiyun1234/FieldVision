"""Model choices and settings for the FieldVision bridge. Change them here, nowhere else.

Scores are from FieldVision's registry README and docs/model.md (Beltsville, 44 field-years,
12 silage; "always grain" scores 32).
"""

from ..settings import COUNTY_DATA, FIELDVISION as FIELDVISION_ROOT, FIELDVISION_FINAL, QUICKSTATS_CSV, RAG_PIPELINE  # noqa: F401

# The final model module (FieldVision HANDOFF.md, GitHub main 0f9e5df). The agent's "standard" route
# runs FINAL_POLICY, whose field classifier is STANDARD_MODEL: a six-feature logistic trained on 8,000
# European fields (Austria, France, Netherlands), 42 of 44 on Beltsville zero-shot.
STANDARD_MODEL = "eu2_logistic_v1"
FINAL_POLICY = "module_v2"         # evidence tiers + STANDARD_MODEL + county total on the clear tier
SUPERSEDED_POLICIES = {"module_v1": "superseded by module_v2"}

# Why each registry model is or is not used by the agent.
MODEL_DECISIONS = {
    "eu2_logistic_v1": "default (module_v2): 42/44 on Beltsville, trained on 8,000 fields in AT, FR, NL",
    "eu_logistic_v1": "previous default (Austria only, 42/44); kept for comparison",
    "eu_barc_logistic_v1": "only through the toolbox's in_site route (FIPS 24033): 43/44, trained on Beltsville labels",
    "eu_timing_rule_v1": "only through the regional_threshold rerun: 38/44",
    "composite_v1": "not used: superseded by module_v2",
    "eu_gbm_v1": "not used: 30/44, below always-grain, did not transfer",
    "ts_model_v1": "not used: same model as eu_gbm_v1",
}

# FieldVision calibrates p_silage on 50/50 balanced training samples (docs/model.md), so county
# totals need the probabilities shifted to the county's expected silage share.
TRAINING_PRIOR = 0.5
# The prior describes a whole county, so it is applied only when the classified fields cover at least this
# share of the county's harvested corn (Quick Stats). One farm, such as Beltsville, keeps its own mix and is
# left at the model's output.
MIN_COUNTY_COVERAGE = 0.5
FALLBACK_PRIOR = (0.07, "U.S. silage share of harvested corn, about 6-8 % (USDA NASS; FieldVision docs/model.md)")

ACRES_PER_HECTARE = 2.47105
RANGE_DRAWS = 2000
RANGE_SEED = 0
