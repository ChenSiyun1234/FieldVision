"""Every number shown in the pitch pack, each with where it comes from.

Two kinds of source:
  "log HH:MM"  a row of the evaluation access log (<data>/eval/ACCESS_LOG.md), named by its time on
               14 Nov 2026 (Pacific);
  "docs/..."   a file in this repository.

A value of None is a result that was not logged yet when the pack was drawn. The figures draw a
marked place for it. To fill one: put the value here with its source, then run draw.py again.
"""

# --- 1. What the small model keys on -------------------------------------------------------------
# docs/features.md, table "The six features the small model uses": drop in held-out ROC-AUC when
# one feature is shuffled, mean and spread over five folds. Austria 2024, 2,400 fields.
FEATURES = [
    ("Moisture just before harvest", "how wet the standing crop is", 0.103, 0.020),
    ("Residue left after harvest", "stalks on the ground after the cut", 0.049, 0.019),
    ("Length of the dry-down", "days from the start of the decline to harvest", 0.038, 0.030),
    ("How fast it browns", "slope of greenness before harvest", 0.030, 0.017),
    ("Greenness just before harvest", "as a share of the field's own peak", 0.008, 0.007),
    ("Days from peak to harvest", "the date on its own", 0.000, 0.001),
]

# --- 2. A country the model never saw ------------------------------------------------------------
# docs/foundation-models.md, table "Three countries, each left out in turn" (8,000 fields:
# Austria 2022-2024, France 2022, Netherlands 2025). Per held-out country: ROC-AUC there, and the
# loss against the same model tested inside that country (spatial cross-validation).
TRANSFER_COUNTRIES = ["Austria", "France", "Netherlands"]
TRANSFER = [
    ("Small model", "6 features, logistic regression", [(0.754, 0.015), (0.826, 0.002), (0.717, -0.001)]),
    ("Big model", "135 features, boosted trees", [(0.770, 0.078), (0.810, 0.036), (0.713, 0.027)]),
    ("Monthly band values", "boosted trees, no hand-made features", [(0.747, 0.093), (0.797, 0.050), (0.715, 0.022)]),
    ("Pretrained satellite model", "Presto embedding", [(0.754, 0.071), (0.650, 0.188), (0.714, 0.025)]),
    ("Big model + pretrained", "both inputs, boosted trees", [(0.798, 0.063), (0.815, 0.047), (0.733, 0.023)]),
]

# --- 3. The organisers' 44 labelled field-years (Beltsville, 2021-2023) --------------------------
# log 18:02, "EVALUATION 2 RESULT": models trained on every European field-year, scored once.
# Small model: accuracy .955, ROC-AUC .951; 11 silage found, 1 missed, 1 grain called silage, 31
# grain right.
LABELLED_TOTAL = 44
LABELLED = [
    ("Small model", "6 features, logistic regression", 42, "main"),
    ("Big model", "135 features, boosted trees", 37, "other"),
    ("Timing + moisture rule", "two thresholds", 36, "other"),
    ("Average of three simple models", "", 36, "other"),
    ("Timing rule", "harvest date only", 35, "other"),
    ("Always say grain", "no model at all", 32, "baseline"),
]
# log 13:53 and 14:07 (first scoring, models trained on Austria 2024 only), docs/module.md table.
LABELLED_FIRST_RUN = {"small": 42, "big": 30, "always_grain": 32}
# log 18:50: one more scoring of the big model trained with dates dropped (eu2_gbm_thin_v1),
# approved beforehand; it passes if it scores more than 37.
THINNED_BIG_MODEL_CORRECT = None
THINNED_BIG_MODEL_SOURCE = "to come: its own row in the access log"

# --- 4. Dropping observation dates in training ---------------------------------------------------
# docs/foundation-models.md, table under "Can augmentation stop the big model memorising?":
# 135-feature boosted trees, mean ROC-AUC over the three held-out countries ("mean out").
AUGMENT = [
    ("Big model, as before", 0.764, False),
    ("+ stronger regularisation", 0.765, False),
    ("+ season shifted by up to 3 weeks", 0.766, False),
    ("+ small gain and offset on the bands", 0.758, False),
    ("+ observation dates dropped", 0.780, True),
    ("+ all three changes", 0.780, False),
    ("+ all three, and regularised", 0.777, False),
]
AUGMENT_SMALL_MODEL = 0.765  # same table, row "logistic (6 features), base"
AUGMENT_LOSS = {"big": 0.047, "big_dates_dropped": 0.035, "small": 0.005}  # same table, "mean loss"

# --- 5. County totals against the 2022 Census ----------------------------------------------------
# Census 2022 silage share of harvested corn, and the range for the scored field set in Frederick
# (part of its silage sits in double-crop fields that are not scored): log 18:09, 18:16, 18:54.
# "first": the county step fixed before the test counties were opened (local threshold on the
#          clearly seen fields, log 18:09). Kent and Carroll were each run once: log 18:11, 18:16.
# "second": harvest-window weighting, weeks without a clear view counted as zero. Found on these
#          four counties after Kent failed: log 18:54. "unseen" = share of corn area in weeks
#          without a clear view (same row; Carroll and Frederick are "under 1 %").
MARYLAND_2022 = [
    # county, Census %, first method % (None = not a test county), verdict, second method %, unseen %
    ("Queen Anne's", 1.3, None, "", 2.0, 57),
    ("Kent", 5.5, 21.0, "failed", 6.2, 54),
    ("Carroll", 8.9, 0.0, "weak", 5.1, 0),
    ("Frederick", 19.6, None, "", 15.1, 0),
]
FREDERICK_SCORED_RANGE = (11.2, 19.9)
KENT_FAIL_LINE = 20.5  # log 17:30: fail above 20.5 %
SECOND_METHOD_WORST_MISS = 3.8  # log 18:54 (Carroll); the plain share of clear fields: 7.4

# Michigan 2022, the clean test of the second method: log 18:25 (criteria), 18:37 and 18:54
# (when it is "not testable"). Census shares from the 18:25 row. Results: None until logged.
MICHIGAN_2022 = [
    # county, Census %, our headline %, unseen %
    ("Gratiot", 24.2, None, None),
    ("Branch", 1.8, None, None),
]
MICHIGAN_VERDICT = None  # "pass", "fail" or "not testable"
MICHIGAN_SOURCE = "to come: its own row in the access log"

# --- State acres, 2025 (what the organisers score) -----------------------------------------------
# USDA NASS 2025 for Maryland: log, section "State targets for 2025". Sample and double-crop acres:
# log 18:25 and docs/us-county-fields.md, "2025 statewide samples".
STATE_2025 = {
    "Maryland": {
        "nass_silage_acres": 40_000, "nass_grain_acres": 410_000, "nass_silage_share": 8.9,
        "blocks_drawn": 150, "blocks_total": 838, "fields_drawn": 3_929,
        "double_crop_acres_not_scored": 17_056,
        "our_silage_acres": None, "our_grain_acres": None, "our_silage_share": None,
        "our_range_share": None, "unseen_share": None, "upper_share": None,
        "source": "to come: its own row in the access log",
    },
    "Pennsylvania": {"our_silage_share": None, "source": "to come"},
    "Wisconsin": {"our_silage_share": None, "source": "to come"},
}
