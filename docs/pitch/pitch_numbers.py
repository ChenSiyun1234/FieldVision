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
    ("Big model + band values", "both inputs, boosted trees", [(0.789, 0.071), (0.823, 0.040), (0.727, 0.030)]),
]
# The two rows of that table left out of the figure are the two hand-written rules (simpler than
# the small model). Paired differences to the small model with Austria held out, same table:
# big model +0.016, big + band values +0.035, big + pretrained +0.045 (intervals clear of zero).

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
# log 18:50 (conditions, written before it ran) and 19:16 (result): one more scoring, of the big
# model trained with dates dropped (eu2_gbm_thin_v1). It passes if it scores more than 37.
# Result: 39 of 44, ROC-AUC .932. After it the set is closed.
THINNED_BIG_MODEL_CORRECT = 39
THINNED_BIG_MODEL_SOURCE = "log 19:16"

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
# Census 2022 silage share of harvested corn. For Frederick and Carroll part of the silage sits in
# double-crop fields that are not scored, so the comparison is a range for the scored fields:
# Frederick log 18:09, Carroll log 19:05.
# "first": the county step fixed before the test counties were opened (a local cut-off on the
#          clearly seen fields, log 18:09). Kent and Carroll were each run once: log 18:11, 18:16.
# "count": the plain count = share of the clearly seen corn area that the small model calls
#          silage ("counted clear share"). Maryland values: log 18:33; it is the headline again
#          since log 19:05 (a harvest-window variant was tried at 18:54 and withdrawn at 19:05
#          because its result flips with the date convention).
MARYLAND_2022 = [
    # county, Census %, range for the scored fields, first method (%, verdict) or None, plain count %
    ("Queen Anne's", 1.3, None, None, 4.0),
    ("Kent", 5.5, None, (21.0, "failed"), 12.9),
    ("Carroll", 8.9, (4.8, 7.5), (0.0, "weak"), 3.6),
    ("Frederick", 19.6, (11.2, 19.9), None, 13.2),
]
KENT_FAIL_LINE = 20.5  # log 17:30: fail above 20.5 %
PLAIN_COUNT_WORST_MISS_MD = 7.4  # log 18:37 (Kent)

# Michigan 2022, the clean test of the plain count. Rule written first (log 18:25): Gratiot at
# least 10 points above Branch, neither more than 10 points off its Census share; minimum of
# clearly seen fields (log 18:37). Each county run once: log 19:07. Result: log 19:11.
MICHIGAN_2022 = [
    # county, Census %, plain count %, clearly seen share of corn area %
    ("Branch", 1.8, 2.1, 64),
    ("Gratiot", 24.2, 30.0, 83),
]
MICHIGAN_VERDICT = "passed"  # log 19:11: 27.9 points apart; off by +0.3 and +5.8
MICHIGAN_GAP = 27.9
MICHIGAN_SOURCE = "log 19:11"

# --- State acres, 2025 (what the organisers score) -----------------------------------------------
# USDA NASS 2025 for Maryland: log, section "State targets for 2025". Procedure: log 18:25, 18:37,
# 18:55, 19:05. Run once: log 19:11. Result: log 19:14 (silage share 4.75 % [3.4, 6.3] of the
# clearly seen corn; acres = share x 477,891 corn acres from field boundaries).
STATE_2025 = {
    "Maryland": {
        "nass_silage_acres": 40_000, "nass_grain_acres": 410_000, "nass_silage_share": 8.9,
        "blocks_drawn": 150, "blocks_total": 838, "fields_scored": 3_611,
        "double_crop_acres_not_scored": 17_056,
        "our_silage_acres": 22_700, "our_silage_acres_range": (16_200, 29_900),
        "our_grain_acres": 455_200,
        "our_silage_share": 4.75, "our_silage_share_range": (3.4, 6.3),
        "clearly_seen_share_of_area": 62,
        "share_if_all_double_crop_is_silage": 8.0,
        "points_low": 4.2,
        "corn_acres_boundaries": 477_891, "corn_acres_nass": 450_000,
        "source": "log 19:14",
        # Shown beside, never instead. Harvest-window variant of the same count, computed in the
        # same run: log 19:18 (Michigan under it: Branch 3.6 %, Gratiot 29.8 %, still a pass).
        "window_variant_share": 3.8, "window_variant_range": (2.6, 4.9),
        # Double-crop corn scored with the same model, AFTER the headline was reported, by a rule
        # written for the states not yet opened: log 19:21, corrected 19:23. No labelled
        # double-crop field exists, so these calls are untested.
        "with_double_crop_silage_acres": 33_135, "with_double_crop_share": 6.7,
        "with_double_crop_range": (5.3, 8.2), "all_corn_acres": 494_947,
    },
    "Pennsylvania": {"our_silage_share": None, "source": "to come"},
    "Wisconsin": {"our_silage_share": None, "source": "to come"},
}
