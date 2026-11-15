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
        # Upper bound: every double-crop acre counted as silage, 22,700 + 17,056 (8.0 %, log 19:14).
        "upper_bound_silage_acres": 39_756,
    },
    "Pennsylvania": {"our_silage_share": None, "source": "to come"},
    "Wisconsin": {"our_silage_share": None, "source": "to come"},
}

# =================================================================================================
# Numbers used only by the slide deck (deck_figs.py, build_deck.py)
# =================================================================================================

# Fields behind the work. Labelled: log 17:40 (8,000 European field-years). Unlabelled US corn
# fields: four Maryland counties 2022, log 18:21 (7,243); Branch and Gratiot, log 19:11 (3,390 and
# 3,463); Maryland 2025 sample drawn, docs/us-county-fields.md (3,929, of which 3,611 corn fields
# were scored and 318 are double crop, log 19:11).
FIELDS_LABELLED_EUROPE = 8_000
FIELDS_US_MARYLAND_2022 = 7_243
FIELDS_US_MICHIGAN_2022 = 3_390 + 3_463
FIELDS_US_MARYLAND_2025 = 3_929
FIELDS_US_TOTAL = FIELDS_US_MARYLAND_2022 + FIELDS_US_MICHIGAN_2022 + FIELDS_US_MARYLAND_2025  # 18,025

# The first county total: Queen Anne's County 2022, check declared before it was opened
# (log 16:12), result log 16:40: 22.4 % of the counted corn area called silage, Census 1.3 %.
FIRST_COUNTY_TOTAL = {"county": "Queen Anne's", "ours": 22.4, "census": 1.3}

# Tuning 1. docs/module.md, "The layers": in Queen Anne's the share of fields called silage rose
# with the gap between the two images around the cut. Same doc, "What was validated where": the
# check on the labelled fields by evidence group (predictions already scored, nothing new run).
GAP_EVIDENCE = [("gap of 5 to 10 days", 6.6), ("gap of 15 to 20 days", 60.0)]
LABELLED_BY_EVIDENCE = {"clearly_seen": (24, 0), "partly_seen": (12, 1)}  # (field-years, wrong)

# Tuning 2. Limits written at 17:30 (log), before Kent or Carroll was opened; results log 18:11, 18:16.
MARYLAND_TEST = [
    # county, Census %, read %, verdict, pass band, weak bands
    ("Kent", 5.5, 21.0, "FAIL", (0.0, 12.5), [(12.5, 20.5)]),
    ("Carroll", 8.9, 0.0, "weak, low side", (1.9, 15.9), [(0.0, 1.9), (15.9, 23.9)]),
]

# Tuning 3. Adding up by two-week harvest window, windows with no clearly seen field counted as
# zero: log 18:54 (county, Census %, plain count %, by window %). Corn cut in the one cloudy
# fortnight (days 252-265): log 19:05. The same idea with the harvest date taken as the middle of
# the gap: docs/module.md on main, "What the imagery could not see".
WINDOW_FIX = [
    ("Queen Anne's", 1.3, 4.0, 2.0, 57),
    ("Kent", 5.5, 12.9, 6.2, 54),
    ("Carroll", 8.9, 3.6, 5.1, 0),
    ("Frederick", 19.6, 13.2, 15.1, 0),
]
WINDOW_FIX_WORST = (7.4, 3.8)
WINDOW_FIX_OTHER_DATE = (10.7, 10.7, 17.1, 15.8)

# Tuning 4. docs/module.md on main, "Why the Eastern Shore reads high": crop history of Kent's
# clearly seen silage calls, from field boundaries (share of their acres that grew each crop).
KENT_CALLS_HISTORY = [("Soybeans", 97), ("Wheat", 59), ("Hay or alfalfa", 6)]
FREDERICK_CALLS_HAY = (16, 22)
KENT_CALLS_CUT_EARLY = (72, 75)  # cut by 2 September

# Slide "why simple won". Mean over the three held-out countries and mean loss:
# docs/foundation-models.md, augmentation table. Sequence model: the stream experiment (season
# read as a sequence of per-date band medians; see below). Last column: the 44 labelled US
# field-years (log 18:02, 19:16).
TRANSFER_SUMMARY = [
    ("Small model", "6 features, logistic regression", 0.765, 0.005, "42 of 44"),
    ("Big model", "135 features, boosted trees", 0.764, 0.047, "37 of 44"),
    ("Big model, dates dropped", "observation dates removed at random in training", 0.780, 0.035, "39 of 44"),
    ("Sequence model", "the whole season as a sequence; three seeds", 0.817, 0.031, "42 of 44"),
]
# The sequence model on the 44 labelled US field-years: 42 right, 11 of 12 silage found, 1 grain
# called silage; pass mark (40 of 44 and 10 of 12 silage) written first. Log 23:03 (model and
# mark), 23:39 (result). This was the fourth scoring of that set. On the eight seen counties it is
# no better than the small model (mean miss 5.7 points for both, Marathon the worst for both), so
# the small model stays. Where it is steadier: fields whose harvest fell between two pictures far
# apart ("partly seen"), share called silage, small model / sequence model (log 23:39).
SEQUENCE_PARTLY_SEEN = [("Queen Anne's", 18.3, 1.7), ("Racine", 35.4, 9.7), ("Kent", 25.5, 12.8)]
LABELLED_SCORINGS = 4
# Sequence model, three seeds, each country held out in turn: mean 0.817 (Austria 0.845, France
# 0.854, Netherlands 0.752), against 0.765 and 0.764 above; loss against in-country 0.042 / 0.024 /
# 0.026, mean 0.031. From the stream experiment's note of 14 Nov, confirmed by the coordinator (it
# replaces the one-seed 0.797). Not tested on the labelled US fields.

# Tuning 1, the replay on labelled fields (modelling work, access log 19:48 and docs/RESULTS.md):
# Beltsville with the small model, labelled grain called silage: 3.1 % on the full series, 27.5 %
# and 39.4 % with 28 and 35 days of observations around the drop removed. Fields that stay
# clearly seen after that (few): 0 % at Beltsville, 0-5 % in France 2022.
REPLAY = {"full": 3, "hidden": (28, 39), "clear": (0, 5)}

# Every place with a figure: docs/RESULTS.md on main, "County and state shares of corn area
# harvested for silage", and the log rows it names (18:09-19:23). Per place: USDA share (Census
# 2022 for counties, NASS 2025 for the state), the USDA range for the fields we score where
# double-crop corn lies outside them, our plain count of the clearly seen fields, the same added
# up by two-week harvest window, and the share of corn area in windows with no clear view
# (log 19:05, 19:11, 19:14). "tuning": looked at while the method was built. "test": never looked
# at before, run once under rules logged first.
PLACES = [
    ("tuning", "Queen Anne's", 1.3, None, 4.0, 2.0, 57.2),
    ("tuning", "Kent", 5.5, None, 12.9, 6.2, 54.4),
    ("tuning", "Carroll", 8.9, (4.8, 7.5), 3.6, 5.1, 0.1),
    ("tuning", "Frederick", 19.6, (11.2, 19.9), 13.2, 15.1, 0.5),
    ("test", "Branch, Michigan", 1.8, None, 2.1, 3.6, 0.8),
    ("test", "Gratiot, Michigan", 24.2, None, 30.0, 29.8, 0.2),
    ("test", "Racine, Wisconsin", 2.4, (2.4, 6.4), 6.4, 6.3, 0.6),
    ("test", "Marathon, Wisconsin", 49.9, (32.1, 49.9), 4.3, 10.0, 0.4),
    ("test", "Maryland 2025 (state)", 8.9, None, 4.75, 3.8, 0.6),
]
# Wisconsin 2023 (log 22:52 and 22:55 for the rule, 22:58 for the result; each county run once).
# USDA figure: the 2023 survey; band: the official figures across sources and years (Census 2022
# to survey 2023). Racine reads inside its band. Marathon (a seeded draw of a third of its corn)
# reads 27.8 points below its band; 48 % of its fields could not be judged and the clearly seen
# fields are 32 % of the area. As a pair the test FAILED.
PLACE_MISSES = {"Marathon, Wisconsin": 27.8}
MARATHON_NOT_JUDGED = 48
# Which of the two is read as "ours": the plain count, except where more than 20 % of the corn
# area lies in windows with no clear view, where it is the count by harvest window (log 19:33; on
# everything seen so far that means Queen Anne's and Kent).
NOT_SEEN_SWITCH = 20
# Not opened: kept as the clean test of the next change to the module (log 23:00).
PLACES_HELD = ["Pennsylvania, Wisconsin 2025"]

# The judges' rubric asks for precision, recall, false positive rate and response time. On the 44
# labelled field-years the small model's calls were (log 18:02): 11 silage found, 1 silage missed,
# 1 grain called silage, 31 grain right. So recall 11 of 12, precision 11 of 12, false positive
# rate 1 of 32.
CONFUSION = {"silage_found": 11, "silage_missed": 1, "grain_called_silage": 1, "grain_right": 31}
# Response time, measured by the coordinator on the laptop with other jobs running (message of 14
# Nov, about 23:40): Gratiot County, 3,463 fields, from stored time series to a silage or grain
# call for every field in 47 seconds (Toolbox.classify, policy module_v2), about 14 ms per field;
# adding the county up took another 57 seconds.
RESPONSE_TIME = {"county": "Gratiot", "fields": 3_463, "seconds": 47, "ms_per_field": 14, "adding_up_seconds": 57}

# Why Marathon read low. Source: log 23:33 ("Marathon diagnosis and safety net"), which also
# holds the coordinator's monthly counts of clear looks per corn field (a look counted when at
# least 60 % of the field is clear), the radar check and the Landsat scenes.
# One cloudy spell, 20 September to 15 October 2023, lay over the silage harvest: 39 % of
# Marathon's fields were cut inside that 25-day gap and could not be judged; the fields seen
# clearly on both sides of the cut are the late harvest, which is grain. The "not seen" measure
# missed it because it placed fields by the first picture after the cut.
MARATHON_GAP = {
    "days": 25, "start": "20 September", "end": "15 October", "fields_cut_inside": 39,
    "landsat_clear": "1 October",
    "months": ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"],
    "marathon": [2.1, 5.4, 6.6, 3.8, 3.8, 1.1, 2.6, 5.9],
    "racine": [1.2, 6.0, 6.2, 6.0, 5.2, 4.0, 4.6, 3.0],
}
# A safety net, not a test (same message): mean miss on the eight counties already seen, against
# the year-matched official figure. The land-use figure was fitted on the 2022 Census, which
# includes these counties.
# What sees into the gap (the county work's check, relayed by the coordinator, 14 Nov about 23:40):
# Sentinel-1 radar has no scene over Marathon between 18 September and 12 October 2023, so radar
# cannot date a harvest there; Landsat 9 passed on 1 October 2023 with 6 % cloud and Landsat 8 on
# 9 October with 24 % (Planetary Computer, landsat-c2-l2). The Landsat extraction was being built
# when this was written: no result yet.
SAFETY_NET = {"counties": 8, "satellite": 8.6, "land_use": 8.2, "both": 6.8}
