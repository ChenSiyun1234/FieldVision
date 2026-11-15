# External evaluation: ZERO-SHOT

2026-11-14 18:02. Run `0484d7e43267`, primary model `logistic` (chosen at training time), training threshold 0.500. Positive class = silage.

Nothing was fit on these fields: model, threshold, feature settings and the low-evidence rule all come from the training run. Evaluated once (ledger entry written).

Frozen model identifier: `e2623ab2bf112150` (model `logistic`, trained 2026-11-14T17:48:36; sha256 of the model file and its training configuration (first 16 hex); matches the value written at training: yes). The ledger refuses a second zero-shot evaluation of this set with the same frozen model.

**Warning:** this field set was evaluated before 1 time(s) (any run). See the ledger.

44 labelled fields (0 unlabelled ignored). Intervals: only 1 10 km block(s). Count metrics carry exact binomial (Clopper-Pearson) 95% intervals; ROC-AUC and PR-AUC a bootstrap over the 34 physical fields (all years of a polygon resampled together). Both treat fields as independent, which fields on one farm are not, so the true uncertainty is larger.

## Claim and validation design

Claim: the timing and state of the harvest drop seen by Sentinel-2 separates silage from grain corn, and a model learned on European parcel registries (Austria 2022-24, Netherlands 2025, France 2022) transfers to US fields without retraining.

Design: all model choices were made on the European data with splits grouped by 10 km block (never random fields), plus leave-one-country-out and leave-one-year-out. This set was not used for any fitting, tuning or feature choice and is scored once. With this few fields it confirms or contradicts the European transfer results below; it cannot replace them.

### What was seen of this set before the freeze

Quoted from the team's access log (kept by the coordinating session; the row for this evaluation is added after this report is written):

| Time | Who | What was looked at | Used to change anything? |
|---|---|---|---|
| 11:29 | lead | Downloaded the GeoPackage from the team workspace; conversion script printed column names, class counts per year (2021: 14 grain, 4 silage; 2022: 3, 4; 2023: 15, 4), field areas and bounds | No. Counts were needed to write the per-year field files |
| 11:33-11:36 | extraction track | Extracted Sentinel-2 time series for all 44 field-years with the same settings as the European runs; printed coverage statistics (usable dates, gaps, pixel counts) | No. Settings were fixed before and not changed after |
| about 11:36 | extraction track, at the lead's request | Plotted NDVI and NDMI for four labelled field-years (silage: barc-280-2021, barc-248-2023; grain: barc-1-2021, barc-1-2023) and described the curves in words | No thresholds, features or settings were set from them. The description (abrupt drop from a green, wet state for silage; gradual dry-down for grain; green cover crop in spring and after harvest) was shown to the team member. The lead then reminded the modelling track to handle double cropping, a trap the organisers list in their own slides; that handling is developed and tested on European data |
| between 11:37 and 11:47 | foundation-model track | Opened `barc_2021/fields.geojson` once: counted 18 features and printed one field's properties, including its label; read one line of the run manifest. No time series read, nothing scored | No |
| about 11:44 (logged afterwards, at 12:00) | modelling track | Format check before the freeze, after the 11:36 rule (I had not seen the rule yet): loaded `barc_<year>/fields.geojson` and the matching time series (whole `barc_<year>` folders) through the project loaders and printed only: number of field-years per year (18, 7, 19), number with time-series rows, number of ids missing (0), the `split` and `country` values, the `collection` values present, and the loader's duplicate-row notes. The loader reads the label column into memory; no label, class count, curve, feature value or score was printed, plotted or computed | No. Nothing in the code, features or thresholds was changed from it |
| about 11:55 | labels track, at the lead's request | Read the Silage/Grain columns of `BARC_Corn.2021-2023.gpkg` once, only to check that the new generic converter (`fieldvision.labels.from_file`) reproduces the existing `barc_2021`, `barc_2022`, `barc_2023` field files: counts per year and class, ids, areas. No imagery, no time series, nothing scored | No. Existing files are not overwritten; output goes to `labels/_check_barc/` |
| 13:53 (written before Evaluation 1 runs) | modelling track, on the lead's decision | Plan declared in advance: the set is scored exactly twice and both results are reported whatever they show. Evaluation 1 (now): run `24500e02a9e0` from `C:\t\fv\data\model\runs\at2024_full` (trained 12:07 on all 2,400 Austria 2024 fields, code at commit bd4cc94), LightGBM frozen model id `72b208f3297397be`, unchanged; the timing-rule baseline from the same run (`dd9379ee699047de`, threshold fitted on Austria) and always-grain scored in the same command. Evaluation 2 (tonight): the model trained on all European data available then, its design frozen on European cross-validation before it is run. Nothing learned from Evaluation 1 may change features, thresholds or model choice for Evaluation 2; anything it suggests is written in the report as an observation | Not yet run when this line was written |
| 13:53 | modelling track | zero-shot evaluation, run once with frozen model `72b208f3297397be`: `python -m fieldvision.model.evaluate --model C:\t\fv\data\model\runs\at2024_full --timeseries C:\t\fv\data\timeseries\barc_2021\timeseries.parquet C:\t\fv\data\timeseries\barc_2022\timeseries.parquet C:\t\fv\data\timeseries\barc_2023\timeseries.parquet --fields C:\t\fv\data\labels\barc_2021\fields.geojson C:\t\fv\data\labels\barc_2022\fields.geojson C:\t\fv\data\labels\barc_2023\fields.geojson --keep-columns OBJECTID --out C:\t\fv\data\model\eval\beltsville_eval1 --access-log C:\t\fv\data\eval\ACCESS_LOG.md`; output `C:\t\fv\data\model\eval\beltsville_eval1` | This is the evaluation |
| 13:58 | modelling track, on the lead's instruction | Evaluation 2 design fixed with nothing left to decide: exactly the `at2024_full` configuration (code at commit bd4cc94 plus the report-layout fix 7d5f78a; same feature set, feature settings, LightGBM primary, fixed hyper-parameters, 0.5 threshold on Platt-calibrated probabilities, same low-evidence rule, timing rule and always-grain scored beside it). The only change is the training data: all European field-years listed in timeseries/READY.txt at the time it is run. Written before any further work that touches Beltsville labels (the team's `timing_rule_v1`, which fits thresholds on Beltsville by leave-one-year-out, is a separate model and cannot feed into it) | No |
| 14:01 | modelling track | Ran `fieldvision.model.classify` with the frozen registry models `ts_model_v1` (= Evaluation 1's LightGBM, 72b208f3297397be) and `eu_timing_rule_v1` (= Evaluation 1's timing rule) on the 44 Beltsville field-years to write the team's result files (C:\t\fv\data\results\MD_<year>.parquet). The field loader reads the label column; classify drops it before predicting and computes no metrics. The predictions are the ones already scored in Evaluation 1 | No |
| 14:04 | modelling track, to the team's specification | Fitting the team's `timing_rule_v1` ON Beltsville labels: thresholds on drop date and pre-drop NDVI chosen by balanced accuracy, validated by leave-one-year-out over 2021/2022/2023 (code fieldvision.beltsville.timing_rule, commit d7c36b2). This is a separate in-site model; it is written after the Evaluation 2 design was fixed (13:58) and cannot feed into it | Yes, by design: this model's thresholds come from these labels |
| 14:07 | modelling track, on the lead's decision | Rule for choosing the Evaluation 2 primary, SET AFTER Evaluation 1 WAS SEEN: the primary is whichever pre-declared family (timing rule, timing+moisture rule, 6-feature logistic, LightGBM) has the highest mean ROC-AUC under leave-one-country-out on European data, or leave-one-year-out across the Austrian years if only one country is available by then; a difference smaller than the spread across held-out groups goes to the simpler model. All four are scored and shown in Evaluation 2 whatever the primary is. The Evaluation 2 report will say that the pre-declared Evaluation 1 primary (LightGBM, accuracy 0.682) did worse than always-grain (0.727) on accuracy, and that the logistic's 42 of 44 was a secondary result | Model choice for Evaluation 2 is decided by this rule on European data only |
| 14:07 | modelling track, on the lead's decision | ADDITION AFTER Evaluation 1: a fifth family, `composite_v1` (combining the timing rule, timing+moisture rule and 6-feature logistic, with abstention), joins the Evaluation 2 candidates under the same rule. It is designed and validated on European data only and is not scored on Beltsville before Evaluation 2 | No |
| 14:09 | modelling track, for the demo | Wrote per-field predictions of the frozen `eu_logistic_v1` (the Evaluation 1 secondary logistic, trained on Austria only) for the 44 field-years in the app's predictions format, and ran `fieldvision.model.classify` with it for the team's result files. Labels are dropped before predicting; no metrics computed | No |
| 14:17 | demo track | Read the Beltsville labels (`barc_<year>/fields.geojson`), time series and the already-scored `eu_logistic_v1` predictions for 2021-2023, to display them in the demo app only; the app shows the stored predictions next to the labels and counts right and wrong. Evidence numbers are recomputed with the unchanged feature code and the Austrian training medians | No. No model, feature, threshold or setting was fitted or changed |
| 14:18 | modelling track | Built the toolbox's pseudo-county (C:\t\fv\data\counties, FIPS 24033) from the Beltsville field files WITH ALL LABEL COLUMNS REMOVED, and ran the toolbox on 2022 as a demonstration: predictions from `composite_v1` and the regionally shifted timing rule, areas and counts only. No labels in those files, nothing compared with labels, no metrics; composite_v1 remains unscored on Beltsville until Evaluation 2 | No |
| 14:54 | modelling track, on the lead's request | IN-SITE models fitted ON Beltsville labels by design (package fieldvision.beltsville, separate from the European models, never feeding them or Evaluation 2): (a) six-feature logistic fitted on Beltsville, (b) timing+moisture rule fitted on Beltsville, (c) the European logistic with only its intercept and threshold refitted on the training years, (d) a logistic on Austria 2024 plus the training years with Beltsville up-weighted; validated by leave-one-year-out (same folds as timing_rule_v1) and leave-one-field-out | Yes, by design: these models are fitted on these labels |
| 16:51 | modelling track, on the lead's request | Descriptive look at the evidence features of the 44 labelled Beltsville field-years (bracket width, drop date, greenness and moisture before the drop, usable dates, member votes) as a comparison for the Queen Anne's false-positive analysis. Nothing fitted; Evaluation 2 design was fixed at 13:58 | Possibly: it may inform a change to the evidence layer, which will be marked as prompted by Queen Anne's and checked on Beltsville only as a regression check |
| Time | Who | What | Used to change anything? |
|---|---|---|---|
| 16:12 | modelling track, on the lead's proposal | HEADLINE RULE for a county estimate, fixed now: the headline is the **counted share**: the share of the non-abstained corn area that the policy's model calls silage. Reported beside it, with their ranges: the adjusted share (counted share corrected with European held-out TPR/FPR), its spread across the European validation designs, the raw sum of probabilities, and a prior-shift (EM) re-estimate from the county's own probabilities. A wide spread of the adjusted share across European designs is shown as the warning that Europe-measured error rates do not carry over. Known bias of the headline, stated in advance: in expectation the counted share is TPR x true share + FPR x (1 - true share), so it reads HIGH wherever the true silage share is below the model's false-positive rate in that county (European FPR about 0.16) and low where silage is common. This rule was chosen AFTER seeing the partial Frederick snapshot (11 % of the area: raw sum 33 %, counted 18.9 %, adjusted 4.9 %, Census 19.6 %), so Frederick is no longer a clean test of the choice. Queen Anne's County (FIPS 24035, Census silage share 1.3 %) has not been run through any model; it is declared the untouched check of this rule and nothing is run on it before this line exists | The rule is fixed here; nothing is tuned towards Census numbers |
| 16:39 | modelling track | PRE-REGISTERED CHECK STARTS: running the toolbox on Queen Anne's County (24035, 2022) for the first time, policies standard (composite_v1) and regional_threshold, under the headline rule logged at 16:12 (code at commit b0fe3cd) | No |
| 16:40 | modelling track | PRE-REGISTERED CHECK RESULT, Queen Anne's 2022 (Census silage 715 of 54,530 acres = 1.3 %). standard (composite_v1), 1,449 fields, 403 abstained (14,323 acres): HEADLINE counted share 22.4 %; adjusted (Europe rates) 11.3 % [4.4, 18.0], across European designs 0-11.9 %; raw sum of p 40.1 %; prior-shift EM 0.0 % (boundary). regional_threshold: headline 50.2 %, adjusted 36.5 %, designs 0-83 %, raw 46.2 %, EM 0 %. The headline fails the check, in the direction stated in advance (reads high where the true share is below the false-positive rate); implied local FPR of composite_v1 about 0.22 | No: reported as it came out; nothing tuned |
| 16:50 | modelling track | MODULE_V1 COUNTY PROCEDURE, fixed now, before the third Maryland county is touched: module headline = reference-calibrated share (counted share corrected with the false-positive rate measured in a nearly all-grain reference county of the same region and season, its Census share known, TPR from European held-out fields) when a reference is configured, otherwise the Europe-rate adjusted share; counted share, raw sum, Europe-adjusted spread and prior-shift reported beside it. Chosen because it beat the counted share on European pseudo-counties (held-out year: MAE 0.099 vs 0.154). Chosen AFTER seeing Queen Anne's (used as check and now as reference) and the partial Frederick snapshot; neither is a clean test. On the partial Frederick snapshot with Queen Anne's as reference it gives 0 % (clipped; Census 19.6 %) | No |
| 17:02 | modelling track | MODULE_V2, PROMPTED BY QUEEN ANNE'S (lead's decision), logged before Kent (24029) and Carroll (24013) are touched. Layer 1: evidence tiers from the harvest bracket: clear <= 12 days, wide 12-20 days, abstain > 20 days or on the existing rules; every field keeps a call; the county headline uses the clear tier only, with the wide tier's share and acreage and the clear tier's share of area reported beside it. Layer 2: the calendar timing rule leaves the vote (stays as a baseline). CLASSIFIER CHOICE RULE, fixed now, before computing it: between eu_logistic_v1 alone and the mean of the logistic and the timing+moisture rule, the higher mean ROC-AUC under leave-one-country-out Austria vs France (France is in READY.txt), using out-of-fold predictions of a run on AT 2022+2023+2024 and FR 2022; a difference smaller than the spread across the held-out groups goes to the logistic alone; nothing frozen is refit (the logistic alone means eu_logistic_v1 as it stands) | Prompted by Queen Anne's; the choice is made by this rule on European data only |
| 17:16 | modelling track | MODULE_V2 CLASSIFIER CHOICE RESULT under the 17:02 rule (run at_fr: AT 2022+2023+2024, FR 2022, out-of-fold, relative features). ROC-AUC held out AT / FR: eu_logistic family 0.766 / 0.822 (mean 0.794); mean of logistic and timing+moisture rule 0.763 / 0.809 (mean 0.786). Difference -0.008, below the spread across held-out groups (0.056 / 0.046). Chosen: eu_logistic_v1 alone, as frozen (no refit). For reference, timing rule 0.655 / 0.692, LightGBM 0.783 / 0.761 | Sets the module_v2 field classifier to eu_logistic_v1; nothing from the US used |
| 17:17 | modelling track | MODULE_V2 BELTSVILLE REGRESSION CHECK starts: reading the Beltsville labels and the already-scored Evaluation 1 predictions of eu_logistic_v1 in results/MD_2021-2023.jsonl, split into the clear (<= 12 days) and wide tiers; no model is run on Beltsville, nothing is scored anew | No: reported only; the tier limit and classifier were fixed at 17:02 and 17:16 on European data |
| 17:18 | modelling track | MODULE_V2 RECOMPUTE ON SEEN COUNTIES: Queen Anne's (24035, 2022) and the partial Frederick snapshot (24021, 2022, 248 of 2,233 fields) through policy module_v2 (eu_logistic_v1, clear tier <= 12 days), Frederick also with Queen Anne's as reference county (Census 715 / 54,530 acres). Both were seen before and are labelled as seen, not clean tests. Kent (24029) data is on disk and is not opened | No: descriptive; the county step is decided afterwards from European pseudo-counties |
| 17:30 | modelling track | MODULE_V2 COUNTY STEP AND KENT / CARROLL FAILURE CRITERIA, fixed now, before Kent (24029) or Carroll (24013) is opened. STEP: local threshold on the clear tier (bracket <= 12 days): threshold = 90th percentile of eu_logistic_v1 scores of the clear-tier fields of the reference county Queen Anne's (24035) in the same season (Census 715 / 54,530 acres = 1.3 %); share = (target clear area flagged - reference clear area flagged) / (matched TPR - reference flagged), clipped to 0-1, matched TPR 0.632 from clear:at2024:spatial in registry/eu_logistic_v1/rates.json. WHY: on European pseudo-counties under module_v2 (eu_logistic family out-of-fold, clear tier, 150-field counties, 200 reps, files eval/pseudo_v2) mean abs. error held-out 2022 / 2023 / FR: local 0.087 / 0.086 / 0.082, reference-calibrated 0.072 / 0.328 / 0.068, counted 0.066 / 0.399 / 0.070, Europe-adjusted 0.071 / 0.668 / 0.073; the local threshold is the only step that survives the 2023 shift (clear-tier FPR 0.47 there); with TPR -/+0.1 it stays <= 0.111. Chosen on the lowest worst-case held-out error. Chosen AFTER seeing Queen Anne's and the partial Frederick snapshot, where it gave 6.4 % against Census 19.6 % for the whole county (worst of the four on that unrepresentative 11 % snapshot); stated here, not hidden. Counted, Europe-adjusted and reference-calibrated clear shares are reported beside it. FAILURE CRITERIA (headline = module_headline_share for policy module_v2, Census 2022 share): pseudo-county error of the local step at 5-10 % silage in held-out settings: mean 0.070, 90th pct 0.147. Per county: PASS if |headline - Census| <= 0.07; WEAK if 0.07 < |error| <= 0.15; FAIL if |error| > 0.15 or no headline (fewer than 50 clear fields in target or reference, or matched TPR - reference flagged <= 0.05). Kent (Census 5.5 %): pass 0-12.5 %, weak 12.5-20.5 %, fail above 20.5 %. Carroll (Census 8.9 %): pass 1.9-15.9 %, weak 0-1.9 % or 15.9-23.9 %, fail above 23.9 %. The module_v2 county step FAILS the check if either county fails, or if both are weak on the same side. Kent below Carroll is not tested (3.4 points is inside the noise). Expected bias of the counted clear share (clear FPR about 0.07): Kent about 10 %, Carroll about 12 %; reported, not tested | Fixes the county step; nothing tuned to Kent or Carroll |
| 17:37 | modelling track | FULL FREDERICK (24021, 2022, all 2,233 fields; landed 17:23) run through policy standard = module_v2 at about 17:35, AFTER the 17:30 procedure was fixed. Not planned as a check: I ran it to confirm the reference config and only then saw PARTIAL.json was gone; Frederick was already seen (partial) and is not one of the clean-test counties. Result (Census 7,748 / 39,500 acres = 19.6 %): 195 abstained; clear tier 67 % of corn area (26,296 ac), wide tier 9,880 ac with 24.6 % called silage; HEADLINE local threshold 8.4 % (reference flags 11.5 %, Frederick 15.8 %, matched TPR 0.632); counted clear 14.0 %, reference-calibrated 12.3 %, Europe-adjusted 6.8 %; counted over both tiers 16.9 %. Under the 17:30 bands it would be WEAK (error 0.112, low side) | No: reported as it came out; the procedure is unchanged |
| 17:40 | modelling track | EVALUATION 2 TRAINING RUN starts (design 13:58, primary rule 14:07): fieldvision.model.train at code db35101 (feature code unchanged since bd4cc94; training code only adds the composite family) on every European field-year in READY.txt: AT 2022, 2023, 2024, FR 2022, NL 2025 (8,000 fields), --n-boot 1000 --n-splits 5, other options as at2024_full; out C:\t\fv\data\model\runs\eu_all. One run does the cross-validation (spatial, leave-one-country-out AT/FR/NL, leave-one-year-out) and fits the final model of every family; the final fits are fixed by the design and do not depend on which family is selected. The leave-one-country-out table and the family it selects are logged before any export and before Beltsville is opened | No |
| 17:41 | modelling track | EVALUATION 2 PRIMARY, how the 14:07 rule is applied, written while the run is still training and before any of its results are read: families ordered simplest first (timing rule, timing+moisture rule, 6-feature logistic, LightGBM, composite_v1); ROC-AUC per held-out country (AT, FR, NL), mean over the three; the family with the highest mean is replaced by the simplest family whose mean is below it by less than the best family's spread (max - min over the held-out countries). Script C:\t\fv\data\model\eval2_loco_table.py | Fixes the reading of the rule before results |
| 17:46 | modelling track | WHAT A PASS ON KENT AND CARROLL CAN AND CANNOT SHOW (lead's point, logged before either county is opened): Kent's Census share is 5.5 % and Carroll's 8.9 %, both low. On full Frederick (19.6 %) the step reads 8.4 % (other track's version 7.0 %), and its flagged area separates Queen Anne's from Frederick by about 3.4 points where the Census differs by 18. A step that returned about 7 % for every county would pass both test counties. So a pass shows that the false-alarm level is under control in a low-silage county; it does NOT show that the step tracks the silage share. Seen-data analysis (frederick_low.py): Frederick's early-cut fields fall mostly in the wide tier (implied recall at the local threshold 0.36 on the clear tier, 0.57 on clear + wide, matched European 0.63), so the clear-tier step under-reads where silage is common. Kent and Carroll run once, with the final all-Europe model in the classifier slot and the Queen Anne's threshold recomputed with it by the 17:30 procedure; the 17:30 bands stand. A test of tracking needs a high/low pair in one state, with its bands logged before it is opened | No |
| 17:49 | modelling track | EVALUATION 2 LEAVE-ONE-COUNTRY-OUT TABLE (run eu_all, done 17:49, 8,000 fields; ROC-AUC held out AT / FR / NL, mean, spread): timing rule .654/.694/.638 mean .662 spread .056; timing+moisture rule .746/.772/.665 mean .728 spread .108; logistic .754/.826/.717 mean .765 spread .109; LightGBM .770/.805/.713 mean .763 spread .091; composite_v1 .756/.800/.724 mean .760 spread .076. Highest mean: logistic. FAMILY SELECTED BY THE 17:41 READING: the timing rule (its mean is .103 below the logistic's, less than the logistic's spread of .109). Reported as it came out. Problem with that reading, stated now: the spread across countries is mostly country difficulty shared by all families (every family is worst on NL); paired by country the logistic beats the timing rule in all three (+.100/+.132/+.079) and the timing+moisture rule in all three (+.008/+.053/+.052). A paired reading (mean difference against the spread of the differences) would select the timing+moisture rule (mean +.038, spread .045). Nothing is changed by me: the lead decides which reading stands; no export choice, module choice or Beltsville access before that | No: the selection is on hold |
| 18:00 | modelling track | SECONDARY COUNTY FIGURE (lead's decision), fixed before Kent or Carroll is opened: local_per_tier_share = the local threshold applied per evidence tier on clear + wide: for each tier, the threshold is the 90th percentile of the classifier's scores over that tier's fields in the reference county, the reference flagged share is measured within that tier, the matched recall is that tier's (from rates.json 'local', per country-year on non-abstained European out-of-fold fields), the adjusted shares are combined by the target's tier areas; none if a reference tier has fewer than 50 fields. Shown beside the headline, never the headline. ADDED AFTER SEEING Frederick and Queen Anne's (the early-cut silage in the wide tier); the headline (17:30 procedure) does not change on the Maryland test counties whatever they show. Also shown beside the headline from now on: the unadjusted locally flagged share of the clear tier (clear_silage_share_local_flagged). The headline's matched recall comes from rates.json 'local' when present (per country-year, the corrected recipe), otherwise from the pooled clear set | No |
| 18:01 | modelling track | EVALUATION 2 PRIMARY, CORRECTION BY THE LEAD to the 17:41 reading, made after the 17:49 table was seen and before Beltsville is opened for Evaluation 2. (1) The 14:07 rule: highest mean ROC-AUC under leave-one-country-out, ties to the simpler family, all families shown. Highest mean: 6-feature logistic .765; LightGBM .763 and composite_v1 .760 are within .005 and less simple, so the tie clause resolves to the logistic; the timing rule is .103 lower and lower in every held-out country (AT .654 vs .754, FR .694 vs .826, NL .638 vs .717), not a tie in any plain sense. (2) The 17:41 text defined a tie by the best family's spread across countries (.109); that spread is mostly difficulty all families share (all worst on NL), so it cannot say whether two families differ; applied literally it selects the timing rule (gap .103 < .109). (3) The three readings and what each selects: (a) 17:41 literal -> timing rule; (b) paired differences against their spread -> timing+moisture rule (mean +.038, spread .045); (c) plain tie clause -> 6-feature logistic. PRIMARY FOR EVALUATION 2: the 6-feature logistic (eu2_logistic_v1, run eu_all). (4) The awkward part: Evaluation 1 already showed the Austria-trained logistic ahead on Beltsville (42 of 44 against 38 for the timing rule), so this choice agrees with something already seen. The protection: all five families (timing rule, timing+moisture rule, logistic, LightGBM, composite) and always-grain are scored once in Evaluation 2 and shown side by side, so a reader sees what readings (a) and (b) would have reported | Names the primary; nothing else changed |
| 18:02 | modelling track | EVALUATION 2 RUNS NOW, once: python -m fieldvision.model.evaluate --model C:\t\fv\data\model\runs\eu_all --which logistic (primary, named at 18:01) on the 44 Beltsville field-years (barc_2021-2023 time series and fields, as Evaluation 1), --keep-columns OBJECTID, out C:\t\fv\data\model\eval\beltsville_eval2, access log appended by the command; all families of run eu_all (run_id 0484d7e43267, trained 17:49 on AT 2022-2024, FR 2022, NL 2025 with the 13:58 design) and always-grain scored in the same command and all reported | This is the evaluation |

### Fields evaluated, by year and class

| year | silage | grain | scored | unlabelled (not scored) |
|---|---|---|---|---|
| 2021 | 4 | 14 | 18 | 0 |
| 2022 | 4 | 3 | 7 | 0 |
| 2023 | 4 | 15 | 19 | 0 |
| total | 12 | 32 | 44 | 0 |

Blank labels are 'not labelled', not negatives, and are not scored. One polygon can appear in several years.

## Results: baselines and primary model

`majority` predicts the training majority class for every field; `timing_rule` is the transparent baseline (silage if the sustained NDVI decline comes before a day-of-year threshold fitted on the European training data only). Precision and recall are given for both classes.

| model | n (silage) | silage precision | silage recall | FPR | grain precision | grain recall | bal. acc. | ROC-AUC | PR-AUC (silage) |
|---|---|---|---|---|---|---|---|---|---|
| majority | 44 (12) | 0.273 [0.15, 0.43] | 1.000 [0.74, 1.00] | 1.000 [0.89, 1.00] | n/a | 0.000 [0.00, 0.11] | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 0.273 [0.15, 0.42] |
| always_grain | 44 (12) | n/a | 0.000 [0.00, 0.26] | 0.000 [0.00, 0.11] | 0.727 [0.57, 0.85] | 1.000 [0.89, 1.00] | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 0.273 [0.15, 0.42] |
| timing_rule | 44 (12) | 0.579 [0.33, 0.80] | 0.917 [0.62, 1.00] | 0.250 [0.11, 0.43] | 0.960 [0.80, 1.00] | 0.750 [0.57, 0.89] | 0.833 [0.71, 0.94] | 0.904 [0.78, 0.99] | 0.684 [0.44, 0.98] |
| logistic | 44 (12) | 0.917 [0.62, 1.00] | 0.917 [0.62, 1.00] | 0.031 [0.00, 0.16] | 0.969 [0.84, 1.00] | 0.969 [0.84, 1.00] | 0.943 [0.84, 1.00] | 0.951 [0.83, 1.00] | 0.949 [0.82, 1.00] |

Confusion matrix, `logistic` (rows = declared, columns = predicted):

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 11 | 1 |
| grain | 1 | 31 |

Confusion matrix, `timing_rule`:

| declared \ predicted | silage | grain |
|---|---|---|
| silage | 11 | 1 |
| grain | 8 | 24 |

### Abstention

Fields flagged low evidence still get a call, but the method says it should not be trusted. Confident subset (low-evidence fields removed):

| model | n (silage) | silage precision | silage recall | FPR | grain precision | grain recall | bal. acc. | ROC-AUC | PR-AUC (silage) |
|---|---|---|---|---|---|---|---|---|---|
| majority | 36 (10) | 0.278 [0.14, 0.45] | 1.000 [0.69, 1.00] | 1.000 [0.87, 1.00] | n/a | 0.000 [0.00, 0.13] | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 0.278 [0.14, 0.45] |
| always_grain | 36 (10) | n/a | 0.000 [0.00, 0.31] | 0.000 [0.00, 0.13] | 0.722 [0.55, 0.86] | 1.000 [0.87, 1.00] | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 0.278 [0.14, 0.45] |
| timing_rule | 36 (10) | 0.643 [0.35, 0.87] | 0.900 [0.55, 1.00] | 0.192 [0.07, 0.39] | 0.955 [0.77, 1.00] | 0.808 [0.61, 0.93] | 0.854 [0.72, 0.95] | 0.929 [0.81, 1.00] | 0.767 [0.51, 1.00] |
| logistic | 36 (10) | 1.000 [0.66, 1.00] | 0.900 [0.55, 1.00] | 0.000 [0.00, 0.13] | 0.963 [0.81, 1.00] | 1.000 [0.87, 1.00] | 0.950 [0.83, 1.00] | 0.950 [0.83, 1.00] | 0.943 [0.79, 1.00] |

Coverage 0.818.

8 labelled fields abstained on (all in `abstentions.csv`):

| field_id | label | p_silage | predicted_label | low_evidence_reason | n_usable_dates | max_gap_days |
|---|---|---|---|---|---|---|
| barc-278-2022 | silage | 0.833 | silage | wide_harvest_interval | 27 | 23.000 |
| barc-279-2022 | grain | 0.625 | silage | wide_harvest_interval | 25 | 23.000 |
| barc-24-2023 | grain | 0.381 | grain | wide_harvest_interval | 19 | 28.000 |
| barc-347-2023 | grain | 0.278 | grain | wide_harvest_interval | 13 | 25.000 |
| barc-348-2023 | grain | 0.344 | grain | wide_harvest_interval | 13 | 25.000 |
| barc-382-2023 | grain | 0.445 | grain | wide_harvest_interval | 15 | 25.000 |
| barc-383-2023 | grain | 0.409 | grain | wide_harvest_interval | 16 | 25.000 |
| barc-292-2023 | silage | 0.730 | silage | wide_harvest_interval | 15 | 25.000 |

## Per year

| year | n (silage) | silage precision | silage recall | FPR | grain precision | grain recall | bal. acc. | ROC-AUC | PR-AUC (silage) |
|---|---|---|---|---|---|---|---|---|---|
| 2021 | 18 (4) | 1.000 [0.29, 1.00] | 0.750 [0.19, 0.99] | 0.000 [0.00, 0.23] | 0.933 [0.68, 1.00] | 1.000 [0.77, 1.00] | 0.875 [0.50, 1.00] | 0.929 [0.73, 1.00] | 0.875 [0.46, 1.00] |
| 2022 | 7 (4) | 0.800 [0.28, 0.99] | 1.000 [0.40, 1.00] | 0.333 [0.01, 0.91] | 1.000 [0.16, 1.00] | 0.667 [0.09, 0.99] | 0.833 [0.50, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| 2023 | 19 (4) | 1.000 [0.40, 1.00] | 1.000 [0.40, 1.00] | 0.000 [0.00, 0.22] | 1.000 [0.78, 1.00] | 1.000 [0.78, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |

Per-year counts are tiny; read these as a check for one bad year, not as estimates.

### By imagery source

Each field-year is assigned the collection most of its usable dates came from ('+mixed' if more than one). For Beltsville, 2022 comes from a different processing source (Planetary Computer) than 2021 and 2023; the training report's same-block check (Austria 2022 vs 2023) shows how much that source moves the features.

| collection | n (silage) | silage precision | silage recall | FPR | grain precision | grain recall | bal. acc. | ROC-AUC | PR-AUC (silage) |
|---|---|---|---|---|---|---|---|---|---|
| planetary:sentinel-2-l2a | 7 (4) | 0.800 [0.28, 0.99] | 1.000 [0.40, 1.00] | 0.333 [0.01, 0.91] | 1.000 [0.16, 1.00] | 0.667 [0.09, 0.99] | 0.833 [0.50, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| sentinel-2-c1-l2a | 37 (8) | 1.000 [0.59, 1.00] | 0.875 [0.47, 1.00] | 0.000 [0.00, 0.12] | 0.967 [0.83, 1.00] | 1.000 [0.88, 1.00] | 0.938 [0.80, 1.00] | 0.931 [0.77, 1.00] | 0.917 [0.71, 1.00] |

## Calibration on this set

Brier 0.087, ECE 0.194. Probabilities are calibrated for the 50/50 training balance, so with fewer silage fields here they will read high on average. Per bin (mean predicted -> observed silage share): 0.14->0.07 (n=14), 0.27->0.00 (n=13), 0.44->0.00 (n=5), 0.74->0.89 (n=9), 0.82->1.00 (n=3).

## Transfer evidence from the European data (main evidence)

From the training run: spatial CV, and each country or year held out entirely. This is where the transfer claim is tested with hundreds of fields per split.

| split | model | n | bal. acc. | ROC-AUC | silage recall | grain recall | FPR |
|---|---|---|---|---|---|---|---|
| spatial | majority | 8000 | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 |
| spatial | timing_rule | 8000 | 0.616 | 0.660 | 0.653 | 0.578 | 0.422 |
| spatial | gbm | 8000 | 0.766 | 0.830 | 0.762 | 0.769 | 0.231 |
| loco:AT | majority | 4800 | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 |
| loco:AT | timing_rule | 4800 | 0.624 | 0.654 | 0.584 | 0.664 | 0.336 |
| loco:AT | gbm | 4800 | 0.719 | 0.770 | 0.710 | 0.728 | 0.272 |
| loco:FR | majority | 1600 | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 |
| loco:FR | timing_rule | 1600 | 0.670 | 0.694 | 0.744 | 0.596 | 0.404 |
| loco:FR | gbm | 1600 | 0.702 | 0.805 | 0.534 | 0.871 | 0.129 |
| loco:NL | majority | 1600 | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 |
| loco:NL | timing_rule | 1600 | 0.601 | 0.638 | 0.639 | 0.562 | 0.438 |
| loco:NL | gbm | 1600 | 0.668 | 0.713 | 0.901 | 0.435 | 0.565 |
| loyo:2022 | majority | 2800 | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 |
| loyo:2022 | timing_rule | 2800 | 0.662 | 0.680 | 0.673 | 0.651 | 0.349 |
| loyo:2022 | gbm | 2800 | 0.720 | 0.794 | 0.588 | 0.852 | 0.148 |
| loyo:2023 | majority | 1200 | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 |
| loyo:2023 | timing_rule | 1200 | 0.591 | 0.714 | 0.272 | 0.910 | 0.090 |
| loyo:2023 | gbm | 1200 | 0.749 | 0.821 | 0.868 | 0.630 | 0.370 |
| loyo:2024 | majority | 2400 | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 |
| loyo:2024 | timing_rule | 2400 | 0.575 | 0.671 | 0.837 | 0.313 | 0.687 |
| loyo:2024 | gbm | 2400 | 0.782 | 0.845 | 0.739 | 0.824 | 0.176 |
| loyo:2025 | majority | 1600 | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 |
| loyo:2025 | timing_rule | 1600 | 0.601 | 0.638 | 0.639 | 0.562 | 0.438 |
| loyo:2025 | gbm | 1600 | 0.668 | 0.713 | 0.901 | 0.435 | 0.565 |

## Limits

- Geography: one research farm in Maryland. It says nothing direct about the Corn Belt, the South (earlier harvest calendars) or irrigated double-cropped regions.
- Time: 2021-2023, with imagery from more than one processing source; see the per-year table and the training report's collection-shift check.
- Labels: European labels are farmer declarations; the Dutch ones are made in spring and can differ from the actual use.
- The harvest date is bracketed by the last clear image before and the first after; cloud gaps widen it, and wide brackets are what drive abstention.

## Other models in the run (secondary; not used to choose the model)

| model | n (silage) | silage precision | silage recall | FPR | grain precision | grain recall | bal. acc. | ROC-AUC | PR-AUC (silage) |
|---|---|---|---|---|---|---|---|---|---|
| rule | 44 (12) | 0.611 [0.36, 0.83] | 0.917 [0.62, 1.00] | 0.219 [0.09, 0.40] | 0.962 [0.80, 1.00] | 0.781 [0.60, 0.91] | 0.849 [0.72, 0.94] | 0.918 [0.78, 1.00] | 0.845 [0.64, 1.00] |
| gbm | 44 (12) | 0.667 [0.38, 0.88] | 0.833 [0.52, 0.98] | 0.156 [0.05, 0.33] | 0.931 [0.77, 0.99] | 0.844 [0.67, 0.95] | 0.839 [0.70, 0.95] | 0.901 [0.74, 0.99] | 0.864 [0.68, 0.99] |
| composite | 44 (12) | 0.611 [0.36, 0.83] | 0.917 [0.62, 1.00] | 0.219 [0.09, 0.40] | 0.962 [0.80, 1.00] | 0.781 [0.60, 0.91] | 0.849 [0.72, 0.94] | 0.948 [0.83, 1.00] | 0.935 [0.80, 1.00] |

## Misclassified fields (primary model)

2 fields, full list in `misclassified.csv`. Most confident mistakes:

| field_id | year | error | p | low_evidence | harv_days_since_peak | pre_ndmi | pre_ndvi_frac | post_ndti | q_n_usable_season |
|---|---|---|---|---|---|---|---|---|---|
| barc-327-2021 | 2021 | silage called grain (FN) | 0.196 | False | 67.500 | 0.013 | 0.453 | 0.165 | 15.000 |
| barc-279-2022 | 2022 | grain called silage (FP) | 0.625 | True | 53.500 | 0.435 | 0.912 | 0.224 | 25.000 |

## Key features: training vs this set (medians per declared class)

Descriptive only; nothing is adjusted from it.

| feature | train_silage | eval_silage | eval_silage_nan | train_grain | eval_grain | eval_grain_nan |
|---|---|---|---|---|---|---|
| harv_days_since_peak | 51.500 | 51.750 | 0.000 | 63.500 | 59.000 | 0.000 |
| harv_mid_doy | 252.500 | 250.750 | 0.000 | 265.000 | 262.500 | 0.000 |
| pre_ndvi_frac | 0.788 | 0.894 | 0.000 | 0.539 | 0.463 | 0.000 |
| pre_ndmi | 0.295 | 0.370 | 0.000 | 0.130 | 0.039 | 0.000 |
| sen_ndvi_slope_rel | -0.005 | -0.002 | 0.000 | -0.011 | -0.015 | 0.000 |
| post_ndti | 0.122 | 0.111 | 0.000 | 0.149 | 0.201 | 0.000 |
| peak_ndvi | 0.892 | 0.910 | 0.000 | 0.889 | 0.878 | 0.000 |
| q_n_usable_season | 17.000 | 24.500 | 0.000 | 18.000 | 25.000 | 0.000 |
| q_max_gap_days | 28.000 | 20.000 | 0.000 | 28.000 | 16.000 | 0.000 |
| q_harv_width_days | 10.000 | 12.500 | 0.000 | 12.000 | 7.000 | 0.000 |

## Low-evidence reasons

ok: 36, wide_harvest_interval: 8


Features: 6.57 s per 1000 fields.
