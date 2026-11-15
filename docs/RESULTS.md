# Results at a glance (14 Nov, tuning stopped at 19:30)

Every number below is from a run logged in the evaluation access log before it was opened; the time
is the log row. "Seen before" says whether the place had already been looked at when the figure was
produced. Method and reasoning: `docs/module.md`; commands and models: `docs/model.md`.

## Field calls (Beltsville, 44 labelled field-years, Maryland)

| what | compared with | logged | seen before | result |
|---|---|---|---|---|
| Evaluation 1, Austria-trained models | labels, once | 13:53 | no | logistic 42 of 44 (secondary), timing rule 38, LightGBM 30 (primary), always grain 32 |
| Evaluation 2, all-Europe models (AT 2022-24, FR 2022, NL 2025) | labels, once | 18:02 | yes (Eval 1) | **logistic `eu2_logistic_v1` 42 of 44 (primary)**, LightGBM 37, composite 36, timing+moisture rule 36, timing rule 35, always grain 32 |
| Third scoring, date-thinned LightGBM `eu2_gbm_thin_v1` | labels, once; expectation > 37 | 19:15-19:16 | yes | 39 of 44, passes; the set is now closed |

The Evaluation 2 primary was chosen by leave-one-country-out ROC-AUC (logistic .765, LightGBM .763,
composite .760); a first reading of the tie rule (17:41) would have picked the timing rule and was
corrected before Beltsville was opened (18:01).

## County and state shares of corn area harvested for silage

| place | compared with | logged | seen before | headline | beside it |
|---|---|---|---|---|---|
| Queen Anne's MD 2022 | Census 1.3 % | 16:40 (module_v1) | no | 22.4 %: **fail** (first check) | module_v2 counted clear 4.0 %, window 2.0 % (57 % of area blind) |
| Frederick MD 2022 | Census 19.6 %; scored-set range 11.2-19.9 % | 17:37, 18:09 | partial snapshot | eu2 local threshold 18.4 % | counted clear 13.2 %, window 15.1 % |
| Kent MD 2022 | Census 5.5 % | 18:11 | no | local threshold 21.0 %: **fail** | counted clear 12.9 %, window 6.2 % (54 % blind) |
| Carroll MD 2022 | Census 8.9 %; scored-set range 4.8-7.5 % | 18:16 | no | local threshold 0 %: **weak** | counted clear 3.6 %, window 5.1 % |
| Branch MI 2022 | Census 1.8 % | 19:11 | no | counted clear 2.1 %: **pass** | window 3.6 % |
| Gratiot MI 2022 | Census 24.2 % | 19:11 | no | counted clear 30.0 %: **pass** (gap 28 points) | window 29.8 % |
| Maryland 2025, 150 random read blocks | NASS 2025 8.9 % (40,000 silage acres) | 19:14 | no | counted clear 4.75 % [3.4, 6.3] = 22,700 acres: **reads 4.2 points low** | window 3.8 %; with double-crop corn scored (after the fact, 19:23) 6.7 % [5.3, 8.2] = 33,100 acres |

The first county step (local threshold against a nearly all-grain reference county) failed on
Kent and Carroll and is kept only as a record. Since 19:05 the headline is the counted clear share
(share of clearly observed corn area the classifier calls silage). The window-weighted share counts
each two-week harvest window by its own clearly seen fields and counts unseen windows as no silage.

**Why the Eastern Shore read high in 2022:** grain cut early while still green, on corn, wheat and
soybean ground (crop history from boundaries: soybeans on 97 % of Kent's early silage calls, hay or
alfalfa on 6 %), in a year when more than half of the corn there was cut in one cloudy fortnight. The
imagery cannot tell it from silage. **Why Maryland 2025 reads low:** not diagnosed. The partly seen
fields are called silage far more often (38 % against 4.75 %), and corn after a small grain is
outside the scored set; scoring it lifts the figure to 6.7 %.

## Open leads, in the order we would try them

1. **Harvest detector taking a later drop.** Example: Frederick field `csb-241825014024676`, cut on
   day 240 with regrowth 18 days later; the detector needs two low looks within 25 days and so took
   the October fall. Counted on seen data it is only 0.2-2.6 % of clear-tier grain calls
   (`<data>/model/diag/earlier_fall.csv`), so a fix moves little. On disk: all county time series.
2. **Double-crop corn.** 22,700 single-crop silage acres plus 17,056 acres of small-grain-then-corn
   would match NASS's 40,000 only if all of it were silage; the classifier calls 61 % of it silage
   (triticale then corn 69 %), giving 33,100 acres. The rule to count it by class is logged (19:21)
   and coded (`state_estimate --double-crop-class-acres`). On disk: the 2025 sample's
   `timeseries_dblcrop.parquet`, per-class results in `<data>/model/diag/dblcrop_2025_*`. No
   labelled double-crop field exists.
3. **Thinned LightGBM for the wide tier.** On the partly seen fields it calls less silage where
   silage is rare (Queen Anne's 10 % against the logistic's 18 %, Branch 5 % against 9 %) and the same
   in dairy counties. On disk: run `<data>/model/runs/eu_all_gbm_thin`.
4. **Radar for blind weeks.** In a replay on labelled fields, a radar harvest date recovers about
   half the recall lost to a 3-5 week optical gap (France 0.33 → 0.54, full 0.65) but adds 3-6
   points of false silage when residue is read from stale optics (`<data>/model/diag/radar_replay.csv`).
   A Kent pilot is built by the extraction work.
5. **Land-use prior, county level only.** The share of hay and alfalfa in the cropland around the
   fields orders the four Maryland counties as the Census does (0.20 / 0.12 / 0.024 / 0.013) but does
   not separate fields within a county. On disk: `landscape.parquet` in the four county folders and
   the 2025 sample folder; a national hold-out test is with the county work.

Not opened and not run: the Pennsylvania and Wisconsin 2025 samples and the 2023 counties. Rules for
them were written in the log at 19:21 (double crops) and 19:33 (headline switch at 20 % blind area);
whoever continues may keep or replace them, since nothing has been scored.

Not finished: `docs/module.md` and `docs/model.md` hold all results above, but the earlier-fall count,
the thinned model on the wide tier and the radar replay are only in the access log and this file.
