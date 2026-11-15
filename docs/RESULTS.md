# Results at a glance (14 Nov, tuning stopped at 19:30)

Every number below is from a run logged in the evaluation access log before it was opened; the time
is the log row. "Seen before" says whether the place had already been looked at when the figure was
produced. Method and reasoning: `docs/module.md`; commands and models: `docs/model.md`.

## For the deck: every county and state figure

"Used in tuning" = the place's result was seen before the method that produced the figure was fixed,
so it is not a clean test. Headline = what the log named as the headline when the place was opened.

| place | year | headline | comparison | used in tuning | logged |
|---|---|---|---|---|---|
| Queen Anne's MD | 2022 | 22.4 % (module_v1 counted share) | Census 1.3 % | no (first check: failed) | 16:40 |
| Queen Anne's MD | 2022 | 4.0 % counted clear / 2.0 % window | Census 1.3 % | yes | 18:07, 18:54 |
| Frederick MD | 2022 | 13.2 % counted clear / 15.1 % window | Census 19.6 %; scored set 11.2-19.9 % | yes | 18:09 |
| Kent MD | 2022 | 21.0 % local threshold (failed) | Census 5.5 % | no for the local threshold; yes for counted clear 12.9 % and window 6.2 % | 18:11 |
| Carroll MD | 2022 | 0 % local threshold (weak) | Census 8.9 %; scored set 4.8-7.5 % | no for the local threshold; yes for counted clear 3.6 % and window 5.1 % | 18:16 |
| Branch MI | 2022 | 2.1 % counted clear (window 3.6 %) | Census 1.8 % | no: pass | 19:11 |
| Gratiot MI | 2022 | 30.0 % counted clear (window 29.8 %) | Census 24.2 % | no: pass | 19:11 |
| Maryland (150 random blocks) | 2025 | 4.75 % [3.4, 6.3] counted clear (window 3.8 %) | NASS 8.9 % | no: reads low | 19:14 |
| Maryland, double crops added | 2025 | 6.7 % [5.3, 8.2] | NASS 8.9 % | after the fact | 19:23 |
| Land-use prior, Maryland | 2025 | 10.6 % | NASS 8.9 % | no (county work) | 19:48 |
| Racine WI | 2023 | 6.4 % counted clear (window 6.3 %) | survey 2023 2.4 %; band 2.4-6.4 % | no: inside the band | 22:58 |
| Marathon WI (drawn blocks) | 2023 | 4.3 % counted clear (window 10.0 %) | survey 2023 49.9 %; band 32.1-49.9 % | no: **fail**, 28 points low; 48 % of fields abstained (harvest bracket over 20 days) | 22:58 |
| Wisconsin (150 random blocks), Sentinel-2 + Landsat | 2025 | 16.6 % [13.8, 19.5] all corn incl. double crops (single-crop 16.5 %); shift guard flagged; prior-updated 12.9 % (last official 19.8 %), 11.6 % (land-use) | NASS 21.6 % | no: **pass by 0.02 point** (4.98 low; first reported as a fail, corrected 01:53) | 01:43, 01:53 |
| Pennsylvania (150 random blocks), Sentinel-2 + Landsat | 2025 | 15.2 % [12.9, 17.5] all corn incl. double crops (single-crop 9.5 %); guard clean; prior-updated 18.7 % (last official 31.4 %), 13.1 % (land-use) | NASS 32.8 % | no: **fail** (17.6 low), as predicted before opening | 01:53 |
| Lancaster PA (70 of 113 blocks), Sentinel-2 + Landsat | 2023 | 26.6 % counted clear, seen 60 %, range 18.0-58.1 %; prior-updated 25.1 % (Census 27.4 %), 9.1 % (land-use) | Census 27.4 %; band 27.4-54.8 % | no: **pass** (0.8 low); teammates' independent run 35.2 % | 02:13 |
| Michigan (150 random blocks), Sentinel-2 + Landsat | 2025 | 14.4 % [10.8, 18.0] all corn incl. double crops; seen 98 %; guard clean; prior-updated 9.9 % (last official 14.3 %), 8.7 % (land-use) | NASS 14.7 % | no: **pass** (0.3 low); the prediction that it would read low failed | 04:42 |
| New York (150 random blocks), Sentinel-2 + Landsat | 2025 | 48.7 % [43.3, 54.2] all corn incl. double crops; seen 88 %; guard clean; prior-updated 46.5 % (last official 45.0 %), 36.5 % (land-use) | NASS 53.2 % | no: **pass** (4.5 low); the prediction that it would read below 40 % failed | 04:55 |

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
| Land-use prior, 2025 states (county work, logged 19:48, no refit) | NASS 2025, shown once | 19:48 | no | Maryland 10.6 % against 8.9 %; 41 states 3.7 points acre-weighted error (15 large corn states 2.6); national 7.2 % against 6.4 % | Pennsylvania 23.2 vs 32.8, Wisconsin 17.2 vs 21.6, Michigan 11.4 vs 14.7; large misses in irrigated western dairy states and New York (California 9.9 vs 83); `docs/landuse-prior.md` |

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

## After 22:50 (work resumed at the lead's request)

**Wisconsin 2023.** Racine read 6.4 % (band 2.4-6.4 %, inside); Marathon read 4.3 % against the 2023
survey's 49.9 % (band 32.1-49.9 %): fail. Why: 39 % of Marathon's fields have a harvest bracket over
20 days (last standing look at median 20 September, next clear look 15 October); clear looks per field
were 1.1 in September against 4.0 in Racine. The silage harvest fell in one cloudy spell; the clear
tier is mostly the late grain harvest, which the classifier calls grain. No Sentinel-1 scene covers
Marathon between 18 September and 12 October 2023, so radar cannot fill the gap; Landsat has scenes on
1 and 9 October and is being extracted.

**Seen and unseen corn (logged 23:17).** A two-week window of the harvest-bracket midpoint counts as
seen when clear-tier fields hold half of its corn area. Seen share / lower / upper bound: Queen
Anne's 14 % / 1.2 / 87.0, Kent 9 % / 0.0 / 91.3, Marathon 29 % / 0.8 / 72.0, Carroll 83 % / 2.3 / 19.7,
Frederick 71 % / 8.2 / 37.7, Branch 90 % / 1.4 / 11.7, Gratiot 74 % / 25.1 / 51.7, Racine 80 % / 2.8 /
22.6, Maryland 2025 67 % / 3.8 / 37.0. Where less than half is seen the headline is the range.

**Safety net beside the headline.** Mean absolute error on the eight seen counties, satellite /
land-use prior / mean of the two: 8.6 / 8.2 / 6.8 points against the official figure, 4.6 / 5.6 / 3.9
against the official band. The prior's fit includes these counties, so its numbers here are
in-sample. Maryland 2025: 4.75 and 10.6 average 7.7 against NASS 8.9.

**Two-ended check (report only).** Wide and abstained fields scored with the harvest at either end
of their bracket, accepted where both agree: mean absolute error against the bands 3.9 points
against 5.7 for the counted clear share (Marathon 17.1 %, Carroll and Frederick inside their bands,
Branch worse at 7.9 %).

**Response time** (measured by the lead at 23:30 on the laptop with other jobs running):
`Toolbox.classify('26057', 2022, 'module_v2')` 46.7 s for Gratiot's 3,463 fields, about 14 ms per
field; `aggregate` another 56.9 s, mostly re-classifying the reference county, which is now cached.

## Satellite update of the last official share: the first clearly positive county result

Start from the last official silage share published before the test year and let the satellite
update it field by field where it sees the harvest (logged 00:11): seen fields take
sigmoid(logit p + logit prior - logit 0.5), unseen fields keep the prior, county share = area mean.

| county | test year | last official share (prior) | prior updated by the satellite | official | moved |
|---|---|---|---|---|---|
| Queen Anne's MD | 2022 | 2.2 (2017 Census) | 2.1 | 1.3 | towards |
| Carroll MD | 2022 | 11.4 (2017 Census) | 5.3 | 8.9 (band 4.8-7.5) | into the band, away from the point |
| Frederick MD | 2022 | 24.3 (2017 Census) | 16.6 | 19.6 | towards |
| Kent MD | 2022 | 5.8 (2017 Census) | 5.5 (3.1 with Landsat) | 5.5 | towards (away with Landsat) |
| Branch MI | 2022 | 3.6 (2017 Census) | 1.9 | 1.8 | towards |
| Gratiot MI | 2022 | 19.8 (2017 Census) | 21.8 | 24.2 | towards |
| Racine WI | 2023 | 4.2 (2022 survey) | 2.6 | 2.4 | towards |
| Marathon WI | 2023 | 44.9 (2022 survey) | 39.4 (33.6 with Landsat) | 49.9 | away (both inside the band 32.1-49.9) |
| Maryland 2025 | 2025 | 9.5 (2024 survey) | 6.8 | 8.9 | away |

Mean absolute error on the eight counties: prior alone 2.7 points, updated 2.6 (0.4 against the
official bands), satellite alone 9.7. Caveats: these counties were seen while tuning; the last
official share is five years old for six of them and one year old for two; Marathon and Maryland
2025 move away, both places where the satellite reads low. The same update with the land-use
prior instead (state left out) helps where that prior is right and wrecks Gratiot (30.0 to 10.5),
so it is shown beside, never as the headline.

**Landsat** (development places): Marathon 4.3 % to 13.8 % (official 49.9), Kent 12.9 % to 7.3 %
(Census 5.5); seen share 29 to 83 % and 9 to 94 %.

**Sequence model on Beltsville** (fourth scoring, logged 23:03 / 23:37): 42 of 44, 11 of 12 silage,
passes its mark; on the seen counties it reads no better than the logistic and not Marathon.

**Wisconsin and Pennsylvania 2025 (locked procedure, each run once).** With Landsat the samples are
96 % and 98 % seen. The headline (share of all corn, including the corn sown after a small grain) is
16.6 % [13.8, 19.5] in Wisconsin against NASS 21.6 %, 4.98 points low, inside its 5-point mark by
0.02 point (first reported as a fail on the single-crop figure, corrected in the log at 01:53; the
written prediction "below 16.6 %" missed by the same 0.02), and 15.2 % [12.9, 17.5] in Pennsylvania
against 32.8 %, 17.6 points low, a fail as predicted. The single-crop share is 16.5 % and 9.5 %:
the classifier calls far too little of the dairy states' corn silage even when it sees all of it.
Wisconsin's shift guard fired (its two lowest-silage reference counties read 15.0 % against a 4.5 %
Census share); Pennsylvania's did not. The prior updates are worse than their priors in both states
(Wisconsin 12.9 % from 19.8 %, Pennsylvania 18.7 % from 31.4 %): where the satellite reads low,
letting it update a good prior pulls the prior down.

**Lancaster 2023.** An independent run by the teammates in their workspace, with our classifier on
a smaller set of fields (1,057 clearly seen, 35.7 % of acres not judged), read 35.2 % against the
2022 Census 27.4 %; it was seen by the lead at 01:50, before our own run, and is recorded in
the log. Our run (02:11, procedure logged at 00:18, Landsat added) read 26.6 % on 6,404 clearly seen fields, 0.8 below the Census, and passes its mark; seen share 60 %, range 18.0-58.1 %.

**Michigan 2025 (clean test, 04:40).** 14.4 % [10.8, 18.0] against NASS 14.7 %: a pass, 0.3 points
low, with 98 % of the corn seen. The prediction logged before opening (below 11.7 %, meaning the year
2025 is the difference) failed; by the reading written in advance, the place is the difference: the
module reads Michigan right in 2022 (Branch, Gratiot) and in 2025, Lancaster right in 2023, and
Wisconsin, Pennsylvania and Maryland low in 2025.

**Tally of the locked procedure, satellite alone:** Wisconsin 2025 passes by 0.02 point, Pennsylvania 2025 fails (17.6 low), Lancaster 2023 passes (0.8 low), Michigan 2025 passes (0.3 low), New York 2025 passes (4.5 low): four of five clean places pass, and the highest-silage state reads right, so the under-read is not 'dairy' as such; Pennsylvania 2025 stands out, with Maryland 2025 and Marathon among the development places. The prior updates failed their marks in all three places.

**Why does Lancaster read right in 2023 and Pennsylvania low in 2025? Not the timing feature.**
Lancaster's own fields in the Pennsylvania 2025 sample read 15.7 % (block-resampling range 11.1-20.4 %,
11 blocks) against 26.6 % in 2023, so the gap is more than a small sample; Frederick's fields in the
Maryland 2025 sample read 6.1 % (1.3-13.0 %, 8 blocks) against 13.2 % in 2022, which a small sample
could explain. The pictures show no level shift between the two Lancaster years (NDVI before the cut
0.562 and 0.561); the 2025 green peak came 16 days earlier. Three logistics without the
peak-anchored timing were tested under a rule logged before fitting (02:18): all equal V0 in Europe
(0.765-0.766 held out by country), none closes the year-to-year gap by a third (18.4 points for V0,
15.1-15.2 for the others; 12.3 needed). By the rule the module stays as it is, and the earlier peak is
not the cause. With the Maryland 2025 Landsat added afterwards (Frederick's 2025 fields 4.8 %),
the gaps are 19.7 for V0 and 16.9-17.1 for the others: the decision does not change. Maryland 2025
itself with Landsat, after the fact: 99 % seen, single-crop 4.2 % [2.9, 5.8], with double crops
6.2 % [4.7, 7.7], against NASS 8.9 %: still low, now with nothing hidden by clouds.

**November-green variant (teammate's lead, report only, 04:52).** Counting abstained fields that are
green again in November as silage moves the development places by +0.3 to +3.9 points (Lancaster
+13.5, to 40.1 % against a Census 27.4 %), helps Carroll, Frederick, Marathon and the 2025 states a
little (Pennsylvania +1.8 of 23.3 missing), and adds 1-2 points to the low-silage counties. It touches
none of the 44 Beltsville field-years. Left as a documented lead; New York was run without it.

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
   not separate fields within a county. National hold-out by the county work (silage share = a + b x
   forage share of cropland, 1,410 counties, Census 2022): leave one state out, county error 5.9
   points and state error 4.2 (national-share baseline 9.0 and 7.7); 2023 survey counties fitted on
   2022, county 5.8 and state 3.6. Its 2025 prediction uses the 2019-2023 mean forage share (logged
   19:48 by the county work). How to report it: beside the satellite figure, labelled as a land-use
   prior (what a usual year looks like, nothing about this season or any field), never replacing it;
   no combined figure, because the satellite figure has only two state-level checks (pseudo-state
   2022 within 0.1 point, Maryland 2025 4.2 low) and that is too little to set weights in advance.
   On disk: `landscape.parquet` in the four county folders and the 2025 sample folder;
   `<data>/us/landuse/holdout_results.csv`.

Not opened and not run: the Pennsylvania and Wisconsin 2025 samples and the 2023 counties. Rules for
them were written in the log at 19:21 (double crops) and 19:33 (headline switch at 20 % blind area);
whoever continues may keep or replace them, since nothing has been scored.

Not finished: `docs/module.md` and `docs/model.md` hold all results above, but the earlier-fall count,
the thinned model on the wide tier and the radar replay are only in the access log and this file.
