# What we do not claim

**Written before the Wisconsin 2023 test (log 22:58), which failed.** Racine read 6.4 %, inside
the range of USDA's figures (2.4 to 6.4 %). Marathon, the heaviest dairy county tried, read 4.3 %
where USDA has 32.1 to 49.9 %: 27.8 points low, with 48 % of its fields not judged. The cause,
as measured afterwards: one cloudy spell of 25 days lay over the silage harvest, so the fields
seen clearly were the late ones, which are grain, and the "not seen" measure did not flag it. The
labelled fields were also scored a fourth time (log 23:39): a sequence model got 42 of 44, a tie
with the small model. The slide deck shows both; the rest of this page does not yet.

**Also written before the two state tests of 2025 (log 01:43, corrected 01:53; log 01:53).** Both
are random samples of a whole state, read with Sentinel-2 plus Landsat, run once under rules
written at log 00:18; the figure is the share over all corn, double-crop corn included.

- *Wisconsin 2025* (8,199 corn fields in 150 blocks, 96 % of the corn area seen): 16.6 % silage
  (16.62; 13.8 to 19.5) where USDA's 2025 figure is 21.6 %. That is 4.98 points low against a pass
  mark of 5: inside it by 0.02 point. The 16.5 % logged first was the single-crop figure, reported
  by mistake and corrected. We had written beforehand that it would read below 16.6 %; that
  misses by the same 0.02. A warning check written into the same rules went off: in two
  low-silage reference counties (Waukesha and Rock, three blocks) the same count read 15.0 % where
  the 2022 Census has 4.5 %, so by those rules the state figure is reported as unreliable, and the
  error is not one uniform shift.
- *Pennsylvania 2025* (3,931 corn fields in 150 blocks, 98 % seen): 15.2 % (12.9 to 17.5) where
  USDA has 32.8 %. 17.6 points low: a fail. Single-crop corn alone reads 9.5 %. The warning check
  did not go off (4.1 % for 3.8 % in the reference counties), so this is neither cloud nor a
  general shift. The written prediction (below 27.8 %) held.
- Starting from USDA's last figure and updating it with the pictures made both worse: Wisconsin
  19.8 % (itself 1.8 off) to 12.9 %, Pennsylvania 31.4 % (1.4 off) to 18.7 %. The update from the
  land-use figure gave 11.6 % and 13.1 %. All four failed.

**And before Lancaster County, Pennsylvania, 2023 (log 02:13), which passed.** 10,533 corn fields
in 70 of the county's 113 blocks (64 % of its corn), Sentinel-2 plus Landsat, 60 % of the corn
area seen, run once. The satellite alone read 26.6 % (range 18.0 to 58.1). The reference, fixed
before the run (log 01:28), is the 2022 Census, 27.4 %: the 2023 county survey says 44.5 % but
counts only about half the corn acres that the Census, the field boundaries and the cropland map
show. So 0.8 points low, and 0.8 below the range of official figures (27.4 to 54.8); the pass mark
was "within 10 points of that range". Beside it: updating from the Census figure gave 25.1 %, no
gain; updating from the land-use figure gave 9.1 %, far off. Teammates had run the same classifier
on 1,057 clearly seen Lancaster fields and read 35.2 %; the coordinator had seen that figure
before this run (log 01:50) and the procedure was not changed.

**And before Michigan 2025 (log 04:42), which passed.** A fourth clean test, added at log 02:22
with its procedure, pass marks and prediction, run once: 4,835 corn fields in 150 blocks,
Sentinel-2 plus Landsat, 98 % of the corn area seen. 14.4 % (10.8 to 18.0) where USDA has 14.7 %:
0.3 points low. The warning check is clean (3.1 % in two grain counties). Our own prediction was
wrong: we had written that it would read below 11.7 %, expecting the year 2025 to be what made
Maryland, Wisconsin and Pennsylvania read low. Updating from USDA's last figure (14.3 %, itself
0.4 off) gave 9.9 %, worse, as in every other state. Ten of the 150 blocks touch Branch and
Gratiot, the two Michigan counties tested for 2022: another year and an independent draw.

**After the fact, not tests.** Maryland 2025 with the second satellite added sees 99 % of the
corn and reads 6.2 % with double crops (4.7 to 7.7), still below 8.9 % (log 02:47); the reported
figure stays 4.75 %. Three replacements for the timing measurement were tried against the low
readings, with the rule for choosing written first (log 02:18); none closed the gap by the third
the rule asked for, so the module is unchanged (log 02:20). A look inside the Pennsylvania 2025
sample (log 02:16) found Lancaster County's own fields reading 15.7 % there (488 fields) against
26.6 % in the 2023 run, so within that county the two years do differ.

**What the four last tests say together.** Satellite alone, each run once: Michigan 2025 right
(14.4 for 14.7), Lancaster 2023 right (26.6 for 27.4), Wisconsin 2025 on its pass mark (16.6 for
21.6, and marked unreliable by the warning check), Pennsylvania 2025 missed (15.2 for 32.8).
Before them: Branch, Gratiot and Racine right, Marathon missed, Maryland 2025 low. So the results
are mixed by place, and we can say where. By the reading written before Michigan was opened, a
Michigan that reads right says the place, not the year, makes the difference; the Lancaster look
above shows the year is not nothing either. Clouds are not the reason (96 to 99 % of the corn
seen in the four states). What separates the places is still open.

"Log HH:MM" is a row of the evaluation access log (`<data>/eval/ACCESS_LOG.md`), named by its time
on 14 Nov 2026 (Pacific). Everything else is a file in this repository.

## The limits, said plainly

- **State acres are not validated: Maryland 2025 reads low.** Our figure is 22,700 acres of
  silage (range 16,200 to 29,900); USDA reports 40,000. As a share of corn that is 4.75 % against
  8.9 %, 4.2 points low, and the USDA figure lies outside our range. The range covers sampling
  only, not the method's own miss.
- **Why it reads low is not diagnosed.** What we know: on the Maryland counties seen in 2022,
  early-cut silage lands more often among the partly seen fields (Frederick), which the count
  leaves out. In the 2025 sample 37.9 % of the partly seen area is called silage, against 4.75 %
  of the clearly seen. Cloud hiding whole harvest weeks is not the reason in 2025 (0.6 % of the
  corn area). The variant weighted by harvest window, computed in the same run, read 3.8 %.
- **Double-crop corn closes part of the gap, after the fact.** Corn sown after a small grain
  (17,056 acres) is outside the scored set. Scored after the fact with the same model, 61 % of
  it is called silage, which lifts the figure to 33,135 acres, 6.7 % (range 5.3 to 8.2), still
  below 8.9 %. No labelled double-crop field exists anywhere, so those calls are untested. The
  reported figure stays 22,700.
- **The first county method failed.** On the Maryland test whose limits were written down before
  it ran, Kent County read 21.0 % silage where the Census has 5.5 % (fail line 20.5 %), and
  Carroll read 0.0 % where the Census has 8.9 % (weak). Two causes are suspected, neither
  proven: cloud at harvest (in Kent and Queen Anne's more than half of the corn area falls in
  harvest windows where too few fields were clearly seen), and false alarms on early-cut grain
  (inside the early windows the clearly seen fields are called silage at 26 to 55 %).
- **The plain count passed one test, and that test is narrow.** On two Michigan counties it had
  never seen it read 2.1 % and 30.0 % where the Census has 1.8 % and 24.2 %. The rule, written
  first, asked for 10 points between the counties and no miss above 10 points. So across a large
  difference it tells a silage county from a grain county. It is not exact (Gratiot is 5.8
  points high), and among the four Maryland counties, where the differences are smaller, it
  misses by up to 7.4 points and its order does not follow the Census (Kent reads 12.9 % at
  5.5 %, Carroll 3.6 % at 8.9 %).
- **Two ways of counting were run together; which one is "the headline" was argued after the
  results.** The plain count and a variant weighted by harvest window were both fixed as outputs
  before Michigan and Maryland 2025 were opened, and the runs were made with the plain count as
  the headline. The variant looked better on the four Maryland counties, but its result there
  flips with how the harvest date is defined. Both give the same Michigan verdict (pass: 2.1 and
  30.0 %, or 3.6 and 29.8 %) and both read low for Maryland 2025 (4.75 %, or 3.8 %). This pack
  shows the plain count and puts the variant beside it.
- **42 of 44 is a small test, not an accuracy figure.** One site, three seasons, 44 field-years;
  one field is 2.3 points. The set was scored three times, each under a rule written beforehand,
  and is now closed. The small model was named as the main model on European data, by the
  held-out-country table of the training run (small model mean 0.765, big model 0.763), with
  near-ties going to the simpler model. That was after the first run had already shown it
  ahead; the log says so, and every model family is shown side by side for that reason.
- **The small model is not better everywhere.** Inside one country the big model is ahead (0.86
  against 0.78 in Austria 2024). When Austria is the new country three bigger models are still
  ahead (by 0.016, 0.035 and 0.045). The claim is only that the small model loses least when the
  country changes.
- **Dropping dates in training helps the big model, and it is still behind.** It was picked
  after seeing six arms, then passed its one confirmation: 39 of 44 on the labelled fields,
  against 37 without it. The small model has 42.
- **Not every field gets a say.** Fields whose harvest was not seen are not judged, and the
  silage share comes from the clearly seen fields only (62 % of the corn area in Maryland 2025).
- **The Maryland 2025 figure was not blind to its target.** The USDA figure was already known to
  us when the steps were written; no 2025 field had been scored then.

Tried for the totals and not kept: the local cut-off (failed in Kent), harvest-window weighting
(withdrawn), and classifiers that also saw the labelled Maryland fields (they raised the false
alarms on the Eastern Shore). Still running: Pennsylvania and Wisconsin 2025 (**to come**), and
a model that also reads the pixels around the field (no result in this pack).

## Where each number comes from

| In the pack | Number | Source |
|---|---|---|
| Small model on the labelled fields | 42 of 44; one silage field missed, one grain field called silage | Log 18:02, "Evaluation 2 result" |
| Other models, same run | big model 37, timing + moisture rule 36, average of three simple models 36, timing rule 35, always grain 32 | Log 18:02 |
| Big model with dates dropped, scored later | 39 of 44; pass line "more than 37" | Log 18:50 (rule), 19:16 (result) |
| First run, models trained on Austria only | small model 42, big model 30, always grain 32 | Log 14:07 (accuracy 0.682 and 0.727 of 44); `docs/module.md`, module_v1 table |
| Scored three times; how the main model was named | training run, each country held out: small model 0.754 / 0.826 / 0.717, mean 0.765; big model 0.770 / 0.805 / 0.713, mean 0.763 | Log 13:53, 14:07, 17:49, 18:01, 18:50 |
| One field is 2.3 points | | `docs/model.md`, under the in-site table |
| What the small model keys on | 0.103, 0.049, 0.038, 0.030, 0.008, 0.000 (with spreads) | `docs/features.md`, "The six features the small model uses" |
| Training data of the classifier | 8,000 fields: Austria 2022-2024, France 2022, Netherlands 2025 | Log 17:40 |
| New country: accuracy and loss per model | fig2, all 36 values | `docs/foundation-models.md`, "Three countries, each left out in turn" |
| "at most 0.015", "0.02 to 0.19" | | `docs/foundation-models.md`, Conclusion |
| Bigger models ahead on Austria | +0.016, +0.035, +0.045 | Same table (paired differences) |
| Inside Austria 2024 | 0.86 against 0.78 | `docs/features.md`, "What did not help" |
| Dropping dates | 0.764 to 0.780; loss 0.047 to 0.035; small model 0.765 and 0.005; seven arms in fig4 | `docs/foundation-models.md`, augmentation table |
| Clearly seen / partly seen / not judged | 12 days or less; 12 to 20; over 20 or no harvest found | Log 17:02; `docs/module.md`, "The layers" |
| First county method, Kent and Carroll | 21.0 % against 5.5 %, fail line 20.5 %; 0.0 % against 8.9 %, weak | Log 17:30 (limits), 18:11, 18:16 |
| Plain count, four Maryland counties | 4.0 / 12.9 / 3.6 / 13.2 % against Census 1.3 / 5.5 / 8.9 / 19.6 %; worst miss 7.4 | Log 18:33, 18:37 |
| Range for the scored fields | Frederick 11.2 to 19.9 %; Carroll 4.8 to 7.5 % | Log 18:09; log 19:05 |
| Corn in harvest windows with too few clearly seen fields | Queen Anne's 57.2 %, Kent 54.4 % | Log 19:05 |
| Early windows: clearly seen fields called silage | 26 to 55 % | Log 18:49 |
| Harvest-window variant on the four Maryland counties | worst miss 3.8 or 10.3, by date convention | Log 18:54, 19:05 |
| Harvest-window variant, Michigan and Maryland 2025; how the headline was argued | 3.6 / 29.8 %, pass; 3.8 % [2.6, 4.9] | Log 19:05, 19:18 |
| Double-crop corn scored after the fact | 60.9 % of its clearly seen area called silage; 33,135 silage acres of 494,947, 6.7 % [5.3, 8.2]; about 10,430 of them double-crop; no labelled double-crop field | Log 19:21, 19:23 |
| Classifiers that saw the labelled Maryland fields | Queen Anne's 10.2 % against 4.0 % | Log 18:33 |
| Michigan test: rule | 10 points apart, no miss above 10 points | Log 18:25, 18:37 |
| Michigan test: result | Branch 2.1 % (Census 1.8 %), Gratiot 30.0 % (24.2 %), 27.9 points apart: pass | Log 19:11 |
| Maryland 2025, USDA | 410,000 grain, 40,000 silage, 8.9 % | Log, "State targets for 2025" |
| Maryland 2025, ours | 4.75 % [3.4, 6.3]; silage 22,700 [16,200-29,900]; grain 455,200; 4.2 points low | Log 19:14 |
| Maryland 2025, beside the figure | 150 of 838 blocks; 3,611 corn fields scored; clearly seen 62 % of area; 17,056 double-crop acres, 8.0 % if all silage; partly seen called silage 37.9 %; 0.6 % in windows with too few clear fields; corn acres 477,891 (boundaries) and 450,000 (USDA) | Log 18:25, 19:11, 19:14 |
| Miss on single Maryland counties | up to about 7 points either way | Log 18:25, 19:14 |
| Silage cut early lands among the partly seen fields | | `docs/module.md`, "Why Frederick reads low" (seen data) |

One difference between two runs, stated so nobody trips on it: the training run of the final
models reports the big model at 0.770 / 0.805 / 0.713 with each country held out (mean 0.763, log
17:49); the comparison run in `docs/foundation-models.md` reports 0.770 / 0.810 / 0.713 (mean
0.764). Figures 2 and 4 use the comparison run, because only it has the in-country values and the
date-dropping arms. Anything said about choosing the main model quotes the training run, because
that table made the logged choice. The small model's values are the same in both.
