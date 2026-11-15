# What we do not claim

"Log HH:MM" is a row of the evaluation access log (`<data>/eval/ACCESS_LOG.md`), named by its time
on 14 Nov 2026 (Pacific). Everything else is a file in this repository.

## The limits, said plainly

- **County and state acres are not validated.** The first county method failed the Maryland test
  whose limits were written down before it ran: Kent County read 21.0 % silage where the Census
  has 5.5 % (fail line 20.5 %), and Carroll read 0.0 % where the Census has 8.9 % (weak).
- **The second method is not confirmed.** Counting silage only where the harvest was seen was
  found on the same four Maryland counties after the failure. Its worst miss there is 3.8 points,
  but that is not a test, and only two of the four counties (Queen Anne's, Kent) have much corn
  in harvest windows without a clear view. Its clean test is Michigan 2022 (Gratiot, Branch):
  **to come**.
- **Cloud is part of the cause, not all of it.** In Queen Anne's and Kent, 57 % and 54 % of the
  corn area falls in two-week harvest windows where too few fields were clearly seen. But inside
  the early windows the clearly seen fields are themselves called silage at 26 to 55 %, so part
  of the error is in the field calls.
- **42 of 44 is a small test, not an accuracy figure.** One site, three seasons, 44 field-years;
  one field is 2.3 points. The set was scored twice, by a plan written before the first run. The
  small model was named as the main model on European data, but after the first run had already
  shown it ahead; the log says so, and every model family is shown side by side for that reason.
- **The small model is not better everywhere.** Inside one country the big model is ahead (0.86
  against 0.78 in Austria 2024). When Austria is the new country three bigger models are still
  ahead (by 0.016, 0.035 and 0.045). The claim is only that the small model loses least when the
  country changes.
- **Dropping dates in training is not proven.** It was picked after seeing six arms. Its one
  confirmation on the labelled US fields is **to come**. Even with it, the big model is level
  with the small one in two of three countries.
- **Not all corn is in the total.** Corn planted after a small grain is outside the scored field
  set (17,056 acres in the Maryland 2025 frame) and is reported beside the total. Fields whose
  harvest was not seen are not judged.
- **The Maryland 2025 figure is not blind to its target.** The USDA figure (40,000 silage,
  410,000 grain) was already known to us when the steps were written; no 2025 field had been
  scored then. Our figure: **to come**. Pennsylvania and Wisconsin: **to come**.

What is being tried: the second method with its Michigan test; harvest windows formed by region
for the Maryland 2025 sample; the big model trained with dates dropped as a possible second
opinion; a model that also reads the pixels around the field (no result in this pack).

## Where each number comes from

| In the pack | Number | Source |
|---|---|---|
| Small model on the labelled fields | 42 of 44; one silage field missed, one grain field called silage | Log 18:02, "Evaluation 2 result" |
| Other models, same run | big model 37, timing + moisture rule 36, average of three simple models 36, timing rule 35, always grain 32 | Log 18:02 |
| First run, models trained on Austria only | small model 42, big model 30, always grain 32 | Log 14:07 (accuracy 0.682 and 0.727 of 44); `docs/module.md`, module_v1 table |
| Scored twice by plan; how the main model was named | | Log 13:53, 14:07, 18:01 |
| One field is 2.3 points | | `docs/model.md`, under the in-site table |
| What the small model keys on | 0.103, 0.049, 0.038, 0.030, 0.008, 0.000 (with spreads) | `docs/features.md`, "The six features the small model uses" |
| Training data of the classifier | 8,000 fields: Austria 2022-2024, France 2022, Netherlands 2025 | Log 17:40 |
| New country: accuracy and loss per model | fig2, all 36 values | `docs/foundation-models.md`, "Three countries, each left out in turn" |
| "at most 0.015", "0.02 to 0.19" | | `docs/foundation-models.md`, Conclusion |
| Bigger models ahead on Austria | +0.016, +0.035, +0.045 | Same table (paired differences) |
| Inside Austria 2024 | 0.86 against 0.78 | `docs/features.md`, "What did not help" |
| Dropping dates | 0.764 to 0.780; loss 0.047 to 0.035; small model 0.765 and 0.005; seven arms in fig4 | `docs/foundation-models.md`, augmentation table |
| The one confirmation and its pass line | more than 37 of 44 | Log 18:50 |
| Clearly seen / partly seen / not judged | 12 days or less; 12 to 20; over 20 or no harvest found | Log 17:02; `docs/module.md`, "The layers" |
| First county method, Kent and Carroll | 21.0 % against 5.5 %, fail line 20.5 %; 0.0 % against 8.9 %, weak | Log 17:30 (limits), 18:11, 18:16 |
| Second method, four seen counties | 2.0 / 6.2 / 5.1 / 15.1 % against Census 1.3 / 5.5 / 8.9 / 19.6 %; worst miss 3.8 | Log 18:54 |
| Frederick, range for the scored fields | 11.2 to 19.9 % | Log 18:09 |
| Corn not seen at harvest | Queen Anne's 57 %, Kent 54 %, Carroll and Frederick under 1 % | Log 18:54 |
| Early windows called silage too often | 26 to 55 % | Log 18:49 |
| Michigan test counties and pass rule | Census 24.2 % and 1.8 % | Log 18:25, 18:37, 18:54 |
| Maryland 2025, USDA | 410,000 grain, 40,000 silage, 8.9 % | Log, "State targets for 2025" |
| Maryland 2025 sample | 150 of 838 blocks; 3,929 fields drawn; 17,056 double-crop acres | Log 18:25; `docs/us-county-fields.md`, "2025 statewide samples" |

One difference between two runs, stated so nobody trips on it: the training run of the final
models reports the big model at 0.770 / 0.805 / 0.713 with each country held out (mean 0.763, log
17:49); the comparison run in `docs/foundation-models.md` reports 0.770 / 0.810 / 0.713 (mean
0.764). Figures 2 and 4 use the comparison run, because only it has the in-country values and the
date-dropping arms. The small model's values are the same in both.
