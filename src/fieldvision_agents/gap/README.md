# gap: fields whose harvest fell in a cloud gap

Added Sunday 15 Nov, about 1:30 AM. Model estimates, not NASS statistics.

## What it does

1. **Sorts every field** into clear, gap (harvest somewhere between two clear looks) or unseen, names the
   main cloud gaps, and gives the county silage share three ways plus a low and high bound.
2. **`inspect_gaps` tool for the supervisor.** Allowed once, after classify. When the critic does not
   pass, the critic result now carries a hint to call it before rerunning or abstaining. It never changes
   the county estimate.
3. **Vision on gap fields.** Nemotron Super VL sees the last clear image before the gap and the first one
   after it, and says bare soil or residue. Results are cached per field, so a rerun costs nothing.
4. **Checked reasons.** Nemotron Ultra writes two or three sentences per gap field from one evidence
   record. Code rejects the text if any number in it is not in the record, and falls back to a template.

## Run it

```bash
python -m fieldvision_agents.gap.gap 36121 2022 --data <counties dir> --vision 40 --vision-clear 30 --reasons 8
python -m fieldvision_agents.gap.test_gap && python -m fieldvision_agents.gap.test_gap_tool   # 10 tests, offline
```

## What we found (all scored once, nothing tuned on these)

**Wyoming County, NY, 2022 (820 fields).** 208 fields abstained. 202 of them have a known gap, and 110
share one: 10 Sep to 5 Oct. Those were green before (NDVI 0.84) and bare after (0.22).

**Agent run `36121-2022-9345f2`:** critic failed 50.7% against the county's 67.2% to 74.8% history, the
agent called `inspect_gaps` with 6 vision fields, then abstained and named the gap and the range
64.5% to 65.1% in its reason. 70,572 tokens, 177 seconds. Census 2022 says 66.7%.

**Counting rules against USDA, mean miss in points:**

| Rule | 9 states 2025 | Frederick + Wyoming 2022 |
|---|---|---|
| Last official share alone, no satellite | 2.5 | 6.4 |
| Model calls, abstained fields dropped | 7.3 | 8.0 |
| Gap fields counted by the model's call | 7.9 | 1.1 |
| Probabilities moved to the prior | 4.6 | 6.4 |

Counting gap fields helps the two dairy counties (Frederick 19.2% vs 19.6%, Wyoming 64.9% vs 66.7%) and
hurts grain states. Both county answers were known when the rule was written, so this is not a clean
test. The tool reports it as a range beside the estimate, not as the estimate.

**Vision check, 44 labelled Beltsville field-years:** picture readable on 28, right on 15 of those. The
model is right on 42 of the same 44. At 10 m true colour the picture cannot tell residue from bare soil,
so the tool labels it as illustration, not evidence. In Wyoming the picture agreed with the model on 24
of 32 readable gap fields and 13 of 20 readable clear fields.

**Reasons:** 8 written, 5 passed the number check, 2 were rejected for a number not in the record, 1 came
back empty. The 3 fell back to the template.

## Bug found on the way

`vision/vision.py` asks Super VL with `max_tokens=400`. The model uses about 570 tokens reasoning before
it answers, so the reply is empty and every field reads "unclear". This module uses 2,048.

## Limits

- The 2025 state samples are 300 fields each; the two-county result is two counties.
- Chips take about 14 seconds each to download, so a vision check is about 20 seconds per field.
- Landsat rows are used when `timeseries_landsat.parquet` exists for the county (`--landsat`); none of the
  counties in the workspace had one when this was run.

## The interval formula (added 2:45 AM)

For an abstained field whose harvest lies between its last standing date a and its drop date b:

    p(silage) = pi * S / (pi * S + (1 - pi) * G)

- S = share of the county's silage harvest that fell between a and b
- G = share of the county's grain harvest that fell between a and b
- pi = share of judged fields called silage (or the last official share before the season)

S and G come from the county's own judged fields: each one spreads its harvest evenly over its own
bracket (harvest_curves). County share: judged fields keep the model's call, abstained fields with a
bracket take p, unseen fields take pi (interval_share). Code: interval_probability in gap.py.

Worked example, Wyoming NY 2022, the 110 fields hidden from 10 Sep to 5 Oct: S = 0.632, G = 0.067,
pi = 0.559, so p = 0.923. In Frederick's main gap (18 Aug to 15 Sep): S = 0.576, G = 0.196, pi = 0.180,
so p = 0.391.

Scored once, rule fixed before the run. Miss in points against USDA:

| Place | Abstains dropped | Gap fields by call | Interval, pi from fields | Interval, pi = prior | Prior alone |
|---|---|---|---|---|---|
| Wyoming NY 2022 | -13.1 | -1.8 | -3.3 | -1.8 | 8.1 |
| Frederick MD 2022 | -3.0 | -0.4 | -2.2 | -1.7 | 4.7 |
| 9 states 2025, mean size | 7.3 | 7.9 | 7.4 | 6.3 | 2.5 |
| MN 2025 | 2.3 | 8.7 | 3.9 | 2.7 | 0.6 |
| SD 2025 | -1.5 | 5.8 | 1.1 | 0.9 | 2.2 |
| PA 2025 | -19.5 | -8.9 | -18.8 | -13.8 | -2.2 |
| NY 2025 | -14.4 | -16.3 | -15.1 | -15.2 | -9.8 |

What the pattern says:
- The formula is safe. It recovers most of the hidden silage in a dairy county and does not overcount in
  grain states, where counting gap fields by the model's call does (MN, SD).
- It cannot fix Pennsylvania. There the judged fields already under-count silage (pi = 13%), and the
  silage and grain harvests are only 12 days apart in the judged fields, so S and G are nearly equal and
  p falls back to pi.
- New York's state miss is not a cloud problem: the model calls only 10% of its gap fields silage.
- Nothing here beats the last official share alone on the state samples.
