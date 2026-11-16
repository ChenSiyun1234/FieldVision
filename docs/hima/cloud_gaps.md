# Cloud gaps at harvest: what we found and how the agent handles it

Hima Verma, Hack the Harvest 2026. Model estimates from satellite data, not NASS statistics.
Code: `src/fieldvision_agents/gap/`. Scripts: `scripts/hima/scoring/`. Tables: `handoff/results/hima/gap/`.
Every rule below was written down before its first run and scored once. Nothing was tuned on USDA figures
or on the labelled Beltsville fields.

## The problem
Silage is cut over a few weeks in September. When those weeks are cloudy, the model cannot see when a
field was cut, so it abstains. The fields it drops are mostly silage, so the county total reads low.

Wyoming County, NY, 2022: 208 of 820 fields abstained, 202 of them with a known cloud gap, and 110 share
one gap from 10 Sep to 5 Oct. Those fields were green before (NDVI 0.84) and bare after (0.22).

## The timing probability
For a field whose harvest lies somewhere between its last clear standing date a and its first clear bare
date b:

    p(silage) = pi * S / (pi * S + (1 - pi) * G)

- S = share of the county's silage harvest that fell between a and b
- G = share of the county's grain harvest that fell between a and b
- pi = share of judged fields called silage, or the last official share before the season

S and G come from the county's own judged fields: each spreads its harvest evenly over its own bracket.
Wyoming's main gap: S = 0.632, G = 0.067, pi = 0.559, so p = 0.92. Frederick's main gap (18 Aug to
15 Sep): S = 0.576, G = 0.196, pi = 0.180, so p = 0.39.

## Counting rules against USDA (miss in points)

| Rule | Wyoming NY 2022 | Frederick MD 2022 | 9 states 2025, mean size |
|---|---|---|---|
| Abstained fields dropped | -13.1 | -3.0 | 7.3 |
| Gap fields counted by the model's call | -1.8 | -0.4 | 7.9 |
| Timing probability, pi from judged fields | -3.3 | -2.2 | 7.4 |
| Timing probability, pi = last official share | -1.8 | -1.7 | 6.3 |
| Probabilities moved to the prior | 9.3 | -3.4 | 4.6 |
| Last official share alone | 8.1 | 4.7 | 2.5 |

- The timing probability recovers most of the hidden silage in dairy counties and does not overcount in
  grain states, where counting gap fields by the model's call does (Minnesota 3.9 vs 8.7, South Dakota
  1.1 vs 5.8).
- It cannot separate silage from grain where the two harvests overlap in time: in the Pennsylvania 2025
  sample they are 12 days apart, so S and G come out nearly equal and p falls back to pi.
- Both county answers were known when the rules were written, so the county columns are not a clean test.

## What separates right from low: the share of corn the model could not judge

| Place | Not judged | Called | USDA | Miss | If every unjudged field were silage | Miss then |
|---|---|---|---|---|---|---|
| PA 2025 | 26.0% | 13.3% | 32.8% | -19.5 | 35.8% | 3.0 |
| NY 2025 | 24.7% | 38.8% | 53.2% | -14.4 | 53.9% | 0.7 |
| Wyoming NY 2022 | 28.0% | 53.6% | 66.7% | -13.1 | 66.6% | -0.1 |
| CA 2025 | 14.0% | 63.4% | 83.1% | -19.7 | 68.5% | -14.6 |
| Frederick MD 2022 | 7.3% | 16.6% | 19.6% | -3.0 | 22.7% | 3.1 |
| WI 2025 | 11.1% | 22.2% | 21.6% | 0.6 | 30.8% | 9.2 |
| MI 2025 | 12.5% | 12.7% | 14.7% | -2.0 | 23.6% | 8.9 |
| IA 2025 | 13.9% | 6.2% | 1.9% | 4.3 | 19.2% | 17.3 |
| NE 2025 | 13.2% | 0.7% | 2.1% | -1.4 | 13.8% | 11.7 |
| MN 2025 | 9.6% | 6.4% | 4.1% | 2.3 | 15.4% | 11.3 |
| SD 2025 | 15.7% | 4.3% | 5.8% | -1.5 | 19.3% | 13.5 |

- The three places with 20% or more not judged read 13 to 20 points low (15.7 on average); the other
  eight missed by 4.3. Correlation 0.69 over 11 places, 0.87 without California. The 20% line was drawn
  after seeing this table, so the agent uses it as a warning, not as a correction.
- In those three places, counting every unjudged field as silage lands within 3 points of USDA. In grain
  states the same move overshoots by 9 to 17, so it is reported as the top of a range.
- California is a different case: 90% of its unjudged fields are green in November (winter forage).

## Green in November
Rule fixed first: median NDVI of clear looks from 25 Oct to 30 Nov; 0.45 or more counts as green.
- 44 labelled Beltsville field-years: 10 of 12 silage fields are green late, 28 of 32 grain fields are
  not. The flag alone is right on 38 of 44 (the model gets 42).
- Pennsylvania 2025: green late on 33.6% of silage-called acres and 2.8% of grain-called acres.
- No signal in NY, WI, MN, SD, NE or IA. It is a mid-Atlantic cover-crop pattern.

## Vision check (Nemotron Super VL)
The model sees the image on the last clear date before a field's gap and the first after it.
- 44 labelled Beltsville field-years: readable on 28, right on 15. At 10 m true colour it cannot tell
  residue from bare soil, so the agent reports it as illustration, never as evidence.
- Wyoming: agreed with the model on 24 of 32 readable gap fields and 13 of 20 readable clear fields.

## Checked explanations (Nemotron Ultra)
Ultra writes two or three sentences per gap field from one evidence record. Code rejects the text if any
number in it is not in the record. Of 8 written for Wyoming, 5 passed, 2 were rejected for a number not
in the record and 1 came back empty; those 3 used the template.
