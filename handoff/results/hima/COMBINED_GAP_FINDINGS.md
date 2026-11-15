# Where the silage gap comes from: Hima's tables and ours together

Model estimates, not NASS statistics. Sources: `gap/separator_by_place.csv`, `gap/interval_rule_vs_usda.csv`,
`gap/late_green_by_place.csv` and `watch/` (Hima, hima_build on Keeboh, 15 Nov), plus Landsat and Lancaster
figures from `docs/pitch/pitch_numbers.py` and `handoff/results/agents/` (us). Silage share = silage / (silage + grain)
acres harvested. "Not judged" = the model abstained or found no harvest.

## 1. Hima's separator: the unjudged share

On his 11 places, the share of corn acres not judged tracks the size of the miss (r = 0.70). The three places with
20% or more not judged (PA 2025, NY 2025, Wyoming NY 2022) miss by 13 to 20 points, and counting every unjudged
field as silage lands them within 3 points of USDA. In grain states the same move overshoots by 9 to 17 points.
The 20% line was drawn after seeing the table.

## 2. Our places show it is two gaps, not one

| Place | Not judged | Ours | USDA | Miss | What it shows |
|---|---|---|---|---|---|
| Marathon WI 2023, Sentinel-2 only | ~71% | 4.3 | 49.9 | -45.6 | cloud gap, dairy county: reads low |
| Marathon WI 2023, with Landsat | ~17% | 13.8 | 49.9 | -36.1 | **seen, still low**: misread, not unseen |
| Kent MD 2022, Sentinel-2 only | ~91% | 12.9 | 5.5 | +7.4 | cloud gap, grain county: reads **high** |
| Kent MD 2022, with Landsat | ~6% | 7.3 | 5.5 | +1.8 | seeing the gap fixes it |
| Lancaster PA 2023, Sentinel-2 only (1,057 clear fields) | 35.7% | 35.2 | 27.4 (Census 2022) | +7.8 | clear fields lean silage: reads **high** |
| Lancaster PA 2023, both satellites (team, 6,404 clear fields) | — | 26.6 | 27.4 | -0.8 | seeing most fields fixes it |

Adding our four "not judged" points to Hima's eleven drops the correlation from 0.70 to 0.40. The unjudged share
predicts how far off a place can be, not which way: Kent and Lancaster read high with large gaps, and Marathon still
reads 36 points low once Landsat has seen 83% of its corn.

## 3. So the gap has three parts

1. **Unseen harvest (cloud).** The cut falls in a gap between clear images, the field is left unjudged, and in dairy
   country those fields are mostly silage (PA 2025, NY 2025, Wyoming). Fix: a second satellite (Landsat: Kent 9% to 94%
   seen, Lancaster right with both), and the bounded range up to "all unjudged are silage", which Hima's agent rule now
   reports at 20% or more unjudged.
2. **Seen but misread (northern late cut).** Northern silage is cut after the crop yellows, so its curve looks like
   grain to a model trained in Europe. Marathon stays 36 points low with 83% seen. Hima's pace check points the same way:
   on Sep 28 USDA has 66% of Wisconsin silage cut but only 32% of our silage fields show the drop, while grain fields
   "drop" early because browning reads as a drop. Fix: labelled northern fields; no setting change closes this.
3. **Which fields are clearly seen.** When only part of a county is seen, the seen part need not look like the rest
   (Lancaster S2-only reads high on 1,057 fields, right on 6,404). Fix: report the clear share beside the headline and
   prefer runs with both satellites.

## 4. The candidate missing feature: green in November

Median NDVI Oct 25 to Nov 30 at 0.45 or more (rule fixed first). On Beltsville it gets 38 of 44 on its own (10 of 12
silage, 28 of 32 grain); the model gets 42. In PA 33.6% of silage-called acres are green late against 2.8% of
grain-called. It shows nothing in NY, WI, MN, SD, NE or IA, so it cannot close the northern gap (part 2), and in
California 90% of unjudged fields are green from winter forage after corn. Worth adding for the mid-Atlantic only,
after the hackathon and tested on held-out places.

## 5. What not to claim

- On the 2025 state samples, USDA's last official share alone (mean miss 2.5 points) beats every satellite variant
  (7.3 to 7.9). The "start from USDA's last figure" slide needs the prior-alone row beside it.
- Counting gap fields helps Frederick (19.2 vs 19.6) and Wyoming (64.9 vs 66.7), both known before the rule was set.
  Two counties, not a test.
- The GPU was idle: the classifier is logistic regression on CPU (1.3 s for 820 fields, 5.8 s for 2,233) and Nemotron
  runs on NVIDIA's hosted API. One agent run took 64 s / 72,800 tokens (Wyoming) and 100 s / 102,246 tokens (Frederick).
- The 5-day replay: Frederick's silage count settles by Oct 10 and its range is within 10 points by Nov 4 with the
  Census inside it; Wyoming's range stays wide to the end because of cloud.

## Open check (for Hima)

Post the unjudged share for Lancaster 2023, Dodge and Marathon with and without Landsat from the same code, so
parts 1 and 3 can be told apart on the same footing. Prediction: low in Lancaster with both satellites, high in
Marathon without Landsat.
