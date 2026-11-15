# The module: silage or grain, field by field and county by county

What we deliver is a module of several small models, each doing one job, not a single model.
The current version is **module_v2**, the toolbox policy `module_v2`
(`fieldvision.toolbox.Toolbox`, see `docs/toolbox.md`). The `standard` policy stays as the team's
wrapper knows it (every field to `composite_v1`), so its figures do not change under teammates. The earlier version stays
available as the policy `module_v1` and is described at the end. Every model in the module is a
frozen folder in `registry/`.

```
Sentinel-2 series per field
   │
   ▼
1 evidence tiers ──► clear / wide / abstain (every field keeps its call)   fieldvision.features (no model)
   │
   ▼
2 field classifier ──► p_silage, label                                     registry/eu2_logistic_v1
   │
   ▼
3 second opinion ──► agree / disagree                                      registry/anomaly_v1 (being built)
   │
   ▼
4 county total on the clear tier ──► headline share                       local threshold, reference county
```

module_v2 was prompted by Queen Anne's County (see "Why module_v2" below). Every rule in it was
written into the evaluation access log, with the time, before it was computed or applied.


## Where it stands (Evaluation 2, the Maryland test, the 2025 state figure)

**The module gives field calls. Its county step failed the Maryland check (Kent read high, Carroll
read low), and the label-free checks did not catch that failure.**

- **Field calls.** Evaluation 2 (run once, logged): the six-feature logistic trained on every
  European field-year (Austria 2022-2024, France 2022, Netherlands 2025), `eu2_logistic_v1`, got 42
  of the 44 Beltsville field-years right (ROC-AUC 0.95). Same run: timing+moisture rule 36,
  LightGBM 37, composite 36, timing rule 35, always grain 32.
- **County step (local threshold).** Kent (Census 5.5 %) read 21.0 %, beyond the fail line written
  down beforehand; Carroll (8.9 %) read 0 %. The flagged clear-tier area was 9.3 % / 4.4 % /
  17.0 % / 18.0 % in Queen Anne's / Carroll / Frederick / Kent against Census 1.3 / 8.9 / 19.6 /
  5.5 %. The two label-free checks, computed afterwards, passed on Kent and failed on Frederick.
- **County figures now: the counted clear share**, the share of clear-tier corn area the classifier
  calls silage. Per county it was off by up to about 7 points in either direction (4.0 / 3.6 /
  13.2 / 12.9 % in the four counties above); pooled over the four it was within half a point
  (8.4 % against 8.1 %). Any figure over many counties rests on that cancelling, which four counties
  cannot prove. Classifiers that have seen the Beltsville labels did worse here (Queen Anne's
  10.2 %, Kent 15-17 %): they raise the false alarms on the Eastern Shore.
- **Which classifier counts.** Chosen by a rule set before the table (lowest worst error over the
  four seen counties, Frederick judged against its scored-set range 11.2-19.9 %): `eu2_logistic_v1`
  worst error 7.4 points, the Europe-plus-Beltsville logistic 11.5, the Beltsville-only one 9.2.
  So the county and state figures use the Evaluation 2 model, not a locally calibrated one.
- **Why Kent reads high: two readings, not settled.** Kent's clear-tier silage calls look like
  silage in the imagery (green and moist right before the cut; labelled European fields in the same
  place are 82 % silage), and 72 of 75 were cut by 2 September. (a) The calls are mostly right and the
  county share is wrong because the clear tier is not a random part of the county: the acres flagged
  inside the clear tier are close to the county's whole Census silage in Kent (about 1,900 against
  2,264) and in Queen Anne's (about 770 against 715). (b) Early-planted grain cut while still green,
  which the training labels never show as grain, is being called silage. Weighting the clear tier's
  share by harvest window did not help (worst error 10.3 points against 7.4), and the plain lower
  bound (clear-tier silage acres over all corn acres) fits Kent and Queen Anne's but reads low in
  Carroll and Frederick, so neither reading explains all four counties yet.
- **Minimum evidence.** No county figure unless the clear tier has at least 100 fields and 15 % of
  the scored corn area; no state figure unless the drawn blocks hold 300 clear fields and 15 %.
  Otherwise the clear + wide counted share is shown as context and the figure is not reported.
- **Michigan 2022** (Gratiot 24.2 %, Branch 1.8 %) is the one clean test left of whether the counted
  clear share follows the silage share: pass only if Gratiot reads at least 10 points above Branch
  and neither is more than 10 points off (logged before opening).

## The 2025 state figure

The organisers judge results against NASS 2025 state figures. The state step was logged before any
2025 US imagery was opened, chosen after the four Maryland county results and with the 2025
Maryland target (8.9 %) already known to us (`fieldvision.model.state_estimate`):

- a simple random sample of 150 of Maryland's 838 read blocks (seed fixed); every class-1 corn field
  in them scored with `eu2_logistic_v1`; the small-grain-then-corn double crops are not scored;
- headline: the counted clear share over all drawn blocks; range: 5-95 % of 2,000 resamples of
  whole blocks; beside it, the per-county error measured in 2022 (up to about 7 points) and the
  pooled one (half a point);
- acres: share x the state's class-1 corn acres from boundaries (477,891), grain the rest; the
  double-crop corn (17,056 acres) reported beside, with the share if none or all of it is silage;
- shift guard: the counted clear share in the drawn blocks of the counties whose 2022 Census silage
  share is below 2 % (Anne Arundel, St Mary's, Queen Anne's, Baltimore; this leans on 2022 Census
  shares, not on the 2025 target); above 0.10 the headline is reported as unreliable for 2025;
- shown beside: the failed local-threshold step as a record, the reference-calibrated share, the
  unadjusted flagged share, the per-tier share, the wide tier, and the counted clear share of the
  Europe-plus-Beltsville logistic;
- the comparison with NASS 2025 is shown once.

A dry run on the four seen 2022 counties pooled as a pseudo-state gave 8.2 % [5.9, 10.7] against
their pooled Census 8.1 %; it is seen data, not evidence.

## The layers (module_v2)

**1. Evidence tiers.** From each field's own series: at least 6 usable dates from July to November,
a harvest drop found after the corn peak (the peak window ignores spring cover crops and a second
green-up), and the drop confirmed by a later observation. The harvest bracket is the number of
days between the last image with standing corn and the first image after the cut.

| tier | rule | used for |
|---|---|---|
| `clear` | evidence checks pass, bracket <= 12 days | the county headline |
| `wide` | evidence checks pass, bracket 12-20 days | reported beside the headline |
| `abstain` | bracket > 20 days, or another evidence check fails | not counted |

Every field still gets a call (`label`, `p_silage`); the tier is in `evidence.evidence_tier` and
`abstain` is true for the third tier. No model and nothing fitted. Why a bracket limit: in Queen
Anne's the share called silage rose with the bracket (6.6 % of fields at 5-10 days, 60 % at 15-20
days); with a wide bracket the drop date, and so the timing evidence, is not known.

**2. Field classifier.** `eu2_logistic_v1`, the six-feature logistic regression trained on every
European field-year (Austria 2022-2024, France 2022, Netherlands 2025; the Evaluation 2 primary).
Features: days from the green peak to the drop, how long the decline took, greenness before the
cut as a share of the peak, moisture before the cut, the relative senescence slope, and residue
after the cut. The calendar timing rule is no longer part of the call; it stays in the registry as
a baseline. Until Evaluation 2 the slot held `eu_logistic_v1` (the same family trained on Austria
2024 only); only the model file changed, so the module is still module_v2.

The Evaluation 2 primary was chosen by the rule logged at 14:07 (highest mean ROC-AUC under
leave-one-country-out, ties to the simpler family). Held out Austria / France / Netherlands:
logistic 0.754 / 0.826 / 0.717 (mean 0.765), LightGBM 0.763, composite 0.760, timing+moisture rule
0.728, timing rule 0.662. A reading of "tie" written at 17:41 (gap below the best family's spread
across countries) would have selected the timing rule; it was corrected by the coordinator after the
table was seen and before Beltsville was opened, and all three readings are in the access log.

How it was chosen (rule logged at 17:02, result at 17:16): the logistic alone against the mean of
the logistic and the timing+moisture rule, by mean ROC-AUC under leave-one-country-out on
Austria 2022-2024 and France 2022, with a difference smaller than the spread across the held-out
groups going to the logistic alone. Result: logistic 0.766 (Austria held out) / 0.822 (France held
out), mean 0.794; mean with the rule 0.763 / 0.809, mean 0.786. The logistic wins outright.

**3. Second opinion.** As in module_v1 (below): until `registry/anomaly_v1` exists the tier field
says `classifier_only`.

**4. County total, on the clear tier.** The headline (`silage_share_headline`,
`module_headline_share`) is the **local-threshold share** of the clear tier:

- threshold = the 90th percentile of classifier scores over the clear-tier fields of a reference
  county: a nearly all-grain county of the same region **and the same season**, whose Census
  silage share is known (Queen Anne's 2022, 1.3 %, for the Maryland counties);
- flagged share = share of clear-tier corn area scoring at or above that threshold, in the target
  and in the reference;
- share = (target flagged - reference flagged) / (matched TPR - reference flagged), clipped to
  0-1, where matched TPR is the recall of European held-out silage at the score that flags 10 % of
  European held-out grain, measured inside each source country-year on non-abstained clear-tier
  fields and averaged (`local` in `rates.json`, 0.508 for eu2_logistic_v1; computed with
  `fieldvision.anomaly.local`, the shared implementation).

Beside it, all on the clear tier: the counted share (area called silage), the Europe-adjusted
share, and the reference-calibrated share (see module_v1); and for context the clear tier's share
of the county's corn area, the wide tier's acreage and the share of it called silage; the
unadjusted locally flagged share; the per-tier local share (the same step inside the clear and the
wide tier, combined by area; added after seeing Frederick and Queen Anne's); and the double-crop
corn acres outside the scored field set (`unscored_corn.json`) with the county-share range they
imply (none of them silage / all of them silage). Without a
same-season reference the headline falls back to the counted share of the clear tier, and
`headline_method` says so.

Why the local threshold (rule logged at 17:30, before the next Maryland counties were opened): on
European pseudo-counties it is the only county step whose error did not blow up when the season
changed (table below). Its cost is a little more error in the ordinary settings.

**The clear tier may not be a fair sample.** A field lands in the clear tier when a clear image
happens to fall on both sides of its harvest. Whether that happens depends on weather and on when
the field was cut, and silage is cut earlier than grain, so the clear tier can hold a different mix
of silage and grain from the county as a whole. Read the clear tier's share of area next to every
headline: the lower it is, the more the headline rests on a subset that may lean one way. In
Queen Anne's 2022 the clear tier is 37 % of the corn area; in the partial Frederick snapshot 86 %.

## What was validated where (module_v2)

European held-out fields, eu_logistic family out-of-fold (run on Austria 2022-2024 and France
2022), by tier. Each cell: false-positive rate / recall / accuracy, then coverage (share of
fields in that tier):

| held out | clear tier | wide tier | clear + wide |
|---|---|---|---|
| year 2022 | 0.07 / 0.59 / 0.76, 36 % | 0.26 / 0.73 / 0.73, 23 % | 0.14 / 0.65 / 0.75, 59 % |
| year 2023 | 0.47 / 0.89 / 0.74, 70 % | 0.32 / 0.63 / 0.66, 18 % | 0.43 / 0.85 / 0.72, 88 % |
| year 2024 | 0.07 / 0.63 / 0.79, 47 % | 0.36 / 0.88 / 0.79, 20 % | 0.14 / 0.72 / 0.79, 67 % |
| Austria | 0.27 / 0.77 / 0.75, 47 % | 0.43 / 0.80 / 0.70, 20 % | 0.32 / 0.78 / 0.74, 67 % |
| France | 0.07 / 0.56 / 0.76, 45 % | 0.31 / 0.79 / 0.76, 24 % | 0.13 / 0.66 / 0.76, 69 % |

The clear tier cuts the false-positive rate to about 0.07 in three of five held-out groups, but in
2023 (and with Austria held out) the classifier's operating point moved and the tier does not fix
that: error rates measured in one season do not carry over to the next. That is why the county
step re-anchors the threshold locally.

| piece | where | result |
|---|---|---|
| Beltsville regression check | Evaluation 1 predictions of eu_logistic_v1 already scored, 44 field-years, no new scoring | clear tier 24 field-years, 0 wrong; wide tier 12, 1 silage missed; all 44: 42 right, as before |
| county step (eu_logistic_v1) | Austrian/French pseudo-counties, 150 fields, 1-50 % silage, 200 draws, clear tier | mean abs. error of the share, held out 2022 / 2023 / France: local threshold 0.087 / 0.086 / 0.082; reference-calibrated 0.072 / 0.328 / 0.068; counted 0.066 / 0.399 / 0.070; Europe-adjusted 0.071 / 0.668 / 0.073 |
| county step at 5-10 % silage | same, held-out settings pooled | local threshold mean abs. error 0.070, 90th percentile 0.147 |
| Queen Anne's 2022 (seen, eu_logistic_v1) | Census 1.3 % | clear tier 37 % of area; counted clear share 7.8 % (module_v1 headline 22.4 %); used as the reference county |
| Frederick 2022, partial (seen, eu_logistic_v1) | 248 of 2,233 fields; Census 19.6 % for the whole county | clear tier 86 %; local threshold 6.4 %, reference-calibrated 12.8 %, counted 14.3 %, Europe-adjusted 7.4 % |
| Frederick 2022, full (seen, eu_logistic_v1) | all 2,233 fields, run after the county step was fixed; Census 19.6 % | clear tier 67 % of area; **headline (local threshold) 8.4 %**, reference-calibrated 12.3 %, counted 14.0 %, Europe-adjusted 6.8 %; wide tier 9,880 acres, 24.6 % called silage. On the Kent/Carroll bands this would be weak (0.11 low) |
| county step (eu2_logistic_v1) | pseudo-counties from the all-Europe run, each held-out country or year with a source outside it, recall per country-year, clear tier | mean abs. error held out AT / FR / NL / 2022 / 2023 / 2024: local threshold (headline) 0.142 / 0.131 / 0.134 / 0.157 / 0.090 / 0.127, over-counting by 0.07-0.11 in four settings; unadjusted flagged share 0.094 / 0.078 / 0.121 / 0.080 / 0.088 / 0.079; counted 0.142 / 0.070 / 0.379 / 0.068 / 0.374 / 0.066 |
| Frederick 2022, full (seen, eu2_logistic_v1) | Census 19.6 %; 3,364 acres of double-crop corn outside the scored set | headline 18.4 %; flagged 17.0 %, per-tier 18.2 %, reference-calibrated 14.8 %, counted 13.2 %; range with the double crops 17.0-24.9 %. Queen Anne's (reference) counted clear share 4.0 % |
| **Kent 2022, clean test** (eu2_logistic_v1, run once) | Census 5.5 %; bands logged before | **headline 21.0 %: FAIL** (fail line 20.5 %); flagged 18.0 %, per-tier 23.4 %, reference-calibrated 14.4 %, counted 12.9 %, Europe-adjusted 0 %; clear tier 33 % of area; no double-crop corn |

Queen Anne's and the Frederick snapshot were both seen before module_v2 was built, so neither is a
clean test. Kent (24029, Census 5.5 %) and Carroll (24013, Census 8.9 %) are the clean tests. The
pass/weak/fail bands were logged at 17:30, before either county was opened: pass when the headline
is within 0.07 of Census, weak within 0.15, fail beyond 0.15 or when there is no headline; the
county step fails if either county fails or both are weak on the same side.

**What a pass on Kent and Carroll can and cannot show.** Both test counties have little silage
(5.5 % and 8.9 %). A county step that returned about 7 % for every county would pass both. So a
pass shows that the false-alarm level is under control in a low-silage county. It does not show
that the step follows the silage share up and down: on Frederick (19.6 %) the step reads 8.4 %, and
its flagged area separates Queen Anne's from Frederick by about 3 points where the Census differs by
18. Testing that needs a high-silage and a low-silage county in the same state, with bands written
down before either is opened.

**Why Frederick reads low** (seen data only: Queen Anne's, Frederick, European scores). The field
set matches the Census corn total (39,029 against 39,500 acres) and Frederick's clear tier has more
high-scoring area than Queen Anne's (12 % against 4 % above 0.6), so neither a missing piece of the
county nor a threshold above Maryland's silage scores explains it. The tiers do: Frederick's wide
tier was cut earlier (median 15 September against 7 October for the clear tier) and is flagged twice
as often. The recall implied by Census at the local threshold is 0.36 on the clear tier alone but
0.57 on clear and wide together, close to the European 0.63. Silage is cut in late summer, when a
clear image on both sides of the cut is less likely, so silage ends up in the wide tier more often
than grain and the clear tier under-reads where silage is common. In Queen Anne's the wide tier is
flagged 39 % with almost no silage, so its flags cannot simply be added back.

Known limits: the reference county's Census share enters the estimate, so the headline is not
independent of Census; the matched TPR comes from Europe and the local step's error grows if it is
off by 0.1 (to at most 0.11 in the pseudo-counties); on Frederick (partial snapshot and full
county) the local threshold reads low, 8.4 % against Census 19.6 %, and was not the closest of the
four clear-tier estimates there.

## module_v1 (kept, policy `module_v1`)

**1. Evidence check and abstain.** As above, with a single limit: bracket at most 20 days, no tiers.

**2. Field classifier.** `composite_v1`: the mean of the calibrated probabilities of
`eu_logistic_v1` and the two transparent rules (`eu_timing_rule_v1`,
`eu_timing_moisture_rule_v1`), all trained on Austrian registry labels only. It also abstains when
the members split their votes and are at least 0.5 apart. For the Beltsville fields only, the
in-site model `eu_barc_logistic_v1` (fitted on those fields) can be used instead (`in_site`
policy).

**3. Second opinion.** An anomaly scorer that has learned what normal grain looks like. Each field
gets a tier in `evidence.tier`: `agree`, `disagree`, or `abstained`. Until `registry/anomaly_v1`
exists the tier is `classifier_only` and `evidence.second_opinion` says so.

**4. County total.** The raw sum of probabilities, the counted share, the counted share corrected
with European error rates and its spread across European validation designs, a prior-shift
re-estimate, and the **reference-calibrated share**: the reference county gives the classifier's
local false-positive rate, FPR_ref = (counted share there - TPR x its Census share) / (1 - its
Census share), and the target county's share is (counted - FPR_ref) / (TPR - FPR_ref), clipped to
0-1, with TPR from Europe. Module headline: the reference-calibrated share when a reference is
configured (`<data root>/references.json` or `set_reference`), otherwise the Europe-adjusted share.

| piece | where | result |
|---|---|---|
| classifier family | Austria 2024, 5-fold spatial CV (2,400 fields) | logistic ROC-AUC 0.78, timing rule 0.67, LightGBM 0.86 |
| European models on US fields | Beltsville 44 field-years, Evaluation 1 (zero-shot, once) | logistic 42 of 44 (secondary), timing rule 38, LightGBM 30, always grain 32 |
| in-site models | Beltsville, leave-one-year-out and leave-one-field-out | 43 of 44; labels-shuffled check: Beltsville-only model at most 40 |
| county estimators | Austrian pseudo-counties, held-out year | counted 0.15, Europe-adjusted 0.10, reference-calibrated 0.10, raw 0.30 |
| headline counted share | Queen Anne's MD 2022, pre-registered check | counted 22.4 % vs Census 1.3 %: failed, in the direction stated in advance |

## Why module_v2

The pre-registered Queen Anne's check failed: composite_v1 called 22.4 % of the corn area silage
where Census has 1.3 %. The breakdown showed two causes. Fields with a wide harvest bracket were
called silage far more often, because the drop date is unknown there; and almost every silage call
had the calendar timing rule's vote even when the field took 65 days from peak to drop, because
the rule reads an early drop as silage and Maryland harvests earlier than Austria. module_v2 removes
the first cause with the evidence tiers and the second by taking the timing rule out of the vote.
