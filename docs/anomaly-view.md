# Anomaly view: learn only "normal grain", flag what departs from it

## Conclusion

A score that only measures how unlike normal grain a field looks does not find silage:
on Austrian data every one-class scorer we tried ranks silage worse than the
supervised 6-feature logistic (best ROC-AUC 0.69 against 0.77), because silage is a
shifted part of the grain distribution, not something outside it, and the score flags
odd grain fields (second green-up, cloudy series, early harvest) just as readily; it
stays in the code only as a diagnostic of unusual fields, never as a silage vote.
What does carry over is learning the classifier's *threshold* from local, mostly-grain
fields: on Austrian pseudo-counties it never fails badly (mean absolute error of the
area share 0.10-0.13 in every held-out year, where counting at the classifier's own
cut-off ranges from 0.08 to 0.33), but in Maryland its one real test so far, Frederick's
partial snapshot, comes out at 15% against the Census 19.6% with both of its failure
checks firing, so the Queen Anne's reference probably does not stand in for that part
of Frederick and the number should not be trusted.

**Known weak point, up front.** Any reference-county method assumes the target
county's grain looks to the classifier like the reference county's grain. The
modelling session found on the same partial Frederick snapshot that the false-positive
rate taken from Queen Anne's at the 0.5 cut-off (0.218) is larger than Frederick's whole
counted share (18.9%), so a reference county's rate need not carry to its neighbour.
This method does not escape that assumption; it only moves it from the 0.5 cut-off to
the reference's 90th percentile. When it fails, the adjusted share is biased in the
direction of the shift (target grain scoring higher -> over-count, lower -> under-count,
clipped at 0), and it shows through two checks computed from the two counties' scores
alone, with no labels: (1) the estimate drifts with alpha (estimates at alpha 0.05,
0.10, 0.20 and 0.30 disagree), and (2) the share of the target's area below the
reference's median score differs from what the estimate implies. On Austrian
pseudo-counties of Frederick's size both checks together reach Frederick's values in
under 1% of draws. When they fire, the county figure should be reported as "reference
does not match", not as a number.

## Stage 1: distance-to-normal scorers on European data

Data: Austria 2022, 2023, 2024, 4,800 labelled fields (2,400 silage). Splits are the
main line's: 5 spatial folds grouped by block (seed 0), then leave-one-year-out.
"Normal" = the grain fields of the training part only. Three inputs:

- phys6: the six physical features of the small logistic model;
- rel: the curve aligned on each field's own NDVI peak (index medians in 15-day bins,
  -30 to +120 days, 80 numbers);
- raw: monthly composites April-November of the ten S2 bands and NDVI (88 numbers).

Three standard scorers, all fit on the normal set alone (median/IQR scaling, PCA to
10 components above 10 inputs): robust Mahalanobis distance (minimum covariance
determinant), isolation forest, and mean distance to the 5 nearest normal fields
(memory bank). The flag threshold is the 98th percentile of the normal set's own
cross-fitted scores. The supervised logistic and LightGBM of the main line are fit on
the same training parts, with labels, for reference.

ROC-AUC with block-bootstrap 95% interval; spatial folds also show fold mean ± SD.
"1/5/20%" = AUC when that share of the normal set is silage (mean of 3 draws).
Recall and FPR are at the 98th-percentile flag threshold (logistic: its 0.5 cut-off).

| model | spatial AUC [95% CI] | folds | recall | FPR | 1% | 5% | 20% | diff vs logistic [95% CI] |
|---|---|---|---:|---:|---:|---:|---:|---|
| phys-logistic (supervised) | 0.772 [0.751, 0.797] | 0.77 ± 0.02 | 0.713 | 0.262 | | | | ref |
| phys-gbm (supervised) | 0.854 [0.837, 0.871] | 0.85 ± 0.03 | 0.779 | 0.202 | | | | +0.081 [+0.065, +0.097] |
| mahalanobis-raw | 0.685 [0.658, 0.719] | 0.68 ± 0.04 | 0.068 | 0.022 | 0.686 | 0.688 | 0.658 | -0.087 [-0.120, -0.052] |
| knn-raw | 0.680 [0.655, 0.708] | 0.68 ± 0.03 | 0.071 | 0.020 | 0.678 | 0.670 | 0.632 | -0.093 [-0.125, -0.059] |
| iforest-phys6 | 0.661 [0.630, 0.691] | 0.67 ± 0.03 | 0.060 | 0.023 | 0.660 | 0.642 | 0.581 | -0.112 [-0.146, -0.081] |
| iforest-raw | 0.635 [0.610, 0.664] | 0.64 ± 0.01 | 0.104 | 0.023 | 0.637 | 0.633 | 0.607 | -0.137 [-0.172, -0.104] |
| iforest-rel | 0.606 [0.581, 0.635] | 0.61 ± 0.02 | 0.109 | 0.028 | 0.606 | 0.603 | 0.593 | -0.166 [-0.205, -0.126] |
| knn-phys6 | 0.570 [0.552, 0.591] | 0.57 ± 0.03 | 0.035 | 0.020 | 0.564 | 0.541 | 0.491 | -0.202 [-0.229, -0.175] |
| knn-rel | 0.563 [0.551, 0.586] | 0.60 ± 0.03 | 0.060 | 0.015 | 0.563 | 0.560 | 0.557 | -0.210 [-0.241, -0.175] |
| mahalanobis-phys6 | 0.550 [0.523, 0.573] | 0.56 ± 0.03 | 0.022 | 0.024 | 0.546 | 0.524 | 0.475 | -0.222 [-0.265, -0.184] |
| mahalanobis-rel | 0.550 [0.538, 0.573] | 0.60 ± 0.02 | 0.027 | 0.007 | 0.551 | 0.554 | 0.554 | -0.222 [-0.246, -0.193] |

Held-out year, ROC-AUC (trained on the other two years):

| model | 2022 | 2023 | 2024 |
|---|---:|---:|---:|
| phys-logistic (supervised) | 0.704 | 0.804 | 0.775 |
| phys-gbm (supervised) | 0.801 | 0.851 | 0.840 |
| mahalanobis-raw | 0.660 | 0.628 | 0.735 |
| knn-raw | 0.624 | 0.574 | 0.679 |
| iforest-phys6 | 0.543 | 0.771 | 0.629 |
| iforest-raw | 0.590 | 0.567 | 0.675 |
| iforest-rel | 0.610 | 0.694 | 0.576 |
| knn-rel / mahalanobis-rel | 0.617 / 0.624 | 0.679 / 0.682 | 0.576 / 0.584 |
| knn-phys6 / mahalanobis-phys6 | 0.415 / 0.432 | 0.693 / 0.718 | 0.506 / 0.477 |

Every scorer is below the logistic, with the paired interval excluding zero, except
mahalanobis-raw on 2022 (-0.044 [-0.100, +0.015], no clear difference). The 98th
percentile threshold does hold on new grain (FPR 1-5%), but it catches only 2-16% of
silage.

**Impure normal set.** 1% or 5% silage in the normal set costs at most 0.02 AUC;
20% costs 0.03-0.08 for the phys6 and raw scorers and about 0.01 for the curve
scorers. So purity is not the problem; separation is.

**Agreement tiers** (logistic call vs mahalanobis-raw flag, spatial folds; "abstain"
= fields the feature code marks low evidence, 33% of fields here):

| tier | share of fields | silage share | classifier accuracy | recall | FPR |
|---|---:|---:|---:|---:|---:|
| agree | 0.36 | 0.31 | 0.73 | 0.15 | 0.01 |
| disagree | 0.30 | 0.79 | 0.79 | 0.98 | 0.94 |
| abstain | 0.33 | 0.45 | 0.66 | 0.70 | 0.37 |
| all | 1.00 | 0.50 | 0.73 | 0.71 | 0.26 |

The same holds for iforest-phys6 and for each held-out year. Because the flag fires
so rarely, "agree" is almost exactly "the classifier says grain" and "disagree" is
"the classifier says silage"; the classifier is no more accurate where the two views
agree, so a tier built this way is not worth showing.

**What gets flagged among grain.** The 50 grain fields with the highest
mahalanobis-raw score (held out, each scaled by its fold's threshold), compared with
all grain fields: a second green-up (double crop or cover crop) in 42% vs 13%; a
sparse or cloudy series in 40% vs 38%; an early harvest (below the grain 10th
percentile of days after peak) in 16% vs 3%; no harvest found in 4% vs 1%; no weak or
failed crops (peak NDVI below 0.6) at all. So the distance view mostly finds unusual
cropping, which is useful as a data-quality note but is not a silage vote.

## What carries over: a local threshold on the classifier

The classifier's problem in a new region is mainly where its cut-off falls (in
Queen Anne's County it calls far more silage than the Census allows), not which
direction is silage. So keep the direction from European labels and learn only the
cut-off from local, mostly-grain fields:

1. score the reference fields (mostly grain, no labels read) with the classifier; the
   threshold is the 90th percentile of those scores (alpha = 0.10), so about 10% of
   local grain is flagged whatever the classifier's calibration;
2. on the labelled source data, with out-of-fold scores, find the score that flags
   the same 10% of source grain and the share of source silage above it (the matched
   recall); this assumes the separation of the two classes carries over, not their
   location;
3. per field, silage = score at or above the local threshold; for the county share,
   adjusted count: (flagged share - reference flagged share) / (matched recall -
   reference flagged share), clipped to 0-1.

European check: for each held-out Austrian year, the logistic is trained on the other
two years; the year's blocks are split at random into a "reference county" (its grain
plus 1.3% silage, as in Queen Anne's) and a "test county" (its grain plus 20% silage);
50 random splits, mean and 5-95% range. alpha = 0.10 was chosen before this check;
0.05 and 0.20 are shown only as sensitivity.

| held out | classifier at 0.5: FPR | counted share | local: FPR | recall | flagged share | adjusted share (true 0.20) |
|---|---|---|---|---|---|---|
| 2022 | 0.33 [0.27, 0.40] | 0.39 [0.32, 0.45] | 0.10 [0.04, 0.18] | 0.36 [0.21, 0.52] | 0.15 [0.08, 0.23] | 0.13 [0.00, 0.33] |
| 2023 | 0.34 [0.28, 0.40] | 0.44 [0.38, 0.49] | 0.10 [0.04, 0.17] | 0.49 [0.31, 0.68] | 0.18 [0.09, 0.28] | 0.21 [0.00, 0.49] |
| 2024 | 0.23 [0.18, 0.29] | 0.33 [0.28, 0.37] | 0.10 [0.04, 0.18] | 0.45 [0.30, 0.61] | 0.17 [0.10, 0.27] | 0.21 [0.00, 0.49] |

Adjusted share with alpha 0.05 / 0.20: 2022 0.14 / 0.13, 2023 0.19 / 0.23,
2024 0.22 / 0.21. The false-positive rate is held where it was set; the adjusted share
is about right on average but a single county's estimate is noisy (the 5-95% range
spans 0 to about 0.5 with 300-750 fields per half), and 2022 comes out low because the
two classes separate less well that year than in the training years, which the matched
recall cannot know. The European year shift is also much smaller than the shift to
Maryland, so this checks the mechanics, not the size of the effect there. (This check
scores every field of the test county, so its matched recall from all source fields
is consistent with it; see the correction below for the county recipe.)

### Correction: the matched recall must come from the same kind of field

The county recipe scores only fields that are not low evidence, but its first version
measured the matched recall on all source fields, low-evidence ones included. Those
separate worse, so the recall was understated (Austria, at a 10% local false-positive
rate on fields that are not low evidence: target-year recall 0.56 in 2022 and 0.67 in
2024, against 0.49 and 0.44 from all source fields) and the adjusted share overstated,
by more the more silage there is. Found on the Austrian pseudo-counties below, after
the Frederick snapshot had been scored once; fixed by dropping low-evidence fields from
the source scores (`local_eval.source_and_target_scores`, default). With it, the
matched recall for Maryland is 0.535 instead of 0.462. Both Frederick numbers are
reported below.

### Pseudo-counties, on the modelling session's footing

The modelling session's county check (`fieldvision.model.county_validation`) draws
pseudo-counties of 250 held-out fields at silage shares from 1% to 50%, with a separate
250-field reference county at about 1%, 200 draws, and scores the area share against
the truth. The same construction here, on the 6-feature logistic trained on the other
two Austrian years, plus a harder version where the reference and the target come from
different random halves of the held-out year's blocks (neighbouring, not identical,
counties). Estimators, all over the area that is not low evidence:

- counted: share of area at or above the classifier's 0.5;
- Europe-adjusted: counted, corrected with the TPR and FPR at 0.5 from the source;
- local: this recipe, one threshold for the whole county (corrected matched recall);
- local, not adjusted: the share of area above the local threshold, as it is;
- local, stratified: the same per stratum of evidence quality (harvest bracket at most
  10 days, or 11-20 days; the cut was fixed from the feature's distribution before any
  result), each with its own threshold and matched recall, combined by area.

Mean absolute error of the area share over all shares (1-50%), block-split version
(the random version in brackets):

| held out | counted | Europe-adjusted | local | local, not adjusted | local, stratified |
|---|---:|---:|---:|---:|---:|
| 2022 | 0.159 (0.152) | 0.082 (0.064) | 0.129 (0.120) | 0.107 (0.098) | 0.122 (0.106) |
| 2023 | 0.327 (0.327) | 0.446 (0.447) | 0.099 (0.059) | 0.093 (0.079) | 0.114 (0.072) |
| 2024 | 0.077 (0.076) | 0.112 (0.112) | 0.111 (0.103) | 0.072 (0.069) | 0.127 (0.118) |
| mean | 0.188 | 0.213 | 0.113 | 0.091 | 0.121 |

- The stratified version is not better than one threshold in any year, so the
  whole-county threshold stays.
- Counting and Europe-adjusted counting are best in one year and fail in another
  (2023, where the classifier's calibration moved: errors 0.33 and 0.45); the two local
  versions never exceed 0.13. That robustness is the reason to use a local reference.
- The plain locally-flagged share has the lower mean error: it is better at 10-30%
  silage and worse at 1-5% (it cannot read below about the alpha share), the same
  pattern the modelling session found on its composite scores. The adjusted version
  still over-counts by about 0.08 in 2022 and 2024, because those years separate the
  classes better than the training years, which the matched recall cannot know.

## Recipe for Maryland, fixed before scoring Frederick

Written down and committed before the target county was read; nothing below was tuned
on Frederick or on any Census number.

- Classifier: the main line's 6-feature logistic (`make_factories(..., "relative")["logistic"]`),
  trained on all 4,800 labelled Austrian fields (2022, 2023, 2024).
- Matched recall: out-of-fold scores from 5 spatial folds on the same 4,800 fields
  (corrected later to the fields that are not low evidence; see the correction above).
- Reference: Queen Anne's County (24035) 2022, every corn field with a feature row that
  is not low evidence (1,054 of 1,449); labels are not read (there are none).
- alpha = 0.10, so the local threshold is the reference's 90th percentile of scores
  (0.559 on Queen Anne's).
- Target: score every field of the target county; report, over fields that are not
  low evidence, the flagged share and the adjusted share, by area (primary, since the
  Census counts acres) and by field count, next to the classifier's own 0.5 cut-off.
- Distance view, reported as a diagnostic only: robust Mahalanobis on the monthly
  spectra, fit on the reference, flag above the reference's cross-fitted 98th percentile.
- Known bias: the reference contains some silage (1.3% of acres by the Census), so the
  true grain false-positive rate is slightly below the reference's flagged share; we do
  not correct for it, because that would use the Census number.

Self-check on the reference itself (by construction): the classifier at 0.5 calls
14.9% of Queen Anne's fields (18.1% of area) silage; the local threshold flags 10.1% of
fields (11.5% of area) and the adjusted share is 0. These differ from the modelling
session's 22.4% because that check used its own trained model and abstention rule.

## Stage 2: Frederick County

Scored at 16:55 on 14 November, after the recipe above was committed, on the partial
snapshot available then: 248 fields in 7 adjacent blocks of 70 (about 11% of the
county's corn acres), so not a sample of the whole county. 238 are not low evidence.
The Census share (19.6%) was not used for anything below.

| | Queen Anne's (reference, Census 1.3%) | Frederick, partial (Census 19.6%) |
|---|---:|---:|
| fields scored | 1,054 | 238 |
| classifier at 0.5, share of area | 0.181 | 0.195 [0.14, 0.26] |
| classifier at 0.5, share of fields | 0.149 | 0.256 |
| local threshold, flagged share of area | 0.115 | 0.180 |
| local threshold, adjusted share of area, first version (recall 0.462) | 0 (by construction) | 0.187 [0.03, 0.36] |
| local threshold, **adjusted share of area, corrected** (recall 0.535) | 0 (by construction) | **0.154** |
| local threshold, adjusted share of fields, corrected | 0 (by construction) | 0.320 |
| local threshold, stratified (not adopted), share of area | | 0.211 |
| distance view, flagged share of area | 0.008 | 0.043 |

Intervals: 2,000 bootstrap resamples of Frederick's scored fields; they leave out the
uncertainty in the reference and the matched recall and the clustering of fields in
blocks, so the real uncertainty is wider. Agreement: the local threshold flags 57
fields, all of which the classifier at 0.5 also flags; the classifier flags 4 more.
The distance view flags 7 fields, one of them also flagged by both.

**Failure checks** (corrected recipe): the adjusted area share at alpha 0.05 / 0.10 /
0.20 / 0.30 is 0.35 / 0.15 / 0 / 0, and 62% of Frederick's scored area lies below
Queen Anne's median score where the estimate implies 40% (ratio 1.56). Frederick's
grain sits mostly lower than Queen Anne's while its top few percent sit higher, so no
single local rate fits both. To judge whether that is just noise at this size, the same
checks were run on 1,200 Austrian pseudo-counties of 240 scored fields with a reference
of 96-340 scored fields (all smaller than Queen Anne's 1,054, so if anything noisier):
the alpha drift reached 0.35 in 4.8% of them (0.8% in 2023 and 2024), the lower-tail
ratio reached 1.56 in 1.9% (none in 2023 and 2024), and both together in 0.8% (none in
2023 and 2024). Where the alpha drift was in its top 10%, the error was twice as large
(0.17 against 0.085). So the checks say the reference does not match this part of
Frederick.

Reading: at its own cut-off the classifier finds about as much silage in Queen Anne's
as in Frederick (18% and 20% of area) while the Census puts them 18 points apart, so
the plain classifier cannot tell these counties apart. The local threshold does
separate them, but on this snapshot its estimate depends on alpha and its checks fire,
so it should be reported as "reference does not match" rather than as 15%. The snapshot
is 7 adjacent blocks in one corner of a county that spans very different terrain from
Queen Anne's; the full Frederick extraction will show whether the mismatch is that
corner or the county. Kent and Carroll, untouched, should be scored with exactly this
recipe and checks, read as numbers only when the checks pass.

## The importable layer

`fieldvision.anomaly.local.LocalThreshold` (fit on reference scores plus source
out-of-fold scores and labels, all from fields that are not low evidence;
`calls(scores)`, `county_share(scores)`) is the county layer recommended here, with
`fieldvision.anomaly.county.failure_checks` reported next to every county figure.
`local.local_share` and `local.stratified_share` are the area-weighted functions used
on the pseudo-counties. `fieldvision.anomaly.api` holds the distance view:
`fit_reference(reference_table)` and `score_target(model, target_table)` work on the
feature table that `features.build.build_features` produces (plus the monthly
composite columns for the raw input), and `confidence_tier(classifier_pred,
anomaly_flag, low_evidence)` gives agree / disagree / abstain. Tests:
`tests/test_anomaly.py`. Stage 1 runs with `python -m fieldvision.anomaly.evaluate`,
the European local-threshold check with `python -m fieldvision.anomaly.local_eval`,
and Stage 2 with `python -m fieldvision.anomaly.county`.

## Next steps

**The image time series as a 3D spatio-temporal input.** Instead of one average per
field and date, the model would see each field as a stack of small images over the
season, so it could use within-field texture and how harvest spreads across a field
(silage fields are often cut in one pass, grain combined over days; edges and
headlands behave differently). Not built today: it needs the pixels re-read for every
field, and a larger model, while today's evidence is that more capacity transfers
worse across regions. What would have to be extracted, from cheapest to fullest: per
field and date, the within-field spread (standard deviation and 10th/90th percentiles
of NDVI, NDMI and NDTI, plus the share of pixels below a bare-soil NDVI), which fits
the current tables and the current models; or per-field pixel chips (for example
32x32 pixels at 10 m, all bands, every usable date), which is roughly a hundred times
the current storage and needs a convolutional or attention model. The spread
statistics should be tried first, on the same folds, with the transfer check.
