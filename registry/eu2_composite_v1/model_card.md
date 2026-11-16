# eu2_composite_v1

Average of the calibrated probabilities of three low-capacity models trained on European registry labels: the timing rule, the timing+moisture rule and the six-feature logistic.

**Not scored on the Beltsville evaluation set. It was added after Evaluation 1 and joins the Evaluation 2 candidates under the rule written in ACCESS_LOG.md.**

Members (frozen, from run `0484d7e43267`, trained on 8000 fields of ['AT', 'FR', 'NL'] ['2022', '2023', '2024', '2025']):

- timing_rule: silage if harv_mid_doy <= 257.5; no harvest detected -> score 0.62; Platt calibration p = sigmoid(0.26*logit(score) +0.09)
- rule: silage if harv_days_since_peak <= 76.0 AND pre_ndmi >= 0.173; no harvest detected -> score 0.62; Platt calibration p = sigmoid(0.59*logit(score) -0.01)
- logistic: logit = -0.25*z(harv_days_since_peak) -0.31*z(harv_decline_days) -0.24*z(pre_ndvi_frac) +0.58*z(pre_ndmi) +0.31*z(sen_ndvi_slope_rel) -0.44*z(post_ndti) -0.00; Platt calibration p = sigmoid(0.98*logit(score) -0.00)

Abstains when a field is low evidence or the members split their votes with probabilities at least 0.5 apart.

## Held-out results of the member average (this model's training run: `eu_all`)

Rebuilt from the members' out-of-fold predictions; exact 95 % intervals.

- loco:AT: silage precision 0.702 [0.68, 0.72], silage recall 0.703 [0.68, 0.72], grain precision 0.703 [0.68, 0.72], grain recall 0.701 [0.68, 0.72], FPR 0.299 [0.28, 0.32], ROC-AUC 0.756, n 4800 (2400 silage), accuracy 0.702; abstain rate 0.33; not abstained: silage precision 0.774 [0.75, 0.79], silage recall 0.716 [0.69, 0.74], grain precision 0.708 [0.69, 0.73], grain recall 0.766 [0.74, 0.79], FPR 0.234 [0.21, 0.26], ROC-AUC 0.792, n 3204 (1688 silage)
- loco:FR: silage precision 0.798 [0.77, 0.83], silage recall 0.656 [0.62, 0.69], grain precision 0.708 [0.68, 0.74], grain recall 0.834 [0.81, 0.86], FPR 0.166 [0.14, 0.19], ROC-AUC 0.800, n 1600 (800 silage), accuracy 0.745; abstain rate 0.32; not abstained: silage precision 0.821 [0.78, 0.85], silage recall 0.673 [0.63, 0.71], grain precision 0.694 [0.66, 0.73], grain recall 0.834 [0.80, 0.87], FPR 0.166 [0.13, 0.20], ROC-AUC 0.810, n 1091 (578 silage)
- loco:NL: silage precision 0.641 [0.61, 0.67], silage recall 0.825 [0.80, 0.85], grain precision 0.754 [0.72, 0.79], grain recall 0.537 [0.50, 0.57], FPR 0.463 [0.43, 0.50], ROC-AUC 0.724, n 1600 (800 silage), accuracy 0.681; abstain rate 0.20; not abstained: silage precision 0.647 [0.61, 0.68], silage recall 0.845 [0.82, 0.87], grain precision 0.746 [0.70, 0.79], grain recall 0.497 [0.46, 0.54], FPR 0.503 [0.46, 0.54], ROC-AUC 0.716, n 1273 (665 silage)
- loyo:2022: silage precision 0.766 [0.74, 0.79], silage recall 0.654 [0.63, 0.68], grain precision 0.698 [0.67, 0.72], grain recall 0.800 [0.78, 0.82], FPR 0.200 [0.18, 0.22], ROC-AUC 0.776, n 2800 (1400 silage), accuracy 0.727; abstain rate 0.41; not abstained: silage precision 0.833 [0.80, 0.86], silage recall 0.663 [0.63, 0.69], grain precision 0.684 [0.65, 0.71], grain recall 0.846 [0.82, 0.87], FPR 0.154 [0.13, 0.18], ROC-AUC 0.803, n 1638 (879 silage)
- loyo:2023: silage precision 0.760 [0.72, 0.79], silage recall 0.712 [0.67, 0.75], grain precision 0.729 [0.69, 0.76], grain recall 0.775 [0.74, 0.81], FPR 0.225 [0.19, 0.26], ROC-AUC 0.789, n 1200 (600 silage), accuracy 0.743; abstain rate 0.12; not abstained: silage precision 0.758 [0.72, 0.79], silage recall 0.739 [0.70, 0.77], grain precision 0.707 [0.66, 0.75], grain recall 0.727 [0.69, 0.77], FPR 0.273 [0.23, 0.31], ROC-AUC 0.776, n 1050 (563 silage)
- loyo:2024: silage precision 0.674 [0.65, 0.70], silage recall 0.750 [0.72, 0.77], grain precision 0.718 [0.69, 0.74], grain recall 0.637 [0.61, 0.66], FPR 0.363 [0.34, 0.39], ROC-AUC 0.760, n 2400 (1200 silage), accuracy 0.693; abstain rate 0.33; not abstained: silage precision 0.778 [0.75, 0.81], silage recall 0.732 [0.70, 0.76], grain precision 0.733 [0.70, 0.76], grain recall 0.779 [0.75, 0.81], FPR 0.221 [0.19, 0.25], ROC-AUC 0.812, n 1605 (825 silage)
- loyo:2025: silage precision 0.641 [0.61, 0.67], silage recall 0.825 [0.80, 0.85], grain precision 0.754 [0.72, 0.79], grain recall 0.537 [0.50, 0.57], FPR 0.463 [0.43, 0.50], ROC-AUC 0.724, n 1600 (800 silage), accuracy 0.681; abstain rate 0.20; not abstained: silage precision 0.647 [0.61, 0.68], silage recall 0.845 [0.82, 0.87], grain precision 0.746 [0.70, 0.79], grain recall 0.497 [0.46, 0.54], FPR 0.503 [0.46, 0.54], ROC-AUC 0.716, n 1273 (665 silage)
- spatial: silage precision 0.710 [0.70, 0.72], silage recall 0.726 [0.71, 0.74], grain precision 0.720 [0.71, 0.73], grain recall 0.704 [0.69, 0.72], FPR 0.296 [0.28, 0.31], ROC-AUC 0.762, n 8000 (4000 silage), accuracy 0.715; abstain rate 0.30; not abstained: silage precision 0.751 [0.73, 0.77], silage recall 0.740 [0.72, 0.76], grain precision 0.716 [0.70, 0.73], grain recall 0.727 [0.71, 0.74], FPR 0.273 [0.26, 0.29], ROC-AUC 0.782, n 5580 (2936 silage)

## Known limits

- On Austria 2024 spatial CV the average is slightly below the logistic alone (ROC-AUC 0.766 vs 0.783); its case is robustness across places and years, which leave-one-country-out has to show.
- The members' strong-disagreement flag rarely fires on Austrian data (about 0.4 % of fields); most abstentions come from low evidence.
- Probabilities assume a 50/50 class balance.
