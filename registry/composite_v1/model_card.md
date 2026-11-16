# composite_v1

Average of the calibrated probabilities of three low-capacity models trained on European registry labels: the timing rule, the timing+moisture rule and the six-feature logistic.

**Not scored on the Beltsville evaluation set. It was added after Evaluation 1 and joins the Evaluation 2 candidates under the rule written in ACCESS_LOG.md.**

Members (frozen, from run `24500e02a9e0`, trained on 2400 fields of ['AT'] ['2024']):

- timing_rule: silage if harv_mid_doy <= 251.5; no harvest detected -> score 0.46; Platt calibration p = sigmoid(0.19*logit(score) +0.15)
- rule: silage if harv_days_since_peak <= 26.7 OR pre_ndmi >= 0.213; no harvest detected -> score 0.46; Platt calibration p = sigmoid(0.60*logit(score) +0.12)
- logistic: logit = -0.03*z(harv_days_since_peak) -0.51*z(harv_decline_days) -0.40*z(pre_ndvi_frac) +0.67*z(pre_ndmi) +0.42*z(sen_ndvi_slope_rel) -0.53*z(post_ndti) +0.01; Platt calibration p = sigmoid(0.94*logit(score) -0.04)

Abstains when a field is low evidence or the members split their votes with probabilities at least 0.5 apart.

## Held-out results of the member average (this model's training run: `at2024_full`)

Rebuilt from the members' out-of-fold predictions; exact 95 % intervals.

- spatial: silage precision 0.714 [0.69, 0.74], silage recall 0.713 [0.69, 0.74], grain precision 0.713 [0.69, 0.74], grain recall 0.715 [0.69, 0.74], FPR 0.285 [0.26, 0.31], ROC-AUC 0.766, n 2400 (1200 silage), accuracy 0.714; abstain rate 0.33; not abstained: silage precision 0.820 [0.79, 0.85], silage recall 0.701 [0.67, 0.73], grain precision 0.725 [0.69, 0.75], grain recall 0.837 [0.81, 0.86], FPR 0.163 [0.14, 0.19], ROC-AUC 0.810, n 1603 (825 silage)

## Held-out results of the member average (family, other run: `at2022_2024`)

Rebuilt from the members' out-of-fold predictions; exact 95 % intervals.

- loyo:2022: silage precision 0.678 [0.64, 0.71], silage recall 0.693 [0.65, 0.73], grain precision 0.686 [0.65, 0.72], grain recall 0.670 [0.63, 0.71], FPR 0.330 [0.29, 0.37], ROC-AUC 0.719, n 1200 (600 silage), accuracy 0.682; abstain rate 0.55; not abstained: silage precision 0.818 [0.76, 0.86], silage recall 0.664 [0.61, 0.72], grain precision 0.671 [0.62, 0.72], grain recall 0.823 [0.77, 0.87], FPR 0.177 [0.13, 0.23], ROC-AUC 0.770, n 546 (298 silage)
- loyo:2024: silage precision 0.646 [0.62, 0.67], silage recall 0.765 [0.74, 0.79], grain precision 0.712 [0.68, 0.74], grain recall 0.582 [0.55, 0.61], FPR 0.418 [0.39, 0.45], ROC-AUC 0.744, n 2400 (1200 silage), accuracy 0.673; abstain rate 0.34; not abstained: silage precision 0.733 [0.70, 0.76], silage recall 0.752 [0.72, 0.78], grain precision 0.728 [0.69, 0.76], grain recall 0.707 [0.67, 0.74], FPR 0.293 [0.26, 0.33], ROC-AUC 0.800, n 1594 (823 silage)
- spatial: silage precision 0.711 [0.69, 0.73], silage recall 0.689 [0.67, 0.71], grain precision 0.698 [0.68, 0.72], grain recall 0.719 [0.70, 0.74], FPR 0.281 [0.26, 0.30], ROC-AUC 0.752, n 3600 (1800 silage), accuracy 0.704; abstain rate 0.40; not abstained: silage precision 0.823 [0.80, 0.85], silage recall 0.687 [0.66, 0.71], grain precision 0.710 [0.68, 0.74], grain recall 0.838 [0.81, 0.86], FPR 0.162 [0.14, 0.19], ROC-AUC 0.803, n 2151 (1124 silage)

## Known limits

- On Austria 2024 spatial CV the average is slightly below the logistic alone (ROC-AUC 0.766 vs 0.783); its case is robustness across places and years, which leave-one-country-out has to show.
- The members' strong-disagreement flag rarely fires on Austrian data (about 0.4 % of fields); most abstentions come from low evidence.
- Probabilities assume a 50/50 class balance.
