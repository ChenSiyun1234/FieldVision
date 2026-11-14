# Worked example of the result shape

`MD_2021.jsonl`, `MD_2022.jsonl`, `MD_2023.jsonl`: one JSON object per field, season and model for
the 44 Beltsville (USDA BARC, Prince George's County, MD) field-years. They carry no labels.

```json
{"field_id": "BARC-112", "year": 2023, "source_field_id": "barc-112-2023", "model": "eu_logistic_v1",
 "p_silage": 0.91, "label": "silage", "abstain": false,
 "evidence": {"drop_date": "2023-08-29", "last_standing_date": "2023-08-23", "days_to_drop": 6,
              "ndvi_before": 0.81, "ndvi_after": 0.24, "ndmi_before": 0.33,
              "greenness_before_share_of_peak": 0.9, "days_peak_to_drop": 35.0,
              "residue_ndti_after": 0.11, "n_usable_dates": 30, "largest_gap_days": 15,
              "abstain_reason": null}}
```

`field_id` is `BARC-<OBJECTID>`; the key is `field_id` + `year`. `abstain` is true when the
evidence is too thin (harvest bracket wider than 20 days, too few usable dates, no or
unconfirmed drop); the call is still given but should not be trusted. `drop_date` is the first
clear observation after the cut and `last_standing_date` the last one before it.

What each model's rows are:

| model | trained on | are these rows a test? |
|---|---|---|
| `ts_model_v1` | Austria 2024 only | yes: never saw Beltsville (Evaluation 1's LightGBM) |
| `eu_timing_rule_v1` | Austria 2024 only | yes: never saw Beltsville |
| `eu_logistic_v1` | Austria 2024 only | yes: never saw Beltsville (secondary model in Evaluation 1) |
| `timing_rule_v1` | Beltsville labels | **out-of-fold**: each year is predicted by the rule fitted on the other two years (`evidence.fold = "test-<year>"`). The frozen `timing_rule_v1` folder is fitted on all three years; its predictions on these same fields would be in-sample and are not shown here |

| `barc_logistic_v1` | Beltsville labels only (in-site) | **out-of-fold**: each field predicted by the model fitted without any season of that field (leave-one-field-out, `evidence.fold = "test-field-<OBJECTID>"`). The frozen folder is fitted on all 44 field-years |
| `eu_barc_logistic_v1` | Austria 2024 plus Beltsville labels (in-site) | **out-of-fold**, same design as the row above |

Produced with `python -m fieldvision.model.classify` (see `registry/README.md`); the
`timing_rule_v1` rows come from `fieldvision.beltsville.timing_rule`'s leave-one-year-out
predictions.
