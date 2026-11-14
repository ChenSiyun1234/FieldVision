# Model registry (pointer)

Frozen model folders live outside git, in `C:\t\fv\data\registry\` (the repo keeps no model
files). Each folder is written once and loaded read-only:

```
<name>/
  model.json            what the model is; full thresholds for rules
  model.joblib          fitted object (ML models; also kept for rules fitted with calibration)
  feature_schema.json   feature names in order, units, gap handling, feature settings
  applicability.json    data the model needs, when it abstains, what it was validated on
  model_card.md         metrics, confusion matrices, calibration, known limits
  provenance.json       training data hashes, git commit, dates, run and frozen ids
```

| name | what | trained on | validated on |
|---|---|---|---|
| `ts_model_v1` | LightGBM on self-referenced harvest features (Platt-calibrated) | Austria 2024 registry labels, 2,400 fields | Austrian spatial CV; Beltsville zero-shot (Evaluation 1) |
| `eu_timing_rule_v1` | Timing rule: silage if the harvest drop mid-point is early enough | Austria 2024 | Austrian spatial CV; Beltsville zero-shot (Evaluation 1) |
| `timing_rule_v1` | Thresholds on drop date and pre-drop NDVI, the team's specification | Beltsville labels (in-site) | Beltsville leave-one-year-out |

Make one: `python -m fieldvision.model.registry export --run <run dir> --which gbm --name ts_model_v2 ...`
(see `docs/model.md`). Use one: `python -m fieldvision.model.classify --registry C:\t\fv\data\registry
--model ts_model_v1 --timeseries ... --fields ... --state MD --out C:\t\fv\data\results`.
