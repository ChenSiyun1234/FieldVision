# Model registry

Frozen model folders, loaded read-only. Each is written once by code and the model files are
never edited. Added after export, as reference values the models themselves do not read:
`eu_timing_rule_v1/training_reference.json` (Austria's median drop day, for the toolbox's
regional threshold) and `evidence_reference.json` in the European folders (class medians of the
evidence features, for explanation sentences). New exports put both into `model.json`. Paths in
`provenance.json` are written relative to the data folder (`data/...`) and commands as
`python -m ...`; no machine-specific path is ever written.

```
registry/<name>/
  model.json            what the model is; full thresholds for rules, members for the composite
  model.joblib          fitted object (ML models, calibrated rules, composite)
  feature_schema.json   feature names in order, units, gap handling, feature settings
  applicability.json    data the model needs, when it abstains, what it was validated on
  model_card.md         metrics, confusion matrices, calibration, known limits
  provenance.json       training data hashes, git commit, dates, run and frozen ids
```

| name | what | trained on | validated on | Beltsville |
|---|---|---|---|---|
| `eu_logistic_v1` | six-feature logistic (timing, dry-down, pre-harvest greenness and moisture, slope, residue), calibrated | Austria 2024 | Austrian spatial CV | Evaluation 1 secondary: 42 of 44 |
| `eu_timing_rule_v1` | silage if the drop mid-point is on or before about day 256 | Austria 2024 | Austrian spatial CV | Evaluation 1: 38 of 44 |
| `eu_timing_moisture_rule_v1` | silage if the drop comes soon after the field's own peak OR the canopy is still wet | Austria 2024 | Austrian spatial CV | Evaluation 1 secondary: 35 of 44 |
| `eu_gbm_v1` | LightGBM, 135 features; **did not transfer** (card says so) | Austria 2024 | Austrian spatial CV | Evaluation 1 primary: 30 of 44, below always-grain (32) |
| `ts_model_v1` | same model as `eu_gbm_v1` (earlier name, kept for the team's links) | Austria 2024 | as above | as above |
| `composite_v1` | mean of the calibrated probabilities of the two rules and the logistic, abstains on strong disagreement | Austria 2024 | Austrian spatial CV and year hold-out | **not scored before Evaluation 2** |
| `timing_rule_v1` | the team's rule: thresholds on drop date and pre-drop NDVI | Beltsville labels (in-site) | Beltsville leave-one-year-out | 40 of 44 out-of-fold |

Classify a county-year from a time-series file (run from the repo root, `src` on the path;
in PowerShell `$env:PYTHONPATH="src"`):

```
python -m fieldvision.model.classify --model eu_logistic_v1 --timeseries <timeseries.parquet> --fields <fields.geojson> --state MD --out results
```

`--registry` defaults to this folder. Output: `results/<STATE>_<year>.parquet` and `.jsonl` in the
shape shown in `results/README.md`; several models share one file and each run replaces only its
own rows.

Make a new folder from a training run (never overwrite an existing version):

```
python -m fieldvision.model.registry export --run <run dir> --which logistic --name eu_logistic_v2 --training-files <files> --git-commit <sha>
python -m fieldvision.model.registry composite --run <run dir> --name composite_v2 --training-files <files> --git-commit <sha>
```

The toolbox for the team's agent (`fieldvision.toolbox.Toolbox`) reads models from here too.
