# FieldVision bridge

The layer between the team's FieldVision models (`/workspace/FieldVision`) and the Nemotron agents
in `agents/`. It decides **which models the agent may use**, makes **county totals realistic**, and
gives the reporter, Q&A and planner **per-field evidence**.

## What it changes, and why

| Change | Why |
|---|---|
| Default model `eu_logistic_v1` behind the `standard` route | 42 of 44 on Beltsville without seeing Maryland. FieldVision's own default, `composite_v1`, is not yet scored |
| Routes come from the toolbox | The supervisor accepts any route `route_options` marks available, so `in_site` (Beltsville-trained, FIPS 24033) is used when FieldVision offers it, and refused elsewhere |
| County totals shifted to a prior silage share | FieldVision calibrates on 50/50 balanced samples; real counties are about 1-20 % silage, so raw sums read high. The prior comes from Quick Stats years **before** the target year, keeping the USDA comparison independent |
| ...but only when the fields stand for the county | The prior is a county average. If the classified fields are under `MIN_COUNTY_COVERAGE` (50 %) of the county's harvested corn, such as the Beltsville farm in Prince George's, totals stay at the model's output and say why |
| Field calls use the model's own probability | A field is called silage or grain from `p_silage_model` at 0.5; the county-prior probability is only for totals, so Q&A never argues a call from it |
| Per-field results with area and evidence | Lets the agents explain a field (drop date, greenness before and after, residue) and lets the planner pick fields to visit |

Model decisions live in `config.py` (`STANDARD_MODEL`, `MODEL_DECISIONS`); change them there only.

## The adjustment

A field's probability `p` at the 50/50 training balance becomes, for a county with prior silage share `π`:

```
odds' = p / (1 - p) × [π / (1 - π)]        p' = odds' / (1 + odds')
```

The prior is, in order: the county's latest Census share before the target year; the state's mean survey share
over the three years before it (used when the county figure is withheld, as Prince George's is); the U.S. share
of about 7 %. Aggregates report `silage_acres` and `silage_acres_unadjusted`, the prior and its source, `prior_applied`,
`county_coverage` (field acres / the county's harvested corn) and the reason in `adjustment`. When the prior applies,
the planner counts abstained fields at the same prior.

## Layout

```
fieldvision_bridge/
  config.py       model choices, paths, constants
  prevalence.py   prior silage share from Quick Stats, the probability shift, the acreage range
  results.py      FieldVision's per-field result files, joined with field areas
  toolbox.py      load_team_toolbox, and FieldVisionToolbox: same tools as the stub, plus fields(), field(), planner_frame()
  field_qa.py     field-level analyst Q&A with FieldVision's evidence (Nemotron, numbers checked)
  rag.py          the team's rag-pipeline: make_rag_critic (critic=...) and make_report_search (reports=...)
  tests/          offline tests with a fake FieldVision toolbox: python -m fieldvision_agents.fieldvision_bridge.tests.run
  15_fieldvision_bridge.ipynb   demo on the Beltsville county (24033)
  16_rag_critic.ipynb           the real critic and NASS report search in a full run
```

## Use

```python
from fieldvision_agents.fieldvision_bridge import FieldVisionToolbox, ask_fields, make_rag_critic, make_report_search
from fieldvision_agents.react.supervisor import NemotronClient, run_county

toolbox = FieldVisionToolbox(critic=make_rag_critic(), reports=make_report_search())
result = run_county("24033", 2022, toolbox, NemotronClient())
ask_fields(toolbox, "24033", 2022, result, "Which fields abstained, and why?")
```

Needs the Parquet reader installed once into FieldVision's private folder:
`pip install --target /workspace/FieldVision/.deps pyarrow`.

## Limits

- Beltsville is 44 field-years on one farm; one field moves accuracy about 2.3 points.
- The prior is a county or state average; a county whose silage share changed a lot recently will be shifted wrongly.
- No radar model exists, so `radar_first` runs the same model as `standard`.
- The county-level check (Frederick 19.6 % vs Queen Anne's 1.3 % silage, Census 2022) is the real test of totals.
