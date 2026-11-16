# fieldvision_agents: the Nemotron side of FieldVision

Agents that run, check and explain the FieldVision silage/grain model for a USDA NASS analyst, the
analyst app built on them, and the checks run along the way. Code computes every number; Nemotron
routes, checks and explains, and code verifies every number it writes. Paths live in one file,
`settings.py`, relative to the repository root.

## How it fits the architecture

```
Public data (Sentinel-2, CDL fields, Quick Stats)
        │
Offline lane (src/fieldvision, registry/)         Online lane, per county-season (this package)
  features → frozen models in registry/   ──►   react/ supervisor (Nemotron ReAct, audit log)
                                                    │  calls
                                                fieldvision_bridge/ tools: scout, route, features,
                                                    classify, aggregate  (code: the numbers)
                                                    │
                                                rag-pipeline/ critic: Quick Stats checks decide,
                                                    Nemotron explains with NASS report passages
                                                    │  one rerun at most, then accept or abstain
                                                reporter/ report + Q&A (every number checked)
                                                    │
                                                app/ console and CLI  ──►  NASS analyst
```

## Run it

From the repository root, once:

```bash
uv venv && uv pip install -e ".[agents]"        # or: python -m venv .venv && pip install -e ".[agents]"
```

Then:

```bash
fieldvision doctor                    # what this machine can run
fieldvision test                      # every offline test, no key, no network
fieldvision run 24033 2022 --report   # Beltsville 2022 through the agent, with the county report
fieldvision serve                     # the app at http://127.0.0.1:8765: dairy owner or USDA analyst
```

Without `NVIDIA_API_KEY` (environment or `.env` at the repository root) everything runs offline: a
rule-based agent (`app/offline.py`) calls the same tools under the same guardrails, and reports and
answers are laid out from the run record. With the key, Nemotron Lightning runs the loop and Ultra
writes the report and explains the critic's verdict.

County data is not in git: put `<fips>/<year>/fields.geojson` and `timeseries.parquet` under
`data/counties/` (from the team's hand-over zip, `handoff/FieldVision_handoff_data.zip`, folder
`3_data/counties`), or point `FIELDVISION_COUNTY_DATA` at them. Runs (audit logs, reports, evaluation
and benchmark results, the LLM reply cache) go to `runs/`, also outside git.

Report search ranks the indexed NASS passages in `rag-pipeline/rag_index` by embedding when
`pip install -e ".[rag]"` is installed (sentence-transformers and torch, about 1 GB) and by keyword
otherwise.

## Modules

| Folder | What it does | Notebook (`notebooks/agents/`) |
|---|---|---|
| `react/` | Nemotron supervisor: a ReAct loop over the county tools, with rules enforced in code (step order, one rerun, critic before accept/abstain, every number grounded), an audit log, and the optional `recommend_sources` step | `07_supervisor_demo` |
| `fieldvision_bridge/` | The tools the agent calls: loads the FieldVision toolbox, puts the final module (`module_v2`, `eu2_logistic_v1`) behind "standard", county totals, field evidence, the RAG critic and NASS report search, field Q&A | `15_fieldvision_bridge`, `16_rag_critic` |
| `reporter/` | County report from the audit log (Ultra writes, Ultra edits, code checks the numbers) and evidence Q&A | `08_reporter_demo` |
| `app/` | The app: `fieldvision` command line and a web page with two roles, map first, with Ask the agent in the corner, PDF download of any result and light/dark themes. **Dairy owner**: chat with the silage finder (`dairy.py`, over `recommender/`) and a map of standing corn they could buy. **USDA analyst**: chat with every tool here (`chat.py`), the run console, and a dashboard of USDA supply, stocks and price beside our silage estimates and the price what-if (`dashboard.py`). Offline agent and router when there is no key (`offline.py`) | — |
| `analyst_screen/` | The same flow as ipywidgets, for notebooks | `09_analyst_screen` |
| `evaluation/` | Scenario and Q&A checks of the agent | `10_evaluation` |
| `benchmark/` | Nemotron models compared on the agent tasks, with the limitations found | `14_benchmark` |
| `planner/` | Which fields to visit to shrink the county range; supply impact in bushels | `13_planner` |
| `vision/` | Nemotron vision check on before/after Sentinel-2 chips; extra evidence, never the label | `11_vision_check` |
| `county_check/` | Every corn field of one county against the Census (Frederick 2022) | `18_frederick_check` |
| `states/` | Four-state 2025 check (PA, WI, IA, NE) against the USDA survey and harvest progress | `01_four_states` |
| `tempdiff/` | TempDiff-SMID (Agronomy 2025) reproduced and scored once on Beltsville: 22 of 44, below always-grain | `17_tempdiff` |
| `market/` | Corn price models and a what-if for silage acres; read `market/README.md` before quoting a number | `19_price_impact`, `22_stock_model_prediction` |
| `gap/` | Fields the model could not judge: the cloud gap each sits in, vision on before/after chips, checked reasons, and the agent's `inspect_gaps` rule (warns at 20% or more unjudged and gives the range up to "all unjudged are silage"). Guide: `gap/README.md` | — |
| `watch/` | Replays a season in 5-day steps using only images taken up to each date, to show how early a county settles; can poll the catalog for new scenes. Guide: `watch/README.md` | — |
| `validation/` | `gap_checks`: the known gaps measured against results on disk (writes `handoff/results/agents/gap_checks.md`) | — |

`21_final_model_everywhere.ipynb` refreshes the four states, Frederick and the price scenario with the
final model in one go. The notebooks were written in the Keeboh workspace; their setup cells now use
`fieldvision_agents.settings.REPO`, and their outputs were cleared.

Also in the repository: `rag-pipeline/` (the critic's checks, Quick Stats and the NASS report index)
and `recommender/` (silage sources for dairies, `recommend_sources`), each with its own README and tests.

## Tests

`fieldvision test` runs every module's offline tests with fakes. The team folders have their own:

```bash
python rag-pipeline/tests/smoke_test.py
python rag-pipeline/tests/test_critic_toolbox.py
python recommender/tests/test_silage_match.py
python recommender/tests/test_supervisor_recommend.py
```

Model estimates throughout are not NASS statistics; the price models are research estimates, not
trading advice.
