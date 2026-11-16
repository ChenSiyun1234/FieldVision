# notebooks

`agents/` holds one notebook per agent module; each starts from `fieldvision_agents.settings.REPO`, and outputs
are cleared. The code they call lives in `src/fieldvision_agents/`; the notebooks show it step by step.

| Notebook | Module |
|---|---|
| `01_four_states` | `states/`: four-state 2025 sample against the USDA survey |
| `07_supervisor_demo` | `react/`: the Nemotron supervisor on one county |
| `08_reporter_demo` | `reporter/`: county report and Q&A |
| `09_analyst_screen` | `analyst_screen/`: the notebook version of the analyst screen |
| `10_evaluation` | `evaluation/`: scenario and Q&A checks (needs `NVIDIA_API_KEY`) |
| `11_vision_check` | `vision/`: Nemotron vision on before/after image chips |
| `13_planner` | `planner/`: which fields to visit |
| `14_benchmark` | `benchmark/`: Nemotron models compared (needs `NVIDIA_API_KEY`) |
| `15_fieldvision_bridge` | `fieldvision_bridge/`: the tools the agent calls |
| `16_rag_critic` | `rag-pipeline/` through the bridge: the critic |
| `17_tempdiff` | `tempdiff/`: a published method reproduced and scored once |
| `18_frederick_check` | `county_check/`: every Frederick field against the Census |
| `19_price_impact`, `22_stock_model_prediction` | `market/`: price models (read `market/README.md` first) |
| `21_final_model_everywhere` | Refreshes the four states, Frederick and the price scenario with the final model |
