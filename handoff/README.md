# handoff

What a judge or a new teammate needs, without running anything.

| Path | What it is |
|---|---|
| `presentation/FieldVision.pdf` | **The final presentation** (17 slides): problem, model, data and features, results against NASS, cloud gaps, the agents, impact, the price outlook, future work |
| `presentation/FieldVision_model_story.pdf` (`.pptx`) | The model team's deck: how the model was built and tested |
| `figures/results/` | The result slides of the final presentation as images (shown in the top-level README) |
| `figures/` | Other figures: agent architecture, roadmap, future scope, price outlook, field strips, county maps, test charts |
| `results/REPORT.md` | The Beltsville evaluation report: metrics with intervals, misclassified fields, every decision with its time |
| `results/ACCESS_LOG.md` | When each evaluation set was opened |
| `results/nass_2025_state_targets.csv` | The organisers' NASS 2025 state figures (corn for grain and silage, acres harvested) |
| `results/state_estimate.json` | Maryland 2025 state estimate |
| `results/rotation_check_md_2022.csv`, `landscape_groups_md_2022.csv`, `thinning_replay.csv` | Context checks behind the Maryland results |
| `results/agents/` | Agent-side results: four-state 2025 comparison, Frederick and Lancaster county checks, price impact, TempDiff reproduction, the known-gap checks (`gap_checks.md`) |
| `results/hima/` | Gap tables, the 5-day replay charts and `COMBINED_GAP_FINDINGS.md`: why silage reads low in dairy states |

Every figure is a model estimate, not a NASS statistic.
