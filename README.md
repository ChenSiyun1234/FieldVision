# FieldVision

Hack the Harvest 2026 — can public satellite data tell whether a corn field was cut early for **silage** or left to mature for **grain**?

| | |
|---|---|
| Build window | Sat Nov 14, 10:30 AM → Sun Nov 15, 9:00 AM PT |
| Submit | In Keeboh by Sun 9:00 AM. Late = not judged |
| Present | 5 minutes, no Q&A, no video |
| Data | Publicly available geospatial data only |

Rules and rubric: <https://hack-the-harvest.devpost.com/rules>

## What the judges score

| Criterion | Weight | What they look for |
|---|---|---|
| Technical quality | 20% | A trained model with precision, recall, false positive rate and response time measured on evaluation data |
| Innovation | 20% | A new data source or harvest signal that improves predictions |
| Feasibility | 20% | Does it actually separate silage-like from grain-like fields, and could it be developed further |
| Presentation | 40% | Clear demo, clear process, clear value |

Presentation is worth as much as technical quality and innovation combined, so the demo and the story get real time, not the last hour.

## Layout

```
src/fieldvision/          the model module: features, frozen models, toolbox, county totals
src/fieldvision_agents/   the Nemotron agents and the analyst app (command line + web console)
registry/                 frozen, versioned models the toolbox loads read-only
rag-pipeline/             the critic: Quick Stats checks and the NASS report index
recommender/              silage sources for dairies (the agent's optional recommend_sources step)
app/                      the field map demo (Beltsville, per-field evidence)
notebooks/                exploration and experiments; notebooks/agents/ for the agent side
data/                     pointers to where data lives (no large files in git)
runs/                     agent audit logs, reports and evaluations (not in git)
docs/, handoff/           notes, decisions, results and the hand-over
```

## Run the analyst app

```bash
uv venv && uv pip install -e ".[agents]"
fieldvision doctor                   # what this machine can run
fieldvision run 24033 2022 --report  # one county-season through the agent
fieldvision serve                    # the app at http://127.0.0.1:8765: dairy owner or USDA analyst
```

County data goes in `data/counties/<fips>/<year>/` (not in git). Without `NVIDIA_API_KEY` in `.env` the
app runs offline with a rule-based agent under the same guardrails. Details:
[src/fieldvision_agents/README.md](src/fieldvision_agents/README.md).

## Working together

- `main` should always run. Work on a branch, open a PR, merge after a teammate takes a quick look.
- Keep imagery, rasters and model weights out of git. Put them on the shared drive or the Keeboh workspace and add a pointer in `data/README.md`.
- Clear heavy notebook outputs before committing.
- When you try something, write down what you ran and what you got in `docs/log.md`, including results that did not work. We will need it for the presentation.

## Resources

Datasets and papers the team has collected: [docs/resources.md](docs/resources.md)
