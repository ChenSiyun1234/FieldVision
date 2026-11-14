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
data/        pointers to where data lives (no large files in git)
notebooks/   exploration and experiments
src/         reusable code: data access, features, models, evaluation
app/         demo / map
docs/        resources, notes, decisions
```

## Working together

- `main` should always run. Work on a branch, open a PR, merge after a teammate takes a quick look.
- Keep imagery, rasters and model weights out of git. Put them on the shared drive or the Keeboh workspace and add a pointer in `data/README.md`.
- Clear heavy notebook outputs before committing.
- When you try something, write down what you ran and what you got in `docs/log.md`, including results that did not work. We will need it for the presentation.

## Resources

Datasets and papers the team has collected: [docs/resources.md](docs/resources.md)
