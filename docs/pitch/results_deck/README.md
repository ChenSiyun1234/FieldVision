# Results deck

`handoff/presentation/FieldVision_results.pptx` (and the PDF beside it): 23 slides, the whole project
from the field signal to the app, in the same dark "harvest" theme as the app. Speaker notes on every slide.

How it was made, so it can be redrawn:

| File | What it does |
|---|---|
| `figs.py` | Charts and maps from the repository's results (field test, country hold-out, counties, states, price check, what-if, planner, US county map, Frederick field map). Run with the project's Python from the repo root: `python docs/pitch/results_deck/figs.py <out_dir>` (needs `counties.geojson`/`states.geojson` from `topo2geo.js` next to it) |
| `topo2geo.js` | Converts the vendored `us-atlas` county shapes to GeoJSON for `figs.py` |
| `shots.js` | App screenshots with `puppeteer-core` driving an installed Chrome, against `fieldvision serve` |
| `build.js` | The deck itself (pptxgenjs); `node build.js out.pptx` with the figures, screenshots and team images beside it |
| `figs/` | The charts and maps as used in the deck |

Every number on the slides comes from `handoff/results/`, `runs/` or the app's own endpoints; the team's
wording rules in `docs/pitch/limits.md` apply ("42 of 44" with "always grain gets 32"; county and state
totals are not called validated).
