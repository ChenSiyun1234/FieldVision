# FieldVision

**Was this corn field chopped for silage or combined for grain?** FieldVision answers that for every corn field in
a county from free public satellite data, adds the fields up to county and state acres, and checks the result
against USDA NASS. Built for Hack the Harvest 2026 (USDA NASS challenge).

Every figure the system produces is a **model estimate, not a NASS statistic**.

- [What it does](#what-it-does)
- [Results](#results)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Repository map](#repository-map)
- [Data](#data)
- [Reproduce the main results](#reproduce-the-main-results)
- [Tests](#tests)
- [Known limits](#known-limits)
- [Working together](#working-together)

## What it does

| For | What they get |
|---|---|
| A USDA NASS analyst | County silage and grain acres with a range, the reason behind each field's call, a check against USDA history, a written report and cited answers, a US map from states down to fields, and a dashboard of corn supply, stocks and price |
| A dairy owner | Standing corn near the dairy that could still be bought for silage: tons, road miles and a delivered price range, on a map |

Code computes every number. Nemotron (NVIDIA's hosted models) decides the order of steps, checks the result and
explains it, and code checks every number it writes.

## Results

All results below are from runs logged before they were opened. The full record is in
[`docs/RESULTS.md`](docs/RESULTS.md) and [`handoff/results/REPORT.md`](handoff/results/REPORT.md).

**Field calls, 44 labelled US field-years (Beltsville, MD, 2021-23), scored once.** The model was trained only on
8,000 labelled European fields.

| Model | Right | Silage precision | Silage recall | False positive rate | ROC-AUC |
|---|---|---|---|---|---|
| `eu2_logistic_v1` (final) | 42 of 44 | 0.917 | 0.917 | 0.031 | 0.951 |
| Always grain | 32 of 44 | – | 0 | 0 | 0.5 |

**States, 2025, against the organisers' NASS figures** (silage share of corn harvested; clean tests run once under
rules written first):

| State | Ours | NASS 2025 | Miss |
|---|---|---|---|
| Maryland | 4.75% [3.4, 6.3] | 8.9% | −4.2 |
| Wisconsin | 16.6% [13.8, 19.5] | 21.6% | −5.0 |
| Pennsylvania | 15.2% [12.9, 17.5] | 32.8% | −17.6 |

**Counties.** Lancaster PA 2023 with Sentinel-2 and Landsat: 26.6% against 27.4% in the 2022 Census. Frederick MD
2022: 13.2% against 19.6% (used in tuning). Every place, test or not, is in [`docs/RESULTS.md`](docs/RESULTS.md).

**What the misses mean.** The model reads silage low in dairy country. Three causes, from
[`handoff/results/hima/COMBINED_GAP_FINDINGS.md`](handoff/results/hima/COMBINED_GAP_FINDINGS.md): harvests hidden by
cloud (the unjudged fields there are mostly silage), northern silage cut after the crop yellows and read as grain,
and a partly seen county whose clear fields are not typical. USDA's last official share alone misses the 2025 state
samples by 2.5 points on average, better than any satellite variant; say so beside any satellite figure.

**Speed.** Classifying a county takes about 14 ms per field on a laptop CPU (Gratiot MI, 3,463 fields, 47 s);
an agent run takes 1 to 2 minutes and 70,000 to 100,000 Nemotron tokens. No GPU is needed.

## Quick start

Python 3.10 or newer.

```bash
git clone https://github.com/ChenSiyun1234/FieldVision && cd FieldVision
uv venv && uv pip install -e ".[agents]"        # or: python -m venv .venv && .venv/bin/pip install -e ".[agents]"
cp .env.example .env                            # optional: add NVIDIA_API_KEY to use Nemotron
```

| Command | What it does |
|---|---|
| `fieldvision doctor` | What this machine can run (data, key, optional packages) |
| `fieldvision test` | Every offline test of the agent side (no key, no network) |
| `fieldvision counties` | County-seasons with data on disk |
| `fieldvision run 24033 2022 --report` | One county-season through the agent, with the written report |
| `fieldvision run 24021 2022 --offline` | The same with the rule-based agent, no language model |
| `fieldvision serve` | The app at <http://127.0.0.1:8765> (dairy owner or USDA analyst) |
| `pytest -q tests` | The model package's tests |

Without `NVIDIA_API_KEY` everything still runs: a rule-based agent calls the same tools under the same guardrails.
County data is not in git; see [Data](#data).

In the app: the map is the main view, history and settings are on the left, **Ask the agent** sits in the corner,
every result has **Download PDF**, and **Light/Dark** switches the theme.

## How it works

![Architecture](handoff/figures/architecture.png)

**Offline lane (trained once).**

1. **Features** (`src/fieldvision/features/`): each field's Sentinel-2 series, plus Landsat 8/9 where Sentinel-2
   is clouded, becomes harvest signals: when greenness dropped, how green and wet the crop was just before, bare soil
   or residue after.
2. **Models** (`src/fieldvision/model/`): timing rule, timing and moisture rule, a 6-feature logistic, LightGBM and a
   composite, chosen by leave-one-country-out on European labels.
3. **Registry** (`registry/`): the chosen models frozen in versioned folders and loaded read-only.

**Online lane (per county-season).**

1. **Supervisor** (`src/fieldvision_agents/react/`): Nemotron in a reason-then-act loop with the rules enforced in
   code (step order, one rerun, the critic before accept or abstain, no county accepted with no field judged) and an
   audit log of every step.
2. **Tools** (`src/fieldvision/toolbox.py` via `src/fieldvision_agents/fieldvision_bridge/`): scout the imagery,
   compute features, classify each field (or abstain), add up to county acres with a range.
3. **Critic** (`rag-pipeline/`): checks the county share against Quick Stats history; Nemotron explains the verdict
   with passages from NASS reports. It may send the run back once.
4. **Gap check** (`src/fieldvision_agents/gap/`): when 20% or more of the corn is unjudged, warns that the estimate
   likely reads low and gives the range up to "all unjudged fields are silage".
5. **Reporter** (`src/fieldvision_agents/reporter/`): the county report and Q&A, every number checked against the
   tool results.
6. **App** (`src/fieldvision_agents/app/`): command line and web app for both roles; the dairy finder runs on the
   recommender (`recommender/`).

## Repository map

```
FieldVision/
├── README.md                    you are here
├── HANDOFF.md                   hand-over notes from the model team
├── pyproject.toml               package, extras ([agents], [rag], [foundation]) and the `fieldvision` command
├── src/
│   ├── fieldvision/             the model: extraction, features, models, toolbox, county and state totals
│   │   ├── s2/  s1/  ls/        Sentinel-2, Sentinel-1 and Landsat per-field extractors
│   │   ├── features/            harvest features and the Landsat merge
│   │   ├── model/               training, metrics, splits, state estimate
│   │   ├── beltsville/          in-site models fitted on the labelled fields (comparison only)
│   │   ├── labels/              European parcel declaration labels
│   │   ├── anomaly/ foundation/ stream/   experiments (anomaly view, Presto, pixel-chip sequence model)
│   │   ├── app/                 data export for the field-map demo in app/
│   │   └── toolbox.py           the tools the agent calls
│   └── fieldvision_agents/      the agents and the analyst app (guide: src/fieldvision_agents/README.md)
├── registry/                    frozen models, one folder per version (guide: registry/README.md)
├── rag-pipeline/                critic: Quick Stats checks, NASS report index (guide: rag-pipeline/README.md)
├── recommender/                 silage sources for dairies (guide: recommender/README.md)
├── app/                         field-map demo and the vendored map libraries the analyst app serves
├── scripts/                     one-off and pipeline scripts, grouped in scripts/README.md
├── colab/                       Colab extraction notebooks and bundles (guide: colab/README.md)
├── notebooks/agents/            notebooks behind each agent module (guide: notebooks/README.md)
├── tests/                       model package tests (guide: tests/README.md)
├── docs/                        method, decisions, experiment log, results (index: docs/README.md)
├── handoff/                     deck, figures and result files for the judges (guide: handoff/README.md)
├── results/                     model-side result notes (results/README.md)
└── data/                        pointers only; county data lives outside git (data/README.md)
```

**Where to start, by question:**

| I want to… | Read |
|---|---|
| See every result and whether it is a clean test | [`docs/RESULTS.md`](docs/RESULTS.md) |
| Understand the model and its features | [`docs/module.md`](docs/module.md), [`docs/model.md`](docs/model.md), [`docs/features.md`](docs/features.md) |
| Understand the agents and the app | [`src/fieldvision_agents/README.md`](src/fieldvision_agents/README.md) |
| Know why the state figures read low | [`handoff/results/hima/COMBINED_GAP_FINDINGS.md`](handoff/results/hima/COMBINED_GAP_FINDINGS.md) |
| Extract a new county | [`docs/s2-extractor.md`](docs/s2-extractor.md), [`colab/README.md`](colab/README.md), `scripts/colab_to_county.py` |
| Use the toolbox from Python | [`docs/toolbox.md`](docs/toolbox.md) |
| See what was tried, including what failed | [`docs/log.md`](docs/log.md) |
| Present it | [`handoff/presentation/FieldVision_results.pdf`](handoff/presentation/FieldVision_results.pdf) |

## Data

All public. Nothing large is in git.

| Data | Source | Used for |
|---|---|---|
| Sentinel-2 L2A | Microsoft Planetary Computer, Element 84 Earth Search, Google Earth Engine | per-field time series |
| Landsat 8/9 Collection 2 L2 | Planetary Computer (`landsat-c2-l2`) | filling Sentinel-2 cloud gaps |
| Sentinel-1 RTC | Planetary Computer | radar companion (extractor built; not in the final runs) |
| Crop Sequence Boundaries, Cropland Data Layer | USDA NASS | corn field outlines and crop history |
| Quick Stats | USDA NASS (`rag-pipeline/dataset/quickstats/`) | Census and survey history for the critic and the map |
| NASS 2025 state acres | USDA NASS Quick Stats (`handoff/results/nass_2025_state_targets.csv`) | the organisers' evaluation figures |
| NASS reports | USDA NASS (`rag-pipeline/rag_index`) | passages the critic cites |
| European maize declarations | Austria, France, the Netherlands (`src/fieldvision/labels/`) | training labels |
| Dairy locations | Maryland dairy permits, EPA ECHO | the dairy finder |

County data for the agent goes in `data/counties/<fips>/<year>/` (`fields.geojson`, `timeseries.parquet`, and
`timeseries_landsat.parquet` where Landsat was pulled), or set `FIELDVISION_COUNTY_DATA`. Convert a Colab extraction
with `python scripts/colab_to_county.py <extract_dir> <year>`; pull Landsat with
`python -m fieldvision.ls.extract --fields data/counties/<fips>/<year>/fields.geojson --year <year> --out <dir>`.

## Reproduce the main results

| Result | How |
|---|---|
| Beltsville scoring | `docs/model.md` (commands) and `handoff/results/REPORT.md` (the logged runs) |
| One county through the agent | `fieldvision run <fips> <year> --report` |
| Agent evaluation and model comparison | `notebooks/agents/10_evaluation.ipynb`, `14_benchmark.ipynb` (need `NVIDIA_API_KEY`) |
| Known gaps checked against the data | `python -m fieldvision_agents.validation.gap_checks` |
| Gap table and 5-day replay | `python -m fieldvision_agents.gap.gap <fips> <year> --data data/counties`, `python -m fieldvision_agents.watch.watch <fips> <year> --data data/counties` (guides in their folders) |
| Price models | `src/fieldvision_agents/market/README.md` before quoting any number |
| Many counties on a server | `python scripts/pod_batch_counties.py list --top 50`, then `run <extract_root> <year>` |

## Tests

```bash
fieldvision test        # agents, app, RAG bridge, gap and watch (106 tests, offline)
pytest -q tests         # model package (184 tests)
```

## Known limits

- Ground truth in the US is 44 field-years from one farm; everywhere else is checked against Census or survey totals.
- Silage reads low in dairy states (above). Closing it needs labelled northern fields, not a setting change.
- Under cloud a county can be only partly judged; the agent reports the range and warns at 20% unjudged.
- The price page is a labelled what-if, not a forecast: on clean tests no price model beats last year's price by much.
- Field data covers past seasons; the dairy view replays a season as of a date.

## Working together

- `main` is the model team's branch; `feature/agent-app` adds the agents and the app on top of it.
- Work on a branch and merge after a teammate looks. Keep imagery, rasters and weights out of git.
- Write what you ran and what you got in `docs/log.md`, including what did not work.
