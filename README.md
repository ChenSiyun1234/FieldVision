<p align="center">
  <img src="docs/readme/00_banner.png" alt="FieldVision: Grand Prize, 1st place, Hack the Harvest 2026, USDA NASS challenge" width="100%">
</p>

<p align="center">
  <img alt="Grand Prize" src="https://img.shields.io/badge/Grand%20Prize-1st%20place-D4AF37?style=for-the-badge">
  <img alt="Prize" src="https://img.shields.io/badge/Prize-%245%2C000-2F6B4F?style=for-the-badge">
  <img alt="Hack the Harvest 2026" src="https://img.shields.io/badge/Hack%20the%20Harvest-2026-2878D6?style=for-the-badge">
  <img alt="USDA NASS challenge" src="https://img.shields.io/badge/USDA%20NASS-challenge-143A2B?style=for-the-badge">
</p>

<p align="center">
  <img src="docs/readme/grand_prize_check.jpg" alt="The Grand Prize check made out to Field Vision, $5,000, Nov 15, 2026" width="62%"><br>
  <sub>The Grand Prize check, Nov 15, 2026. Hosted and organized by the Human Machine Collaboration Institute, with USDA, UC Davis, NVIDIA, Solidigm, FarmGPU and Keeboh. Team makai.</sub>
</p>

# FieldVision

**An explainable, satellite-driven corn harvest intelligence system.** Was this corn field chopped green for
**silage** or left to dry for **grain**? FieldVision answers that for every corn field from free public satellite
data, explains each call, adds the fields up to county and state acres, and checks them against USDA NASS.
Built for Hack the Harvest 2026 (USDA NASS challenge) by **Team makai**: Hetvi Bhadani, Hima Verma, Krishi Shah,
Priyanka R, Siyun Chen. **Grand Prize, 1st place.**

**Presentation:** [`handoff/presentation/FieldVision.pdf`](handoff/presentation/FieldVision.pdf) (17 slides).
Every figure the system produces is a **model estimate, not a NASS statistic**.

![Fields called grain or silage from Sentinel-2 pictures and greenness curves](handoff/figures/deck/strips_sheet.png)

**Results at a glance**

| Test | Result |
|---|---|
| 44 labelled US fields (Beltsville, MD), trained with zero US labels | **42 of 44 right**; precision and recall 91.7% |
| Clean tests against USDA (rules written first, run once) | **4 of 5 within 5 points**; Pennsylvania 2025 missed |
| Model alone, 7 states | average gap **2.0 points** |
| Model + USDA history (Texas, California, Colorado) | average gap **13.3 → 1.4 points** |
| Landsat 8/9 added for cloud gaps | hidden fields **217 → 136** |
| Sentinel-1 radar replay (France, 3-5 week optical gap) | recall **0.33 → 0.54** |
| Nemotron agents | 5/5 routing, 4/4 Q&A, **0 unsupported numbers** |

[Model](#the-classification-model) · [What it does](#what-it-does) · [Results](#results) · [Quick start](#quick-start) · [How it works](#how-it-works) · [Repository map](#repository-map) · [Data](#data) · [Reproduce](#reproduce-the-main-results) · [Tests](#tests) · [Known limits](#known-limits)

## The classification model

I built the part that makes the field-by-field call and turns it into county and state totals: the data, the training set, the model, how it was validated, and the toolbox the rest of the system calls. This section describes the model alone. The Bayesian update with USDA history, the Nemotron agents and the app are covered further down. The numbers here come from [`docs/RESULTS.md`](docs/RESULTS.md) and [`handoff/results/ACCESS_LOG.md`](handoff/results/ACCESS_LOG.md), and every figure is drawn by [`scripts/make_readme_figures.py`](scripts/make_readme_figures.py) from those files and from `git log`.

![From a corn parcel to a silage-area estimate](docs/readme/01_pipeline.png)

| | What I did | Why |
|---|---|---|
| **Data** | A Sentinel-2 time series for every USDA Crop Sequence Boundaries field, extracted on Google Colab the same way everywhere. Landsat 8/9 came later as a second satellite. | Cloud gaps stop hiding the harvest, and a result in one county means the same thing in another. |
| **Training set** | 8,000 labelled fields from Austria, France and the Netherlands. The model never sees a US label. | Only 44 US field-years carry labels: too few to train on and too valuable to spend. |
| **Model** | A logistic regression on six crop-timing features: greenness and moisture before the cut, how fast the field dries, days from peak to cut, and the stubble afterwards. I also tried boosted trees on 135 features, a composite, a timing rule and a sequence model. | I picked by a rule written down first: best mean ROC-AUC with a whole country held out. A gap smaller than the spread across held-out groups goes to the simpler model. |
| **Evidence tiers** | A call counts toward a total only when two clear looks around the cut are at most 12 days apart. Fields with wide or no evidence are reported separately, with the area the model could not see. | A total that hides what it did not see would look better and mean less. |
| **Totals** | Field calls add up to county and state silage shares, area-weighted, with a sampling range. | The range shows how far each share can be trusted. |
| **Validation** | For every test I wrote the rule, the pass mark and my own prediction before opening the data, and ran it once. | Failures stay in the record: the first county method failed on Kent County, and Pennsylvania 2025 failed the final checks. |
| **Hand-over** | One Python interface, `fieldvision.toolbox.Toolbox`, with [`HANDOFF.md`](HANDOFF.md), a results document and a slide deck. | The rest of the system calls the model through it. |

| | |
|---|---|
| ![The six features and their weights](docs/readme/02_features.png) | ![Transfer to a new country and to labelled US fields](docs/readme/03_validation.png) |
| ![Confusion matrix on 44 labelled US field-years](docs/readme/04_confusion.png) | ![Five clean checks against USDA](docs/readme/05_nass_comparison.png) |

**42 of 44 fields.** On the organisers' 44 labelled Beltsville field-years, scored once, the model gets 42 right: 11 of 12 silage fields found, one false alarm. A rule that always says "grain" gets 32.

**Four of five checks within 5 points.** Michigan 2025, New York 2025, Lancaster County 2023 and Wisconsin 2025 land within 5 points of USDA. Wisconsin passes by 0.02 points and its own guard flags it as unreliable. Pennsylvania 2025 misses by 17.6 points. I kept that in the README because the fix is not a tuning knob: dairy country with long cloud gaps at harvest hides exactly the fields that matter.

![Commits across the build window](docs/readme/07_timeline.png)

**22.5 hours.** Of my 271 commits on `main`, 266 fall inside the build window (Nov 14 10:30 to Nov 15 09:00, Pacific time). The labelled milestones come from the evaluation access log, which every scoring and every rule change was written into before the run.

<details>
<summary><b>More figures</b>: Landsat as a second satellite, and why the evidence tiers exist</summary>

![Landsat as a second satellite](docs/readme/06_landsat.png)

![Why the evidence tiers exist](docs/readme/08_evidence_tiers.png)

</details>

## What it does

| For | What they get |
|---|---|
| A USDA NASS analyst | County silage and grain acres with a range, the reason behind each field's call, a check against USDA history, a written report and cited answers, a US map from states down to fields, and a dashboard of corn supply, stocks and price |
| A dairy owner | Standing corn near the dairy that could still be bought for silage: tons, road miles and a delivered price range, on a map |

Code computes every number. Nemotron (NVIDIA's hosted models) decides the order of steps, checks the result and
explains it, and code checks every number it writes.

## Results

The figures below come from the logged runs and match the presentation. The full record, with which results were clean tests and
when each was logged, is in [`docs/RESULTS.md`](docs/RESULTS.md) and [`handoff/results/REPORT.md`](handoff/results/REPORT.md).

Open any line for the tables.

<details>
<summary><b>Field level: 44 labelled US field-years (Beltsville, MD, 2021-23)</b>: 42 of 44 right, trained with zero US labels</summary>

Trained with **zero US labels** (8,000 labelled European fields from three countries), scored once.

| | Predicted silage | Predicted grain |
|---|---|---|
| **Truly silage** | 11 | 1 |
| **Truly grain** | 1 | 31 |

| Accuracy | Precision | Recall | F1 | False positive rate | Speed |
|---|---|---|---|---|---|
| 95.5% (42 of 44) | 91.7% | 91.7% | 91.7% | 3.1% | 14 ms per field |

For comparison: a larger gradient-boosted model gets 37 of 44, and always saying "grain" gets 32 of 44.

</details>

<details>
<summary><b>Clean tests against USDA</b>: 4 of 5 within 5 points; Pennsylvania 2025 missed</summary>

Each place was run once, under rules written down before it was opened, with Sentinel-2 and Landsat 8/9.
Silage share of harvested corn acres; ranges in brackets.

| Place | FieldVision | USDA | Gap (points) | Logged result |
|---|---|---|---|---|
| Michigan 2025 (state) | 14.4% [10.8, 18.0] | 14.7% (NASS 2025) | −0.3 | pass |
| Lancaster County, PA 2023 | 26.6% | 27.4% (Census 2022) | −0.8 | pass |
| New York 2025 (state) | 48.7% [43.3, 54.2] | 53.2% (NASS 2025) | −4.5 | pass |
| Wisconsin 2025 (state) | 16.6% [13.8, 19.5] | 21.6% (NASS 2025) | −5.0 | pass, by 0.02 point |
| Pennsylvania 2025 (state) | 15.2% [12.9, 17.5] | 32.8% (NASS 2025) | −17.6 | **missed** |

**Four of these five are within 5 points of USDA; Pennsylvania is missed.** Earlier clean tests: Maryland 2025,
4.75% [3.4, 6.3] against 8.9% (4.2 low); Branch County MI 2022, 2.1% against 1.8% (pass); Gratiot County MI
2022, 30.0% against 24.2% (pass under its logged rule).

</details>

<details>
<summary><b>Earlier state samples: model alone</b>: 7 states, average gap 2.0 points</summary>

300 fields per state, Sentinel-2, from the runs used while building the method (not clean tests). Within 5
points of USDA in 7 states:

| State | NASS 2025 | Model alone | Gap (points) |
|---|---|---|---|
| Iowa | 1.9% | 6.2% | 4.3 |
| Minnesota | 4.1% | 6.4% | 2.3 |
| Kansas | 3.4% | 1.3% | 2.1 |
| Michigan | 14.7% | 12.7% | 2.0 |
| South Dakota | 5.8% | 4.3% | 1.5 |
| Nebraska | 2.1% | 0.7% | 1.4 |
| Wisconsin | 21.6% | 22.2% | 0.6 |
| **Average** | | | **2.0** |

</details>

<details>
<summary><b>Model + USDA history: where images mislead, USDA history steers</b>: average gap 13.3 → 1.4 points</summary>

Each field's silage probability *p* is adjusted with the state's recent USDA silage share π (2021, 2023, 2024) by
Bayes' rule, p′ = p·π ÷ (p·π + (1 − p)(1 − π)), and the state estimate is the area-weighted mean of p′. On the
earlier samples (not clean tests):

| State | NASS 2025 | Model alone | Model + USDA history | Gap |
|---|---|---|---|---|
| Texas | 10.3% | 34.1% | 10.2% | 23.8 → 0.1 |
| California | 83.1% | 75.0% | 86.0% | 8.1 → 2.9 |
| Colorado | 14.0% | 21.9% | 12.7% | 7.9 → 1.3 |
| **Average** | | | | **13.3 → 1.4** |

Counting the fields hidden by cloud by the model's own call brings two counties seen while tuning close to the
Census: Frederick County MD 2022, 19.2% against 19.6%, and Wyoming County NY 2022, 64.9% against 66.7%.

</details>

<details>
<summary><b>Cloud gaps: Landsat 8/9 as a second satellite</b>: hidden fields 217 → 136</summary>

| | Sentinel-2 only | + Landsat 8/9 |
|---|---|---|
| Fields hidden by clouds at harvest (8 states) | 217 | 136 |
| Model-alone miss, in points | 3.5 | 2.8 |
| Iowa silage share (NASS 1.9%) | 6.2% | 2.5% |

</details>

<details>
<summary><b>Cloud gaps: Sentinel-1 radar (pilot)</b>: recall 0.33 → 0.54 in a replay</summary>

Radar sees through cloud, so it can date a harvest the optical pictures missed. In a replay on labelled fields
(France), where a 3-5 week optical gap was cut out around the harvest, a radar harvest date recovered about half
the lost recall:

| | Optical, with the gap | Optical + radar date | Optical, no gap |
|---|---|---|---|
| Silage recall (France) | 0.33 | 0.54 | 0.65 |

The cost: 3-6 points of extra false silage, because residue is then read from older optical pictures. Radar is
not in the final runs. The extractor is built and tested (`src/fieldvision/s1/`, `tests/test_s1.py`), and
the Colab notebooks for Kent County MD 2022 and the Maryland 2025 sample are in
[`colab/extraction/s1_24029_2022/`](colab/extraction/s1_24029_2022/README.md) and
`colab/extraction/us_24_2025_s1/`. Over Marathon County WI, no Sentinel-1 scene falls in the cloudy weeks, so
radar cannot close that gap. The replay's table (`radar_replay.csv`) is not in this repository; the figures above
are as logged in [`docs/RESULTS.md`](docs/RESULTS.md).

</details>

<details>
<summary><b>Corn price scenario</b>: what silage acres do to the grain market</summary>

A ridge regression on USDA stocks-to-use, last year's price and yield (`src/fieldvision_agents/market/`) sizes
what silage acres do to the grain market: 0.8 points more silage nationally is about 131 million fewer bushels
of grain a year and 2 to 2.5% on price. Its 2026-28 path ($3.42, $3.06, $2.77) rests on acres, yield and use
written into the code, and USDA's September 2026 price is $4.81. Use it to size silage's effect, not to call
next year's price. The agents reach it through the `silage_what_if` tool.

</details>

<details>
<summary><b>Agents (Nemotron)</b>: 5/5 routing, 4/4 Q&A, 0 unsupported numbers</summary>

| Correct routing decisions | Evidence Q&A checks | Unsupported numbers after guardrails | County run time |
|---|---|---|---|
| 5 / 5 | 4 / 4 | 0 | Bolt 6.9 s · Lightning 15.3 s · Ultra 37.6 s |

Failures found and fixed: empty replies from reasoning tokens (larger per-call output budget); a tool result that
tried to steer the agent (next steps enforced in code); numbers added by the writer in 2 of 8 reasons (every number
checked, with a template fallback); vision correct on 15 of 28 at 10 m (used as context, never as the label).

</details>

<details>
<summary><b>Where it still reads low</b>: Pennsylvania 2025, and what closes it</summary>

Pennsylvania 2025 is the clean test that missed: 15.2% against NASS's 32.8%. Dairy country with long cloud gaps
at harvest reads silage low, because the fields the model cannot judge there are mostly silage, and northern silage
cut after the crop yellows can look like grain. Why, and what closes it:
[`handoff/results/hima/COMBINED_GAP_FINDINGS.md`](handoff/results/hima/COMBINED_GAP_FINDINGS.md).

</details>

<details>
<summary><b>Where each result lives</b>: the file behind every number</summary>

| Result | File |
|---|---|
| Field-level test (Beltsville) | [`handoff/results/REPORT.md`](handoff/results/REPORT.md), per-year calls in `results/MD_202*.jsonl` |
| Clean tests and every logged result | [`docs/RESULTS.md`](docs/RESULTS.md) |
| State samples, 2025 | `handoff/results/agents/states_2025_comparison.json`, `handoff/results/nass_2025_state_targets.csv` |
| County checks (Frederick, Lancaster) | `handoff/results/agents/*_county_check.json` |
| Gap checks and harvest pace | `handoff/results/agents/gap_checks.md`, `handoff/results/hima/` |
| Agent benchmark and evaluation | `src/fieldvision_agents/benchmark/`, `src/fieldvision_agents/evaluation/` |
| Price scenario | `handoff/results/agents/price_impact.json` |
| Numbers quoted in the talk | [`docs/pitch/pitch_numbers.py`](docs/pitch/pitch_numbers.py) |

</details>

## Results slides

<details>
<summary>Show the eight slides</summary>

From the final presentation; the images are in [`handoff/figures/results/`](handoff/figures/results/).

| | |
|---|---|
| ![Model alone, 7 states](handoff/figures/results/05_model_alone.jpg) | ![Model + USDA history](handoff/figures/results/06_model_plus_usda_history.jpg) |
| ![Counties against USDA](handoff/figures/results/07_counties_vs_usda.jpg) | ![Cloud gaps](handoff/figures/results/08_cloud_gaps.jpg) |
| ![Field-level test](handoff/figures/results/09_field_level_test.jpg) | ![Agent architecture](handoff/figures/results/10_agent_architecture.jpg) |
| ![Agent reliability](handoff/figures/results/11_agent_reliability.jpg) | ![Price scenario](handoff/figures/results/14_price_scenario.jpg) |

</details>

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

<details>
<summary><b>The two lanes, step by step</b>: offline (trained once) and online (per county-season)</summary>

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

</details>

<details>
<summary><b>The agentic loop (NVIDIA Nemotron)</b></summary>

![Agent architecture](handoff/figures/results/10_agent_architecture.jpg)

An analyst or dairy owner asks about a county. **Nemotron 3.5 Lightning** is the supervisor: in a
reason-then-act loop it picks the next tool, reads the result and decides what comes next. Code makes every
number; Nemotron decides and explains.

| Step | Tool | What it does |
|---|---|---|
| 1 | `scout` · `extract_features` | What imagery exists per field (Sentinel-2 + Landsat 8/9), then the harvest features |
| 2 | `classify` · `aggregate` | The frozen model calls each field or abstains; fields add up to county acres with a range |
| 3 | `run_critic` | Checks the county share against USDA history; retrieves NASS report passages only when the check fails (agentic RAG) |
| 4 | `inspect_gaps` | Finds fields hidden by cloud at harvest, scores them by harvest timing with Bayes' rule, warns "likely low" at 20% unjudged |
| 5 | `recommend_sources` | Silage Match: dairies short on feed matched to nearby standing corn |
| 6 | report · Q&A | The written report and answers, every number traced to a tool result |

| Model | Role |
|---|---|
| Nemotron 3.5 Lightning | Supervisor: picks each step and calls the tools |
| Nemotron 3 Ultra | Explains the critic's verdict, writes and edits the report, writes a reason for each cloud-hidden field |
| Nemotron 3.5 Super VL | Looks at a field's last clear image before a cloud gap and the first one after it |
| Nemotron 3.5 Bolt | Benchmarked for fast Q&A (6.9 s per county run) |

**Guardrails in code:** fixed step order, one rerun, the critic before accept or abstain, abstain instead of
guessing, and every number Nemotron writes checked against tool results. The loop returns county acres, the
range, and the reason for each field. Between seasons the **5-day watch loop** (`src/fieldvision_agents/watch/`)
reruns the agent only when new clear imagery moves the estimate.

</details>

<details>
<summary><b>What we learned from each Nemotron model</b></summary>

Measured in our benchmark (`src/fieldvision_agents/benchmark/`, `notebooks/agents/14_benchmark.ipynb`) and in
the agent runs' audit logs.

| Model | What worked | What did not, at first | What we changed |
|---|---|---|---|
| Nemotron 3.5 Lightning (supervisor) | 5/5 routing decisions; 15.3 s per county run | Called tools without saying why; returned empty replies at a 900-token limit; followed an instruction injected into a tool result; picked the policy with the fewest abstentions first; reran a policy it had already used | `reason` is a required argument on every tool; 2,048-token limit; after a critic fail only a rerun or abstain is allowed, enforced in code; prompt rule on the first policy; code refuses a repeat rerun |
| Nemotron 3 Ultra (explainer and writer) | 5/5 routing, 4/4 evidence Q&A; resisted prompt injection; 0 invented numbers under guardrails | With thinking on, used its whole budget and returned nothing; added numbers not in the evidence in 2 of 8 field reasons; misread a harvest-pace table once; about a minute per report | Thinking off for explanations; every number checked, with a template fallback; the critic's verdict is decided in code and Ultra only explains it; reply cache, and Lightning drafts the Q&A |
| Nemotron 3.5 Super VL (vision) | Clear before-and-after descriptions in about 6 s per field | Empty replies at a 400-token limit (it reasons for about 570 tokens first); no tool calling; right on 15 of 28 readable labelled fields at 10 m | 2,048-token limit; asks for plain JSON, validated in code; used as visual context, never as the label |
| Nemotron 3.5 Bolt (benchmarked) | 5/5 routing; fastest at 6.9 s per county run; resisted prompt injection | Without guardrails, once called the pipeline's output "the NASS estimate" | Prompt rule plus a code check that sends the answer back once |

**Access and limits.** Nemotron 3 Super 120B returned 404 on our key, so Ultra runs the critic. No embedding
model was available, so the RAG index uses local search. A shared free-tier key (about 40 requests a minute) meant
caching every reply, running counties one at a time, and an offline rule-based agent that calls the same tools.

**Design decisions.**
- Small model routes, large model explains: Lightning keeps the loop fast, Ultra's quality is worth its time
  where the wording matters.
- Code owns every number: Nemotron decides the steps and writes the explanation, and code checks every
  number it writes.
- Wake the agent only when it matters: in a 5-day replay of Wyoming County NY 2022 the agent would run 4 times
  instead of 21.

**Feedback for the Nemotron team:** a thinking budget per call, tool calling in Super VL, and an embedding model
on the same key.

</details>

## Repository map

<details>
<summary>Show the repository map</summary>

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
├── handoff/                     final deck, result slides as images, result files (guide: handoff/README.md)
├── results/                     model-side result notes (results/README.md)
└── data/                        pointers only; county data lives outside git (data/README.md)
```

**Where to start, by question:**

| I want to… | Read |
|---|---|
| See every result and whether it is a clean test | [`docs/RESULTS.md`](docs/RESULTS.md) |
| See the radar (Sentinel-1) result | [Cloud gaps: Sentinel-1 radar](#cloud-gaps-sentinel-1-radar-pilot), [`colab/extraction/s1_24029_2022/`](colab/extraction/s1_24029_2022/README.md) |
| Understand the model and its features | [`docs/module.md`](docs/module.md), [`docs/model.md`](docs/model.md), [`docs/features.md`](docs/features.md) |
| Understand the agents and the app | [`src/fieldvision_agents/README.md`](src/fieldvision_agents/README.md) |
| Know why the state figures read low | [`handoff/results/hima/COMBINED_GAP_FINDINGS.md`](handoff/results/hima/COMBINED_GAP_FINDINGS.md) |
| Extract a new county | [`docs/s2-extractor.md`](docs/s2-extractor.md), [`colab/README.md`](colab/README.md), `scripts/colab_to_county.py` |
| Use the toolbox from Python | [`docs/toolbox.md`](docs/toolbox.md) |
| See what was tried, including what failed | [`docs/log.md`](docs/log.md) |
| Present it | [`handoff/presentation/FieldVision.pdf`](handoff/presentation/FieldVision.pdf) |

</details>

## Data

<details>
<summary>Show data sources and layout</summary>

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

</details>

## Reproduce the main results

<details>
<summary>Show the commands</summary>

| Result | How |
|---|---|
| Beltsville scoring | `docs/model.md` (commands) and `handoff/results/REPORT.md` (the logged runs) |
| One county through the agent | `fieldvision run <fips> <year> --report` |
| Agent evaluation and model comparison | `notebooks/agents/10_evaluation.ipynb`, `14_benchmark.ipynb` (need `NVIDIA_API_KEY`) |
| Known gaps checked against the data | `python -m fieldvision_agents.validation.gap_checks` |
| Gap table and 5-day replay | `python -m fieldvision_agents.gap.gap <fips> <year> --data data/counties`, `python -m fieldvision_agents.watch.watch <fips> <year> --data data/counties` (guides in their folders) |
| Price models | `src/fieldvision_agents/market/README.md` before quoting any number |
| Many counties on a server | `python scripts/pod_batch_counties.py list --top 50`, then `run <extract_root> <year>` |

</details>

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
