# README assets

Every image on the main [README](../../README.md) lives in this folder. To change something, find it below,
update it here, and the README picks it up.

```
docs/readme/
├── README.md                this index
├── charts/                  charts drawn by code (edit make_charts.py, then rerun it)
│   ├── make_charts.py
│   ├── highlights.svg
│   ├── gap_closing.svg
│   ├── state_2025.svg
│   ├── us_tests.svg
│   ├── pipeline.svg
│   └── field_check.svg
├── slides/                  key slides exported from the final deck
├── strips_sheet.png         satellite picture strips of real fields
├── hero.svg                 banner at the top
├── grand_prize_check_card.png
└── 02_features.png ... 08_evidence_tiers.png   model figures
```

## Charts (`charts/`)

Drawn by [`charts/make_charts.py`](charts/make_charts.py) with Python only, no extra packages. Every number is
written in the script next to the chart that uses it. After an edit, rerun from the repo root:

```bash
python docs/readme/charts/make_charts.py
```

| File | README section | Function in `make_charts.py` | Numbers come from |
|---|---|---|---|
| `highlights.svg` | top, under the title | `highlights()` | `docs/RESULTS.md`, `docs/hima/cloud_gaps.md` |
| `gap_closing.svg` | A classifier was not enough | `gap_closing()` | `docs/RESULTS.md`, `docs/hima/cloud_gaps.md`, `handoff/results/hima/COMBINED_GAP_FINDINGS.md` |
| `state_2025.svg` | Graded against USDA NASS 2025 | `state_2025()` | `docs/RESULTS.md`, `handoff/results/nass_2025_state_targets.csv` |
| `us_tests.svg` | Graded against USDA NASS 2025 | `us_tests()` | state lists in the function; map from `app/vendor/us-atlas/` |
| `pipeline.svg` | The pipeline: 9 stages | `pipeline()` | stage cards in the function; Nemotron sizes from NVIDIA model cards |
| `field_check.svg` | How the classifier was built | `field_check()` | `handoff/results/ACCESS_LOG.md`, `docs/module.md` |

Colours are set once at the top of the script (`ACCENT`, `INK`, `MUTED`, `LINE`, `BADGE`, `MAPGOLD`, `MAPGREEN`).

## Slides (`slides/`)

Exported from the final deck ([`handoff/presentation/FieldVision.pdf`](../../handoff/presentation/FieldVision.pdf))
as JPG, 96 dpi, quality 82. To replace one, export the same slide again and keep the file name, for example:

```bash
pdftoppm -f 6 -l 6 -r 96 -jpeg -jpegopt quality=82 FieldVision.pdf slide
```

| File | Deck slide |
|---|---|
| `04_data_features.jpg` | 4, Data and Feature Engineering |
| `05_model_alone.jpg` | 5, Model alone |
| `06_bayes_usda_history.jpg` | 6, Model + USDA history |
| `07_counties_vs_usda.jpg` | 7, Results and validation (counties) |
| `08_cloud_gaps.jpg` | 8, Cloud gaps |
| `09_field_level_test.jpg` | 9, Results and validation (field level) |
| `10_agent_architecture.jpg` | 10, Agentic intelligence (architecture) |
| `11_agent_reliability.jpg` | 11, Agentic intelligence (what broke, what we fixed) |

## Other images

| File | README section | Made by |
|---|---|---|
| `hero.svg` | banner | `scripts/make_readme_art.py` |
| `strips_sheet.png` | How the classifier was built | copy of `handoff/figures/deck/strips_sheet.png` |
| `02_features.png`, `03_validation.png`, `04_confusion.png`, `06_landsat.png`, `08_evidence_tiers.png` | How the classifier was built, Model figures | `scripts/make_readme_figures.py` |
| `grand_prize_check_card.png` | Team | photo of the Grand Prize check |

Files here that the README does not use (`00_banner.png`, `01_pipeline.png`, `05_nass_comparison.png`,
`07_timeline.png`, `stats.svg`, `model_stats.svg`, `grand_prize_check.jpg`) are kept because other scripts write them.
