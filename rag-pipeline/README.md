# rag-pipeline

The critic's memory for FieldVision. It answers "does this county result make sense?" with
QuickStats numbers (exact, from the CSV) and cited passages from NASS reports (RAG).
Numbers never come from the vector index.

## Layout

| Path | What goes there |
| --- | --- |
| `rag_pipeline.py` | Ingest, index, the 3 tools, critic loop, CLI |
| `scripts/fetch_crop_progress.py` | 53 national weekly Crop Progress .txt files, Aug-Nov 2021-2023 -> `dataset/national/` |
| `scripts/fetch_md_progress.py` | Maryland weekly Crop Progress & Condition PDFs -> `dataset/maryland/` |
| `dataset/handbooks/` | Usual Planting and Harvesting Dates (2010) and other national references |
| `dataset/quickstats/` | `corn_grain_silage_acres_2012_2025.csv` (copy from the challenge `02_QuickStats` folder) |
| `rag_index/` | Built by `build`. Not committed |
| `tests/smoke_test.py` | Offline test, no GPU or key needed |

Files in `national/` and `handbooks/` are tagged as national (`US`). Elsewhere, start file names
with the state code (`MD_`, `TX_`) so retrieval can filter by state.

## Run

```
pip install -r rag-pipeline/requirements.txt
python rag-pipeline/tests/smoke_test.py
python rag-pipeline/scripts/fetch_crop_progress.py
python rag-pipeline/scripts/fetch_md_progress.py --years 2021 2022 2023
python rag-pipeline/rag_pipeline.py build
python rag-pipeline/rag_pipeline.py ask "When is corn for grain usually harvested in Maryland?" --state Maryland
python rag-pipeline/rag_pipeline.py check --state Maryland --silage 60000 --grain 380000
```

## Tools the supervisor calls (names and return shapes are fixed)

- `retrieve_reports(query, state=None, year=None, k=5)` -> passages with `source`, `page`, `state`, `year`, `score`
- `silage_share_history(state, county=None, years=None)` -> silage share by year + min/max/mean
- `check_prediction(state, pred_silage_acres, pred_grain_acres, planted_acres=None, county=None)` -> pass/warn/fail per check
- `run_critic(prediction)` -> flags JSON with `suggested_action` (needs `NVIDIA_API_KEY` in `FieldVision/.env`)

Embeddings: `EMBED_BACKEND=local` (default, GPU if available, else CPU) or `nvidia`.
