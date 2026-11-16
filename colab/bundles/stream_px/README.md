# stream_px: upload 15 files to /content (bundle.json, code_stream_efb675be.zip, 8 fields_*.geojson, 5 ref_*.parquet) and open extract_stream_px.ipynb; run all cells top to bottom (1-7).

This pixel-chip job covers 4282 labelled European fields plus 44 Beltsville field-years with labels removed. It reads with the frozen Sentinel-2 code (fingerprint 9cd0447b) and saves a 12 x 12 pixel chip per field and usable date next to the same full-field medians.

Priority: run this AFTER the Maryland 2025 sample. Select a CPU high-RAM runtime with at least 8 cores; no GPU is needed. The job is about 200 read blocks (roughly 2 hours on one runtime). It can be split across up to three runtimes by shortening RUNS in cell 1: at_2022, at_2023, at_2024, barc_* / fr_2022 / nl_2025 (upload the same files to each).

Each run's files (parts + stream_px_<run>_SHA256SUMS.txt) go to C:\t\fv\colab_results\stream_px_<run>\ (watcher: -Run stream_px_<run>). The training job (C:\t\fv\colab\stream_train) can also read them from Drive (MyDrive/fieldvision_stream_px) if cell 2 mounted it.

Open extract_stream_px.ipynb in Colab. Upload these files into /content:

- bundle.json
- code_stream_efb675be.zip
- fields_at_2022.geojson
- fields_at_2023.geojson
- fields_at_2024.geojson
- fields_fr_2022.geojson
- fields_nl_2025.geojson
- fields_barc_2021.geojson
- fields_barc_2022.geojson
- fields_barc_2023.geojson
- ref_at_2022.parquet
- ref_at_2023.parquet
- ref_at_2024.parquet
- ref_fr_2022.parquet
- ref_nl_2025.parquet

Cell 2 optionally mounts Drive. Finished block checkpoints are backed up every 300 seconds under MyDrive/fieldvision_stream_px and restored after a runtime restart. Rerun cell 5 to resume; complete blocks are skipped. Network errors mean rerun this cell. Cell 6 prints comparison counts and requires at least 99% agreement for European references; BARC prints manifest counts only.

Dependencies: build_colab_bundle.PINS fallback; no restricted template bundle was read. Explicit --pins-from accepts a JSON package/version mapping or a bundle JSON containing pins. The notebook prints Python, package, GDAL, and PROJ versions.

Cell 7 returns one stream_px_<run>_colab_chips.zip per run containing chips.npz, timeseries.parquet, manifest.json, and colab_environment.txt. It also writes <=50 MiB .zip.000, .zip.001, ... parts and stream_px_<run>_SHA256SUMS.txt. Parts and the full ZIP have SHA256 checksums. Drive receives the ZIPs, parts, and checksum files. No automatic browser downloads are started.

For each run, download every part and its checksum file into a stream_px_<run> folder, then unpack into that run's configured results folder:

```sh
python scripts/stream_unpack.py --parts-dir DOWNLOADED_PARTS --out RESULTS_FOLDER
```

The four result files are extracted directly into RESULTS_FOLDER. Pass a parts folder containing exactly one run's checksum file. Verified result files replace existing files atomically. The unpacker verifies every part, the full ZIP, ZIP CRCs, and safe member names before installing a run.
