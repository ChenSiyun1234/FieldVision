# us_55101_2023: upload bundle.json, check_blocks.zip, code_9cd0447b.zip, corn_grain_silage_acres_2012_2025.csv, corn_grain_silage_acres_2021_2023.csv, extract_us_55101_2023.ipynb to /content; run all cells top to bottom (1-4, A, B, S, C, 5-7).

Cell C downloads `us_55101_2023_fields.zip` + `us_55101_2023_fields_SHA256SUMS.txt` (fields.geojson = the extracted set,
fields_all.geojson = every corn field of the county, sample.json = the block draw, summary, corn-area check);
cell 7 downloads the result parts `us_55101_2023_colab_blocks.zip.NNN` + `us_55101_2023_SHA256SUMS.txt`, as for the other runs.
Untouched test county: no model on it until the modelling session has logged the test.

Back on the laptop (repository root, PYTHONPATH=src):
1. check the fields zip's SHA256, unzip it to C:\t\fv\data\labels\us_55101_2023\;
2. create the run folder with the extractor's signature for that file:
   python -c "import sys; sys.path.insert(0, 'scripts'); from pathlib import Path; import build_colab_run as b; b.start_fresh_run(Path(r'C:\t\fv\data\timeseries\us_55101_2023'), Path(r'C:\t\fv\data\labels\us_55101_2023\fields.geojson'), Path(r'C:\t\fv\frozen_src_154133e'), Path(r'C:\t\fv\colab\_templates\9cd0447b_2023'))"
3. python scripts/merge_colab_blocks.py --parts-dir C:/t/fv/colab_results/us_55101_2023 --run-dir C:/t/fv/data/timeseries/us_55101_2023
4. finalise with the frozen code: PYTHONPATH=C:/t/fv/frozen_src_154133e python -m fieldvision.s2.extract --fields C:/t/fv/data/labels/us_55101_2023/fields.geojson --year 2023 --out C:/t/fv/data/timeseries/us_55101_2023 --workers 16
