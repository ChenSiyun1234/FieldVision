# us_26057_2022: upload bundle.json, check_blocks.zip, code_9cd0447b.zip, corn_grain_silage_acres_2012_2025.csv, corn_grain_silage_acres_2021_2023.csv, extract_us_26057_2022.ipynb to /content; run all cells top to bottom (1-4, A, B, S, C, 5-7).

Cell C downloads `us_26057_2022_fields.zip` + `us_26057_2022_fields_SHA256SUMS.txt` (fields.geojson = the extracted set,
fields_all.geojson = every corn field of the county, sample.json = the block draw, summary, corn-area check);
cell 7 downloads the result parts `us_26057_2022_colab_blocks.zip.NNN` + `us_26057_2022_SHA256SUMS.txt`, as for the other runs.
Untouched test county: no model on it until the modelling track has logged the test.

Back on the laptop (repository root, PYTHONPATH=src):
1. check the fields zip's SHA256, unzip it to <FV_DATA>\data\labels\us_26057_2022\;
2. create the run folder with the extractor's signature for that file:
   python -c "import sys; sys.path.insert(0, 'scripts'); from pathlib import Path; import build_colab_run as b; b.start_fresh_run(Path(r'<FV_DATA>\data\timeseries\us_26057_2022'), Path(r'<FV_DATA>\data\labels\us_26057_2022\fields.geojson'), Path(r'<FV_DATA>\frozen_src_154133e'), Path(r'<FV_DATA>\data\timeseries\us_24029_2022'))"
3. python scripts/merge_colab_blocks.py --parts-dir <FV_DATA>/colab_results/us_26057_2022 --run-dir <FV_DATA>/data/timeseries/us_26057_2022
4. finalise with the frozen code: PYTHONPATH=<FV_DATA>/frozen_src_154133e python -m fieldvision.s2.extract --fields <FV_DATA>/data/labels/us_26057_2022/fields.geojson --year 2022 --out <FV_DATA>/data/timeseries/us_26057_2022 --workers 16
