# hima_data: extra counties for the county check

Downloads per-field Sentinel-2 time series for extra counties and exports them in the FieldVision
county format, so the toolbox, agent, critic and recommender can run on them. Adapted from the team's
Colab download notebook to run in the Keeboh workspace.

## Files
| File | What it is |
|---|---|
| hima_data_download.ipynb | The notebook (run all; set TEST = True first) |
| hima_data_download.py | Same code in percent-cell format (the notebook is generated from it) |
| scratch/py2nb.py | Turns the .py into the .ipynb: python3 scratch/py2nb.py hima_data_download.py hima_data_download.ipynb |
| scratch/census_check.py | Census 2012/2017/2022 silage shares for candidate counties from the QuickStats CSV |
| scratch/toolbox_check.py | Runs the FieldVision toolbox on the exported counties |
| scratch/csb_probe*.py | Checks on the national Crop Sequence Boundaries file |
| DATA_LOG.md, run_NY_36121.log | What ran, when, with which settings |

## Settings
- Year 2022 (Census of Agriculture year, the county ground truth).
- Field outlines: national CSB 2018-2025 zip read in place; corn = CSB majority crop for the year plus at least
  80% corn pixels in the CDL raster; at least 2 ha.
- 800 fields per county as random whole 5 km tiles; Sentinel-2 L2A from Planetary Computer, all 10 bands,
  SCL 4/5, at least 80% clear pixels per field-date, 15 m inward buffer; window Apr 1 to Dec 15 (CA from Mar 1).
- Radar and weather are off by default. County Census numbers come from the workspace QuickStats CSV
  (all 50 states), so no API key is needed.

## Run from a terminal
    cd /workspace/hima_data
    HIMA_TEST=1 python3 hima_data_download.py                                         # 4-field test
    HIMA_COUNTIES=NY:36121 nohup python3 hima_data_download.py > run_NY_36121.log 2>&1 &

## Done so far
- Wyoming, NY (36121), 2022: 820 of 2,059 corn fields (17,299 acres), 18,385 field-date rows, median 25 clear
  dates, 30 tiles in 29.5 minutes. Census 2022: 66.7% silage (38,593 silage acres, 19,251 grain).
- Tulare, CA (06107): test only. Only 282 corn fields pass the corn filter against about 118,000 Census corn
  acres (double-crop corn is labelled differently in CDL), so Gooding, ID (16047) is the suggested swap.

Data outputs (processed/, fieldvision_export/, raw/) stay out of git; they are in hima_data_outputs.zip.
