# Bringing Colab results back (all four runs)

All four runs use the same frozen extractor code (fingerprint `9cd0447b…`). Each run is
independent: handle the downloads in whatever order they arrive. Run every command from
the repository root `<REPO>`
in Git Bash. Nothing here needs the network except where a step says so.

| Run | Colab bundle | What comes back | Laptop run signature | Who writes READY |
|---|---|---|---|---|
| `nl_2025` | `colab\nl_2025` (fresh run) | one zip `nl_2025_colab_result.zip` (full outputs) | `5ba8187c8ae7` | coordinator / s2 session |
| `us_24021_2022` Frederick | `colab\us_24021_2022` (continue) | `us_24021_2022_colab_blocks.zip.000…` + `us_24021_2022_SHA256SUMS.txt` | `af46982b81c5` | county session |
| `fr_2022` France | `colab\fr_2022_rest` (continue) | `fr_2022_colab_blocks.zip.000…` + `fr_2022_SHA256SUMS.txt` | `569e1fa48ddb` | coordinator / s2 session |
| `us_24035_2022` Queen Anne's | `colab\us_24035_2022` (continue) | `us_24035_2022_colab_blocks.zip.000…` + `us_24035_2022_SHA256SUMS.txt` | `2fb9dbcedda2` | county session |

Before downloading: the notebook's step 6 must have ended with exit code 0, or 2 with
**no** "network read" scene errors left (rerun step 6 on Colab until that holds; a few
plausibility rejections are normal and final).

## A. The three "continue" runs (Frederick, France, Queen Anne's)

Replace `RUN` with `us_24021_2022`, `fr_2022` or `us_24035_2022`.

1. Put **all** downloaded parts and the `RUN_SHA256SUMS.txt` file into
   `<FV_DATA>\colab_results\RUN\`. A part that arrived truncated is caught in step 3.
2. Check that no laptop process is writing to that run (must print nothing for RUN):

```bash
powershell -NoProfile -Command 'Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*fieldvision.s2.extract*" } | Select-Object ProcessId, CommandLine | Format-List'
```

3. Dry run, then the merge. It verifies every checksum (re-download any part it names),
   refuses unless the Colab run had the same code fingerprint and run signature as the
   laptop run (table above), keeps blocks already complete on the laptop, copies the rest:

```bash
PYTHONPATH=src <FV_DATA>/venv/Scripts/python.exe scripts/merge_colab_blocks.py --parts-dir <FV_DATA>/colab_results/RUN --run-dir <FV_DATA>/data/timeseries/RUN --dry-run
```
```bash
PYTHONPATH=src <FV_DATA>/venv/Scripts/python.exe scripts/merge_colab_blocks.py --parts-dir <FV_DATA>/colab_results/RUN --run-dir <FV_DATA>/data/timeseries/RUN
```

   The last line must show only `complete` (and possibly `partial`) blocks and the full
   block count (France 80, Frederick 70, Queen Anne's 52). If it lists a block as
   `partial` with network errors, or the count is short, stop: rerun step 6 on Colab and
   download again, rather than retrying on this slow link.

4. Finalise with the run's own command and the frozen code. With every block finished it
   reads nothing from the network and rebuilds `timeseries.parquet`,
   `fields_coverage.csv` and `manifest.json` from the block files:

```bash
PYTHONPATH=<FV_DATA>/frozen_src_154133e <FV_DATA>/venv/Scripts/python.exe -m fieldvision.s2.extract --fields <FV_DATA>/data/labels/RUN/fields.geojson --year 2022 --out <FV_DATA>/data/timeseries/RUN --workers 16
```

   It must end with `INFO complete:` or `INFO partial:` and a `n/n fields usable` count.

5. Check, then READY. Print the line first (no `--append`); it must say `status=complete`,
   or `partial` with only `plausibility` / `missing_asset` failures:

```bash
PYTHONPATH=src <FV_DATA>/venv/Scripts/python.exe scripts/report_s2_run.py --run <FV_DATA>/data/timeseries/RUN
```

   * Frederick and Queen Anne's: tell the county session the merge is done; it runs
     step 4 (if not done), its toolbox copy and its own READY line.
   * France: append with the note below (the script refuses if network failures remain):

```bash
PYTHONPATH=src <FV_DATA>/venv/Scripts/python.exe scripts/report_s2_run.py --run <FV_DATA>/data/timeseries/fr_2022 --append <FV_DATA>/data/timeseries/READY.txt --note "26 blocks on the laptop, the rest and all scene reads that failed during the 14:51-15:30 network outage on Google Colab, same frozen code 9cd0447b and arguments; same-result check passed; merged by checksum and run signature"
```

## B. The Netherlands (fresh run, one zip with everything)

1. Put `nl_2025_colab_result.zip` into `<FV_DATA>\colab_results\nl_2025\` and test it
   (a truncated download fails here):

```bash
<FV_DATA>/venv/Scripts/python.exe -c "import zipfile; z=zipfile.ZipFile('<FV_DATA>/colab_results/nl_2025/nl_2025_colab_result.zip'); print('bad member:', z.testzip()); print(len(z.namelist()), 'files')"
```

   It must print `bad member: None`.
2. Move the laptop's partial folder aside (it holds the prefetched catalog and one half
   block from the 14:15 false start; keep it until READY is written):

```bash
mv <FV_DATA>/data/timeseries/nl_2025 <FV_DATA>/data/timeseries/nl_2025_laptop_partial
```

3. Unpack and compare code fingerprint and run signature with the laptop's:

```bash
mkdir -p <FV_DATA>/data/timeseries/nl_2025 && <FV_DATA>/venv/Scripts/python.exe -c "import zipfile; zipfile.ZipFile('<FV_DATA>/colab_results/nl_2025/nl_2025_colab_result.zip').extractall('<FV_DATA>/data/timeseries/nl_2025')"
```
```bash
<FV_DATA>/venv/Scripts/python.exe -c "import json; a=json.load(open('<FV_DATA>/data/timeseries/nl_2025/manifest.json')); b=json.load(open('<FV_DATA>/data/timeseries/nl_2025_laptop_partial/manifest.json')); print(a['status'], a['source_sha256'][:8], a['signature'][:12], 'same as laptop:', a['signature']==b['signature']); print(open('<FV_DATA>/data/timeseries/nl_2025/colab_environment.txt').read()[-200:])"
```

   It must print fingerprint `9cd0447b`, `same as laptop: True` and `same-result check PASSED`.
4. Check, then READY (the zip already contains `timeseries.parquet`; no finalising run):

```bash
PYTHONPATH=src <FV_DATA>/venv/Scripts/python.exe scripts/report_s2_run.py --run <FV_DATA>/data/timeseries/nl_2025
```
```bash
PYTHONPATH=src <FV_DATA>/venv/Scripts/python.exe scripts/report_s2_run.py --run <FV_DATA>/data/timeseries/nl_2025 --append <FV_DATA>/data/timeseries/READY.txt --note "extracted on Google Colab (free tier, 2 CPU, Python 3.13), same frozen code 9cd0447b and arguments as the laptop lane; same-result check passed on 2 laptop blocks"
```

## If something does not match

Do not edit files by hand. A checksum, fingerprint or signature mismatch means the
download or the Colab run is not the one described here: re-download, or tell the s2
session with the exact message the script printed.
