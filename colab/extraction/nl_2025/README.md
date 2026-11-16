# Netherlands 2025 extraction on Google Colab

Same extractor code as France 2022 on the laptop (frozen copy, fingerprint
`9cd0447bcc1272ba…`), same command and settings. Only the machine differs, and the
notebook proves that this does not change results before it starts.

## Files (upload all of them into `/content`, one per call, each < 9 MB)

| File | Size | What |
|---|---:|---|
| `bundle.json` | 1 KB | names, fingerprint, settings, pinned versions |
| `code_9cd0447b.zip` | 14 KB | the frozen `fieldvision` package |
| `check_blocks.zip` | 0.15 MB | two blocks the laptop finished + their reference rows |
| `fields_nl_2025.geojson` | 4.4 MB | the 1,600 Dutch fields, byte-identical to `<FV_DATA>\data\labels\nl_2025\fields.geojson` |
| `catalog_nl_2025.zip.000` | 8.4 MB | prefetched scene lists for all 95 blocks, part 1 |
| `catalog_nl_2025.zip.001` | 3.9 MB | part 2 |
| `extract_nl_2025.ipynb` | 14 KB | open this one in Colab (File > Upload notebook) |

## Run order

1. Open `extract_nl_2025.ipynb` in Colab with a plain CPU runtime; upload the six
   data files into `/content` (file panel, or drag and drop).
2. Run the cells top to bottom:
   1. settings (prints CPU count and any missing upload),
   2. mount Google Drive (optional, strongly recommended: checkpoints are copied
      there every 5 minutes so a recycled runtime does not lose work),
   3. installs the laptop's exact package versions (binary wheels only) and prints them
      with the GDAL and PROJ versions,
   4. unpacks the code, checks the fingerprint, then **re-extracts two blocks the laptop
      already finished** (Austria 2024 `32633_AT-E4720N2790_2`, Collection 1, 24 scenes;
      France 2022 `32630_FR-E3440N2860_0`, Planetary Computer, 30 scenes) and compares
      every row: field, date, chosen scene, pixel counts and all ten band values exactly.
      Any difference stops the notebook with the first differing value. Takes ~2-4 min.
      (Tested on the laptop: both blocks IDENTICAL.)
   5. prepares `/content/fv/nl_2025` (restores from Drive if present, adds the catalog),
   6. runs the extraction, printing one line per block with `done/95`,
   7. prints row counts and builds `nl_2025_colab_result.zip` (also copied to Drive),
      then offers it for download.
3. If step 4 fails: stop, do not run step 6, tell the coordinator. The laptop fallback
   is unchanged (below).

## Expected run time

Measured on the laptop: about 1.2 CPU-seconds per scene-block read; the Netherlands
needs 12,911 reads (95 blocks; the extractor streams roughly 10 MB per read).

| Colab runtime | Reads per second (estimate) | Netherlands run |
|---|---:|---|
| 2 vCPU (standard) | 1.0-1.5 | about 2.5-3.5 h |
| 8 vCPU (high-CPU) | 4-6, if the network keeps up | about 40-60 min |

Add ~5 minutes for installs and the check. The first `done/95` lines give the real rate.

## Network read errors

Planetary Computer and S3 sometimes refuse a burst of range requests (seen on the
laptop at 14:51 while two jobs read in parallel). The check retries such scenes up to
three times before comparing. If step 6 ends with exit code 2 and "scene errors by kind:
network read", run step 6 again: only the failed scenes are read again. A run is
complete when step 6 prints exit code 0 (or 2 with no network-read errors left, e.g. a
plausibility rejection, which is final and also happens on the laptop).

## Resume

Just run all cells again. The check is skipped if it already passed in this runtime with
the same code and libraries; the extractor skips finished blocks and finished scenes of
the block it was in. After a recycled runtime, step 5 restores the checkpoints from
Drive (if mounted). Never run two copies of step 6 at once.

## Getting the result back on the laptop

Download `nl_2025_colab_result.zip` (or take it from Drive), then in Git Bash:

```bash
mv <FV_DATA>/data/timeseries/nl_2025 <FV_DATA>/data/timeseries/nl_2025_laptop_partial
```
```bash
mkdir -p <FV_DATA>/data/timeseries/nl_2025 && unzip -q ~/Downloads/nl_2025_colab_result.zip -d <FV_DATA>/data/timeseries/nl_2025
```
```bash
grep -o '"source_sha256": "[0-9a-f]*' <FV_DATA>/data/timeseries/nl_2025/manifest.json && cat <FV_DATA>/data/timeseries/nl_2025/colab_environment.txt
```
The fingerprint must start with `9cd0447b`. Then, from the repository root:
```bash
PYTHONPATH=src <FV_DATA>/venv/Scripts/python.exe scripts/report_s2_run.py --run <FV_DATA>/data/timeseries/nl_2025 --append <FV_DATA>/data/timeseries/READY.txt --note "extracted on Google Colab, same frozen code 9cd0447b, same-result check passed on 2 laptop blocks"
```

## Laptop fallback (if Colab does not work)

```bash
PYTHONPATH=<FV_DATA>/frozen_src_154133e <FV_DATA>/venv/Scripts/python.exe -m fieldvision.s2.extract --fields <FV_DATA>/data/labels/nl_2025/fields.geojson --year 2025 --start 04-01 --end 11-30 --out <FV_DATA>/data/timeseries/nl_2025 --workers 32 --scene-batch 6
```
(after France is in READY.txt, at below-normal priority until Frederick is listed).
