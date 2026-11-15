# Frederick County 2022 (`us_24021_2022`): continue the laptop run on Colab

Same code as the laptop run (frozen copy, fingerprint `9cd0447bcc1272ba…`, equal to the
run manifest's `source_sha256`) and the same arguments (`--year 2022`, all other options
at their defaults, exactly as in the laptop manifest). The laptop's 9 block checkpoints
(7 complete, 1 partial, 1 running at 15:5x) are shipped: Colab skips the 7 finished
blocks, re-reads only the scenes that failed on the laptop's network, and does the
remaining blocks of 70.

## Files (upload all into `/content`, each well under 9 MB)

| File | Size | What |
|---|---:|---|
| `bundle.json` | 1 KB | names, fingerprint, arguments, pinned versions |
| `code_9cd0447b.zip` | 14 KB | frozen `fieldvision` package |
| `fields_us_24021_2022.geojson` | 4.0 MB | byte-identical to `C:\t\fv\data\labels\us_24021_2022\fields.geojson` |
| `blocks_us_24021_2022.zip.000` | 0.7 MB | the laptop's block checkpoints (parquet + json) |
| `catalog_us_24021_2022.zip.000` | 0.1 MB | prefetched scene lists of the 2 unfinished laptop blocks (Colab queries the others itself) |
| `check_blocks.zip` | 0.08 MB | finished laptop block `32618_US-E270N4350_1` (121 scenes, 11 fields) + reference rows |
| `extract_us_24021_2022.ipynb` | 16 KB | open this in Colab |

## Run order

Same as the Netherlands notebook: run the cells top to bottom.
1. settings (lists missing uploads), 2. mount Drive (recommended), 3. pinned installs,
4. code fingerprint + same-result check (re-extracts block `32618_US-E270N4350_1` and
compares field, date, chosen scene, pixel counts and all 10 bands exactly; any
difference stops the notebook), 5. unpack checkpoints and catalog,
6. extraction with progress `done/70`, 7. result parts.

Step 7 sends back only the blocks finished on Colab (block parquet + checkpoint json)
and the Colab manifest, zipped, in parts of at most 50 MB, plus
`us_24021_2022_SHA256SUMS.txt`. Download all parts and the checksum file. The full
time series is rebuilt on the laptop from the blocks, so nothing else has to travel.

If step 6 ends with "network read" scene errors, run step 6 again.

## Expected time

61 unfinished blocks, mostly 121 scenes each (two MGRS tiles, 2022 = Collection 1 +
Planetary Computer): about 7,000 scene-block reads. At ~1.2 CPU-s per read: 2 vCPU
about 1.5-2 h; 8 vCPU about 25-40 min. Catalog queries for the 59 blocks without a
prefetched list add ~30 s each, interleaved.

## Back on the laptop

1. Stop the laptop's Frederick extraction (the county session's process) so two
   writers never touch the same folder.
2. Put the downloaded parts and `us_24021_2022_SHA256SUMS.txt` in one folder, e.g.
   `C:\t\fv\colab_results\us_24021_2022\`, then from the repository root:

```bash
PYTHONPATH=src /c/t/fv/venv/Scripts/python.exe scripts/merge_colab_blocks.py --parts-dir C:/t/fv/colab_results/us_24021_2022 --run-dir C:/t/fv/data/timeseries/us_24021_2022
```

   It checks every checksum, refuses to merge unless the Colab run had the same code
   fingerprint and the same run signature as the laptop run, keeps blocks already
   complete on the laptop, and copies the rest.
3. Finalise with the run's own command and code (no network needed when all 70 blocks
   are finished; any block still partial is retried):

```bash
PYTHONPATH=C:/t/fv/frozen_src_154133e /c/t/fv/venv/Scripts/python.exe -m fieldvision.s2.extract --fields C:/t/fv/data/labels/us_24021_2022/fields.geojson --year 2022 --out C:/t/fv/data/timeseries/us_24021_2022 --workers 16
```

4. READY line (the county session may prefer to write its own):

```bash
PYTHONPATH=src /c/t/fv/venv/Scripts/python.exe scripts/report_s2_run.py --run C:/t/fv/data/timeseries/us_24021_2022 --append C:/t/fv/data/timeseries/READY.txt --note "61 of 70 blocks extracted on Google Colab with the same frozen code 9cd0447b and arguments; same-result check passed; merged by checksum and run signature"
```
