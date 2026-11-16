# Queen Anne's County 2022 (`us_24035_2022`): continue the laptop run on Colab

Same code as the laptop run (frozen copy, fingerprint `9cd0447bcc1272ba…`, equal to the
run manifest's `source_sha256`) and the same arguments (`--year 2022`, all other options
at their defaults, exactly as in the laptop manifest). The laptop's 12 block checkpoints
(10 complete, 1 partial, 1 interrupted; the laptop process stopped at 14:58) are shipped: Colab skips the 10 finished
blocks, re-reads only the scenes that failed on the laptop's network, and does the
remaining blocks of 52.

## Files (upload all into `/content`, each well under 9 MB)

| File | Size | What |
|---|---:|---|
| `bundle.json` | 1 KB | names, fingerprint, arguments, pinned versions |
| `code_9cd0447b.zip` | 14 KB | frozen `fieldvision` package |
| `fields_us_24035_2022.geojson` | 3.1 MB | byte-identical to `<FV_DATA>\data\labels\us_24035_2022\fields.geojson` |
| `blocks_us_24035_2022.zip.000` | 0.2 MB | the laptop's block checkpoints (parquet + json) |
| `catalog_us_24035_2022.zip.000` | 0.04 MB | prefetched scene lists of the 2 unfinished laptop blocks (Colab queries the others itself) |
| `check_blocks.zip` | 0.02 MB | finished laptop block `32618_US-E380N4320_0` (32 scenes, 3 fields) + reference rows |
| `extract_us_24035_2022.ipynb` | 16 KB | open this in Colab |

## Run order

Same as the Netherlands notebook: run the cells top to bottom.
1. settings (lists missing uploads), 2. mount Drive (recommended), 3. pinned installs,
4. code fingerprint + same-result check (re-extracts block `32618_US-E380N4320_0` and
compares field, date, chosen scene, pixel counts and all 10 bands exactly; any
difference stops the notebook), 5. unpack checkpoints and catalog,
6. extraction with progress `done/52`, 7. result parts.

Step 7 sends back only the blocks finished on Colab (block parquet + checkpoint json)
and the Colab manifest, zipped, in parts of at most 50 MB, plus
`us_24035_2022_SHA256SUMS.txt`. Download all parts and the checksum file. The full
time series is rebuilt on the laptop from the blocks, so nothing else has to travel.

If step 6 ends with "network read" scene errors, run step 6 again.

## Expected time

42 unfinished blocks, 32-65 scenes each (one or two MGRS tiles, 2022 = Collection 1 +
Planetary Computer): about 2,000 scene-block reads. At ~1.2 CPU-s per read: 2 vCPU
about 30-45 min; 8 vCPU about 10-15 min. Catalog queries for the 40 blocks without a
prefetched list add ~30 s each, interleaved.

## Back on the laptop

1. Make sure no laptop process writes to the Queen Anne's run (none was running at 16:10) so two
   writers never touch the same folder.
2. Put the downloaded parts and `us_24035_2022_SHA256SUMS.txt` in one folder, e.g.
   `<FV_DATA>\colab_results\us_24035_2022\`, then from the repository root:

```bash
PYTHONPATH=src <FV_DATA>/venv/Scripts/python.exe scripts/merge_colab_blocks.py --parts-dir <FV_DATA>/colab_results/us_24035_2022 --run-dir <FV_DATA>/data/timeseries/us_24035_2022
```

   It checks every checksum, refuses to merge unless the Colab run had the same code
   fingerprint and the same run signature as the laptop run, keeps blocks already
   complete on the laptop, and copies the rest.
3. Finalise with the run's own command and code (no network needed when all 52 blocks
   are finished; any block still partial is retried):

```bash
PYTHONPATH=<FV_DATA>/frozen_src_154133e <FV_DATA>/venv/Scripts/python.exe -m fieldvision.s2.extract --fields <FV_DATA>/data/labels/us_24035_2022/fields.geojson --year 2022 --out <FV_DATA>/data/timeseries/us_24035_2022 --workers 16
```

4. READY line (the county session may prefer to write its own):

```bash
PYTHONPATH=src <FV_DATA>/venv/Scripts/python.exe scripts/report_s2_run.py --run <FV_DATA>/data/timeseries/us_24035_2022 --append <FV_DATA>/data/timeseries/READY.txt --note "42 of 52 blocks extracted on Google Colab with the same frozen code 9cd0447b and arguments; same-result check passed; merged by checksum and run signature"
```
