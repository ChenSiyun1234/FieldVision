# France 2022 (`fr_2022`): finish the laptop run on Colab

Same frozen code as the laptop run (`9cd0447bcc1272ba…`, equal to the manifest) and the
same arguments (`--year 2022`, everything else default, as on the laptop). All laptop
block checkpoints are shipped (snapshot ~16:00: 26 complete, 3 partial with network
failures, 1 running): Colab skips the 26, re-reads only the scenes recorded as network
failures, continues the running block and does the 50 blocks the laptop never reached.
The result is the same as an uninterrupted laptop run: the same code, arguments and
scene lists produce identical rows (checked in step 4), and the merge refuses
anything with a different code fingerprint or run signature.

## Files (upload all into `/content`)

| File | Size | What |
|---|---:|---|
| `bundle.json` | 1 KB | names, fingerprint, arguments, pinned versions |
| `code_9cd0447b.zip` | 14 KB | frozen `fieldvision` package |
| `fields_fr_2022.geojson` | 2.0 MB | byte-identical to `C:\t\fv\data\labels\fr_2022\fields.geojson` |
| `blocks_fr_2022.zip.000` | 1.4 MB | all laptop block checkpoints (parquet + json) |
| `catalog_fr_2022.zip.000` | 1.8 MB | prefetched scene lists of the 54 unfinished blocks (same lists the laptop uses) |
| `check_blocks.zip` | 0.04 MB | finished laptop block `32630_FR-E3440N2860_0` (30 scenes) + reference rows |
| `extract_fr_2022.ipynb` | 16 KB | open this in Colab |

## Run order

Cells top to bottom, as for the other notebooks: settings, Drive (recommended), pinned
installs, fingerprint + same-result check (stops on any difference), unpack checkpoints
and catalog, extraction (`done/80`), result parts. If step 6 ends with "network read"
scene errors, run step 6 again.

## Expected time

About 54 blocks with ~3,900 scene-block reads plus ~110 re-reads (France 2022 is mostly
Planetary Computer, 30-70 scenes per block). At ~1.2 CPU-s per read: 2 vCPU about
1-1.5 h; 8 vCPU about 20-30 min.

## Back on the laptop

Do these in order. The laptop France lane must be stopped first (the coordinator asks
the s2 session to do that as soon as Colab has started this notebook).

1. Put the downloaded parts and `fr_2022_SHA256SUMS.txt` into
   `C:\t\fv\colab_results\fr_2022\`, then from the repository root:

```bash
PYTHONPATH=src /c/t/fv/venv/Scripts/python.exe scripts/merge_colab_blocks.py --parts-dir C:/t/fv/colab_results/fr_2022 --run-dir C:/t/fv/data/timeseries/fr_2022
```

   It verifies every checksum, refuses unless code fingerprint and run signature equal
   the laptop run's, keeps blocks already complete on the laptop (including any the
   laptop finished after this snapshot) and copies every other block from Colab.

2. Finalise with the run's own command and the same frozen code. With all 80 blocks
   finished it reads nothing from the network; it rebuilds `timeseries.parquet`,
   `fields_coverage.csv` and the manifest from the block files exactly as at the end
   of an uninterrupted run (any block still partial is retried first):

```bash
PYTHONPATH=C:/t/fv/frozen_src_154133e /c/t/fv/venv/Scripts/python.exe -m fieldvision.s2.extract --fields C:/t/fv/data/labels/fr_2022/fields.geojson --year 2022 --out C:/t/fv/data/timeseries/fr_2022 --workers 16
```

3. READY line (refused automatically if any network failure remains):

```bash
PYTHONPATH=src /c/t/fv/venv/Scripts/python.exe scripts/report_s2_run.py --run C:/t/fv/data/timeseries/fr_2022 --append C:/t/fv/data/timeseries/READY.txt --note "26 blocks on the laptop, the rest and all scene reads that failed during the 14:51-15:30 network outage on Google Colab, same frozen code 9cd0447b and arguments; same-result check passed; merged by checksum and run signature"
```
