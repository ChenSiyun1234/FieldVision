# Carroll County 2022 (`us_24013_2022`): full extraction on Colab

No laptop extraction exists for this county; Colab runs all 56 read blocks from scratch.
Same frozen code as the other county runs (fingerprint `9cd0447bcc1272ba…`) and the same
arguments (`--year 2022`, everything else default; copied from the Queen Anne's run). The
laptop folder `<FV_DATA>\data\timeseries\us_24013_2022\` holds only a stub manifest with the
run signature, so the merge can verify Colab's run and the laptop finalising run accepts it.
This is an untouched test county: extraction only, no model runs on it.

## Files (upload all into `/content`)

| File | Size | What |
|---|---:|---|
| `bundle.json` | 1 KB | names, fingerprint, arguments, pinned versions |
| `code_9cd0447b.zip` | 14 KB | frozen `fieldvision` package |
| `fields_us_24013_2022.geojson` | 3.8 MB | byte-identical to `<FV_DATA>\data\labels\us_24013_2022\fields.geojson` (2,198 fields) |
| `check_blocks.zip` | 0.02 MB | finished Queen Anne's block `32618_US-E380N4320_0` (32 scenes, 3 fields) with its label-block fields and reference rows |
| `extract_us_24013_2022.ipynb` | 16 KB | open this in Colab |

No catalog or block files: Colab queries the scene lists itself over its fast link.

## Run order

Cells top to bottom as for Frederick: settings, Drive (recommended), pinned installs,
fingerprint + same-result check (stops on any difference), prepare folder, extraction
(`done/56`), result parts. If step 6 ends with "network read" scene errors, run step 6
again. Expected: 56 blocks, ~4,000-6,000 scene-block reads, about 20-35 min on 8 vCPU
(1.5-2.5 h on 2 vCPU).

## Back on the laptop

Follow section A of `<FV_DATA>\colab\README.md` with `RUN=us_24013_2022` (merge, then
finalise with the frozen code, then print the READY line). The county track writes the
toolbox copy and the READY line.
