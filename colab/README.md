# Colab notebooks

Everything here runs in a plain Google Colab CPU runtime. No account, key or token of ours is
needed: imagery comes from the public Earth Search catalog and from Microsoft Planetary
Computer, which hands out anonymous read tokens.

Open any notebook straight from GitHub:
`https://colab.research.google.com/github/ChenSiyun1234/FieldVision/blob/main/colab/<path>.ipynb`

## Look at fields: `view_field_images.ipynb`

<https://colab.research.google.com/github/ChenSiyun1234/FieldVision/blob/main/colab/view_field_images.ipynb>

Picture strip for one field at a time: six Sentinel-2 true-colour chips across the season (two
summer looks, the last clear look before the cut, the first after, two later) with the field
outline, and the field's greenness (NDVI) curve below with the chip dates marked.

Inputs for one county-year, uploaded in step 2 or read from a folder (`COUNTY_DIR`, e.g. on
Google Drive):

* `fields.geojson`: field outlines (the label/field file the extraction used);
* `timeseries.parquet`: the extracted per-field series of that county-year;
* optional, a per-field calls file (`*_module_v2.parquet`, or any parquet/CSV with `field_id`,
  `label`, `abstain` and either an `evidence` JSON column or `evidence_tier`, `drop_date`,
  `last_standing_date`). With it you can filter (called silage/grain/abstained, evidence tier,
  drop date range) and the cut window is shaded on the curve.

Step 4 picks a field by id, at random, or at random among those matching the filter; step 5
draws it (about six small image reads, a few MB) and saves the PNG under `/content/strips`.

## Extraction notebooks: `extraction/<run>/`

One folder per extraction that was run on Colab (2026-11-14), each with its notebook and the
README that went with it. They are kept as the record of how each data set was made and as
templates; the data files they need are not in git. Each notebook expects, in `/content`, the
files its bundle builder wrote:

| Kind | Folders | Files the notebook expects |
|---|---|---|
| Fresh optical run (first version) | `nl_2025` | `bundle.json`, `code_<fp>.zip`, `fields_<run>.geojson`, `catalog_<run>.zip.NNN`, `check_blocks.zip` |
| Continue / fresh optical run | `fr_2022_rest`, `us_240*_2022`, `us_24_2025_s*`, `us_26*_2022`, `us_42071_2023`, `us_55*_2023` | `bundle.json`, `code_<fp>.zip`, `fields_<run>.geojson`, `check_blocks.zip`, and when the laptop had started the run `blocks_<run>.zip.NNN` and `catalog_<run>.zip.NNN` |
| Radar pilot (Sentinel-1 RTC) | `s1_24029_2022` | `bundle.json`, `code_<fp>.zip`, `fields_<run>.geojson`, `check_blocks.zip` |
| Field frames (boundaries only) | `state_frames_2025` | see its README |

Every extraction notebook runs the same steps: settings, optional Drive mount (checkpoints copied
every 5 minutes), pinned package installs, code fingerprint check, a same-result check that
re-extracts blocks already extracted elsewhere and stops on any difference, the extraction
(resumable: rerun the cell), and the result parts with a SHA-256 list.

How the bundles are made and brought back (repository scripts):

* `scripts/build_colab_run.py`: bundle for a run (`--product s2` optical, `--product s1` radar;
  continues a laptop run from its checkpoints, or starts a fresh one from a template or the
  extractor defaults);
* `scripts/build_colab_bundle.py`: the first, fresh-run bundle format (`nl_2025`);
* `scripts/merge_colab_blocks.py`: checks the parts' checksums, the code fingerprint and the run
  signature, and merges the blocks into the run folder; `extraction/BRING_BACK.md` has the steps.

Placeholders in the READMEs: `<FV_DATA>` is the data/work folder the team used
(`<FV_DATA>/data/timeseries/<run>`, `<FV_DATA>/venv`, frozen code copies), `<REPO>` the
repository checkout, `<HOME>` a user folder.

Not here: the `stream_px` / `stream_train` notebooks of the separate stream experiment.
