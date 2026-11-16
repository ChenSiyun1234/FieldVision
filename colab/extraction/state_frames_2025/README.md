# State frames 2025 (PENNSYLVANIA, WISCONSIN): upload code_frames_a14bc9b.zip, nass_2025_state_targets.csv, state_frames_2025.ipynb to /content; run all cells top to bottom (1-6).

Boundaries only (CSB 2018-2025 rev23), no imagery, nothing scored. Returns frames_<fips>_2025.zip + _SHA256SUMS.txt per state: s1..s3/fields.geojson (150 read blocks, seed 42), draw.json, state_acres.json. Unzip to <FV_DATA>\data\labels\us_<fips>_2025\, then build one extraction bundle per part with build_colab_run.py (template <FV_DATA>\colab\_templates\9cd0447b_2025).
