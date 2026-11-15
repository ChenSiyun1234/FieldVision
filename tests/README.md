# tests

Tests for the model package (`src/fieldvision`). Run `pytest -q tests` from the repository root; they use small
synthetic data (`synthetic.py`) and need no network. Tests needing optional packages (torch, rasterio extras)
skip themselves when those are missing.

The agent side has its own tests next to the code; run them with `fieldvision test`.
