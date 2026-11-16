import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope="session")
def synth():
    """A small synthetic dataset shared by several tests: (fields, timeseries)."""
    from synthetic import make_dataset
    return make_dataset(n_fields=450, seed=3, blocks_per_country=8)


@pytest.fixture(scope="session")
def synth_features(synth):
    from fieldvision.features.build import build_features
    fields, ts = synth
    X, info = build_features(ts, fields)
    return X, info
