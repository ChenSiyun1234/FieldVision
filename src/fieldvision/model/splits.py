"""Train/test splits that respect space and time.

Neighbouring fields share weather, soil and satellite passes, so a field
and its neighbour must never sit on opposite sides of a split. Every split
here is made from whole `block_id` groups (10 km cells) or whole
countries / years. Nothing is split at random field by field.
"""
from __future__ import annotations

import fnmatch
import math

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def spatial_folds(blocks, y, n_splits=5, seed=0):
    """Group k-fold over blocks, stratified by label as far as blocks allow.

    Returns a list of (train_idx, test_idx). The number of folds is capped
    at the number of blocks.
    """
    blocks = np.asarray(blocks).astype(str)
    y = np.asarray(y).astype(int)
    n_blocks = len(np.unique(blocks))
    k = min(n_splits, n_blocks)
    if k < 2:
        raise ValueError(f"spatial CV needs at least 2 blocks, got {n_blocks}")
    if len(np.unique(y)) < 2:
        raise ValueError("spatial CV needs both classes in the labelled data")
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [(tr, te) for tr, te in cv.split(np.zeros(len(y)), y, groups=blocks)]
    for tr, te in folds:
        assert_disjoint(blocks, tr, te)
    return folds


def leave_one_group_out(groups, blocks=None, exclude_test_blocks=False, min_test=1):
    """One split per distinct group value: train on the rest, test on it.

    groups               country or year per field
    exclude_test_blocks  also drop training fields whose block appears in the
                         test group (for leave-one-year-out this removes the
                         same places seen in another year)
    Returns a list of (group_value, train_idx, test_idx, n_dropped).
    """
    groups = np.asarray(groups).astype(str)
    out = []
    values = sorted(np.unique(groups))
    if len(values) < 2:
        return out
    for g in values:
        te = np.flatnonzero(groups == g)
        tr_mask = groups != g
        n_dropped = 0
        if exclude_test_blocks and blocks is not None:
            b = np.asarray(blocks).astype(str)
            clash = tr_mask & np.isin(b, np.unique(b[te]))
            n_dropped = int(clash.sum())
            tr_mask &= ~clash
        if len(te) < min_test or not tr_mask.any():
            continue
        out.append((g, np.flatnonzero(tr_mask), te, n_dropped))
    return out


def holdout_mask(blocks, patterns):
    """True for fields whose block_id matches any of the patterns (fnmatch, e.g. 'AT-E46*')."""
    blocks = np.asarray(blocks).astype(str)
    if not patterns:
        return np.zeros(len(blocks), dtype=bool)
    mask = np.zeros(len(blocks), dtype=bool)
    for pat in patterns:
        mask |= np.array([fnmatch.fnmatchcase(b, pat) for b in blocks])
    return mask


def assert_disjoint(blocks, train_idx, test_idx):
    blocks = np.asarray(blocks).astype(str)
    shared = set(blocks[train_idx]) & set(blocks[test_idx])
    if shared:
        raise AssertionError(f"blocks on both sides of a split: {sorted(shared)[:5]}")


def grid_block_id(lon, lat, size_km=10.0, prefix="G"):
    """A rough 10 km grid cell for fields that came without a block_id.

    Uses a sinusoidal approximation, which is enough for grouping
    neighbours; it does not match the label fetcher's EPSG:3035 grid.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    km_per_deg = 111.32
    y = np.floor(lat * km_per_deg / size_km)
    x = np.floor(lon * km_per_deg * np.cos(np.radians(lat)) / size_km)
    out = np.empty(len(lon), dtype=object)
    for i, (a, b) in enumerate(zip(x, y)):
        out[i] = "unknown" if not (math.isfinite(a) and math.isfinite(b)) else f"{prefix}{int(a)}_{int(b)}"
    return out
