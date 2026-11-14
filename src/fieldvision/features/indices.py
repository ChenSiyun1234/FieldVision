"""Spectral indices from Sentinel-2 surface reflectance (0-1).

All functions are pure: arrays in, arrays out. Where the denominator is
zero or an input is NaN the result is NaN, never 0.
"""
from __future__ import annotations

import numpy as np

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
INDEX_NAMES = ["ndvi", "ndmi", "evi", "nbr", "ndre", "psri", "ndti"]


def _safe_ratio(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(out) & (np.abs(den) > 1e-6), out, np.nan)


def normalized_difference(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return _safe_ratio(a - b, a + b)


def ndvi(b08, b04):
    """Greenness: (NIR - red) / (NIR + red)."""
    return normalized_difference(b08, b04)


def ndmi(b8a, b11):
    """Canopy water content: (narrow NIR - SWIR1) / (narrow NIR + SWIR1)."""
    return normalized_difference(b8a, b11)


def evi(b08, b04, b02):
    """Enhanced vegetation index. NaN when the denominator is not positive."""
    nir = np.asarray(b08, dtype=float)
    red = np.asarray(b04, dtype=float)
    blue = np.asarray(b02, dtype=float)
    den = nir + 6.0 * red - 7.5 * blue + 1.0
    out = _safe_ratio(2.5 * (nir - red), den)
    return np.where(den > 0, out, np.nan)


def nbr(b8a, b12):
    """Normalized burn ratio (NIR vs SWIR2). Drops when green biomass is removed."""
    return normalized_difference(b8a, b12)


def ndre(b8a, b05):
    """Red-edge index: (narrow NIR - red edge 1) / (narrow NIR + red edge 1)."""
    return normalized_difference(b8a, b05)


def psri(b04, b02, b06):
    """Plant senescence reflectance index: (red - blue) / red edge 2.

    Rises as chlorophyll breaks down and the canopy yellows.
    """
    return _safe_ratio(np.asarray(b04, dtype=float) - np.asarray(b02, dtype=float), b06)


def ndti(b11, b12):
    """Normalized difference tillage index: (SWIR1 - SWIR2) / (SWIR1 + SWIR2).

    Higher over dry crop residue than over bare soil.
    """
    return normalized_difference(b11, b12)


def compute_indices(bands):
    """All indices from a mapping band name -> array. Returns dict name -> array."""
    return {
        "ndvi": ndvi(bands["B08"], bands["B04"]),
        "ndmi": ndmi(bands["B8A"], bands["B11"]),
        "evi": evi(bands["B08"], bands["B04"], bands["B02"]),
        "nbr": nbr(bands["B8A"], bands["B12"]),
        "ndre": ndre(bands["B8A"], bands["B05"]),
        "psri": psri(bands["B04"], bands["B02"], bands["B06"]),
        "ndti": ndti(bands["B11"], bands["B12"]),
    }
