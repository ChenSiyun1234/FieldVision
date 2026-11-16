"""Indices from reflectance tables; zero denominators return NaN.

PSRI uses (B04 - B02) / B06: a Sentinel-2 approximation to the
plant senescence reflectance index, whose reference wavelengths are
680, 500 and 750 nm. These are indices of band medians, not median indices.
"""
import numpy as np


def _ratio(numerator, denominator):
    return numerator / np.where(np.abs(denominator) > 1e-12, denominator, np.nan)


def ndvi(table):
    return _ratio(table["B08"] - table["B04"], table["B08"] + table["B04"])


def ndmi(table):
    return _ratio(table["B8A"] - table["B11"], table["B8A"] + table["B11"])


def evi(table):
    return _ratio(2.5 * (table["B08"] - table["B04"]),
                  table["B08"] + 6 * table["B04"] - 7.5 * table["B02"] + 1)


def nbr(table):
    return _ratio(table["B08"] - table["B12"], table["B08"] + table["B12"])


def psri(table):
    return _ratio(table["B04"] - table["B02"], table["B06"])


def add_indices(table):
    result = table.copy()
    for name, function in (("NDVI", ndvi), ("NDMI", ndmi), ("EVI", evi), ("NBR", nbr), ("PSRI", psri)):
        result[name] = function(table)
    return result
