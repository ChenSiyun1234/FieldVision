"""From our long per-field time series to Presto's input tensors.

Presto was pre-trained on monthly composites: one value per band per
calendar month, 12 months, with Sentinel-1, Sentinel-2, ERA5 weather,
SRTM terrain and Dynamic World land cover. Every channel group can be
masked, and masked tokens are dropped before the transformer, so absent
inputs never have to be invented. What we give it:

- time steps   one per calendar month from `start_month` to `end_month` of the
               field's season year (default April-November, the extraction
               window: 8 steps). `month` is the 0-based index of the first one.
- Sentinel-2   median of each band over the field's usable acquisitions in
               that month (usable = same rule as the feature code:
               features.season.usable_mask). A month without any usable
               acquisition is masked for all S2 groups and NDVI.
- scaling      Presto divides Sentinel-2 digital numbers by 10,000, so its
               model input is reflectance on a 0-1 scale. Our table is already
               0-1 surface reflectance, so the values go in unchanged.
               Presto's own pre-training data was top-of-atmosphere (L1C);
               ours is surface reflectance (L2A). See docs/foundation-models.md.
- NDVI         (B8 - B4) / (B8 + B4) of the composite, as Presto computes it
- S1, ERA5,
  SRTM         masked (we do not have them)
- Dynamic World class 9 = "no data" for every month (masked by the model)
- lat/lon      the field centroid in degrees, [lat, lon] order
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.indices import BANDS as OUR_BANDS
from ..features.season import FeatureConfig, usable_mask
from .presto_bands import BAND_GROUPS, DW_MISSING, PRESTO_BANDS

# our column -> Presto band name
OUR_TO_PRESTO = {"B02": "B2", "B03": "B3", "B04": "B4", "B05": "B5", "B06": "B6", "B07": "B7",
                 "B08": "B8", "B8A": "B8A", "B11": "B11", "B12": "B12"}
S2_GROUPS = ["S2_RGB", "S2_Red_Edge", "S2_NIR_10m", "S2_NIR_20m", "S2_SWIR", "NDVI"]


@dataclass
class PrestoBatch:
    """Inputs for N fields. Arrays are numpy; convert with torch.from_numpy."""
    field_id: np.ndarray        # [N] str
    x: np.ndarray               # [N, T, 17] float32, Presto channel order
    mask: np.ndarray            # [N, T, 17] float32, 1 = missing
    dynamic_world: np.ndarray   # [N, T] int64
    latlons: np.ndarray         # [N, 2] float32, [lat, lon]
    month: int                  # 0-based month of the first time step
    n_obs: np.ndarray           # [N, T] usable acquisitions per month

    @property
    def n_months_observed(self):
        return (self.n_obs > 0).sum(axis=1)


def monthly_composites(ts, fields, start_month=4, end_month=11, cfg=FeatureConfig()):
    """Median reflectance per field and calendar month.

    ts      long table: field_id, date, clear_fraction, B02 ... B12 (0-1)
    fields  frame with field_id and year (the season year); output rows
            follow its order. A field without rows gets all-NaN months.

    Returns (values [N, T, 10] in OUR_BANDS order, n_obs [N, T]).
    """
    if not 1 <= start_month <= end_month <= 12:
        raise ValueError("need 1 <= start_month <= end_month <= 12")
    T = end_month - start_month + 1
    fid = fields["field_id"].astype(str).to_numpy()
    pos = pd.Series(np.arange(len(fid)), index=fid)
    year_of = pd.Series(pd.to_numeric(fields["year"], errors="coerce").to_numpy(), index=fid)
    values = np.full((len(fid), T, len(OUR_BANDS)), np.nan)
    n_obs = np.zeros((len(fid), T), dtype=int)
    if len(ts) == 0:
        return values, n_obs

    t = ts[ts["field_id"].astype(str).isin(pos.index)].copy()
    t["field_id"] = t["field_id"].astype(str)
    bands = t[OUR_BANDS].to_numpy(dtype=float)
    n_clear = t["n_clear"].to_numpy(dtype=float) if "n_clear" in t.columns else None
    ok = usable_mask(t["clear_fraction"].to_numpy(dtype=float), bands, cfg, n_clear)
    date = pd.to_datetime(t["date"])
    season = year_of.reindex(t["field_id"]).to_numpy()
    step = date.dt.month.to_numpy() - start_month
    ok &= (date.dt.year.to_numpy() == season) & (step >= 0) & (step < T)
    t = t.loc[ok, ["field_id"] + OUR_BANDS].copy()
    t["step"] = step[ok]
    if t.empty:
        return values, n_obs
    med = t.groupby(["field_id", "step"])[OUR_BANDS].median()
    cnt = t.groupby(["field_id", "step"]).size()
    rows = pos.reindex(med.index.get_level_values(0)).to_numpy()
    steps = med.index.get_level_values(1).to_numpy()
    values[rows, steps] = med.to_numpy()
    n_obs[pos.reindex(cnt.index.get_level_values(0)).to_numpy(),
          cnt.index.get_level_values(1).to_numpy()] = cnt.to_numpy()
    return values, n_obs


def to_presto(values, n_obs, field_id, lat, lon, start_month=4):
    """Monthly composites -> PrestoBatch (normalised exactly as Presto does)."""
    N, T, _ = values.shape
    x = np.zeros((N, T, len(PRESTO_BANDS)), dtype=np.float32)
    mask = np.ones((N, T, len(PRESTO_BANDS)), dtype=np.float32)
    have = np.isfinite(values).all(axis=2)              # [N, T]
    for j, ours in enumerate(OUR_BANDS):
        k = PRESTO_BANDS.index(OUR_TO_PRESTO[ours])
        # Presto: (DN + 0) / 1e4 == reflectance; ours is already reflectance
        x[:, :, k] = np.where(have, values[:, :, j], 0.0)
        mask[:, :, k] = np.where(have, 0.0, 1.0)
    b8 = x[:, :, PRESTO_BANDS.index("B8")]
    b4 = x[:, :, PRESTO_BANDS.index("B4")]
    s = b8 + b4
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(s > 0, (b8 - b4) / s, 0.0)
    k = PRESTO_BANDS.index("NDVI")
    x[:, :, k] = np.where(have, ndvi, 0.0)
    mask[:, :, k] = np.where(have, 0.0, 1.0)
    latlons = np.column_stack([np.asarray(lat, dtype=np.float32),
                               np.asarray(lon, dtype=np.float32)])
    return PrestoBatch(field_id=np.asarray(field_id).astype(str), x=x, mask=mask,
                       dynamic_world=np.full((N, T), DW_MISSING, dtype=np.int64),
                       latlons=latlons, month=start_month - 1,
                       n_obs=np.where(have, n_obs, 0))


def build_presto_inputs(ts, fields, start_month=4, end_month=11, cfg=FeatureConfig()):
    """Long time series + field table -> PrestoBatch, rows in the order of `fields`."""
    values, n_obs = monthly_composites(ts, fields, start_month, end_month, cfg)
    lat = pd.to_numeric(fields["centroid_lat"], errors="coerce").to_numpy()
    lon = pd.to_numeric(fields["centroid_lon"], errors="coerce").to_numpy()
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("every field needs centroid_lat/centroid_lon for Presto's location token")
    return to_presto(values, n_obs, fields["field_id"], lat, lon, start_month)


def masked_groups(batch):
    """Which channel groups are masked at each step, as Presto will see it ([N, T] per group)."""
    return {g: batch.mask[:, :, idx].max(axis=2) for g, idx in BAND_GROUPS.items()}
