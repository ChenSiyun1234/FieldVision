"""Synthetic silage / grain maize fields for testing the pipeline.

This is a software fixture, not a simulation anyone should trust: the
spectra are three hand-written endmembers (green canopy, dry plant
material, bare soil) mixed linearly. It exists so that the feature code,
the models and the evaluation can be exercised before real data arrives.
Scores obtained on it say nothing about real accuracy.

silage  green until an abrupt cut around day 250-275, little residue left
grain   gradual senescence, cut around day 285-320, stover left on the ground

Run as a script to write files in the same layout as the real data:
    python tests/synthetic.py --out <dir> --n-fields 1500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

#                  B02   B03   B04   B05   B06   B07   B08   B8A   B11   B12
GREEN = np.array([0.03, 0.06, 0.03, 0.10, 0.32, 0.42, 0.45, 0.47, 0.20, 0.09])
DRY = np.array([0.07, 0.11, 0.16, 0.20, 0.24, 0.27, 0.30, 0.31, 0.36, 0.24])
SOIL = np.array([0.07, 0.10, 0.14, 0.15, 0.17, 0.19, 0.20, 0.21, 0.30, 0.26])
# wet soil is darker, most of all in SWIR2
WET_DARKENING = np.array([0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.35, 0.45])

COUNTRIES = {
    #       silage share, phenology shift (days), chance a cloudy day follows a cloudy day
    "NL": dict(silage=0.70, shift=0.0, p_clear_after_cloud=0.28, lon=5.5, lat=52.2),
    "AT": dict(silage=0.35, shift=-8.0, p_clear_after_cloud=0.36, lon=15.5, lat=48.0),
    "FR": dict(silage=0.50, shift=-4.0, p_clear_after_cloud=0.40, lon=1.5, lat=47.5),
}


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _block_calendar(rng, p_clear_after_cloud, start=91, stop=345):
    """Acquisition days and cloud state shared by all fields of a block."""
    days = np.arange(start + rng.integers(0, 5), stop, 5)
    if rng.random() < 0.3:                       # second orbit
        days = np.sort(np.r_[days, days + rng.choice([2, 3])])
    clear = np.zeros(len(days), dtype=bool)
    state = rng.random() < 0.4
    for i, d in enumerate(days):
        p = 0.6 if state else p_clear_after_cloud
        if d > 305:                              # November is worse
            p *= 0.6
        state = rng.random() < p
        clear[i] = state
    wet = rng.random(len(days))                  # soil wetness per date
    return days, clear, wet


def _field_series(rng, cls, shift, days, block_clear, wet):
    greenup = rng.normal(160, 6) + shift
    cover_max = rng.uniform(0.85, 0.97)
    if cls == "silage":
        harvest = float(np.clip(rng.normal(262, 9) + shift, 236, 292))
        sen_start = harvest - 25
        dry_share = rng.uniform(0.03, 0.25)
        residue = rng.uniform(0.03, 0.15)
        regreen = rng.random() < 0.5
    else:
        harvest = float(np.clip(rng.normal(302, 11) + shift, 270, 335))
        sen_start = harvest - rng.uniform(35, 60)
        dry_share = rng.uniform(0.50, 0.85)
        residue = rng.uniform(0.40, 0.80)
        regreen = rng.random() < 0.15
    regreen_max = rng.uniform(0.25, 0.55)

    t = days.astype(float)
    cover = cover_max / (1 + np.exp(-0.12 * (t - greenup)))
    sen = dry_share * _smoothstep((t - sen_start) / (harvest - sen_start))
    f_green = cover * (1 - sen)
    f_dry = cover * sen
    after = t > harvest
    f_dry = np.where(after, residue, f_dry)
    f_green = np.where(after, 0.0, f_green)
    if regreen:
        f_green = np.where(after, regreen_max * np.clip((t - harvest - 10) / 35, 0, 1), f_green)
        f_dry = np.where(after, residue * (1 - f_green), f_dry)
    f_soil = 1 - f_green - f_dry

    soil = SOIL[None, :] * (1 - WET_DARKENING[None, :] * wet[:, None])
    refl = f_green[:, None] * GREEN + f_dry[:, None] * DRY + f_soil[:, None] * soil
    n = len(t)
    refl = refl * rng.normal(1, 0.04, (n, 1)) * rng.normal(1, 0.03, (n, len(BANDS)))
    refl = refl + rng.normal(0, 0.004, (n, len(BANDS)))

    # per-field clear fraction given the block's cloud state
    cf = np.where(block_clear,
                  np.where(rng.random(n) < 0.85, rng.uniform(0.85, 1.0, n), rng.uniform(0.4, 0.85, n)),
                  np.where(rng.random(n) < 0.8, 0.0, rng.uniform(0.05, 0.5, n)))
    # haze left over in partly cloudy scenes, and the odd cloud the mask missed
    haze = (1 - cf) * rng.uniform(0.0, 0.10, n) * (rng.random(n) < 0.5)
    haze = haze + np.where(rng.random(n) < 0.02, rng.uniform(0.05, 0.15, n), 0.0)
    refl = np.clip(refl + haze[:, None], 0.001, 1.0)
    refl[cf == 0] = np.nan

    keep = rng.random(n) > 0.04                  # some acquisitions never happen
    return t[keep], cf[keep], refl[keep], harvest


def make_dataset(n_fields=600, seed=0, countries=("NL", "AT", "FR"), year=2023,
                 blocks_per_country=12, label_noise=0.02, unlabelled_fraction=0.0):
    """Returns (fields, timeseries) as data frames following the data contracts.

    `fields` carries two extra columns that the pipeline must not use:
    true_class and true_harvest_doy.
    """
    rng = np.random.default_rng(seed)
    field_rows, ts_frames = [], []
    per_country = np.full(len(countries), n_fields // len(countries))
    per_country[: n_fields % len(countries)] += 1
    jan1 = pd.Timestamp(year=year, month=1, day=1)

    for country, n_c in zip(countries, per_country):
        par = COUNTRIES[country]
        blocks = []
        for b in range(blocks_per_country):
            days, clear, wet = _block_calendar(rng, par["p_clear_after_cloud"])
            blocks.append(dict(ix=b % 4, iy=b // 4, days=days, clear=clear, wet=wet,
                               shift=rng.normal(0, 4)))
        for k in range(n_c):
            b = int(rng.integers(0, blocks_per_country))
            blk = blocks[b]
            cls = "silage" if rng.random() < par["silage"] else "grain"
            t, cf, refl, harvest = _field_series(rng, cls, par["shift"] + blk["shift"],
                                                 blk["days"], blk["clear"], blk["wet"])
            label = cls
            if rng.random() < label_noise:
                label = "grain" if cls == "silage" else "silage"
            if rng.random() < unlabelled_fraction:
                label = None
            fid = f"{country}{year}_{k:05d}"
            area = float(np.round(rng.uniform(1.0, 12.0), 2))
            lon = par["lon"] + 0.14 * blk["ix"] + rng.uniform(0, 0.13)
            lat = par["lat"] + 0.09 * blk["iy"] + rng.uniform(0, 0.08)
            field_rows.append(dict(
                field_id=fid, label=label, year=year, country=country, source="synthetic",
                crop_code="SYN", area_ha=area, block_id=f"{country}-{blk['ix']:03d}-{blk['iy']:03d}",
                centroid_lon=round(lon, 5), centroid_lat=round(lat, 5),
                true_class=cls, true_harvest_doy=round(harvest, 1)))
            n_pix = int(area * 100)
            frame = pd.DataFrame(refl, columns=BANDS)
            frame.insert(0, "field_id", fid)
            frame.insert(1, "date", jan1 + pd.to_timedelta(t - 1, unit="D"))
            frame.insert(2, "scene_id", [f"S2_{country}_{int(d):03d}" for d in t])
            frame.insert(3, "collection", "sentinel-2-l2a")
            frame.insert(4, "n_pixels", n_pix)
            frame.insert(5, "n_clear", np.round(cf * n_pix).astype(int))
            frame.insert(6, "clear_fraction", cf)
            ts_frames.append(frame)

    fields = pd.DataFrame(field_rows)
    ts = pd.concat(ts_frames, ignore_index=True)
    return fields, ts


def fields_to_geojson(fields):
    feats = []
    for row in fields.to_dict("records"):
        lon, lat = row["centroid_lon"], row["centroid_lat"]
        d = 0.001
        ring = [[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d], [lon - d, lat + d],
                [lon - d, lat - d]]
        feats.append({"type": "Feature", "properties": row,
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return {"type": "FeatureCollection", "features": feats}


def write_dataset(out_dir, **kwargs):
    """Write one GeoJSON and one Parquet per country and year. Returns (fields_dir, ts_dir)."""
    out_dir = Path(out_dir)
    fields, ts = make_dataset(**kwargs)
    fdir, tdir = out_dir / "fields", out_dir / "timeseries"
    fdir.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    for (country, year), part in fields.groupby(["country", "year"]):
        with open(fdir / f"fields_{country}_{year}.geojson", "w", encoding="utf-8") as fh:
            json.dump(fields_to_geojson(part), fh)
        ts[ts["field_id"].isin(part["field_id"])].to_parquet(
            tdir / f"timeseries_{country}_{year}.parquet", index=False)
    return fdir, tdir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-fields", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks-per-country", type=int, default=12)
    ap.add_argument("--unlabelled-fraction", type=float, default=0.0)
    a = ap.parse_args()
    fdir, tdir = write_dataset(a.out, n_fields=a.n_fields, seed=a.seed,
                               blocks_per_country=a.blocks_per_country,
                               unlabelled_fraction=a.unlabelled_fraction)
    print(f"fields: {fdir}\ntimeseries: {tdir}")
