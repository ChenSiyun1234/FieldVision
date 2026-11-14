"""Synthetic sample regions so the demo shows something before real model output exists.

    python -m fieldvision.app.sample_data            # writes app/data/<region>/ for every sample region
    python -m fieldvision.app.sample_data --only at-2024 --n 40

It writes the same inputs the real pipeline produces (fields GeoJSON, long-format
Sentinel-2 Parquet, per-field predictions, predictions at early cutoff dates, and
for the US a county table) into a
work folder, then runs the normal export on them. Every region is marked synthetic,
and the app shows a "sample data" badge for it. Nothing here is a result.

The "model" is a toy: it reads the observed curve around the true harvest day and
turns greenness, moisture, timing and the ground after the cut into a probability,
so the curves, the harvest band and the prediction in the app agree with each other.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import export

DEFAULT_WORK = Path(r"C:\t\fv\data\app\sample")
DEFAULT_APP_DATA = Path(__file__).resolve().parents[3] / "app" / "data"

# id, title, group, country, year, centre (lon, lat), n fields, silage share,
# labels?, predictions?, counties?, mean silage / grain harvest day of year
REGIONS = [
    dict(id="at-2024", title="Austria 2024", group="Austria", country="AT", year=2024,
         centre=(13.62, 48.16), n=150, silage=0.4, labels=True, preds=True, counties=False,
         h_sil=258, h_grn=296, field_ha=(1.5, 9)),
    dict(id="nl-2025", title="Netherlands 2025", group="Netherlands", country="NL", year=2025,
         centre=(6.42, 52.27), n=70, silage=0.75, labels=True, preds=True, counties=False,
         h_sil=262, h_grn=300, field_ha=(1.5, 7)),
    dict(id="fr-2022", title="France 2022", group="France", country="FR", year=2022,
         centre=(-1.46, 48.08), n=70, silage=0.6, labels=True, preds=True, counties=False,
         h_sil=250, h_grn=290, field_ha=(2, 10)),
    dict(id="barc-2021", title="BARC Maryland 2021", group="BARC Maryland", country="US", year=2021,
         centre=(-76.905, 39.035), n=32, silage=0.4, labels=True, preds=True, counties=False,
         h_sil=252, h_grn=292, field_ha=(1.5, 8), farm="barc"),
    dict(id="barc-2022", title="BARC Maryland 2022", group="BARC Maryland", country="US", year=2022,
         centre=(-76.905, 39.035), n=32, silage=0.45, labels=True, preds=True, counties=False,
         h_sil=255, h_grn=295, field_ha=(1.5, 8), farm="barc"),
    dict(id="barc-2023", title="BARC Maryland 2023", group="BARC Maryland", country="US", year=2023,
         centre=(-76.905, 39.035), n=32, silage=0.4, labels=False, preds=False, counties=False,
         h_sil=254, h_grn=294, field_ha=(1.5, 8), farm="barc"),
    dict(id="us-2022", title="US counties 2022", group="US counties", country="US", year=2022,
         centre=(-89.75, 43.55), n=150, silage=None, labels=False, preds=True, counties=True,
         h_sil=252, h_grn=292, field_ha=(8, 40)),
]

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


# ---------------------------------------------------------------- geometry

def _rect(lon, lat, area_ha, aspect, angle):
    """Rotated rectangle around (lon, lat) as a GeoJSON polygon ring."""
    a = area_ha * 1e4
    w = math.sqrt(a * aspect)
    h = a / w
    kx = 111320.0 * math.cos(math.radians(lat))
    ky = 110540.0
    ca, sa = math.cos(angle), math.sin(angle)
    ring = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2), (-w / 2, -h / 2)):
        x = dx * ca - dy * sa
        y = dx * sa + dy * ca
        ring.append([round(lon + x / kx, 6), round(lat + y / ky, 6)])
    return ring, w, h


def make_layout(rng, centre, n, field_ha, n_blocks, spread_km, block_km):
    """Fields clustered in a few blocks, no overlaps (centroid spacing check)."""
    lon0, lat0 = centre
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110540.0
    blocks = []
    for _ in range(n_blocks):
        blocks.append((rng.uniform(-spread_km, spread_km) * 1000, rng.uniform(-spread_km, spread_km) * 1000))
    placed = []
    tries = 0
    while len(placed) < n and tries < n * 400:
        tries += 1
        bx, by = blocks[len(placed) % n_blocks]
        x = bx + rng.normal(0, block_km * 1000 / 2)
        y = by + rng.normal(0, block_km * 1000 / 2)
        area = float(np.exp(rng.uniform(math.log(field_ha[0]), math.log(field_ha[1]))))
        aspect = float(rng.uniform(1.0, 2.8))
        radius = math.sqrt(area * 1e4 * aspect) / 2 * 1.5 + 25
        if any((x - px) ** 2 + (y - py) ** 2 < (radius + pr) ** 2 for px, py, pr, *_ in placed):
            continue
        angle = float(rng.choice([0.0, math.pi / 2]) + rng.normal(0, 0.15))
        placed.append((x, y, radius, area, aspect, angle))
    out = []
    for k, (x, y, _, area, aspect, angle) in enumerate(placed):
        lon, lat = lon0 + x / kx, lat0 + y / ky
        ring, _, _ = _rect(lon, lat, area, aspect, angle)
        block = f"B{k % n_blocks + 1:02d}"
        out.append(dict(lon=lon, lat=lat, area=area, ring=ring, block=block))
    return out


# ---------------------------------------------------------------- curves

def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def field_truth(rng, cls, h_sil, h_grn):
    """Season parameters for one field. About 1 in 12 is atypical (late dry silage, early green grain).

    Greenness and moisture at the cut follow how the field looks (typical or not); the ground
    after the cut follows what really happened: chopping removes the whole plant and leaves
    bare soil (low NDTI), combining leaves stalks on the ground (higher NDTI)."""
    atypical = rng.random() < 0.08
    p = {"cls": cls, "atypical": atypical}
    p["greenup"] = rng.normal(158, 7)
    p["peak_ndvi"] = rng.uniform(0.83, 0.92)
    p["peak_ndmi"] = rng.uniform(0.34, 0.44)
    p["peak_ndti"] = rng.uniform(0.28, 0.34)
    p["soil_ndvi"] = rng.uniform(0.14, 0.21)
    p["soil_ndmi"] = rng.uniform(-0.16, -0.08)
    p["soil_ndti"] = rng.uniform(0.04, 0.09)
    silage_like = (cls == "silage") != atypical
    if cls == "silage":
        p["harvest"] = rng.normal(h_grn - 12, 6) if atypical else rng.normal(h_sil, 8)
        p["post_ndti"] = rng.uniform(0.02, 0.08)
    else:
        p["harvest"] = rng.normal(h_sil + 12, 6) if atypical else rng.normal(h_grn, 10)
        p["post_ndti"] = rng.uniform(0.12, 0.2)
    if silage_like:
        p["pre_ndvi"] = rng.uniform(0.66, 0.84)
        p["pre_ndmi"] = rng.uniform(0.16, 0.32)
        p["pre_ndti"] = rng.uniform(0.24, 0.3)
        p["sen_days"] = rng.uniform(15, 30)
        p["post_ndvi"] = rng.uniform(0.15, 0.22)
        p["post_ndmi"] = rng.uniform(-0.17, -0.09)
    else:
        p["pre_ndvi"] = rng.uniform(0.32, 0.52)
        p["pre_ndmi"] = rng.uniform(-0.06, 0.1)
        p["pre_ndti"] = rng.uniform(0.16, 0.22)
        p["sen_days"] = rng.uniform(35, 55)
        p["post_ndvi"] = rng.uniform(0.21, 0.3)
        p["post_ndmi"] = rng.uniform(-0.12, -0.04)
    return p


def curve(p, day):
    day = np.asarray(day, dtype=float)
    rise = 1 / (1 + np.exp(-(day - p["greenup"]) / 7))
    sen0 = p["harvest"] - p["sen_days"]
    fade = _smooth((day - sen0) / p["sen_days"])
    after = day >= p["harvest"]
    out = []
    for k in ("ndvi", "ndmi", "ndti"):
        v = p[f"soil_{k}"] + (p[f"peak_{k}"] - p[f"soil_{k}"]) * rise - (p[f"peak_{k}"] - p[f"pre_{k}"]) * fade
        out.append(np.where(after, p[f"post_{k}"], v))
    return out


def bands_from(ndvi, ndmi, ndti, rng):
    """Reflectances that give back the requested NDVI (B08/B04), NDMI (B8A/B11) and NDTI (B11/B12)."""
    nir = 0.18 + 0.2 * np.clip(ndvi, 0, 1) + rng.normal(0, 0.01, np.shape(ndvi))
    b04 = nir * (1 - ndvi) / (1 + ndvi)
    b8a = nir * 1.02
    b11 = b8a * (1 - ndmi) / (1 + ndmi)
    b12 = b11 * (1 - ndti) / (1 + ndti)
    return {
        "B02": b04 * 0.75, "B03": b04 * 1.15, "B04": b04, "B05": b04 * 1.3 + 0.02,
        "B06": nir * 0.75, "B07": nir * 0.92, "B08": nir, "B8A": b8a, "B11": b11, "B12": b12,
    }


def acquisitions(rng, year):
    """Revisit dates Apr 1 to Nov 30: a 5-day cycle plus some overlap passes."""
    start = pd.Timestamp(year=year, month=4, day=1).day_of_year
    end = pd.Timestamp(year=year, month=11, day=30).day_of_year
    days = list(range(start + int(rng.integers(0, 5)), end + 1, 5))
    days += [d + 2 for d in days if rng.random() < 0.3 and d + 2 <= end]
    return np.array(sorted(set(days)))


def weather(rng, days):
    """Regional cloud cover per acquisition with a few long overcast spells."""
    cover = np.clip(rng.beta(0.9, 2.6, len(days)), 0, 1)
    # two long spells in spring/summer, one short one anywhere
    spells = [(rng.uniform(days[0] + 20, 220), rng.uniform(14, 24)) for _ in range(2)]
    spells.append((rng.uniform(days[0] + 20, days[-1] - 10), rng.uniform(6, 12)))
    for mid, width in spells:
        cover = np.where(np.abs(days - mid) < width / 2, np.maximum(cover, rng.uniform(0.85, 1.0, len(days))), cover)
    return cover


def cutoff_days(year):
    """Early-season cutoffs: every two weeks from Aug 1 to early November (day of year)."""
    start = pd.Timestamp(year=year, month=8, day=1).day_of_year
    return [start + 14 * k for k in range(8)]


# ---------------------------------------------------------------- one region

def simulate_region(cfg, seed, n=None):
    rng = np.random.default_rng(seed)
    n = n or cfg["n"]
    year = cfg["year"]
    layout_rng = np.random.default_rng(1000 + (7 if cfg.get("farm") == "barc" else seed))
    if cfg["counties"]:
        layout = make_layout(layout_rng, cfg["centre"], n, cfg["field_ha"], n_blocks=10, spread_km=40, block_km=8)
    elif cfg.get("farm") == "barc":
        layout = make_layout(layout_rng, cfg["centre"], n, cfg["field_ha"], n_blocks=4, spread_km=1.6, block_km=0.9)
    else:
        layout = make_layout(layout_rng, cfg["centre"], n, cfg["field_ha"], n_blocks=5, spread_km=9, block_km=2.5)
    n = len(layout)

    county_shapes, county_of, share_of, county_names = None, [None] * n, {}, {}
    if cfg["counties"]:
        county_shapes, county_of, share_of, county_names = _counties(rng, layout)

    days = acquisitions(rng, year)
    cover = weather(rng, days)
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    cuts = cutoff_days(year)

    features, rows, preds, asof, truths = [], [], [], [], []
    special = rng.permutation(n)
    no_clear = set(special[:2].tolist())                 # nothing usable all season
    gap_at_harvest = set(special[2:7].tolist())          # long overcast spell around harvest
    unlabelled = set(special[7:10].tolist()) if cfg["labels"] else set()

    for i, f in enumerate(layout):
        share = share_of[county_of[i]] if cfg["counties"] else cfg["silage"]
        cls = "silage" if rng.random() < share else "grain"
        truth = field_truth(rng, cls, cfg["h_sil"], cfg["h_grn"])
        truths.append((cls, f["area"], county_of[i]))
        fid = f"{cfg['id'].replace('-', '')}-s{i:04d}"
        label = cls if cfg["labels"] and i not in unlabelled else None
        props = {
            "field_id": fid, "label": label, "year": year, "country": cfg["country"],
            "source": "synthetic", "crop_code": None, "area_ha": round(f["area"], 2), "block_id": f["block"],
            "centroid_lon": round(f["lon"], 6), "centroid_lat": round(f["lat"], 6),
        }
        if county_of[i] is not None:
            props["county_fips"] = county_of[i]
        features.append({"type": "Feature", "properties": props,
                         "geometry": {"type": "Polygon", "coordinates": [f["ring"]]}})

        ndvi, ndmi, ndti = curve(truth, days)
        cf = np.clip(1 - cover + rng.normal(0, 0.12, len(days)), 0, 1)
        if i in gap_at_harvest:
            cf = np.where(np.abs(days - truth["harvest"]) < rng.uniform(14, 22), 0.0, cf)
        if i in no_clear:
            cf = np.where(rng.random(len(days)) < 0.85, 0.0, np.minimum(cf, 0.35))
        cf = np.where(cf < 0.04, 0.0, cf)
        # thin cloud and haze pull NDVI down and scramble NDMI on partly clear scenes
        haze = (1 - cf) * rng.uniform(0.0, 0.5, len(days))
        obs_ndvi = ndvi - haze * (ndvi - 0.05) + rng.normal(0, 0.018, len(days))
        obs_ndmi = ndmi - haze * 0.25 + rng.normal(0, 0.02, len(days))
        obs_ndti = ndti - haze * 0.1 + rng.normal(0, 0.012, len(days))
        b = bands_from(np.clip(obs_ndvi, -0.2, 0.97), np.clip(obs_ndmi, -0.5, 0.8), np.clip(obs_ndti, -0.3, 0.6), rng)
        n_pix = int(round(f["area"] * 100 * 0.7))
        for k, d in enumerate(days):
            n_clear = int(round(n_pix * cf[k]))
            row = {"field_id": fid, "date": jan1 + pd.Timedelta(days=int(d) - 1), "scene_id": f"S2_{year}_{d:03d}",
                   "collection": "synthetic", "n_pixels": n_pix, "n_clear": n_clear,
                   "clear_fraction": round(n_clear / n_pix, 3) if n_pix else 0.0}
            for name in BANDS:
                row[name] = float(b[name][k]) if n_clear > 0 else np.nan
            rows.append(row)

        if cfg["preds"]:
            usable = cf >= 0.7
            obs = (days[usable], obs_ndvi[usable], obs_ndmi[usable], obs_ndti[usable])
            pred, state = _toy_prediction(rng, fid, jan1, truth, obs, cfg)
            preds.append(pred)
            asof.extend(_toy_asof(rng, fid, jan1, pred, state, obs, cuts, cfg))

    county_rows = _usda_rows(rng, truths, county_names) if cfg["counties"] else None
    fields = {"type": "FeatureCollection", "features": features}
    return {
        "fields": fields,
        "timeseries": pd.DataFrame(rows),
        "predictions": pd.DataFrame(preds) if preds else None,
        "asof": pd.DataFrame(asof) if asof else None,
        "counties": county_rows,
        "county_shapes": county_shapes,
    }


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _toy_prediction(rng, fid, jan1, truth, obs, cfg):
    doy, v, m, r = obs
    h = truth["harvest"]
    before, after = doy < h, doy >= h
    reasons = []
    hs = he = None
    pre_v = pre_m = post_r = None
    if before.any() and after.any():
        i0 = np.where(before)[0][-1]
        i1 = np.where(after)[0][0]
        hs, he = int(doy[i0]), int(doy[i1])
        pre_v, pre_m = float(v[i0]), float(m[i0])
        late = (doy >= he) & (doy <= he + 30)
        post_r = float(np.median(r[late][:3])) if late.any() else None
    contrib = {}
    if hs is None:
        reasons.append("no harvest drop seen between clear observations")
        prob = float(np.clip(rng.normal(0.5, 0.12), 0.05, 0.95))
    else:
        mid = (hs + he) / 2
        contrib = {
            "green": 6 * (pre_v - 0.58),
            "moist": 7 * (pre_m - 0.12),
            "timing": 0.6 * ((cfg["h_sil"] + cfg["h_grn"]) / 2 - mid) / 12,
            "ground": 0.0 if post_r is None else 4 * (0.1 - post_r),
        }
        prob = _sigmoid(sum(contrib.values()) + rng.normal(0, 0.5))
        if he - hs > 21:
            reasons.append(f"harvest window {he - hs} days wide")
    if len(doy) < 8:
        reasons.append(f"only {len(doy)} clear observations")
    low = bool(reasons)

    # a short reason string for most fields; the app must also work without one
    explanation = flip = None
    if contrib and rng.random() < 0.7:
        words = {
            "green": ("green at the cut", "dried down before the cut"),
            "moist": ("leaves still moist", "leaves already dry"),
            "timing": ("early cut", "late cut"),
            "ground": ("bare soil afterwards", "stalks left afterwards"),
        }
        top = sorted(contrib, key=lambda k: -abs(contrib[k]))[:2]
        explanation = "Mostly: " + ", ".join(words[k][0 if contrib[k] > 0 else 1] for k in top)
        if rng.random() < 0.2:
            weakest = min(contrib, key=lambda k: abs(contrib[k]))
            other = "grain" if prob >= 0.5 else "silage"
            flip = f"Would tip to {other} if the {weakest} signal pointed the other way"

    def iso(d):
        return None if d is None else (jan1 + pd.Timedelta(days=int(d) - 1)).date().isoformat()

    pred = {
        "field_id": fid, "p_silage": round(prob, 4),
        "harvest_start": iso(hs), "harvest_end": iso(he),
        "pre_ndvi": None if pre_v is None else round(pre_v, 4),
        "pre_ndmi": None if pre_m is None else round(pre_m, 4),
        "low_evidence": low, "low_evidence_reason": "; ".join(reasons) if reasons else None,
        "explanation": explanation, "flip_reason": flip,
    }
    return pred, {"hs": hs, "he": he}


def _toy_asof(rng, fid, jan1, pred, state, obs, cuts, cfg):
    """What the toy model would have said with data up to each cutoff."""
    doy, v, m, _ = obs
    out = []
    for c in cuts:
        if state["he"] is not None and state["he"] <= c:
            prob = float(np.clip(pred["p_silage"] + rng.normal(0, 0.03), 0.01, 0.99))
            low = pred["low_evidence"]
        else:
            seen = doy <= c
            last_m = float(m[seen][-1]) if seen.any() else 0.12
            late = c - (cfg["h_sil"] + 5)
            if late < 0:
                prob = float(np.clip(rng.normal(0.5, 0.08), 0.2, 0.8))
            else:
                # still standing well after the silage harvest: grain becomes likely
                prob = _sigmoid(-late / 8 + 3 * (last_m - 0.12) + rng.normal(0, 0.4))
            low = bool(late < 15 or not seen.any())
        out.append({"field_id": fid, "cutoff_date": (jan1 + pd.Timedelta(days=c - 1)).date().isoformat(),
                    "p_silage": round(prob, 4), "low_evidence": low})
    return out


def _counties(rng, layout):
    """Ten square sample counties on a 5 x 2 grid covering the fields."""
    lons = [f["lon"] for f in layout]
    lats = [f["lat"] for f in layout]
    x0, x1 = min(lons) - 0.02, max(lons) + 0.02
    y0, y1 = min(lats) - 0.02, max(lats) + 0.02
    nx, ny = 5, 2
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    shapes, share_of, names = [], {}, {}
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            fips = f"99{k + 1:03d}"
            share_of[fips] = float(np.clip(0.1 + 0.07 * k + rng.normal(0, 0.05), 0.03, 0.9))
            names[fips] = f"Sample County {chr(65 + k)}"
            ring = [[round(x0 + i * dx, 5), round(y0 + j * dy, 5)], [round(x0 + (i + 1) * dx, 5), round(y0 + j * dy, 5)],
                    [round(x0 + (i + 1) * dx, 5), round(y0 + (j + 1) * dy, 5)],
                    [round(x0 + i * dx, 5), round(y0 + (j + 1) * dy, 5)], [round(x0 + i * dx, 5), round(y0 + j * dy, 5)]]
            shapes.append({"type": "Feature", "properties": {"GEOID": fips},
                           "geometry": {"type": "Polygon", "coordinates": [ring]}})
    county_of = []
    for f in layout:
        i = min(int((f["lon"] - x0) / dx), nx - 1)
        j = min(int((f["lat"] - y0) / dy), ny - 1)
        county_of.append(f"99{j * nx + i + 1:03d}")
    return {"type": "FeatureCollection", "features": shapes}, county_of, share_of, names


def _usda_rows(rng, truths, names):
    """USDA-style county acres consistent with the simulated fields: we 'map' 80-100 % of
    each county's corn, and the reported numbers carry a few percent of survey noise."""
    rows = []
    for fips, name in names.items():
        sil = sum(a for c, a, f in truths if f == fips and c == "silage") * export.ACRES_PER_HA
        grn = sum(a for c, a, f in truths if f == fips and c == "grain") * export.ACRES_PER_HA
        cover = rng.uniform(0.8, 1.0)
        rows.append({"county_fips": fips, "county_name": name, "state": "WI",
                     "silage_acres": round(sil / cover * rng.normal(1, 0.04)),
                     "grain_acres": round(grn / cover * rng.normal(1, 0.04))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- driver

def build(region_ids=None, work=DEFAULT_WORK, app_data=DEFAULT_APP_DATA, seed=7, n=None):
    work, app_data = Path(work), Path(app_data)
    done = []
    for k, cfg in enumerate(REGIONS):
        if region_ids and cfg["id"] not in region_ids:
            continue
        sim = simulate_region(cfg, seed + 13 * k, n)
        src = work / cfg["id"]
        src.mkdir(parents=True, exist_ok=True)
        (src / "fields.geojson").write_text(json.dumps(sim["fields"]), encoding="utf-8")
        sim["timeseries"].to_parquet(src / "timeseries.parquet", index=False)
        pred_path = asof_path = counties = shapes = None
        if sim["predictions"] is not None:
            pred_path = src / "predictions.parquet"
            sim["predictions"].to_parquet(pred_path, index=False)
        if sim["asof"] is not None:
            asof_path = src / "predictions_by_cutoff.parquet"
            sim["asof"].to_parquet(asof_path, index=False)
        if sim["counties"] is not None:
            counties = src / "usda_counties.csv"
            sim["counties"].to_csv(counties, index=False)
            shapes = src / "county_shapes.geojson"
            shapes.write_text(json.dumps(sim["county_shapes"]), encoding="utf-8")
        meta = export.export_region(
            src / "fields.geojson", app_data / cfg["id"], src / "timeseries.parquet", pred_path,
            asof_path=asof_path, title=cfg["title"], group=cfg["group"], year=cfg["year"], synthetic=True,
            note="synthetic sample, not a model result", counties=counties, county_shapes=shapes,
        )
        done.append(meta)
        m = meta["metrics"]
        print(f"{cfg['id']}: {m['n_fields']} fields, {m['n_predicted']} predicted -> {app_data / cfg['id']}")
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write synthetic sample regions for the demo app.")
    ap.add_argument("--only", nargs="*", help="Region ids to build (default: all)")
    ap.add_argument("--work", default=str(DEFAULT_WORK), help="Where the contract-format inputs are written")
    ap.add_argument("--app-data", default=str(DEFAULT_APP_DATA), help="The app's data folder")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, help="Override the number of fields per region")
    args = ap.parse_args(argv)
    build(args.only, args.work, args.app_data, args.seed, args.n)


if __name__ == "__main__":
    main()
