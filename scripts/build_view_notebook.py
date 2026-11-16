"""Write colab/view_field_images.ipynb: picture strips for any extracted county, in Colab.

The notebook is self-contained (no fieldvision package, no account): it reads the
public Sentinel-2 catalogs (Earth Search, Planetary Computer with its anonymous token)
and only a small window of each scene's 10 m true-colour asset per chip.
"""
import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# View field images: true-colour chips across the season + greenness curve

For one corn field at a time: six Sentinel-2 true-colour chips (two summer looks, the last
clear look before the cut, the first look after, two later), each a small window around the
field with its outline, and the field's NDVI (greenness) over the season below, with the chip
dates marked. Works for any county the team extracted.

**Inputs** (one county-year): `fields.geojson` (the field outlines), `timeseries.parquet` (the
extracted per-field series), and optionally a per-field calls file (`*_module_v2.parquet` or a
CSV/parquet with `field_id`, `label`, `abstain`, and `evidence` JSON or the columns
`evidence_tier`, `drop_date`, `last_standing_date`). Either upload them in step 2 or point
`COUNTY_DIR` at a folder (for example on your Google Drive).

No account or key is needed: images come from the public Earth Search and Microsoft Planetary
Computer catalogs (Planetary Computer hands out anonymous read tokens). Each chip downloads one
small window, so a strip costs a few megabytes.

Run the cells top to bottom; then change the choice in step 4 and rerun steps 4-5 for more fields.
""")

code("""
# 1. Install (rasterio is the only package Colab lacks)
!pip install -q rasterio
""")

code("""
# 2. Inputs: either set COUNTY_DIR to a folder, or leave it empty and upload the files
COUNTY_DIR = ""          # e.g. "/content/drive/MyDrive/fieldvision/us_24021_2022"
CALLS_FILE = ""          # optional; empty = look for *module*.parquet / *calls* in the folder

import glob, json, os
from pathlib import Path
import pandas as pd

if COUNTY_DIR:
    folder = Path(COUNTY_DIR)
else:
    from google.colab import files
    folder = Path("/content/county")
    folder.mkdir(exist_ok=True)
    print("Upload fields.geojson, timeseries.parquet and (optional) the calls file")
    for name, data in files.upload().items():
        (folder / name).write_bytes(data)

fields_path = next(iter(sorted(folder.glob("*fields*.geojson"))), None)
series_path = next(iter(sorted(folder.glob("*timeseries*.parquet"))), None)
calls_path = Path(CALLS_FILE) if CALLS_FILE else next(
    iter(sorted([*folder.glob("*module*.parquet"), *folder.glob("*calls*.parquet"), *folder.glob("*calls*.csv")])), None)
assert fields_path and series_path, f"need fields.geojson and timeseries.parquet in {folder}"
FIELDS = json.loads(fields_path.read_text())["features"]
SERIES = pd.read_parquet(series_path)
SERIES["date"] = pd.to_datetime(SERIES["date"])
YEAR = int(SERIES["date"].dt.year.mode().iloc[0])

def load_calls(path):
    if path is None:
        return None
    t = pd.read_csv(path) if str(path).endswith(".csv") else pd.read_parquet(path)
    if "evidence" in t.columns:
        ev = t["evidence"].map(lambda s: json.loads(s) if isinstance(s, str) else (s or {})).apply(pd.Series)
        t = pd.concat([t.drop(columns="evidence"), ev], axis=1)
    return t.set_index("field_id")

CALLS = load_calls(calls_path)
print(f"{len(FIELDS)} fields, {SERIES.field_id.nunique()} with a series, year {YEAR}, "
      f"calls: {'none' if CALLS is None else f'{len(CALLS)} rows from {calls_path.name}'}")
""")

code('''
# 3. Helpers (catalog lookup, small-window reads, the strip figure)
import io, threading, time
import numpy as np, requests, rasterio
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from rasterio.windows import from_bounds
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from IPython.display import Image as ShowImage, display

ES = "https://earth-search.aws.element84.com/v1"
PC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
GDAL = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1")
BG, INK, MUTED, GREEN, CUT = "#fcfcfb", "#0b0b0b", "#52514e", "#2f6f4f", "#b5651d"
_token = {"value": None, "until": 0}

def pc_token():
    if _token["value"] is None or time.time() > _token["until"]:
        _token["value"] = requests.get(PC_TOKEN, timeout=60).json()["token"]
        _token["until"] = time.time() + 1800
    return _token["value"]

def fetch_item(scene_id, collection):
    if str(collection).startswith("planetary"):
        url = f"{PC}/collections/sentinel-2-l2a/items/{scene_id}"
    else:
        url = f"{ES}/collections/{collection}/items/{scene_id}"
    r = requests.get(url, timeout=60); r.raise_for_status(); return r.json()

def read_chip(item, geom_ll, collection, margin_m=120):
    epsg = item["properties"].get("proj:epsg") or item["assets"]["visual"].get("proj:epsg")
    if epsg is None and item["properties"].get("proj:code"):
        epsg = int(str(item["properties"]["proj:code"]).split(":")[-1])
    to_local = Transformer.from_crs(4326, epsg, always_xy=True).transform
    local = shp_transform(to_local, geom_ll)
    w, s, e, n = local.bounds
    side = max(e - w, n - s) + 2 * margin_m
    cx, cy = (w + e) / 2, (s + n) / 2
    href = item["assets"]["visual"]["href"]
    if str(collection).startswith("planetary"):
        href = href.split("?")[0] + "?" + pc_token()
    with rasterio.Env(**GDAL):
        with rasterio.open(href) as src:
            win = from_bounds(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, src.transform)
            win = win.round_offsets().round_lengths()
            rgb = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
            inv = ~src.window_transform(win)
    return np.moveaxis(rgb, 0, -1), shp_transform(lambda x, y, z=None: inv * (x, y), local)

def usable_series(field_id):
    t = SERIES[(SERIES.field_id == field_id) & (SERIES.n_clear >= 1) & (SERIES.clear_fraction >= 0.2)].copy()
    t["ndvi"] = (t.B08 - t.B04) / (t.B08 + t.B04)
    return t.sort_values("date").reset_index(drop=True)

def pick_dates(series, last_standing, drop):
    clear = series[series.clear_fraction >= 0.9]
    pool = clear if len(clear) >= 6 else series
    chosen = {}
    def nearest(target, after=None, before=None):
        c = pool
        if after is not None: c = c[c.date > after]
        if before is not None: c = c[c.date < before]
        c = c[~c.date.isin(list(chosen))]
        return None if c.empty else c.iloc[(c.date - target).abs().argsort().iloc[0]].date
    last = pd.Timestamp(last_standing) if last_standing else None
    for target in (f"{YEAR}-07-01", f"{YEAR}-08-01"):
        d = nearest(pd.Timestamp(target), before=last)
        if d is not None: chosen[d] = "summer"
    if last is not None and (series.date == last).any():
        chosen[last] = "last look before cut"
    first_after = None
    if drop:
        after = series[series.date >= pd.Timestamp(drop)]
        if len(after): first_after = after.date.iloc[0]; chosen[first_after] = "first look after"
    base = first_after if first_after is not None else pd.Timestamp(f"{YEAR}-10-01")
    for days in (21, 45):
        d = nearest(base + pd.Timedelta(days=days), after=base)
        if d is not None: chosen[d] = "later"
    return [(d, chosen[d], series[series.date == d].iloc[0]) for d in sorted(chosen)]

def call_of(field_id):
    if CALLS is None or field_id not in CALLS.index:
        return {}
    r = CALLS.loc[field_id]
    return {k: (None if pd.isna(r.get(k)) else r.get(k)) for k in
            ("label", "abstain", "evidence_tier", "drop_date", "last_standing_date")}

def draw_strip(field_id, save_to=None):
    feat = next(f for f in FIELDS if f["properties"]["field_id"] == field_id)
    geom, props = shape(feat["geometry"]), feat["properties"]
    series, call = usable_series(field_id), call_of(field_id)
    chips = pick_dates(series, call.get("last_standing_date"), call.get("drop_date"))
    n = max(len(chips), 6)
    fig = plt.figure(figsize=(2.1 * n, 5.6), facecolor=BG)
    g = fig.add_gridspec(2, n, height_ratios=[1, 0.9], hspace=0.35, wspace=0.06, left=0.05, right=0.98, top=0.82, bottom=0.1)
    for i, (day, role, row) in enumerate(chips):
        ax = fig.add_subplot(g[0, i])
        try:
            for attempt in range(3):           # slow or flaky links: retry a chip a few times
                try:
                    rgb, outline = read_chip(fetch_item(row.scene_id, row.collection), geom, row.collection)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(3 * (attempt + 1))
            ax.imshow(rgb, interpolation="nearest")
            for poly in getattr(outline, "geoms", [outline]):
                xs, ys = poly.exterior.xy
                ax.plot(xs, ys, color="white", lw=1.6)
                ax.plot(xs, ys, color=CUT if role in ("first look after", "later") else GREEN, lw=0.9)
        except Exception as exc:
            ax.text(0.5, 0.5, f"image not available\\n{type(exc).__name__}", ha="center", va="center", fontsize=7, color=MUTED)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(day.strftime("%d %b"), fontsize=10, color=INK, pad=3)
        ax.text(0.5, -0.08, role, transform=ax.transAxes, ha="center", va="top", fontsize=8, color=MUTED)
    ax = fig.add_subplot(g[1, :]); ax.set_facecolor(BG)
    ax.plot(series.date, series.ndvi, color=GREEN, lw=1.2, marker="o", ms=3.5)
    for day, _, _ in chips:
        ax.axvline(day, color=MUTED, lw=0.8, ls=":")
    if call.get("last_standing_date") and call.get("drop_date"):
        ax.axvspan(pd.Timestamp(call["last_standing_date"]), pd.Timestamp(call["drop_date"]), color=CUT, alpha=0.15, lw=0,
                   label="cut happened in here")
        ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=MUTED)
    ax.set_ylim(-0.05, 1.0); ax.set_xlim(pd.Timestamp(f"{YEAR}-04-01"), pd.Timestamp(f"{YEAR}-11-30"))
    ax.set_ylabel("NDVI (greenness)", color=MUTED, fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    if call:
        what = "abstained" if call.get("abstain") else f"called {call.get('label')}"
        head = f"{what} ({call.get('evidence_tier')} evidence)"
    else:
        head = f"label in field file: {props.get('label')}"
    fig.text(0.05, 0.94, f"Field {field_id}, {YEAR}: {head}", fontsize=13, color=INK, weight="bold")
    sub = f"{props.get('area_ha', float('nan')):.1f} ha; {len(series)} usable dates"
    if call.get("last_standing_date") and call.get("drop_date"):
        gap = (pd.Timestamp(call["drop_date"]) - pd.Timestamp(call["last_standing_date"])).days
        sub += f"; last look standing {call['last_standing_date']}, first look cut {call['drop_date']} ({gap} days apart)"
    fig.text(0.05, 0.885, sub + ". Sentinel-2 true colour, 10 m.", fontsize=9, color=MUTED)
    buf = io.BytesIO(); fig.savefig(buf, dpi=100, facecolor=BG); plt.close(fig)
    if save_to:
        Path(save_to).write_bytes(buf.getvalue())
    display(ShowImage(buf.getvalue()))
''')

code("""
# 4. Choose a field: by id, at random, or at random among those matching a filter
MODE = "filter"            # "id", "random" or "filter"
FIELD_ID = ""              # for MODE = "id"
CALL = "silage"            # filter: "silage", "grain", "abstained" or "" (any)
TIER = ""                  # filter: "clear", "wide", "abstain" or "" (any)
DROP_FROM, DROP_TO = "", ""  # filter: e.g. "2022-09-09", "2022-09-22"; "" = no limit
SEED = None                # an int gives the same pick every time

import random
have_series = set(SERIES.field_id)
candidates = [f["properties"]["field_id"] for f in FIELDS if f["properties"]["field_id"] in have_series]
if MODE == "filter":
    if CALLS is None:
        raise SystemExit("filters need a calls file (step 2)")
    c = CALLS[CALLS.index.isin(candidates)].copy()
    if CALL == "abstained":
        c = c[c["abstain"].astype(bool)]
    elif CALL:
        c = c[(c["label"] == CALL) & ~c["abstain"].astype(bool)]
    if TIER:
        c = c[c["evidence_tier"] == TIER]
    drop = pd.to_datetime(c.get("drop_date"))
    if DROP_FROM:
        c = c[drop >= DROP_FROM]
        drop = pd.to_datetime(c.get("drop_date"))
    if DROP_TO:
        c = c[drop <= DROP_TO]
    candidates = list(c.index)
print(len(candidates), "candidate fields")
if MODE == "id":
    chosen = FIELD_ID
else:
    chosen = random.Random(SEED).choice(candidates)
print("field:", chosen, call_of(chosen))
""")

code("""
# 5. Draw it (about 6 small image reads); the PNG is also saved under /content/strips
os.makedirs("/content/strips", exist_ok=True)
draw_strip(chosen, save_to=f"/content/strips/strip_{chosen}.png")
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent.parent / "colab" / "view_field_images.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
