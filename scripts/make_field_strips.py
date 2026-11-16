"""Picture strips for a few fields: true-colour chips across the season plus the NDVI curve.

For each field: six chips (two summer looks, the last clear look before the cut, the
first after, two later), each a small window of the scene's 10 m true-colour
("visual") asset around the field with its outline, dates underneath, and the field's
NDVI from the extracted time series below with the chip dates marked. Only those
small windows are read (one per chip); the extractor itself is not used or changed.

Usage: python scripts/make_field_strips.py --picks picks.json --out C:/t/fv/data/share/strips
"""
import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from fieldvision.s2.catalog import PC_URL, URL as ES_URL, http_session, pc_token
from fieldvision.s2.extract import GDAL_OPTIONS
from fieldvision.s2.geometry import project

BG, INK, MUTED, ACCENT, CUT = "#fcfcfb", "#0b0b0b", "#52514e", "#2f6f4f", "#b5651d"
COUNTY = {"24021": "Frederick County, MD", "24029": "Kent County, MD"}
MARGIN_M = 120


def usable_series(run_dir, field_id):
    t = pd.read_parquet(run_dir / "timeseries.parquet", filters=[("field_id", "==", field_id)])
    t = t[(t.n_clear >= 1) & (t.clear_fraction >= 0.2)].copy()
    t["date"] = pd.to_datetime(t.date)
    t["ndvi"] = (t.B08 - t.B04) / (t.B08 + t.B04)
    return t.sort_values("date").reset_index(drop=True)


def pick_dates(series, last_standing, drop, year):
    """Six chip dates from fully clear looks (clear_fraction >= 0.9) where possible."""
    clear = series[series.clear_fraction >= 0.9]
    pool = clear if len(clear) >= 6 else series
    chosen = []

    def nearest(target, after=None, before=None):
        cand = pool
        if after is not None:
            cand = cand[cand.date > after]
        if before is not None:
            cand = cand[cand.date < before]
        cand = cand[~cand.date.isin([c[0] for c in chosen])]
        if cand.empty:
            return None
        return cand.iloc[(cand.date - target).abs().argsort().iloc[0]].date

    last = pd.Timestamp(last_standing) if last_standing else None
    first_after = None
    if drop:
        after_rows = series[series.date >= pd.Timestamp(drop)]
        first_after = after_rows.date.iloc[0] if len(after_rows) else None
    for role, target, kw in [("summer", pd.Timestamp(f"{year}-07-01"), {"before": last}),
                             ("summer", pd.Timestamp(f"{year}-08-01"), {"before": last})]:
        d = nearest(target, **kw)
        if d is not None:
            chosen.append((d, role))
    if last is not None and (series.date == last).any():
        chosen.append((last, "last look before cut"))
    if first_after is not None:
        chosen.append((first_after, "first look after"))
    base = first_after if first_after is not None else pd.Timestamp(f"{year}-10-01")
    for days in (21, 45):
        d = nearest(base + pd.Timedelta(days=days), after=base)
        if d is not None:
            chosen.append((d, "later"))
    chosen = sorted({d: r for d, r in chosen}.items())
    return [(d, r, series[series.date == d].iloc[0]) for d, r in chosen]


def fetch_item(scene_id, collection):
    url = (PC_URL + "/collections/sentinel-2-l2a/items/" if collection.startswith("planetary")
           else ES_URL + f"/collections/{collection}/items/")
    with http_session() as session:
        response = session.get(url + scene_id, timeout=(15, 60))
        response.raise_for_status()
        return response.json()


def read_chip(item, geometry_ll, collection):
    epsg = item["properties"].get("proj:epsg") or item["assets"]["visual"].get("proj:epsg")
    local = project(geometry_ll, 4326, epsg)
    west, south, east, north = local.bounds
    side = max(east - west, north - south) + 2 * MARGIN_M
    cx, cy = (west + east) / 2, (south + north) / 2
    href = item["assets"]["visual"]["href"]
    if collection.startswith("planetary"):
        href = href.split("?")[0] + "?" + pc_token()
    with rasterio.Env(**GDAL_OPTIONS):
        with rasterio.open(href) as src:
            window = from_bounds(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, src.transform)
            window = window.round_offsets().round_lengths()
            rgb = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
            win_transform = src.window_transform(window)
    inverse = ~win_transform
    outline = shp_transform(lambda x, y, z=None: inverse * (x, y), local)
    return np.moveaxis(rgb, 0, -1), outline


DPI, SIZE_LIMIT = 100, 400_000


def draw(pick, run_dir, labels_path, out_path):
    field_id, year = pick["field_id"], 2022
    feats = json.loads(Path(labels_path).read_text())["features"]
    geometry = shape(next(f["geometry"] for f in feats if f["properties"]["field_id"] == field_id))
    series = usable_series(run_dir, field_id)
    chips = pick_dates(series, pick.get("last_standing_date"), pick.get("drop_date"), year)
    n = len(chips)
    fig = plt.figure(figsize=(2.0 * max(n, 6), 5.4), facecolor=BG)
    grid = fig.add_gridspec(2, max(n, 6), height_ratios=[1.0, 0.9], hspace=0.35, wspace=0.06,
                            left=0.05, right=0.98, top=0.82, bottom=0.1)
    for i, (day, role, row) in enumerate(chips):
        ax = fig.add_subplot(grid[0, i])
        for attempt in range(4):              # slow or flaky links: retry a chip
            try:
                rgb, outline = read_chip(fetch_item(row.scene_id, row.collection), geometry, row.collection)
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                print(f"  retry {row.scene_id}: {type(exc).__name__}", flush=True)
                import time
                time.sleep(5 * (attempt + 1))
        ax.imshow(rgb, interpolation="nearest")
        for poly in getattr(outline, "geoms", [outline]):
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color="#ffffff", lw=1.6)
            ax.plot(xs, ys, color=CUT if "after" in role or role == "later" else ACCENT, lw=0.9)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d9d8d4")
        ax.set_title(day.strftime("%d %b"), fontsize=10, color=INK, pad=3)
        ax.text(0.5, -0.08, role, transform=ax.transAxes, ha="center", va="top", fontsize=8, color=MUTED)
        print(f"  chip {day.date()} {role}: {row.scene_id}", flush=True)
    ax = fig.add_subplot(grid[1, :])
    ax.set_facecolor(BG)
    ax.plot(series.date, series.ndvi, color=ACCENT, lw=1.2, marker="o", ms=3.5)
    for day, role, _ in chips:
        ax.axvline(day, color=MUTED, lw=0.8, ls=":")
    if pick.get("last_standing_date") and pick.get("drop_date"):
        ax.axvspan(pd.Timestamp(pick["last_standing_date"]), pd.Timestamp(pick["drop_date"]), color=CUT, alpha=0.15,
                   lw=0, label="cut happened in here")
        ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=MUTED)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlim(pd.Timestamp(f"{year}-04-01"), pd.Timestamp(f"{year}-11-30"))
    ax.set_ylabel("NDVI (greenness)", color=MUTED, fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbcb7")
    call = "abstained" if pick.get("abstain") else f"called {pick['label']}"
    fig.text(0.05, 0.94, f"{COUNTY.get(str(pick['fips']), pick['fips'])}, 2022: {call} ({pick['evidence_tier']} evidence)",
             fontsize=13, color=INK, weight="bold")
    gap = ""
    if pick.get("last_standing_date") and pick.get("drop_date"):
        gap = (f"; last clear look standing {pick['last_standing_date']}, first look cut {pick['drop_date']} "
               f"({(pd.Timestamp(pick['drop_date']) - pd.Timestamp(pick['last_standing_date'])).days} days apart)")
    fig.text(0.05, 0.885, f"Field {field_id}, {pick['area_ha']:.1f} ha{gap}. Sentinel-2 true colour, 10 m.",
             fontsize=9, color=MUTED)
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=DPI, facecolor=BG)
    plt.close(fig)
    save_small(Image.open(buffer), out_path, SIZE_LIMIT)
    return out_path


def save_small(image, path, limit=400_000):
    """PNG under the size limit: adaptive palette, then shrink if still too big."""
    image = image.convert("RGB")
    scale = 1.0
    while True:
        img = image if scale == 1.0 else image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
        img.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE).save(path, optimize=True)
        if path.stat().st_size <= limit or scale < 0.4:
            return path
        scale *= 0.85


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--picks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--dpi", type=int, default=100, help="160 gives 1920 px wide strips for slides")
    parser.add_argument("--size-limit", type=int, default=400_000)
    args = parser.parse_args()
    global DPI, SIZE_LIMIT
    DPI, SIZE_LIMIT = args.dpi, args.size_limit
    args.out.mkdir(parents=True, exist_ok=True)
    picks = json.loads(args.picks.read_text())
    made = []
    for name, pick in picks.items():
        if args.only and name not in args.only:
            continue
        fips = str(pick["fips"])
        print(name, pick["field_id"], flush=True)
        path = draw(pick, Path(f"C:/t/fv/data/timeseries/us_{fips}_2022"),
                    Path(f"C:/t/fv/data/labels/us_{fips}_2022/fields.geojson"), args.out / f"strip_{name}.png")
        print("  wrote", path, f"{path.stat().st_size / 1e3:.0f} KB", flush=True)
        made.append(path)
    strips = sorted(args.out.glob("strip_*.png"))
    if len(strips) > 1:
        images = [Image.open(p).convert("RGB") for p in strips]
        width = max(i.width for i in images)
        sheet = Image.new("RGB", (width, sum(i.height for i in images)), BG)
        y = 0
        for img in images:
            sheet.paste(img, (0, y))
            y += img.height
        path = save_small(sheet, args.out / "strips_sheet.png")
        print("sheet", path, f"{path.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
