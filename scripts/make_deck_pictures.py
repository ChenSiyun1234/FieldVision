"""Slide-size pictures for the deck: county call maps and a before/after-the-cut sheet.

maps:   every corn field of a county coloured by the per-field call (silage, grain, not judged).
before_after: for N fields called silage and N called grain in a county's clear evidence tier,
        the last look standing and the first look cut side by side. Rule (no picking by eye):
        keep fields whose two looks are both >= 90% clear, then take the N with area closest
        to the group's median area.
Images are read as small windows of each scene's 10 m true-colour asset (see make_field_strips.py).
"""
import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import shape

from fieldvision.s2.geometry import project
from make_field_strips import fetch_item, read_chip

BG, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
COLORS = {"silage": "#2f6f4f", "grain": "#d4a72c", "not judged": "#bdbcb7"}
COUNTY = {"24021": "Frederick County, Maryland", "24029": "Kent County, Maryland",
          "24035": "Queen Anne's County, Maryland", "24013": "Carroll County, Maryland"}
SLIDE = (13.333, 7.5)      # inches; 1920 x 1080 at 144 dpi
DPI = 144


def load_calls(fips, year=2022):
    t = pd.read_parquet(rf"C:\t\fv\data\diag\dairy\results\MD_{year}_{fips}_module_v2.parquet")
    ev = t.evidence.map(json.loads).apply(pd.Series)
    t = pd.concat([t.drop(columns="evidence"), ev], axis=1)
    t["call"] = np.where(t.abstain.astype(bool), "not judged", t.label)
    return t.set_index("field_id")


def save_png(fig, path, limit=None):
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=DPI, facecolor=BG)
    plt.close(fig)
    image = Image.open(buffer).convert("RGB")
    image.save(path, optimize=True)
    if limit and path.stat().st_size > limit:
        image.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE).save(path, optimize=True)
    return path


def county_map(fips, out, year=2022):
    calls = load_calls(fips, year)
    feats = json.loads(Path(rf"C:\t\fv\data\labels\us_{fips}_{year}\fields.geojson").read_text())["features"]
    patches = {k: [] for k in COLORS}
    xs, ys = [], []
    for f in feats:
        fid = f["properties"]["field_id"]
        call = calls.call.get(fid, "not judged")
        geom = project(shape(f["geometry"]), 4326, 32618)
        for poly in getattr(geom, "geoms", [geom]):
            coords = np.asarray(poly.exterior.coords)
            patches[call].append(MplPolygon(coords, closed=True))
            xs += [coords[:, 0].min(), coords[:, 0].max()]
            ys += [coords[:, 1].min(), coords[:, 1].max()]
    fig = plt.figure(figsize=SLIDE, facecolor=BG)
    ax = fig.add_axes([0.03, 0.05, 0.64, 0.82])
    ax.set_facecolor(BG)
    for call, items in patches.items():
        ax.add_collection(PatchCollection(items, facecolor=COLORS[call], edgecolor=COLORS[call], linewidth=0.3))
    pad = 0.03 * max(max(xs) - min(xs), max(ys) - min(ys))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    # 5 km scale bar
    x0, y0 = min(xs), min(ys) - pad * 0.5
    ax.plot([x0, x0 + 5000], [y0, y0], color=INK, lw=2)
    ax.text(x0 + 2500, y0 + pad * 0.25, "5 km", ha="center", va="bottom", fontsize=10, color=INK)
    counts = calls.call.value_counts()
    total = len(feats)
    fig.text(0.03, 0.93, f"{COUNTY[fips]}, {year}: every corn field and its call", fontsize=22, color=INK, weight="bold")
    handles = [Patch(facecolor=COLORS[c], edgecolor=COLORS[c],
                     label=f"{c}: {int(counts.get(c, 0)):,} fields ({counts.get(c, 0) / total:.0%})") for c in COLORS]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.70, 0.78), frameon=False, fontsize=15,
                     labelcolor=INK, handlelength=1.4, handleheight=1.2)
    tiers = calls.evidence_tier.value_counts()
    fig.text(0.70, 0.55, f"{total:,} corn fields\n(USDA Crop Sequence Boundaries)", fontsize=12, color=MUTED,
             va="top", linespacing=1.4)
    fig.text(0.70, 0.46, f"Evidence: clear {int(tiers.get('clear', 0)):,}, wide {int(tiers.get('wide', 0)):,},\n"
             f"abstained {int(tiers.get('abstain', 0)):,}", fontsize=12, color=MUTED, va="top", linespacing=1.4)
    fig.text(0.70, 0.37, "Each call comes from the field's\nSentinel-2 greenness series: when the\ncut happened and how "
             "green the crop\nwas just before it. 'Not judged': the\nseries could not show the cut clearly.",
             fontsize=12, color=MUTED, va="top", linespacing=1.4)
    return save_png(fig, out / f"map_{fips}_{year}.png", limit=1_500_000)


def series_for(fips, year=2022):
    t = pd.read_parquet(rf"C:\t\fv\data\timeseries\us_{fips}_{year}\timeseries.parquet")
    t["date"] = pd.to_datetime(t.date)
    return t


def before_after(fips, out, n=6, year=2022):
    calls = load_calls(fips, year)
    series = series_for(fips, year)
    feats = {f["properties"]["field_id"]: f for f in
             json.loads(Path(rf"C:\t\fv\data\labels\us_{fips}_{year}\fields.geojson").read_text())["features"]}
    clear = series[series.clear_fraction >= 0.9].set_index(["field_id", "date"])
    chosen = {}
    for label in ("silage", "grain"):
        g = calls[(calls.evidence_tier == "clear") & (calls.label == label) & ~calls.abstain.astype(bool)].copy()
        g["area_ha"] = [feats[i]["properties"]["area_ha"] for i in g.index]
        ok = [i for i, r in g.iterrows()
              if (i, pd.Timestamp(r.last_standing_date)) in clear.index and (i, pd.Timestamp(r.drop_date)) in clear.index]
        g = g.loc[ok]
        median = g.area_ha.median()
        chosen[label] = g.iloc[(g.area_ha - median).abs().argsort().iloc[:n]]
        print(label, len(ok), "eligible; median area", round(median, 2), "->", list(chosen[label].index))
    fig = plt.figure(figsize=SLIDE, facecolor=BG)
    grid = fig.add_gridspec(4, n, left=0.17, right=0.99, top=0.83, bottom=0.03, hspace=0.32, wspace=0.05)
    row_axes = {}
    for block, label in enumerate(("silage", "grain")):
        for col, (fid, r) in enumerate(chosen[label].iterrows()):
            geom = shape(feats[fid]["geometry"])
            for k, (when, role) in enumerate(((r.last_standing_date, "standing"), (r.drop_date, "cut"))):
                row = clear.loc[(fid, pd.Timestamp(when))]
                ax = fig.add_subplot(grid[2 * block + k, col])
                row_axes.setdefault(2 * block + k, ax)
                rgb, outline = read_chip(fetch_item(row.scene_id, row.collection), geom, row.collection)
                ax.imshow(rgb, interpolation="nearest")
                for poly in getattr(outline, "geoms", [outline]):
                    px, py = poly.exterior.xy
                    ax.plot(px, py, color="white", lw=1.4)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#d9d8d4")
                ax.set_title(pd.Timestamp(when).strftime("%d %b"), fontsize=10, color=INK if k else MUTED, pad=2)
                print(f"  {label} {fid} {role} {when}", flush=True)
    # Row and group labels aligned to the chip rows.
    centre = {r: (ax.get_position().y0 + ax.get_position().y1) / 2 for r, ax in row_axes.items()}
    for r, text in ((0, "last look\nstanding"), (1, "first look\ncut"), (2, "last look\nstanding"), (3, "first look\ncut")):
        fig.text(0.155, centre[r], text, fontsize=10, color=MUTED, va="center", ha="right")
    for block, label in enumerate(("silage", "grain")):
        fig.text(0.015, (centre[2 * block] + centre[2 * block + 1]) / 2, f"called\n{label}", fontsize=18,
                 color=COLORS[label], weight="bold", va="center")
    fig.text(0.015, 0.945, f"{COUNTY[fips]}, {year}: just before and just after the cut", fontsize=22, color=INK,
             weight="bold")
    fig.text(0.015, 0.875, f"{n} fields called silage and {n} called grain, clear evidence. Chosen by rule, not by eye: "
             "both looks >= 90% clear, then the fields closest to their group's median area.\n"
             "Sentinel-2 true colour, 10 m, field outlined. Silage is cut green; grain is left to dry brown first.",
             fontsize=10.5, color=MUTED, linespacing=1.5)
    return save_png(fig, out / f"before_after_{fips}_{year}.png", limit=2_500_000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(r"C:\t\fv\data\share\deck"))
    parser.add_argument("--what", nargs="+", default=["maps", "before_after"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if "maps" in args.what:
        for fips in ("24021", "24029"):
            path = county_map(fips, args.out)
            print("wrote", path, f"{path.stat().st_size / 1e3:.0f} KB", flush=True)
    if "before_after" in args.what:
        path = before_after("24021", args.out)
        print("wrote", path, f"{path.stat().st_size / 1e3:.0f} KB", flush=True)


if __name__ == "__main__":
    main()
