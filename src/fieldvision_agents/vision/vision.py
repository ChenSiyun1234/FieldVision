"""Vision check: Nemotron Super VL looks at before- and after-harvest images of one field.

It answers two questions a person would ask looking at the pictures: was the crop still green
before it was cut, and is the ground afterwards bare soil (silage-like) or covered in stalk
residue (grain-like)? The answer is extra evidence for the field's record. It never sets the label.

In a notebook:
    from fieldvision_agents.vision.vision import check_barc_field
    check_barc_field(object_id=112, year=2022)
"""

import base64
import io
import json
import re

from ..react.supervisor import VISION_MODEL, NemotronClient

BARC_GPKG = "04_Evaluation/BARC_Corn.2021-2023.gpkg"
BARC_LAYER = "barc_silage_grain_corn20212023"

# Default windows: green canopy before any cut, then the ground after harvest.
BEFORE = ("07-15", "08-20")
AFTER = ("10-01", "11-30")

GROUND = ("bare_soil", "residue", "green_cover", "unclear")
GREEN = ("green", "brown", "unclear")

VISION_PROMPT = """You are looking at two true-colour satellite images of the same corn field (outlined in yellow).
Image 1 is before harvest, image 2 is after harvest.
Answer only from what you can see. Reply with JSON only, no other text:
{"before_canopy": "green" | "brown" | "unclear",
 "after_ground": "bare_soil" | "residue" | "green_cover" | "unclear",
 "confidence": a number from 0 to 1,
 "reason": "one short sentence about what you see"}
Bare dark soil after harvest suggests the whole plant was removed (silage-like). Light brown stalks and leaves
lying on the ground suggest grain harvest. A green field in image 2 may be a cover crop. Clouds or haze mean unclear."""


def parse_reply(text):
    """Pull the JSON out of the model's reply and keep only allowed values."""
    m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", text or "", flags=re.S), re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        obj = {}
    try:
        conf = max(0.0, min(1.0, float(obj.get("confidence", 0))))
    except (TypeError, ValueError):
        conf = 0.0
    out = {
        "before_canopy": obj.get("before_canopy") if obj.get("before_canopy") in GREEN else "unclear",
        "after_ground": obj.get("after_ground") if obj.get("after_ground") in GROUND else "unclear",
        "confidence": conf,
        "reason": str(obj.get("reason", ""))[:300],
    }
    out["hint"] = {"bare_soil": "silage-like", "residue": "grain-like"}.get(out["after_ground"], "no hint")
    return out


def build_messages(before_png, after_png):
    def url(png):
        return "data:image/png;base64," + base64.b64encode(png).decode()
    return [{"role": "user", "content": [
        {"type": "text", "text": VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": url(before_png)}},
        {"type": "image_url", "image_url": {"url": url(after_png)}},
    ]}]


def ask_vision(before_png, after_png, llm=None):
    llm = llm or NemotronClient(model=VISION_MODEL, max_tokens=400)
    return parse_reply(llm.chat(build_messages(before_png, after_png))["content"])


def load_barc_field(object_id, path=BARC_GPKG, layer=BARC_LAYER):
    """One BARC field polygon in lon/lat (the GeoPackage is in Maryland State Plane feet)."""
    import geopandas as gpd
    fields = gpd.read_file(path, layer=layer)
    row = fields[fields["OBJECTID"] == object_id]
    if row.empty:
        raise ValueError(f"OBJECTID {object_id} not in {path}")
    return row.to_crs("EPSG:4326").geometry.iloc[0]


def true_colour_chip(geom, start, end, max_cloud=40, pad_m=150, size_in=3.2):
    """Clearest Sentinel-2 true-colour image of the field between two dates, as PNG bytes."""
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import odc.stac
    import planetary_computer
    import pystac_client

    field = gpd.GeoSeries([geom], crs="EPSG:4326")
    utm = field.estimate_utm_crs()
    box = field.to_crs(utm).buffer(pad_m).to_crs("EPSG:4326").total_bounds
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                        modifier=planetary_computer.sign_inplace)
    items = catalog.search(collections=["sentinel-2-l2a"], bbox=box, datetime=f"{start}/{end}",
                           query={"eo:cloud_cover": {"lt": max_cloud}}).item_collection()
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes between {start} and {end}")
    ds = odc.stac.load(items, bands=["B04", "B03", "B02", "SCL"], bbox=box, crs=utm.to_string(),
                       resolution=10, groupby="solar_day")
    clear = ds.SCL.isin([4, 5]).mean(dim=["x", "y"]).values
    i = int(np.argmax(clear))
    scene = ds.isel(time=i)
    rgb = np.stack([scene[b].values.astype("float32") for b in ("B04", "B03", "B02")], axis=-1)
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1), 0, 1)

    x, y = scene.x.values, scene.y.values
    fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=100)
    ax.imshow(rgb, extent=(x.min(), x.max(), y.min(), y.max()))
    field.to_crs(utm).boundary.plot(ax=ax, color="yellow", linewidth=1.5)
    ax.set_axis_off()
    date = str(scene.time.values)[:10]
    ax.set_title(date, fontsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue(), date, float(clear[i])


def check_field(geom, year, before=BEFORE, after=AFTER, llm=None):
    """Fetch both chips for a field-season and ask the vision model what it sees."""
    b_png, b_date, b_clear = true_colour_chip(geom, f"{year}-{before[0]}", f"{year}-{before[1]}")
    a_png, a_date, a_clear = true_colour_chip(geom, f"{year}-{after[0]}", f"{year}-{after[1]}")
    verdict = ask_vision(b_png, a_png, llm)
    verdict.update({"before_date": b_date, "after_date": a_date,
                    "before_clear_fraction": round(b_clear, 2), "after_clear_fraction": round(a_clear, 2)})
    return verdict, b_png, a_png


def check_barc_field(object_id, year, **kw):
    """Vision check for one labelled BARC field, so it can be compared with the reference label."""
    return check_field(load_barc_field(object_id), year, **kw)
