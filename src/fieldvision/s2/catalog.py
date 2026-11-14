"""Public Earth Search queries and guarded reflectance calibration."""
import time
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = "https://earth-search.aws.element84.com/v1"
PRIMARY = "sentinel-2-c1-l2a"
LEGACY = "sentinel-2-l2a"
ASSETS = dict(zip(("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
                  ("blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22")))


def http_session():
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "POST"), respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def search(collection, bbox, start, end, max_cloud=80, session=None):
    session = session or http_session()
    params = {"collections": collection, "bbox": ",".join(map(str, bbox)),
              "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 200}
    url = URL + "/search"
    items = {}
    seen = set()
    method, body = "GET", None
    while url:
        response = session.request(method, url, params=params, json=body, timeout=(15, 90))
        response.raise_for_status()
        data = response.json()
        for item in data.get("features", []):
            if float(item["properties"].get("eo:cloud_cover", 100)) < max_cloud:
                items[item["id"]] = item
        next_link = next((link for link in data.get("links", []) if link["rel"] == "next"), None)
        if not next_link:
            break
        token = str(next_link)
        if token in seen:
            raise RuntimeError("STAC pagination repeated a page")
        seen.add(token)
        url, method = next_link["href"], next_link.get("method", "GET")
        params, body = None, next_link.get("body")
    return list(items.values())


def acquisition_key(item):
    p = item["properties"]
    tile = p.get("s2:mgrs_tile") or "".join(str(p.get(k, "")) for k in
                                          ("mgrs:utm_zone", "mgrs:latitude_band", "mgrs:grid_square"))
    if not tile:
        # Earth Search IDs encode the MGRS tile in the second component.
        tile = item["id"].split("_")[1].removeprefix("T")
    return tile, p["datetime"][:10], p.get("platform", item["id"][:3])


def find_scenes(bbox, start, end, max_cloud=80, collection=PRIMARY, fallback=True):
    with http_session() as session:
        primary = search(collection, bbox, start, end, max_cloud, session)
        added = []
        # The 2022 growing season is almost entirely absent from Collection 1.
        # Query legacy even when a handful of Collection 1 scenes exist.
        if fallback and collection == PRIMARY and (int(start[:4]) <= 2022 or not primary):
            known = {acquisition_key(item) for item in primary}
            added = [item for item in search(LEGACY, bbox, start, end, max_cloud, session)
                     if acquisition_key(item) not in known]
    items = sorted(primary + added, key=lambda item: (item["properties"]["datetime"], item["id"]))
    return items, {"primary_scenes": len(primary), "fallback_scenes": len(added)}


def calibration(item, asset):
    metadata = asset.get("raster:bands", [{}])[0]
    if "scale" not in metadata or "offset" not in metadata:
        raise ValueError("Reflectance asset lacks explicit raster:bands scale/offset")
    scale, offset = float(metadata["scale"]), float(metadata["offset"])
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
        raise ValueError("Invalid reflectance scale/offset")
    if item["collection"] == LEGACY:
        applied = item["properties"].get("earthsearch:boa_offset_applied")
        if applied is True:
            offset = 0.0
        elif applied is not False and offset != 0:
            raise ValueError("Legacy offset flag missing; refusing ambiguous calibration")
    return scale, offset, metadata.get("nodata", 0)


def scale_reflectance(raw, item, asset):
    scale, offset, nodata = calibration(item, asset)
    result = raw.astype("float32") * scale + offset
    result[(raw == nodata) | ~np.isfinite(raw)] = np.nan
    return result


def check_vegetation(bands, scl, labels):
    """Reject a likely offset error before clipping or writing any observations.

    A scene needs 20 vegetation pixels to support a distribution-level check.
    SCL vegetation includes sparse/senescent plants, so only check the aggregate.
    """
    red, nir = bands["B04"], bands["B08"]
    mask = (scl == 4) & (labels > 0) & np.isfinite(red) & np.isfinite(nir)
    count = int(mask.sum())
    if count < 20:
        return {"vegetation_pixels": count, "checked": False}
    r, n = float(np.median(red[mask])), float(np.median(nir[mask]))
    if not (0 < r < 0.35 and n > r + 0.02 and n < 1):
        raise ValueError(f"Vegetation reflectance sanity check failed: red={r:.4f}, nir={n:.4f}, pixels={count}")
    return {"vegetation_pixels": count, "checked": True, "median_red": r, "median_nir": n}
