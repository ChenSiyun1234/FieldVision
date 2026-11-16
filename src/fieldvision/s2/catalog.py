"""Public STAC queries (Earth Search, Planetary Computer) and guarded calibration."""
from datetime import datetime
import threading
import time
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = "https://earth-search.aws.element84.com/v1"
PC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
PRIMARY = "sentinel-2-c1-l2a"
LEGACY = "sentinel-2-l2a"
# Planetary Computer serves the unmodified ESA products under the same collection
# name as the Earth Search legacy archive, so its rows get a prefixed source name.
PLANETARY = "planetary:sentinel-2-l2a"
SOURCES = (PRIMARY, LEGACY, PLANETARY)
ASSETS = dict(zip(("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
                  ("blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22")))


def http_session():
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "POST"), respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def search(collection, bbox, start, end, max_cloud=80, session=None, url=URL):
    session = session or http_session()
    params = {"collections": collection, "bbox": ",".join(map(str, bbox)),
              "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 200}
    url = url + "/search"
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
    # Planetary Computer says "Sentinel-2A", Earth Search "sentinel-2a".
    platform = str(p.get("platform", item["id"][:3])).lower().replace("sentinel-", "s")
    return tile.removeprefix("T"), p["datetime"][:10], platform[:3]


def baseline(item):
    try:
        return float(item["properties"].get("s2:processing_baseline"))
    except (TypeError, ValueError):
        return None


def search_source(source, bbox, start, end, max_cloud=80, session=None):
    """Search one source; every item gets ``collection`` set to the source name."""
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}")
    if source == PLANETARY:
        items = search(LEGACY, bbox, start, end, max_cloud, session, url=PC_URL)
        # Planetary Computer keeps the original product and, for some dates, ESA's
        # reprocessed one. Keep one item per acquisition, newest baseline first.
        best = {}
        for item in items:
            item["collection"] = PLANETARY
            key = acquisition_key(item)
            if key not in best or (baseline(item) or 0) > (baseline(best[key]) or 0):
                best[key] = item
        return list(best.values())
    items = search(source, bbox, start, end, max_cloud, session)
    for item in items:
        item["collection"] = source
    return items


def find_scenes(bbox, start, end, max_cloud=80, collection=PRIMARY, fallback=PLANETARY):
    """Scenes from ``collection`` plus acquisitions it lacks from ``fallback`` (None disables)."""
    with http_session() as session:
        primary = search_source(collection, bbox, start, end, max_cloud, session)
        added = []
        # The 2022 growing season is almost entirely absent from Collection 1.
        # Query the fallback even when a handful of Collection 1 scenes exist.
        if fallback and fallback != collection and collection == PRIMARY and (int(start[:4]) <= 2022 or not primary):
            known = {acquisition_key(item) for item in primary}
            added = [item for item in search_source(fallback, bbox, start, end, max_cloud, session)
                     if acquisition_key(item) not in known]
    items = sorted(primary + added, key=lambda item: (item["properties"]["datetime"], item["id"]))
    return items, {"primary_scenes": len(primary), "fallback_scenes": len(added), "fallback_source": fallback}


_PC_TOKEN = {"token": None, "expires": 0.0}
_PC_LOCK = threading.Lock()


def pc_token():
    """Anonymous read token for the Planetary Computer Sentinel-2 container (no account)."""
    with _PC_LOCK:
        if _PC_TOKEN["token"] is None or time.time() > _PC_TOKEN["expires"] - 300:
            with http_session() as session:
                response = session.get(PC_TOKEN_URL, timeout=(15, 60))
                response.raise_for_status()
                data = response.json()
            try:
                expires = datetime.fromisoformat(data["msft:expiry"].replace("Z", "+00:00")).timestamp()
            except (KeyError, ValueError):
                expires = time.time() + 1800
            _PC_TOKEN.update(token=data["token"], expires=expires)
        return _PC_TOKEN["token"]


def asset_for(item, name):
    """Asset for a band name (``B04``) or ``SCL`` in any supported source."""
    if item["collection"] == PLANETARY:
        return item["assets"][name]
    return item["assets"]["scl" if name == "SCL" else ASSETS[name]]


def asset_href(item, name):
    href = asset_for(item, name)["href"]
    if item["collection"] == PLANETARY:
        href = href.split("?")[0] + "?" + pc_token()
    return href


def calibration(item, asset):
    if item["collection"] == PLANETARY:
        # Unmodified ESA products: from processing baseline 04.00 on, L2A digital
        # numbers carry BOA_ADD_OFFSET = -1000 with quantification value 10000.
        value = baseline(item)
        if value is None:
            raise ValueError("Planetary Computer item lacks s2:processing_baseline")
        return 1e-4, (-0.1 if value >= 4.0 else 0.0), 0
    metadata = asset.get("raster:bands", [{}])[0]
    if "scale" not in metadata or "offset" not in metadata:
        raise ValueError("Reflectance asset lacks explicit raster:bands scale/offset")
    scale, offset = float(metadata["scale"]), float(metadata["offset"])
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
        raise ValueError("Invalid reflectance scale/offset")
    if item["collection"] == LEGACY:
        # Measured at Beltsville (scripts/compare_s2_sources.py): flag=True scenes
        # equal Collection 1 / original ESA numbers minus 1000, i.e. the offset is
        # already in the data. All five flag=False 2022 scenes tested were ALSO
        # already shifted, so the flag cannot be trusted when False. Those scenes
        # are refused and logged, not guessed; use the Planetary Computer source.
        applied = item["properties"].get("earthsearch:boa_offset_applied")
        if applied is True:
            offset = 0.0
        elif applied is False:
            raise ValueError("Legacy boa_offset_applied=False is unreliable; refusing scene")
        elif offset != 0:
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
