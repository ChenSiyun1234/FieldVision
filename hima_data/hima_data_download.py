# %% [markdown]
# # Hack the Harvest: data download for 2 new counties (Keeboh version)
#
# Adapted from the team's Colab notebook to run in the Keeboh workspace, in its own folder (/workspace/hima_data), so it never touches the agents/ folder or the Iowa/Nebraska download.
#
# **Counties:** Tulare, California (06107) and Wyoming, New York (36121). They cover what the other downloads miss: the irrigated West with a different season, and the cloudy dairy Northeast.
#
# **Year: 2022**, so every county can be checked against the USDA Census of Agriculture 2022 (our county ground truth). Change YEAR in the config cell if the team settles on another year.
#
# **What changed from the Colab version**
# - No Google Drive: everything goes to /workspace/hima_data.
# - Reuses the national Crop Sequence Boundaries zip already in the workspace (/workspace/data/states/csb, 3.7 GB), read straight from the zip, filtered by state and county code. No TIGER download needed.
# - Sentinel-2 keeps all 10 bands (B02 to B12), so the output can be exported in the FieldVision format and run through our toolbox, agent, critic and recommender.
# - Per-state date window (California starts in March for early and double-cropped corn).
# - Sampling picks whole 5 km tiles at random (cluster sample), so 800 fields means about 20 tiles instead of every tile in the county.
# - Radar (Sentinel-1) and weather are off by default: no model uses them yet. Turn them on in the config if needed.
# - Last cell exports fields.geojson + timeseries.parquet + fields_coverage.csv for FieldVision.
#
# **How to run:** set TEST = True and run all (about 4 fields, a few minutes). If it works, set TEST = False and run all again. Finished tiles are skipped on reruns.
#
# No raw satellite imagery is saved, only per-field tables.

# %%
# Install the few packages the Keeboh image is missing, into this folder only (does not touch the shared environment)
import subprocess, sys
from pathlib import Path
DEPS = Path("/workspace/hima_data/.deps")
need = []
for mod, pkg in [("tqdm", "tqdm"), ("h5netcdf", "h5netcdf"), ("h5py", "h5py"), ("netCDF4", "netCDF4"), ("cftime", "cftime"), ("pyarrow", "pyarrow")]:
    try:
        __import__(mod)
    except ImportError:
        need.append(pkg)
if need:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(DEPS)] + need, check=False)
if str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))
print("installed into hima_data/.deps:", need or "nothing needed")

# %% [markdown]
# ## 1. Config

# %%
import os, re, io, json, time, glob, zipfile, warnings, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, geopandas as gpd
import requests
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

# ---- Where things go ----
BASE  = Path("/workspace/hima_data")
RAW   = BASE / "raw"          # small cached inputs (CDL county tifs)
OUT   = BASE / "processed"    # per-region outputs
LOCAL = BASE / "cache"        # big temporary downloads (gridMET, only if weather is on)
FV_EXPORT = BASE / "fieldvision_export" / "counties"   # FieldVision format for the toolbox
for p in (RAW, OUT, LOCAL, FV_EXPORT): p.mkdir(parents=True, exist_ok=True)

# ---- What to download ----
YEAR = 2022
COUNTIES = [
    ("CA", "06107"),   # Tulare, California: irrigated dairy, different season
    ("NY", "36121"),   # Wyoming, New York: cloudy Northeast dairy. Backup: ("ID", "16047") Gooding, Idaho
]
if os.environ.get("HIMA_COUNTIES"):   # e.g. HIMA_COUNTIES="NY:36121,ID:16047" when run from a terminal
    COUNTIES = [tuple(x.split(":")) for x in os.environ["HIMA_COUNTIES"].split(",")]
N_FIELDS    = 800       # target fields per county (whole 5 km tiles picked at random), or "all"
RANDOM_SEED = 42
TEST        = os.environ.get("HIMA_TEST", "0") == "1"   # set True = 4 fields from one tile per county, to check quickly
RUN_S1      = False     # Sentinel-1 radar: no model uses it yet
RUN_WEATHER = False     # gridMET weather: not used by the model yet (downloads about 1 GB)

# Crop Sequence Boundaries: national zip already downloaded by a teammate (read-only, read straight from the zip)
CSB_ZIP   = Path("/workspace/data/states/csb/NationalCSB_2018-2025_rev23.zip")
CSB_URL   = "https://www.nass.usda.gov/Research_and_Science/Crop-Sequence-Boundaries/datasets/NationalCSB_2018-2025_rev23.zip"
STATE_FIPS = {"CA": "06", "NY": "36", "ID": "16", "PA": "42", "WI": "55", "IA": "19", "NE": "31", "MD": "24", "TX": "48"}

# ---- Shared processing settings (same as the team's Colab, except the window) ----
MIN_CORN_FRAC = 0.8
MIN_AREA_HA   = 2.0
BUFFER_M      = 15          # inward buffer to avoid edge pixels
DATE_WINDOW   = {"default": ("04-01", "12-15"),   # FieldVision's Frederick series starts in early April
                 "CA": ("03-01", "12-15")}       # earlier planting and double crops
MAX_CLOUD     = 60          # scene-level cloud filter (%)
SCL_OK        = [4, 5]      # Sentinel-2 SCL: vegetation, bare soil
MIN_CLEAR     = 0.8         # drop field-dates with < 80% clear pixels
MIN_S1_COVER  = 0.5
RES           = 10
TILE_SIZE_M   = 5000
N_WORKERS     = 3           # tiles processed in parallel (shared machine: keep it modest)
READ_THREADS  = 8

# ---- Context data ----
def _env_key(name):
    if os.environ.get(name):
        return os.environ[name]
    for f in (BASE / ".env", Path("/workspace/.env")):
        if f.exists():
            for line in f.read_text().splitlines():
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip('"')
    return ""
QUICKSTATS_KEY = _env_key("QUICKSTATS_KEY")   # free key: https://quickstats.nass.usda.gov/api , put QUICKSTATS_KEY=... in /workspace/hima_data/.env
DAIRY_CSV      = Path("/workspace/data/usa_dairy_farms.csv")   # team file (EPA FRS); lat/lon or latitude/longitude columns
CA_DAIRY_SERVICE = "https://gispublic.waterboards.ca.gov/portalserver/rest/services/California_Dairies_Layers/MapServer"

LOG = BASE / "DATA_LOG.md"
def log(msg):
    line = f"- {dt.datetime.now():%Y-%m-%d %H:%M} {msg}"
    print(line)
    with open(LOG, "a") as f: f.write(line + "\n")

print("Quick Stats key:", "found" if QUICKSTATS_KEY else "missing (county context will be skipped)")
print("CSB zip:", CSB_ZIP if CSB_ZIP.exists() else "missing, will download")

# %% [markdown]
# ## 2. Helpers

# %%
UA = {"User-Agent": "Mozilla/5.0 (hack-the-harvest data pipeline)"}

def download(url, dest, verify=True):
    """Download with progress bar; skip if the file already exists."""
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, headers=UA, stream=True, timeout=300, verify=verify) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk); bar.update(len(chunk))
    tmp.rename(dest)
    return dest

def retry(fn, tries=3, wait=5):
    """Call fn(); retry with exponential backoff (covers HTTP 429 and flaky reads)."""
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(wait * 2 ** i)

def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1

def region_dir(region):
    d = OUT / region; d.mkdir(parents=True, exist_ok=True); return d

def window_for(region):
    st = region.split("_")[0]
    return DATE_WINDOW.get(st, DATE_WINDOW["default"])

# %% [markdown]
# ## 3. Field boundaries: Crop Sequence Boundaries + CDL corn fraction

# %%
import pyogrio, rasterio
from rasterio.features import rasterize

def csb_source():
    zp = CSB_ZIP if CSB_ZIP.exists() else download(CSB_URL, LOCAL / CSB_ZIP.name)
    gdb_name = next(n for n in zipfile.ZipFile(zp).namelist() if n.endswith(".gdb/"))
    path = f"/vsizip/{zp}/{gdb_name.rstrip('/')}"
    layer = pyogrio.list_layers(path)[0][0]
    return path, layer

_csb_attr = {}
def csb_attributes():
    """All 16 million CSB rows, attributes only (about 1 minute, once per session).
    A SQL 'where' on the zipped geodatabase returns nothing, so we filter in pandas and read geometry by row id."""
    if "df" not in _csb_attr:
        path, layer = csb_source()
        t0 = time.time()
        a = pyogrio.read_dataframe(path, layer=layer, columns=["CSBID", "STATEFIPS", "CNTYFIPS", f"CDL{YEAR}"],
                                   read_geometry=False, fid_as_index=True)
        _csb_attr["df"] = a
        print(f"CSB attributes: {len(a):,} polygons in {time.time()-t0:.0f} s")
    return _csb_attr["df"]

def get_cdl_county(year, fips):
    """County CDL GeoTIFF from CropScape (small, cached)."""
    dest = RAW / "cdl" / f"CDL_{year}_{fips}.tif"
    if dest.exists():
        return dest
    url = f"https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year={year}&fips={fips}"
    r = retry(lambda: requests.get(url, headers=UA, timeout=300))
    r.raise_for_status()
    tif_url = re.search(r"<returnURL>(.*?)</returnURL>", r.text).group(1)
    return download(tif_url, dest)

def corn_fraction(fields, cdl_path, corn_code=1):
    """Fraction of CDL corn pixels inside each polygon (one rasterize + bincount)."""
    with rasterio.open(cdl_path) as src:
        cdl = src.read(1)
        f = fields.to_crs(src.crs)
        ids = np.arange(1, len(f) + 1)
        lab = rasterize(zip(f.geometry, ids), out_shape=cdl.shape, transform=src.transform, fill=0, dtype="int32")
    tot  = np.bincount(lab.ravel(), minlength=len(f) + 1)
    corn = np.bincount(lab.ravel(), weights=(cdl.ravel() == corn_code), minlength=len(f) + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return corn[1:] / tot[1:]

def collect_fields(state, fips):
    region = f"{state}_{fips}"
    fpath = region_dir(region) / "fields.gpkg"
    if fpath.exists():
        return gpd.read_file(fpath)
    path, layer = csb_source()
    st, cnty = fips[:2], fips[2:]
    t0 = time.time()
    a = csb_attributes()
    col = f"CDL{YEAR}"
    sel = a[(a.STATEFIPS.astype(str) == st) & (a.CNTYFIPS.astype(str) == cnty)]
    n_total = len(sel)
    sel = sel[pd.to_numeric(sel[col], errors="coerce") == 1]   # CSB's majority crop for the year is corn
    if sel.empty:
        raise ValueError(f"No corn polygons for {fips} in {YEAR}")
    csb = pyogrio.read_dataframe(path, layer=layer, fids=sel.index.values, columns=["CSBID"])
    csb["area_ha"] = csb.to_crs(5070).area / 1e4
    csb = csb[csb.area_ha >= MIN_AREA_HA]
    try:
        csb["corn_frac"] = corn_fraction(csb, get_cdl_county(YEAR, fips))
        corn_src = f"CDL {YEAR} raster (CropScape)"
    except Exception as e:
        print(f"CropScape failed ({e}); using CSB column {col} == 1 instead")
        csb["corn_frac"] = 1.0
        corn_src = f"CSB attribute {col}"
    q = csb[csb.corn_frac >= MIN_CORN_FRAC].copy()
    q["field_id"] = "csb-" + q["CSBID"].astype(str)
    q["fips"] = fips
    q = q[["field_id", "fips", "area_ha", "corn_frac", "geometry"]].to_crs(4326)
    q.to_file(fpath, driver="GPKG")
    log(f"{region}: CSB polygons in county {n_total}, corn fields kept {len(q)}, "
        f"area {q.area_ha.sum():,.0f} ha ({q.area_ha.sum()*2.471:,.0f} ac); corn mask: {corn_src}; "
        f"read {time.time()-t0:.0f} s")
    return q

fields_by_region = {}
for st, fp in COUNTIES:
    fields_by_region[f"{st}_{fp}"] = collect_fields(st, fp)
    print(f"{st}_{fp}: {len(fields_by_region[f'{st}_{fp}'])} corn fields in the county")

# %% [markdown]
# ## 4. Sentinel-2 (and optionally Sentinel-1) per-field time series
#
# Each county is split into 5 km tiles. Each tile does one STAC search and one load, extracts every field in it, and saves a small CSV part. Reruns skip finished tiles. With N_FIELDS set, whole tiles are picked at random until about N_FIELDS fields are covered.
#
# Indices per pixel, then median per field: NDVI, EVI2, NDRE, NDMI (canopy water), NDTI (residue), NDSVI. All 10 band medians are kept too.

# %%
import pystac_client, planetary_computer, odc.stac
from odc.geo.geom import box as gbox
from collections import defaultdict

odc.stac.configure_rio(cloud_defaults=True)
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)

S2_REFL = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S2_BANDS = S2_REFL + ["SCL"]

def nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (a - b) / (a + b)
    x[~np.isfinite(x)] = np.nan
    return x

def label_raster(fields_utm, geobox):
    shapes = [(g, i + 1) for i, g in enumerate(fields_utm.geometry.buffer(-BUFFER_M))
              if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(tuple(geobox.shape), dtype="int32")
    return rasterize(shapes, out_shape=tuple(geobox.shape), transform=geobox.affine, fill=0, dtype="int32")

def extract_s2(arrs, times, lab, fids):
    m = lab > 0
    L = lab[m]; nf = len(fids)
    tot = np.bincount(L, minlength=nf + 1)
    px = {b: arrs[b][:, m] for b in S2_BANDS}
    out = []
    for ti, t in enumerate(pd.to_datetime(times)):
        ok = np.isin(px["SCL"][ti], SCL_OK)
        off = 1000 if t >= pd.Timestamp("2022-01-25") else 0      # L2A processing baseline 04.00 offset
        r = {}
        for b in S2_REFL:
            v = px[b][ti].astype("float32")
            ok &= v > 0
            r[b] = np.clip(v - off, 0, None) / 10000.0
        if not ok.any():
            continue
        n_ok = np.bincount(L[ok], minlength=nf + 1)
        clear = n_ok / np.maximum(tot, 1)
        good = np.where(clear >= MIN_CLEAR)[0]
        good = good[good > 0]
        if len(good) == 0:
            continue
        sel = ok & np.isin(L, good)
        b = {k: v[sel] for k, v in r.items()}
        df = pd.DataFrame({
            "lab": L[sel],
            "ndvi": nd(b["B08"], b["B04"]),
            "evi2": 2.5 * (b["B08"] - b["B04"]) / (b["B08"] + 2.4 * b["B04"] + 1.0),
            "ndre": nd(b["B8A"], b["B05"]),
            "ndmi": nd(b["B8A"], b["B11"]),
            "ndti": nd(b["B11"], b["B12"]),
            "ndsvi": nd(b["B11"], b["B04"]),
            **{k.lower(): v for k, v in b.items()},
        })
        med = df.groupby("lab").median()
        med["clear_frac"] = clear[med.index]
        med["n_pixels"] = n_ok[med.index]
        med["n_pixels_total"] = tot[med.index]
        med["field_id"] = np.asarray(fids)[med.index - 1]
        med["date"] = t.date().isoformat()
        med["doy"] = t.dayofyear
        out.append(med.reset_index(drop=True))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def extract_s1(vv, vh, times, lab, fids):
    m = lab > 0
    L = lab[m]; nf = len(fids)
    tot = np.bincount(L, minlength=nf + 1)
    VV, VH = vv[:, m], vh[:, m]
    out = []
    for ti, t in enumerate(pd.to_datetime(times)):
        a, b = VV[ti].astype("float32"), VH[ti].astype("float32")
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if not ok.any():
            continue
        n_ok = np.bincount(L[ok], minlength=nf + 1)
        cover = n_ok / np.maximum(tot, 1)
        good = np.where(cover >= MIN_S1_COVER)[0]; good = good[good > 0]
        if len(good) == 0:
            continue
        sel = ok & np.isin(L, good)
        med = pd.DataFrame({"lab": L[sel], "vv": a[sel], "vh": b[sel]}).groupby("lab").median()
        med["vv_db"] = 10 * np.log10(med.vv); med["vh_db"] = 10 * np.log10(med.vh)
        med["vh_vv_db"] = med.vh_db - med.vv_db
        med["n_pixels"] = n_ok[med.index]
        med["field_id"] = np.asarray(fids)[med.index - 1]
        med["date"] = t.date().isoformat(); med["doy"] = t.dayofyear
        out.append(med.drop(columns=["vv", "vh"]).reset_index(drop=True))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def load_cube(items, bands, geom, epsg):
    return odc.stac.load(items, bands=bands, crs=f"EPSG:{epsg}", resolution=RES, geopolygon=geom,
                         groupby="solar_day", pool=READ_THREADS, fail_on_error=False)

def process_tile(sensor, year, sub, epsg, window):
    minx, miny, maxx, maxy = sub.total_bounds
    geom = gbox(minx - 20, miny - 20, maxx + 20, maxy + 20, crs=f"EPSG:{epsg}")
    bb = geom.to_crs("EPSG:4326").boundingbox
    bbox_ll = [bb.left, bb.bottom, bb.right, bb.top]
    when = f"{year}-{window[0]}/{year}-{window[1]}"
    fids = sub.field_id.values
    if sensor == "s2":
        items = retry(lambda: list(catalog.search(collections=["sentinel-2-l2a"], bbox=bbox_ll, datetime=when,
                                                  query={"eo:cloud_cover": {"lt": MAX_CLOUD}}).items()))
        if not items:
            return pd.DataFrame(), 0
        ds = retry(lambda: load_cube(items, S2_BANDS, geom, epsg))
        lab = label_raster(sub, ds.odc.geobox)
        return extract_s2({b: ds[b].values for b in S2_BANDS}, ds.time.values, lab, fids), len(items)
    items = retry(lambda: list(catalog.search(collections=["sentinel-1-rtc"], bbox=bbox_ll, datetime=when).items()))
    if not items:
        return pd.DataFrame(), 0
    groups = defaultdict(list)
    for it in items:
        groups[(it.properties.get("sat:orbit_state"), it.properties.get("sat:relative_orbit"))].append(it)
    frames = []
    for (orb, rel), its in groups.items():
        ds = retry(lambda: load_cube(its, ["vv", "vh"], geom, epsg))
        lab = label_raster(sub, ds.odc.geobox)
        d = extract_s1(ds.vv.values, ds.vh.values, ds.time.values, lab, fids)
        if len(d):
            d["orbit_state"], d["relative_orbit"] = orb, rel
            frames.append(d)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), len(items)

def build_jobs(region):
    """List of (year, tile_id, fields_subset_utm, epsg, window). Picks whole tiles at random when N_FIELDS is set."""
    f = fields_by_region[region]
    c = f.to_crs(5070).unary_union.centroid
    lon, lat = gpd.GeoSeries([c], crs=5070).to_crs(4326).iloc[0].coords[0]
    epsg = utm_epsg(lon, lat)
    fu = f.to_crs(epsg)
    cen = fu.geometry.centroid
    tile = (np.floor(cen.x / TILE_SIZE_M).astype(int).astype(str) + "_" +
            np.floor(cen.y / TILE_SIZE_M).astype(int).astype(str))
    groups = {t: sub for t, sub in fu.groupby(tile.values)}
    order = list(groups)
    np.random.default_rng(RANDOM_SEED).shuffle(order)
    if isinstance(N_FIELDS, int):
        keep, n = [], 0
        for t in order:
            if n >= N_FIELDS:
                break
            keep.append(t); n += len(groups[t])
        order = keep
    jobs = [(YEAR, t, groups[t], epsg, window_for(region)) for t in order]
    if TEST:
        y, t, sub, e, w = max(jobs, key=lambda j: len(j[2]))
        jobs = [(y, t, sub.head(4), e, w)]
    return jobs

def run_region(region, sensor):
    rdir = region_dir(region); pdir = rdir / "parts"; pdir.mkdir(exist_ok=True)
    jobs = build_jobs(region)
    tag = "_test" if TEST else ""
    todo = [j for j in jobs if not (pdir / f"{sensor}{tag}_{j[0]}_{j[1]}.done").exists()]
    print(f"{region} {sensor}: {len(jobs)} tiles, {sum(len(j[2]) for j in jobs)} fields, {len(todo)} to do, window {jobs[0][4] if jobs else ''}")
    n_scenes, failed, t0 = 0, [], time.time()

    def work(job):
        y, t, sub, epsg, window = job
        df, n = process_tile(sensor, y, sub, epsg, window)
        if len(df):
            df["year"] = y
            df["tile"] = t
        df.to_csv(pdir / f"{sensor}{tag}_{y}_{t}.csv", index=False)
        (pdir / f"{sensor}{tag}_{y}_{t}.done").touch()
        return len(sub), n, len(df)

    with ThreadPoolExecutor(N_WORKERS) as ex, tqdm(total=len(todo), desc=f"{region} {sensor} tiles") as bar:
        futs = {ex.submit(work, j): j for j in todo}
        for fu in as_completed(futs):
            try:
                nf, n, nr = fu.result(); n_scenes += n
            except Exception as e:
                j = futs[fu]; failed.append(f"{j[0]}_{j[1]}"); print(f"  tile {j[1]} failed: {e}")
            bar.update(1); bar.set_postfix(elapsed=f"{(time.time()-t0)/60:.1f} min", failed=len(failed))

    parts = sorted(glob.glob(str(pdir / f"{sensor}{tag}_*.csv")))
    dfs = [pd.read_csv(p, dtype={"field_id": str}) for p in parts if os.path.getsize(p) > 1]
    dfs = [d for d in dfs if len(d)]
    if dfs:
        allp = pd.concat(dfs, ignore_index=True).sort_values(["field_id", "date"])
        lead = ["field_id", "year", "date", "doy"]
        allp = allp[lead + [c for c in allp.columns if c not in lead]]
        allp.to_csv(rdir / f"{sensor}_timeseries{tag}.csv", index=False)
        print(f"  saved {rdir / f'{sensor}_timeseries{tag}.csv'}: {len(allp):,} rows, {allp.field_id.nunique()} fields")
    log(f"{region} {sensor}{tag}: tiles {len(jobs)} (ran {len(todo)}), scenes {n_scenes}, failed {failed}, "
        f"window {window_for(region)}, cloud<{MAX_CLOUD}, SCL {SCL_OK}, clear>={MIN_CLEAR}, buffer {BUFFER_M} m")
    if failed:
        print("  Rerun this cell to retry failed tiles.")

# %%
# Sentinel-2 (the main signal)
for region in fields_by_region:
    run_region(region, "s2")

# %%
# Sentinel-1 radar (off by default; set RUN_S1 = True in the config cell)
if RUN_S1:
    for region in fields_by_region:
        run_region(region, "s1")
else:
    print("RUN_S1 is False: skipping radar")

# %% [markdown]
# ## 5. Weather (gridMET, optional)

# %%
GRIDMET = {"tmmx": "tmax_c", "tmmn": "tmin_c", "pr": "pr_mm"}

def gridmet_points(var, year, lon, lat):
    import xarray as xr
    path = download(f"https://www.northwestknowledge.net/metdata/data/{var}_{year}.nc", LOCAL / "gridmet" / f"{var}_{year}.nc")
    ds = xr.open_dataset(path)
    da = ds[[v for v in ds.data_vars if ds[v].ndim == 3][0]]
    tdim = [d for d in da.dims if d not in ("lat", "lon")][0]
    latsl = slice(lat.max() + 0.1, lat.min() - 0.1) if da.lat[0] > da.lat[-1] else slice(lat.min() - 0.1, lat.max() + 0.1)
    da = da.sel(lat=latsl, lon=slice(lon.min() - 0.1, lon.max() + 0.1)).load()
    pts = da.sel(lat=xr.DataArray(lat, dims="f"), lon=xr.DataArray(lon, dims="f"), method="nearest")
    vals = pts.transpose(tdim, "f").values.astype("float32")
    if da.attrs.get("units", "").upper() in ("K", "KELVIN"):
        vals -= 273.15
    return pd.to_datetime(da[tdim].values), vals

def weather_region(region, years):
    f = fields_by_region[region]
    if TEST: f = f.head(4)
    cen = f.to_crs(5070).centroid.to_crs(4326)
    lon, lat = cen.x.values, cen.y.values
    frames = []
    for y in years:
        cols = {}
        for var, name in GRIDMET.items():
            days, cols[name] = gridmet_points(var, y, lon, lat)
        nd_, nf = cols["tmax_c"].shape
        df = pd.DataFrame({"field_id": np.tile(f.field_id.values, nd_), "year": y, "date": np.repeat(days.date, nf)})
        for name, v in cols.items():
            df[name] = v.ravel()
        tavg = np.minimum((df.tmax_c + df.tmin_c) / 2, 30)
        df["gdd"] = np.maximum(0, tavg - 10)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    tag = "_test" if TEST else ""
    out.to_csv(region_dir(region) / f"weather{tag}.csv", index=False)
    log(f"{region} weather{tag}: gridMET {list(GRIDMET)} years {years}, {len(out):,} rows")

if RUN_WEATHER:
    for region in fields_by_region:
        weather_region(region, [YEAR])
else:
    print("RUN_WEATHER is False: skipping gridMET")

# %% [markdown]
# ## 6. County and dairy context
#
# - NASS Quick Stats (needs the free key): county grain and silage acres from the 2022 Census, and milk cows. This is the ground truth for the county check.
# - California: dairy points from the CA Water Boards map service.
# - Any state: the team's dairy CSV (lat/lon or latitude/longitude). Distance to the nearest dairy and counts within 10 and 25 km.

# %%
from sklearn.neighbors import BallTree

def quickstats(**params):
    r = retry(lambda: requests.get("https://quickstats.nass.usda.gov/api/api_GET/",
                                   params={"key": QUICKSTATS_KEY, "format": "JSON", **params}, timeout=180))
    if r.status_code != 200:
        print("  Quick Stats:", r.status_code, r.text[:200]); return pd.DataFrame()
    return pd.DataFrame(r.json().get("data", []))

def qs_value(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")  # (D), (Z) -> NaN

def county_context(states):
    rows = []
    for st in states:
        for short, src, yr, name in [
            ("CORN, GRAIN - ACRES HARVESTED",  "CENSUS", 2022, "grain_acres_2022"),
            ("CORN, SILAGE - ACRES HARVESTED", "CENSUS", 2022, "silage_acres_2022"),
            ("CORN, GRAIN - ACRES HARVESTED",  "CENSUS", 2017, "grain_acres_2017"),
            ("CORN, SILAGE - ACRES HARVESTED", "CENSUS", 2017, "silage_acres_2017"),
            ("CATTLE, COWS, MILK - INVENTORY", "CENSUS", 2022, "milk_cows_2022"),
        ]:
            d = quickstats(state_alpha=st, short_desc=short, source_desc=src, year=yr,
                           agg_level_desc="COUNTY", domain_desc="TOTAL")
            if d.empty: continue
            if "reference_period_desc" in d:
                d = d.sort_values("reference_period_desc").drop_duplicates(["state_fips_code", "county_code"], keep="last")
            d = d[d.county_code != "998"]
            rows.append(pd.DataFrame({"fips": d.state_fips_code + d.county_code, "state": st,
                                      "county": d.county_name, "var": name, "value": qs_value(d.Value)}))
    if not rows:
        return pd.DataFrame()
    t = pd.concat(rows).pivot_table(index=["fips", "state", "county"], columns="var", values="value").reset_index()
    for y in (2022, 2017):
        g, s = t.get(f"grain_acres_{y}"), t.get(f"silage_acres_{y}")
        if g is not None and s is not None:
            t[f"silage_share_{y}"] = s / (g + s)   # 2017 is the prior (years before the target year), 2022 the check
    if "milk_cows_2022" in t and "grain_acres_2022" in t:
        t["cows_per_1000_corn_ac"] = 1000 * t.milk_cows_2022 / (t.grain_acres_2022 + t.silage_acres_2022)
    return t

def arcgis_points(service):
    info = requests.get(service, params={"f": "json"}, timeout=60).json()
    for l in info.get("layers", []):
        if l.get("geometryType") not in (None, "esriGeometryPoint"):
            continue
        feats, off = [], 0
        while True:
            r = requests.get(f"{service}/{l['id']}/query", timeout=120, params={
                "where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson",
                "resultOffset": off, "resultRecordCount": 1000}).json()
            fs = r.get("features", []); feats += fs; off += len(fs)
            if not fs or not r.get("exceededTransferLimit", r.get("properties", {}).get("exceededTransferLimit")):
                break
        g = gpd.GeoDataFrame.from_features(feats, crs=4326) if feats else gpd.GeoDataFrame()
        if len(g) and (g.geom_type == "Point").all():
            print(f"  using layer {l['id']} '{l['name']}': {len(g)} points")
            return g
    return gpd.GeoDataFrame()

def dairy_features(fields, dairies):
    R = 6371.0
    cen = fields.to_crs(5070).centroid.to_crs(4326)
    fx = np.radians(np.c_[cen.y, cen.x])
    tree = BallTree(np.radians(np.c_[dairies.geometry.y, dairies.geometry.x]), metric="haversine")
    out = pd.DataFrame({"field_id": fields.field_id.values})
    out["dairy_dist_km"] = tree.query(fx, k=1)[0][:, 0] * R
    for km in (10, 25):
        out[f"dairies_within_{km}km"] = tree.query_radius(fx, r=km / R, count_only=True)
    return out

try:
    display
except NameError:          # running as a plain script
    display = print

def county_context_local(counties):
    """No API key needed: the workspace QuickStats CSV covers all 50 states. Same clean rules as our critic
    (final values, all practices, Census for county rows)."""
    sys.path[:0] = ["/workspace/FieldVision/.deps", "/workspace/FieldVision/rag-pipeline"]
    os.environ.setdefault("QUICKSTATS_CSV", "/workspace/02_QuickStats/corn_grain_silage_acres_2012_2025.csv")
    import rag_pipeline as rp, critic_toolbox as ct
    rows = []
    for st, fips in counties:
        state, county = ct.fips_to_names(fips)
        by = {r["year"]: r for r in rp.silage_share_history(state, county)["by_year"]} if county else {}
        row = {"fips": fips, "state": st, "county": county}
        for y in (2012, 2017, 2022):
            if y in by:
                row[f"silage_acres_{y}"] = by[y]["silage_acres"]
                row[f"grain_acres_{y}"] = by[y]["grain_acres"]
                row[f"silage_share_{y}"] = by[y]["silage_share"]
        rows.append(row)
    return pd.DataFrame(rows)

states = sorted({st for st, _ in COUNTIES})
cc = county_context(states) if QUICKSTATS_KEY else county_context_local(COUNTIES)
if len(cc):
    cc.to_csv(OUT / "county_context.csv", index=False)
    log(f"County context ({'Quick Stats API' if QUICKSTATS_KEY else 'workspace QuickStats CSV'}): {len(cc)} counties")
    display(cc[cc.fips.isin([f for _, f in COUNTIES])])
else:
    print("No county context found")

dairy_sets = []
if DAIRY_CSV and Path(DAIRY_CSV).exists():
    d = pd.read_csv(DAIRY_CSV)
    lat = "lat" if "lat" in d else "latitude"
    lon = "lon" if "lon" in d else "longitude"
    dairy_sets.append(gpd.GeoDataFrame(geometry=gpd.points_from_xy(d[lon], d[lat]), crs=4326))
if any(st == "CA" for st, _ in COUNTIES):
    try:
        ca = arcgis_points(CA_DAIRY_SERVICE)
        if len(ca):
            ca.to_file(RAW / "ca_dairies.gpkg", driver="GPKG"); dairy_sets.append(ca[["geometry"]])
    except Exception as e:
        print("  CA dairy service failed:", e)
dairies = pd.concat([d[["geometry"]] for d in dairy_sets], ignore_index=True) if dairy_sets else None

for region, f in fields_by_region.items():
    ctx = pd.DataFrame({"field_id": f.field_id.values, "fips": f.fips.values,
                        "area_ha": f.area_ha.values, "corn_frac": f.corn_frac.values})
    if len(cc):
        ctx = ctx.merge(cc.drop(columns=["state", "county"]), on="fips", how="left")
    if dairies is not None and len(dairies):
        ctx = ctx.merge(dairy_features(f, gpd.GeoDataFrame(dairies, crs=4326)), on="field_id", how="left")
    ctx.to_csv(region_dir(region) / "field_context.csv", index=False)
    print(region, "field_context.csv:", ctx.shape)

# %% [markdown]
# ## 7. Export in FieldVision format
#
# Writes, per county: fieldvision_export/counties/<fips>/<YEAR>/ with fields.geojson, timeseries.parquet and fields_coverage.csv (same columns as the Frederick files), only for fields that got satellite data. Then the toolbox can run on it:
#
#     Toolbox(data_root="/workspace/hima_data/fieldvision_export/counties")

# %%
def export_fieldvision(region):
    st, fips = region.split("_")
    tag = "_test" if TEST else ""
    ts_path = OUT / region / f"s2_timeseries{tag}.csv"
    if not ts_path.exists():
        print(region, "no Sentinel-2 table yet"); return
    ts = pd.read_csv(ts_path, dtype={"field_id": str})
    f = fields_by_region[region]
    f = f[f.field_id.isin(set(ts.field_id))].copy()
    cen = f.to_crs(5070).centroid.to_crs(4326)
    out = FV_EXPORT / fips / str(YEAR); out.mkdir(parents=True, exist_ok=True)
    gj = gpd.GeoDataFrame({
        "id": f.field_id.values, "field_id": f.field_id.values, "label": None, "year": YEAR, "country": "US",
        "source": "nass-csb", "crop_code": 1, "crop_name": "Corn", "area_ha": f.area_ha.values,
        "block_id": "", "centroid_lon": cen.x.values, "centroid_lat": cen.y.values,
        "acres_reported": f.area_ha.values * 2.47105, "county_fips": fips}, geometry=f.geometry.values, crs=4326)
    gj.to_file(out / "fields.geojson", driver="GeoJSON")
    t = pd.DataFrame({"field_id": ts.field_id, "date": ts.date, "scene_id": "S2_" + ts.date.astype(str),
                      "collection": "planetary:sentinel-2-l2a",
                      "n_pixels": ts.n_pixels_total, "n_clear": ts.n_pixels, "clear_fraction": ts.clear_frac})
    for b in S2_REFL:
        t[b] = ts[b.lower()]
    t.sort_values(["field_id", "date"]).to_parquet(out / "timeseries.parquet", index=False)
    cov = []
    for fid, g in t.groupby("field_id"):
        d = pd.to_datetime(g.date).sort_values()
        gap = int(d.diff().dt.days.max()) if len(d) > 1 else 999
        cov.append({"field_id": fid, "n_pixels": int(g.n_pixels.max()), "usable_dates": len(d), "largest_gap_days": gap,
                    "status": "ok" if len(d) >= 5 else "too_few_dates",
                    "first_usable_date": d.iloc[0].date().isoformat(), "last_usable_date": d.iloc[-1].date().isoformat()})
    pd.DataFrame(cov).to_csv(out / "fields_coverage.csv", index=False)
    log(f"{region}: exported {len(gj)} fields, {len(t):,} rows to {out}")

for region in fields_by_region:
    export_fieldvision(region)

# %% [markdown]
# ## 8. Quick look: NDVI curves

# %%
import matplotlib.pyplot as plt
tag = "_test" if TEST else ""
for region in fields_by_region:
    p = OUT / region / f"s2_timeseries{tag}.csv"
    if not p.exists(): continue
    df = pd.read_csv(p, dtype={"field_id": str})
    ids = pd.Series(df.field_id.unique()).sample(min(50, df.field_id.nunique()), random_state=RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(10, 4))
    for fid in ids:
        s = df[df.field_id == fid]
        ax.plot(s.doy, s.ndvi, color="gray", alpha=0.4, lw=1)
    ax.set(title=f"{region} {YEAR}: NDVI, {len(ids)} fields", xlabel="day of year", ylabel="NDVI", ylim=(-0.1, 1))
    fig.savefig(OUT / region / f"ndvi_curves_{YEAR}{tag}.png", dpi=120, bbox_inches="tight"); plt.show()
