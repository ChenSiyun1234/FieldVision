"""Daily gridMET temperature, precipitation, and growing degree days at each field centroid."""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .common import append_log, load_progress, mark_progress, part_path, progress_path, rebuild_master, region_dir, write_frame
from .config import (
    GRIDMET_URL_TEMPLATE,
    GRIDMET_VARS,
    RAW,
    SEASON_END,
    SEASON_START,
    TRANSFER_YEAR,
    WEATHER_COLUMNS,
    YEARS,
)
from .fields import prepare_region

# gridMET cells are 1/24 degree, about 4 km. These are the published cell-center origins.
_CELL_DEG = 1 / 24
_LON0 = -124.76666663333334
_LAT0 = 25.066666666666666
_CACHE: dict[int, dict] = {}


def _clear_partial_downloads() -> None:
    folder = RAW / "gridmet"
    if not folder.exists():
        return
    for part in folder.glob("*.part"):
        part.unlink()


def _opens(path: Path) -> bool:
    """A complete download is a NetCDF file xarray can open."""
    try:
        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            _data_var(dataset)
    except Exception:
        return False
    return True


def _download_once(url: str, part: Path, dest: Path) -> None:
    if part.exists():
        part.unlink()
    try:
        urllib.request.urlretrieve(url, part)
        if not _opens(part):
            raise RuntimeError(f"{part.name} did not open with xarray")
        os.replace(part, dest)
    except Exception:
        if part.exists():
            part.unlink()
        raise


def _download(year: int) -> dict:
    """Download one NetCDF per variable. A 404 is remembered for the pygridmet fallback."""
    if year in _CACHE:
        return _CACHE[year]
    _clear_partial_downloads()
    folder = RAW / "gridmet"
    folder.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path | None] = {}
    notes: list[str] = []
    not_found = False
    for var in GRIDMET_VARS:
        dest = folder / f"{var}_{year}.nc"
        part = Path(str(dest) + ".part")
        if dest.exists() and _opens(dest):
            print(f"Using existing {dest.name}")
            paths[var] = dest
            continue
        if dest.exists():
            dest.unlink()
        url = GRIDMET_URL_TEMPLATE.format(var=var, year=year)
        delay = 2.0
        saved = False
        saw_404 = False
        last_error: Exception | None = None
        for attempt in range(1, 4):
            print(f"Downloading {url} (attempt {attempt}/3)")
            try:
                _download_once(url, part, dest)
                saved = True
                break
            except Exception as exc:  # noqa: BLE001  HTTP, network, and NetCDF errors
                last_error = exc
                code = getattr(exc, "code", None)
                if code == 404:
                    saw_404 = True
                print(f"{var}_{year} attempt {attempt} failed ({code}): {exc}")
                if attempt < 3:
                    time.sleep(delay)
                    delay *= 2
        if saved:
            paths[var] = dest
            continue
        paths[var] = None
        if saw_404:
            not_found = True
            note = f"{var}_{year} returned 404 from {url}"
        else:
            note = f"{var}_{year} failed after 3 attempts: {last_error}"
        notes.append(note)
        print(f"gridMET {note}. Skipping this variable-year.")
        if not_found:
            break
    status = {"paths": paths, "not_found": not_found, "notes": notes}
    _CACHE[year] = status
    return status


def _data_var(dataset: xr.Dataset) -> str:
    for name, variable in dataset.data_vars.items():
        if variable.ndim == 3:
            return name
    raise RuntimeError(f"No 3-D variable in {list(dataset.data_vars)}")


def _axes(dataset: xr.Dataset) -> tuple[str, str, str]:
    names = set(dataset.dims) | set(dataset.coords)
    lon = "lon" if "lon" in names else "longitude"
    lat = "lat" if "lat" in names else "latitude"
    time = "day" if "day" in names else "time"
    return lon, lat, time


def _cell(lon: float, lat: float) -> tuple[float, float]:
    """Nearest 4 km gridMET cell center. Fields in one cell share one request."""
    lon_c = _LON0 + np.round((lon - _LON0) / _CELL_DEG) * _CELL_DEG
    lat_c = _LAT0 + np.round((lat - _LAT0) / _CELL_DEG) * _CELL_DEG
    return (round(float(lon_c), 6), round(float(lat_c), 6))


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=WEATHER_COLUMNS)


def _season_frame(field_ids: np.ndarray, year: int, times, tmax, tmin, rain) -> pd.DataFrame:
    """Same columns and GDD formula for the file download and the pygridmet fallback."""
    if np.nanmedian(tmax) > 100:
        tmax = tmax - 273.15
        tmin = tmin - 273.15
    start = pd.Timestamp(year=year, month=SEASON_START[0], day=SEASON_START[1])
    end = pd.Timestamp(year=year, month=SEASON_END[0], day=SEASON_END[1])
    keep = (times >= start) & (times <= end)
    times = times[keep]
    tmax = tmax[keep]
    tmin = tmin[keep]
    rain = rain[keep]
    average = (tmax + tmin) / 2.0
    gdd = np.maximum(0, np.minimum(average, 30) - 10)
    frames = []
    for index, field_id in enumerate(field_ids):
        frames.append(
            pd.DataFrame(
                {
                    "field_id": field_id,
                    "year": year,
                    "date": pd.DatetimeIndex(times).strftime("%Y-%m-%d"),
                    "tmax_c": tmax[:, index],
                    "tmin_c": tmin[:, index],
                    "pr_mm": rain[:, index],
                    "gdd": gdd[:, index],
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else _empty()


def _sample_files(subset, year: int, paths: dict[str, Path]) -> pd.DataFrame:
    points = subset.to_crs("EPSG:4326")
    lons = points.geometry.centroid.x.to_numpy()
    lats = points.geometry.centroid.y.to_numpy()
    sampled = {}
    times = None
    for var in GRIDMET_VARS:
        dataset = xr.open_dataset(paths[var], engine="h5netcdf")
        try:
            lon_name, lat_name, time_name = _axes(dataset)
            query_lons = lons
            if float(dataset[lon_name].min()) >= 0 and np.nanmin(lons) < 0:
                query_lons = np.where(lons < 0, lons + 360, lons)
            values = dataset[_data_var(dataset)].sel(
                {
                    lon_name: xr.DataArray(query_lons, dims="field"),
                    lat_name: xr.DataArray(lats, dims="field"),
                },
                method="nearest",
            )
            sampled[var] = np.asarray(values.transpose(time_name, "field").values, dtype="float64")
            times = pd.to_datetime(values[time_name].values)
        finally:
            dataset.close()
    if times is None:
        return _empty()
    return _season_frame(subset["field_id"].to_numpy(), year, times, sampled["tmmx"], sampled["tmmn"], sampled["pr"])


def _series(frame: pd.DataFrame, var: str) -> pd.Series:
    for column in frame.columns:
        if str(column).split("(", 1)[0].strip() == var:
            return frame[column]
    raise RuntimeError(f"pygridmet frame has no {var} column: {list(frame.columns)}")


def _request_cell(lon: float, lat: float, year: int) -> pd.DataFrame | None:
    import pygridmet

    start = f"{year}-{SEASON_START[0]:02d}-{SEASON_START[1]:02d}"
    end = f"{year}-{SEASON_END[0]:02d}-{SEASON_END[1]:02d}"
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            return pygridmet.get_bycoords((lon, lat), (start, end), variables=list(GRIDMET_VARS))
        except Exception as exc:  # noqa: BLE001  the service raises several HTTP and IO errors
            last_error = exc
            print(f"pygridmet {lon}, {lat} {year} attempt {attempt} failed: {exc}")
            if attempt < 3:
                time.sleep(delay)
                delay *= 2
    print(f"pygridmet {lon}, {lat} {year} failed after 3 attempts: {last_error}. Skipping this cell.")
    return None


def _sample_pygridmet(subset, year: int) -> pd.DataFrame:
    points = subset.to_crs("EPSG:4326")
    lons = points.geometry.centroid.x.to_numpy()
    lats = points.geometry.centroid.y.to_numpy()
    field_ids = subset["field_id"].to_numpy()
    cells = [_cell(float(lon), float(lat)) for lon, lat in zip(lons, lats)]
    unique = list(dict.fromkeys(cells))
    print(f"pygridmet {year}: {len(field_ids)} fields, {len(unique)} gridMET cells")
    try:
        import pygridmet  # noqa: F401
    except ImportError:
        print("pygridmet is not installed. Run pip install pygridmet. Skipping this year.")
        return _empty()
    fetched: dict[tuple[float, float], pd.DataFrame] = {}
    for lon, lat in unique:
        frame = _request_cell(lon, lat, year)
        if frame is not None and not frame.empty:
            fetched[(lon, lat)] = frame
    kept_ids = []
    columns = {var: [] for var in GRIDMET_VARS}
    times = None
    for field_id, cell in zip(field_ids, cells):
        frame = fetched.get(cell)
        if frame is None:
            continue
        if times is None:
            times = pd.to_datetime(frame.index)
            if getattr(times, "tz", None) is not None:
                times = times.tz_localize(None)
        kept_ids.append(field_id)
        for var in GRIDMET_VARS:
            columns[var].append(_series(frame, var).to_numpy(dtype="float64"))
    if times is None:
        return _empty()
    return _season_frame(
        np.asarray(kept_ids),
        year,
        times,
        np.column_stack(columns["tmmx"]),
        np.column_stack(columns["tmmn"]),
        np.column_stack(columns["pr"]),
    )


def _sample_year(fields, year: int) -> tuple[pd.DataFrame, str]:
    subset = fields.loc[fields["year"] == year].drop_duplicates("field_id")
    if subset.empty:
        return _empty(), "no fields"
    status = _download(year)
    if status["not_found"]:
        print(f"gridMET file URL returned 404 for {year}. Using pygridmet.")
        return _sample_pygridmet(subset, year), "pygridmet"
    if any(status["paths"].get(var) is None for var in GRIDMET_VARS):
        print(f"gridMET {year} is missing a variable after the file download. Skipping this year.")
        return _empty(), "skipped"
    return _sample_files(subset, year, status["paths"]), f"direct NetCDF {GRIDMET_URL_TEMPLATE}"


def main(region: str, test: bool = False) -> None:
    fields, _tiles, _epsg, years, notes = prepare_region(region, test)
    download_notes: list[str] = []
    # Download every configured year once, even when this region only uses one of them.
    for year in sorted(set(YEARS) | {TRANSFER_YEAR}):
        download_notes.extend(_download(year)["notes"])
    progress = progress_path(region, "weather", test=False)
    done = set() if test else load_progress(progress)
    parts = []
    sources = []
    for year in years:
        part = part_path(region, "weather", year, "all", test)
        if (region, "all", int(year)) in done and part.exists():
            print(f"Skipping finished weather {region} {year}")
            parts.append(part)
            sources.append(f"{year}: already extracted")
            continue
        print(f"{region}: extracting gridMET {year} for {fields.loc[fields['year'] == year, 'field_id'].nunique()} fields")
        frame, source = _sample_year(fields, int(year))
        sources.append(f"{year}: {source}")
        write_frame(part, frame, WEATHER_COLUMNS)
        parts.append(part)
        # A failed year stays unfinished so the next run tries it again.
        if not test and len(frame) and source != "skipped":
            mark_progress(progress, region, "all", int(year))
    master = region_dir(region) / ("weather_test.csv" if test else "weather.csv")
    n_rows = rebuild_master(master, parts, WEATHER_COLUMNS)
    print(f"Wrote {n_rows:,} rows to {master}")
    skip_notes = list(notes) + download_notes
    append_log(
        f"gridMET {region}" + (" test" if test else ""),
        [
            ("url", GRIDMET_URL_TEMPLATE),
            ("source", "; ".join(sources) if sources else "none"),
            ("variables", ", ".join(GRIDMET_VARS)),
            ("years downloaded", sorted(set(YEARS) | {TRANSFER_YEAR})),
            ("years extracted", years),
            ("date range", f"{SEASON_START[0]:02d}-{SEASON_START[1]:02d} to {SEASON_END[0]:02d}-{SEASON_END[1]:02d}"),
            ("gdd", "max(0, min((tmax+tmin)/2, 30) - 10) with temperature in C"),
            ("rows written", n_rows),
            ("output", master),
            ("skip notes", "; ".join(skip_notes) if skip_notes else "none"),
        ],
    )
