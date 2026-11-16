"""Pure pixel aggregation, geometry selection and coverage helpers."""
import numpy as np
import pandas as pd
from rasterio.features import rasterize

BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
COLUMNS = ["field_id", "date", "scene_id", "collection", "n_pixels", "n_clear", "clear_fraction", *BANDS]


def scl_clear(scl):
    return np.isin(scl, (4, 5, 6, 7))


def polygon_mask(geometry, transform, shape):
    if geometry.is_empty:
        return np.zeros(shape, dtype=bool)
    return rasterize([(geometry, 1)], out_shape=shape, transform=transform, dtype="uint8").astype(bool)


def interior_geometry(geometry, transform, shape, buffer_m=5, min_pixels=4):
    """Use a 5 m inset only if four actual grid pixel centers remain."""
    inset = geometry.buffer(-buffer_m)
    if not inset.is_empty and polygon_mask(inset, transform, shape).sum() >= min_pixels:
        return inset, True
    return geometry, False


def zonal_medians(labels, bands, clear, count):
    """Aggregate label IDs 1..count; missing data never enters a median."""
    valid = clear.copy()
    for band in bands.values():
        valid &= np.isfinite(band)
    n_pixels = np.bincount(labels.ravel(), minlength=count + 1)[1:count + 1]
    zone = labels[valid]
    order = np.argsort(zone, kind="stable")
    zone = zone[order]
    starts = np.searchsorted(zone, np.arange(1, count + 2))
    n_clear = np.diff(starts)
    result = {"n_pixels": n_pixels, "n_clear": n_clear,
              "clear_fraction": np.divide(n_clear, n_pixels, out=np.zeros(count, dtype=float), where=n_pixels > 0)}
    for name, band in bands.items():
        values = band[valid][order]
        result[name] = np.array([np.median(values[a:b]) if b > a else np.nan
                                 for a, b in zip(starts[:-1], starts[1:])])
    return pd.DataFrame(result)


def mask_bright_rows(table, max_blue):
    """Optional fog/cloud guard, off unless ``max_blue`` is given.

    SCL sometimes labels fog or low cloud as bare soil (class 5); those field
    medians are bright in every band (B02 ~0.45). A row whose median B02 exceeds
    ``max_blue`` is turned into a no-clear-pixel observation: bands NaN, n_clear 0.
    Returns the table and the number of rows masked.
    """
    if max_blue is None:
        return table, 0
    bright = table["B02"] > max_blue
    table = table.copy()
    table.loc[bright, list(BANDS)] = np.nan
    table.loc[bright, ["n_clear", "clear_fraction"]] = 0
    return table, int(bright.sum())


def resolve_duplicates(table):
    if table.empty:
        return table.copy()
    # Collection 1 wins ties; cloud-free pixel count always takes priority.
    return (table.sort_values(["field_id", "date", "n_clear", "clear_fraction", "collection", "scene_id"],
                              ascending=[True, True, False, False, True, True], kind="stable")
            .drop_duplicates(["field_id", "date"]).sort_values(["field_id", "date"]).reset_index(drop=True))


def coverage_table(fields, table, start, end, min_clear_pixels=1, min_clear_fraction=0.2):
    groups = {key: group for key, group in table.groupby("field_id")}
    rows = []
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for field in fields:
        field_id = field["field_id"]
        group = groups.get(field_id, table.iloc[:0])
        pixels = int(group.n_pixels.max()) if len(group) else 0
        usable = group[(group.n_clear >= min_clear_pixels) & (group.clear_fraction >= min_clear_fraction)]
        dates = sorted(pd.to_datetime(usable.date).drop_duplicates())
        # Include both window edges so late starts and early stops are visible.
        edges = [start, *dates, end]
        largest_gap = max((b - a).days for a, b in zip(edges[:-1], edges[1:]))
        status = field.get("error") or ("no_scenes" if group.empty else
                 "no_pixels" if pixels == 0 else "no_clear_dates" if not dates else "ok")
        rows.append({"field_id": field_id, "n_pixels": pixels, "usable_dates": len(dates),
                     "largest_gap_days": largest_gap, "status": status,
                     "first_usable_date": dates[0].date().isoformat() if dates else None,
                     "last_usable_date": dates[-1].date().isoformat() if dates else None})
    return pd.DataFrame(rows)
