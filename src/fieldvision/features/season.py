"""Per-field season features built around the harvest event.

Input is one field's acquisitions for one season, sorted by date. Only
"usable" observations are used (clear fraction above a threshold, all
bands finite and physically plausible). Nothing is interpolated across
cloud gaps except inside the two greenness integrals, which have their
own coverage check.

Naming convention for the output columns
    peak_*, plateau_*, base_ndvi, amplitude   season shape
    harv_*                                    harvest timing
    pre_*                                     state at the last standing observation
    sen_*                                     senescence slope before harvest
    gi_*                                      greenness integrals
    post_*, drop_*                            after harvest: residue, regreening
    cal_<var>_<halfmonth>                     calendar half-month medians, Jul-Nov
    rel_<var>_<offset>                        15-day medians relative to the field's own peak
    q_*                                       data quality (kept out of the models by default)
    reason_*                                  why a feature group is NaN ("ok" otherwise)

What "harvest" means here: the first sustained fall of NDVI below
base + drop_fraction * (peak - base) after the peak. For silage this is
the chopper. For grain the canopy can dry down below the threshold
before the combine arrives, so the date is really "end of green
canopy". It is still later and more gradual than for silage, which is
what the classifier needs, but it should not be read as the exact
combine date.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict

import numpy as np

from .indices import BANDS, INDEX_NAMES, compute_indices

OK = "ok"
NAN = float("nan")

VAR_NAMES = BANDS + INDEX_NAMES
HALF_MONTHS = ["jul1", "jul2", "aug1", "aug2", "sep1", "sep2", "oct1", "oct2", "nov1", "nov2"]
REL_VARS = INDEX_NAMES + ["ndvifrac"]
PRE_VARS = INDEX_NAMES + ["B11", "B12"]
POST_VARS = ["ndvi", "ndmi", "nbr", "psri", "ndti", "B11", "B12"]

REASON_NAMES = [
    "reason_peak", "reason_plateau", "reason_harvest", "reason_pre", "reason_slope",
    "reason_integral_cal", "reason_integral_rel", "reason_post", "reason_regreen",
    "reason_bins_cal", "reason_bins_rel",
]


@dataclass(frozen=True)
class FeatureConfig:
    # which observations count
    min_clear_fraction: float = 0.6
    min_clear_pixels: int = 1
    # peak search window (day of year) and sanity limits
    peak_window: tuple = (160, 280)       # set per region for early harvests (e.g. Texas)
    min_peak_ndvi: float = 0.5
    min_amplitude: float = 0.25
    # harvest = first sustained fall below base + drop_fraction * amplitude
    drop_fraction: float = 0.35
    confirm_days: float = 25.0
    confirm_obs: int = 2
    # fallback when regrowth (catch crop) follows too soon to confirm the drop:
    # one observation below the threshold that also looks like bare soil
    bare_soil_ndmi: float = 0.05
    # windows
    slope_window_days: float = 30.0
    integral_window: tuple = (213, 304)      # 1 Aug - 31 Oct
    integral_rel_days: float = 75.0          # after the field's own peak
    integral_edge_tolerance: float = 20.0
    post_window_days: float = 20.0
    post_max_obs: int = 3
    regreen_window_days: float = 45.0
    regreen_min_lag_days: float = 10.0
    regreen_min_delta: float = 0.15
    season_window: tuple = (182, 334)        # 1 Jul - 30 Nov, used for quality counts
    rel_bin_start: int = -30
    rel_bin_stop: int = 120
    rel_bin_width: int = 15
    # abstention
    low_evidence_max_interval_days: float = 20.0
    low_evidence_min_usable: int = 6

    def to_dict(self):
        return asdict(self)

    @property
    def rel_bin_edges(self):
        return list(range(self.rel_bin_start, self.rel_bin_stop + 1, self.rel_bin_width))


def _rel_label(start):
    return f"m{-start:03d}" if start < 0 else f"p{start:03d}"


def rel_bin_labels(cfg):
    return [_rel_label(e) for e in cfg.rel_bin_edges[:-1]]


def numeric_feature_names(cfg=FeatureConfig()):
    names = [
        "peak_ndvi", "peak_doy", "base_ndvi", "amplitude", "plateau_start_doy",
        "peak_ndmi", "peak_ndre", "peak_b11",
        "harv_lo_doy", "harv_hi_doy", "harv_mid_doy",
        "harv_days_since_peak", "harv_days_since_plateau_start", "harv_decline_days",
    ]
    names += [f"pre_{v.lower()}" for v in PRE_VARS]
    names += ["pre_ndvi_frac", "pre_ndmi_rel", "pre_ndre_frac", "pre_b11_ratio"]
    names += ["sen_ndvi_slope", "sen_ndmi_slope", "sen_psri_slope", "sen_ndvi_slope_rel"]
    names += ["gi_ndvi_aug_oct", "gi_rel_after_peak"]
    names += [f"post_{v.lower()}" for v in POST_VARS]
    names += ["drop_ndvi", "drop_ndvi_frac", "drop_ndmi", "drop_ndti", "drop_rate_ndvi",
              "post_regreen_delta", "post_regreen_frac", "post_regreen_flag"]
    names += [f"cal_{v.lower()}_{h}" for v in VAR_NAMES for h in HALF_MONTHS]
    names += [f"rel_{v}_{lab}" for v in REL_VARS for lab in rel_bin_labels(cfg)]
    names += [
        "q_n_rows", "q_n_usable", "q_n_usable_season", "q_first_usable_doy",
        "q_last_usable_doy", "q_max_gap_days", "q_mean_clear_fraction",
        "q_harv_width_days", "q_harv_confirmed", "q_sen_n_obs", "q_post_n_obs",
        "q_n_green_cycles", "q_green_cycle_days",
    ]
    return names


_TEMPLATES = {}


def _template(cfg):
    if cfg not in _TEMPLATES:
        _TEMPLATES[cfg] = dict.fromkeys(numeric_feature_names(cfg), NAN)
    return _TEMPLATES[cfg]


# --------------------------------------------------------------------------
# building blocks; each returns its outputs plus a reason code
# --------------------------------------------------------------------------

def usable_mask(clear_fraction, bands, cfg=FeatureConfig(), n_clear=None):
    """True where an acquisition can be trusted.

    A row with NaN bands means the satellite looked and saw nothing clear;
    it is dropped here and never turned into a number.
    """
    cf = np.asarray(clear_fraction, dtype=float)
    bands = np.asarray(bands, dtype=float)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(cf) & (cf >= cfg.min_clear_fraction)
        ok &= np.isfinite(bands).all(axis=1)
        ok &= (bands > -0.05).all(axis=1) & (bands < 1.5).all(axis=1)
        ok &= (bands[:, BANDS.index("B08")] + bands[:, BANDS.index("B04")]) > 0.02
        if n_clear is not None:
            nc = np.asarray(n_clear, dtype=float)
            ok &= ~(nc < cfg.min_clear_pixels)   # unknown n_clear does not disqualify
    return ok


def green_cycles(t, ndvi, level, ndmi=None, bare_soil_ndmi=0.05):
    """Runs of observations with NDVI at or above `level` (start index, end index).

    A single low observation inside a run is tolerated (residual haze or a
    shadow) unless it looks like bare soil, which means the crop was cut.
    """
    n = len(t)
    high = np.isfinite(ndvi) & (ndvi >= level)
    runs = []
    i = 0
    while i < n:
        if not high[i]:
            i += 1
            continue
        a = b = i
        j = i + 1
        while j < n:
            if high[j]:
                b = j
                j += 1
                continue
            soil = ndmi is not None and np.isfinite(ndmi[j]) and ndmi[j] <= bare_soil_ndmi
            if j + 1 < n and high[j + 1] and not soil:
                b = j + 1
                j += 2
                continue
            break
        runs.append((a, b))
        i = b + 1
    return runs


def find_peak(t, ndvi, cfg=FeatureConfig(), ndmi=None):
    """Peak greenness of the main green cycle inside the search window.

    The main cycle is the longest run of observations above half of the
    in-window range, not simply the greenest date: a cover crop in spring, a
    catch crop after harvest or a second crop (double cropping) can be as
    green as the corn but do not last as long. Ties go to the greener run.

    Returns (values, index of the peak observation or None, reason).
    """
    vals = {"peak_ndvi": NAN, "peak_doy": NAN, "base_ndvi": NAN, "amplitude": NAN,
            "q_n_green_cycles": NAN, "q_green_cycle_days": NAN}
    finite = np.isfinite(ndvi)
    in_win = finite & (t >= cfg.peak_window[0]) & (t <= cfg.peak_window[1])
    if not in_win.any():
        return vals, None, "no_obs_in_peak_window"
    base = float(np.clip(np.percentile(ndvi[finite], 5), 0.05, 0.35))
    top = float(np.max(ndvi[in_win]))
    # only runs whose greenest observation falls inside the window compete, so a
    # spring cover crop or an autumn catch crop that merely touches it cannot win
    runs = [(a, b) for a, b in green_cycles(t, ndvi, base + 0.5 * (top - base), ndmi,
                                            cfg.bare_soil_ndmi)
            if in_win[a + int(np.nanargmax(ndvi[a:b + 1]))]]
    if not runs:                       # cannot happen: the top observation is in a run
        runs = [(int(np.argmax(np.where(in_win, ndvi, -np.inf))),) * 2]
    best = max(runs, key=lambda r: (t[r[1]] - t[r[0]],
                                    np.nanmax(np.where(in_win[r[0]:r[1] + 1],
                                                       ndvi[r[0]:r[1] + 1], -np.inf))))
    a, b = best
    seg = np.where(in_win[a:b + 1], ndvi[a:b + 1], -np.inf)
    i = a + int(np.argmax(seg))
    peak = float(ndvi[i])
    vals.update(peak_ndvi=peak, peak_doy=float(t[i]), base_ndvi=base, amplitude=peak - base,
                q_n_green_cycles=float(sum(1 for r in runs if t[r[1]] - t[r[0]] >= 15)),
                q_green_cycle_days=float(t[b] - t[a]))
    if peak < cfg.min_peak_ndvi:
        return vals, None, "peak_too_low"
    if peak - base < cfg.min_amplitude:
        return vals, None, "low_amplitude"
    return vals, i, OK


def find_plateau_start(t, ndvi, i_peak, base, amp):
    """First observation of the green plateau that contains the peak.

    Walk back from the peak to the last observation below half amplitude;
    the plateau starts at the first observation above 90 % after that. If
    the series never goes below half amplitude before the peak, the start
    is not observed and we say so instead of returning the first date.
    """
    half = base + 0.5 * amp
    high = base + 0.9 * amp
    before = np.where(ndvi[:i_peak] < half)[0]
    if len(before) == 0:
        return NAN, "plateau_start_not_observed"
    k0 = int(before[-1]) + 1
    cand = np.where(ndvi[k0:i_peak + 1] >= high)[0]
    if len(cand) == 0:      # cannot happen, the peak itself qualifies
        return NAN, "plateau_start_not_observed"
    return float(t[k0 + int(cand[0])]), OK


def detect_harvest(t, ndvi, i_peak, base, amp, cfg=FeatureConfig(), ndmi=None):
    """First sustained drop after the peak.

    Returns (i_lo, i_hi, confirmed, reason): i_lo is the last observation
    still standing, i_hi the first one clearly harvested. The true date is
    somewhere in between; clouds decide how wide that is.

    A low observation only counts if the next ones (up to `confirm_obs`
    within `confirm_days`) are low too, so a single cloud shadow does not
    look like a harvest. If nothing was observed afterwards the drop is
    accepted but marked unconfirmed.
    """
    n = len(t)
    if i_peak >= n - 1:
        return None, None, False, "no_obs_after_peak"
    thr = base + cfg.drop_fraction * amp
    with np.errstate(invalid="ignore"):
        below = ndvi < thr
    for i in range(i_peak + 1, n):
        if not below[i]:
            continue
        j_end = int(np.searchsorted(t, t[i] + cfg.confirm_days, side="right"))
        stop = min(j_end, i + 1 + cfg.confirm_obs)
        if stop > i + 1 and not below[i + 1:stop].all():
            continue
        confirmed = stop > i + 1
        standing = np.where(~below[i_peak:i])[0]
        i_lo = i_peak + int(standing[-1])
        return i_lo, i, confirmed, OK
    # Nothing sustained. After silage a catch crop is often sown at once, and
    # with a cloud gap the only low observation is followed by regrowth.
    # Accept the first low observation that also looks like bare soil (NDMI
    # near or below zero, which a cloud shadow does not produce), unconfirmed.
    if ndmi is not None:
        with np.errstate(invalid="ignore"):
            soil = below & (np.asarray(ndmi) <= cfg.bare_soil_ndmi)
        cand = np.flatnonzero(soil[i_peak + 1:])
        if len(cand):
            i = i_peak + 1 + int(cand[0])
            standing = np.where(~below[i_peak:i])[0]
            return i_peak + int(standing[-1]), i, False, "single_low_obs"
    return None, None, False, "no_sustained_drop"


def _slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return NAN
    x = x[m] - x[m].mean()
    den = float((x * x).sum())
    if den <= 0:
        return NAN
    return float((x * (y[m] - y[m].mean())).sum() / den)


def senescence_slope(t, series, i_lo, amp, cfg=FeatureConfig()):
    """Linear trend (per day) over the window ending at the last standing observation."""
    vals = {"sen_ndvi_slope": NAN, "sen_ndmi_slope": NAN, "sen_psri_slope": NAN,
            "sen_ndvi_slope_rel": NAN}
    k0 = int(np.searchsorted(t, t[i_lo] - cfg.slope_window_days, side="left"))
    n_obs = i_lo - k0 + 1
    span = t[i_lo] - t[k0]
    if n_obs < 2 or (n_obs == 2 and span < 10):
        return vals, n_obs, "too_few_obs_before_harvest"
    tt = t[k0:i_lo + 1]
    vals["sen_ndvi_slope"] = _slope(tt, series["ndvi"][k0:i_lo + 1])
    vals["sen_ndmi_slope"] = _slope(tt, series["ndmi"][k0:i_lo + 1])
    vals["sen_psri_slope"] = _slope(tt, series["psri"][k0:i_lo + 1])
    vals["sen_ndvi_slope_rel"] = vals["sen_ndvi_slope"] / amp
    return vals, n_obs, OK


def window_integral(t, y, start, end, tol):
    """Trapezoid integral of y over [start, end] on a daily grid.

    Linear interpolation between observations, flat beyond the first and
    last one, and only if the observations reach within `tol` days of both
    ends; otherwise NaN.
    """
    m = np.isfinite(y) & (t >= start - tol) & (t <= end + tol)
    if m.sum() < 3:
        return NAN, "too_few_obs_in_window"
    tt, yy = t[m], y[m]
    if tt[0] > start + tol or tt[-1] < end - tol:
        return NAN, "window_edges_not_observed"
    grid = np.arange(start, end + 1.0)
    vals = np.interp(grid, tt, yy)
    return float(((vals[1:] + vals[:-1]) * 0.5).sum()), OK


def _nanmedian_rows(a):
    """Column-wise median ignoring NaN; all-NaN or empty columns give NaN."""
    if a.shape[0] == 0:
        return np.full(a.shape[1], NAN)
    if a.shape[0] == 1:
        return a[0].astype(float)
    return np.nanmedian(a, axis=0)


# --------------------------------------------------------------------------
# the per-field function
# --------------------------------------------------------------------------

def field_features(doy, month, day, clear_fraction, bands, cfg=FeatureConfig(), n_clear=None):
    """All season features for one field.

    doy            day of the field's season year (1 = 1 Jan), sorted ascending
    month, day     calendar month and day of each acquisition
    clear_fraction share of clear pixels per acquisition
    bands          array (n_acquisitions, 10), columns in indices.BANDS order

    Returns (values, reasons): two dicts. Anything that could not be
    computed is NaN and the matching reason says why.
    """
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _field_features(doy, month, day, clear_fraction, bands, cfg, n_clear)


def _field_features(doy, month, day, clear_fraction, bands, cfg, n_clear):
    out = dict(_template(cfg))
    reasons = dict.fromkeys(REASON_NAMES, "no_timeseries")

    doy = np.asarray(doy, dtype=float)
    bands = np.asarray(bands, dtype=float).reshape(-1, len(BANDS))
    out["q_n_rows"] = float(len(doy))
    out["q_n_usable"] = 0.0
    out["q_n_usable_season"] = 0.0
    if len(doy) == 0:
        return out, reasons

    use = usable_mask(clear_fraction, bands, cfg, n_clear)
    n_use = int(use.sum())
    out["q_n_usable"] = float(n_use)
    if n_use == 0:
        return out, dict.fromkeys(REASON_NAMES, "no_usable_obs")

    t = doy[use]
    B = bands[use]
    month = np.asarray(month)[use]
    day = np.asarray(day)[use]
    idx = compute_indices({b: B[:, j] for j, b in enumerate(BANDS)})
    V = np.column_stack([B] + [idx[k] for k in INDEX_NAMES])
    col = {name: j for j, name in enumerate(VAR_NAMES)}
    ndvi = idx["ndvi"]

    # ---- quality
    s0, s1 = cfg.season_window
    in_season = (t >= s0) & (t <= s1)
    out["q_n_usable_season"] = float(in_season.sum())
    out["q_first_usable_doy"] = float(t[0])
    out["q_last_usable_doy"] = float(t[-1])
    out["q_mean_clear_fraction"] = float(np.mean(np.asarray(clear_fraction, dtype=float)[use]))
    knots = np.concatenate([[s0], t[in_season], [s1]])
    out["q_max_gap_days"] = float(np.diff(knots).max())

    # ---- calendar half-month medians, July to November
    bin_id = (month.astype(int) - 7) * 2 + (day.astype(int) > 15)
    any_bin = False
    for b, label in enumerate(HALF_MONTHS):
        rows = bin_id == b
        if not rows.any():
            continue
        any_bin = True
        med = _nanmedian_rows(V[rows])
        for j, name in enumerate(VAR_NAMES):
            out[f"cal_{name.lower()}_{label}"] = float(med[j])
    reasons["reason_bins_cal"] = OK if any_bin else "no_obs_jul_nov"

    # ---- calendar greenness integral
    out["gi_ndvi_aug_oct"], reasons["reason_integral_cal"] = window_integral(
        t, ndvi, cfg.integral_window[0], cfg.integral_window[1], cfg.integral_edge_tolerance)

    # ---- peak
    peak_vals, i_peak, reason = find_peak(t, ndvi, cfg, idx["ndmi"])
    out.update(peak_vals)
    reasons["reason_peak"] = reason
    if i_peak is None:
        for k in ("reason_plateau", "reason_harvest", "reason_pre", "reason_slope",
                  "reason_integral_rel", "reason_post", "reason_regreen", "reason_bins_rel"):
            reasons[k] = reason
        return out, reasons

    peak, base, amp = out["peak_ndvi"], out["base_ndvi"], out["amplitude"]
    t_peak = float(t[i_peak])
    plateau = np.isfinite(ndvi) & (ndvi >= base + 0.9 * amp) \
        & (t >= cfg.peak_window[0]) & (t <= cfg.peak_window[1]) & (np.abs(t - t_peak) <= 60)
    out["peak_ndmi"] = float(np.nanmedian(idx["ndmi"][plateau]))
    out["peak_ndre"] = float(np.nanmedian(idx["ndre"][plateau]))
    out["peak_b11"] = float(np.nanmedian(B[plateau, col["B11"]]))
    out["plateau_start_doy"], reasons["reason_plateau"] = find_plateau_start(
        t, ndvi, i_peak, base, amp)

    # ---- medians in 15-day steps relative to the field's own peak
    rel_t = t - t_peak
    ndvifrac = (ndvi - base) / amp
    R = np.column_stack([idx[k] for k in INDEX_NAMES] + [ndvifrac])
    edges = cfg.rel_bin_edges
    any_bin = False
    for b, label in enumerate(rel_bin_labels(cfg)):
        rows = (rel_t >= edges[b]) & (rel_t < edges[b + 1])
        if not rows.any():
            continue
        any_bin = True
        med = _nanmedian_rows(R[rows])
        for j, name in enumerate(REL_VARS):
            out[f"rel_{name}_{label}"] = float(med[j])
    reasons["reason_bins_rel"] = OK if any_bin else "no_obs_around_peak"

    # ---- greenness after the peak, as a fraction of the field's own amplitude
    out["gi_rel_after_peak"], reasons["reason_integral_rel"] = window_integral(
        t, np.clip(ndvifrac, 0.0, 1.2), t_peak, t_peak + cfg.integral_rel_days,
        cfg.integral_edge_tolerance)

    # ---- harvest interval
    i_lo, i_hi, confirmed, reason = detect_harvest(t, ndvi, i_peak, base, amp, cfg,
                                                     ndmi=idx["ndmi"])
    reasons["reason_harvest"] = reason
    if i_lo is None:
        for k in ("reason_pre", "reason_slope", "reason_post", "reason_regreen"):
            reasons[k] = reason
        return out, reasons

    t_lo, t_hi = float(t[i_lo]), float(t[i_hi])
    t_mid = 0.5 * (t_lo + t_hi)
    out["harv_lo_doy"] = t_lo
    out["harv_hi_doy"] = t_hi
    out["harv_mid_doy"] = t_mid
    out["q_harv_width_days"] = t_hi - t_lo
    out["q_harv_confirmed"] = 1.0 if confirmed else 0.0
    out["harv_days_since_peak"] = t_mid - t_peak
    if np.isfinite(out["plateau_start_doy"]):
        out["harv_days_since_plateau_start"] = t_mid - out["plateau_start_doy"]
    still_high = np.where(ndvi[i_peak:i_lo + 1] >= base + 0.9 * amp)[0]
    out["harv_decline_days"] = t_mid - float(t[i_peak + int(still_high[-1])])

    # ---- state at the last standing observation
    for name in PRE_VARS:
        out[f"pre_{name.lower()}"] = float(V[i_lo, col[name]])
    out["pre_ndvi_frac"] = (out["pre_ndvi"] - base) / amp
    out["pre_ndmi_rel"] = out["pre_ndmi"] - out["peak_ndmi"]
    if out["peak_ndre"] > 1e-3:
        out["pre_ndre_frac"] = out["pre_ndre"] / out["peak_ndre"]
    if out["peak_b11"] > 1e-3:
        out["pre_b11_ratio"] = out["pre_b11"] / out["peak_b11"]
    reasons["reason_pre"] = OK

    # ---- senescence slope before harvest
    slope_vals, n_slope, reasons["reason_slope"] = senescence_slope(t, idx, i_lo, amp, cfg)
    out.update(slope_vals)
    out["q_sen_n_obs"] = float(n_slope)

    # ---- right after harvest: what is left on the ground
    j_end = int(np.searchsorted(t, t_hi + cfg.post_window_days, side="right"))
    j_end = min(j_end, i_hi + cfg.post_max_obs)
    post = _nanmedian_rows(V[i_hi:j_end])
    for name in POST_VARS:
        out[f"post_{name.lower()}"] = float(post[col[name]])
    out["q_post_n_obs"] = float(j_end - i_hi)
    out["drop_ndvi"] = out["pre_ndvi"] - out["post_ndvi"]
    out["drop_ndvi_frac"] = out["drop_ndvi"] / amp
    out["drop_ndmi"] = out["pre_ndmi"] - out["post_ndmi"]
    out["drop_ndti"] = out["post_ndti"] - out["pre_ndti"]
    out["drop_rate_ndvi"] = (out["pre_ndvi"] - float(ndvi[i_hi])) / max(t_hi - t_lo, 1.0)
    reasons["reason_post"] = OK

    # ---- does it green up again
    k_end = int(np.searchsorted(t, t_hi + cfg.regreen_window_days, side="right"))
    later = np.arange(i_hi, k_end)
    if len(later) < 2 or t[later[-1]] - t_hi < cfg.regreen_min_lag_days:
        reasons["reason_regreen"] = "no_obs_after_harvest"
    else:
        seg = ndvi[later]
        k_min = int(np.nanargmin(seg[: max(j_end - i_hi, 1)]))
        delta = float(np.nanmax(seg[k_min:]) - seg[k_min])
        out["post_regreen_delta"] = delta
        out["post_regreen_frac"] = delta / amp
        out["post_regreen_flag"] = 1.0 if delta >= cfg.regreen_min_delta else 0.0
        reasons["reason_regreen"] = OK

    return out, reasons


# --------------------------------------------------------------------------
# feature bookkeeping
# --------------------------------------------------------------------------

_CALENDAR = {"peak_doy", "plateau_start_doy", "harv_lo_doy", "harv_hi_doy", "harv_mid_doy",
             "gi_ndvi_aug_oct"}
_RELATIVE = {"harv_days_since_peak", "harv_days_since_plateau_start", "harv_decline_days",
             "pre_ndvi_frac", "pre_ndmi_rel", "pre_ndre_frac", "pre_b11_ratio",
             "sen_ndvi_slope_rel", "drop_ndvi_frac", "post_regreen_frac", "gi_rel_after_peak"}


def feature_kind(name):
    """'calendar', 'relative', 'neutral', 'quality' or 'reason' for a column name.

    calendar  depends on the absolute day of year
    relative  measured against the field's own season (days since its peak,
              fraction of its own peak)
    neutral   a physical value at a season-defined moment (e.g. NDMI at the
              last standing observation); used by both variants
    """
    if name.startswith("q_"):
        return "quality"
    if name.startswith("reason_"):
        return "reason"
    if name.startswith("cal_") or name in _CALENDAR:
        return "calendar"
    if name.startswith("rel_") or name in _RELATIVE:
        return "relative"
    return "neutral"


def feature_group(name):
    """Coarse physical group, used for grouped importance and the report."""
    for prefix, group in (("q_", "quality"), ("reason_", "reason"), ("cal_", "calendar_bins"),
                          ("rel_", "relative_bins"), ("harv_", "harvest_timing"),
                          ("pre_", "pre_harvest_state"), ("sen_", "senescence_slope"),
                          ("gi_", "greenness_integral"), ("post_regreen", "regreening")):
        if name.startswith(prefix):
            return group
    if name in ("post_ndti", "drop_ndti", "post_b11", "post_b12"):
        return "residue"
    if name.startswith(("post_", "drop_")):
        return "post_harvest_other"
    return "season_shape"


def signal_features(columns, variant="relative"):
    """Model input columns for a variant: 'relative', 'calendar' or 'all'."""
    if variant not in ("relative", "calendar", "all"):
        raise ValueError(f"unknown variant {variant!r}")
    keep = {"relative": ("relative", "neutral"), "calendar": ("calendar", "neutral"),
            "all": ("relative", "calendar", "neutral")}[variant]
    return [c for c in columns if is_feature_column(c) and feature_kind(c) in keep]


def is_feature_column(name):
    """True for numeric feature columns produced by field_features.

    Keeps ids, labels, the low-evidence flag and any metadata merged into
    the table out of the models.
    """
    if name.startswith(("rel_", "cal_")):
        return True
    return name in set(numeric_feature_names(FeatureConfig()))


def evidence_flags(q_n_usable_season, harv_mid_doy, q_harv_width_days, q_harv_confirmed,
                   cfg=FeatureConfig()):
    """Low-evidence flag and the reason for it.

    Decided from data quality only, never from labels or model output:
    no harvest found, harvest seen only once at the end of a gap, harvest
    interval wider than the limit, or too few usable dates in the season.
    """
    n = np.asarray(q_n_usable_season, dtype=float)
    mid = np.asarray(harv_mid_doy, dtype=float)
    width = np.asarray(q_harv_width_days, dtype=float)
    conf = np.asarray(q_harv_confirmed, dtype=float)
    reason = np.full(len(n), "", dtype=object)

    def add(mask, text):
        for i in np.where(mask)[0]:
            reason[i] = text if not reason[i] else reason[i] + "+" + text

    with np.errstate(invalid="ignore"):
        add(~(n >= cfg.low_evidence_min_usable), "few_usable_dates")
        add(~np.isfinite(mid), "no_harvest_detected")
        add(np.isfinite(mid) & (width > cfg.low_evidence_max_interval_days), "wide_harvest_interval")
        add(np.isfinite(mid) & (conf == 0), "harvest_unconfirmed")
    low = reason != ""
    return low, reason
