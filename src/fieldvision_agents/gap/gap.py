"""Gap path: corn fields whose harvest fell inside a cloud gap.

The model abstains when the harvest bracket (last clear look with the crop standing, first clear
look after the cut) is too wide. Those fields are not random: silage is cut in a few weeks, and
if those weeks are cloudy the county loses mostly silage fields. This module does three things,
all from data already on disk plus two image chips per checked field:

1. gap_table / gap_summary   sort every field into clear, gap or unseen, name the main cloud gaps,
                             and give the county share three ways with a lower and an upper bound
2. check_gap_field           Nemotron Super VL looks at the last clear image before the gap and the
                             first clear image after it: bare soil (silage-like) or residue (grain-like)
3. evidence_record / explain one traceable record per gap field; Nemotron Ultra writes the reason
                             and code checks every number in it against the record

Code decides every number. The vision answer is extra evidence: it marks a gap field as
supported, conflicting or unclear. It never changes the model's call or probability.
Nothing here reads a USDA figure or a field label.

In a terminal:
    python -m fieldvision_agents.gap.gap 36121 2022 --data <counties dir> --vision 40
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HA_TO_ACRE = 2.471053814
PAD_DAYS = 1              # image windows: the last standing date and the drop date, plus or minus this
MAX_SCENE_CLOUD = 100     # no scene-level cloud filter: the field itself was clear on both dates
MIN_VISION_CONFIDENCE = 0.6
KINDS = ("clear", "gap", "unseen")


# ------------------------------------------------------------------ 1. sort the fields

def gap_table(rows, area_ha):
    """One row per field from the toolbox's field rows and a field_id -> hectares mapping."""
    out = []
    for r in rows:
        e = r.get("evidence") or {}
        tier = e.get("evidence_tier") or ("abstain" if r.get("abstain") else "wide")
        both = bool(e.get("drop_date")) and bool(e.get("last_standing_date"))
        kind = "clear" if tier == "clear" else ("gap" if both else "unseen")
        fid = r.get("source_field_id", r.get("field_id"))
        out.append({
            "field_id": fid, "acres": float(area_ha.get(fid, np.nan)) * HA_TO_ACRE,
            "tier": tier, "kind": kind, "abstain": bool(r.get("abstain")),
            "p_silage": float(r["p_silage"]) if r.get("p_silage") is not None else np.nan,
            "call": r.get("label"),
            "last_standing_date": e.get("last_standing_date"), "drop_date": e.get("drop_date"),
            "bracket_days": e.get("days_to_drop"),
            "ndvi_before": e.get("ndvi_before"), "ndvi_after": e.get("ndvi_after"),
            "greenness_before_share_of_peak": e.get("greenness_before_share_of_peak"),
            "residue_ndti_after": e.get("residue_ndti_after"),
            "abstain_reason": e.get("abstain_reason"),
        })
    return pd.DataFrame(out)


def _share(d):
    a = d["acres"].to_numpy(float)
    ok = np.isfinite(a)
    tot = float(a[ok].sum())
    if tot <= 0:
        return None
    return round(float((a[ok] * (d["call"].to_numpy()[ok] == "silage")).sum()) / tot, 4)


def _r(x, n=1):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def gap_summary(df, top=3):
    """Counts and acres by kind, the main cloud gaps, and the county silage share three ways.

    clear_only        clear-tier fields only (what a clear-tier headline counts)
    called            every field the model did not abstain on
    with_gap_fields   clear fields plus every gap field, abstained ones included, by the model's call
    lower / upper     silage acres among clear + gap fields over all corn; upper adds the unseen corn
    """
    total = float(df["acres"].sum())
    kinds = {}
    for k in KINDS:
        d = df[df["kind"] == k]
        kinds[k] = {"fields": int(len(d)), "acres": _r(d["acres"].sum()),
                    "share_of_corn_acres": _r(d["acres"].sum() / total, 4) if total else None}
    g = df[df["kind"] == "gap"]
    ga = g[g["abstain"]]
    gaps = []
    if len(g):
        grp = g.groupby(["last_standing_date", "drop_date"], dropna=True)
        order = grp["acres"].sum().sort_values(ascending=False).head(top)
        for (a, b), acres in order.items():
            d = grp.get_group((a, b))
            gaps.append({"from": a, "to": b, "days": int((pd.Timestamp(b) - pd.Timestamp(a)).days),
                         "fields": int(len(d)), "abstained_fields": int(d["abstain"].sum()),
                         "acres": _r(acres),
                         "median_ndvi_before": _r(d["ndvi_before"].median(), 2),
                         "median_ndvi_after": _r(d["ndvi_after"].median(), 2),
                         "called_silage_share": _share(d)})
    seen = df[df["kind"] != "unseen"]
    sil = float((seen["acres"] * (seen["call"] == "silage")).sum())
    unseen = float(df.loc[df["kind"] == "unseen", "acres"].sum())
    return {
        "fields": int(len(df)), "corn_acres": _r(total),
        "kinds": kinds,
        "abstained_fields": int(df["abstain"].sum()),
        "abstained_with_bracket": int(len(ga)),
        "abstained_with_bracket_acres": _r(ga["acres"].sum()),
        "abstained_with_bracket_called_silage_share": _share(ga) if len(ga) else None,
        "main_gaps": gaps,
        "silage_share": {
            "clear_only": _share(df[df["kind"] == "clear"]),
            "called": _share(df[~df["abstain"]]),
            "with_gap_fields": _share(seen),
            "lower": _r(sil / total, 4) if total else None,
            "upper": _r((sil + unseen) / total, 4) if total else None,
        },
        "note": "Model estimates from satellite data. Not a NASS statistic. No USDA figure was used.",
    }


def prior_shifted_share(df, prior, training_prior=0.5):
    """County share when every field with a harvest bracket keeps its probability, moved from the
    50/50 balance the model was trained on to the prior, and every unseen field takes the prior.

    prior: the last official silage share from years BEFORE the season (county Census, else the
    state survey mean), so the season's own USDA figure is never an input. Acre-weighted mean."""
    p = np.clip(df["p_silage"].to_numpy(float), 1e-6, 1 - 1e-6)
    odds = p / (1 - p) * (prior / (1 - prior)) / (training_prior / (1 - training_prior))
    shifted = odds / (1 + odds)
    val = np.where((df["kind"] != "unseen").to_numpy() & np.isfinite(shifted), shifted, prior)
    a = df["acres"].to_numpy(float)
    ok = np.isfinite(a)
    return round(float(np.sum(a[ok] * val[ok]) / np.sum(a[ok])), 4) if ok.any() and a[ok].sum() > 0 else None


# ------------------------------------------------------------------ 1b. the cloud gap as evidence

CURVE_SMOOTH = 0.5        # field-equivalents spread evenly over the season per class, so no day has zero mass
CURVE_MIN_FIELDS = 10     # judged fields needed per class before the county's own curves are used


def _doy(s):
    return pd.to_datetime(s, errors="coerce").dt.dayofyear.to_numpy(float)


def harvest_curves(df, smooth=CURVE_SMOOTH):
    """Share of each class's harvest falling on each day, from the fields the model did judge.

    A judged field's harvest lies somewhere in its bracket (last standing date, drop date], so it
    puts one field of mass spread evenly over those days. Returns (days, f_silage, f_grain,
    n_silage, n_grain); each f sums to 1 over the season seen in this county."""
    j = df[(~df["abstain"]) & (df["kind"] != "unseen")]
    a, b = _doy(j["last_standing_date"]), _doy(j["drop_date"])
    ok = np.isfinite(a) & np.isfinite(b) & (b > a)
    allb = df[df["kind"] != "unseen"]
    lo = int(np.nanmin(_doy(allb["last_standing_date"]))) + 1
    hi = int(np.nanmax(_doy(allb["drop_date"])))
    days = np.arange(lo, hi + 1)
    out, n = {}, {}
    for cls in ("silage", "grain"):
        f = np.full(len(days), smooth / len(days))
        m = ok & (j["call"].to_numpy() == cls)
        for x, y in zip(a[m], b[m]):
            sel = (days > x) & (days <= y)
            if sel.any():
                f[sel] += 1.0 / sel.sum()
        out[cls], n[cls] = f / f.sum(), int(m.sum())
    return days, out["silage"], out["grain"], n["silage"], n["grain"]


def interval_probability(df, prior=None, min_fields=CURVE_MIN_FIELDS):
    """P(silage) for every field with a bracket, from WHEN it was harvested and nothing else:

        p = pi * S / (pi * S + (1 - pi) * G)

    S, G = share of the county's silage / grain harvest that fell inside this field's bracket
    (from harvest_curves); pi = the share of judged fields called silage, or the prior argument when given.
    Returns (p as an array aligned with df, info). p is NaN for unseen fields and for every field
    when either class has fewer than min_fields judged fields."""
    p = np.full(len(df), np.nan)
    j = df[(~df["abstain"]) & (df["kind"] != "unseen")]
    pi = float(prior) if prior is not None else (float((j["call"] == "silage").mean()) if len(j) else np.nan)
    info = {"pi": None if not np.isfinite(pi) else round(pi, 4), "used": False}
    if not len(j) or not np.isfinite(pi):
        return p, info
    days, fs, fg, ns, ng = harvest_curves(df)
    info.update({"judged_silage_fields": ns, "judged_grain_fields": ng,
                 "median_silage_harvest_doy": int(days[np.searchsorted(np.cumsum(fs), 0.5)]),
                 "median_grain_harvest_doy": int(days[np.searchsorted(np.cumsum(fg), 0.5)])})
    if ns < min_fields or ng < min_fields:
        return p, info
    pi = float(np.clip(pi, 1e-4, 1 - 1e-4))
    a, b = _doy(df["last_standing_date"]), _doy(df["drop_date"])
    cs, cg = np.concatenate([[0], np.cumsum(fs)]), np.concatenate([[0], np.cumsum(fg)])
    for i in np.flatnonzero(np.isfinite(a) & np.isfinite(b) & (b > a) & (df["kind"] != "unseen").to_numpy()):
        i0 = int(np.clip(a[i] + 1 - days[0], 0, len(days)))
        i1 = int(np.clip(b[i] + 1 - days[0], 0, len(days)))
        S, G = cs[i1] - cs[i0], cg[i1] - cg[i0]
        if S + G > 0:
            p[i] = pi * S / (pi * S + (1 - pi) * G)
    info["used"] = True
    return p, info


def interval_share(df, prior=None):
    """County share when judged fields keep the model's call, abstained fields with a bracket take
    their interval probability, and unseen fields (or all of them, if the curves cannot be built)
    take pi. Acre-weighted. Returns (share, info)."""
    p, info = interval_probability(df, prior)
    pi = info["pi"]
    if pi is None:
        return None, info
    judged = (~df["abstain"]).to_numpy() & (df["kind"] != "unseen").to_numpy()
    val = np.where(judged, (df["call"] == "silage").to_numpy().astype(float), np.where(np.isfinite(p), p, pi))
    a = df["acres"].to_numpy(float)
    ok = np.isfinite(a)
    info["abstained_fields_given_interval_p"] = int((~judged & np.isfinite(p)).sum())
    return (round(float(np.sum(a[ok] * val[ok]) / np.sum(a[ok])), 4) if a[ok].sum() > 0 else None), info


def interval_window(df, date_from, date_to, prior=None):
    """S, G and p for one bracket (date_from, date_to], from this county's harvest curves."""
    p, info = interval_probability(df, prior)
    if not info["used"]:
        return None
    days, fs, fg, _, _ = harvest_curves(df)
    a, b = pd.Timestamp(date_from).dayofyear, pd.Timestamp(date_to).dayofyear
    sel = (days > a) & (days <= b)
    S, G, pi = float(fs[sel].sum()), float(fg[sel].sum()), float(np.clip(info["pi"], 1e-4, 1 - 1e-4))
    if S + G <= 0:
        return None
    return {"silage_harvest_in_gap_pct": round(100 * S, 1), "grain_harvest_in_gap_pct": round(100 * G, 1),
            "judged_fields_called_silage_pct": round(100 * pi, 1),
            "p_silage_from_timing": round(pi * S / (pi * S + (1 - pi) * G), 2)}


# ------------------------------------------------------------------ 2. vision on gap fields

def vision_windows(row, pad=PAD_DAYS):
    """Image windows for one field: the two dates that bracket its gap, when the field was seen clearly."""
    a, b = pd.Timestamp(row["last_standing_date"]), pd.Timestamp(row["drop_date"])
    day = lambda t: t.strftime("%Y-%m-%d")
    d = pd.Timedelta(days=pad)
    return (day(a - d), day(a + d)), (day(b - d), day(b + d))


def check_gap_field(geom, row, llm=None, chip=None, ask=None):
    """Two chips at the edges of the field's own gap, then the vision model's answer."""
    if chip is None or ask is None:
        from ..vision.vision import ask_vision, true_colour_chip
        chip, ask = chip or true_colour_chip, ask or ask_vision
    (b0, b1), (a0, a1) = vision_windows(row)
    b_png, b_date, b_clear = chip(geom, b0, b1, max_cloud=MAX_SCENE_CLOUD)
    a_png, a_date, a_clear = chip(geom, a0, a1, max_cloud=MAX_SCENE_CLOUD)
    v = ask(b_png, a_png, llm)
    v.update({"field_id": row["field_id"], "before_date": b_date, "after_date": a_date,
              "before_clear_fraction": round(float(b_clear), 2),
              "after_clear_fraction": round(float(a_clear), 2)})
    return v, b_png, a_png


def vision_status(call, verdict, min_conf=MIN_VISION_CONFIDENCE):
    """supported / conflict / unclear: does the picture point the same way as the model's call?"""
    hint = (verdict or {}).get("hint")
    if hint not in ("silage-like", "grain-like") or float((verdict or {}).get("confidence", 0)) < min_conf:
        return "unclear"
    return "supported" if (hint == "silage-like") == (call == "silage") else "conflict"


def agreement(results):
    """How often the picture agrees with the model, over fields where the picture was readable."""
    st = [r["status"] for r in results]
    n_read = sum(s != "unclear" for s in st)
    n_sup = sum(s == "supported" for s in st)
    return {"checked": len(st), "readable": n_read, "supported": n_sup,
            "conflict": n_read - n_sup, "unclear": len(st) - n_read,
            "agreement": round(n_sup / n_read, 3) if n_read else None}


def run_vision(df, geoms, kind="gap", n=20, seed=0, out_dir=None, llm=None, chip=None, ask=None,
               log=print):
    """Check up to n fields of one kind, largest gaps of acres first within a seeded shuffle.

    Results are appended to <out_dir>/vision_<kind>.jsonl, and fields already there are not
    asked again, so a stopped run continues where it left off."""
    d = df[(df["kind"] == kind) & df["drop_date"].notna() & df["last_standing_date"].notna()]
    d = d.sample(frac=1, random_state=seed).head(n)
    done, path = {}, None
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = Path(out_dir) / f"vision_{kind}.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    j = json.loads(line)
                    done[j["field_id"]] = j
    results = []
    for _, row in d.iterrows():
        fid = row["field_id"]
        if fid in done:
            results.append(done[fid])
            continue
        try:
            v, b_png, a_png = check_gap_field(geoms[fid], row, llm=llm, chip=chip, ask=ask)
            if out_dir:
                cd = Path(out_dir) / "chips"
                cd.mkdir(exist_ok=True)
                (cd / f"{fid}_before.png").write_bytes(b_png)
                (cd / f"{fid}_after.png").write_bytes(a_png)
        except Exception as e:  # a missing scene or a failed call skips the field, never the run
            v = {"field_id": fid, "hint": "no hint", "confidence": 0.0, "error": str(e)[:200]}
        v["call"] = row["call"]
        v["kind"] = kind
        v["status"] = vision_status(row["call"], v)
        results.append(v)
        if path:
            with path.open("a") as f:
                f.write(json.dumps(v, default=str) + "\n")
        log(f"{kind} {len(results)}/{len(d)} {fid} {v.get('after_ground', v.get('error'))} "
            f"-> {v['status']}")
    return results


# ------------------------------------------------------------------ 3. reasons

def evidence_record(row, verdict=None, timing=None):
    """Everything said about one gap field, as plain values. The reason may use only these.
    timing: interval_window() for this field's bracket, when the county's curves could be built."""
    rec = {
        "field_id": row["field_id"], "acres": _r(row["acres"]),
        "gap_from": row["last_standing_date"], "gap_to": row["drop_date"],
        "gap_days": None if row["bracket_days"] is None else int(row["bracket_days"]),
        "ndvi_before": _r(row["ndvi_before"], 2), "ndvi_after": _r(row["ndvi_after"], 2),
        "model_call": row["call"], "model_p_silage": _r(row["p_silage"], 2),
        "model_abstained": bool(row["abstain"]),
    }
    if timing:
        rec.update(timing)
    if verdict and not verdict.get("error"):
        rec.update({"picture_before": verdict.get("before_canopy"),
                    "picture_after": verdict.get("after_ground"),
                    "picture_confidence": _r(verdict.get("confidence"), 2),
                    "picture_dates": [verdict.get("before_date"), verdict.get("after_date")],
                    "picture_says": verdict.get("reason"),
                    "picture_status": verdict.get("status") or vision_status(row["call"], verdict)})
    return rec


def template_reason(rec):
    """The reason laid out by code, used when there is no key or the model's text fails the check."""
    s = (f"Cloud hid this field from {rec['gap_from']} to {rec['gap_to']} ({rec['gap_days']} days). "
         f"Greenness was {rec['ndvi_before']} before and {rec['ndvi_after']} after. "
         f"The model reads {rec['model_call']} at {rec['model_p_silage']}")
    s += ", but abstained because the harvest window is wide." if rec["model_abstained"] else "."
    if rec.get("p_silage_from_timing") is not None:
        s += (f" {rec['silage_harvest_in_gap_pct']}% of the county's silage harvest and "
              f"{rec['grain_harvest_in_gap_pct']}% of its grain harvest fell in that window, "
              f"which puts silage at {rec['p_silage_from_timing']} on timing alone.")
    if rec.get("picture_status"):
        s += (f" The picture after the gap shows {str(rec['picture_after']).replace('_', ' ')} "
              f"(confidence {rec['picture_confidence']}): {rec['picture_status']}.")
    return s


_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def numbers_ok(text, rec):
    """True when every number in the text is in the record (dates and the field id count)."""
    allowed = set()
    for v in rec.values():
        for item in (v if isinstance(v, list) else [v]):
            allowed.update(_NUM.findall(str(item)))
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                allowed.add(str(item))
                allowed.add(str(int(item)) if float(item).is_integer() else str(item))
    return all(n in allowed or n.lstrip("0") in allowed for n in _NUM.findall(text or ""))


REASON_PROMPT = """You explain one corn field to a USDA analyst. Use only the facts in the record below.
Write two or three short plain sentences: when cloud hid the field, what it looked like before and after,
what share of the county's silage and grain harvest fell in that window and the timing probability if the
record has them, what the picture after the gap shows if there is one, and what that suggests (whole plant
removed while green suggests silage; residue left suggests grain). Say plainly if the picture conflicts with the model
or is unclear. Do not add any number, date or fact that is not in the record. No bullet points.

Record:
"""


def explain(rec, llm=None):
    """Nemotron Ultra's reason for one field, kept only if every number in it is in the record."""
    if llm is None:
        return {"reason": template_reason(rec), "written_by": "template"}
    try:
        text = llm.chat([{"role": "user", "content": REASON_PROMPT + json.dumps(rec, default=str)}])["content"]
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    except Exception as e:
        return {"reason": template_reason(rec), "written_by": "template", "error": str(e)[:200]}
    if text and numbers_ok(text, rec):
        return {"reason": text, "written_by": "nemotron"}
    return {"reason": template_reason(rec), "written_by": "template",
            "rejected": "empty reply" if not text else "a number not in the record"}


# ------------------------------------------------------------------ the supervisor's tool

GAP_FLAG_SHARE = 0.10      # flag a county when abstained gap fields hold at least this share of corn acres
# Scored once on the 44 labelled Beltsville field-years (15 Nov, nothing tuned on them): the picture was
# readable on 28 and matched the reference label on 15 of those, while the model matched 42 of 44.
VISION_TRUST_NOTE = ("Illustration, not evidence: on 44 labelled Beltsville fields the picture was readable "
                     "on 28 and right on 15 of those, about a coin flip at 10 m true colour.")
# Seen across 11 places on 15 Nov (9 state samples of 2025, Frederick and Wyoming NY 2022): the three with
# 20% or more of corn acres not judged read 13 to 20 points low; the other eight missed by 4.3 on average.
# The line was drawn after seeing that table, so it is a warning, not a tested rule.
NOT_JUDGED_WARN = 0.20
MAX_VISION_FIELDS = 12


def _pct(x):
    return None if x is None else round(100 * float(x), 1)


def gap_tool_result(toolbox, fips, year, policy="module_v2", vision_fields=0, out_dir=None,
                    llm=None, chip=None, ask=None):
    """What inspect_gaps returns to the supervisor: a small, flat record with every share in percent.

    Code decides the flag and the direction. The county estimate is not touched."""
    df, s, _ = county_gap_report(toolbox, fips, year, policy)
    sh = s["silage_share"]
    total = s["corn_acres"] or 0
    gap_share = (s["abstained_with_bracket_acres"] or 0) / total if total else 0
    flag = gap_share >= GAP_FLAG_SHARE
    acres = df["acres"].to_numpy(float)
    judged = (~df["abstain"]).to_numpy() & (df["kind"] != "unseen").to_numpy()
    not_judged = float(acres[~judged].sum()) / total if total else 0
    sil_judged = float(acres[judged & (df["call"] == "silage").to_numpy()].sum())
    interval, _ = interval_share(df)
    among, called = s["abstained_with_bracket_called_silage_share"], sh["called"]
    direction = "unknown"
    if among is not None and called is not None:
        direction = "low" if among > called else ("high" if among < called else "unchanged")
    out = {
        "fields": s["fields"], "abstained_fields": s["abstained_fields"],
        "abstained_fields_with_cloud_gap": s["abstained_with_bracket"],
        "abstained_cloud_gap_acres": s["abstained_with_bracket_acres"],
        "abstained_cloud_gap_share_of_corn_pct": _pct(gap_share),
        "unseen_fields": s["kinds"]["unseen"]["fields"],
        "cloud_gap_at_harvest": bool(flag),
        "estimate_without_gap_fields_is_likely": direction if flag else "unchanged",
        "silage_share_called_pct": _pct(called),
        "silage_share_if_gap_fields_counted_pct": _pct(sh["with_gap_fields"]),
        "range_with_gap_fields_low_pct": _pct(sh["lower"]), "range_with_gap_fields_high_pct": _pct(sh["upper"]),
        "not_judged_share_of_corn_pct": _pct(not_judged),
        "reads_low_warning": bool(not_judged >= NOT_JUDGED_WARN),
        "silage_share_if_all_not_judged_were_silage_pct": _pct((sil_judged + not_judged * total) / total) if total else None,
        "silage_share_interval_formula_pct": _pct(interval),
        "main_gap": None,
        "note": ("Extra evidence only: the county estimate is unchanged. Where 20% or more of corn acres were "
                 "not judged, estimates read 13 to 20 points low in 3 of 11 places checked, and the fields not "
                 "judged were almost all silage there; in grain states they were not. Report the range beside "
                 "the estimate, not as the estimate."),
    }
    ab = [g for g in s["main_gaps"] if g["abstained_fields"]]
    if ab:
        g = max(ab, key=lambda x: x["abstained_fields"])
        out["main_gap"] = {"from": g["from"], "to": g["to"], "days": g["days"],
                           "abstained_fields": g["abstained_fields"],
                           "median_ndvi_before": g["median_ndvi_before"],
                           "median_ndvi_after": g["median_ndvi_after"]}
    if out["main_gap"]:
        w = interval_window(df, out["main_gap"]["from"], out["main_gap"]["to"])
        if w:
            out["main_gap"].update(w)
    folder = Path(toolbox._folder(str(fips), int(year)))
    out["landsat_rows_on_disk"] = (folder / "timeseries_landsat.parquet").exists()
    out["landsat_in_use"] = bool(getattr(toolbox, "landsat", False))
    n = max(0, min(int(vision_fields or 0), MAX_VISION_FIELDS))
    if n and flag:
        if chip is None and ask is None and llm is None:
            from ..react.supervisor import VISION_MODEL, NemotronClient
            llm = NemotronClient(model=VISION_MODEL, max_tokens=2048)
        import geopandas as gpd
        gf = gpd.read_file(folder / "fields.geojson").to_crs("EPSG:4326")
        res = run_vision(df[df["abstain"]], gf.set_index("field_id").geometry.to_dict(), "gap", n, 0,
                         out_dir, llm, chip, ask, log=lambda *_: None)
        a = agreement(res)
        out["vision"] = {"fields_shown": a["checked"], "picture_readable": a["readable"],
                         "picture_supports_model_call": a["supported"],
                         "picture_conflicts_with_model_call": a["conflict"],
                         "bare_soil_after_gap": sum(r.get("after_ground") == "bare_soil" for r in res),
                         "residue_after_gap": sum(r.get("after_ground") == "residue" for r in res),
                         "how_far_to_trust_it": VISION_TRUST_NOTE}
    elif n:
        out["vision"] = {"fields_shown": 0, "why": "no cloud gap flagged, so no fields were shown"}
    return out


# ------------------------------------------------------------------ county run

def county_gap_report(toolbox, fips, year, policy="module_v2"):
    """Table and summary for one county from a FieldVision Toolbox (the inner one, not the bridge)."""
    rows, _, _ = toolbox._policy_predictions(str(fips), int(year), policy)
    fields = toolbox._data(str(fips), int(year))[0]
    area = fields.set_index("field_id")["area_ha"].astype(float).to_dict()
    df = gap_table(rows, area)
    return df, gap_summary(df), fields


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cloud-gap report for one county")
    ap.add_argument("fips")
    ap.add_argument("year", type=int)
    ap.add_argument("--data", required=True, help="counties folder: <fips>/<year>/fields.geojson ...")
    ap.add_argument("--registry", default="registry")
    ap.add_argument("--out", default="results/gap")
    ap.add_argument("--landsat", action="store_true")
    ap.add_argument("--vision", type=int, default=0, help="gap fields to show the vision model")
    ap.add_argument("--vision-clear", type=int, default=0, help="clear fields to show it, for agreement")
    ap.add_argument("--reasons", type=int, default=0, help="gap fields to write a reason for")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from fieldvision.toolbox import Toolbox
    tb = Toolbox(data_root=a.data, registry=a.registry, out_root=str(Path(a.out) / "toolbox"),
                 landsat=a.landsat)
    df, summary, fields = county_gap_report(tb, a.fips, a.year)
    out = Path(a.out) / f"{a.fips}_{a.year}" / ("landsat" if a.landsat else "s2")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "gap_fields.csv", index=False)
    result = {"fips": a.fips, "year": a.year, "landsat": a.landsat, "summary": summary}

    if a.vision or a.vision_clear or a.reasons:
        from ..react.supervisor import VISION_MODEL, NemotronClient
        import geopandas as gpd
        gf = gpd.read_file(Path(tb._folder(a.fips, a.year)) / "fields.geojson").to_crs("EPSG:4326")
        geoms = gf.set_index("field_id").geometry.to_dict()
        # 2,048 tokens: the vision model spends about 600 on reasoning before it answers, and at
        # 400 the reply comes back empty (seen 15 Nov), which reads as 'unclear' for every field
        vlm = NemotronClient(model=VISION_MODEL, max_tokens=2048)
        res_gap = run_vision(df, geoms, "gap", a.vision, a.seed, out, vlm) if a.vision else []
        res_clear = run_vision(df, geoms, "clear", a.vision_clear, a.seed, out, vlm) if a.vision_clear else []
        result["vision_gap"] = agreement(res_gap)
        result["vision_clear"] = agreement(res_clear)
        if a.reasons:
            from ..react.supervisor import NemotronClient as NC
            ultra = NC(model="nvidia/nemotron-3-ultra-550b-a55b", max_tokens=1500)
            by_id = df.set_index("field_id", drop=False)
            reasons = []
            for v in res_gap[:a.reasons]:
                row = by_id.loc[v["field_id"]]
                rec = evidence_record(row, v, interval_window(df, row["last_standing_date"], row["drop_date"]))
                reasons.append({**rec, **explain(rec, ultra)})
            (out / "gap_reasons.json").write_text(json.dumps(reasons, indent=1, default=str))
            result["reasons_written"] = len(reasons)
            result["reasons_by_nemotron"] = sum(r["written_by"] == "nemotron" for r in reasons)
    (out / "gap_report.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
