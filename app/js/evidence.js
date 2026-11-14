// "Defend this call": three pieces of evidence a skeptic can check against the curves,
// built only from numbers the export provides.
//
// The cue words ("silage-like", "grain-like") come from display thresholds in
// meta.cues and from the region's own harvest dates, not from the model, so a row
// can disagree with the prediction. When the model supplies its own reason or
// counterfactual text, the app shows it next to these rows.

import { isNum, shortDate, daysBetween, dayToLabel, isoToDay, esc } from "./format.js";

const DEFAULT_CUES = { green_ndvi: 0.6, dry_ndvi: 0.45, wet_ndmi: 0.2, dry_ndmi: 0.05, residue_ndti: 0.1 };
const FEW_CLEAR = 8;        // fewer clear dates than this is worth saying out loud
const WIDE_WINDOW = 21;     // harvest window wider than this many days
const TIMING_MARGIN = 4;    // days around the silage/grain midpoint called "in between"

export const cuesOf = (meta) => ({ ...DEFAULT_CUES, ...(meta?.cues || {}) });

function cutWindow(f) {
  if (!f.hs || !f.he) return null;
  return f.hs === f.he ? shortDate(f.hs) : `${shortDate(f.hs)} – ${shortDate(f.he)}`;
}

/** Midpoint between the region's typical silage and grain harvest (day offset), or null. */
export function timingSplit(meta) {
  const r = meta?.harvest_ref;
  if (!r || !isNum(r.silage_mid) || !isNum(r.grain_mid) || r.grain_mid <= r.silage_mid) return null;
  return { split: (r.silage_mid + r.grain_mid) / 2, ...r };
}

function timingRow(f, meta) {
  const year = meta.year;
  const row = { key: "timing", title: "When it was cut", chart: "NDVI", value: cutWindow(f) || "—", cue: null };
  if (!f.hs || !f.he) {
    row.sentence = "No harvest drop seen between clear views, so the cutting date is unknown.";
    return row;
  }
  const mid = (isoToDay(f.hs, year) + isoToDay(f.he, year)) / 2;
  const drop = isNum(f.pre_ndvi) && isNum(f.post_ndvi) ? f.pre_ndvi - f.post_ndvi : null;
  const days = daysBetween(f.hs, f.he);
  row.detail = isNum(drop) ? `NDVI ${f.pre_ndvi.toFixed(2)} → ${f.post_ndvi.toFixed(2)}` : null;
  const parts = [];
  if (isNum(drop)) parts.push(`NDVI fell by ${drop.toFixed(2)} within ${days} day${days === 1 ? "" : "s"}`);
  const t = timingSplit(meta);
  if (t) {
    const margin = Math.max(TIMING_MARGIN, 0.2 * (t.grain_mid - t.silage_mid));
    row.cue = Math.abs(mid - t.split) <= margin ? "mixed" : mid < t.split ? "silage" : "grain";
    const rel = (ref, name) => {
      const d = Math.round(mid - ref);
      if (Math.abs(d) <= 2) return `about when ${name} happens here (${dayToLabel(ref, year)})`;
      return `${Math.abs(d)} days ${d < 0 ? "before" : "after"} the typical ${name} here (${dayToLabel(ref, year)})`;
    };
    parts.push(`${rel(t.silage_mid, "silage cut")} and ${rel(t.grain_mid, "grain harvest")}`);
  }
  row.sentence = parts.length ? `${cap1(parts.join("; cut "))}.` : `Cut ${cutWindow(f)}.`;
  return row;
}

function stateRow(f, meta) {
  const c = cuesOf(meta);
  const row = { key: "state", title: "Crop just before the cut", chart: "NDVI · NDMI", cue: null };
  const v = f.pre_ndvi, m = f.pre_ndmi;
  if (!isNum(v) && !isNum(m)) {
    row.value = "—";
    row.sentence = f.hs ? "No clear view just before the cut." : "Unknown without a detected cut.";
    return row;
  }
  row.value = [isNum(v) ? `NDVI ${v.toFixed(2)}` : null, isNum(m) ? `NDMI ${m.toFixed(2)}` : null].filter(Boolean).join(" · ");
  let score = 0;
  const words = [];
  if (isNum(v)) {
    if (v >= c.green_ndvi) { score++; words.push("still green"); }
    else if (v < c.dry_ndvi) { score--; words.push("dried down"); }
    else words.push("partly dried");
  }
  if (isNum(m)) {
    if (m >= c.wet_ndmi) { score++; words.push("leaves wet"); }
    else if (m < c.dry_ndmi) { score--; words.push("leaves dry"); }
    else words.push("leaves drying");
  }
  row.cue = score > 0 ? "silage" : score < 0 ? "grain" : "mixed";
  let trend = "";
  if (isNum(v) && isNum(f.peak_ndvi) && isNum(f.peak_day)) {
    const loss = f.peak_ndvi - v;
    trend = loss < 0.25
      ? `; it stayed close to its ${f.peak_ndvi.toFixed(2)} peak until the cut`
      : `; a slow dry-down from its ${f.peak_ndvi.toFixed(2)} peak on ${dayToLabel(f.peak_day, meta.year)}`;
  }
  row.sentence = `${cap1(words.join(", "))}${trend}.`;
  return row;
}

function groundRow(f, meta) {
  const c = cuesOf(meta);
  const row = { key: "ground", title: "Ground after the cut", chart: "NDTI", cue: null };
  const r = f.post_ndti;
  if (!f.he) {
    row.value = "—";
    row.sentence = "Unknown without a detected cut.";
    return row;
  }
  if (!isNum(r)) {
    row.value = "—";
    row.sentence = `No clear view in the ${meta.post_window_days ?? 30} days after the cut, so the ground was not seen.`;
    return row;
  }
  row.value = `NDTI ${r.toFixed(2)}`;
  if (r >= c.residue_ndti) {
    row.cue = "grain";
    row.sentence = `Stalks left on the ground (residue index above ${c.residue_ndti.toFixed(2)}): what a combine leaves.`;
  } else {
    row.cue = "silage";
    row.sentence = `Short stubble and bare soil (residue index below ${c.residue_ndti.toFixed(2)}): what a chopper leaves.`;
  }
  return row;
}

const cap1 = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

/** The three evidence rows, each {key,title,value,detail?,sentence,cue,chart}. */
export function defendRows(f, meta) {
  return [timingRow(f, meta), stateRow(f, meta), groundRow(f, meta)];
}

export function tally(rows) {
  const n = { silage: 0, grain: 0, mixed: 0, unknown: 0 };
  for (const r of rows) n[r.cue || "unknown"]++;
  return n;
}

/**
 * What would have made it the other call, from the same display thresholds.
 * Returns {other, already: [..], needed: [..], closest} with HTML-escaped strings.
 */
export function flipConditions(f, meta) {
  if (!f.pred) return null;
  const c = cuesOf(meta);
  const other = f.pred === "silage" ? "grain" : "silage";
  const toGrain = other === "grain";
  const items = [];
  const t = timingSplit(meta);
  if (t && f.hs && f.he) {
    const mid = (isoToDay(f.hs, meta.year) + isoToDay(f.he, meta.year)) / 2;
    const ok = toGrain ? mid > t.split : mid < t.split;
    items.push({ ok, dist: Math.abs(mid - t.split) / 30,
      text: `cut ${toGrain ? "after" : "before"} about ${dayToLabel(Math.round(t.split), meta.year)} (this field: ${dayToLabel(Math.round(mid), meta.year)})` });
  }
  if (isNum(f.pre_ndvi)) {
    const th = toGrain ? c.dry_ndvi : c.green_ndvi;
    const ok = toGrain ? f.pre_ndvi < th : f.pre_ndvi >= th;
    items.push({ ok, dist: Math.abs(f.pre_ndvi - th) / 0.5,
      text: `NDVI ${toGrain ? "below" : "at or above"} ${th.toFixed(2)} just before the cut (this field: ${f.pre_ndvi.toFixed(2)})` });
  }
  if (isNum(f.pre_ndmi)) {
    const th = toGrain ? c.dry_ndmi : c.wet_ndmi;
    const ok = toGrain ? f.pre_ndmi < th : f.pre_ndmi >= th;
    items.push({ ok, dist: Math.abs(f.pre_ndmi - th) / 0.4,
      text: `NDMI ${toGrain ? "below" : "at or above"} ${th.toFixed(2)} just before the cut (this field: ${f.pre_ndmi.toFixed(2)})` });
  }
  if (isNum(f.post_ndti)) {
    const th = c.residue_ndti;
    const ok = toGrain ? f.post_ndti >= th : f.post_ndti < th;
    items.push({ ok, dist: Math.abs(f.post_ndti - th) / 0.2,
      text: `residue index ${toGrain ? "at or above" : "below"} ${th.toFixed(2)} after the cut (this field: ${f.post_ndti.toFixed(2)})` });
  }
  if (!items.length) return { other, already: [], needed: [], closest: null };
  const needed = items.filter((i) => !i.ok).sort((a, b) => a.dist - b.dist);
  return {
    other,
    already: items.filter((i) => i.ok).map((i) => esc(i.text)),
    needed: needed.map((i) => esc(i.text)),
    closest: needed.length ? esc(needed[0].text) : null,
  };
}

/** Plain-language reasons why a call is low evidence, from the field's own numbers. */
export function lowEvidenceLines(f) {
  const out = [];
  if (isNum(f.n_usable) && f.n_usable < FEW_CLEAR) {
    out.push(f.n_usable === 0
      ? `No clear date all season: all ${f.n_obs} acquisitions were cloudy or hazy.`
      : `Only ${f.n_usable} clear date${f.n_usable === 1 ? "" : "s"} all season (of ${f.n_obs} acquisitions; the rest were cloudy).`);
  }
  const w = daysBetween(f.hs, f.he);
  if (isNum(w) && w > WIDE_WINDOW) {
    out.push(`Clouds hid the field for ${w} days around harvest (${shortDate(f.hs)} – ${shortDate(f.he)}), so the cutting date and the crop's state at the cut are blurred.`);
  }
  if (!f.hs && f.pred && !(isNum(f.n_usable) && f.n_usable === 0)) {
    out.push("No harvest drop was seen between clear views.");
  }
  return out;
}

/** 0 = coin flip, 1 = fully certain. Null without a probability. */
export function certainty(f) {
  return isNum(f.p) ? Math.abs(f.p - 0.5) * 2 : null;
}

/** Probability of the predicted class, e.g. 0.87 for p_silage 0.13 predicted grain. */
export function confidence(f) {
  if (!isNum(f.p) || !f.pred) return null;
  return f.pred === "silage" ? f.p : 1 - f.p;
}
