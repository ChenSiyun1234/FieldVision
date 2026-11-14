// "Explainable to a skeptic": three pieces of evidence that can be checked against the
// curves, in plain words first (cliff or fade, bare soil or residue) with the index
// names as secondary labels, plus the traps that make a call less certain.
//
// The cue words ("silage-like", "grain-like") come from display thresholds in
// meta.cues and from the region's own harvest dates, not from the model, so a row
// can disagree with the prediction. When the model supplies its own reason or
// counterfactual text, the app shows it next to these rows.

import { isNum, shortDate, daysBetween, dayToLabel, isoToDay, esc } from "./format.js";

const DEFAULT_CUES = {
  green_ndvi: 0.6, dry_ndvi: 0.45, wet_ndmi: 0.2, dry_ndmi: 0.05, residue_ndti: 0.1,
  cliff_drop: 0.35, fade_loss: 0.25, regreen_ndvi: 0.45,
};
const FEW_CLEAR = 8;        // fewer clear looks than this is worth saying out loud
const WIDE_WINDOW = 21;     // harvest window wider than this many days
const TIMING_MARGIN = 4;    // minimum days around the silage/grain midpoint called "in between"

export const cuesOf = (meta) => ({ ...DEFAULT_CUES, ...(meta?.cues || {}) });

function cutWindow(f) {
  if (!f.hs || !f.he) return null;
  return f.hs === f.he ? shortDate(f.hs) : `${shortDate(f.hs)} – ${shortDate(f.he)}`;
}

/** Midpoint between the region's typical silage and grain harvest (day offset), or null. */
export function timingSplit(meta) {
  const r = meta?.harvest_ref;
  if (!r || !isNum(r.silage_mid) || !isNum(r.grain_mid) || r.grain_mid <= r.silage_mid) return null;
  return { split: (r.silage_mid + r.grain_mid) / 2, margin: Math.max(TIMING_MARGIN, 0.2 * (r.grain_mid - r.silage_mid)), ...r };
}

function cutMid(f, year) {
  return f.hs && f.he ? (isoToDay(f.hs, year) + isoToDay(f.he, year)) / 2 : null;
}

/** "silage" if cut clearly before the region's midpoint, "grain" if clearly after, else "mixed" / null. */
function timingCue(f, meta) {
  const t = timingSplit(meta), mid = cutMid(f, meta.year);
  if (!t || mid === null) return null;
  if (Math.abs(mid - t.split) <= t.margin) return "mixed";
  return mid < t.split ? "silage" : "grain";
}

function timingText(f, meta) {
  const t = timingSplit(meta), mid = cutMid(f, meta.year), year = meta.year;
  if (!t || mid === null) return "";
  const rel = (ref, name) => {
    const d = Math.round(mid - ref);
    if (Math.abs(d) <= 2) return `about when the typical ${name} happens here (${dayToLabel(ref, year)})`;
    return `${Math.abs(d)} days ${d < 0 ? "before" : "after"} the typical ${name} here (${dayToLabel(ref, year)})`;
  };
  return ` Cut ${rel(t.silage_mid, "silage cut")}, ${rel(t.grain_mid, "grain harvest")}.`;
}

function dropRow(f, meta) {
  const c = cuesOf(meta);
  const row = { key: "drop", title: "The drop: cliff or fade?", chart: "greenness", value: "—", cue: null };
  if (!f.hs || !f.he) {
    row.sentence = "No drop seen between clear views, so the shape and date of the harvest are unknown.";
    return row;
  }
  const days = daysBetween(f.hs, f.he);
  const drop = isNum(f.pre_ndvi) && isNum(f.post_ndvi) ? f.pre_ndvi - f.post_ndvi : null;
  const loss = isNum(f.pre_ndvi) && isNum(f.peak_ndvi) ? f.peak_ndvi - f.pre_ndvi : null;
  const weeks = isNum(f.peak_day) ? Math.round((isoToDay(f.hs, meta.year) - f.peak_day) / 7) : null;
  row.detail = `${cutWindow(f)}${isNum(drop) ? ` · NDVI ${f.pre_ndvi.toFixed(2)} → ${f.post_ndvi.toFixed(2)}` : ""}`;
  let shape = null;
  if (isNum(loss) && loss >= c.fade_loss) shape = "fade";
  else if (isNum(loss) && isNum(drop) && drop >= c.cliff_drop) shape = "cliff";
  else if (isNum(loss) || isNum(drop)) shape = "between";
  if (shape === "cliff") {
    row.value = "Cliff";
    row.cue = "silage";
    row.sentence = `Fell off a cliff: green one look (NDVI ${f.pre_ndvi.toFixed(2)}), bare the next (${f.post_ndvi.toFixed(2)}), ${days} day${days === 1 ? "" : "s"} apart.`;
  } else if (shape === "fade") {
    row.value = "Fade";
    row.cue = "grain";
    const since = isNum(weeks) && weeks > 0 ? ` over about ${weeks} week${weeks === 1 ? "" : "s"} after its peak on ${dayToLabel(f.peak_day, meta.year)}` : "";
    row.sentence = `Faded out: lost ${loss.toFixed(2)} of its greenness${since}, then the harvest.`;
  } else if (shape === "between") {
    row.value = "In between";
    row.cue = timingCue(f, meta) || "mixed";
    row.sentence = "Neither a clean cliff nor a long fade.";
  } else {
    row.value = cutWindow(f);
    row.cue = timingCue(f, meta);
    row.sentence = `Cut ${cutWindow(f)}; no clear look right before it.`;
  }
  row.sentence += timingText(f, meta);
  return row;
}

function stateRow(f, meta) {
  const c = cuesOf(meta);
  const row = { key: "state", title: "Right before the drop", chart: "greenness · moisture", value: "—", cue: null };
  const v = f.pre_ndvi, m = f.pre_ndmi;
  if (!isNum(v) && !isNum(m)) {
    row.sentence = f.hs ? "No clear look right before the drop." : "Unknown without a detected drop.";
    return row;
  }
  row.detail = [isNum(v) ? `NDVI ${v.toFixed(2)}` : null, isNum(m) ? `NDMI ${m.toFixed(2)}` : null].filter(Boolean).join(" · ");
  let score = 0;
  const words = [];
  if (isNum(v)) {
    if (v >= c.green_ndvi) { score++; words.push("green"); }
    else if (v < c.dry_ndvi) { score--; words.push("browned"); }
    else words.push("partly browned");
  }
  if (isNum(m)) {
    if (m >= c.wet_ndmi) { score++; words.push("moist"); }
    else if (m < c.dry_ndmi) { score--; words.push("dry"); }
    else words.push("drying");
  }
  row.value = words.map((w, i) => (i ? w : w[0].toUpperCase() + w.slice(1))).join(", ");
  row.cue = score > 0 ? "silage" : score < 0 ? "grain" : "mixed";
  const why = row.cue === "silage"
    ? "silage is chopped while the plant is still green and wet"
    : row.cue === "grain"
      ? "grain is left standing for weeks to dry down"
      : "part-way through drying down, so this could be either";
  row.sentence = `${row.value} right before the drop: ${why}.`;
  return row;
}

function groundRow(f, meta) {
  const c = cuesOf(meta);
  const row = { key: "ground", title: "After harvest: bare soil or residue?", chart: "ground residue", value: "—", cue: null };
  const r = f.post_ndti;
  if (!f.he) {
    row.sentence = "Unknown without a detected drop.";
    return row;
  }
  if (!isNum(r)) {
    row.sentence = `No clear look in the ${meta.post_window_days ?? 30} days after harvest, so the ground was not seen.`;
    return row;
  }
  row.detail = `NDTI ${r.toFixed(2)}`;
  if (r >= c.residue_ndti) {
    row.value = "Stalk residue";
    row.cue = "grain";
    row.sentence = "Stalks left on the ground: what a combine leaves.";
  } else {
    row.value = "Bare soil";
    row.cue = "silage";
    row.sentence = "Almost nothing left on the ground: what a chopper leaves.";
  }
  if (isNum(f.regreen_ndvi) && f.regreen_ndvi >= c.regreen_ndvi) row.sentence += " A second green-up followed, so read this with care.";
  return row;
}

/** The three evidence rows, each {key,title,value,detail?,sentence,cue,chart}. */
export function defendRows(f, meta) {
  return [dropRow(f, meta), stateRow(f, meta), groundRow(f, meta)];
}

export function tally(rows) {
  const n = { silage: 0, grain: 0, mixed: 0, unknown: 0 };
  for (const r of rows) n[r.cue || "unknown"]++;
  return n;
}

/**
 * The organisers' traps that apply to this field, in their words:
 * clouds hide the week a field was cut; early does not mean silage; double crops.
 * Each {key, tag, text}; text is plain (escape before inserting).
 */
export function traps(f, meta) {
  const c = cuesOf(meta);
  const out = [];
  if (isNum(f.n_usable) && f.n_usable === 0 && isNum(f.n_obs)) {
    out.push({ key: "clouds", tag: "cloudy season", text: `Clouds hid the whole season: all ${f.n_obs} looks were cloudy or hazy.` });
  } else if (isNum(f.n_usable) && f.n_usable < FEW_CLEAR) {
    out.push({ key: "clouds", tag: "few clear looks", text: `Clouds: only ${f.n_usable} clear look${f.n_usable === 1 ? "" : "s"} all season (of ${f.n_obs}).` });
  }
  const w = daysBetween(f.hs, f.he);
  if (isNum(w) && w > WIDE_WINDOW) {
    out.push({ key: "cloudcut", tag: "cloud gap at harvest", text: `Clouds hid the week it was cut: no clear look for ${w} days (${shortDate(f.hs)} – ${shortDate(f.he)}), so the cliff-or-fade shape is blurred.` });
  }
  if (!f.hs && f.pred && !(isNum(f.n_usable) && f.n_usable === 0)) {
    out.push({ key: "nodrop", tag: "no drop seen", text: "No harvest drop seen between clear looks." });
  }
  if (isNum(f.regreen_ndvi) && f.regreen_ndvi >= c.regreen_ndvi) {
    out.push({ key: "regreen", tag: "second green-up", text: `A second green-up after harvest (greenness back to ${f.regreen_ndvi.toFixed(2)}${isNum(f.regreen_day) ? ` by ${dayToLabel(f.regreen_day, meta.year)}` : ""}): a double crop or cover crop, which can hide the bare-soil-or-residue signal.` });
  }
  if (timingCue(f, meta) === "silage" && isNum(f.pre_ndvi) && f.pre_ndvi < c.dry_ndvi) {
    out.push({ key: "early", tag: "early ≠ silage", text: "Early does not mean silage: this field was cut early but had already browned, the way grain is combined in summer in hot regions." });
  }
  return out;
}

/**
 * What would have made it the other call, from the same display thresholds.
 * Returns {other, already: [..], needed: [..]} with HTML-escaped strings.
 */
export function flipConditions(f, meta) {
  if (!f.pred) return null;
  const c = cuesOf(meta);
  const other = f.pred === "silage" ? "grain" : "silage";
  const toGrain = other === "grain";
  const items = [];
  const t = timingSplit(meta), mid = cutMid(f, meta.year);
  if (t && mid !== null) {
    const ok = toGrain ? mid > t.split : mid < t.split;
    items.push({ ok, dist: Math.abs(mid - t.split) / 30,
      text: `a cut ${toGrain ? "after" : "before"} about ${dayToLabel(Math.round(t.split), meta.year)} (this field: ${dayToLabel(Math.round(mid), meta.year)})` });
  }
  if (isNum(f.pre_ndvi)) {
    const th = toGrain ? c.dry_ndvi : c.green_ndvi;
    const ok = toGrain ? f.pre_ndvi < th : f.pre_ndvi >= th;
    items.push({ ok, dist: Math.abs(f.pre_ndvi - th) / 0.5,
      text: `${toGrain ? "a browned crop" : "a still-green crop"} right before the drop (NDVI ${toGrain ? "below" : "at or above"} ${th.toFixed(2)}; this field: ${f.pre_ndvi.toFixed(2)})` });
  }
  if (isNum(f.pre_ndmi)) {
    const th = toGrain ? c.dry_ndmi : c.wet_ndmi;
    const ok = toGrain ? f.pre_ndmi < th : f.pre_ndmi >= th;
    items.push({ ok, dist: Math.abs(f.pre_ndmi - th) / 0.4,
      text: `${toGrain ? "dry" : "moist"} leaves right before the drop (NDMI ${toGrain ? "below" : "at or above"} ${th.toFixed(2)}; this field: ${f.pre_ndmi.toFixed(2)})` });
  }
  if (isNum(f.post_ndti)) {
    const th = c.residue_ndti;
    const ok = toGrain ? f.post_ndti >= th : f.post_ndti < th;
    items.push({ ok, dist: Math.abs(f.post_ndti - th) / 0.2,
      text: `${toGrain ? "stalk residue" : "bare soil"} after harvest (NDTI ${toGrain ? "at or above" : "below"} ${th.toFixed(2)}; this field: ${f.post_ndti.toFixed(2)})` });
  }
  const needed = items.filter((i) => !i.ok).sort((a, b) => a.dist - b.dist);
  return {
    other,
    already: items.filter((i) => i.ok).map((i) => esc(i.text)),
    needed: needed.map((i) => esc(i.text)),
  };
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

/**
 * Earliest cutoff from which the call matches the full-season call and stays confident.
 * Returns {iso} (null iso = only with the full season), or null without as-of data.
 */
export function settledFrom(fieldId, asof, full, threshold = 0.5) {
  if (!asof || !full?.pred) return null;
  const a = asof.fields?.[fieldId];
  if (!a) return null;
  let from = null;
  for (let i = asof.cutoffs.length - 1; i >= 0; i--) {
    const p = a.p[i];
    const pred = isNum(p) ? (p >= threshold ? "silage" : "grain") : null;
    if (pred === full.pred && a.low[i] !== true) from = i;
    else break;
  }
  return { iso: from === null ? null : asof.cutoffs[from] };
}

/** First cutoff from which region accuracy stays within `tol` of the full season (or confident share for unlabelled regions). */
export function regionSettled(asof, full, hasLabels, tol = 0.03) {
  if (!asof?.cutoffs?.length || !full) return null;
  const val = (m) => (hasLabels ? m?.all?.accuracy : isNum(m?.low_evidence_share) ? 1 - m.low_evidence_share : null);
  const target = val(full);
  if (!isNum(target)) return null;
  let from = null;
  for (let i = asof.cutoffs.length - 1; i >= 0; i--) {
    const v = val(asof.metrics[i]);
    if (isNum(v) && v >= target - tol) from = i;
    else break;
  }
  return from === null ? null : { iso: asof.cutoffs[from], value: val(asof.metrics[from]), full: target };
}
