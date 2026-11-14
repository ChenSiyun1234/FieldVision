// Plain-English evidence for one field, built only from numbers the export provides.
// The cue words ("silage-like", "grain-like") come from display thresholds in
// meta.cues, not from the model, so a sentence can disagree with the prediction.

import { isNum, shortDate, daysBetween, esc } from "./format.js";

const DEFAULT_CUES = { green_ndvi: 0.6, dry_ndvi: 0.45, wet_ndmi: 0.2, dry_ndmi: 0.05 };

function cutWindow(f) {
  if (!f.hs || !f.he) return null;
  if (f.hs === f.he) return `on ${shortDate(f.hs)}`;
  return `between ${shortDate(f.hs)} and ${shortDate(f.he)}`;
}

/** Returns a list of HTML strings (already escaped). Empty when there is nothing honest to say. */
export function evidenceSentences(f, cuesIn) {
  const cues = { ...DEFAULT_CUES, ...(cuesIn || {}) };
  const out = [];
  const win = cutWindow(f);

  if (!win) {
    if (f.pred) out.push("No harvest drop seen between clear observations, so cutting date and crop state are unknown.");
    return out;
  }

  if (isNum(f.pre_ndvi)) {
    const v = f.pre_ndvi;
    let state, cue;
    if (v >= cues.green_ndvi) { state = "while still green"; cue = "silage-like"; }
    else if (v < cues.dry_ndvi) { state = "after dry-down"; cue = "grain-like"; }
    else { state = "part-way through dry-down"; cue = "between the two"; }
    out.push(`Cut ${esc(win)} ${state} (NDVI ${v.toFixed(2)}): <b>${cue}</b>.`);
  } else {
    out.push(`Cut ${esc(win)}; no clear view just before the cut, so its greenness is unknown.`);
  }

  if (isNum(f.pre_ndmi)) {
    const m = f.pre_ndmi;
    let state, cue;
    if (m >= cues.wet_ndmi) { state = "still wet"; cue = "silage-like"; }
    else if (m < cues.dry_ndmi) { state = "already dry"; cue = "grain-like"; }
    else { state = "drying"; cue = "in between"; }
    out.push(`Leaves ${state} just before the cut (NDMI ${m.toFixed(2)}): <b>${cue}</b>.`);
  }
  return out;
}

export function lowEvidenceText(f) {
  if (f.low !== true) return null;
  const parts = [];
  if (f.reason) parts.push(f.reason);
  else {
    const w = daysBetween(f.hs, f.he);
    if (!f.hs) parts.push("no harvest drop seen");
    else if (isNum(w)) parts.push(`harvest window ${w} days wide`);
    if (isNum(f.n_usable)) parts.push(`${f.n_usable} clear observations`);
  }
  return parts.join("; ");
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
