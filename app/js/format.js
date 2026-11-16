// Formatting helpers. Every function returns an em dash for missing values
// instead of inventing a number.

export const DASH = "—";
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAY = 86400000;

export const isNum = (v) => typeof v === "number" && Number.isFinite(v);

export function pct(v, { digits } = {}) {
  if (!isNum(v)) return DASH;
  const p = v * 100;
  const d = digits ?? (p > 0 && p < 10 ? 1 : 0);
  return `${p.toFixed(d)}%`;
}

export const fixed = (v, d = 2) => (isNum(v) ? v.toFixed(d) : DASH);
export const int = (v) => (isNum(v) ? Math.round(v).toLocaleString("en-US") : DASH);
export const ha = (v) => (isNum(v) ? `${v < 10 ? v.toFixed(1) : Math.round(v)} ha` : DASH);

/** "2024-09-09" -> {y, m, d} without going through local time zones. */
export function parseIso(s) {
  if (typeof s !== "string") return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(s);
  return m ? { y: +m[1], m: +m[2] - 1, d: +m[3] } : null;
}

export function shortDate(s) {
  const p = parseIso(s);
  return p ? `${MONTHS[p.m]} ${p.d}` : DASH;
}

/** Day offset from Jan 1 of `year` (0 = Jan 1), as stored in series.json. */
export function isoToDay(s, year) {
  const p = parseIso(s);
  if (!p) return null;
  return Math.round((Date.UTC(p.y, p.m, p.d) - Date.UTC(year, 0, 1)) / DAY);
}

export function dayToLabel(t, year) {
  const dt = new Date(Date.UTC(year, 0, 1) + t * DAY);
  return `${MONTHS[dt.getUTCMonth()]} ${dt.getUTCDate()}`;
}

/** Month starts (as day offsets) between t0 and t1. */
export function monthTicks(t0, t1, year) {
  const out = [];
  for (let m = 0; m < 24; m++) {
    const t = Math.round((Date.UTC(year, m, 1) - Date.UTC(year, 0, 1)) / DAY);
    if (t > t1) break;
    if (t >= t0) out.push({ t, label: MONTHS[m % 12] });
  }
  return out;
}

export function daysBetween(a, b) {
  const pa = parseIso(a), pb = parseIso(b);
  if (!pa || !pb) return null;
  return Math.round((Date.UTC(pb.y, pb.m, pb.d) - Date.UTC(pa.y, pa.m, pa.d)) / DAY);
}

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

export const acres = (v) => (isNum(v) ? `${Math.round(v).toLocaleString("en-US")} ac` : DASH);

/** Relative gap of ours against a reference, e.g. +0.13 for 13 % more. Null when undefined. */
export function relGap(ours, ref) {
  return isNum(ours) && isNum(ref) && ref > 0 ? (ours - ref) / ref : null;
}

export function signedPct(v) {
  if (!isNum(v)) return DASH;
  const p = Math.round(v * 100);
  return `${p > 0 ? "+" : p < 0 ? "\u2212" : ""}${Math.abs(p)}%`;
}
