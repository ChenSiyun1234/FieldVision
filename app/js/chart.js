// Season curve as plain SVG. Every acquisition is drawn:
//   filled dot   usable (clear_fraction >= meta.min_clear)
//   hollow dot   excluded (some clear pixels, below the threshold)
//   tick at foot no clear pixel at all
// Lines join usable observations only, and never across gaps longer than gapDays.

import { isNum, monthTicks, dayToLabel, esc } from "./format.js";

const Q_NONE = 0, Q_EXCLUDED = 1, Q_USABLE = 2;

export function seasonChart(opts) {
  const {
    width = 420, height = 170, year, domain, series, key, label, color,
    yMin, yMax, yTicks, band, pre, gapDays = 20, bandColor = "#eef2f6",
  } = opts;
  const m = { l: 38, r: 12, t: 24, b: 24 };
  const iw = width - m.l - m.r, ih = height - m.t - m.b;
  const [t0, t1] = domain;
  const x = (t) => m.l + ((t - t0) / (t1 - t0)) * iw;
  const clampY = (v) => Math.max(yMin, Math.min(yMax, v));
  const y = (v) => m.t + (1 - (clampY(v) - yMin) / (yMax - yMin)) * ih;
  const out = [];

  out.push(`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(label)} season curve">`);
  out.push(`<text x="${m.l}" y="14" fill="${color}" font-size="13" font-weight="700">${esc(label)}</text>`);

  // grid
  for (const v of yTicks) {
    out.push(`<line x1="${m.l}" x2="${width - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="#2a3541" stroke-width="1"/>`);
    out.push(`<text x="${m.l - 6}" y="${y(v) + 4}" fill="#9fb0c0" font-size="11" text-anchor="end">${v.toFixed(1)}</text>`);
  }
  for (const tk of monthTicks(t0, t1, year)) {
    out.push(`<line x1="${x(tk.t)}" x2="${x(tk.t)}" y1="${m.t}" y2="${m.t + ih}" stroke="#222c36" stroke-width="1"/>`);
    out.push(`<text x="${x(tk.t) + 3}" y="${height - 8}" fill="#9fb0c0" font-size="11">${tk.label}</text>`);
  }

  // harvest interval band
  if (band && isNum(band[0]) && isNum(band[1])) {
    const a = x(Math.max(t0, band[0])), b = x(Math.min(t1, band[1]));
    const w = Math.max(3, b - a);
    out.push(`<rect x="${a}" y="${m.t}" width="${w}" height="${ih}" fill="${bandColor}" fill-opacity="0.16" stroke="${bandColor}" stroke-opacity="0.55" stroke-dasharray="3 3"/>`);
    out.push(`<text x="${a + w / 2}" y="${m.t - 4}" fill="${bandColor}" font-size="11" font-weight="600" text-anchor="middle">harvest</text>`);
  }

  const n = series ? series.t.length : 0;
  const vals = series ? series[key] : [];
  const usable = [];
  for (let i = 0; i < n; i++) if (series.q[i] === Q_USABLE && isNum(vals[i])) usable.push(i);

  // lines between consecutive usable observations; long gaps stay open
  let path = "";
  for (let k = 0; k < usable.length; k++) {
    const i = usable[k];
    const joined = k > 0 && series.t[i] - series.t[usable[k - 1]] <= gapDays;
    path += `${joined ? "L" : "M"}${x(series.t[i]).toFixed(1)},${y(vals[i]).toFixed(1)}`;
  }
  if (path) out.push(`<path d="${path}" fill="none" stroke="${color}" stroke-width="2.2" stroke-linejoin="round"/>`);

  // label the long gaps so nobody reads them as missing ink
  for (let k = 1; k < usable.length; k++) {
    const ta = series.t[usable[k - 1]], tb = series.t[usable[k]];
    if (tb - ta > gapDays && x(tb) - x(ta) > 46) {
      const xm = (x(ta) + x(tb)) / 2;
      out.push(`<text x="${xm}" y="${m.t + ih - 6}" fill="#6c7c8c" font-size="10" text-anchor="middle">${tb - ta} d no clear view</text>`);
    }
  }

  // observations
  for (let i = 0; i < n; i++) {
    const t = series.t[i], v = vals[i], q = series.q[i];
    const cf = isNum(series.cf[i]) ? `${Math.round(series.cf[i] * 100)}% clear` : "no clear pixels";
    if (q === Q_NONE || !isNum(v)) {
      out.push(`<line x1="${x(t)}" x2="${x(t)}" y1="${m.t + ih}" y2="${m.t + ih + 5}" stroke="#6c7c8c" stroke-width="1.5"><title>${dayToLabel(t, year)}: ${cf}</title></line>`);
    } else if (q === Q_EXCLUDED) {
      out.push(`<circle cx="${x(t)}" cy="${y(v)}" r="3.2" fill="#151c23" stroke="#9fb0c0" stroke-width="1.2"><title>${dayToLabel(t, year)}: ${label} ${v.toFixed(2)} (${cf}, excluded)</title></circle>`);
    } else {
      out.push(`<circle cx="${x(t)}" cy="${y(v)}" r="3.6" fill="${color}"><title>${dayToLabel(t, year)}: ${label} ${v.toFixed(2)} (${cf})</title></circle>`);
    }
  }

  // value just before harvest, as reported by the model
  if (pre && isNum(pre.t) && isNum(pre.v)) {
    const px = x(pre.t), py = y(pre.v);
    const anchor = px > m.l + iw * 0.75 ? "end" : "start";
    const dx = anchor === "end" ? -10 : 10;
    out.push(`<circle cx="${px}" cy="${py}" r="7.5" fill="none" stroke="#eef2f6" stroke-width="2"/>`);
    out.push(`<text x="${px + dx}" y="${py - 9}" fill="#eef2f6" font-size="13" font-weight="700" text-anchor="${anchor}" paint-order="stroke" stroke="#151c23" stroke-width="4">${pre.v.toFixed(2)}</text>`);
  }

  out.push(`</svg>`);
  return out.join("");
}

export function chartLegendHtml(color, minClear) {
  const pctClear = isNum(minClear) ? Math.round(minClear * 100) : null;
  return `<div class="chart-legend">
    <span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="${color}"/></svg>usable${pctClear !== null ? ` (≥${pctClear}% clear)` : ""}</span>
    <span><svg width="12" height="12"><circle cx="6" cy="6" r="3.5" fill="none" stroke="#9fb0c0" stroke-width="1.3"/></svg>excluded (cloud)</span>
    <span><svg width="10" height="12"><line x1="5" x2="5" y1="3" y2="11" stroke="#6c7c8c" stroke-width="1.5"/></svg>no clear pixels</span>
    <span><svg width="16" height="12"><rect x="1" y="1" width="14" height="10" fill="#eef2f6" fill-opacity="0.16" stroke="#eef2f6" stroke-opacity="0.55" stroke-dasharray="2 2"/></svg>harvest window</span>
  </div>`;
}
