// County comparison: predicted silage share (area weighted, from field predictions)
// against the USDA-reported share of corn acres harvested for silage.

import { isNum, pct, int, esc, DASH } from "./format.js";

export function scatterSvg(rows, { width = 400, height = 300, highlight = null } = {}) {
  const m = { l: 46, r: 14, t: 14, b: 40 };
  const iw = width - m.l - m.r, ih = height - m.t - m.b;
  const pts = rows.filter((r) => isNum(r.pred_silage_share) && isNum(r.usda_silage_share));
  const max = Math.min(1, Math.max(0.2, ...pts.flatMap((r) => [r.pred_silage_share, r.usda_silage_share])) * 1.1);
  const x = (v) => m.l + (v / max) * iw;
  const y = (v) => m.t + (1 - v / max) * ih;
  const step = max > 0.6 ? 0.2 : 0.1;
  const out = [`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Predicted against USDA silage share per county">`];
  for (let v = 0; v <= max + 1e-9; v += step) {
    out.push(`<line x1="${m.l}" x2="${m.l + iw}" y1="${y(v)}" y2="${y(v)}" stroke="#2a3541"/>`);
    out.push(`<line y1="${m.t}" y2="${m.t + ih}" x1="${x(v)}" x2="${x(v)}" stroke="#222c36"/>`);
    out.push(`<text x="${m.l - 6}" y="${y(v) + 4}" fill="#9fb0c0" font-size="11" text-anchor="end">${Math.round(v * 100)}%</text>`);
    out.push(`<text x="${x(v)}" y="${m.t + ih + 15}" fill="#9fb0c0" font-size="11" text-anchor="middle">${Math.round(v * 100)}%</text>`);
  }
  out.push(`<line x1="${x(0)}" y1="${y(0)}" x2="${x(max)}" y2="${y(max)}" stroke="#6c7c8c" stroke-dasharray="5 4"/>`);
  out.push(`<text x="${x(max) - 4}" y="${y(max) + 14}" fill="#6c7c8c" font-size="11" text-anchor="end">perfect agreement</text>`);
  out.push(`<text x="${m.l + iw / 2}" y="${height - 6}" fill="#eef2f6" font-size="12" text-anchor="middle">USDA reported silage share</text>`);
  out.push(`<text transform="translate(12 ${m.t + ih / 2}) rotate(-90)" fill="#eef2f6" font-size="12" text-anchor="middle">Predicted silage share</text>`);
  for (const r of pts) {
    const rad = 4 + Math.sqrt(r.n_predicted || 1) * 0.9;
    const hl = r.fips === highlight;
    out.push(`<circle data-fips="${esc(r.fips)}" cx="${x(r.usda_silage_share)}" cy="${y(r.pred_silage_share)}" r="${rad}" fill="#56b4e9" fill-opacity="${hl ? 1 : 0.75}" stroke="${hl ? "#ffffff" : "#0e1318"}" stroke-width="${hl ? 3 : 1}" style="cursor:pointer"><title>${esc(r.name)}: predicted ${pct(r.pred_silage_share)}, USDA ${pct(r.usda_silage_share)} (${r.n_predicted} fields)</title></circle>`);
  }
  out.push(`</svg>`);
  return out.join("");
}

export function countyTableHtml(rows, highlight) {
  const sorted = [...rows].sort((a, b) => (b.usda_silage_share ?? -1) - (a.usda_silage_share ?? -1));
  const body = sorted.map((r) => {
    const diff = isNum(r.pred_silage_share) && isNum(r.usda_silage_share) ? (r.pred_silage_share - r.usda_silage_share) * 100 : null;
    const d = diff === null ? DASH : `${diff > 0 ? "+" : diff < 0 ? "−" : ""}${Math.abs(diff).toFixed(0)}`;
    return `<tr data-fips="${esc(r.fips)}" class="${r.fips === highlight ? "hl" : ""}">
      <td>${esc(r.name)}, ${esc(r.state)}</td>
      <td class="n">${int(r.n_predicted)}</td>
      <td class="n">${pct(r.pred_silage_share)}</td>
      <td class="n">${pct(r.usda_silage_share)}</td>
      <td class="n">${d}</td>
    </tr>`;
  }).join("");
  return `<table class="counties">
    <thead><tr><th>County</th><th class="n">Fields</th><th class="n">Predicted</th><th class="n">USDA</th><th class="n" title="percentage points">Δ pts</th></tr></thead>
    <tbody>${body}</tbody></table>`;
}
