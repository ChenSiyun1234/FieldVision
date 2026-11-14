// Totals check: our predicted silage and grain acres per county and state next to
// USDA's reported acres, plus the silage share comparison as a scatter.

import { isNum, pct, int, acres, relGap, signedPct, esc, DASH } from "./format.js";

function gapWords(ours, usda) {
  const g = relGap(ours, usda);
  if (!isNum(g)) return "no comparison possible";
  const p = Math.round(Math.abs(g) * 100);
  if (p === 0) return "the same as USDA";
  return `${p}% ${g > 0 ? "more" : "less"} than USDA`;
}

/** One card per state with the gap stated in words. */
export function stateCardsHtml(states) {
  if (!states?.length) return "";
  return states.map((s) => {
    const usdaCorn = isNum(s.usda_silage_acres) && isNum(s.usda_grain_acres) ? s.usda_silage_acres + s.usda_grain_acres : null;
    const coverage = isNum(s.mapped_acres) && isNum(usdaCorn) && usdaCorn > 0 ? s.mapped_acres / usdaCorn : null;
    const share = (a, b) => (isNum(a) && isNum(b) && a + b > 0 ? a / (a + b) : null);
    return `<div class="state-card">
      <h4>${esc(s.state)} · ${int(s.n_counties)} counties</h4>
      <div class="tot-grid">
        <div></div><div class="h">We map</div><div class="h">USDA</div><div class="h">Gap</div>
        <div class="k"><span class="sw" style="background:var(--silage)"></span>Silage</div>
        <div class="n">${acres(s.pred_silage_acres)}</div><div class="n">${acres(s.usda_silage_acres)}</div>
        <div class="n gap">${signedPct(relGap(s.pred_silage_acres, s.usda_silage_acres))}</div>
        <div class="k"><span class="sw" style="background:var(--grain)"></span>Grain</div>
        <div class="n">${acres(s.pred_grain_acres)}</div><div class="n">${acres(s.usda_grain_acres)}</div>
        <div class="n gap">${signedPct(relGap(s.pred_grain_acres, s.usda_grain_acres))}</div>
      </div>
      <p class="plain">We put silage at <b>${gapWords(s.pred_silage_acres, s.usda_silage_acres)}</b> and grain at <b>${gapWords(s.pred_grain_acres, s.usda_grain_acres)}</b>.
        Silage share: ours ${pct(share(s.pred_silage_acres, s.pred_grain_acres))}, USDA ${pct(share(s.usda_silage_acres, s.usda_grain_acres))}.</p>
      <p class="caveat">${isNum(coverage) ? `Our fields cover ${pct(coverage)} of the corn acres USDA reports, so part of any shortfall is fields we did not map. ` : ""}${isNum(s.pred_low_acres) && s.pred_low_acres > 0 ? `${acres(s.pred_low_acres)} of our acres are low-evidence calls.` : ""}</p>
    </div>`;
  }).join("");
}

export function countyTableHtml(rows, highlight) {
  const sorted = [...rows].sort((a, b) => (b.usda_silage_acres ?? -1) - (a.usda_silage_acres ?? -1));
  const cell = (ours, usda) => `<td class="n"><b>${isNum(ours) ? Math.round(ours).toLocaleString("en-US") : DASH}</b><br><small>USDA ${isNum(usda) ? Math.round(usda).toLocaleString("en-US") : DASH}</small></td>`;
  const body = sorted.map((r) => `<tr data-fips="${esc(r.fips)}" class="${r.fips === highlight ? "hl" : ""}">
      <td>${esc(r.name)}<br><small>${int(r.n_predicted)} fields</small></td>
      ${cell(r.pred_silage_acres, r.usda_silage_acres)}
      ${cell(r.pred_grain_acres, r.usda_grain_acres)}
      <td class="n">${signedPct(relGap(r.pred_silage_acres, r.usda_silage_acres))}<br><small>${signedPct(relGap(r.pred_grain_acres, r.usda_grain_acres))}</small></td>
    </tr>`).join("");
  return `<table class="counties">
    <thead><tr><th>County</th><th class="n">Silage ac</th><th class="n">Grain ac</th><th class="n" title="Our acres against USDA's, silage then grain">Gap</th></tr></thead>
    <tbody>${body}</tbody></table>`;
}

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
  out.push(`<text x="${m.l + iw / 2}" y="${height - 6}" fill="#eef2f6" font-size="12" text-anchor="middle">USDA silage share</text>`);
  out.push(`<text transform="translate(12 ${m.t + ih / 2}) rotate(-90)" fill="#eef2f6" font-size="12" text-anchor="middle">Our silage share</text>`);
  for (const r of pts) {
    const rad = 4 + Math.sqrt(r.n_predicted || 1) * 0.9;
    const hl = r.fips === highlight;
    out.push(`<circle data-fips="${esc(r.fips)}" cx="${x(r.usda_silage_share)}" cy="${y(r.pred_silage_share)}" r="${rad}" fill="#56b4e9" fill-opacity="${hl ? 1 : 0.75}" stroke="${hl ? "#ffffff" : "#0e1318"}" stroke-width="${hl ? 3 : 1}" style="cursor:pointer"><title>${esc(r.name)}: ours ${pct(r.pred_silage_share)}, USDA ${pct(r.usda_silage_share)} (${r.n_predicted} fields)</title></circle>`);
  }
  out.push(`</svg>`);
  return out.join("");
}
