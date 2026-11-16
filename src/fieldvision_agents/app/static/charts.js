// Two chart forms, plain SVG: a line over years (crosshair + tooltip) and grouped bars (per-bar tooltip).
// One y-axis per chart, recessive grid, 2px lines, rounded bar ends, a legend for two or more series.
import { el, hideTip, showTip } from "./ui.js";

const NS = "http://www.w3.org/2000/svg";
const s = (tag, attrs = {}) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); return n; };

function niceTicks(min, max, n = 4) {
  const span = max - min || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((x) => span / x <= n) || mag * 10;
  const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step, out = [];
  for (let v = lo; v <= hi + step / 2; v += step) out.push(+v.toFixed(10));
  return out;
}

/** points: [{x: year, y, forecast?}], one series. */
export function lineChart({ title, points, yFormat, color = "var(--s1)", note }) {
  const W = 520, H = 220, m = { l: 52, r: 16, t: 10, b: 26 };
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const ticks = niceTicks(Math.min(0, ...ys) === 0 ? Math.min(...ys) * 0.9 : Math.min(...ys), Math.max(...ys));
  const y0 = ticks[0], y1 = ticks[ticks.length - 1], x0 = Math.min(...xs), x1 = Math.max(...xs);
  const X = (x) => m.l + ((x - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (y) => H - m.b - ((y - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
  const grid = s("g", { class: "grid" }), axis = s("g", { class: "axis" });
  ticks.forEach((t) => {
    grid.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
    const tx = s("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }); tx.textContent = yFormat(t); axis.append(tx);
  });
  const step = Math.max(1, Math.round((x1 - x0) / 6));
  for (let x = x0; x <= x1; x += step) { const tx = s("text", { x: X(x), y: H - 6, "text-anchor": "middle" }); tx.textContent = x; axis.append(tx); }
  svg.append(grid, axis);
  const solid = points.filter((p) => !p.forecast), fc = points.filter((p) => p.forecast);
  const path = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
  svg.append(s("path", { d: path(solid), fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round" }));
  if (fc.length && solid.length) {
    svg.append(s("path", { d: path([solid[solid.length - 1], ...fc]), fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": "4 4" }));
    fc.forEach((p) => svg.append(s("circle", { cx: X(p.x), cy: Y(p.y), r: 4.5, fill: "var(--panel)", stroke: color, "stroke-width": 2 })));
  }
  const last = solid[solid.length - 1];
  if (last) svg.append(s("circle", { cx: X(last.x), cy: Y(last.y), r: 4, fill: color, stroke: "var(--panel)", "stroke-width": 2 }));
  // crosshair + tooltip on the nearest year
  const cross = s("line", { y1: m.t, y2: H - m.b, stroke: "var(--faint)", "stroke-width": 1, visibility: "hidden" });
  const dot = s("circle", { r: 5, fill: color, stroke: "var(--panel)", "stroke-width": 2, visibility: "hidden" });
  const hit = s("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
  svg.append(cross, dot, hit);
  hit.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect(), vx = ((e.clientX - r.left) / r.width) * W;
    const p = points.reduce((a, b) => (Math.abs(X(b.x) - vx) < Math.abs(X(a.x) - vx) ? b : a));
    cross.setAttribute("x1", X(p.x)); cross.setAttribute("x2", X(p.x)); cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", X(p.x)); dot.setAttribute("cy", Y(p.y)); dot.setAttribute("visibility", "visible");
    showTip(e, `${p.x}${p.forecast ? " (forecast)" : ""}<br><b>${yFormat(p.y, true)}</b>`);
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); hideTip(); });
  return el("div", { class: "card chart" }, el("h3", {}, title), svg, note ? el("p", { class: "small faint" }, note) : null);
}

/** categories: [label], series: [{name, color, values: [number|null]}] */
export function groupedBars({ title, categories, series, yFormat, note, width = 560 }) {
  const W = width, H = 230, m = { l: 44, r: 10, t: 10, b: 28 };
  const vals = series.flatMap((x) => x.values).filter((v) => v != null);
  const ticks = niceTicks(0, Math.max(...vals));
  const y1 = ticks[ticks.length - 1];
  const Y = (y) => H - m.b - (y / y1) * (H - m.t - m.b);
  const band = (W - m.l - m.r) / categories.length, gap = 2, bw = Math.min(28, (band * 0.7 - gap * (series.length - 1)) / series.length);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
  const grid = s("g", { class: "grid" }), axis = s("g", { class: "axis" });
  ticks.forEach((t) => {
    grid.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
    const tx = s("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }); tx.textContent = yFormat(t); axis.append(tx);
  });
  categories.forEach((c, i) => { const tx = s("text", { x: m.l + band * (i + 0.5), y: H - 8, "text-anchor": "middle" }); tx.textContent = c; axis.append(tx); });
  svg.append(grid, axis);
  categories.forEach((c, i) => {
    const groupW = series.length * bw + (series.length - 1) * gap, start = m.l + band * (i + 0.5) - groupW / 2;
    series.forEach((ser, j) => {
      const v = ser.values[i]; if (v == null) return;
      const x = start + j * (bw + gap), h = Math.max(1, H - m.b - Y(v)), r = Math.min(4, bw / 2, h);
      const d = `M${x},${H - m.b} V${Y(v) + r} Q${x},${Y(v)} ${x + r},${Y(v)} H${x + bw - r} Q${x + bw},${Y(v)} ${x + bw},${Y(v) + r} V${H - m.b} Z`;
      const bar = s("path", { d, fill: ser.color });
      const hit = s("rect", { x: x - 2, y: m.t, width: bw + 4, height: H - m.t - m.b, fill: "transparent" });
      hit.addEventListener("mousemove", (e) => showTip(e, `${c} · ${ser.name}<br><b>${yFormat(v, true)}</b>`));
      hit.addEventListener("mouseleave", hideTip);
      svg.append(bar, hit);
    });
  });
  svg.append(s("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
  const legend = el("div", { class: "legend" }, series.map((x) => el("span", {}, el("i", { style: `background:${x.color}` }), x.name)));
  const table = el("details", {}, el("summary", { class: "small" }, "Show as a table"),
    el("div", { class: "scroll-x" }, el("table", { class: "data" },
      el("thead", {}, el("tr", {}, el("th", {}, ""), series.map((x) => el("th", { class: "num" }, x.name)))),
      el("tbody", {}, categories.map((c, i) => el("tr", {}, el("td", {}, c), series.map((x) => el("td", { class: "num" }, x.values[i] == null ? "—" : yFormat(x.values[i], true)))))))));
  return el("div", { class: "card chart" }, el("h3", {}, title), legend, svg, table, note ? el("p", { class: "small faint" }, note) : null);
}
