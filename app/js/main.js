// FieldVision demo: wiring between data files, map and side panel.

import { FieldMap, COLORS, RAMP } from "./map.js";
import { seasonChart, chartLegendHtml } from "./chart.js";
import { evidenceSentences, lowEvidenceText, certainty, confidence } from "./evidence.js";
import { scatterSvg, countyTableHtml } from "./counties.js";
import { pct, int, ha, fixed, shortDate, isoToDay, daysBetween, esc, cap, isNum, DASH } from "./format.js";

const DATA = "data";
const LIST_LEN = 12;

const state = {
  regions: [],
  region: null,     // entry from regions.json
  meta: null,
  fields: [],       // feature properties
  byId: new Map(),
  series: null,     // null while loading, {} when missing
  seriesError: false,
  counties: null,
  countyShapes: null,
  selected: null,
  tab: "check",
  mode: "pred",
  base: "sat",
  county: null,
};
let loadToken = 0;

const $ = (id) => document.getElementById(id);
const fieldMap = new FieldMap($("map"), {
  onSelect: (id) => selectField(id),
  onCounty: (fips) => selectCounty(fips, { zoom: false }),
});

// ---------------------------------------------------------------- data loading

async function getJson(path, { optional = false } = {}) {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) {
    if (optional && res.status === 404) return null;
    throw new Error(`${path}: HTTP ${res.status}`);
  }
  return res.json();
}

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  return { r: h.get("r"), f: h.get("f"), tab: h.get("tab"), mode: h.get("mode") };
}

function writeHash() {
  const h = new URLSearchParams();
  if (state.region) h.set("r", state.region.id);
  if (state.selected) h.set("f", state.selected);
  if (state.tab !== "check") h.set("tab", state.tab);
  if (state.mode !== "pred") h.set("mode", state.mode);
  history.replaceState(null, "", `#${h.toString()}`);
}

async function boot() {
  if (location.protocol === "file:") {
    fatal(`Browsers block data loading from <code>file://</code>. Start the local server from the repository root:<br><br><code>python -m http.server 8000 -d app</code><br><br>then open <code>http://localhost:8000</code>.`);
    return;
  }
  let index;
  try {
    index = await getJson(`${DATA}/regions.json`);
  } catch (err) {
    fatal(`No region index found (<code>app/data/regions.json</code>). Create sample data with<br><br><code>python -m fieldvision.app.sample_data</code><br><br>or export a region with <code>python -m fieldvision.app.export</code>.<br><br><small>${esc(err.message)}</small>`);
    return;
  }
  state.regions = index.regions || [];
  if (!state.regions.length) { fatal("The region index is empty. Export a region first."); return; }
  const h = readHash();
  if (h.tab) state.tab = h.tab;
  if (h.mode) state.mode = h.mode;
  const first = state.regions.find((r) => r.id === h.r) || state.regions[0];
  await loadRegion(first.id, h.f);
}

async function loadRegion(id, fieldId = null) {
  const region = state.regions.find((r) => r.id === id);
  if (!region) return;
  const token = ++loadToken;
  state.region = region;
  state.selected = null;
  state.county = null;
  state.series = null;
  state.seriesError = false;
  state.counties = null;
  state.countyShapes = null;
  renderRegions();
  $("view").innerHTML = `<p class="empty">Loading ${esc(region.title)}…</p>`;
  const base = `${DATA}/${region.path}`;
  let meta, fc;
  try {
    [meta, fc] = await Promise.all([getJson(`${base}/meta.json`), getJson(`${base}/fields.geojson`)]);
  } catch (err) {
    if (token !== loadToken) return;
    state.meta = null;
    state.fields = [];
    fieldMap.clear();
    $("metrics").innerHTML = `<div class="head"><h2>${esc(region.title)}</h2></div><div class="notice">Could not load this region's files. Re-run the export for it.<br><small>${esc(err.message)}</small></div>`;
    $("view").innerHTML = "";
    $("badge").hidden = !region.synthetic;
    renderLegend();
    return;
  }
  if (token !== loadToken) return;
  state.meta = meta;
  state.fields = fc.features.map((f) => f.properties);
  state.byId = new Map(state.fields.map((p) => [p.field_id, p]));

  if (!meta.has_predictions && state.mode === "check") state.mode = "pred";
  fieldMap.mode = state.mode;
  fieldMap.setFields(fc.features);

  if (meta.has_counties) {
    try {
      state.counties = await getJson(`${base}/counties.json`, { optional: true });
      if (meta.has_county_shapes) state.countyShapes = await getJson(`${base}/counties.geojson`, { optional: true });
    } catch { state.counties = null; }
    if (token !== loadToken) return;
    const rows = new Map((state.counties?.rows || []).map((r) => [r.fips, r]));
    fieldMap.setCounties(state.countyShapes, rows);
  }

  if (!availableTabs().includes(state.tab)) state.tab = "check";
  renderAll();
  if (fieldId && state.byId.has(fieldId)) selectField(fieldId);

  // the curves are the biggest file; load them after the map is up
  if (meta.has_series) {
    getJson(`${base}/series.json`)
      .then((s) => { if (token === loadToken) { state.series = s; if (state.selected) renderView(); } })
      .catch(() => { if (token === loadToken) { state.series = {}; state.seriesError = true; if (state.selected) renderView(); } });
  } else {
    state.series = {};
  }
}

function fatal(html) {
  document.querySelector(".layout").innerHTML = `<div class="fatal">${html}</div>`;
}

// ---------------------------------------------------------------- top bar

function renderRegions() {
  const groups = [];
  for (const r of state.regions) {
    let g = groups.find((x) => x.name === r.group);
    if (!g) { g = { name: r.group, items: [] }; groups.push(g); }
    g.items.push(r);
  }
  const cur = state.region;
  $("regions").innerHTML = groups.map((g) => {
    const on = g.items.some((r) => r.id === cur?.id);
    if (g.items.length === 1) {
      const r = g.items[0];
      return `<span class="group${on ? " on" : ""}"><button type="button" class="${on ? "on" : ""}" data-region="${esc(r.id)}">${esc(r.title)}</button></span>`;
    }
    const years = g.items.map((r) => `<button type="button" class="year${r.id === cur?.id ? " on" : ""}" data-region="${esc(r.id)}">${esc(r.year)}</button>`).join("");
    return `<span class="group${on ? " on" : ""}"><button type="button" class="name" data-region="${esc(g.items[0].id)}" data-group="1">${esc(g.name)}</button>${years}</span>`;
  }).join("");
}

$("regions").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-region]");
  if (!b) return;
  if (b.dataset.group && state.region && state.region.group === state.regions.find((r) => r.id === b.dataset.region)?.group) return;
  loadRegion(b.dataset.region).then(writeHash);
});

// ---------------------------------------------------------------- metrics

function metricTile(label, value, detail, title = "") {
  const na = value === DASH;
  return `<div class="metric" title="${esc(title)}"><div class="label">${label}</div><div class="value${na ? " na" : ""}">${value}</div><div class="detail">${detail || "&nbsp;"}</div></div>`;
}

function renderMetrics() {
  const meta = state.meta;
  const m = meta.metrics || {};
  const a = m.all;
  const sub = [`${int(m.n_fields)} fields`];
  if (meta.year) sub.unshift(String(meta.year));
  let tiles;
  const noAcc = !meta.has_predictions ? "no predictions" : "no declared labels";
  tiles = [
    metricTile("Precision", a ? pct(a.precision) : DASH, a ? `TP ${a.tp} · FP ${a.fp}` : noAcc, "Of fields called silage, share declared silage"),
    metricTile("Recall", a ? pct(a.recall) : DASH, a ? `TP ${a.tp} · FN ${a.fn}` : noAcc, "Of declared silage fields, share the model found"),
    metricTile("False pos. rate", a ? pct(a.fpr) : DASH, a ? `FP ${a.fp} · TN ${a.tn}` : noAcc, "Of declared grain fields, share wrongly called silage"),
  ];
  const tiles2 = [
    metricTile("Fields", int(m.n_fields), meta.has_predictions ? `${int(m.n_predicted)} predicted` : "no predictions"),
    metricTile("Low evidence", meta.has_predictions ? pct(m.low_evidence_share) : DASH, meta.has_predictions ? `${int(m.n_low_evidence)} fields flagged` : "", "Share of predicted fields flagged low evidence (clouds, no clear drop)"),
    metricTile("Called silage", meta.has_predictions ? pct(m.predicted_silage_share) : DASH, meta.has_predictions ? "of predicted fields" : "", "Share of predicted fields the model calls silage"),
  ];
  let foot = "";
  if (!meta.has_predictions) {
    foot = `<div class="notice">No predictions for this region yet. Fields and satellite curves only.</div>`;
  } else if (!a) {
    foot = `<div class="foot">No declared labels here, so no accuracy numbers.${meta.note ? `<br>${esc(meta.note)}` : ""}</div>`;
  } else {
    const c = m.confident;
    const parts = [`Silage = positive class · scored on <b>${int(a.n)}</b> declared fields`];
    if (c && c.n !== a.n) parts.push(`Confident fields only (${int(c.n)}): precision <b>${pct(c.precision)}</b>, recall <b>${pct(c.recall)}</b>, FPR <b>${pct(c.fpr)}</b>`);
    if (meta.note) parts.push(esc(meta.note));
    foot = `<div class="foot">${parts.join("<br>")}</div>`;
  }
  $("metrics").innerHTML = `<div class="head"><h2>${esc(meta.title)}</h2><div class="sub">${sub.join(" · ")}</div></div>
    <div class="metric-grid">${tiles.join("")}</div><div class="metric-grid small">${tiles2.join("")}</div>${foot}`;
}

// ---------------------------------------------------------------- legend

function renderLegend() {
  const el = $("legend");
  const meta = state.meta;
  if (!meta) { el.innerHTML = ""; return; }
  const f = state.fields;
  const count = (pred) => f.filter(pred).length;
  let rows;
  if (state.mode === "check") {
    rows = [
      [`<span class="sw" style="background:${COLORS.ok}"></span>`, "Matches declared", count((p) => p.correct === true)],
      [`<span class="sw" style="background:${COLORS.wrong}"></span>`, "Model wrong", count((p) => p.correct === false)],
      [`<span class="sw hollow"></span>`, "No label or no prediction", count((p) => p.correct == null)],
    ];
  } else {
    rows = [
      [`<span class="sw" style="background:${COLORS.silage}"></span>`, "Silage", count((p) => p.pred === "silage" && p.low !== true)],
      [`<span class="sw" style="background:${COLORS.grain}"></span>`, "Grain", count((p) => p.pred === "grain" && p.low !== true)],
      [`<span class="sw dashed" style="border-color:${COLORS.silage}"></span>`, "Silage, low evidence", count((p) => p.pred === "silage" && p.low === true)],
      [`<span class="sw dashed" style="border-color:${COLORS.grain}"></span>`, "Grain, low evidence", count((p) => p.pred === "grain" && p.low === true)],
      [`<span class="sw hollow"></span>`, "No prediction", count((p) => !p.pred)],
    ].filter((r) => (meta.has_predictions ? r[2] > 0 || r[1] !== "No prediction" : r[1] === "No prediction"));
  }
  let html = `<h4>${state.mode === "check" ? "Model vs declared label" : "Predicted harvest use"}</h4>` +
    rows.map(([sw, name, n]) => `<div class="row">${sw}<span>${name}</span><span class="count">${n}</span></div>`).join("");
  if (state.tab === "counties" && state.countyShapes) {
    html = `<h4>County: predicted silage share</h4><div class="ramp">${RAMP.map((c) => `<span style="background:${c}"></span>`).join("")}</div>
      <div class="ramp-labels"><span>0%</span><span>50%</span><span>100%</span></div><div style="height:0.6rem"></div>` + html;
  }
  el.innerHTML = html;
}

// ---------------------------------------------------------------- tabs + views

function availableTabs() {
  const meta = state.meta;
  const tabs = ["check"];
  if (meta?.has_predictions && meta?.has_labels) tabs.push("wrong");
  if (meta?.has_counties && state.counties) tabs.push("counties");
  return tabs;
}

function renderTabs() {
  const avail = availableTabs();
  for (const b of $("tabs").querySelectorAll("button")) {
    b.hidden = !avail.includes(b.dataset.tab);
    const on = !state.selected && b.dataset.tab === state.tab;
    b.classList.toggle("on", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  }
}

$("tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-tab]");
  if (!b) return;
  state.tab = b.dataset.tab;
  state.selected = null;
  fieldMap.select(null);
  renderAll();
  $("view").scrollTop = 0;
  writeHash();
});

function itemRow(p, i, metaLine) {
  const conf = confidence(p);
  const cert = certainty(p);
  const pill = p.pred ? `<span class="pill ${p.pred}">${cap(p.pred)} ${pct(conf)}</span>` : `<span class="pill none">no prediction</span>`;
  return `<li><button type="button" class="item" data-field="${esc(p.field_id)}">
    <span class="rank">${i + 1}</span>
    <span><span class="id">${esc(p.field_id)}</span><br><span class="meta">${metaLine}</span></span>
    <span class="right">${pill}${isNum(cert) ? `<span class="cert" title="certainty ${Math.round(cert * 100)}%"><span style="width:${Math.max(3, cert * 100)}%"></span></span>` : ""}</span>
  </button></li>`;
}

function renderCheckList() {
  const meta = state.meta;
  if (!meta.has_predictions) {
    return `<p class="empty">No predictions yet, so nothing to rank.<br>Click any field on the map to see its satellite curves.</p>`;
  }
  const ranked = state.fields.filter((p) => isNum(p.p)).sort((a, b) => certainty(a) - certainty(b)).slice(0, LIST_LEN);
  if (!ranked.length) return `<p class="empty">No field has a probability to rank.</p>`;
  const area = ranked.reduce((s, p) => s + (p.area_ha || 0), 0);
  const rows = ranked.map((p, i) => {
    const bits = [ha(p.area_ha)];
    if (p.low === true) bits.push(`<span class="tag">low evidence</span>`);
    if (p.correct === false) bits.push(`<span class="tag wrong">wrong</span>`);
    else if (p.label) bits.push(`declared ${p.label}`);
    return itemRow(p, i, bits.join(" · "));
  }).join("");
  return `<p class="intro">The ${ranked.length} least certain predictions: where a survey call or field visit tells us the most (${ha(area)} in total).</p><ol class="list">${rows}</ol>`;
}

function renderWrongList() {
  const wrong = state.fields.filter((p) => p.correct === false).sort((a, b) => (confidence(b) ?? 0) - (confidence(a) ?? 0));
  if (!wrong.length) return `<p class="empty">The model agrees with every declared label here.</p>`;
  const rows = wrong.map((p, i) => {
    const bits = [`declared <b>${p.label}</b>`];
    if (p.hs && p.he) bits.push(`cut ${shortDate(p.hs)}–${shortDate(p.he)}`);
    if (p.low === true) bits.push(`<span class="tag">low evidence</span>`);
    return itemRow(p, i, bits.join(" · "));
  }).join("");
  return `<p class="intro">${wrong.length} field${wrong.length === 1 ? "" : "s"} where the declared label disagrees with the model. Most confident mistakes first.</p><ol class="list">${rows}</ol>`;
}

function renderCounties() {
  const c = state.counties;
  if (!c || !c.rows?.length) return `<p class="empty">No county table for this region.</p>`;
  const s = c.summary || {};
  const width = Math.max(280, $("view").clientWidth - 32);
  const skipped = c.rows.length - c.rows.filter((r) => isNum(r.pred_silage_share) && isNum(r.usda_silage_share)).length;
  return `<div class="county-summary">
      <div class="metric"><div class="label">Counties</div><div class="value">${int(s.n_compared)}</div><div class="detail">of ${int(s.n_counties)} compared</div></div>
      <div class="metric" title="Mean absolute difference between predicted and USDA silage share"><div class="label">Mean abs. diff.</div><div class="value${isNum(s.mae) ? "" : " na"}">${isNum(s.mae) ? `${(s.mae * 100).toFixed(0)} pts` : DASH}</div><div class="detail">percentage points</div></div>
      <div class="metric" title="Pearson correlation across counties"><div class="label">Correlation</div><div class="value${isNum(s.r) ? "" : " na"}">${fixed(s.r, 2)}</div><div class="detail">Pearson r</div></div>
    </div>
    <div class="scatter" id="scatter">${scatterSvg(c.rows, { width, height: Math.round(width * 0.72), highlight: state.county })}</div>
    ${skipped ? `<p class="intro">${skipped} count${skipped === 1 ? "y" : "ies"} without predicted fields or USDA numbers not plotted.</p>` : ""}
    ${countyTableHtml(c.rows, state.county)}
    <p class="intro" style="margin-top:0.7rem">Predicted share: area of fields called silage over area of all predicted fields in the county. USDA share: acres harvested for silage over silage plus grain acres.</p>`;
}

function renderDetail(p) {
  const meta = state.meta;
  const year = meta.year;
  const s = state.series === null ? undefined : state.series[p.field_id];
  const conf = confidence(p);
  const head = `<div class="detail-head">
      <div><h3>${esc(p.field_id)}</h3><div class="sub">${ha(p.area_ha)}${p.block_id ? ` · block ${esc(p.block_id)}` : ""}</div></div>
      <button type="button" class="back" data-action="back" title="Back to the list (Esc)">← Back</button>
    </div>`;
  const verdict = p.pred
    ? `<div class="verdict"><span class="big ${p.pred}">${cap(p.pred)}</span><span class="prob">${pct(conf)} confident · p(silage) ${fixed(p.p, 2)}</span></div>`
    : `<div class="verdict"><span class="big none">No prediction yet</span></div>`;
  const low = lowEvidenceText(p);
  const lowBox = low ? `<div class="lowbox"><b>Low evidence:</b> ${esc(low)}</div>` : "";
  let declared;
  if (!p.label) declared = `<div class="declared">No declared label for this field.</div>`;
  else if (p.correct === false) declared = `<div class="declared">Declared: <b>${p.label}</b> <span class="miss">✗ model says ${p.pred}</span></div>`;
  else if (p.correct === true) declared = `<div class="declared">Declared: <b>${p.label}</b> <span class="match">✓ matches</span></div>`;
  else declared = `<div class="declared">Declared: <b>${p.label}</b></div>`;
  const ev = evidenceSentences(p, meta.cues);
  const evHtml = ev.length ? `<ul class="evidence">${ev.map((x) => `<li>${x}</li>`).join("")}</ul>` : "";

  const w = daysBetween(p.hs, p.he);
  const stats = `<div class="stats">
      <div>Harvest window<b>${p.hs && p.he ? `${shortDate(p.hs)} – ${shortDate(p.he)}` : DASH}</b>${isNum(w) ? `${w} days between clear views` : p.pred ? "no drop detected" : ""}</div>
      <div>Clear observations<b>${int(p.n_usable)} of ${int(p.n_obs)}</b>${isNum(p.max_gap) ? `longest gap ${p.max_gap} days` : ""}</div>
      <div>NDVI before cut<b>${fixed(p.pre_ndvi, 2)}</b>greenness</div>
      <div>NDMI before cut<b>${fixed(p.pre_ndmi, 2)}</b>leaf moisture</div>
    </div>`;

  let charts;
  if (!meta.has_series) charts = `<div class="chart-empty">No satellite time series was exported for this region.</div>`;
  else if (state.series === null) charts = `<div class="chart-empty">Loading satellite curves…</div>`;
  else if (state.seriesError) charts = `<div class="chart-empty">Could not load the satellite curves for this region.</div>`;
  else if (!s || !s.t.length) charts = `<div class="chart-empty">No Sentinel-2 acquisitions for this field.</div>`;
  else {
    const width = Math.max(280, $("view").clientWidth - 32);
    const domain = meta.season ? [meta.season.start, meta.season.end] : [Math.min(...s.t), Math.max(...s.t)];
    const hs = isoToDay(p.hs, year), he = isoToDay(p.he, year);
    const band = isNum(hs) && isNum(he) ? [hs, he] : null;
    const common = { width, height: 165, year, domain, series: s, band, gapDays: meta.gap_days ?? 20 };
    const noClear = p.n_usable === 0 ? `<div class="chart-empty">No cloud-free observation all season: every acquisition below is cloudy or excluded, so nothing is plotted as a curve.</div>` : "";
    charts = noClear +
      `<div class="chart">${seasonChart({ ...common, key: "ndvi", label: "NDVI · greenness", color: "#8fd17f", yMin: -0.1, yMax: 1, yTicks: [0, 0.2, 0.4, 0.6, 0.8, 1], pre: isNum(hs) ? { t: hs, v: p.pre_ndvi } : null })}</div>` +
      `<div class="chart">${seasonChart({ ...common, key: "ndmi", label: "NDMI · leaf moisture", color: "#5ec8f2", yMin: -0.4, yMax: 0.6, yTicks: [-0.4, -0.2, 0, 0.2, 0.4, 0.6], pre: isNum(hs) ? { t: hs, v: p.pre_ndmi } : null })}</div>` +
      chartLegendHtml("#cfd8e3", meta.min_clear);
  }
  return head + verdict + declared + lowBox + evHtml + stats + charts;
}

function renderView() {
  const view = $("view");
  if (!state.meta) { view.innerHTML = ""; return; }
  const p = state.selected && state.byId.get(state.selected);
  if (p) view.innerHTML = renderDetail(p);
  else if (state.tab === "wrong") view.innerHTML = renderWrongList();
  else if (state.tab === "counties") view.innerHTML = renderCounties();
  else view.innerHTML = renderCheckList();
  $("metrics").classList.toggle("compact", !!p);
  renderTabs();
}

function renderAll() {
  $("badge").hidden = !state.meta?.synthetic;
  renderRegions();
  renderMetrics();
  renderTabs();
  renderView();
  renderLegend();
  renderModeButtons();
  fieldMap.showCounties(state.tab === "counties" && !state.selected);
}

// ---------------------------------------------------------------- interactions

function selectField(id) {
  if (!state.byId.has(id)) return;
  state.selected = id;
  fieldMap.select(id);
  fieldMap.showCounties(false);
  renderView();
  renderLegend();
  $("view").scrollTop = 0;
  // stacked layout on narrow screens: bring the detail into view
  if (window.matchMedia("(max-width: 960px)").matches) $("metrics").scrollIntoView({ behavior: "smooth", block: "start" });
  writeHash();
}

function closeField() {
  if (!state.selected) return;
  state.selected = null;
  fieldMap.select(null);
  fieldMap.showCounties(state.tab === "counties");
  renderView();
  renderLegend();
  writeHash();
}

function selectCounty(fips, { zoom = true } = {}) {
  if (state.tab !== "counties") { state.tab = "counties"; state.selected = null; fieldMap.select(null); renderAll(); }
  state.county = fips;
  fieldMap.highlightCounty(fips);
  if (zoom) fieldMap.zoomToCounty(fips);
  renderView();
  const row = $("view").querySelector(`tr[data-fips="${CSS.escape(fips)}"]`);
  if (row) row.scrollIntoView({ block: "nearest" });
}

$("view").addEventListener("click", (e) => {
  const item = e.target.closest("[data-field]");
  if (item) { selectField(item.dataset.field); return; }
  if (e.target.closest("[data-action=back]")) { closeField(); return; }
  const c = e.target.closest("[data-fips]");
  if (c) selectCounty(c.dataset.fips);
});

$("view").addEventListener("mouseover", (e) => {
  const c = e.target.closest("tr[data-fips]");
  if (c) fieldMap.highlightCounty(c.dataset.fips);
});

function renderModeButtons() {
  const canCheck = state.meta?.has_predictions && state.meta?.has_labels;
  for (const b of document.querySelectorAll("[data-mode]")) {
    b.classList.toggle("on", b.dataset.mode === state.mode);
    if (b.dataset.mode === "check") {
      b.disabled = !canCheck;
      b.title = canCheck ? "Colour fields by whether the model agrees with the declared label" : "Needs predictions and declared labels";
    }
  }
}

document.querySelector(".map-tools").addEventListener("click", (e) => {
  const m = e.target.closest("button[data-mode]");
  if (m && !m.disabled) {
    state.mode = m.dataset.mode;
    fieldMap.setMode(state.mode);
    renderModeButtons();
    renderLegend();
    writeHash();
    return;
  }
  const b = e.target.closest("button[data-base]");
  if (b) {
    state.base = b.dataset.base;
    fieldMap.setBase(state.base);
    for (const x of document.querySelectorAll("[data-base]")) x.classList.toggle("on", x.dataset.base === state.base);
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeField();
});

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { fieldMap.invalidate(); if (state.meta) renderView(); }, 150);
});

boot();
