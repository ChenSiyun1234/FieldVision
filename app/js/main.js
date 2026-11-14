// FieldVision demo: wiring between data files, map and side panel.

import { FieldMap, COLORS, RAMP } from "./map.js";
import { seasonChart, chartLegendHtml } from "./chart.js";
import { defendRows, tally, flipConditions, traps, certainty, confidence, cuesOf, settledFrom, regionSettled, hasModelEvidence, plainModelText } from "./evidence.js";
import { scatterSvg, countyTableHtml, stateCardsHtml } from "./counties.js";
import { Timeline } from "./timeline.js";
import { pct, int, ha, fixed, shortDate, isoToDay, dayToLabel, esc, cap, isNum, DASH } from "./format.js";

const DATA = "data";
const LIST_LEN = 12;

const state = {
  regions: [],
  region: null,     // entry from regions.json
  meta: null,
  fields: [],       // full-season feature properties
  byId: new Map(),
  view: [],         // properties as shown: full season, or as of the chosen cutoff
  viewById: new Map(),
  series: null,     // null while loading, {} when missing
  seriesError: false,
  asof: null,
  cut: null,        // index into asof.cutoffs; null = full season
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
const timeline = new Timeline($("timeline"), (i) => setCut(i));

// ---------------------------------------------------------------- as-of view

const asofActive = () => state.asof && state.cut !== null && state.cut < state.asof.cutoffs.length;
const cutoffIso = () => (asofActive() ? state.asof.cutoffs[state.cut] : null);
const cutoffDay = () => (asofActive() ? isoToDay(cutoffIso(), state.meta.year) : null);

/** Field properties as they would have looked with data up to the cutoff. */
function viewOf(p) {
  if (!asofActive()) return p;
  const a = state.asof.fields[p.field_id];
  const prob = a ? a.p[state.cut] : null;
  const pred = isNum(prob) ? (prob >= (state.meta.threshold ?? 0.5) ? "silage" : "grain") : null;
  const cut = cutoffDay();
  const seen = (iso) => Boolean(iso) && isoToDay(iso, state.meta.year) <= cut;
  const cutSeen = seen(p.he);
  return {
    ...p,
    p: isNum(prob) ? prob : null,
    pred,
    low: a ? a.low[state.cut] : null,
    reason: null,
    correct: p.label && pred ? p.label === pred : null,
    // evidence from after the cutoff does not exist yet
    hs: cutSeen ? p.hs : null,
    he: cutSeen ? p.he : null,
    pre_ndvi: cutSeen ? p.pre_ndvi : null,
    pre_ndmi: cutSeen ? p.pre_ndmi : null,
    post_ndvi: cutSeen ? p.post_ndvi : null,
    post_ndti: cutSeen ? p.post_ndti : null,
    explanation: null,
    flip: null,
    asofHidden: !cutSeen,
  };
}

function rebuildView() {
  state.view = state.fields.map(viewOf);
  state.viewById = new Map(state.view.map((p) => [p.field_id, p]));
}

function currentMetrics() {
  return asofActive() ? state.asof.metrics[state.cut] : state.meta.metrics;
}

function setCut(i) {
  if (!state.asof) return;
  const n = state.asof.cutoffs.length;
  state.cut = i >= n ? null : i;
  rebuildView();
  fieldMap.setProps(state.viewById);
  renderMetrics();
  renderView();
  renderLegend();
  renderTimeline();
  writeHash();
}

function renderTimeline() {
  if (!state.meta) { timeline.render(null); return; }
  const n = state.asof?.cutoffs?.length || 0;
  timeline.render(state.asof, state.meta.metrics, state.cut === null ? n : state.cut, {
    hasLabels: state.meta.has_labels,
    baseline: state.meta.metrics?.always_grain?.accuracy ?? null,
  });
}

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
  return { r: h.get("r"), f: h.get("f"), tab: h.get("tab"), mode: h.get("mode"), asof: h.get("asof"), intro: h.get("intro") };
}

function writeHash() {
  const h = new URLSearchParams();
  if (state.region) h.set("r", state.region.id);
  if (state.selected) h.set("f", state.selected);
  if (state.tab !== "check") h.set("tab", state.tab);
  if (state.mode !== "pred") h.set("mode", state.mode);
  if (asofActive()) h.set("asof", cutoffIso());
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
  showIntro(!h.f && !h.tab && h.intro !== "0");
  const first = state.regions.find((r) => r.id === h.r) || state.regions[0];
  await loadRegion(first.id, h.f, h.asof);
}

/** Sum of the scored results over every region in the same group, e.g. three years of one farm. */
async function loadGroupSummary(region) {
  const members = state.regions.filter((r) => r.group === region.group);
  if (members.length < 2) return null;
  const metas = await Promise.all(members.map((r) => getJson(`${DATA}/${r.path}/meta.json`).catch(() => null)));
  if (metas.some((m) => !m || !m.metrics?.all)) return null;
  const sum = (f) => metas.reduce((a, m) => a + f(m), 0);
  const tp = sum((m) => m.metrics.all.tp), fp = sum((m) => m.metrics.all.fp);
  const tn = sum((m) => m.metrics.all.tn), fn = sum((m) => m.metrics.all.fn);
  return {
    group: region.group,
    years: members.map((r) => r.year),
    n: tp + fp + tn + fn, right: tp + tn, tp, fp, tn, fn,
    silage: tp + fn, grain: tn + fp,
    low: sum((m) => m.metrics.n_low_evidence || 0),
    synthetic: metas.some((m) => m.synthetic),
    model: metas[0].model,
    labelName: metas[0].label_name,
  };
}

async function loadRegion(id, fieldId = null, asofIso = null) {
  const region = state.regions.find((r) => r.id === id);
  if (!region) return;
  const token = ++loadToken;
  Object.assign(state, {
    region, selected: null, county: null, series: null, seriesError: false,
    counties: null, countyShapes: null, asof: null, cut: null, group: null,
  });
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
    renderTimeline();
    return;
  }
  if (token !== loadToken) return;
  state.meta = meta;
  state.group = await loadGroupSummary(region);
  if (token !== loadToken) return;
  state.fields = fc.features.map((f) => f.properties);
  state.byId = new Map(state.fields.map((p) => [p.field_id, p]));

  if (meta.has_asof) {
    try { state.asof = await getJson(`${base}/asof.json`, { optional: true }); } catch { state.asof = null; }
    if (token !== loadToken) return;
    const k = asofIso && state.asof ? state.asof.cutoffs.indexOf(asofIso) : -1;
    state.cut = k >= 0 ? k : null;
  }
  rebuildView();

  if (!meta.has_predictions && state.mode === "check") state.mode = "pred";
  fieldMap.mode = state.mode;
  const wide = window.matchMedia("(min-width: 961px)").matches;
  fieldMap.setFields(fc.features, state.asof && wide ? { topLeft: [40, 60], bottomRight: [60, 170] } : null);
  fieldMap.setProps(state.viewById);

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
  renderOpening();
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

// ---------------------------------------------------------------- opening card

function showIntro(on) {
  $("intro").hidden = !on;
}

const median = (xs) => {
  const v = xs.filter(isNum).sort((a, b) => a - b);
  return v.length ? v[Math.floor(v.length / 2)] : null;
};

/** The organisers' three requirements, each answered with this region's numbers. */
function renderOpening() {
  const meta = state.meta;
  if (!meta) return;
  const looks = median(state.fields.map((p) => p.n_obs));
  const clear = median(state.fields.map((p) => p.n_usable));
  const data = isNum(looks)
    ? `Sentinel-2, free from the EU's Copernicus programme: about ${looks} looks per field this season, ${clear} of them clear.`
    : "Sentinel-2, free from the EU's Copernicus programme.";
  const s = regionSettled(state.asof, meta.metrics, meta.has_labels);
  let early;
  if (s && meta.has_labels) early = `Calls settle by <b>${shortDate(s.iso)}</b>: ${pct(s.value)} accurate, against ${pct(s.full)} with the whole season. NASS publishes the split in January.`;
  else if (s) early = `By <b>${shortDate(s.iso)}</b>, ${pct(s.value)} of calls are confident. NASS publishes the split in January.`;
  else if (meta.season) early = `Each call uses imagery only up to ${dayToLabel(meta.season.end, meta.year)}. NASS publishes the split in January.`;
  else early = "Each call is made as soon as the harvest drop is seen. NASS publishes the split in January.";
  const skeptic = "Every call shows its curve: silage falls off a cliff, grain fades out; after harvest, bare soil or stalk residue.";
  $("yardsticks").innerHTML = [
    ["Public satellite data", data],
    ["Earlier than January", early],
    ["Explainable to a skeptic", skeptic],
  ].map(([h, t]) => `<div class="yard"><div class="yard-h"><span class="tick">\u2713</span>${h}</div><div class="yard-t">${t}</div></div>`).join("");
  $("opening-region").textContent = `Showing ${meta.title}${meta.synthetic ? " \u00b7 synthetic sample data" : ""}`;
  $("result").innerHTML = resultHtml();
  $("result").hidden = !$("result").innerHTML;
}

/** Ten-second summary of a real, scored test: how many right, against always-grain, and how small it is. */
function resultHtml() {
  const g = state.group;
  if (!g || g.synthetic || !g.n) return "";
  const model = g.model?.name ? `<code>${esc(g.model.name)}</code>` : "The model";
  const trained = g.model?.trained_on ? `Trained on ${esc(g.model.trained_on)}.` : "";
  const years = g.years.length > 1 ? `${Math.min(...g.years)}\u2013${Math.max(...g.years)}` : `${g.years[0]}`;
  return `<div class="result-big"><b>${g.right} of ${g.n}</b> ${esc(g.group.split(",")[0])} field-years called right <span>(${years})</span></div>
    <div class="result-row">
      <div><b>${g.tn} of ${g.grain}</b>grain fields right</div>
      <div><b>${g.tp} of ${g.silage}</b>silage fields found</div>
      <div class="vs"><b>${g.grain} of ${g.n}</b>"always grain" baseline</div>
    </div>
    <p class="result-note">${model}: ${trained}</p>
    <p class="result-note small">A small test: ${g.n} field-years on one farm, ${g.silage} of them silage. ${g.low ? `${g.low} calls are flagged low evidence. ` : ""}The truth comes from the organisers.</p>`;
}

$("intro").addEventListener("click", () => showIntro(false));
$("brand").addEventListener("click", () => showIntro($("intro").hidden));

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
  showIntro(false);
  loadRegion(b.dataset.region).then(writeHash);
});

// ---------------------------------------------------------------- metrics

function metricTile(label, value, detail, title = "") {
  const na = value === DASH;
  return `<div class="metric" title="${esc(title)}"><div class="label">${label}</div><div class="value${na ? " na" : ""}">${value}</div><div class="detail">${detail || "&nbsp;"}</div></div>`;
}

function renderMetrics() {
  const meta = state.meta;
  if (!meta) return;
  const m = currentMetrics() || {};
  const a = m.all;
  const sub = [`${int(m.n_fields)} fields`];
  if (meta.year) sub.unshift(String(meta.year));
  const noAcc = !meta.has_predictions ? "no predictions" : "no declared labels";
  const tiles = [
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
    const parts = [`Silage = positive class \u00b7 scored on <b>${int(a.n)}</b> fields with ${esc((meta.label_name || "declared").toLowerCase())}`];
    if (m.always_grain) parts.push(`Accuracy <b>${pct(a.accuracy)}</b>; "always grain" would score <b>${pct(m.always_grain.accuracy)}</b> here and find no silage at all`);
    if (c && c.n !== a.n) parts.push(`Confident fields only (${int(c.n)}): precision <b>${pct(c.precision)}</b>, recall <b>${pct(c.recall)}</b>, FPR <b>${pct(c.fpr)}</b>`);
    if (state.group && !state.group.synthetic) parts.unshift(`<span class="group-line">All ${state.group.years.length} years: <b>${state.group.right} of ${state.group.n}</b> right \u00b7 always grain ${state.group.grain} of ${state.group.n}</span>`);
    if (meta.note) parts.push(esc(meta.note));
    foot = `<div class="foot">${parts.join("<br>")}</div>`;
  }
  const asofChip = asofActive() ? `<span class="asof-chip">as of ${shortDate(cutoffIso())}</span>` : "";
  $("metrics").innerHTML = `<div class="head"><h2>${esc(meta.title)} ${asofChip}</h2><div class="sub">${sub.join(" · ")}</div></div>
    <div class="metric-grid">${tiles.join("")}</div><div class="metric-grid small">${tiles2.join("")}</div>${foot}`;
}

// ---------------------------------------------------------------- legend

function renderLegend() {
  const el = $("legend");
  const meta = state.meta;
  if (!meta) { el.innerHTML = ""; return; }
  const f = state.view;
  const count = (pred) => f.filter(pred).length;
  let rows;
  if (state.mode === "check") {
    rows = [
      [`<span class="sw" style="background:${COLORS.ok}"></span>`, `Matches ${esc((meta.label_name || "declared").toLowerCase())}`, count((p) => p.correct === true)],
      [`<span class="sw" style="background:${COLORS.wrong}"></span>`, "Model wrong", count((p) => p.correct === false)],
      [`<span class="sw hollow"></span>`, "No label or no prediction", count((p) => p.correct == null)],
    ];
  } else {
    rows = [
      [`<span class="sw" style="background:${COLORS.silage}"></span>`, "Silage (chopped)", count((p) => p.pred === "silage" && p.low !== true)],
      [`<span class="sw" style="background:${COLORS.grain}"></span>`, "Grain (combined)", count((p) => p.pred === "grain" && p.low !== true)],
      [`<span class="sw dashed" style="border-color:${COLORS.silage}"></span>`, "Silage, abstains (low evidence)", count((p) => p.pred === "silage" && p.low === true)],
      [`<span class="sw dashed" style="border-color:${COLORS.grain}"></span>`, "Grain, abstains (low evidence)", count((p) => p.pred === "grain" && p.low === true)],
      [`<span class="sw miss-sw"></span>`, `Miss: ${esc((meta.label_name || "declared label").toLowerCase())} differs`, count((p) => p.correct === false)],
      [`<span class="sw hollow"></span>`, "No prediction", count((p) => !p.pred)],
    ].filter((r) => (meta.has_predictions ? (r[2] > 0 || (r[1] !== "No prediction" && !r[1].startsWith("Miss"))) : r[1] === "No prediction"));
  }
  const title = state.mode === "check" ? `Model vs ${esc((meta.label_name || "declared label").toLowerCase())}` : "Predicted harvest use";
  let html = `<h4>${title}</h4>` +
    rows.map(([sw, name, n]) => `<div class="row">${sw}<span>${name}</span><span class="count">${n}</span></div>`).join("");
  if (state.tab === "counties" && state.countyShapes && !state.selected) {
    html = `<h4>County: our silage share</h4><div class="ramp">${RAMP.map((c) => `<span style="background:${c}"></span>`).join("")}</div>
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
  if (!state.meta.has_predictions) {
    return `<p class="empty">No predictions yet, so nothing to rank.<br>Click any field on the map to see its satellite curves.</p>`;
  }
  const ranked = state.view.filter((p) => isNum(p.p)).sort((a, b) => certainty(a) - certainty(b)).slice(0, LIST_LEN);
  if (!ranked.length) return `<p class="empty">No field has a probability to rank${asofActive() ? " at this date" : ""}.</p>`;
  const area = ranked.reduce((s, p) => s + (p.area_ha || 0), 0);
  const rows = ranked.map((p, i) => {
    const bits = [ha(p.area_ha)];
    const tr = traps(p, state.meta);
    if (tr.length) bits.push(tr.map((t) => `<span class="tag">${esc(t.tag)}</span>`).join(" "));
    else if (p.low === true) bits.push(`<span class="tag">low evidence</span>`);
    if (p.correct === false) bits.push(`<span class="tag wrong">wrong</span>`);
    else if (p.label) bits.push(`${esc((state.meta.label_name || "declared").toLowerCase())}: ${p.label}`);
    return itemRow(p, i, bits.join(" · "));
  }).join("");
  return `<p class="intro">The ${ranked.length} least certain predictions${asofActive() ? ` as of ${shortDate(cutoffIso())}` : ""}: where a survey call or field visit tells us the most (${ha(area)} in total).</p><ol class="list">${rows}</ol>`;
}

function renderWrongList() {
  const wrong = state.view.filter((p) => p.correct === false).sort((a, b) => (confidence(b) ?? 0) - (confidence(a) ?? 0));
  if (!wrong.length) return `<p class="empty">The model agrees with every declared label here.</p>`;
  const rows = wrong.map((p, i) => {
    const bits = [`${esc((state.meta.label_name || "declared").toLowerCase())}: <b>${p.label}</b>`];
    if (p.hs && p.he) bits.push(`cut ${shortDate(p.hs)}–${shortDate(p.he)}`);
    if (p.low === true) bits.push(`<span class="tag">low evidence</span>`);
    return itemRow(p, i, bits.join(" · "));
  }).join("");
  return `<p class="intro">${wrong.length} field${wrong.length === 1 ? "" : "s"} where the ${esc((state.meta.label_name || "declared label").toLowerCase())} disagrees with the model. Most confident mistakes first.</p><ol class="list">${rows}</ol>`;
}

function renderCounties() {
  const c = state.counties;
  if (!c || !c.rows?.length) return `<p class="empty">No county table for this region.</p>`;
  const s = c.summary || {};
  const width = Math.max(280, $("view").clientWidth - 32);
  const skipped = c.rows.length - c.rows.filter((r) => isNum(r.pred_silage_share) && isNum(r.usda_silage_share)).length;
  const asofNote = asofActive() ? `<div class="notice">County totals use the full-season predictions; the date slider does not change them.</div>` : "";
  return `${asofNote}<p class="intro">Sanity check: do our totals match NASS? Our acres next to the acres USDA NASS reports, with the gap stated plainly.</p>
    ${stateCardsHtml(c.states)}
    <h4 class="section">By county</h4>
    ${countyTableHtml(c.rows, state.county)}
    <h4 class="section">Silage share by county</h4>
    <div class="county-summary">
      <div class="metric"><div class="label">Counties</div><div class="value">${int(s.n_compared)}</div><div class="detail">of ${int(s.n_counties)} compared</div></div>
      <div class="metric" title="Mean absolute difference between our and USDA's silage share"><div class="label">Mean abs. diff.</div><div class="value${isNum(s.mae) ? "" : " na"}">${isNum(s.mae) ? `${(s.mae * 100).toFixed(0)} pts` : DASH}</div><div class="detail">percentage points</div></div>
      <div class="metric" title="Pearson correlation across counties"><div class="label">Correlation</div><div class="value${isNum(s.r) ? "" : " na"}">${fixed(s.r, 2)}</div><div class="detail">Pearson r</div></div>
    </div>
    <div class="scatter" id="scatter">${scatterSvg(c.rows, { width, height: Math.round(width * 0.72), highlight: state.county })}</div>
    ${skipped ? `<p class="intro">${skipped} count${skipped === 1 ? "y" : "ies"} without predicted fields or USDA numbers not plotted.</p>` : ""}
    <p class="intro" style="margin-top:0.7rem">Our acres: area of the fields we call silage or grain (all predictions, low-evidence ones included). NASS: corn acres harvested for silage and for grain.</p>`;
}

function probBar(p) {
  if (!isNum(p.p)) return "";
  const pos = Math.max(0, Math.min(100, p.p * 100));
  return `<div class="pbar${p.low === true ? " low" : ""}" title="Probability of silage">
    <div class="pbar-track"><span class="mid"></span><span class="mark" style="left:${pos}%"></span></div>
    <div class="pbar-labels"><span>← Grain</span><span>p(silage) ${p.p.toFixed(2)}${p.low === true ? " · low evidence" : ""}</span><span>Silage →</span></div>
  </div>`;
}

function defendHtml(p) {
  const meta = state.meta;
  if (!p.pred) return "";
  if (asofActive() && p.asofHidden) {
    return `<section class="defend"><h4>Explainable to a skeptic</h4><p class="muted">As of ${shortDate(cutoffIso())} no harvest had been seen yet, so this early call rests on the curve so far (a field still standing late in the season leans grain). Move the slider on to see the evidence.</p></section>`;
  }
  const rows = defendRows(p, meta);
  const t = tally(rows);
  const word = (c) => ({ silage: "silage-like", grain: "grain-like", mixed: "in between" }[c] || "not seen");
  const rowHtml = rows.map((r) => `<div class="ev ${r.cue || "unknown"}">
      <div class="ev-head"><span class="ev-title">${r.title}</span><span class="cue ${r.cue || "unknown"}">${word(r.cue)}</span></div>
      <div class="ev-value">${esc(r.value)}${r.detail ? ` <small>${esc(r.detail)}</small>` : ""}</div>
      <div class="ev-text">${esc(r.sentence)} <span class="ev-chart">see ${r.chart}</span></div>
    </div>`).join("");
  const other = p.pred === "silage" ? "grain" : "silage";
  const sig = rows.signals;
  const summary = sig
    ? `${sig.silage} of ${sig.seen} signals look like silage`
    : `${t[p.pred]} of 3 point to ${p.pred}${t[other] ? `, ${t[other]} to ${other}` : ""}${t.unknown ? `, ${t.unknown} not seen` : ""}`;
  const model = p.explanation ? `<p class="model-says"><b>The model's own summary:</b> ${esc(plainModelText(p.explanation))}</p>` : "";

  const flip = flipConditions(p, meta);
  let flipHtml = "";
  if (p.flip || (flip && (flip.needed.length || flip.already.length))) {
    const lines = [];
    if (p.flip) lines.push(`<p class="model-says"><b>Biggest pull the other way:</b> ${esc(plainModelText(p.flip))}</p>`);
    if (flip?.already.length) lines.push(`<p>Already pointing to ${flip.other}: ${flip.already.join("; ")}.</p>`);
    if (flip?.needed.length) lines.push(hasModelEvidence(p, meta)
      ? `<p>Still pointing to ${p.pred}, closest to flipping first: ${flip.needed.join("; ")}.</p>`
      : `<p>A ${flip.other} call would also expect ${flip.needed.join("; ")}.</p>`);
    flipHtml = `<div class="flip"><h5>What would have made it ${flip?.other || other}</h5>${lines.join("")}
      <p class="fine">${hasModelEvidence(p, meta)
        ? "Typical values are the medians of silage and grain fields the model was trained on; a signal points to whichever is closer, as in the model's own summary."
        : `Lines from the display thresholds behind these sentences${meta.harvest_ref ? " and this region's typical cutting dates" : ""}, not the model's internal rule.`}</p></div>`;
  }
  return `<section class="defend"><h4>Explainable to a skeptic <span class="tally">${summary}</span></h4>
    <p class="slogan">Silage falls off a cliff. Grain fades out.</p>${rowHtml}${model}${flipHtml}</section>`;
}

/** The three requirements, answered for this one field. */
function yardstickStrip(p) {
  const data = `Sentinel-2 \u00b7 ${int(p.n_usable)} clear of ${int(p.n_obs)} looks`;
  let early = null;
  if (state.asof) {
    const full = state.byId.get(p.field_id);
    const s = settledFrom(p.field_id, state.asof, full, state.meta.threshold ?? 0.5);
    if (s) early = s.iso ? `Call settled by ${shortDate(s.iso)}` : "Settled only with the full season";
  }
  const chips = [
    ["Public data", data],
    early ? ["Earlier than January", early] : null,
    ["Skeptic", "3 checks below"],
  ].filter(Boolean);
  return `<div class="ystrip">${chips.map(([h, t]) => `<div><span>${h}</span>${esc(t)}</div>`).join("")}</div>`;
}

function renderDetail(p) {
  const meta = state.meta;
  const year = meta.year;
  const s = state.series === null ? undefined : state.series[p.field_id];
  const conf = confidence(p);
  const head = `<div class="detail-head">
      <div><h3>${esc(p.name || p.field_id)}</h3><div class="sub">${p.name ? `${esc(p.field_id)} \u00b7 ` : ""}${ha(p.area_ha)}${p.block_id ? ` · block ${esc(p.block_id)}` : ""} · ${int(p.n_usable)} clear of ${int(p.n_obs)} dates${isNum(p.max_gap) ? `, longest gap ${p.max_gap} d` : ""}</div></div>
      <button type="button" class="back" data-action="back" title="Back to the list (Esc)">← Back</button>
    </div>`;
  let asofLine = "";
  if (asofActive()) {
    const full = state.byId.get(p.field_id);
    asofLine = `<div class="asof-line">Call with data up to <b>${shortDate(cutoffIso())}</b>${full?.pred ? ` · full season: ${cap(full.pred)} ${pct(confidence(full))}` : ""}</div>`;
  }
  const verdict = p.pred
    ? `<div class="verdict"><span class="big ${p.pred}">${cap(p.pred)} <small>${p.pred === "silage" ? "chopped" : "combined"}</small></span><span class="prob">${pct(conf)} confident</span></div>${probBar(p)}`
    : `<div class="verdict"><span class="big none">No prediction${asofActive() ? " at this date" : " yet"}</span></div>`;
  let lowBox = "";
  const tr = asofActive() && p.asofHidden ? [] : traps(p, meta);
  if (p.low === true || tr.length) {
    const lines = asofActive() && p.asofHidden ? ["The harvest has not been seen yet at this date."] : tr.map((x) => x.text);
    if (!lines.length) lines.push(p.reason || "The model flagged this call as low evidence.");
    const flag = p.reason && !lines.includes(p.reason) ? `<div class="fine">Model's flag: ${esc(plainModelText(p.reason))}</div>` : "";
    const title = p.low === true ? "Why this is low evidence" : "Traps to keep in mind";
    lowBox = `<div class="lowbox"><b>${title}</b><ul>${lines.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>${flag}</div>`;
  }
  const strip = p.pred ? yardstickStrip(p) : "";
  let declared;
  const ln = state.meta.label_name || "Declared";
  if (!p.label) declared = `<div class="declared">No ${esc(ln.toLowerCase())} for this field.</div>`;
  else if (p.correct === false) declared = `<div class="declared">${esc(ln)}: <b>${p.label}</b> <span class="miss">\u2717 model says ${p.pred}</span></div>`;
  else if (p.correct === true) declared = `<div class="declared">${esc(ln)}: <b>${p.label}</b> <span class="match">\u2713 model is right</span></div>`;
  else declared = `<div class="declared">${esc(ln)}: <b>${p.label}</b></div>`;

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
    const common = { width, year, domain, series: s, band, gapDays: meta.gap_days ?? 20, cutoff: cutoffDay() };
    const noClear = p.n_usable === 0 ? `<div class="chart-empty">No cloud-free observation all season: every acquisition below is cloudy or excluded, so nothing is plotted as a curve.</div>` : "";
    const hasNdti = Array.isArray(s.ndti) && s.ndti.some(isNum);
    // with the model's evidence, the line sits halfway between its typical silage and grain values
    const rr = meta.evidence_ref?.post_ndti;
    const res = rr && isNum(rr.silage) && isNum(rr.grain) ? (rr.silage + rr.grain) / 2 : cuesOf(meta).residue_ndti;
    charts = noClear +
      `<div class="chart">${seasonChart({ ...common, height: 165, key: "ndvi", label: "Greenness (NDVI)", color: "#8fd17f", yMin: -0.1, yMax: 1, yTicks: [0, 0.2, 0.4, 0.6, 0.8, 1],
        pre: isNum(hs) ? { t: hs, v: p.pre_ndvi } : null, post: isNum(he) ? { t: he, v: p.post_ndvi } : null })}</div>` +
      `<div class="chart">${seasonChart({ ...common, height: 150, key: "ndmi", label: "Leaf moisture (NDMI)", color: "#5ec8f2", yMin: -0.4, yMax: 0.6, yTicks: [-0.4, -0.2, 0, 0.2, 0.4, 0.6],
        pre: isNum(hs) ? { t: hs, v: p.pre_ndmi } : null })}</div>` +
      (hasNdti ? `<div class="chart">${seasonChart({ ...common, height: 140, key: "ndti", label: "Ground residue (NDTI)", color: "#d9b38c", yMin: -0.1, yMax: 0.4, yTicks: [0, 0.1, 0.2, 0.3, 0.4],
        post: isNum(he) ? { t: he, v: p.post_ndti } : null, refLine: isNum(res) ? { v: res, label: "stalk residue above \u00b7 bare soil below" } : null })}</div>` : "") +
      chartLegendHtml("#cfd8e3", meta.min_clear);
  }
  return head + asofLine + verdict + declared + strip + lowBox + defendHtml(p) + charts;
}

function renderView() {
  const view = $("view");
  if (!state.meta) { view.innerHTML = ""; return; }
  const p = state.selected && state.viewById.get(state.selected);
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
  renderTimeline();
  fieldMap.showCounties(state.tab === "counties" && !state.selected);
}

// ---------------------------------------------------------------- interactions

function selectField(id) {
  if (!state.byId.has(id)) return;
  showIntro(false);
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
      b.textContent = /truth/i.test(state.meta?.label_name || "") ? "Check vs truth" : "Check vs declared";
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
  if (!$("intro").hidden && ["Escape", "Enter", " "].includes(e.key)) { e.preventDefault(); showIntro(false); return; }
  if (e.key === "Escape") closeField();
});

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { fieldMap.invalidate(); if (state.meta) { renderView(); renderTimeline(); } }, 150);
});

boot();
