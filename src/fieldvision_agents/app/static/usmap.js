// US map: states shaded by USDA's 2025 silage share; a state opens its counties shaded by the Census
// share; a county with field data opens every field's call on satellite imagery. Dairies and place
// labels are layers on top. Picking a county shows everything the pipeline has on it.
import { lineChart } from "./charts.js";
import { STATUS, api, el, fmt } from "./ui.js";

const BINS = [2, 5, 10, 20];                          // silage share of harvested corn, %
const BIN_LABELS = ["under 2%", "2–5%", "5–10%", "10–20%", "20% or more"];
const US = [[24.5, -125], [49.5, -66.5]];
// Esri basemaps need no key: a gray canvas for shading, imagery for fields, and labels drawn on top of both.
const ESRI_TILE = (svc) => `https://server.arcgisonline.com/ArcGIS/rest/services/${svc}/MapServer/tile/{z}/{y}/{x}`;
const ESRI = ESRI_TILE("World_Imagery");
let topo = null;

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const binOf = (v) => { const i = BINS.findIndex((b) => v < b); return i === -1 ? BINS.length : i; };
const ramp = (v) => cssVar(`--q${binOf(v) + 1}`);
const light = () => matchMedia("(prefers-color-scheme: light)").matches;

async function shapes() {
  if (!topo) topo = await (await fetch("/vendor/us-atlas/counties-10m.json")).json();
  return { states: window.topojson.feature(topo, topo.objects.states), counties: window.topojson.feature(topo, topo.objects.counties) };
}

function table(headers, rows) {
  return el("div", { class: "scroll-x" }, el("table", { class: "data" },
    el("thead", {}, el("tr", {}, headers.map((h, i) => el("th", { class: i ? "num" : "" }, h)))),
    el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c, i) => el("td", { class: i ? "num" : "" }, c)))))));
}

// --- side panels -----------------------------------------------------------------------

function nationPanel(states) {
  const rows = Object.entries(states).filter(([, s]) => s.silage).sort((a, b) => b[1].silage - a[1].silage).slice(0, 8);
  return el("div", { class: "card" }, el("h2", {}, "Corn silage across the US, 2025"),
    el("p", { class: "muted" }, "States are shaded by USDA's 2025 silage share of harvested corn. Pick a state for its counties (Census), then a county for its data. Orange outlines mark field data on this machine."),
    el("h3", {}, "Most silage acres (NASS 2025)"),
    table(["State", "Silage ac", "Share"], rows.map(([, s]) => [s.name, fmt(s.silage, 0), `${s.silage_share_pct}%`])));
}

function statePanel(s, pickCounty, allStates) {
  const n = s.nass_2025, survey = s.history.filter((h) => h.silage_share_pct != null);
  return el("div", { class: "card" },
    el("div", { class: "row between" }, el("h2", {}, s.name), el("button", { class: "small ghost", onclick: allStates }, "← All states")),
    n ? el("div", { class: "kv" },
      el("div", {}, el("div", { class: "k" }, "Silage, 2025 (NASS)"), el("div", { class: "v silage-t" }, `${fmt(n.silage, 0)} ac`)),
      el("div", {}, el("div", { class: "k" }, "Grain, 2025 (NASS)"), el("div", { class: "v grain-t" }, `${fmt(n.grain, 0)} ac`)),
      el("div", {}, el("div", { class: "k" }, "Silage share"), el("div", { class: "v" }, n.silage_share_pct == null ? "—" : `${n.silage_share_pct}%`))) : null,
    s.ours.length ? el("div", {}, el("h3", {}, "Our estimate vs USDA"),
      table(["Run", "Ours", "USDA"], s.ours.map((o) => [o.what, `${o.ours_pct}% · ${fmt(o.ours_acres, 0)} ac`, `${o.usda_pct}% · ${fmt(o.usda_acres, 0)} ac`])),
      el("p", { class: "small faint" }, "Model estimates, not NASS statistics; state totals are not validated yet.")) : null,
    survey.length > 1 ? lineChart({ title: "Silage share of harvested corn, state survey", points: survey.map((h) => ({ x: h.year, y: h.silage_share_pct })),
      yFormat: (v, f) => (f ? `${v.toFixed(1)}%` : `${v}%`), color: "var(--silage)" }) : null,
    s.fields_on_disk.length ? el("div", {}, el("h3", {}, "Counties with field data"),
      el("div", { class: "chips" }, s.fields_on_disk.map((f) => el("button", { class: "chip", onclick: () => pickCounty(f) }, f)))) : null,
    el("p", { class: "small muted" }, `${Object.keys(s.counties).length} counties have a Census silage share. Click a county.`));
}

function countyPanel(c, app, back, fieldInfo) {
  const census = c.history.filter((h) => h.source === "CENSUS"), survey = c.history.filter((h) => h.source === "SURVEY");
  const ex = c.extras || {}, check = ex.county_check_2022;
  const useSeason = (s, then) => { app.ctx.fips = s.fips; app.ctx.year = Number(s.year); app.ctx.run_id = null;
    if (!String(app.ctx.as_of || "").startsWith(String(s.year))) app.ctx.as_of = `${s.year}-09-01`; app.save(); app.refreshSide(); then?.(); };
  const fc = fieldInfo?.counts, judged = fc ? fc.silage + fc.grain : 0;
  return el("div", { class: "card" },
    el("div", { class: "row between" }, el("h2", {}, c.name), el("button", { class: "small ghost", onclick: back }, "← State")),
    el("p", { class: "small faint" }, `FIPS ${c.fips}`),
    fc ? el("div", {}, el("h3", {}, `Our field calls · ${fieldInfo.year} · ${fieldInfo.model}`),
      el("div", { class: "kv" },
        el("div", {}, el("div", { class: "k" }, "Called silage"), el("div", { class: "v silage-t" }, fmt(fc.silage, 0))),
        el("div", {}, el("div", { class: "k" }, "Called grain"), el("div", { class: "v grain-t" }, fmt(fc.grain, 0))),
        el("div", {}, el("div", { class: "k" }, "Not judged"), el("div", { class: "v" }, fmt(fc["not judged"], 0))),
        el("div", {}, el("div", { class: "k" }, "Silage share of fields judged"), el("div", { class: "v" }, judged ? `${fmt((100 * fc.silage) / judged)}%` : "—"))),
      el("p", { class: "small faint" }, "Fields counted, not acres; the county headline in a run counts clearly seen area only. Hover a field on the map for its evidence.")) : null,
    census.length ? el("div", {}, el("h3", {}, "Corn harvested, USDA Census"),
      table(["Year", "Grain ac", "Silage ac", "Silage share"], census.map((h) => [h.year, fmt(h.grain, 0), h.silage == null ? "withheld" : fmt(h.silage, 0),
        h.silage_share_pct == null ? "—" : `${h.silage_share_pct}%`]))) : el("p", { class: "muted" }, "No Census corn rows for this county."),
    survey.length ? el("p", { class: "small muted" }, `Survey grain acres: ${survey.map((h) => `${h.year} ${fmt(h.grain, 0)}`).join(" · ")}`) : null,
    c.prior_2026 ? el("p", { class: "small" }, "The pipeline's prior for this county: ", el("b", {}, `${c.prior_2026.share_pct}%`), ` silage (${c.prior_2026.source}).`) : null,
    el("h3", {}, "Seasons on this machine"),
    c.seasons.length ? el("div", {}, c.seasons.map((s) => el("div", { class: "row between", style: "margin:5px 0" },
      el("span", {}, `${s.year} season${s.partial ? " (partial)" : ""}`),
      el("span", { class: "row" },
        el("button", { class: "small", onclick: () => useSeason(s, () => app.goTab("chat")) }, "Ask about it"),
        app.role === "usda" ? el("button", { class: "small primary", onclick: () => useSeason(s, () => app.goTab("runs")) }, "Run the agent")
          : el("button", { class: "small primary", onclick: () => useSeason(s, () => app.goTab("map")) }, "Find silage")))))
      : el("p", { class: "small muted" }, "No field outlines or satellite series for this county here. The Census figures above are USDA's."),
    c.runs.length ? el("div", {}, el("h3", {}, "Agent runs"), el("ul", { class: "runs" }, c.runs.map((r) => el("li", {},
      el("button", { onclick: () => app.openRun(r.run_id) }, el("span", { class: "when" }, `${r.when} · ${STATUS[r.status] || r.status}${r.offline ? " · offline" : ""}`),
        `${r.year} · ${r.policies.join(" → ") || "no classify"}`))))) : null,
    check ? el("div", {}, el("h3", {}, "Every field against the Census, 2022"),
      table(["Model", "Fields", "Called silage", "Census"], check.rows.map((r) => [r.model, fmt(r.fields, 0), `${r.called_silage_pct}%`, `${check.census_silage_pct}%`])),
      el("p", { class: "small muted" }, `Final module headline: ${fmt(100 * check.module_v2.silage_share_headline)}% on ${fmt(check.module_v2.clear_tier_fields, 0)} clearly seen fields.`)) : null,
    ex.rotation_2022 ? el("details", {}, el("summary", {}, "Crop history of these fields (2022)"),
      table(["Group", "Fields", "Hay/alfalfa", "Soybeans", "Wheat"], ex.rotation_2022.map((r) => [r.group, fmt(r.fields, 0),
        `${fmt(100 * r.alfalfa_or_hay_fields, 0)}%`, `${fmt(100 * r.soybeans_fields, 0)}%`, `${fmt(100 * r.wheat_or_wheat_double_fields, 0)}%`]))) : null,
    ex.landscape_2022 ? el("details", {}, el("summary", {}, "Hay and forage within 5 km (2022)"),
      table(["Group", "Fields", "Forage share, median"], ex.landscape_2022.map((r) => [r.group, fmt(r.fields, 0), `${fmt(100 * r.forage_share_5km_year_p50)}%`]))) : null);
}

// --- the map ---------------------------------------------------------------------------

function legendControl() {
  const ctl = L.control({ position: "bottomright" });
  ctl.onAdd = () => { ctl.node = L.DomUtil.create("div", "map-legend"); L.DomEvent.disableClickPropagation(ctl.node); return ctl.node; };
  ctl.set = (mode) => {
    const share = `<b>Silage share</b> (${mode === "nation" ? "NASS 2025" : "Census"})<br>` +
      BIN_LABELS.map((l, i) => `<i style="background:${cssVar(`--q${i + 1}`)}"></i>${l}`).join("<br>");
    const fields = mode === "fields" ? `<br><b>Field calls</b><br><i style="background:${cssVar("--silage")}"></i>silage<br>`
      + `<i style="background:${cssVar("--grain")}"></i>grain<br><i style="background:${cssVar("--abstain")}"></i>not judged` : "";
    ctl.node.innerHTML = share + fields + `<br><i class="dot" style="background:${cssVar("--usda")}"></i>permitted dairy`
      + `<br><i style="border:2px solid ${cssVar("--gold")};background:transparent"></i>field data here`;
  };
  return ctl;
}

export async function usMapPage(app) {
  const panel = el("div", { class: "usmap-panel" }, el("div", { class: "card" }, el("p", { class: "muted" }, "Loading the map…")));
  const mapNode = el("div", { class: "usmap" });
  const page = el("div", { class: "page usmap-page" }, el("div", { class: "usmap-grid" }, mapNode, panel));
  const [{ states, counties }, stateShares, dairies] = await Promise.all([shapes(), api("/api/geo/states"), api("/api/geo/dairies").catch(() => null)]);
  panel.replaceChildren(nationPanel(stateShares));

  requestAnimationFrame(() => {
    const map = L.map(mapNode, { zoomSnap: 1, scrollWheelZoom: true });   // whole zooms keep label tiles crisp
    map.createPane("labels"); map.getPane("labels").style.zIndex = 650; map.getPane("labels").style.pointerEvents = "none";
    const tone = light() ? "Light" : "Dark";
    const street = L.tileLayer(ESRI_TILE(`Canvas/World_${tone}_Gray_Base`), { maxZoom: 16, attribution: "Basemap © Esri" });
    const satellite = L.tileLayer(ESRI, { maxZoom: 18, attribution: "Imagery © Esri" });
    const labels = L.tileLayer(ESRI_TILE("Reference/World_Boundaries_and_Places"), { maxZoom: 18, pane: "labels", opacity: 0.85 });
    street.addTo(map); labels.addTo(map);
    map.fitBounds(US);
    const line = cssVar("--line"), gold = cssVar("--gold"), text = cssVar("--text");
    const legend = legendControl(); legend.addTo(map); legend.set("nation");

    const dairyLayer = dairies ? L.geoJSON(dairies, {
      pointToLayer: (f, ll) => L.circleMarker(ll, { radius: Math.min(9, 4 + Math.sqrt(f.properties.head || 100) / 10), color: "#04140e", weight: 1.5,
        fillColor: cssVar("--usda"), fillOpacity: 0.95 }),
      onEachFeature: (f, l) => l.bindTooltip(`<b>${f.properties.dairy_id}</b><br>${f.properties.county || ""} ${f.properties.state || ""}<br>${f.properties.head ? `${fmt(f.properties.head, 0)} head` : "herd size unknown"}`),
    }) : null;
    const layersCtl = L.control.layers({ "Map": street, "Satellite": satellite }, { "Place labels": labels, ...(dairyLayer ? { "Permitted dairies": dairyLayer } : {}) },
      { position: "topright", collapsed: true }).addTo(map);
    dairyLayer?.addTo(map);

    let countyLayer = null, fieldLayer = null, selectedState = null, stateData = null;
    const stateLayer = L.geoJSON(states, {
      style: (f) => { const s = stateShares[f.id]; return {
        color: s?.fields_on_disk ? gold : line, weight: s?.fields_on_disk ? 2.5 : 0.8,
        fillColor: s?.silage_share_pct != null ? ramp(s.silage_share_pct) : "transparent", fillOpacity: s ? 0.9 : 0 }; },
      onEachFeature: (f, l) => {
        const s = stateShares[f.id];
        l.bindTooltip(`<b>${f.properties.name}</b>${s ? `<br>${s.silage_share_pct}% silage (NASS 2025)<br>${fmt(s.silage, 0)} silage · ${fmt(s.grain, 0)} grain ac` : ""}`, { sticky: true });
        l.on("click", () => pickState(f.id, l.getBounds()));
        l.on("mouseover", () => !selectedState && l.setStyle({ weight: 2.5, color: text }));
        l.on("mouseout", () => !selectedState && stateLayer.resetStyle(l));
      },
    }).addTo(map);

    function clearFields() {
      if (fieldLayer) { fieldLayer.remove(); layersCtl.removeLayer(fieldLayer); fieldLayer = null; }
    }

    async function showFields(fips, seasons) {
      clearFields();
      if (!seasons.length) return null;
      const s = seasons.find((x) => Number(x.year) === Number(app.ctx.year)) || seasons[seasons.length - 1];
      const fc = await api(`/api/geo/fields/${fips}/${s.year}`);
      const color = { silage: cssVar("--silage"), grain: cssVar("--grain"), "not judged": cssVar("--abstain") };
      fieldLayer = L.geoJSON(fc, {
        style: (f) => ({ color: color[f.properties.call], weight: 1, fillColor: color[f.properties.call], fillOpacity: 0.62 }),
        onEachFeature: (f, l) => { const p = f.properties;
          l.bindTooltip(`<b>${p.call}</b> · P(silage) ${p.p_silage}<br>${fmt(p.acres)} acres${p.drop_date ? ` · cut seen ${p.drop_date}` : ""}${p.reason ? `<br>not judged: ${p.reason.replaceAll("_", " ")}` : ""}`, { sticky: true }); },
      }).addTo(map);
      layersCtl.addOverlay(fieldLayer, `Field calls ${s.year}`);
      map.removeLayer(street); satellite.addTo(map);                 // fields read best on the imagery
      legend.set("fields");
      return { ...fc, year: s.year };
    }

    async function pickCounty(fips) {
      panel.replaceChildren(el("div", { class: "card" }, el("p", { class: "muted" }, "Loading the county…")));
      let target = null;
      countyLayer?.eachLayer((l) => {
        const on = l.feature.id === fips, has = stateData?.fields_on_disk.includes(l.feature.id);
        l.setStyle({ weight: on ? 3 : has ? 2 : 0.5, color: on ? text : has ? gold : line, fillOpacity: on && has ? 0 : 0.72 });
        if (on) target = l;
      });
      if (target) map.fitBounds(target.getBounds(), { padding: [16, 16] });
      try {
        const c = await api(`/api/geo/county/${fips}`);
        const fieldInfo = await showFields(fips, c.seasons).catch(() => null);
        if (fieldInfo) map.fitBounds(fieldLayer.getBounds(), { padding: [12, 12] });
        panel.replaceChildren(countyPanel(c, app, () => { clearFields(); satellite.remove(); street.addTo(map); legend.set("state");
          countyLayer?.eachLayer((l) => countyLayer.resetStyle(l)); map.fitBounds(stateBounds, { padding: [10, 10] });
          panel.replaceChildren(statePanel(stateData, pickCounty, allStates)); }, fieldInfo));
      } catch (e) { panel.replaceChildren(el("div", { class: "card" }, el("p", { class: "err" }, e.message))); }
    }

    let stateBounds = null;
    async function pickState(sf, bounds) {
      selectedState = sf; stateBounds = bounds;
      map.fitBounds(bounds, { padding: [10, 10] });
      stateLayer.setStyle((f) => ({ fillOpacity: f.id === sf ? 0 : 0.25, weight: 0.8, color: line }));
      panel.replaceChildren(el("div", { class: "card" }, el("p", { class: "muted" }, "Loading the state…")));
      try { stateData = await api(`/api/geo/state/${sf}`); }
      catch (e) { panel.replaceChildren(el("div", { class: "card" }, el("p", { class: "err" }, e.message))); return; }
      countyLayer?.remove(); clearFields();
      countyLayer = L.geoJSON({ type: "FeatureCollection", features: counties.features.filter((f) => f.id.startsWith(sf)) }, {
        style: (f) => {
          const v = stateData.counties[f.id], has = stateData.fields_on_disk.includes(f.id);
          return { fillColor: v ? ramp(v.share) : "transparent", fillOpacity: v ? 0.72 : 0,
            color: has ? gold : line, weight: has ? 2 : 0.5, dashArray: v ? null : "2 3" };
        },
        onEachFeature: (f, l) => {
          const v = stateData.counties[f.id];
          l.bindTooltip(`<b>${f.properties.name}</b><br>${v ? `${v.share}% silage (Census ${v.year})<br>${fmt(v.silage, 0)} silage · ${fmt(v.grain, 0)} grain ac` : "silage withheld or no corn"}`
            + (stateData.fields_on_disk.includes(f.id) ? "<br><b>field data here</b>" : ""), { sticky: true });
          l.on("click", () => pickCounty(f.id));
        },
      }).addTo(map);
      dairyLayer?.bringToFront?.();
      legend.set("state");
      panel.replaceChildren(statePanel(stateData, pickCounty, allStates));
    }

    function allStates() {
      selectedState = null; stateData = null;
      countyLayer?.remove(); countyLayer = null; clearFields();
      satellite.remove(); street.addTo(map); legend.set("nation");
      stateLayer.eachLayer((l) => stateLayer.resetStyle(l));
      map.fitBounds(US);
      panel.replaceChildren(nationPanel(stateShares));
    }

    if (app.ctx.fips) {        // open on the season already chosen
      const sf = String(app.ctx.fips).slice(0, 2);
      stateLayer.eachLayer((l) => { if (l.feature.id === sf) pickState(sf, l.getBounds()).then(() => pickCounty(String(app.ctx.fips))); });
    }
    setTimeout(() => map.invalidateSize(), 60);
  });
  return page;
}
