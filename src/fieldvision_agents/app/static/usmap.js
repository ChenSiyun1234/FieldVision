// US map picker: every state; pick one to see its counties shaded by the Census silage share; pick a county
// to see everything the pipeline has on it, and use one of its seasons on the rest of the page.
import { lineChart } from "./charts.js";
import { STATUS, api, el, fmt } from "./ui.js";

const BINS = [2, 5, 10, 20];                          // silage share of harvested corn, % (Census)
const BIN_LABELS = ["under 2%", "2–5%", "5–10%", "10–20%", "20% or more"];
let topo = null;

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const binOf = (v) => BINS.findIndex((b) => v < b) === -1 ? BINS.length : BINS.findIndex((b) => v < b);

async function shapes() {
  if (!topo) topo = await (await fetch("/vendor/us-atlas/counties-10m.json")).json();
  return {
    states: window.topojson.feature(topo, topo.objects.states),
    counties: window.topojson.feature(topo, topo.objects.counties),
  };
}

function legend() {
  return el("div", { class: "legend" },
    el("span", { class: "small" }, "Census silage share:"),
    BIN_LABELS.map((l, i) => el("span", {}, el("i", { style: `background:${cssVar(`--q${i + 1}`)}` }), l)),
    el("span", {}, el("i", { style: `background:transparent;border:1px dashed ${cssVar("--faint")}` }), "withheld or no corn"),
    el("span", {}, el("i", { style: `background:transparent;border:2px solid ${cssVar("--grain")}` }), "field data on this machine"));
}

function table(headers, rows) {
  return el("div", { class: "scroll-x" }, el("table", { class: "data" },
    el("thead", {}, el("tr", {}, headers.map((h, i) => el("th", { class: i ? "num" : "" }, h)))),
    el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c, i) => el("td", { class: i ? "num" : "" }, c)))))));
}

function statePanel(s, pickCounty, allStates) {
  const n = s.nass_2025;
  const survey = s.history.filter((h) => h.silage_share_pct != null);
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
      yFormat: (v, f) => (f ? `${v.toFixed(1)}%` : `${v}%`) }) : null,
    s.fields_on_disk.length ? el("div", {}, el("h3", {}, "Counties with field data"),
      el("div", { class: "chips" }, s.fields_on_disk.map((f) => el("button", { class: "chip", onclick: () => pickCounty(f) }, f)))) : null,
    el("p", { class: "small muted" }, `${Object.keys(s.counties).length} counties have a Census silage share. Click a county.`));
}

function countyPanel(c, app, back) {
  const census = c.history.filter((h) => h.source === "CENSUS");
  const survey = c.history.filter((h) => h.source === "SURVEY");
  const ex = c.extras || {}, check = ex.county_check_2022;
  const useSeason = (s, then) => { app.ctx.fips = s.fips; app.ctx.year = Number(s.year); app.ctx.run_id = null;
    if (!String(app.ctx.as_of || "").startsWith(String(s.year))) app.ctx.as_of = `${s.year}-09-01`; app.save(); app.refreshSide(); then?.(); };
  return el("div", { class: "card" },
    el("div", { class: "row between" }, el("h2", {}, c.name), el("button", { class: "small ghost", onclick: back }, "← State")),
    el("p", { class: "small faint" }, `FIPS ${c.fips}`),
    census.length ? el("div", {}, el("h3", {}, "Corn harvested, USDA Census"),
      table(["Year", "Grain ac", "Silage ac", "Silage share"], census.map((h) => [h.year, fmt(h.grain, 0), h.silage == null ? "withheld" : fmt(h.silage, 0),
        h.silage_share_pct == null ? "—" : `${h.silage_share_pct}%`]))) : el("p", { class: "muted" }, "No Census corn rows for this county."),
    survey.length ? el("p", { class: "small muted" }, `Survey grain acres: ${survey.map((h) => `${h.year} ${fmt(h.grain, 0)}`).join(" · ")}`) : null,
    c.prior_2026 ? el("p", { class: "small" }, `The pipeline's prior for this county: `, el("b", {}, `${c.prior_2026.share_pct}%`), ` silage (${c.prior_2026.source}).`) : null,
    el("h3", {}, "Our field data"),
    c.seasons.length ? el("div", {}, c.seasons.map((s) => el("div", { class: "row between", style: "margin:4px 0" },
      el("span", {}, `${s.year} season${s.partial ? " (partial)" : ""}`),
      el("span", { class: "row" },
        el("button", { class: "small", onclick: () => useSeason(s, () => app.goTab("chat")) }, "Use in chat"),
        app.role === "usda" ? el("button", { class: "small primary", onclick: () => useSeason(s, () => app.goTab("runs")) }, "Run the agent")
          : el("button", { class: "small primary", onclick: () => useSeason(s, () => app.goTab("map")) }, "Find silage")))))
      : el("p", { class: "small muted" }, "No field outlines or satellite series for this county on this machine. Census figures above are still USDA's."),
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

const intro = () => el("div", { class: "card" }, el("h2", {}, "Pick a state"),
  el("p", { class: "muted" }, "States outlined in orange have county field data on this machine. Pick any state to see every county's silage share from the USDA Census, then a county for its data."));

export async function usMapPage(app) {
  const panel = el("div", { class: "usmap-panel" }, intro());
  const mapNode = el("div", { class: "usmap" });
  const page = el("div", { class: "page usmap-page" }, el("div", { class: "usmap-grid" }, el("div", {}, mapNode, legend()), panel));
  const { states, counties } = await shapes();
  const seasons = await api("/api/counties");
  const dataStates = new Set(seasons.map((s) => s.fips.slice(0, 2)));

  requestAnimationFrame(() => {
    const map = L.map(mapNode, { zoomSnap: 0.25, attributionControl: false, scrollWheelZoom: true });
    map.fitBounds([[24.5, -125], [49.5, -66.5]]);
    const line = cssVar("--line"), panelC = cssVar("--panel-2"), grain = cssVar("--grain"), text = cssVar("--text");
    let countyLayer = null, selectedState = null, stateData = null;

    const stateLayer = L.geoJSON(states, {
      style: (f) => ({ color: dataStates.has(f.id) ? grain : line, weight: dataStates.has(f.id) ? 2 : 1, fillColor: panelC, fillOpacity: 1 }),
      onEachFeature: (f, l) => {
        l.bindTooltip(f.properties.name, { sticky: true });
        l.on("click", () => pickState(f.id, l.getBounds()));
        l.on("mouseover", () => f.id !== selectedState && l.setStyle({ fillColor: cssVar("--panel") }));
        l.on("mouseout", () => l.setStyle({ fillColor: panelC }));
      },
    }).addTo(map);

    async function pickCounty(fips) {
      panel.replaceChildren(el("div", { class: "card" }, el("p", { class: "muted" }, "Loading the county…")));
      countyLayer?.eachLayer((l) => l.setStyle({ weight: l.feature.id === fips ? 3 : (stateData?.fields_on_disk.includes(l.feature.id) ? 2 : 0.5),
        color: l.feature.id === fips ? text : (stateData?.fields_on_disk.includes(l.feature.id) ? grain : line) }));
      try { panel.replaceChildren(countyPanel(await api(`/api/geo/county/${fips}`), app, () => panel.replaceChildren(statePanel(stateData, pickCounty, allStates)))); }
      catch (e) { panel.replaceChildren(el("div", { class: "card" }, el("p", { class: "err" }, e.message))); }
    }

    async function pickState(sf, bounds) {
      selectedState = sf;
      map.fitBounds(bounds, { padding: [10, 10] });
      panel.replaceChildren(el("div", { class: "card" }, el("p", { class: "muted" }, "Loading the state…")));
      try { stateData = await api(`/api/geo/state/${sf}`); }
      catch (e) { panel.replaceChildren(el("div", { class: "card" }, el("p", { class: "err" }, e.message))); return; }
      countyLayer?.remove();
      countyLayer = L.geoJSON({ type: "FeatureCollection", features: counties.features.filter((f) => f.id.startsWith(sf)) }, {
        style: (f) => {
          const v = stateData.counties[f.id], has = stateData.fields_on_disk.includes(f.id);
          return { fillColor: v ? cssVar(`--q${binOf(v.share) + 1}`) : "transparent", fillOpacity: v ? 1 : 0,
            color: has ? grain : line, weight: has ? 2 : 0.5, dashArray: v ? null : "2 3" };
        },
        onEachFeature: (f, l) => {
          const v = stateData.counties[f.id];
          l.bindTooltip(`<b>${f.properties.name}</b><br>${v ? `${v.share}% silage (Census ${v.year})<br>${fmt(v.silage, 0)} silage · ${fmt(v.grain, 0)} grain ac` : "silage withheld or no corn"}`
            + (stateData.fields_on_disk.includes(f.id) ? "<br>field data on this machine" : ""), { sticky: true });
          l.on("click", () => pickCounty(f.id));
        },
      }).addTo(map);
      countyLayer.eachLayer((l) => stateData.fields_on_disk.includes(l.feature.id) && l.bringToFront());
      panel.replaceChildren(statePanel(stateData, pickCounty, allStates));
    }

    function allStates() {
      selectedState = null; stateData = null;
      countyLayer?.remove(); countyLayer = null;
      map.fitBounds([[24.5, -125], [49.5, -66.5]]);
      panel.replaceChildren(intro());
    }

    if (app.ctx.fips) {        // open on the season already chosen
      const sf = String(app.ctx.fips).slice(0, 2);
      stateLayer.eachLayer((l) => { if (l.feature.id === sf) pickState(sf, l.getBounds()).then(() => pickCounty(String(app.ctx.fips))); });
    }
    setTimeout(() => map.invalidateSize(), 50);
  });
  return page;
}
