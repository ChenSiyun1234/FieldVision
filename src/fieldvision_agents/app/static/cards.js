// What a chat answer can carry besides text: one renderer per card type the server sends.
import { groupedBars } from "./charts.js";
import { $, STATUS, big, el, fmt, mdEl, money } from "./ui.js";

const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

/** A Leaflet map of one silage search: the dairy, the matched fields, a line to each. */
export function silageMap(node, r, { onPick } = {}) {
  if (!window.L) { node.append(el("p", { class: "muted small" }, "Map library did not load.")); return null; }
  const css = getComputedStyle(document.documentElement);
  const map = L.map(node, { scrollWheelZoom: false, attributionControl: true });
  const satellite = L.tileLayer(ESRI, { maxZoom: 18, attribution: "Imagery © Esri" }).addTo(map);
  const tone = matchMedia("(prefers-color-scheme: light)").matches ? "Light" : "Dark";
  const street = L.tileLayer(`https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_${tone}_Gray_Base/MapServer/tile/{z}/{y}/{x}`,
    { maxZoom: 16, attribution: "Basemap © Esri" });
  L.control.layers({ "Satellite": satellite, "Map": street }, null, { position: "topright" }).addTo(map);
  const grain = css.getPropertyValue("--grain").trim(), silage = css.getPropertyValue("--silage").trim();
  const d = r.dairy;
  const layer = L.geoJSON(r.map, {
    style: { color: grain, weight: 2, fillColor: grain, fillOpacity: 0.35 },
    onEachFeature: (f, l) => {
      const p = f.properties;
      l.bindTooltip(`<b>${fmt(p.tons, 0)} tons</b> · ${fmt(p.road_miles)} road mi<br>${fmt(p.acres)} acres · P(silage) ${p.p_silage_adj}<br>green on ${p.ndvi_date} (NDVI ${p.ndvi_as_of})`);
    },
  });
  // Leaflet draws vectors against the current view, so the view is set before any shape is added.
  const b = layer.getBounds();
  b.isValid() ? map.fitBounds(b.extend([d.lat, d.lon]), { padding: [24, 24], maxZoom: 13 }) : map.setView([d.lat, d.lon], 11);
  layer.addTo(map);
  r.fields.forEach((f) => L.polyline([[d.lat, d.lon], [f.lat, f.lon]], { color: grain, weight: 1, opacity: 0.7, dashArray: "3 4" }).addTo(map));
  const miles = r.params?.max_road_miles || 20;
  L.circle([d.lat, d.lon], { radius: (miles / 1.3) * 1609, color: silage, weight: 1, fill: false, dashArray: "4 6", interactive: false }).addTo(map);
  L.circleMarker([d.lat, d.lon], { radius: 8, color: "#071018", weight: 2, fillColor: silage, fillOpacity: 1 })
    .bindTooltip(`Your dairy${d.dairy_id && d.dairy_id !== "YOUR-DAIRY" ? ` (${d.dairy_id})` : ""}`).addTo(map);
  if (onPick) map.on("click", (e) => onPick(e.latlng));
  setTimeout(() => map.invalidateSize(), 50);
  return map;
}

function checksList(checks) {
  return el("div", {}, (checks || []).map((c) => el("div", { class: `flag ${c.status}` },
    el("span", { class: "name" }, c.check.replaceAll("_", " ")), ` (${c.status}) `, c.why || "")));
}

export function silageSummary(r) {
  const d = r.dairy;
  const price = d.delivered_usd_per_ton || [];
  return el("div", { class: "kv" },
    el("div", {}, el("div", { class: "k" }, "Tons matched"), el("div", { class: "v silage-t" }, fmt(d.matched_tons, 0)),
      el("div", { class: "k" }, `of ${fmt(d.request_tons, 0)} wanted`)),
    el("div", {}, el("div", { class: "k" }, "Fields"), el("div", { class: "v" }, fmt(d.n_fields ?? r.fields.length, 0))),
    el("div", {}, el("div", { class: "k" }, "Delivered, per wet ton"), el("div", { class: "v" }, price.length ? `${money(price[0])}–${money(price[1])}` : "—")),
    el("div", {}, el("div", { class: "k" }, "Still standing in the county"), el("div", { class: "v" }, fmt(r.funnel?.candidate, 0)),
      el("div", { class: "k" }, `of ${fmt(r.funnel?.fields, 0)} corn fields`)));
}

export function silageTable(r) {
  return el("div", { class: "scroll-x" }, el("table", { class: "data" },
    el("thead", {}, el("tr", {}, ["Field", "Tons", "Road mi", "Acres", "P(silage)", "Green on", "Haul $/t"].map((h, i) => el("th", { class: i ? "num" : "" }, h)))),
    el("tbody", {}, r.fields.map((f) => el("tr", {},
      el("td", {}, f.field_id), el("td", { class: "num" }, fmt(f.tons, 0)), el("td", { class: "num" }, fmt(f.road_miles)),
      el("td", { class: "num" }, fmt(f.acres)), el("td", { class: "num" }, fmt(f.p_silage_adj, 2)), el("td", { class: "num" }, f.ndvi_date),
      el("td", { class: "num" }, `${money(f.haul_usd_per_ton[0])}–${money(f.haul_usd_per_ton[1])}`))))));
}

const RENDER = {
  silage(r, ctx) {
    const box = el("div", { class: "card" }, el("div", { class: "row between" },
      el("h3", {}, `Standing corn near you · ${r.name} · as of ${r.as_of}`), el("span", { class: `badge ${r.overall}` }, STATUS[r.overall])));
    box.append(silageSummary(r));
    const mapNode = el("div", { class: "minimap" });
    box.append(mapNode, r.fields.length ? silageTable(r) : el("p", { class: "muted" }, "No field in reach."), checksList(r.checks),
      el("p", { class: "note" }, r.note));
    if (ctx?.openMap) box.append(el("button", { class: "small", onclick: () => ctx.openMap(r) }, "Open on the big map"));
    requestAnimationFrame(() => silageMap(mapNode, r));
    return box;
  },
  run(d, ctx) {
    const ev = d.evidence, est = ev.estimate || {};
    return el("div", { class: "card" },
      el("div", { class: "row between" }, el("h3", {}, `${d.name || ev.county.fips} · ${ev.county.year} · run ${d.result.run_id}`),
        el("span", { class: `badge ${d.result.status}` }, STATUS[d.result.status] || d.result.status)),
      el("div", { class: "kv" },
        el("div", {}, el("div", { class: "k" }, "Silage acres"), el("div", { class: "v silage-t" }, fmt(est.silage_acres)),
          el("div", { class: "k" }, `range ${fmt(est.silage_acres_low)}–${fmt(est.silage_acres_high)}`)),
        el("div", {}, el("div", { class: "k" }, "Grain acres"), el("div", { class: "v grain-t" }, fmt(est.grain_acres))),
        el("div", {}, el("div", { class: "k" }, "Abstained acres"), el("div", { class: "v" }, fmt(est.abstained_acres)))),
      el("p", { class: "small muted" }, `Routing ${ev.run.policy_sequence.join(" → ") || "none"} · critic ${ev.run.critic_sequence.join(" → ") || "not run"}`),
      ctx?.openRun ? el("button", { class: "small", onclick: () => ctx.openRun(d.result.run_id) }, "Open every step in Runs") : null);
  },
  report(d) {
    return el("div", { class: "card" }, el("h3", {}, `County report · run ${d.run_id}`), mdEl(d.report_md),
      el("p", { class: "small faint" }, `${d.grounded ? "Every number checked against the evidence." : "Some numbers are marked [unverified]."} Saved to ${d.path}`));
  },
  visits(d) {
    const p = d.plan, st = d.county_status;
    return el("div", { class: "card" }, el("h3", {}, `Field visit plan · ${d.name} ${d.year}`),
      el("div", { class: "kv" },
        el("div", {}, el("div", { class: "k" }, "Silage estimate"), el("div", { class: "v" }, `${fmt(st.silage_acres, 0)} ac`),
          el("div", { class: "k" }, `90% range ${fmt(st.range_low, 0)}–${fmt(st.range_high, 0)}`)),
        el("div", {}, el("div", { class: "k" }, `Range after ${p.visits} visits`), el("div", { class: "v" }, `±${fmt(p.plus_minus_after, 0)}`),
          el("div", { class: "k" }, `from ±${fmt(p.plus_minus_before, 0)} (${p.range_reduction_pct}% smaller)`)),
        el("div", {}, el("div", { class: "k" }, "Grain not produced"), el("div", { class: "v" }, `${big(d.supply.bushels_not_in_grain_supply)} bu`))),
      el("div", { class: "scroll-x" }, el("table", { class: "data" },
        el("thead", {}, el("tr", {}, ["#", "Field", "Acres", "P(silage)", "Why visit"].map((h) => el("th", {}, h)))),
        el("tbody", {}, p.fields.map((f) => el("tr", {}, el("td", {}, f.rank), el("td", {}, f.field_id), el("td", { class: "num" }, fmt(f.area_acres)),
          el("td", { class: "num" }, f.p_silage ?? "abstained"), el("td", {}, f.why)))))),
      el("p", { class: "note" }, d.supply.note));
  },
  passages(rows) {
    return el("div", { class: "card" }, el("h3", {}, "From NASS reports"),
      rows.map((p) => el("div", { class: "quote" }, p.text.trim(), el("span", { class: "src" }, `${p.source}, page ${p.page}${p.year ? ` · ${p.year}` : ""}`))));
  },
  supply(d) { return supplyTiles(d.tiles); },
  whatif(w) { return whatIfCard(w); },
  silage_compare(s) { return silageCompare(s); },
  price_models(rows) { return priceModelTable(rows); },
  quality(q) {
    return el("div", { class: "card" }, el("h3", {}, "How the agent tested"),
      q.evaluation_md ? mdEl(q.evaluation_md) : el("p", { class: "muted" }, "No evaluation run on this machine yet."),
      q.benchmark_md ? el("div", {}, el("h3", {}, "Nemotron models compared"), mdEl(q.benchmark_md)) : null,
      q.limitations_md ? el("details", {}, el("summary", {}, "Limitations found and what was done"), mdEl(q.limitations_md)) : null);
  },
  fields(d) {
    return el("div", { class: "card" }, el("h3", {}, `${d.count} fields · ${d.kind.replaceAll("_", " ")}`),
      el("div", { class: "scroll-x" }, el("table", { class: "data" },
        el("thead", {}, el("tr", {}, ["Field", "Acres", "Call", "P(silage)", "Cut seen", "Abstain reason"].map((h) => el("th", {}, h)))),
        el("tbody", {}, d.fields.map((f) => el("tr", {}, el("td", {}, f.field_id), el("td", { class: "num" }, fmt(f.area_acres)),
          el("td", {}, f.call || "—"), el("td", { class: "num" }, f.p_silage_model), el("td", {}, f.drop_date || "—"), el("td", {}, f.abstain_reason || "")))))));
  },
  error(e) { return el("div", { class: "card" }, el("p", { class: "err" }, `${e.tool}: ${e.error}`)); },
};

export function renderCard(card, ctx) {
  const fn = RENDER[card.type];
  return fn ? fn(card.data, ctx) : el("pre", { class: "card small" }, JSON.stringify(card.data, null, 1).slice(0, 2000));
}

// --- pieces shared with the dashboard ---------------------------------------------

const UNIT = { bu: (v) => `${big(v)} bu`, "bu/acre": (v) => `${fmt(v)} bu/ac`, acres: (v) => `${big(v)} ac`, "%": (v) => `${fmt(v)}%`, "$/bu": (v) => money(v) };

export function supplyTiles(tiles) {
  return el("div", { class: "tiles" }, tiles.map((t) => {
    const arrow = t.change_pct == null ? "" : t.change_pct > 0 ? "▲" : t.change_pct < 0 ? "▼" : "■";
    const delta = t.change_pct != null ? el("span", {}, el("b", {}, `${arrow} ${fmt(Math.abs(t.change_pct))}%`), ` vs ${t.vs}`)
      : t.change_pts != null ? el("span", {}, el("b", {}, `${t.change_pts > 0 ? "+" : ""}${fmt(t.change_pts)} pts`), ` vs ${t.vs}`) : "";
    return el("div", { class: "tile" }, el("div", { class: "label" }, t.label),
      el("div", { class: "value" }, (UNIT[t.unit] || ((v) => fmt(v)))(t.value)), el("div", { class: "delta" }, delta), el("div", { class: "src" }, t.source));
  }));
}

export function whatIfCard(w) {
  const lo = w.range_pct?.[0], hi = w.range_pct?.[1];
  return el("div", { class: "card" }, el("h3", {}, `What if national silage share were ${w.points > 0 ? "+" : ""}${w.points} points?`),
    el("div", { class: "kv" },
      el("div", {}, el("div", { class: "k" }, "Acres moved from grain to silage"), el("div", { class: "v" }, big(w.acres))),
      el("div", {}, el("div", { class: "k" }, "Grain not produced"), el("div", { class: "v" }, `${big(-w.grain_bushels)} bu`)),
      el("div", {}, el("div", { class: "k" }, "Stocks-to-use, 2025/26"), el("div", { class: "v" }, `${w.stu_before_pct}% → ${w.stu_after_pct}%`)),
      el("div", {}, el("div", { class: "k" }, "Price effect, by model"), el("div", { class: "v" }, `${lo > 0 ? "+" : ""}${lo}% to ${hi > 0 ? "+" : ""}${hi}%`))),
    el("div", { class: "scroll-x" }, el("table", { class: "data" },
      el("thead", {}, el("tr", {}, ["Model (explains a season from its ending stocks)", "2025 price in the model", "Change", "Change $/bu"].map((h, i) => el("th", { class: i ? "num" : "" }, h)))),
      el("tbody", {}, w.models.map((m) => el("tr", {}, el("td", {}, m.model.replace("explain: ", "")), el("td", { class: "num" }, money(m.price_before)),
        el("td", { class: "num" }, `${m.change_pct > 0 ? "+" : ""}${m.change_pct}%`), el("td", { class: "num" }, `${m.change_usd > 0 ? "+" : ""}${money(m.change_usd)}`)))))),
    Object.keys(w.other_models_per_point_pct || {}).length ? el("p", { class: "small muted" },
      `Per point, the other price models give: ${Object.entries(w.other_models_per_point_pct).map(([k, v]) => `${k} +${v}%`).join(", ")}.`) : null,
    el("p", { class: "note" }, w.note));
}

export function silageCompare(s) {
  const st = s.states_2025 || [];
  const la = s.lancaster_2023;
  const cats = [...st.map((x) => x.state + " 2025"), ...(s.maryland_2025 ? ["MD 2025"] : []), ...(s.frederick_2022 ? ["Frederick 2022"] : []),
    ...(la ? ["Lancaster 2023*"] : [])];
  const usda = [...st.map((x) => x.usda_pct), ...(s.maryland_2025 ? [s.maryland_2025.usda_pct] : []), ...(s.frederick_2022 ? [s.frederick_2022.census_pct] : []),
    ...(la ? [la.census_pct] : [])];
  const ours = [...st.map((x) => x.ours_pct), ...(s.maryland_2025 ? [s.maryland_2025.ours_pct] : []), ...(s.frederick_2022 ? [s.frederick_2022.ours_pct] : []),
    ...(la ? [la.ours_pct] : [])];
  const chart = groupedBars({
    title: "Silage share of harvested corn: USDA vs our model", categories: cats, width: 900,
    series: [{ name: "USDA (survey or Census)", color: "var(--s1)", values: usda }, { name: "Our model", color: "var(--s2)", values: ours }],
    yFormat: (v, full) => (full ? `${fmt(v)}%` : `${v}%`),
    note: "Our county and state totals are not validated yet: Pennsylvania and Nebraska read low, Iowa and Wisconsin high."
      + (la ? ` *Lancaster County, PA 2023 against the ${la.census_year} Census: USDA has no 2023 county figure.` : ""),
  });
  const ft = s.field_test;
  return el("div", {}, chart, ft ? el("p", { class: "small muted" },
    `Field level: ${ft.right} of ${ft.of} labelled Beltsville field-years right (${ft.silage_found} of ${ft.silage_total} silage fields found; always-grain gets ${ft.always_grain}). ${ft.source}.`) : null);
}

export function priceModelTable(rows) {
  const base = rows.find((r) => r.model.startsWith("no change"));
  return el("div", { class: "card" }, el("h3", {}, "Price models held out one year at a time, 2001–2024"),
    el("div", { class: "scroll-x" }, el("table", { class: "data" },
      el("thead", {}, el("tr", {}, ["Model", "Mean abs. error", "Direction right", "2025 predicted", "2025 error"].map((h, i) => el("th", { class: i ? "num" : "" }, h)))),
      el("tbody", {}, rows.map((r) => el("tr", { class: r === base ? "base" : r.mape_pct < base.mape_pct ? "hl" : "" },
        el("td", {}, r.model), el("td", { class: "num" }, `${r.mape_pct}%`), el("td", { class: "num" }, r.direction_right_pct == null ? "—" : `${r.direction_right_pct}%`),
        el("td", { class: "num" }, money(r.pred_2025)), el("td", { class: "num" }, `${r.error_2025_pct > 0 ? "+" : ""}${r.error_2025_pct}%`)))))),
    el("p", { class: "note" }, `"Explain" models use the season's ending stocks, known only after the season; "forecast" models use last season's. `
      + `Last year's price (${base?.mape_pct}% average error, ${base?.error_2025_pct}% on 2025) is the bar to beat; with forecast-time data no model beats it.`));
}
