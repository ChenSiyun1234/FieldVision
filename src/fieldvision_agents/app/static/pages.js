// Full-page views: the analyst's run console and dashboard, the dairy owner's map.
import { priceModelTable, silageCompare, silageMap, silageSummary, silageTable, supplyTiles, whatIfCard } from "./cards.js";
import { lineChart } from "./charts.js";
import { $, STATUS, api, big, el, fmt, mdEl, money } from "./ui.js";

// --- Runs: every step of one agent run, its critic, report and questions ------------------

function stepNote(s) {
  const r = s.result || {};
  if (r.error) return el("span", { class: "refused" }, `Refused by guardrail: ${r.error}`);
  switch (s.tool) {
    case "scout": return `${fmt(r.n_fields)} fields · ${fmt(r.fields_with_clear_series)} clear · ${fmt(r.fields_with_little_data)} little data`;
    case "route_options": return `Compared: ${Object.keys(r.options || {}).join(", ")}`;
    case "extract_features": return `${fmt(r.rows ?? r.n_fields)} feature rows`;
    case "classify": return `${Object.entries(r.models_used || {}).map(([k, v]) => `${k} ${fmt(v)}`).join(", ")} · ${fmt(r.n_abstain)} abstain`;
    case "aggregate": return `silage ${fmt(r.silage_acres)} (${fmt(r.silage_acres_low)}–${fmt(r.silage_acres_high)}) · grain ${fmt(r.grain_acres)}`;
    case "run_critic": return `${r.overall} → ${r.suggested_action}`;
    case "recommend_sources": return r.summary ? r.summary.split("\n")[0] : `${r.overall}`;
    case "abstain_region": return `Abstained: ${r.reason ?? s.args.reason ?? ""}`;
    case "finish": return `Closed as ${STATUS[s.args.status] || s.args.status}`;
    default: return JSON.stringify(r).slice(0, 140);
  }
}

export function runPage(d, app) {
  const { result, evidence: ev, steps } = d, est = ev.estimate || {};
  const total = (est.silage_acres || 0) + (est.grain_acres || 0) + (est.abstained_acres || 0) || 1;
  const say = el("p", { class: "say" });
  const reportBox = d.report_md ? mdEl(d.report_md) : el("p", { class: "muted" }, "No report yet.");
  const answers = el("div");
  const q = el("input", { type: "text", placeholder: "Ask about this run: how sure are we? why that route?" });
  const ask = async () => {
    if (!q.value.trim()) return;
    say.textContent = "Answering…"; say.className = "say";
    try {
      const a = await api(`/api/runs/${result.run_id}/ask`, { question: q.value, offline: app.offline() });
      answers.prepend(el("div", { class: "quote" }, el("b", {}, q.value), el("div", {}, a.answer),
        el("span", { class: "src" }, `evidence: ${(a.sections_used || []).join(", ")} · ${a.grounded ? "every number checked" : "contains [unverified]"}`)));
      say.textContent = ""; q.value = "";
    } catch (e) { say.textContent = e.message; say.className = "say err"; }
  };
  q.addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
  return el("div", { class: "page" },
    el("div", { class: "card" },
      el("div", { class: "row between" }, el("h2", { class: "title" }, `${d.name || ev.county.fips} · ${ev.county.year}`),
        el("span", { class: `badge ${result.status}` }, STATUS[result.status] || result.status)),
      el("div", { class: "kv" },
        el("div", {}, el("div", { class: "k" }, "Silage acres (model estimate)"), el("div", { class: "v big silage-t" }, fmt(est.silage_acres)),
          el("div", { class: "k" }, `range ${fmt(est.silage_acres_low)} – ${fmt(est.silage_acres_high)}`)),
        el("div", {}, el("div", { class: "k" }, "Grain acres"), el("div", { class: "v big grain-t" }, fmt(est.grain_acres))),
        el("div", {}, el("div", { class: "k" }, "Abstained acres"), el("div", { class: "v big" }, fmt(est.abstained_acres)))),
      el("div", { class: "bar", "aria-hidden": "true" },
        el("span", { class: "s", style: `width:${(100 * (est.silage_acres || 0)) / total}%` }),
        el("span", { class: "g", style: `width:${(100 * (est.grain_acres || 0)) / total}%` }),
        el("span", { class: "a", style: `width:${(100 * (est.abstained_acres || 0)) / total}%` })),
      el("p", {}, `Routing: ${ev.run.policy_sequence.join(" → ") || "none"} · Critic: ${ev.run.critic_sequence.join(" → ") || "not run"}`),
      est.adjustment ? el("p", { class: "muted small" }, `Totals: ${est.adjustment}`) : null,
      el("p", { class: "faint small" }, `Run ${result.run_id}${result.tokens ? ` · ${fmt(result.tokens)} tokens` : " · offline"} · model estimates, not NASS statistics`)),
    el("div", { class: "card" }, el("h3", {}, "What the agent did"),
      el("ol", { class: "steps" }, steps.map((s) => el("li", {}, el("span", { class: "tool" }, s.tool + (s.args.policy ? ` ${s.args.policy}` : "")), " ", stepNote(s),
        s.args.reason ? el("div", { class: "why" }, `Why: ${s.args.reason}`) : null)))),
    el("div", { class: "card" }, el("h3", {}, "Critic"),
      ev.critic.length ? ev.critic.map((c, i) => el("div", {},
        el("p", {}, `Check ${i + 1}: `, el("span", { class: `verdict ${c.overall}` }, c.overall), ` → ${c.suggested_action}`),
        c.summary ? el("p", { class: "muted" }, c.summary) : null,
        (c.flags || []).map((f) => el("div", { class: `flag ${f.status}` }, el("span", { class: "name" }, f.check), ` (${f.status}) ${f.why || ""}`,
          (f.evidence || []).length ? el("div", { class: "small faint" }, "Sources: " + f.evidence.map((e) => (typeof e === "string" ? e : [e.source, e.page && `p.${e.page}`].filter(Boolean).join(" "))).join("; ")) : null))))
        : el("p", { class: "muted" }, "The critic did not run.")),
    el("div", { class: "card" }, el("div", { class: "row between" }, el("h3", {}, "County report"),
      el("button", { class: "small", onclick: async (e) => {
        e.target.disabled = true; say.textContent = app.offline() ? "Writing from the run record…" : "Nemotron Ultra is writing (about 30 s)…";
        try { const rep = await api(`/api/runs/${result.run_id}/report`, { offline: app.offline() }); reportBox.replaceWith(mdEl(rep.report_md)); say.textContent = ""; }
        catch (err) { say.textContent = err.message; say.className = "say err"; } finally { e.target.disabled = false; }
      } }, "Write the report")), reportBox),
    el("div", { class: "card" }, el("h3", {}, "Ask about this run"), el("div", { class: "row" }, q, el("button", { onclick: ask }, "Ask")), say, answers));
}

export function runsPage(app) {
  const page = el("div", { class: "page" });
  const say = el("p", { class: "say" });
  const go = el("button", { class: "primary", onclick: async () => {
    const { fips, year } = app.ctx;
    if (!fips) { say.textContent = "Pick a county and season on the left."; return; }
    go.disabled = true; say.textContent = app.offline() ? "Running with the offline agent…" : "Nemotron is running the county (up to a minute)…";
    try { const d = await api("/api/runs", { fips, year: Number(year), offline: app.offline() }); app.ctx.run_id = d.result.run_id; app.refreshSide(); page.replaceWith(runPage(d, app)); }
    catch (e) { say.textContent = e.message; say.className = "say err"; go.disabled = false; }
  } }, "Run the agent on this season");
  page.append(el("div", { class: "card" }, el("h2", {}, "Run a county-season through the agent"),
    el("p", { class: "muted" }, "Scout the imagery, pick a route, classify every field with the frozen model, add up acres, check against Quick Stats history, rerun once if the critic asks, then accept or abstain. Every step and number is logged."),
    el("div", { class: "row" }, go), say));
  return page;
}

// --- Dashboard -----------------------------------------------------------------------

export async function dashboardPage(app) {
  const page = el("div", { class: "page" }, el("p", { class: "muted" }, "Loading the dashboard…"));
  const d = await api("/api/dashboard");
  const s = d.supply.series;
  const slider = el("input", { type: "range", min: -2, max: 3, step: 0.5, value: 1, "aria-label": "Silage share change in points" });
  const label = el("b", {}, "+1 point");
  const whatBox = el("div", {}, whatIfCard(d.what_if));
  let pending;
  slider.addEventListener("input", () => {
    const v = Number(slider.value); label.textContent = `${v > 0 ? "+" : ""}${v} point${Math.abs(v) === 1 ? "" : "s"}`;
    clearTimeout(pending);
    pending = setTimeout(async () => { if (v === 0) return; whatBox.replaceChildren(whatIfCard(await api(`/api/whatif?points=${v}`))); }, 200);
  });
  const runsTable = el("div", { class: "scroll-x" }, el("table", { class: "data" },
    el("thead", {}, el("tr", {}, ["When", "County", "Season", "Status", "Routing", "Critic"].map((h) => el("th", {}, h)))),
    el("tbody", {}, d.runs.map((r) => el("tr", {},
      el("td", {}, r.when), el("td", {}, el("button", { class: "small ghost", onclick: () => app.openRun(r.run_id) }, r.name)), el("td", {}, r.year),
      el("td", {}, el("span", { class: `badge ${r.status}` }, STATUS[r.status] || r.status)), el("td", {}, r.policies.join(" → ")), el("td", {}, r.critic.join(" → ")))))));
  page.replaceChildren(
    el("div", {}, el("h2", {}, "Corn supply, stocks and price (USDA)"), supplyTiles(d.supply.tiles)),
    el("div", { class: "charts" },
      lineChart({ title: "Corn grain production, billion bushels", points: s.production.map((p) => ({ x: p.year, y: p.value / 1e9, forecast: p.forecast })),
        yFormat: (v, f) => (f ? `${v.toFixed(2)} B bu` : v.toFixed(0)), note: "2026 is USDA's November forecast (hollow point)." }),
      lineChart({ title: "Price received, marketing year, $ per bushel", points: s.price.map((p) => ({ x: p.year, y: p.value })),
        yFormat: (v, f) => (f ? money(v) : `$${v}`) }),
      lineChart({ title: "Corn stocks on September 1, billion bushels", points: s.stocks.map((p) => ({ x: p.year, y: p.value / 1e9 })),
        yFormat: (v, f) => (f ? `${v.toFixed(2)} B bu` : v.toFixed(1)) }),
      lineChart({ title: "Corn harvested for silage, million acres (USDA)", points: s.silage_acres.map((p) => ({ x: p.year, y: p.value / 1e6 })),
        yFormat: (v, f) => (f ? `${v.toFixed(2)} M ac` : v.toFixed(1)) })),
    el("div", {}, el("h2", {}, "Our silage estimates next to USDA's"), silageCompare(d.silage)),
    el("div", {}, el("h2", {}, "Silage what-if on price"),
      el("div", { class: "card whatif-controls" }, el("span", {}, "National silage share:"), slider, label,
        el("span", { class: "small muted" }, "Moves acres from grain to silage in 2025 and asks each price model what that does.")), whatBox),
    el("div", {}, el("h2", {}, "Can the price models be trusted?"), priceModelTable(d.price_models)),
    el("div", {}, el("h2", {}, "Recent agent runs"), el("div", { class: "card" }, d.runs.length ? runsTable : el("p", { class: "muted" }, "No runs yet."))),
    d.agent_quality.evaluation_md ? el("div", {}, el("h2", {}, "How the agent tested"), el("div", { class: "card" }, mdEl(d.agent_quality.evaluation_md),
      d.agent_quality.benchmark_md ? el("div", {}, el("h3", {}, "Nemotron models compared"), mdEl(d.agent_quality.benchmark_md)) : null)) : null,
    d.agent_quality.gaps_md ? el("div", {}, el("h2", {}, "Known gaps, checked"), el("div", { class: "card" }, mdEl(d.agent_quality.gaps_md))) : null);
  return page;
}

// --- Dairy map -------------------------------------------------------------------------

export function dairyMapPage(app, result) {
  const page = el("div", { class: "page" });
  const say = el("p", { class: "say" });
  const mapWrap = el("div");
  const body = el("div");
  const draw = (r) => {
    body.replaceChildren(silageSummary(r), r.fields.length ? silageTable(r) : el("p", { class: "muted" }, "No field in reach. Try a later date or a longer haul."),
      el("div", {}, r.checks.map((c) => el("div", { class: `flag ${c.status}` }, el("span", { class: "name" }, c.check.replaceAll("_", " ")), ` (${c.status}) ${c.why || ""}`))),
      el("p", { class: "note" }, r.note));
    const fresh = el("div", { class: "minimap", style: "height:460px" });   // a new node per search: Leaflet binds once
    mapWrap.replaceChildren(fresh);
    silageMap(fresh, r, { onPick: (ll) => { app.setDairyLocation(ll.lat, ll.lng); say.textContent = `Location set to ${ll.lat.toFixed(4)}, ${ll.lng.toFixed(4)}. Press "Find silage".`; } });
  };
  const find = el("button", { class: "primary", onclick: async () => {
    find.disabled = true; say.textContent = "Matching standing corn to your dairy…"; say.className = "say";
    try { const r = await api("/api/silage", app.silageRequest()); app.lastSilage = r; draw(r); say.textContent = ""; }
    catch (e) { say.textContent = e.message; say.className = "say err"; } finally { find.disabled = false; }
  } }, "Find silage");
  page.append(el("div", { class: "card" }, el("div", { class: "row between" }, el("h2", { class: "title" }, "Standing corn you could buy"), find),
    el("p", { class: "muted small" }, "Fields that are still green and standing on the date, likely headed for grain, at least 5 acres, within your haul distance. Click the map to set your dairy's location."), say, mapWrap, body));
  if (result) requestAnimationFrame(() => draw(result));
  else requestAnimationFrame(() => find.click());
  return page;
}
