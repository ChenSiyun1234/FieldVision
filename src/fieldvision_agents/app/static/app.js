// FieldVision: one page, two roles. The shell keeps who you are, the season, the dairy profile and the
// chat, and swaps the main view between Chat and the role's pages.
import { renderCard } from "./cards.js";
import { dairyMapPage, dashboardPage, runPage, runsPage } from "./pages.js";
import { $, STATUS, api, el, fmt, mdEl, store } from "./ui.js";

const ROLES = {
  dairy: { name: "Dairy owner", tabs: [["chat", "Ask"], ["map", "Silage map"]],
    intro: "Ask where you can still buy standing corn for silage, how much, how far and roughly what it costs delivered. Set your dairy on the left." },
  usda: { name: "USDA analyst", tabs: [["chat", "Ask"], ["runs", "Runs"], ["dashboard", "Dashboard"]],
    intro: "Ask the agent to run a county, explain how sure it is, write the report, plan field visits, compare our silage estimates with USDA's, or walk through corn supply and price." },
};

const app = {
  role: null, tab: "chat", status: null, seasons: [], dairies: [], suggestions: {},
  ctx: store.get("fv.ctx", { fips: "24021", year: 2022, as_of: "2022-09-01" }),
  profile: store.get("fv.dairy", { mode: "permit", dairy_id: "MDE-21912", lat: 39.45, lon: -77.4, head: 400, tons: 2000, max_miles: 20 }),
  threads: { dairy: [], usda: [] }, lastSilage: null, mode: "nemotron",
  offline() { return this.mode === "offline" || !this.status?.nemotron.ok; },
  save() { store.set("fv.ctx", { fips: this.ctx.fips, year: this.ctx.year, as_of: this.ctx.as_of }); store.set("fv.dairy", this.profile); },
  silageRequest() {
    const p = this.profile, base = { fips: this.ctx.fips, year: Number(this.ctx.year), as_of: this.ctx.as_of,
      tons: p.tons ? Number(p.tons) : null, max_miles: p.max_miles ? Number(p.max_miles) : null };
    return p.mode === "permit" ? { ...base, dairy_id: p.dairy_id }           // the permit file has the herd size
      : { ...base, lat: Number(p.lat), lon: Number(p.lon), head: p.head ? Number(p.head) : null };
  },
  chatContext() {
    const c = { fips: this.ctx.fips, year: Number(this.ctx.year), as_of: this.ctx.as_of, offline: this.offline() };
    if (this.ctx.run_id) c.run_id = this.ctx.run_id;
    if (this.role === "dairy") {
      const r = this.silageRequest();
      c.dairy = { dairy_id: r.dairy_id, lat: r.lat, lon: r.lon, head: r.head, tons: r.tons, max_miles: r.max_miles };
    }
    return c;
  },
  setDairyLocation(lat, lon) { this.profile = { ...this.profile, mode: "location", lat: +lat.toFixed(5), lon: +lon.toFixed(5) }; this.save(); this.refreshSide(); },
  async openRun(runId) { this.ctx.run_id = runId; this.tab = "runs"; this.renderTabs(); await this.show(runPage(await api(`/api/runs/${runId}`), this)); this.refreshSide(); },
  openMap(r) { this.tab = "map"; this.renderTabs(); this.show(dairyMapPage(this, r)); },
};

// --- shell -------------------------------------------------------------------------

async function loadStatus() {
  const s = await api("/api/status");
  app.status = s;
  $("#status").replaceChildren(
    el("span", { class: `chip ${s.nemotron.ok ? "ok" : "off"}` }, s.nemotron.ok ? "Nemotron ready" : "Offline: no NVIDIA key"),
    el("span", { class: `chip ${s.report_search.ok ? "ok" : "off"}` }, `Report search: ${s.report_search.mode}`),
    el("span", { class: `chip ${s.county_data.ok ? "ok" : "off"}` }, `${s.county_data.n} county-seasons`));
  if (!s.nemotron.ok) app.mode = "offline";
}

function chooseRole(role) {
  app.role = role; app.tab = "chat"; store.set("fv.role", role);
  $("#landing").hidden = true; $("#workspace").hidden = false; $("#switch-role").hidden = false;
  app.renderTabs(); app.refreshSide(); showTab();
}

function goHome() {
  app.role = null; store.set("fv.role", null);
  $("#landing").hidden = false; $("#workspace").hidden = true; $("#switch-role").hidden = true; $("#tabs").replaceChildren();
}

app.renderTabs = () => {
  $("#tabs").replaceChildren(...ROLES[app.role].tabs.map(([id, label]) =>
    el("button", { "aria-selected": String(app.tab === id), onclick: () => { app.tab = id; app.renderTabs(); showTab(); } }, label)));
};

app.show = async (node) => { $("#view").replaceChildren(await node); };

async function showTab() {
  try {
    if (app.tab === "chat") return app.show(chatView());
    if (app.tab === "map") return app.show(dairyMapPage(app, app.lastSilage));
    if (app.tab === "runs") return app.show(app.ctx.run_id ? runPage(await api(`/api/runs/${app.ctx.run_id}`), app) : runsPage(app));
    if (app.tab === "dashboard") return app.show(dashboardPage(app));
  } catch (e) { app.show(el("div", { class: "page" }, el("p", { class: "err" }, e.message))); }
}

// --- sidebar: season, agent mode, and the role's own settings ----------------------------

app.refreshSide = async () => {
  const side = $("#side");
  const seasons = app.seasons.filter((c) => app.role !== "dairy" || !c.partial);
  const season = el("select", { onchange: (e) => {
    const [f, y] = e.target.value.split("|"); app.ctx.fips = f; app.ctx.year = Number(y); app.ctx.run_id = null;
    if (!String(app.ctx.as_of || "").startsWith(y)) app.ctx.as_of = `${y}-09-01`;
    app.save(); app.refreshSide();
  } }, seasons.map((c) => el("option", { value: `${c.fips}|${c.year}`, selected: c.fips === app.ctx.fips && Number(c.year) === Number(app.ctx.year) },
    `${c.name} · ${c.year}`)));
  const online = app.status?.nemotron.ok;
  const mode = el("div", { class: "seg", role: "group", "aria-label": "Agent" },
    ["nemotron", "offline"].map((m) => el("button", { "aria-pressed": String(app.mode === m), disabled: m === "nemotron" && !online,
      onclick: () => { app.mode = m; app.refreshSide(); } }, m === "nemotron" ? "Nemotron" : "Offline rules")));
  const blocks = [el("div", {}, el("h3", {}, ROLES[app.role].name), el("label", {}, "County and season", season), el("div", {}, el("span", { class: "k" }, "Agent"), mode))];

  if (app.role === "dairy") {
    const p = app.profile, y = app.ctx.year;
    const set = (k) => (e) => { app.profile = { ...app.profile, [k]: e.target.value }; app.save(); };
    const seg = el("div", { class: "seg" }, [["permit", "Permitted dairy"], ["location", "My location"]].map(([m, l]) =>
      el("button", { "aria-pressed": String(p.mode === m), onclick: () => { app.profile = { ...p, mode: m }; app.save(); app.refreshSide(); } }, l)));
    const who = p.mode === "permit"
      ? el("label", {}, "Dairy (permit id)", el("select", { onchange: set("dairy_id") }, app.dairies.map((d) =>
          el("option", { value: d.dairy_id, selected: d.dairy_id === p.dairy_id }, `${d.dairy_id} · ${d.county || "?"} · ${d.head ? fmt(d.head, 0) + " head" : "herd ?"}`))))
      : el("div", { class: "grid2" }, el("label", {}, "Latitude", el("input", { type: "number", step: "0.0001", value: p.lat, onchange: set("lat") })),
          el("label", {}, "Longitude", el("input", { type: "number", step: "0.0001", value: p.lon, onchange: set("lon") })));
    blocks.push(el("div", {}, el("h3", {}, "Your dairy"), seg, who,
      el("div", { class: "grid2" },
        p.mode === "location" ? el("label", {}, "Herd (head)", el("input", { type: "number", min: 0, value: p.head ?? "", onchange: set("head") })) : null,
        el("label", {}, "Tons wanted", el("input", { type: "number", min: 0, step: 100, value: p.tons ?? "", onchange: set("tons") }))),
      el("label", {}, "Longest haul, road miles", el("input", { type: "number", min: 1, max: 60, value: p.max_miles ?? 20, onchange: set("max_miles") })),
      el("label", {}, `As of (replays the ${y} season)`, el("input", { type: "date", min: `${y}-08-01`, max: `${y}-11-30`, value: app.ctx.as_of,
        onchange: (e) => { app.ctx.as_of = e.target.value; app.save(); } })),
      el("p", { class: "small faint" }, "Permit ids come from Maryland's dairy permit map; no names are kept.")));
  } else {
    const runs = await api("/api/runs").catch(() => []);
    blocks.push(el("div", {}, el("h3", {}, "Runs"),
      app.ctx.run_id ? el("p", { class: "small" }, "Chat questions use run ", el("b", {}, app.ctx.run_id)) : el("p", { class: "small muted" }, "No run selected: the agent will run the season when you ask."),
      el("ul", { class: "runs" }, runs.slice(0, 12).map((r) => el("li", {}, el("button", { class: r.run_id === app.ctx.run_id ? "active" : "", onclick: () => app.openRun(r.run_id) },
        el("span", { class: "when" }, `${r.when} · ${STATUS[r.status] || r.status}${r.offline ? " · offline" : ""}`), `${r.name} ${r.year ?? ""}`))))));
  }
  side.replaceChildren(...blocks);
};

// --- chat ---------------------------------------------------------------------------------

function messageNode(m) {
  if (m.role === "user") return el("div", { class: "msg user" }, m.content);
  const node = el("div", { class: "msg assistant" });
  if (m.pending) { node.append(el("div", { class: "thinking" }, m.pending)); return node; }
  node.append(mdEl(m.content, "text md"));
  if (m.cards?.length) node.append(el("div", { class: "cards" }, m.cards.map((c) => renderCard(c, { openRun: (id) => app.openRun(id), openMap: (r) => app.openMap(r) }))));
  if (m.meta) node.append(el("div", { class: "meta" }, m.meta));
  return node;
}

function chatView() {
  const role = app.role, thread = el("div", { class: "thread" }), input = el("textarea", { rows: 1, placeholder: "Ask anything about silage, fields, supply or price…" });
  const draw = () => {
    const msgs = app.threads[role];
    thread.replaceChildren(...(msgs.length ? msgs.map(messageNode)
      : [el("div", { class: "intro" }, el("h2", {}, role === "dairy" ? "Find silage near your dairy" : "Ask the silage agent"), el("p", {}, ROLES[role].intro))]));
    thread.scrollTop = thread.scrollHeight;
  };
  const send = async (text) => {
    text = (text ?? input.value).trim(); if (!text) return;
    input.value = "";
    const msgs = app.threads[role];
    msgs.push({ role: "user", content: text });
    const pending = { role: "assistant", pending: app.offline() ? "Working" : "Nemotron is choosing tools" };
    msgs.push(pending); draw();
    try {
      const history = msgs.filter((m) => !m.pending).map((m) => ({ role: m.role, content: m.content }));
      const r = await api("/api/chat", { role, messages: history, context: app.chatContext() });
      if (r.context?.run_id && r.context.run_id !== app.ctx.run_id) { app.ctx.run_id = r.context.run_id; app.refreshSide(); }
      const silage = r.cards.find((c) => c.type === "silage"); if (silage) app.lastSilage = silage.data;
      Object.assign(pending, { pending: null, content: r.reply, cards: r.cards,
        meta: `${r.mode === "offline" ? "Offline rules" : "Nemotron"}${r.tools.length ? ` · tools: ${[...new Set(r.tools)].join(", ")}` : ""} · ${r.grounded ? "every number checked" : "contains [unverified] numbers"}` });
    } catch (e) { Object.assign(pending, { pending: null, content: `Something went wrong: ${e.message}` }); }
    draw();
  };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
  input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 160)}px`; });
  const suggest = el("div", { class: "suggest" }, (app.suggestions[role] || []).map((s) => el("button", { class: "chip", onclick: () => send(s) }, s)));
  draw();
  return el("div", { class: "chat" }, thread, el("div", { class: "composer" }, suggest,
    el("form", { onsubmit: (e) => { e.preventDefault(); send(); } }, input, el("button", { class: "primary", type: "submit" }, "Send")),
    el("div", { class: "hint" }, "Enter to send, Shift+Enter for a new line. Answers use the season and settings on the left. Acres are model estimates, not NASS statistics.")));
}

// --- start --------------------------------------------------------------------------------

document.querySelectorAll(".role-card").forEach((b) => b.addEventListener("click", () => chooseRole(b.dataset.role)));
$("#home").addEventListener("click", goHome);
$("#switch-role").addEventListener("click", goHome);

(async () => {
  try {
    await loadStatus();
    [app.seasons, app.dairies, app.suggestions.dairy, app.suggestions.usda] = await Promise.all([
      api("/api/counties"), api("/api/dairies").catch(() => []), api("/api/suggestions/dairy"), api("/api/suggestions/usda")]);
    const saved = store.get("fv.role", null);
    if (saved && ROLES[saved]) chooseRole(saved);
  } catch (e) {
    $("#landing").append(el("p", { class: "err" }, `Could not reach the server: ${e.message}`));
  }
})();
