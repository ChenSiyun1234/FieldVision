// FieldVision analyst console: talks to the JSON API in server.py. No build step, no dependencies.

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === "class" ? (n.className = v) : n.setAttribute(k, v);
  for (const k of kids) n.append(k?.nodeType ? k : document.createTextNode(k ?? ""));
  return n;
};
const fmt = (x) => (typeof x === "number" ? x.toLocaleString("en-US", { maximumFractionDigits: 1 }) : "—");
const STATUS = { accepted: "Accepted", accepted_with_warnings: "Accepted with warnings", abstained: "Abstained", incomplete: "Incomplete" };
const SAMPLES = ["How many silage acres, and how sure are we?", "What did the critic flag, and what happened next?",
                 "Why did the agent choose that routing policy?"];

let current = null;     // run_id shown
let online = false;     // NVIDIA key present

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}
const say = (text, kind = "") => { const s = $("#say"); s.textContent = text; s.className = `say ${kind}`; };
const offline = () => !online || document.querySelector('input[name="mode"]:checked')?.value === "offline";

async function loadStatus() {
  const s = await api("/api/status");
  online = s.nemotron.ok;
  const chips = [
    [s.nemotron.ok, s.nemotron.ok ? "Nemotron ready" : "Offline: no NVIDIA key"],
    [s.report_search.ok, `Report search: ${s.report_search.mode}`],
    [s.county_data.ok, `${s.county_data.n} county-seasons`],
    [s.quickstats.ok, "Quick Stats"],
    [s.registry.ok, "Model registry"],
  ];
  $("#status").replaceChildren(...chips.map(([ok, t]) => el("span", { class: `chip ${ok ? "ok" : "off"}` }, t)));
  const radios = document.querySelectorAll('input[name="mode"]');
  radios.forEach((r) => { r.checked = r.value === (online ? "nemotron" : "offline"); });
  radios[0].disabled = !online;
}

async function loadCounties() {
  const list = await api("/api/counties");
  const sel = $("#county");
  sel.replaceChildren(...list.map((c) => el("option", { value: `${c.fips}|${c.year}` },
    `${c.name} · ${c.year}${c.partial ? " (partial)" : ""}`)));
  if (!list.length) { sel.append(el("option", { value: "" }, "No county data: see README")); $("#run").disabled = true; }
}

async function loadRuns() {
  const list = await api("/api/runs");
  $("#runs").replaceChildren(...(list.length ? list.map((r) => {
    const b = el("button", { class: r.run_id === current ? "active" : "" },
      el("span", { class: "when" }, `${r.when} · ${STATUS[r.status] || r.status}${r.offline ? " · offline" : ""}`),
      `${r.name} ${r.year ?? ""}`);
    b.onclick = () => show(r.run_id);
    return el("li", {}, b);
  }) : [el("li", { class: "muted" }, "No runs yet.")]));
}

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
    case "abstain_region": return `Abstained: ${r.reason ?? s.args.reason ?? ""}`;
    case "finish": return `Closed as ${STATUS[s.args.status] || s.args.status}`;
    default: return JSON.stringify(r).slice(0, 140);
  }
}

// Minimal markdown for the reports: headings, bold, italics, paragraphs. Text is escaped first.
function markdown(md) {
  const esc = md.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc.split(/\n{2,}/).map((block) => {
    const b = block.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|\s)_(.+?)_(?=\s|$)/g, "$1<em>$2</em>");
    const h = b.match(/^(#{1,4})\s+(.*)$/s);
    return h ? `<h${h[1].length + 1}>${h[2]}</h${h[1].length + 1}>` : `<p>${b.replace(/\n/g, "<br>")}</p>`;
  }).join("");
}

function render(d) {
  const { result, evidence: ev, steps } = d;
  const view = $("#tpl-run").content.cloneNode(true);
  const est = ev.estimate || {};
  $(".title", view).textContent = `${d.name || ev.county.fips} · ${ev.county.year}`;
  const badge = $(".badge", view);
  badge.textContent = STATUS[result.status] || result.status;
  badge.classList.add(result.status);
  $('[data-f="silage"]', view).textContent = fmt(est.silage_acres);
  $('[data-f="range"]', view).textContent = `range ${fmt(est.silage_acres_low)} – ${fmt(est.silage_acres_high)}`;
  $('[data-f="grain"]', view).textContent = fmt(est.grain_acres);
  $('[data-f="abstained"]', view).textContent = fmt(est.abstained_acres);
  const total = (est.silage_acres || 0) + (est.grain_acres || 0) + (est.abstained_acres || 0);
  if (total) {
    $(".bar .s", view).style.width = `${(100 * (est.silage_acres || 0)) / total}%`;
    $(".bar .g", view).style.width = `${(100 * (est.grain_acres || 0)) / total}%`;
    $(".bar .a", view).style.width = `${(100 * (est.abstained_acres || 0)) / total}%`;
  }
  $('[data-f="route"]', view).textContent =
    `Routing: ${ev.run.policy_sequence.join(" → ") || "none"} · Critic: ${ev.run.critic_sequence.join(" → ") || "not run"}`;
  $('[data-f="adjust"]', view).textContent = est.adjustment ? `Totals: ${est.adjustment}` : "";
  $('[data-f="meta"]', view).textContent =
    `Run ${result.run_id}${result.tokens ? ` · ${fmt(result.tokens)} tokens` : ""}${ev.is_stub ? " · stub tools" : ""} · model estimates, not NASS statistics`;

  $(".steps", view).replaceChildren(...steps.map((s) => el("li", {},
    el("span", { class: "tool" }, s.tool + (s.args.policy ? ` ${s.args.policy}` : "")), " ", stepNote(s),
    ...(s.args.reason ? [el("div", { class: "why" }, `Why: ${s.args.reason}`)] : []))));

  const critic = $(".critic", view);
  if (!ev.critic.length) critic.append(el("p", { class: "muted" }, "The critic did not run."));
  ev.critic.forEach((c, i) => {
    critic.append(el("p", {}, `Check ${i + 1}: `, el("span", { class: `verdict ${c.overall}` }, c.overall),
      ` → ${c.suggested_action}`));
    if (c.summary) critic.append(el("p", { class: "muted" }, c.summary));
    (c.flags || []).forEach((f) => {
      const cites = (f.evidence || []).map((e) => (typeof e === "string" ? e : [e.source, e.page && `p.${e.page}`].filter(Boolean).join(" "))).join("; ");
      critic.append(el("div", { class: `flag ${f.status}` }, el("span", { class: "name" }, f.check), ` (${f.status}) ${f.why || ""}`,
        ...(cites ? [el("div", { class: "small muted" }, `Sources: ${cites}`)] : [])));
    });
  });

  const reportBox = $(".report", view);
  if (d.report_md) { reportBox.innerHTML = markdown(d.report_md); reportBox.classList.remove("muted"); }
  const reportBtn = $(".report-btn", view);
  reportBtn.onclick = async () => {
    reportBtn.disabled = true;
    say(offline() ? "Writing the report from the run record…" : "Nemotron Ultra is writing the report (about a minute)…");
    try {
      const rep = await api(`/api/runs/${result.run_id}/report`, { offline: offline() });
      reportBox.innerHTML = markdown(rep.report_md) +
        `<p class="small muted">${rep.grounding.grounded ? "Every number checked against the evidence." : "Some numbers are marked [unverified]."} Saved to ${rep.path}</p>`;
      reportBox.classList.remove("muted");
      say("Report ready.", "good");
    } catch (e) { say(`Report failed: ${e.message}`, "err"); } finally { reportBtn.disabled = false; }
  };

  const q = $(".question", view), askBtn = $(".ask-btn", view), answers = $(".answers", view);
  const ask = async () => {
    const question = q.value.trim();
    if (!question) return;
    askBtn.disabled = true;
    say("Answering…");
    try {
      const a = await api(`/api/runs/${result.run_id}/ask`, { question, offline: offline() });
      answers.prepend(el("div", { class: "answer" }, el("div", { class: "q" }, question), el("div", {}, a.answer),
        el("div", { class: "src" }, `Evidence: ${(a.sections_used || []).join(", ") || "none"} · ${a.grounded ? "every number checked" : "contains [unverified] numbers"}`)));
      say("Answered.", "good");
    } catch (e) { say(`Question failed: ${e.message}`, "err"); } finally { askBtn.disabled = false; }
  };
  askBtn.onclick = ask;
  q.onkeydown = (e) => { if (e.key === "Enter") ask(); };
  $(".samples", view).replaceChildren(...SAMPLES.map((s) => {
    const b = el("button", { class: "chip" }, s);
    b.onclick = () => { q.value = s; ask(); };
    return b;
  }));
  $("#main").replaceChildren(view);
}

async function show(runId) {
  try {
    say("Opening the run…");
    current = runId;
    render(await api(`/api/runs/${runId}`));
    say("");
    loadRuns();
  } catch (e) { say(`Could not open that run: ${e.message}`, "err"); }
}

$("#run").onclick = async () => {
  const [fips, year] = $("#county").value.split("|");
  if (!fips) return;
  const btn = $("#run");
  btn.disabled = true;
  say(offline() ? "Running the tools with the offline agent…" : "Nemotron is running the county (20–60 seconds)…");
  try {
    const d = await api("/api/runs", { fips, year: Number(year), offline: offline() });
    current = d.result.run_id;
    render(d);
    say(`Done in ${d.result.seconds ?? "?"} s.`, "good");
    loadRuns();
  } catch (e) { say(`Run failed: ${e.message}`, "err"); } finally { btn.disabled = false; }
};

Promise.all([loadStatus(), loadCounties(), loadRuns()]).catch((e) => say(`Could not reach the server: ${e.message}`, "err"));
