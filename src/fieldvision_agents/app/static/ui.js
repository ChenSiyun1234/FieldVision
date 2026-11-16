// Small shared helpers: DOM building, number formats, API calls, minimal markdown, the hover tooltip.

export const $ = (sel, root = document) => root.querySelector(sel);
export const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) n.append(k.nodeType ? k : document.createTextNode(String(k)));
  return n;
};
export const fmt = (x, d = 1) => (typeof x === "number" && isFinite(x)
  ? x.toLocaleString("en-US", { maximumFractionDigits: d }) : "—");
export const money = (x) => (typeof x === "number" ? `$${x.toFixed(2)}` : "—");
export const big = (x) => (typeof x !== "number" ? "—" : Math.abs(x) >= 1e9 ? `${(x / 1e9).toFixed(2)} B`
  : Math.abs(x) >= 1e6 ? `${(x / 1e6).toFixed(2)} M` : fmt(x, 0));
export const STATUS = { accepted: "Accepted", accepted_with_warnings: "Accepted with warnings", abstained: "Abstained",
  incomplete: "Incomplete", pass: "Pass", warn: "Warn", fail: "Fail" };

export async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

// Storage can be blocked (private windows, previews); the page must work without it.
export const store = {
  get(k, fallback) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : fallback; } catch { return fallback; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* ignore */ } },
};

// Headings, bold, italics, bullet lists and paragraphs. Text is escaped before any markup is added.
export function markdown(md) {
  const esc = String(md || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|[\s(])_(.+?)_(?=[\s.,)]|$)/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  return esc.split(/\n{2,}/).map((block) => {
    const h = block.match(/^(#{1,4})\s+(.*)$/s);
    if (h) return `<h${h[1].length + 1}>${inline(h[2])}</h${h[1].length + 1}>`;
    const lines = block.split("\n");
    if (lines.every((l) => /^\s*([-*•]|\d+\.)\s+/.test(l)))
      return `<ul>${lines.map((l) => `<li>${inline(l.replace(/^\s*([-*•]|\d+\.)\s+/, ""))}</li>`).join("")}</ul>`;
    if (lines.length > 1 && lines.every((l) => l.trim().startsWith("|"))) {
      const rows = lines.filter((l) => !/^\s*\|[\s:|-]+\|\s*$/.test(l)).map((l) => l.trim().replace(/^\||\|$/g, "").split("|"));
      return `<div class="scroll-x"><table class="data"><thead><tr>${rows[0].map((c) => `<th>${inline(c.trim())}</th>`).join("")}</tr></thead>`
        + `<tbody>${rows.slice(1).map((r) => `<tr>${r.map((c) => `<td>${inline(c.trim())}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    }
    return `<p>${inline(block).replace(/\n/g, "<br>")}</p>`;
  }).join("");
}
export const mdEl = (md, cls = "md") => { const d = el("div", { class: cls }); d.innerHTML = markdown(md); return d; };

const tip = () => document.getElementById("tip");
export function showTip(evt, html) {
  const t = tip(); t.innerHTML = html; t.hidden = false;
  const x = Math.min(evt.clientX + 14, window.innerWidth - t.offsetWidth - 8);
  const y = Math.min(evt.clientY + 14, window.innerHeight - t.offsetHeight - 8);
  t.style.left = `${x}px`; t.style.top = `${y}px`;
}
export const hideTip = () => { tip().hidden = true; };
