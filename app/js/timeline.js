// "As of" control: a slider over the season's cutoff dates whose track doubles as the
// x axis of a small chart of accuracy (and share of confident calls) per cutoff.
// Position n (after the last cutoff) is the full season.

import { isNum, pct, shortDate, esc } from "./format.js";

const THUMB = 18; // px, must match the CSS thumb width so points sit over the thumb

export class Timeline {
  constructor(el, onChange) {
    this.el = el;
    this.onChange = onChange;
    this.timer = null;
    el.addEventListener("input", (e) => {
      if (e.target.matches("input[type=range]")) { this.stop(); this.onChange(+e.target.value); }
    });
    el.addEventListener("click", (e) => {
      const b = e.target.closest("[data-tl]");
      if (!b) return;
      if (b.dataset.tl === "play") this.timer ? this.stop() : this.play();
      if (b.dataset.tl === "full") { this.stop(); this.onChange(this.n); }
    });
  }

  /** asof: asof.json; full: region metrics for the full season; index: 0..n */
  render(asof, full, index, { hasLabels }) {
    if (!asof || !asof.cutoffs?.length) { this.el.hidden = true; this.stop(); return; }
    this.el.hidden = false;
    this.asof = asof;
    this.n = asof.cutoffs.length;
    this.index = index;
    const all = [...asof.metrics, full];
    const acc = all.map((m) => (hasLabels ? m?.all?.accuracy ?? null : null));
    const conf = all.map((m) => (isNum(m?.low_evidence_share) ? 1 - m.low_evidence_share : null));
    const cur = all[index];
    const label = index === this.n ? "Full season" : `As of ${shortDate(asof.cutoffs[index])}`;
    const read = [
      hasLabels ? `accuracy <b>${pct(cur?.all?.accuracy)}</b>` : null,
      `<b>${pct(conf[index])}</b> confident calls`,
    ].filter(Boolean).join(" · ");

    const width = Math.max(240, this.el.clientWidth - 24);
    const h = 64, pad = THUMB / 2;
    const x = (i) => pad + (i / this.n) * (width - 2 * pad);
    const y = (v) => 6 + (1 - v) * (h - 12);
    const line = (vals, attrs) => {
      let d = "";
      vals.forEach((v, i) => { if (isNum(v)) d += `${d && isNum(vals[i - 1]) ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`; });
      return d ? `<path d="${d}" fill="none" ${attrs}/>` : "";
    };
    const dots = (vals, fill) => vals.map((v, i) => (isNum(v) ? `<circle cx="${x(i)}" cy="${y(v)}" r="${i === index ? 4.5 : 2.6}" fill="${fill}"/>` : "")).join("");
    const svg = `<svg viewBox="0 0 ${width} ${h}" width="100%" height="${h}" preserveAspectRatio="none" aria-hidden="true">
      <line x1="${pad}" x2="${width - pad}" y1="${y(0.5)}" y2="${y(0.5)}" stroke="#2a3541" stroke-dasharray="3 3"/>
      <line x1="${pad}" x2="${width - pad}" y1="${y(1)}" y2="${y(1)}" stroke="#2a3541"/>
      <line x1="${x(index)}" x2="${x(index)}" y1="0" y2="${h}" stroke="#f0e442" stroke-width="2"/>
      ${line(conf, 'stroke="#9fb0c0" stroke-width="1.6" stroke-dasharray="4 3"')}
      ${line(acc, 'stroke="#eef2f6" stroke-width="2.4"')}
      ${dots(conf, "#9fb0c0")}${dots(acc, "#eef2f6")}
    </svg>`;
    const first = shortDate(asof.cutoffs[0]);
    this.el.innerHTML = `
      <div class="tl-head">
        <button type="button" class="tl-play" data-tl="play" title="Play through the season">${this.timer ? "❚❚" : "▶"}</button>
        <div class="tl-title">${esc(label)}</div>
        <div class="tl-read">${read}</div>
      </div>
      <div class="tl-chart">${svg}</div>
      <input type="range" min="0" max="${this.n}" step="1" value="${index}" aria-label="Use satellite data up to this date">
      <div class="tl-foot"><span>${esc(first)}</span><span class="tl-key">${hasLabels ? '<i class="k-acc"></i>accuracy ' : ""}<i class="k-conf"></i>confident calls</span><button type="button" data-tl="full" class="${index === this.n ? "on" : ""}">Full season</button></div>`;
  }

  play() {
    if (!this.asof) return;
    let i = this.index >= this.n ? 0 : this.index;
    this.timer = setInterval(() => {
      i += 1;
      if (i > this.n) { this.stop(); return; }
      this.onChange(i);
    }, 1100);
    this.onChange(i);
  }

  stop() {
    if (this.timer) { clearInterval(this.timer); this.timer = null; if (this.asof) this.onChange(this.index); }
  }
}
