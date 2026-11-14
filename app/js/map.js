// Leaflet map: field polygons (dots when zoomed out), optional county choropleth.
// Works without the network: if satellite tiles fail, fields stay on a plain background.

/* global L */
import { pct, esc, cap } from "./format.js";
import { confidence } from "./evidence.js";

export const COLORS = {
  silage: "#56b4e9",
  grain: "#e69f00",
  wrong: "#f25cd6",
  ok: "#8b97a5",
  none: "#cfd8e3",
};
// sequential ramp for county silage share (light = low, saturated blue = high)
export const RAMP = ["#22323f", "#24506b", "#2a7299", "#3f95c8", "#7cc3ef"];
const DOT_ZOOM = 13.5;
const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

export function rampColor(v) {
  if (typeof v !== "number" || !Number.isFinite(v)) return null;
  return RAMP[Math.min(RAMP.length - 1, Math.floor(v * RAMP.length))];
}

export function fieldStyle(p, mode) {
  if (mode === "check") {
    if (p.correct === false) return { color: "#ffffff", weight: 2, opacity: 1, dashArray: null, fillColor: COLORS.wrong, fillOpacity: 0.95 };
    if (p.correct === true) return { color: "#dfe6ee", weight: 1, opacity: 0.7, dashArray: null, fillColor: COLORS.ok, fillOpacity: 0.45 };
    return { color: COLORS.none, weight: 1.5, opacity: 0.9, dashArray: "4 3", fillColor: COLORS.none, fillOpacity: 0.04 };
  }
  if (!p.pred) return { color: COLORS.none, weight: 1.5, opacity: 0.9, dashArray: "4 3", fillColor: COLORS.none, fillOpacity: 0.04 };
  const c = COLORS[p.pred];
  if (p.low === true) return { color: c, weight: 2.5, opacity: 1, dashArray: "5 4", fillColor: c, fillOpacity: 0.12 };
  return { color: "#ffffff", weight: 1, opacity: 0.85, dashArray: null, fillColor: c, fillOpacity: 0.85 };
}

function tipHtml(p) {
  const conf = confidence(p);
  let line;
  if (!p.pred) line = "No prediction";
  else line = `${cap(p.pred)} ${pct(conf)}${p.low === true ? " · low evidence" : ""}`;
  let decl = "";
  if (p.label) decl = `<br>Declared: ${p.label}${p.correct === false ? " ✗" : ""}`;
  return `<b>${esc(p.field_id)}</b><br>${line}${decl}`;
}

export class FieldMap {
  constructor(el, { onSelect, onCounty }) {
    this.el = el;
    this.onSelect = onSelect;
    this.onCounty = onCounty;
    this.mode = "pred";
    this.map = L.map(el, { zoomSnap: 0.25, zoomControl: false, attributionControl: true, minZoom: 3, maxZoom: 19 });
    L.control.zoom({ position: "bottomright" }).addTo(this.map);
    this.map.createPane("counties").style.zIndex = 350;
    this.map.createPane("dots").style.zIndex = 450;
    // SVG, not canvas: stacked canvas panes swallow each other's clicks
    this.fieldRenderer = L.svg({ padding: 0.3 });
    this.dotRenderer = L.svg({ pane: "dots", padding: 0.3 });
    this.countyRenderer = L.svg({ pane: "counties", padding: 0.3 });
    this.tileErrors = 0;
    this.tileLoads = 0;
    this.sat = L.tileLayer(ESRI, {
      maxZoom: 19, maxNativeZoom: 18,
      attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics",
    });
    this.sat.on("tileerror", () => { this.tileErrors++; this._tileNote(); });
    this.sat.on("tileload", () => { this.tileLoads++; this._tileNote(); });
    this.setBase("sat");
    this.map.setView([48, 10], 4);
    this.map.on("zoomend", () => this._toggleDots());
    this.layers = [];
    this.byId = new Map();
    this.selectedId = null;
    this.highlight = null;
  }

  _tileNote() {
    const note = document.getElementById("map-note");
    if (!note) return;
    const offline = this.base === "sat" && this.tileErrors > 4 && this.tileLoads === 0;
    note.hidden = !offline;
    if (offline) note.textContent = "Satellite basemap unavailable (offline?). Fields are drawn on a plain background.";
  }

  setBase(base) {
    this.base = base;
    if (base === "sat") { this.sat.addTo(this.map); this.el.classList.remove("plain"); }
    else { this.map.removeLayer(this.sat); this.el.classList.add("plain"); }
    this._tileNote();
  }

  clear() {
    for (const l of this.layers) this.map.removeLayer(l);
    this.layers = [];
    this.byId.clear();
    if (this.highlight) { this.map.removeLayer(this.highlight); this.highlight = null; }
    this.counties = null;
    this.countyById = new Map();
  }

  /** padding: {topLeft:[x,y], bottomRight:[x,y]} keeps fields clear of overlays */
  setFields(features, padding = null) {
    this.clear();
    const valid = features.filter((f) => f.geometry);
    this.polys = L.geoJSON({ type: "FeatureCollection", features: valid }, {
      renderer: this.fieldRenderer,
      style: (f) => fieldStyle(f.properties, this.mode),
      onEachFeature: (f, layer) => {
        const id = f.properties.field_id;
        layer.bindTooltip(() => tipHtml(this.byId.get(id).props), { className: "fv-tip", sticky: true, direction: "top" });
        layer.on("click", () => this.onSelect(f.properties.field_id));
        this.byId.set(f.properties.field_id, { poly: layer, props: f.properties });
      },
    });
    this.dots = L.layerGroup();
    for (const f of valid) {
      const p = f.properties;
      if (typeof p.lat !== "number" || typeof p.lon !== "number") continue;
      const dot = L.circleMarker([p.lat, p.lon], { pane: "dots", renderer: this.dotRenderer, radius: 6.5, ...fieldStyle(p, this.mode) });
      dot.bindTooltip(() => tipHtml(this.byId.get(p.field_id).props), { className: "fv-tip", direction: "top", offset: [0, -6] });
      dot.on("click", () => this.onSelect(p.field_id));
      this.dots.addLayer(dot);
      const entry = this.byId.get(p.field_id);
      if (entry) entry.dot = dot;
    }
    this.polys.addTo(this.map);
    this.layers.push(this.polys, this.dots);
    if (valid.length) {
      const b = this.polys.getBounds();
      const pad = padding ? { paddingTopLeft: padding.topLeft, paddingBottomRight: padding.bottomRight } : { padding: [40, 40] };
      if (b.isValid()) this.map.fitBounds(b, { ...pad, maxZoom: 15 });
    }
    this._toggleDots();
  }

  _toggleDots() {
    if (!this.dots) return;
    const show = this.map.getZoom() < DOT_ZOOM;
    if (show && !this.map.hasLayer(this.dots)) this.dots.addTo(this.map);
    if (!show && this.map.hasLayer(this.dots)) this.map.removeLayer(this.dots);
    this._drawHighlight();
  }

  /** Swap the properties used for styling and tooltips (e.g. predictions as of a cutoff). */
  setProps(byId) {
    for (const [id, e] of this.byId) {
      const p = byId.get(id);
      if (p) e.props = p;
    }
    this.setMode(this.mode);
  }

  setMode(mode) {
    this.mode = mode;
    for (const { poly, dot, props } of this.byId.values()) {
      const s = fieldStyle(props, mode);
      poly.setStyle(s);
      if (dot) dot.setStyle(s);
    }
  }

  select(id, { pan = true } = {}) {
    this.selectedId = id;
    this._drawHighlight();
    const e = id && this.byId.get(id);
    if (e && pan) {
      const b = e.poly.getBounds();
      if (!this.map.getBounds().pad(-0.1).contains(b)) this.map.panTo(b.getCenter());
    }
  }

  _drawHighlight() {
    if (this.highlight) { this.map.removeLayer(this.highlight); this.highlight = null; }
    const e = this.selectedId && this.byId.get(this.selectedId);
    if (!e) return;
    const dotsOn = this.dots && this.map.hasLayer(this.dots);
    if (dotsOn && e.dot) {
      this.highlight = L.circleMarker(e.dot.getLatLng(), { pane: "dots", renderer: this.dotRenderer, radius: 12, color: "#ffffff", weight: 3, fill: false, interactive: false });
    } else {
      this.highlight = L.geoJSON(e.poly.toGeoJSON(), { pane: "dots", renderer: this.dotRenderer, interactive: false, style: { color: "#ffffff", weight: 4, fill: false, dashArray: null } });
    }
    this.highlight.addTo(this.map);
  }

  setCounties(shapes, rowsByFips) {
    if (this.counties) { this.map.removeLayer(this.counties); this.counties = null; }
    this.countyById = new Map();
    if (!shapes) return;
    this.counties = L.geoJSON(shapes, {
      pane: "counties",
      renderer: this.countyRenderer,
      style: (f) => {
        const r = rowsByFips.get(f.properties.fips);
        const c = r ? rampColor(r.pred_silage_share) : null;
        return { color: "#cfd8e3", weight: 1.2, opacity: 0.8, fillColor: c || "#000", fillOpacity: c ? 0.55 : 0 };
      },
      onEachFeature: (f, layer) => {
        const r = rowsByFips.get(f.properties.fips);
        if (r) {
          layer.bindTooltip(`<b>${esc(r.name)}, ${esc(r.state)}</b><br>Predicted silage ${pct(r.pred_silage_share)}<br>USDA silage ${pct(r.usda_silage_share)}`, { className: "fv-tip", sticky: true });
          layer.on("click", () => this.onCounty && this.onCounty(r.fips));
        }
        this.countyById.set(f.properties.fips, layer);
      },
    });
  }

  showCounties(on) {
    if (!this.counties) return;
    if (on && !this.map.hasLayer(this.counties)) this.counties.addTo(this.map);
    if (!on && this.map.hasLayer(this.counties)) this.map.removeLayer(this.counties);
  }

  highlightCounty(fips) {
    for (const [k, layer] of this.countyById) {
      layer.setStyle({ weight: k === fips ? 3.5 : 1.2, color: k === fips ? "#ffffff" : "#cfd8e3" });
    }
  }

  zoomToCounty(fips) {
    const l = this.countyById.get(fips);
    if (l) this.map.fitBounds(l.getBounds(), { padding: [30, 30] });
  }

  invalidate() { this.map.invalidateSize(); }
}
