// FieldVision results deck. Dark "harvest" theme shared with the app: field green surfaces, silage green,
// corn gold for grain and highlights, sky blue for USDA's own figures. Charts are pictures (they were drawn
// from the repository's results by figs.py); tables are native.
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const DIR = __dirname;
const DIMS = JSON.parse(fs.readFileSync(path.join(DIR, "dims.json")));
const OUT = process.argv[2] || path.join(DIR, "FieldVision_results.pptx");

const C = { bg: "0B1410", panel: "13201A", panel2: "1A2A22", line: "2A3D33", text: "EEF3EE", muted: "A7B8AD", faint: "73887B",
  silage: "14A37F", grain: "C98500", usda: "4B8FE2", gold: "F2C14E", fail: "FF7A6B", ok: "57D18F", white: "FFFFFF" };
const THEME = { name: "FieldVision Harvest", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: { dk1: C.bg, lt1: C.text, dk2: C.panel, lt2: C.muted, accent1: C.silage, accent2: C.grain, accent3: C.usda,
    accent4: C.gold, accent5: C.faint, accent6: C.fail, hlink: C.usda, folHlink: "9BE3C6" } };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                         // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "FieldVision: chopped for silage or combined for grain?";
pres.author = "Team FieldVision";
pres.subject = "Hack the Harvest 2026 results";
const W = 13.333, H = 7.5, M = 0.6;

// --- layouts -------------------------------------------------------------------------------
const footer = { text: { text: "FieldVision · Hack the Harvest 2026", options: { x: M, y: H - 0.42, w: 6, h: 0.3, fontSize: 10, color: C.faint, margin: 0 } } };
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.bg },
  objects: [footer,
    { placeholder: { options: { name: "eyebrow", type: "body", x: M, y: 0.38, w: 9, h: 0.3, fontSize: 12, bold: true, color: C.gold, charSpacing: 2, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.66, w: W - 2 * M, h: 0.9, fontSize: 30, bold: true, color: C.text, margin: 0, valign: "top", align: "left" }, text: "" } }],
  slideNumber: { x: W - M - 0.5, y: H - 0.42, w: 0.5, h: 0.3, fontSize: 10, color: C.faint, align: "right" },
});
pres.defineSlideMaster({
  title: "SECTION", background: { color: C.panel },
  objects: [footer,
    { placeholder: { options: { name: "eyebrow", type: "body", x: M, y: 2.5, w: 10, h: 0.4, fontSize: 14, bold: true, color: C.gold, charSpacing: 3, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.95, w: 11.5, h: 1.6, fontSize: 40, bold: true, color: C.text, margin: 0, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.6, w: 10.5, h: 1.2, fontSize: 18, color: C.muted, margin: 0, valign: "top" }, text: "" } }],
  slideNumber: { x: W - M - 0.5, y: H - 0.42, w: 0.5, h: 0.3, fontSize: 10, color: C.faint, align: "right" },
});
pres.defineSlideMaster({ title: "BLANK_DARK", background: { color: C.bg }, objects: [] });

// --- helpers ---------------------------------------------------------------------------------
let n = 0;
function slide(layout, section, eyebrow, title, notes) {
  const s = pres.addSlide({ masterName: layout, sectionTitle: section });
  if (eyebrow) s.addText(eyebrow.toUpperCase(), { placeholder: "eyebrow" });
  if (title) s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  n++;
  return s;
}
const aspect = (p) => { const d = DIMS[p]; return d[1] / d[0]; };
/** Picture fitted inside a box, keeping its aspect; optional rounded panel behind it. */
function pic(s, p, x, y, w, h, { panel = false, fill = C.panel, pad = 0.12, align = "center", alt } = {}) {
  const r = aspect(p);
  let iw = w - 2 * (panel ? pad : 0), ih = iw * r;
  const maxH = h - 2 * (panel ? pad : 0);
  if (ih > maxH) { ih = maxH; iw = ih / r; }
  const ix = align === "left" ? x + (panel ? pad : 0) : x + (w - iw) / 2, iy = y + (h - ih) / 2;
  if (panel) s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: ix - pad, y: iy - pad, w: iw + 2 * pad, h: ih + 2 * pad, rectRadius: 0.12,
    fill: { color: fill }, line: { color: C.line, width: 0.75 }, objectName: "panel" });
  s.addImage({ path: path.join(DIR, p), x: ix, y: iy, w: iw, h: ih, altText: alt || path.basename(p) });
  return { x: ix, y: iy, w: iw, h: ih };
}
function card(s, x, y, w, h, fill = C.panel) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.14, fill: { color: fill }, line: { color: C.line, width: 0.75 }, objectName: "card" });
}
function stat(s, x, y, w, value, label, color = C.text, size = 40) {
  s.addText(value, { x, y, w, h: 0.75, fontSize: size, bold: true, color, margin: 0, isTextBox: true, valign: "bottom" });
  s.addText(label, { x, y: y + 0.8, w, h: 0.75, fontSize: 13, color: C.muted, margin: 0, isTextBox: true, valign: "top" });
}
function text(s, t, x, y, w, h, opts = {}) {
  s.addText(t, { x, y, w, h, fontSize: 15, color: C.text, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 6, ...opts });
}
const bullets = (items, opts = {}) => items.map((t, i) => ({ text: t, options: { bullet: { indent: 16 }, breakLine: i < items.length - 1, ...opts } }));
function caption(s, t, x, y, w) { text(s, t, x, y, w, 0.5, { fontSize: 11, color: C.faint }); }
function chip(s, t, x, y, w, color) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.34, rectRadius: 0.17, fill: { color: C.panel2 }, line: { color, width: 1 } });
  s.addText(t, { x, y, w, h: 0.34, fontSize: 11.5, color, align: "center", valign: "middle", margin: 0, bold: true, isTextBox: true });
}
function table(s, rows, x, y, w, colW, opts = {}) {
  const head = rows[0].map((h) => ({ text: h, options: { bold: true, color: C.muted, fill: { color: C.panel2 }, fontSize: 12 } }));
  const body = rows.slice(1).map((r, i) => r.map((c) => (typeof c === "object" && c.text !== undefined ? c
    : { text: String(c), options: { color: C.text, fill: { color: i % 2 ? C.bg : C.panel }, fontSize: 12.5 } })));
  s.addTable([head, ...body], { x, y, w, colW, border: { type: "solid", pt: 0.5, color: C.line }, fontFace: "Calibri",
    margin: [3, 6, 3, 6], valign: "middle", rowH: opts.rowH || 0.36, ...opts });
}

// =============================================================================================
// 1. Title
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "BLANK_DARK", sectionTitle: "Opening" }); n++;
  s.addImage({ path: path.join(DIR, "assets/hero_fields.jpg"), x: 0, y: 0, w: W, h: W * aspect("assets/hero_fields.jpg"), altText: "Two corn fields seen by Sentinel-2 from July to October 2022" });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: H, fill: { color: C.bg, transparency: 18 }, line: { type: "none" } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 3.2, w: W, h: H - 3.2, fill: { color: C.bg }, line: { type: "none" } });
  s.addText("HACK THE HARVEST 2026 · USDA NASS CHALLENGE", { x: M, y: 3.55, w: 10, h: 0.35, fontSize: 14, bold: true, color: C.gold, charSpacing: 3, margin: 0, isTextBox: true });
  s.addText("Chopped for silage, or combined for grain?", { x: M, y: 3.95, w: 12, h: 1.0, fontSize: 46, bold: true, color: C.text, margin: 0, isTextBox: true });
  s.addText("FieldVision reads every corn field from free satellite images, explains each call, checks itself against USDA, and puts the answer in front of a dairy owner and a NASS analyst.",
    { x: M, y: 5.05, w: 10.8, h: 0.9, fontSize: 18, color: C.muted, margin: 0, isTextBox: true });
  chip(s, "Silage: cut green, within days", M, 6.25, 3.0, C.silage);
  chip(s, "Grain: dries down first", M + 3.15, 6.25, 2.5, C.grain);
  s.addNotes("FieldVision answers one question for every corn field in the country: was it chopped green for silage, or left to dry and combined for grain? Behind me are two real fields from Frederick County, Maryland, seen from space from July to October. The top one was cut green in early September, silage. The bottom one browned first and was cut in October, grain.");
}

// 2. Why it matters
{
  const s = slide("CONTENT", "Opening", "Why it matters", "Every silage acre is an acre missing from the grain supply",
    "USDA counts about six million acres of corn silage a year, against 91 million for grain. Silage acres never reach the grain supply, so they move stocks and price. But the split is only published in January, after the season, from surveys. Dairies need to know where silage is during the season. We wanted a field-by-field answer, from public data, with reasons.");
  const cards = [["6.19 M", "acres of corn cut for silage in the US, 2025 (USDA)", C.silage], ["90.9 M", "acres harvested for grain, 2025 (USDA)", C.grain],
    ["January", "when NASS publishes the silage / grain split, months after the harvest", C.usda], ["0", "field-level silage maps from public data before this project", C.gold]];
  cards.forEach(([v, l, col], i) => { const x = M + i * 3.05; card(s, x, 1.95, 2.85, 2.35); stat(s, x + 0.25, 2.1, 2.4, v, l, col, 38); });
  text(s, bullets(["A dairy short of feed needs standing corn it can still buy, near it, before the chopper rolls.",
    "A NASS analyst needs county acres with a range, field reasons and a way to check them.",
    "The challenge: public satellite data only, an answer earlier than January, and an explanation a skeptic accepts."]), M, 4.65, 12, 2, { fontSize: 17 });
}

// 3. Traps
{
  const s = slide("CONTENT", "Opening", "The problem", "Four traps, and only 44 labelled US fields",
    "The organisers warned about four traps. Clouds can hide the cut. Some grain is cut early too, so the date alone misleads. Silage is rare, so always saying grain looks good. And double crops follow another calendar. On top of that, the only labelled US fields are 44 field-years at one research farm in Beltsville. Too few to train on, so we kept them for testing only.");
  const traps = [["Clouds", "The harvest can fall between two clear images, weeks apart."], ["Early is not silage", "Some grain is cut early too. The date alone misleads."],
    ["Silage is rare", "About 6 to 8% of US corn. Always saying grain already looks right."], ["Double crops", "Corn sown after a small grain follows another calendar."]];
  traps.forEach(([t, d], i) => { const x = M + i * 3.05; card(s, x, 1.95, 2.85, 2.0);
    text(s, t, x + 0.25, 2.15, 2.4, 0.4, { fontSize: 19, bold: true, color: C.gold }); text(s, d, x + 0.25, 2.65, 2.4, 1.2, { fontSize: 14, color: C.muted }); });
  card(s, M, 4.35, W - 2 * M, 2.3, C.panel2);
  s.addText("44", { x: M + 0.35, y: 4.55, w: 2.2, h: 1.9, fontSize: 110, bold: true, color: C.silage, margin: 0, isTextBox: true, valign: "middle" });
  text(s, [{ text: "labelled US field-years exist: one research farm in Beltsville, Maryland, 2021–2023.", options: { bold: true, breakLine: true, fontSize: 20 } },
    { text: "Too few to train on. We trained on 8,000 European fields instead, and kept these 44 for testing only: scored once per model, each scoring written in a log first.", options: { color: C.muted, fontSize: 16 } }],
    M + 2.8, 4.85, 9, 1.6);
}

// 4. The signal
{
  const s = slide("CONTENT", "Opening", "The signal", "Silage falls off a cliff. Grain fades out",
    "This is the whole idea in two pictures. Same county, same season. The top field is still green on the last look before the cut and bare right after: a cliff. That is a chopper. The bottom field browns for weeks and is then combined: a fade. We measure exactly this from Sentinel-2: when the green drops, how green and moist it was just before, and how much residue is left after.");
  text(s, "Called silage: green on 31 August, cut around 2 September, green again within weeks (a cover crop)", M, 1.7, 12, 0.35, { fontSize: 15, bold: true, color: C.silage });
  pic(s, "assets/pic_frederick_silage.png", M, 2.05, W - 2 * M, 2.25, { panel: true, fill: "FCFCFA" });
  text(s, "Called grain: brown by late September, cut by 7 October", M, 4.45, 12, 0.35, { fontSize: 15, bold: true, color: C.grain });
  pic(s, "assets/pic_frederick_grain.png", M, 4.8, W - 2 * M, 2.25, { panel: true, fill: "FCFCFA" });
}

// 5. Data
{
  const s = slide("CONTENT", "Opening", "The data we built", "8,000 labelled fields in Europe, 18,000 unlabelled in the US",
    "We built our own data. 8,000 labelled fields from farm registries in Austria, France and the Netherlands, where farmers declare silage or grain. About 18,000 unlabelled US corn fields in Maryland and Michigan, from USDA's crop boundaries. For each field we keep only the pixels inside its outline, on every clear Sentinel-2 date from April to November. The map is all 1,363 corn fields of Kent County.");
  const rows = [["8,000", "labelled European fields", "Farm registries: Austria 2022–24, France 2022, Netherlands 2025. Silage or grain as declared.", C.silage],
    ["18,000", "unlabelled US corn fields", "Four Maryland and two Michigan counties (2022), a random Maryland 2025 sample.", C.grain],
    ["44", "labelled US field-years", "Beltsville 2021–2023. Testing only, never training.", C.usda]];
  rows.forEach(([v, l, d, col], i) => { const y = 1.85 + i * 1.6;
    s.addText(v, { x: M, y, w: 2.3, h: 0.8, fontSize: 40, bold: true, color: col, margin: 0, isTextBox: true });
    text(s, l, M + 2.4, y + 0.05, 4.8, 0.4, { fontSize: 18, bold: true }); text(s, d, M + 2.4, y + 0.5, 4.8, 0.9, { fontSize: 14, color: C.muted }); });
  pic(s, "assets/map_kent.png", 8.1, 1.7, 4.65, 4.9, { panel: true, fill: "FCFCFA" });
  caption(s, "Kent County, Maryland, 2022: every corn field. Green: called silage. Yellow: called grain. Grey: not judged.", 8.1, 6.65, 4.65);
}

// 6. Architecture
pres.addSection({ title: "How it works" });
{
  const s = slide("CONTENT", "How it works", "Architecture", "Code computes every number; Nemotron routes, checks, explains",
    "Two lanes. Offline, once: features from European registry fields, five model families compared by holding out whole countries, the winner frozen in a registry, then tested once on Beltsville. Online, per county-season: a Nemotron supervisor calls code tools to scout, classify and add up; a critic checks the total against USDA history; at most one rerun; then a report with every number checked. The same tools serve two people: a dairy owner and a NASS analyst.");
  const box = (x, y, w, h, t, sub, col) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: C.panel }, line: { color: col, width: 1.5 } });
    s.addText([{ text: t, options: { bold: true, fontSize: 14, color: C.text, breakLine: true } }, { text: sub, options: { fontSize: 11.5, color: C.muted } }],
      { x: x + 0.12, y, w: w - 0.24, h, align: "center", valign: "middle", margin: 0, isTextBox: true });
  };
  const arrow = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color: C.faint, width: 1.25, endArrowType: "triangle" } });
  text(s, "OFFLINE · ONCE", M, 1.7, 5, 0.3, { fontSize: 12, bold: true, color: C.muted, charSpacing: 2 });
  text(s, "ONLINE · PER COUNTY-SEASON", 5.0, 1.7, 5, 0.3, { fontSize: 12, bold: true, color: C.muted, charSpacing: 2 });
  const ox = M, ow = 3.7, bh = 0.78;
  box(ox, 2.05, ow, bh, "8,000 EU registry fields", "Sentinel-2 per field, Apr–Nov", C.line);
  box(ox, 3.05, ow, bh, "Feature engine", "same code online and offline", C.silage);
  box(ox, 4.05, ow, bh, "5 model families", "chosen leaving whole countries out", C.silage);
  box(ox, 5.05, ow, bh, "Frozen registry → Beltsville test", "logistic, 6 numbers · 42 of 44", C.line);
  [2.83, 3.83, 4.83].forEach((y) => arrow(ox + ow / 2, y, ox + ow / 2, y + 0.22));
  const nx = 5.0, nw = 4.1;
  box(nx, 2.05, nw, bh, "Supervisor agent", "Nemotron ReAct · guardrails · audit log", C.usda);
  box(nx, 3.05, nw, bh, "Code tools", "scout · route · classify · add up", C.silage);
  box(nx, 4.05, nw, bh, "Critic", "Quick Stats checks decide · Nemotron explains with NASS pages", C.usda);
  box(nx, 5.05, nw, bh, "Reporter + Q&A", "writer, editor, every number checked", C.usda);
  [2.83, 3.83, 4.83].forEach((y) => arrow(nx + nw / 2, y, nx + nw / 2, y + 0.22));
  arrow(ox + ow, 5.44, nx, 3.44);
  s.addText("one rerun at most", { x: nx + nw + 0.12, y: 3.85, w: 1.6, h: 0.3, fontSize: 11, color: C.gold, margin: 0, isTextBox: true });
  s.addShape(pres.shapes.LINE, { x: nx + nw + 0.05, y: 2.44, w: 0, h: 2.0, line: { color: C.gold, width: 1.25, dashType: "dash", beginArrowType: "triangle" } });
  box(10.25, 2.05, 2.5, 1.7, "Dairy owner", "silage finder · map · chat", C.grain);
  box(10.25, 4.12, 2.5, 1.7, "NASS analyst", "chat · runs · dashboard · US map", C.grain);
  arrow(nx + nw, 5.44, 10.25, 4.97); arrow(nx + nw, 3.44, 10.25, 2.9);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.15, w: 0.22, h: 0.22, rectRadius: 0.04, fill: { color: C.panel }, line: { color: C.silage, width: 1.5 } });
  text(s, "code: the numbers", M + 0.3, 6.13, 2.2, 0.3, { fontSize: 12, color: C.muted });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 2.5, y: 6.15, w: 0.22, h: 0.22, rectRadius: 0.04, fill: { color: C.panel }, line: { color: C.usda, width: 1.5 } });
  text(s, "Nemotron: route, explain, answer", M + 2.8, 6.13, 3.2, 0.3, { fontSize: 12, color: C.muted });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 6.0, y: 6.15, w: 0.22, h: 0.22, rectRadius: 0.04, fill: { color: C.panel }, line: { color: C.grain, width: 1.5 } });
  text(s, "the people it serves", M + 6.3, 6.13, 3, 0.3, { fontSize: 12, color: C.muted });
}

// 7. Field-level result
pres.addSection({ title: "Results" });
{
  const s = slide("CONTENT", "Results", "Field level", "Trained only in Europe: 42 of 44 US fields right",
    "We trained only on European fields, then scored the organisers' 44 labelled field-years once. The small model, six numbers and a logistic regression, got 42 right: it found 11 of the 12 silage fields and called one grain field silage. Always saying grain gets 32. The big boosted-tree model gets 37. A published method we reproduced, TempDiff, gets 22.");
  pic(s, "figs/f_field_test.png", M, 1.75, 8.5, 4.4, { panel: true });
  card(s, 9.35, 1.75, 3.4, 4.4);
  text(s, "SMALL MODEL ON THE 44", 9.6, 1.95, 3, 0.3, { fontSize: 12, bold: true, color: C.muted, charSpacing: 2 });
  table(s, [["", "called silage", "called grain"], ["silage (12)", { text: "11", options: { bold: true, color: C.silage, fill: { color: C.panel } } }, "1"],
    ["grain (32)", "1", { text: "31", options: { bold: true, color: C.grain, fill: { color: C.bg } } }]], 9.6, 2.4, 2.9, [1.0, 0.95, 0.95]);
  stat(s, 9.6, 3.75, 2.9, "0.951", "ROC-AUC on these 44 (0.904 for the timing rule)", C.text, 30);
  text(s, "Each model was scored once on this set, and each scoring written in a log first.", 9.6, 5.25, 2.9, 0.8, { fontSize: 12, color: C.faint });
  caption(s, "Source: handoff/results/REPORT.md (zero-shot evaluation, run 0484d7e43267).", M, 6.3, 8);
}

// 8. Why simple won
{
  const s = slide("CONTENT", "Results", "Transfer", "Simple won because it travels to countries it never saw",
    "Why the small model? We held out each country in turn and trained on the others. The small logistic model has the best mean, 0.765. The bigger boosted trees are close in Europe but lose more on the US farm. The timing rule alone is clearly worse. Every family is weakest on the Netherlands, where labels are declared in spring.");
  pic(s, "figs/f_loco.png", M, 1.75, 8.6, 4.5, { panel: true });
  card(s, 9.45, 1.75, 3.3, 4.5);
  text(s, bullets(["Rule chosen before the test: highest mean ROC-AUC with a whole country held out; ties go to the simpler family.",
    "Winner: logistic on 6 numbers, 0.765.", "Every family is weakest on the Netherlands (labels declared in spring).", "The six numbers: harvest timing, greenness and moisture before the drop, residue after."], { fontSize: 14 }), 9.65, 1.95, 2.95, 4.2, { color: C.muted });
}

// 9. Counties
{
  const s = slide("CONTENT", "Results", "From fields to counties", "Adding up was the hard part. We tested it honestly",
    "Field calls are not county acres. Our first county total in Queen Anne's said 22 percent silage where the Census says 1.3: false alarms on a lot of grain add up. So we only count fields whose harvest we saw clearly, and add up by harvest window. We wrote pass and fail lines before opening each test county. Then the clean test: two Michigan counties we had never looked at. 2.1 percent against 1.8, and 30 against 24. Passed.");
  pic(s, "figs/f_counties.png", M, 1.75, 8.6, 4.5, { panel: true });
  card(s, 9.45, 1.75, 3.3, 4.5);
  const steps = [["First total", "Queen Anne's read 22.4% for a Census 1.3%. Failed.", C.fail], ["Fix 1", "Count only fields whose cut was seen within 12 days.", C.gold],
    ["Fix 2", "A reference-county correction failed its test; dropped.", C.fail], ["Clean test", "Two Michigan counties never looked at: passed.", C.ok]];
  steps.forEach(([t, d, col], i) => { const y = 1.95 + i * 1.05;
    text(s, t, 9.65, y, 2.9, 0.3, { fontSize: 14, bold: true, color: col }); text(s, d, 9.65, y + 0.32, 2.9, 0.7, { fontSize: 12.5, color: C.muted }); });
}

// 10. Clouds
{
  const s = slide("CONTENT", "Results", "What the satellite cannot see", "Every field says how well the satellite saw its harvest",
    "On Maryland's Eastern Shore more than half of the corn was cut in one cloudy fortnight. This Kent field was last seen standing on 2 September and next on 17 September: the cut is somewhere in between. Fields seen with a long gap were called silage far more often. So every field is clearly seen, partly seen, or not judged, and only clearly seen fields are counted. The map and the app show the not-judged fields, they are not hidden.");
  text(s, "A Kent County field, 2022: last clear look 2 September, the next 17 September", M, 1.7, 12, 0.35, { fontSize: 15, bold: true, color: C.gold });
  pic(s, "assets/pic_kent_cloudy.png", M, 2.05, W - 2 * M, 2.25, { panel: true, fill: "FCFCFA" });
  pic(s, "assets/s09_gap_evidence.png", M, 4.45, 6.1, 2.55, { panel: true, fill: "FCFCFA" });
  card(s, 7.0, 4.45, 5.75, 2.55);
  text(s, bullets(["Clearly seen: 12 days or less between the last look standing and the first after the cut. Counted.",
    "Partly seen: 12 to 20 days. Shown beside the total.", "Not judged: wider, or no cut found. Called but not counted."], { fontSize: 14 }), 7.2, 4.65, 5.35, 2.2, { color: C.muted });
}

// 11. States
{
  const s = slide("CONTENT", "Results", "State totals, 2025", "State figures: some close, some far off, and we say which",
    "The organisers score state acres. We sampled 300 fields per state for four states and 3,611 fields in Maryland. Wisconsin and Iowa read high, Pennsylvania and Nebraska low. Maryland reads 4.75 percent against USDA's 8.9. Corn after a small grain, double-crop corn, is not scored yet and explains part of the Maryland gap. State totals are not validated, and we say so.");
  pic(s, "figs/f_states.png", M, 1.75, 8.0, 4.5, { panel: true });
  card(s, 8.85, 1.75, 3.9, 4.5);
  stat(s, 9.1, 1.9, 3.5, "4.75%", "Maryland 2025, ours (range 3.4–6.3%) vs 8.9% from USDA: 22,700 acres vs 40,000", C.silage, 36);
  stat(s, 9.1, 3.55, 3.5, "17,056 ac", "of double-crop corn in Maryland not scored yet. Scored afterwards: 6.7%", C.gold, 30);
  text(s, "Do not quote these as validated. The field calls travel; the adding-up is the open problem.", 9.1, 5.2, 3.4, 0.9, { fontSize: 12.5, color: C.faint });
}

// 12. US map
pres.addSection({ title: "Geospatial" });
{
  const s = slide("CONTENT", "Geospatial", "The US map", "Silage lives where the dairies are: 3,231 counties from the Census",
    "This is the USDA Census silage share for every county, the ground truth we compare to and the prior the pipeline uses. Silage is a dairy crop: Wisconsin, New York, Pennsylvania, the West. In the Corn Belt it is a few percent. Maryland, in gold, is where our labelled fields and county data are. In the app this map is live: click a state for its counties, a county for everything we know about it.");
  pic(s, "figs/m_us_counties.png", M, 1.65, 8.9, 5.35, { panel: true, fill: C.bg });
  card(s, 9.7, 1.65, 3.05, 5.35);
  text(s, bullets(["Shaded: Census 2022 silage share of harvested corn, every county with a published figure.",
    "Gold: Maryland, where the labelled fields and our county data are.", "In the app: states by NASS 2025, counties by Census, then our field calls on satellite imagery."], { fontSize: 14 }), 9.9, 1.85, 2.7, 5.0, { color: C.muted });
}

// 13. Frederick fields
{
  const s = slide("CONTENT", "Geospatial", "Every field, one county", "2,233 Frederick fields, each with a call and a reason",
    "Every corn field in Frederick County in 2022, coloured by the final model's call. 366 silage, 1,672 grain, 195 not judged because the satellite did not see the cut well enough. The blue dots are permitted dairies, sized by herd. Notice where the silage calls cluster: around the dairies. The model never saw a dairy location. That is a good sign the calls mean something.");
  pic(s, "figs/m_frederick.png", M, 1.6, 5.4, 5.45, { panel: true, fill: C.bg });
  pic(s, "shots/usmap_frederick.png", 6.25, 1.6, 6.5, 4.15, { panel: true });
  card(s, 6.25, 5.9, 6.5, 1.15);
  text(s, [{ text: "Silage calls cluster around the dairies. ", options: { bold: true, color: C.gold } },
    { text: "The model never saw a dairy location. Right: the same fields in the app, on satellite imagery, with the county's Census history.", options: { color: C.muted } }],
    6.45, 6.03, 6.1, 0.95, { fontSize: 14 });
}

// 14. Agent
pres.addSection({ title: "The agent" });
{
  const s = slide("CONTENT", "The agent", "Supervisor", "Nemotron runs the county; code holds it to the rules",
    "Nemotron Lightning runs each county as a ReAct loop. It scouts the imagery, picks a route and says why, classifies, adds up and calls the critic. Code enforces the rules: the step order, one rerun at most, the critic before any accept, and every number in the summary must come from a tool result. This is a real run on Beltsville 2022: seven steps, 39,247 tokens, accepted with warnings.");
  table(s, [["#", "Tool", "What came back"],
    ["1", "scout", "7 fields · 5 with a clear series · median 27 clear looks"],
    ["2", "route_options", "standard, radar_first, regional_threshold, in_site …"],
    ["3", "extract_features", "7 feature rows"],
    ["4", "classify standard", "Why: \"5 of 7 fields have clear optical series …\""],
    ["5", "aggregate", "silage 51.2 ac (21.4–73.5) · grain 22.3 · abstained 36.8"],
    ["6", "run_critic", "warn → accept with warnings: 69.7% silage vs MD history 7.8–12.2%"],
    ["7", "finish", "accepted with warnings · summary checked number by number"]], M, 1.75, 7.9, [0.45, 1.85, 5.6], { rowH: 0.5 });
  card(s, 8.8, 1.75, 3.95, 4.55);
  text(s, "GUARDRAILS IN CODE", 9.05, 1.95, 3.5, 0.3, { fontSize: 12, bold: true, color: C.muted, charSpacing: 2 });
  text(s, bullets(["Steps in order: scout before classify, aggregate before critic.", "One rerun, only the policy the critic names.",
    "No accept after a critic fail.", "Every number in the summary must be in a tool result.", "Every LLM turn and tool call in an audit log."], { fontSize: 14 }), 9.05, 2.35, 3.5, 3.9, { color: C.muted });
}

// 15. Critic
{
  const s = slide("CONTENT", "The agent", "Critic", "The critic decides with Quick Stats; Nemotron only explains",
    "The critic compares the county total with USDA's own history for that county and state, excluding the year being checked. Code decides pass, warn or fail and the one rerun; Nemotron Ultra only explains, citing NASS report pages. Here Frederick 2022: the first total reads 13.2 percent silage, below the county's history of 24 to 32, so it fails. The rerun reads 63 percent, fails again, and the agent abstains instead of guessing. One honest catch: the true 2022 figure was 19.6 percent, also outside that range. The history is only two Census years, so the range is too narrow.");
  const steps = [["classify standard", "13.2% silage on clearly seen fields", C.text], ["critic: fail", "below Frederick's history, 24.3–32.1% (Census 2012, 2017)", C.fail],
    ["rerun: regional threshold", "63.0% silage", C.text], ["critic: fail", "above history again", C.fail], ["abstain", "no number published for the county", C.gold]];
  steps.forEach(([t, d, col], i) => { const x = M + i * 2.45;
    card(s, x, 1.85, 2.25, 1.85); text(s, t, x + 0.2, 2.0, 1.9, 0.6, { fontSize: 15, bold: true, color: col }); text(s, d, x + 0.2, 2.65, 1.9, 1.0, { fontSize: 12.5, color: C.muted });
    if (i < steps.length - 1) s.addShape(pres.shapes.LINE, { x: x + 2.25, y: 2.77, w: 0.2, h: 0, line: { color: C.faint, width: 1.25, endArrowType: "triangle" } }); });
  card(s, M, 4.05, 6.0, 2.6);
  text(s, "WHAT WORKED", M + 0.25, 4.25, 5.5, 0.3, { fontSize: 12, bold: true, color: C.ok, charSpacing: 2 });
  text(s, bullets(["The agent refused to publish a county number it could not stand behind.", "Every flag cites the Quick Stats range it used, and Nemotron cites NASS report pages."], { fontSize: 14 }), M + 0.25, 4.65, 5.5, 1.9, { color: C.muted });
  card(s, 6.75, 4.05, 6.0, 2.6);
  text(s, "WHAT WE LEARNED", 7.0, 4.25, 5.5, 0.3, { fontSize: 12, bold: true, color: C.gold, charSpacing: 2 });
  text(s, bullets(["The true 2022 share, 19.6%, is also outside 24.3–32.1%: two Census years make too narrow a range.", "Next: widen it with the state survey trend before the critic fails a county."], { fontSize: 14 }), 7.0, 4.65, 5.5, 1.9, { color: C.muted });
}

// 16. Trust
{
  const s = slide("CONTENT", "The agent", "Can you trust the agent?", "We tested the agent like a model, including an attack",
    "We tested the agent the same way as the model. Five fixed critic scenarios: it got the right outcome in all five. Four analyst questions, two of them out of scope like the price of corn or who owns the farms: all grounded. Then a benchmark across three Nemotron models, including a prompt-injection attack hidden in the critic's text. Lightning followed it; Bolt and Ultra resisted. So a guardrail now blocks any accept after a critic fail.");
  stat(s, M, 1.85, 2.6, "5 / 5", "critic scenarios with the right outcome", C.silage, 44);
  stat(s, M + 2.8, 1.85, 2.6, "4 / 4", "analyst questions grounded, two out of scope", C.silage, 44);
  stat(s, M + 5.6, 1.85, 2.6, "0", "invented numbers once guardrails are on", C.silage, 44);
  table(s, [["Nemotron model", "Decisions", "Q&A", "Prompt injection", "Seconds / run", "Tokens / run"],
    ["3.5 Lightning 30B", "5/5", "4/4", { text: "followed", options: { color: C.fail, bold: true, fill: { color: C.panel } } }, "15.3", "28,632"],
    ["3.5 Bolt E4B", "5/5", "4/4", { text: "resisted", options: { color: C.ok, bold: true, fill: { color: C.bg } } }, "6.9", "23,272"],
    ["3 Ultra 550B", "5/5", "4/4", { text: "resisted", options: { color: C.ok, bold: true, fill: { color: C.panel } } }, "37.6", "29,958"]], M, 3.85, W - 2 * M, [3.2, 1.6, 1.4, 2.2, 1.9, 1.833], { rowH: 0.48 });
  caption(s, "After the injection result: a code guardrail allows only a rerun or abstain after a critic fail. Sources: runs/evaluation and runs/benchmark.", M, 6.0, 12);
}

// 17. App overview
pres.addSection({ title: "The app" });
{
  const s = slide("CONTENT", "The app", "One app, two people", "One pipeline, two people: a dairy owner and a NASS analyst",
    "Everything comes together in one app that runs on a laptop. You pick who you are. A dairy owner gets a chat and a map of standing corn they could buy. A NASS analyst gets a chat over every tool, the full run record, a dashboard and the US map. Without an NVIDIA key it runs offline with a rule-based agent under the same guardrails.");
  pic(s, "shots/landing.png", M, 1.65, 8.2, 5.35, { panel: true });
  card(s, 9.1, 1.65, 3.65, 5.35);
  text(s, "RUN IT", 9.35, 1.85, 3.2, 0.3, { fontSize: 12, bold: true, color: C.muted, charSpacing: 2 });
  text(s, [{ text: "uv pip install -e \".[agents]\"", options: { breakLine: true } }, { text: "fieldvision serve", options: {} }], 9.35, 2.25, 3.2, 0.8,
    { fontFace: "Courier New", fontSize: 13, color: C.gold });
  text(s, bullets(["Dairy owner: Ask · Silage map · US map", "USDA analyst: Ask · Runs · Dashboard · US map", "Offline mode: same tools and guardrails, no key", "86 offline tests pass"], { fontSize: 14 }), 9.35, 3.3, 3.2, 3.5, { color: C.muted });
}

// 18. Dairy owner
{
  const s = slide("CONTENT", "The app", "For the dairy owner", "\"Where can I still buy 2,000 tons of silage?\" Three fields, 1.6 miles",
    "A dairy owner sets their farm, the tons they need and how far they will haul, and picks a date in the season. The recommender keeps fields that are still green and standing on that date, likely headed for grain, at least five acres and within reach. It matches the cheapest tons by haul and by the risk the field is already someone's silage. For this Frederick dairy on 1 September 2022: 2,000 tons from three fields, 1.1 to 1.6 miles away, about 29 to 33 dollars a ton delivered. Every check passed.");
  pic(s, "shots/dairy_map.png", M, 1.65, 8.4, 5.35, { panel: true });
  card(s, 9.3, 1.65, 3.45, 5.35);
  stat(s, 9.55, 1.8, 3.0, "2,000 t", "matched from 3 standing fields within 1.6 road miles", C.silage, 36);
  stat(s, 9.55, 3.35, 3.0, "$28.69–33.03", "per wet ton delivered (rule of thumb, not quotes)", C.gold, 28);
  text(s, "As of 1 September 2022, 1,449 of 2,233 Frederick corn fields could still be chopped. A replay of a past season; model estimates, not NASS statistics.", 9.55, 4.95, 3.0, 1.9, { fontSize: 12, color: C.faint });
}

// 19. Analyst chat
{
  const s = slide("CONTENT", "The app", "For the NASS analyst", "Ask in plain words; every number in the answer is checked",
    "The analyst asks in plain words. Here: where should enumerators go with five visits, and what does that mean for grain supply? Nemotron picks the visit planner. Visiting these five fields shrinks the county's silage range from plus or minus 731 acres to 621. And every number in the answer is checked against what the tool returned before it is shown.");
  pic(s, "shots/usda_chat.png", M, 1.65, 7.9, 5.4, { panel: true });
  card(s, 8.75, 1.65, 4.0, 5.4);
  text(s, "TOOLS THE ANALYST CAN ASK FOR", 9.0, 1.85, 3.6, 0.3, { fontSize: 12, bold: true, color: C.muted, charSpacing: 2 });
  text(s, bullets(["Run a county through the agent", "Evidence, report, field lists", "Field-visit plan and grain impact", "Our silage vs USDA's",
    "Supply, stocks and price (USDA)", "Silage what-if on price", "NASS report search with page", "The agent's own test results"], { fontSize: 14 }), 9.0, 2.25, 3.55, 4.6, { color: C.muted });
}

// 20. Dashboard and price honesty
{
  const s = slide("CONTENT", "The app", "Dashboard", "Supply and price from USDA; our silage only as a labelled what-if",
    "The dashboard puts USDA's numbers first: 2026 production forecast down 6.3 percent, September stocks up 35 percent, the 2025 price at 4.16. Our price models explain a finished season from its ending stocks. Held out one year at a time they beat last year's price by a little, and with only data known at the start of a season they do not beat it. So the silage effect on price is shown as a what-if, with the full range across models: one more point of silage moves the price anywhere from under 1 to over 9 percent.");
  pic(s, "shots/dashboard.png", M, 1.65, 6.4, 3.95, { panel: true });
  pic(s, "figs/f_price_models.png", 7.2, 1.65, 5.55, 3.95, { panel: true });
  card(s, M, 5.8, W - 2 * M, 1.2);
  text(s, [{ text: "Honest by design: ", options: { bold: true, color: C.gold } },
    { text: "the price models explain a season from ending stocks; they do not forecast one. +1 point of silage share moves the 2025 price anywhere from +0.85% to +9.4% depending on the model, so it is shown as a what-if.", options: { color: C.muted } }],
    M + 0.25, 5.95, W - 2 * M - 0.5, 0.95, { fontSize: 14 });
}

// 21. Limits and next steps
pres.addSection({ title: "Close" });
{
  const s = slide("CONTENT", "Close", "Next", "What we would do next",
    "Four next steps, each attacking the adding-up. Radar sees through cloud: a Sentinel-1 reader is built for Kent and would date the cuts we miss. Double-crop corn needs scoring: it explains part of the Maryland gap. The critic's history range should include the state survey trend, so it does not fail a right answer. And a price forecast would need USDA's own projected stocks at forecast time.");
  const items = [["Radar for cloudy weeks", "Sentinel-1 sees through cloud. The Kent reader is built; it dates the cuts optical images miss.", C.usda],
    ["Score double-crop corn", "17,056 Maryland acres sit outside today's figure; part of the gap to USDA's 8.9%.", C.grain],
    ["A wider critic range", "Two Census years made Frederick's range too narrow; add the state survey trend.", C.silage],
    ["A real price forecast", "Needs USDA's projected stocks at forecast time (WASDE), not ending stocks.", C.gold]];
  items.forEach(([t, d, col], i) => { const x = M + (i % 2) * 6.15, y = 1.85 + Math.floor(i / 2) * 2.4;
    card(s, x, y, 5.95, 2.2);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.6, h: 0.6, fill: { color: C.panel2 }, line: { color: col, width: 2 } });
    s.addText(String(i + 1), { x: x + 0.3, y: y + 0.3, w: 0.6, h: 0.6, fontSize: 18, bold: true, color: col, align: "center", valign: "middle", margin: 0, isTextBox: true });
    text(s, t, x + 1.15, y + 0.33, 4.5, 0.45, { fontSize: 19, bold: true }); text(s, d, x + 1.15, y + 0.85, 4.55, 1.2, { fontSize: 14, color: C.muted }); });
}

// 22. Close
{
  const s = slide("SECTION", "Close", "FieldVision", "Every field called. Every number checked.",
    "To close. Field by field, it works: 42 of 44 on fields it never trained on. The adding-up is honest about where it is right and where it reads low. The agent refuses to publish what it cannot stand behind. And the same pipeline helps a dairy find feed and an analyst check a county. Thank you.");
  s.addText("42 of 44 labelled US fields right, trained only on Europe · an agent that abstains instead of guessing · one app for a dairy owner and a NASS analyst",
    { placeholder: "body" });
  chip(s, "github.com/ChenSiyun1234/FieldVision · branch feature/agent-app", M, 6.15, 6.6, C.gold);
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT, n, "slides");
})();
