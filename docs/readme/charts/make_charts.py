"""Draw the README charts in this folder (animated SVG, GitHub-safe).

    python docs/readme/charts/make_charts.py

Writes highlights.svg, pipeline.svg, us_tests.svg, gap_closing.svg, field_check.svg and state_2025.svg
next to this script. To change a number or a label, edit it below and rerun. Every number is copied from
docs/RESULTS.md, docs/hima/cloud_gaps.md, src/fieldvision_agents/watch/README.md and the deck.
The US map uses the vendored us-atlas topology in app/vendor/us-atlas/.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

FONT = "text { font-family: 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif; }"
ACCENT, INK, MUTED, LINE = "#1d6b47", "#13261c", "#4f6358", "#cfdcd4"
BADGE, MAPGOLD, MAPGREEN, BEFORE = "#f2c94c", "#c98f0c", "#3d8f62", "#7f8e86"
GOLD = ACCENT
CARD_GRAD = """<linearGradient id="card" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#f3f8f5"/></linearGradient>"""
BG_GRAD = """<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#ffffff"/></linearGradient>"""
FADE = """.in { animation: rise 1s ease-out both; }
      @keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }"""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def svg(w, h, label, defs, body, style=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{esc(label)}">\n  <defs>\n    {defs}\n    <style>\n      {FONT}\n      {style}\n'
            f'    </style>\n  </defs>\n{body}\n</svg>\n')


# ---------------------------------------------------------------- highlights
def highlights():
    tiles = [
        ("4 of 5", "states within 5 pts of NASS 2025", "graded test, each run once"),
        ("95.5%", "field accuracy, 42 of 44", "USDA-labelled Beltsville fields"),
        ("13.3 → 1.4", "pt gap: Bayes + USDA history", "Texas, California, Colorado"),
        ("13.1 → 1.8", "pt miss: cloud-gap Bayes", "Wyoming County, NY"),
        ("0", "unsupported LLM numbers", "5/5 routing, 4/4 Q&A"),
    ]
    w, gap = 228, 15
    parts = []
    for i, (big, a, b) in enumerate(tiles):
        x = i * (w + gap)
        parts.append(
            f'  <g class="in" style="animation-delay:{i * 0.15:.2f}s">'
            f'<rect x="{x + 1}" y="1" width="{w - 2}" height="128" rx="14" fill="url(#card)" stroke="{LINE}"/>'
            f'<text x="{x + 20}" y="56" font-size="34" font-weight="800" fill="{GOLD}">{esc(big)}</text>'
            f'<text x="{x + 20}" y="85" font-size="14" fill="{INK}">{esc(a)}</text>'
            f'<text x="{x + 20}" y="107" font-size="13" fill="{MUTED}">{esc(b)}</text></g>')
    label = "; ".join(f"{t[0]} {t[1]}" for t in tiles)
    return svg(1200, 130, label, CARD_GRAD, "\n".join(parts), FADE)


# ---------------------------------------------------------------- pipeline
MONO = "font-family: Consolas, Menlo, 'DejaVu Sans Mono', monospace;"


def stage(x, y, w, h, n, title, bullets, result, delay, formula=None):
    s = [f'  <g class="in" style="animation-delay:{delay:.2f}s">',
         f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="url(#card)" stroke="{LINE}" stroke-width="1.5"/>',
         f'<circle cx="{x + 24}" cy="{y + 25}" r="14" fill="{BADGE}"/>',
         f'<text x="{x + 24}" y="{y + 30}" font-size="14" font-weight="800" fill="{INK}" text-anchor="middle">{n}</text>',
         f'<text x="{x + 47}" y="{y + 31}" font-size="16.5" font-weight="700" fill="{INK}">{esc(title)}</text>']
    ty = y + 62
    if formula:
        s.append(f'<rect x="{x + 14}" y="{y + 46}" width="{w - 28}" height="30" rx="7" fill="#fdf6e0" stroke="#e8cf86"/>'
                 f'<text x="{x + w / 2:.0f}" y="{y + 66}" font-size="14" font-weight="700" fill="{INK}" '
                 f'text-anchor="middle" style="{MONO}">{esc(formula)}</text>')
        ty = y + 98
    for b in bullets:
        s.append(f'<text x="{x + 16}" y="{ty}" font-size="13.5" fill="{ACCENT}" font-weight="800">•</text>'
                 f'<text x="{x + 30}" y="{ty}" font-size="13.5" fill="{INK}">{esc(b)}</text>')
        ty += 21
    s.append(f'<line x1="{x + 14}" y1="{y + h - 36}" x2="{x + w - 14}" y2="{y + h - 36}" stroke="{LINE}"/>'
             f'<text x="{x + 16}" y="{y + h - 13}" font-size="14" font-weight="700" fill="{ACCENT}">{esc(result)}</text></g>')
    return "".join(s)


def arrow(d, label=None, lx=0, ly=0, anchor="middle"):
    s = f'  <path d="{d}" fill="none" stroke="{MUTED}" stroke-width="1.8" marker-end="url(#head)"/>'
    if label:
        s += (f'<text x="{lx}" y="{ly}" font-size="12" fill="{MUTED}" text-anchor="{anchor}" '
              f'font-style="italic">{esc(label)}</text>')
    return s


def pipeline():
    W, H = 1200, 1090
    X = [80, 450, 820]
    w = 330
    yA, hA = 146, 320
    yB, yC = yA + hA + 40, yA + hA + 280
    mA, mB = yA + hA // 2, yA + hA + 40 + 90
    y3, y4 = yA + 65, yA + 140 + 90
    body = [f'  <rect width="{W}" height="{H}" rx="20" fill="url(#bg)" stroke="#dde7e1" stroke-width="1.5"/>']
    # 9: watch loop band on top, it triggers the run
    body.append(
        f'  <g class="in"><rect x="60" y="16" width="1125" height="64" rx="12" fill="#f3f8f5" stroke="{LINE}" stroke-width="1.5"/>'
        f'<circle cx="88" cy="48" r="14" fill="{BADGE}"/><text x="88" y="53" font-size="14" font-weight="800" fill="{INK}" '
        f'text-anchor="middle">9</text><text x="112" y="54" font-size="16.5" font-weight="700" fill="{INK}">Watch loop</text>'
        f'<text x="220" y="54" font-size="13.5" fill="{INK}">checks for new Sentinel-2 scenes every 5 days and reruns the '
        f'pipeline only when the estimate moves 1+ point</text>'
        f'<text x="1165" y="54" font-size="14" font-weight="700" fill="{ACCENT}" text-anchor="end">agent ran 4×, not 21×</text></g>')
    body += [f'  <rect x="60" y="96" width="1125" height="890" rx="16" fill="none" stroke="#c98f0c" stroke-opacity="0.7" '
             f'stroke-dasharray="6 6"/>',
             f'  <text x="80" y="126" font-size="13" font-weight="700" fill="{ACCENT}" letter-spacing="1.2">'
             f'NEMOTRON SUPERVISOR  ·  REACT LOOP OVER 10 TOOLS  ·  CODE COMPUTES AND CHECKS EVERY NUMBER</text>']
    body += [
        stage(X[0], yA, w, hA, 1, "Ingest: data in",
              ["Sentinel-2 L2A: 10 bands, every 5 days", "Landsat 8/9: fills the cloudy weeks",
               "USDA Crop Sequence Boundaries (fields)", "USDA Cropland Data Layer: where corn is",
               "EU declarations: 8,000 labelled fields",
               "USDA Beltsville: 44 labelled US fields"],
              "Hidden fields 217 → 136 with Landsat", 0.1),
        stage(X[1], yA, w, hA, 2, "Cloud check",
              ["Needs 2 clear images that bracket the cut", "Bracket of 12 days or less → stage 3",
               "Cut hidden by cloud → stage 4", "Nothing seen → not judged, area reported"],
              "Queen Anne's MD 22.4% → 4.0% (USDA 1.3%)", 0.3),
        stage(X[2], yA, w, 130, 3, "Classify (seen fields)",
              ["Class-balanced logistic regression", "6 features shaped by US signals"],
              "42 / 44 US fields right (95.5%)", 0.5),
        stage(X[2], yA + 140, w, 180, 4, "Cloud-gap Bayes (hidden fields)",
              ["S, G: county silage, grain cut in the gap", "π: last official silage share"],
              "Wyoming NY miss 13.1 → 1.8 pts", 0.6, formula="p(silage) = πS / (πS + (1−π)G)"),
    ]
    body += [
        stage(X[2], yB, w, 200, 5, "USDA-history prior",
              ["p: the model's silage probability", "π: state's USDA silage share, 2021-24",
               "Σ p′ × acres → county and state + range"],
              "TX, CA, CO gap 13.3 → 1.4 pts", 0.8, formula="p′ = pπ / (pπ + (1−p)(1−π))"),
        stage(X[1], yB, w, 200, 6, "RAG critic",
              ["USDA QuickStats: acres, 2012-2025", "NASS Crop Progress reports, 2021-2023",
               "Checks the county vs its history", "On fail: cites NASS passages, reruns"],
              "Every flag comes with a source", 1.0),
        stage(X[0], yB, w, 200, 7, "Silage Match",
              ["Dairies: MD permits + EPA ECHO", "Min-cost flow, 20 road miles max",
               "Cost: haul distance + silage odds"],
              "2,000 t from 3 fields within 1.6 mi", 1.2),
        stage(X[0], yC, w, 220, 8, "Report + decision checks",
              ["Fixed step order, one rerun", "Abstains instead of guessing",
               "Every LLM number checked in code", "Template fallback if a check fails"],
              "0 unsupported numbers", 1.4),
    ]
    # Nemotron models panel
    nx, nw, nh = X[1], X[2] + w - X[1], 220
    models = [
        ("Nemotron 3.5 Lightning", "30B total · 3B active · text", "Supervisor: picks each tool · 15.3 s/county"),
        ("Nemotron 3 Ultra", "550B total · 55B active · 1M context", "Critic verdict + report · 37.6 s/county"),
        ("Nemotron 3.5 Super VL", "120B total · 12B active · vision + text", "Before/after field images · ~6 s/field"),
        ("Nemotron 3.5 Bolt", "74B total · 9B active · runs locally", "Fast Q&A, benchmarked · 6.9 s/county"),
    ]
    p = [f'  <g class="in" style="animation-delay:1.6s"><rect x="{nx}" y="{yC}" width="{nw}" height="{nh}" rx="12" '
         f'fill="url(#card)" stroke="{LINE}" stroke-width="1.5"/>',
         f'<text x="{nx + 18}" y="{yC + 31}" font-size="16.5" font-weight="700" fill="{INK}">NVIDIA Nemotron models</text>'
         f'<text x="{nx + 236}" y="{yC + 31}" font-size="13" fill="{MUTED}">open mixture-of-experts, hybrid Mamba-Transformer</text>']
    cw = (nw - 46) / 2
    for i, (name, spec, role) in enumerate(models):
        mx = nx + 16 + (i % 2) * (cw + 14)
        my = yC + 46 + (i // 2) * 84
        p.append(f'<rect x="{mx:.0f}" y="{my}" width="{cw:.0f}" height="76" rx="9" fill="#ffffff" stroke="{LINE}"/>'
                 f'<text x="{mx + 14:.0f}" y="{my + 23}" font-size="14.5" font-weight="700" fill="{INK}">{esc(name)}</text>'
                 f'<text x="{mx + 14:.0f}" y="{my + 43}" font-size="12.5" fill="{MUTED}">{esc(spec)}</text>'
                 f'<text x="{mx + 14:.0f}" y="{my + 63}" font-size="13" font-weight="600" fill="{ACCENT}">{esc(role)}</text>')
    p.append("</g>")
    body += p
    # outputs
    body.append(f'  <text x="80" y="1054" font-size="12.5" font-weight="700" fill="{ACCENT}" letter-spacing="1.2">OUTPUTS</text>')
    ox = 170
    for i, o in enumerate(["Analyst dashboard", "County report", "US map, state to field", "Dairy silage map",
                           "Price scenario 2026-28"]):
        tw = len(o) * 7.3 + 30
        body.append(f'  <g class="in" style="animation-delay:{1.8 + i * 0.1:.2f}s"><rect x="{ox:.0f}" y="1032" width="{tw:.0f}" '
                    f'height="36" rx="18" fill="#fdf6e0" stroke="#c98f0c" stroke-opacity="0.8"/>'
                    f'<text x="{ox + tw / 2:.0f}" y="1055" font-size="13.5" fill="{INK}" text-anchor="middle">{esc(o)}</text></g>')
        ox += tw + 12
    c = [x + w / 2 for x in X]
    body += [
        arrow(f"M60,48 L36,48 L36,{mA} L76,{mA}"),
        arrow(f"M{X[0] + w},{mA} L{X[1] - 4},{mA}"),
        arrow(f"M{X[1] + w},{y3} L{X[2] - 4},{y3}", "seen", X[1] + w + 20, y3 - 9),
        arrow(f"M{X[1] + w},{y4} L{X[2] - 4},{y4}", "cloud", X[1] + w + 20, y4 - 9),
        arrow(f"M{X[2] + w},{y3} L1170,{y3} L1170,{mB + 10} L{X[2] + w + 4},{mB + 10}"),
        arrow(f"M{c[2]:.0f},{yA + hA} L{c[2]:.0f},{yB - 4}"),
        arrow(f"M{X[2]},{mB} L{X[1] + w + 4},{mB}"),
        arrow(f"M{X[1] + 250},{yB} C{X[1] + 250},{yB - 26} {X[2] + 60},{yB - 26} {X[2] + 60},{yB - 4}",
              "fail: rerun once", (X[1] + 250 + X[2] + 60) / 2, yB - 26),
        arrow(f"M{X[1]},{mB} L{X[0] + w + 4},{mB}", "pass", X[0] + w + 20, mB - 9),
        arrow(f"M{c[0]:.0f},{yB + 200} L{c[0]:.0f},{yC - 4}"),
        arrow(f"M{c[0]:.0f},{yC + 220} L{c[0]:.0f},1028"),
    ]
    main = (f"M48,48 L36,48 L36,{mA} L{c[0]:.0f},{mA} L{c[1]:.0f},{mA} L{c[2]:.0f},{y3} L1170,{y3} L1170,{mB + 10} "
            f"L{c[2]:.0f},{mB + 10} L{c[1]:.0f},{mB} L{c[0]:.0f},{mB} L{c[0]:.0f},{yC + 110} L{c[0]:.0f},1050")
    body.append(f'  <circle r="6" fill="#e0a800" opacity="0.9"><animateMotion dur="9s" repeatCount="indefinite" '
                f'path="{main}"/><animate attributeName="r" values="5;8;5" dur="1.2s" repeatCount="indefinite"/></circle>')
    defs = (CARD_GRAD + "\n    " + BG_GRAD + f'\n    <marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>')
    label = ("FieldVision pipeline: 9 stages orchestrated by a Nemotron supervisor. Ingest Sentinel-2, Landsat 8/9 and USDA "
             "field data; cloud check; classify with a 6-feature model trained on 8,000 EU fields; cloud-gap Bayes "
             "p(silage) = pi S / (pi S + (1 - pi) G); USDA-history prior p' = p pi / (p pi + (1 - p)(1 - pi)); RAG critic over "
             "QuickStats and NASS reports; Silage Match; report and decision checks; 5-day watch loop. Models: Nemotron 3.5 "
             "Lightning 30B-A3B, Nemotron 3 Ultra 550B-A55B, Nemotron 3.5 Super VL 120B-A12B, Nemotron 3.5 Bolt 74B-A9B.")
    return svg(W, H, label, defs, "\n".join(body), FADE)


# ---------------------------------------------------------------- US map
def decode_topology(t):
    sx, sy = t["transform"]["scale"]
    tx, ty = t["transform"]["translate"]
    arcs = []
    for arc in t["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx
            y += dy
            pts.append((x * sx + tx, y * sy + ty))
        arcs.append(pts)
    return arcs


def ring_points(ring, arcs):
    pts = []
    for i in ring:
        a = arcs[i] if i >= 0 else arcs[~i][::-1]
        pts.extend(a if not pts else a[1:])
    return pts


def albers(lon, lat):
    p1, p2, p0, l0 = map(math.radians, (29.5, 45.5, 37.5, -96.0))
    n = (math.sin(p1) + math.sin(p2)) / 2
    c = math.cos(p1) ** 2 + 2 * n * math.sin(p1)
    r0 = math.sqrt(c - 2 * n * math.sin(p0)) / n
    phi, lam = math.radians(lat), math.radians(lon)
    r = math.sqrt(c - 2 * n * math.sin(phi)) / n
    th = n * (lam - l0)
    return r * math.sin(th), -(r0 - r * math.cos(th))


def us_tests():
    t = json.loads((ROOT / "app/vendor/us-atlas/counties-10m.json").read_text())
    arcs = decode_topology(t)
    skip = {"02", "15", "60", "66", "69", "72", "78"}
    clean = {"26": "MI", "36": "NY", "55": "WI", "42": "PA", "24": "MD"}
    dev = {"19": "IA", "27": "MN", "20": "KS", "46": "SD", "31": "NE", "48": "TX", "06": "CA", "08": "CO"}
    shapes = []
    for g in t["objects"]["states"]["geometries"]:
        if g["id"] in skip:
            continue
        polys = g["arcs"] if g["type"] == "MultiPolygon" else [g["arcs"]]
        rings = [[albers(*p) for p in ring_points(r, arcs)] for poly in polys for r in poly]
        shapes.append((g["id"], rings))
    xs = [x for _, rs in shapes for r in rs for x, _ in r]
    ys = [y for _, rs in shapes for r in rs for _, y in r]
    mx, my, Mx, My = min(xs), min(ys), max(xs), max(ys)
    box_x, box_y, box_w, box_h = 30, 70, 790, 470
    k = min(box_w / (Mx - mx), box_h / (My - my))
    ox = box_x + (box_w - k * (Mx - mx)) / 2
    oy = box_y + (box_h - k * (My - my)) / 2

    def P(x, y):
        return ox + (x - mx) * k, oy + (y - my) * k

    def path(rings):
        out = []
        for r in rings:
            pts, last = [], None
            for x, y in r:
                q = P(x, y)
                if last is None or abs(q[0] - last[0]) + abs(q[1] - last[1]) > 1.2:
                    pts.append(q)
                    last = q
            if len(pts) > 2:
                out.append("M" + "L".join(f"{a:.1f},{b:.1f}" for a, b in pts) + "Z")
        return "".join(out)

    def centroid(rings):
        best = max(rings, key=len)
        pts = [P(*p) for p in best]
        a = cxs = cys = 0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
            f = x0 * y1 - x1 * y0
            a += f
            cxs += (x0 + x1) * f
            cys += (y0 + y1) * f
        return cxs / (3 * a), cys / (3 * a)

    nudge = {"MI": (12, 22), "MD": (14, -6), "CA": (-6, 0), "NY": (6, 4), "PA": (0, 2), "TX": (8, -10)}
    base, hl, labels = [], [], []
    order = sorted(shapes, key=lambda s: centroid(s[1])[0])
    for i, (fid, rings) in enumerate(order):
        d = path(rings)
        if fid in clean or fid in dev:
            col = MAPGOLD if fid in clean else MAPGREEN
            hl.append(f'  <path class="in" style="animation-delay:{0.3 + i * 0.04:.2f}s" d="{d}" fill="{col}" '
                      f'stroke="#ffffff" stroke-width="1"/>')
            code = clean.get(fid) or dev.get(fid)
            if code == "MD":
                continue
            x, y = centroid(rings)
            dx, dy = nudge.get(code, (0, 0))
            labels.append(f'  <text x="{x + dx:.0f}" y="{y + dy + 4:.0f}" font-size="12" font-weight="800" '
                          f'fill="#ffffff" text-anchor="middle">{code}</text>')
        else:
            base.append(f'  <path d="{d}" fill="#e6ece8" stroke="#ffffff" stroke-width="1"/>')
    bx, by = P(*albers(-76.93, 39.03))
    star = (f'  <circle cx="{bx:.1f}" cy="{by:.1f}" r="5" fill="#13261c"/><circle cx="{bx:.1f}" cy="{by:.1f}" r="5" fill="none" '
            f'stroke="#13261c" stroke-width="2"><animate attributeName="r" values="5;16" dur="2s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="0.9;0" dur="2s" repeatCount="indefinite"/></circle>'
            f'<text x="{bx + 8:.0f}" y="{by + 30:.0f}" font-size="11.5" fill="{INK}">Beltsville, MD: 44 labelled US</text>'
            f'<text x="{bx + 8:.0f}" y="{by + 44:.0f}" font-size="11.5" fill="{INK}">fields, develop + validate</text>')
    # Europe card and arc
    ex = 860
    eu = [f'  <g class="in" style="animation-delay:0s"><rect x="{ex}" y="70" width="310" height="196" rx="14" fill="url(#card)" stroke="{LINE}"/>',
          f'<text x="{ex + 20}" y="100" font-size="12" font-weight="700" fill="{GOLD}" letter-spacing="1.2">LEARNED FROM EUROPE</text>',
          f'<text x="{ex + 20}" y="140" font-size="34" font-weight="800" fill="{INK}">8,000</text>',
          f'<text x="{ex + 20}" y="164" font-size="13.5" fill="{INK}">labelled maize fields from</text>',
          f'<text x="{ex + 20}" y="183" font-size="13.5" fill="{INK}">Austria, France, the Netherlands</text>',
          f'<text x="{ex + 20}" y="214" font-size="13.5" font-weight="700" fill="{GOLD}">+ US signals shaped the features</text>',
          f'<text x="{ex + 20}" y="236" font-size="12" fill="{MUTED}">cloud gaps, cover crops, early</text>',
          f'<text x="{ex + 20}" y="252" font-size="12" fill="{MUTED}">Texas cuts, double-crop corn</text></g>',
          f'  <path d="M{ex},170 C{ex - 60},170 {bx + 60:.0f},{by - 80:.0f} {bx + 6:.0f},{by - 10:.0f}" fill="none" stroke="#c98f0c" '
          f'stroke-width="2" stroke-dasharray="7 6" marker-end="url(#goldhead)"><animate attributeName="stroke-dashoffset" '
          f'values="26;0" dur="1.2s" repeatCount="indefinite"/></path>']
    legend = [f'  <g class="in" style="animation-delay:0.6s"><rect x="{ex}" y="290" width="310" height="230" rx="14" fill="url(#card)" stroke="{LINE}"/>',
              f'<text x="{ex + 20}" y="320" font-size="12" font-weight="700" fill="{GOLD}" letter-spacing="1.2">TESTED IN 13 STATES</text>',
              f'<rect x="{ex + 20}" y="338" width="18" height="18" rx="4" fill="{MAPGOLD}"/>',
              f'<text x="{ex + 48}" y="352" font-size="13.5" fill="{INK}">Clean test vs USDA, rule logged</text>',
              f'<text x="{ex + 48}" y="369" font-size="12" fill="{MUTED}">MI, NY, WI, PA, MD</text>',
              f'<rect x="{ex + 20}" y="388" width="18" height="18" rx="4" fill="{MAPGREEN}"/>',
              f'<text x="{ex + 48}" y="402" font-size="13.5" fill="{INK}">300-field state samples</text>',
              f'<text x="{ex + 48}" y="419" font-size="12" fill="{MUTED}">IA, MN, KS, SD, NE, TX, CA, CO</text>',
              f'<text x="{ex + 20}" y="456" font-size="13.5" fill="{INK}">Grain belt to dairy country,</text>',
              f'<text x="{ex + 20}" y="474" font-size="13.5" fill="{INK}">1.9% to 83% silage, one model.</text>',
              f'<text x="{ex + 20}" y="500" font-size="12" fill="{MUTED}">Results by state: Results section</text></g>']
    title = (f'  <text x="30" y="46" font-size="20" font-weight="800" fill="{INK}">Learn from Europe, adapt to the US</text>')
    defs = (CARD_GRAD + "\n    " + BG_GRAD + f'\n    <marker id="goldhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#c98f0c"/></marker>')
    body = [f'  <rect width="1200" height="560" rx="20" fill="url(#bg)" stroke="#dde7e1" stroke-width="1.5"/>', title] + base + hl + labels + eu + legend + [star]
    label = ("Map: learned from 8,000 European fields, adapted with US harvest signals, tested in 13 US states. Clean tests in MI, NY, WI, PA, MD; "
             "state samples in IA, MN, KS, SD, NE, TX, CA, CO; 44 labelled fields at Beltsville, MD.")
    return svg(1200, 560, label, defs, "\n".join(body), FADE)


# ---------------------------------------------------------------- gap closing
def gap_closing():
    rows = [  # place, method, miss before, miss after (points from USDA, absolute)
        ("Texas 2025", "USDA-history prior", 23.8, 0.1),
        ("Queen Anne's MD 2022", "cloud check (seen fields only)", 21.1, 2.7),
        ("Wyoming County NY 2022", "cloud-gap Bayes", 13.1, 1.8),
        ("California 2025", "USDA-history prior", 8.1, 2.9),
        ("Colorado 2025", "USDA-history prior", 7.9, 1.3),
        ("Kent MD 2022", "Landsat as second satellite", 7.4, 1.8),
        ("Frederick MD 2022", "cloud-gap Bayes", 3.0, 1.7),
    ]
    W, top, rh = 1200, 96, 46
    H = top + rh * len(rows) + 64
    x0, x1, vmax = 400, 1150, 25.0

    def X(v):
        return x0 + (x1 - x0) * v / vmax

    b = [f'  <rect width="{W}" height="{H}" rx="20" fill="url(#bg)" stroke="#dde7e1" stroke-width="1.5"/>',
         f'  <text x="30" y="44" font-size="20" font-weight="800" fill="{INK}">The classifier alone was not enough</text>',
         f'  <text x="30" y="68" font-size="13" fill="{MUTED}">Miss against USDA, in points of silage share. Each stage fixes one way the images mislead.</text>',
         f'  <circle cx="870" cy="52" r="6" fill="none" stroke="{BEFORE}" stroke-width="2.5"/>'
         f'<text x="884" y="57" font-size="13" fill="{INK}">model alone</text>',
         f'  <circle cx="990" cy="52" r="7" fill="{GOLD}"/><text x="1004" y="57" font-size="13" fill="{INK}">with the stage</text>']
    for v in range(0, 26, 5):
        b.append(f'  <line x1="{X(v):.0f}" y1="{top - 8}" x2="{X(v):.0f}" y2="{top + rh * len(rows) - 10}" stroke="#13261c" '
                 f'stroke-opacity="0.08"/><text x="{X(v):.0f}" y="{top + rh * len(rows) + 10}" font-size="12" fill="{MUTED}" '
                 f'text-anchor="middle">{v}</text>')
    b.append(f'  <text x="{x1}" y="{top + rh * len(rows) + 30}" font-size="12" fill="{MUTED}" text-anchor="end">points from USDA</text>')
    for i, (place, method, a, c) in enumerate(rows):
        y = top + i * rh + 12
        xa, xc = X(a), X(c)
        d = xa - xc
        b.append(f'  <text x="30" y="{y + 1}" font-size="14" font-weight="700" fill="{INK}">{esc(place)}</text>'
                 f'<text x="30" y="{y + 18}" font-size="12" fill="{MUTED}">{esc(method)}</text>')
        b.append(f'  <line x1="{xc:.1f}" y1="{y + 4}" x2="{xa:.1f}" y2="{y + 4}" stroke="{ACCENT}" stroke-opacity="0.25" stroke-width="3" '
                 f'stroke-linecap="round"/>')
        b.append(f'  <circle cx="{xa:.1f}" cy="{y + 4}" r="6" fill="#ffffff" stroke="{BEFORE}" stroke-width="2.5"/>'
                 f'<text x="{xa + 12:.1f}" y="{y + 9}" font-size="12.5" fill="{MUTED}">{a}</text>')
        b.append(f'  <g style="animation: slide{i} 1.6s {0.3 + i * 0.15:.2f}s ease-out both">'
                 f'<circle cx="{xc:.1f}" cy="{y + 4}" r="8" fill="{ACCENT}" stroke="#ffffff" stroke-width="2"/>'
                 f'<text x="{xc - 14:.1f}" y="{y + 9}" font-size="12.5" font-weight="700" fill="{GOLD}" text-anchor="end">'
                 f'{c}</text></g>')
        b.append(f'  <style>@keyframes slide{i} {{ from {{ transform: translateX({d:.1f}px); }} to {{ transform: none; }} }}</style>')
    b.append(f'  <text x="30" y="{H - 18}" font-size="12" fill="{MUTED}">These places were looked at while each method was built, '
             f'so they show what the stage does, not a clean test. Clean tests are in the Results section.</text>')
    label = "Before and after each stage, miss against USDA in points: " + "; ".join(f"{r[0]} {r[2]} to {r[3]}" for r in rows)
    return svg(W, H, label, BG_GRAD, "\n".join(b))


# ---------------------------------------------------------------- field-level check
def field_check():
    rows = [  # label, detail, correct of 44, style
        ("Always say \"grain\"", "the baseline to beat", 32, "base"),
        ("135-feature boosted trees", "trained on Austria", 30, "base"),
        ("135-feature boosted trees", "trained on all of Europe", 37, "base"),
        ("Harvest-timing rule", "date of the green-to-bare drop", 38, "base"),
        ("6-feature logistic regression", "the model we ship, trained on 8,000 EU fields", 42, "final"),
        ("+ Beltsville labels in training", "cross-validated: train on 2 seasons, test the 3rd", 43, "cv"),
    ]
    W, top, rh = 1200, 104, 50
    H = top + rh * len(rows) + 70
    x0, x1 = 470, 1050

    def X(v):
        return x0 + (x1 - x0) * v / 44

    fills = {"base": "#b9c6be", "final": ACCENT, "cv": "#7fb898"}
    b = [f'  <rect width="{W}" height="{H}" rx="20" fill="url(#bg)" stroke="#dde7e1" stroke-width="1.5"/>',
         f'  <text x="30" y="44" font-size="20" font-weight="800" fill="{INK}">Field-level check: the simple model wins</text>',
         f'  <text x="30" y="68" font-size="13" fill="{MUTED}">44 USDA-labelled Beltsville fields, Maryland 2021-23. '
         f'Big models fit Europe better and the US worse.</text>']
    for v in (0, 11, 22, 33, 44):
        b.append(f'  <line x1="{X(v):.0f}" y1="{top - 14}" x2="{X(v):.0f}" y2="{top + rh * len(rows) - 14}" stroke="{INK}" '
                 f'stroke-opacity="0.08"/><text x="{X(v):.0f}" y="{top + rh * len(rows) + 4}" font-size="12" fill="{MUTED}" '
                 f'text-anchor="middle">{v}</text>')
    b.append(f'  <text x="{x1}" y="{top + rh * len(rows) + 24}" font-size="12" fill="{MUTED}" text-anchor="end">fields right, of 44</text>')
    for i, (lab, det, n, kind) in enumerate(rows):
        y = top + i * rh
        b.append(f'  <text x="30" y="{y + 4}" font-size="14.5" font-weight="700" fill="{INK}">{esc(lab)}</text>'
                 f'<text x="30" y="{y + 22}" font-size="12.5" fill="{MUTED}">{esc(det)}</text>')
        b.append(f'  <rect class="grow" style="animation-delay:{0.2 + i * 0.2:.2f}s" x="{x0}" y="{y - 12}" '
                 f'width="{X(n) - x0:.1f}" height="26" rx="4" fill="{fills[kind]}"/>')
        weight = "800" if kind == "final" else "600"
        col = MUTED if kind == "base" else ACCENT
        b.append(f'  <text x="{X(n) + 10:.1f}" y="{y + 6}" font-size="14" font-weight="{weight}" fill="{col}">'
                 f'{n} / 44 · {100 * n / 44:.1f}%</text>')
    b.append(f'  <text x="30" y="{H - 20}" font-size="12" fill="{MUTED}">Beltsville is the only labelled US set and was looked at '
             f'during development, so this is a field-level check; the graded test is the 2025 state acres.</text>')
    style = (".grow { transform-box: fill-box; transform-origin: left center; animation: grow 1.1s ease-out both; }\n"
             "      @keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }")
    label = "Field-level check on 44 labelled US fields: " + "; ".join(f"{r[0]} ({r[1]}) {r[2]} of 44" for r in rows)
    return svg(W, H, label, BG_GRAD, "\n".join(b), style)


# ---------------------------------------------------------------- graded 2025 states
def state_2025():
    rows = [  # state, ours, lo, hi, nass
        ("Michigan", 14.4, 10.8, 18.0, 14.7),
        ("New York", 48.7, 43.3, 54.2, 53.2),
        ("Maryland", 4.75, 3.4, 6.3, 8.9),
        ("Wisconsin", 16.6, 13.8, 19.5, 21.6),
        ("Pennsylvania", 15.2, 12.9, 17.5, 32.8),
    ]
    W, top, rh = 1200, 120, 54
    H = top + rh * len(rows) + 74
    x0, x1, vmax = 260, 1010, 60.0

    def X(v):
        return x0 + (x1 - x0) * v / vmax

    b = [f'  <rect width="{W}" height="{H}" rx="20" fill="url(#bg)" stroke="#dde7e1" stroke-width="1.5"/>',
         f'  <text x="30" y="44" font-size="20" font-weight="800" fill="{INK}">Graded against USDA NASS 2025</text>',
         f'  <text x="30" y="68" font-size="13" fill="{MUTED}">Silage share of harvested corn, by state. Each state run once on '
         f'150 random blocks, under a rule logged before any 2025 imagery was scored.</text>',
         f'  <circle cx="760" cy="94" r="7" fill="{ACCENT}"/><text x="774" y="99" font-size="13" fill="{INK}">FieldVision, with range</text>',
         f'  <path d="M945,86 L953,94 L945,102 L937,94 Z" fill="{MAPGOLD}"/><text x="960" y="99" font-size="13" fill="{INK}">USDA NASS 2025</text>',
         f'  <text x="1170" y="{top - 22}" font-size="12" font-weight="700" fill="{MUTED}" text-anchor="end">GAP</text>']
    for v in range(0, 61, 10):
        b.append(f'  <line x1="{X(v):.0f}" y1="{top - 14}" x2="{X(v):.0f}" y2="{top + rh * len(rows) - 18}" stroke="{INK}" '
                 f'stroke-opacity="0.08"/><text x="{X(v):.0f}" y="{top + rh * len(rows)}" font-size="12" fill="{MUTED}" '
                 f'text-anchor="middle">{v}%</text>')
    for i, (st, ours, lo, hi, nass) in enumerate(rows):
        y = top + i * rh
        gap = ours - nass
        ok = abs(gap) <= 5.01
        b.append(f'  <text x="30" y="{y + 5}" font-size="15" font-weight="700" fill="{INK}">{st}</text>')
        b.append(f'  <line x1="{X(lo):.1f}" y1="{y}" x2="{X(hi):.1f}" y2="{y}" stroke="{ACCENT}" stroke-opacity="0.35" '
                 f'stroke-width="8" stroke-linecap="round"/>')
        b.append(f'  <g class="in" style="animation-delay:{0.2 + i * 0.15:.2f}s"><circle cx="{X(ours):.1f}" cy="{y}" r="8" fill="{ACCENT}" '
                 f'stroke="#ffffff" stroke-width="2"/></g>')
        b.append(f'  <path d="M{X(nass):.1f},{y - 9} L{X(nass) + 9:.1f},{y} L{X(nass):.1f},{y + 9} L{X(nass) - 9:.1f},{y} Z" '
                 f'fill="{MAPGOLD}" stroke="#ffffff" stroke-width="1.5"/>')
        lx = max(X(hi), X(nass) + 9) + 14
        b.append(f'  <text x="{lx:.1f}" y="{y + 5}" font-size="12.5" fill="{MUTED}">{ours:g}% vs {nass:g}%</text>')
        b.append(f'  <text x="1170" y="{y + 5}" font-size="14" font-weight="800" fill="{ACCENT if ok else "#a4471b"}" '
                 f'text-anchor="end">{gap:+.1f}</text>')
    b.append(f'  <text x="30" y="{H - 20}" font-size="12" fill="{MUTED}">4 of 5 within 5 points. Pennsylvania is the miss: '
             f'northern silage cut after the crop yellows reads as grain. Maryland\'s method was set after its county runs.</text>')
    label = "Graded against USDA NASS 2025: " + "; ".join(f"{r[0]} {r[1]}% vs {r[4]}%" for r in rows)
    return svg(W, H, label, BG_GRAD, "\n".join(b), FADE)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("highlights", highlights), ("pipeline", pipeline), ("us_tests", us_tests), ("gap_closing", gap_closing), ("field_check", field_check), ("state_2025", state_2025)]:
        (OUT / f"{name}.svg").write_text(fn(), encoding="utf-8")
        print("wrote", OUT / f"{name}.svg")
