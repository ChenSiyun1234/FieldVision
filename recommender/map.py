"""Build a one-file Leaflet map from a match result.

Usage:  python3 recommender/map.py recommender/out/match_24021_2022_2022-09-01.json
Open:   <jupyter base>/files/FieldVision/recommender/out/map_24021_2022_2022-09-01.html
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import silage_match as sm  # noqa: E402

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Silage Match __TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 body{margin:0;font:14px/1.4 system-ui,sans-serif;display:flex;height:100vh}
 #map{flex:1} #side{width:360px;overflow:auto;padding:14px 16px;border-left:1px solid #ddd;background:#fafafa}
 h2{font-size:17px;margin:0 0 6px} h3{font-size:14px;margin:14px 0 4px} .muted{color:#666;font-size:12px}
 table{border-collapse:collapse;width:100%;font-size:12px} td,th{padding:3px 4px;border-bottom:1px solid #eee;text-align:right}
 td:first-child,th:first-child{text-align:left} .pass{color:#1a7f37} .warn{color:#b35900} .info{color:#555}
 .legend{background:#fff;padding:6px 8px;border-radius:4px;font-size:12px;box-shadow:0 1px 4px #0003}
</style></head><body><div id="map"></div><div id="side"></div>
<script>
const R=__RESULT__, G=__GEOJSON__;
const map=L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:18,attribution:'&copy; OpenStreetMap'}).addTo(map);
const fmt=n=>Number(n).toLocaleString();
const pal=p=>p<0.1?'#1a7f37':p<0.3?'#9a6700':'#cf222e';
const layer=L.geoJSON(G,{
 style:f=>f.properties.kind==='haul'?{color:'#555',weight:1,opacity:0.5}:{},
 pointToLayer:(f,ll)=>{const p=f.properties;
  if(p.kind==='dairy'){const fill=p.request_tons?p.matched_tons/p.request_tons:1;
   return L.circleMarker(ll,{radius:6+Math.sqrt(p.head)/6,color:'#0b3d91',weight:2,fillColor:fill>=0.8?'#4c8bf5':'#f5a623',fillOpacity:0.85})}
  return L.circleMarker(ll,{radius:3+Math.sqrt(p.tons)/12,color:pal(p.p_silage_adj),weight:1,fillOpacity:0.7})},
 onEachFeature:(f,l)=>{const p=f.properties;
  if(p.kind==='dairy')l.bindPopup(`<b>${p.dairy_id}</b> (${p.county})<br>${fmt(p.head)} head, need ~${fmt(p.annual_need_tons)} t/yr<br>asked ${fmt(p.request_tons)} t, matched ${fmt(p.matched_tons)} t from ${p.n_fields} fields<br>delivered ~$${p.delivered_usd_per_ton[0]} to $${p.delivered_usd_per_ton[1]} per ton`);
  else if(p.kind==='source_field')l.bindPopup(`<b>${p.field_id}</b><br>${p.acres} ac, ${fmt(p.tons)} t to ${p.dairy_id}<br>${p.road_miles} road mi, haul $${p.haul_usd_per_ton[0]} to $${p.haul_usd_per_ton[1]} /t<br>P(already silage) ${p.p_silage_adj}<br>NDVI ${p.ndvi_as_of} on ${p.ndvi_date}`);}
}).addTo(map);
map.fitBounds(layer.getBounds().pad(0.1));
const lg=L.control({position:'bottomleft'});lg.onAdd=()=>{const d=L.DomUtil.create('div','legend');
 d.innerHTML='<b>Dairies</b>: blue = request met, orange = short<br><b>Fields</b> by P(already silage): <span style="color:#1a7f37">&lt;0.1</span> <span style="color:#9a6700">0.1 to 0.3</span> <span style="color:#cf222e">&ge;0.3</span>';return d};lg.addTo(map);
const fu=R.funnel, req=R.dairies.reduce((a,d)=>a+d.request_tons,0), met=R.dairies.reduce((a,d)=>a+d.matched_tons,0);
let h=`<h2>Silage Match, FIPS ${R.fips}, as of ${R.as_of}</h2>
<div class="muted">model ${R.model} &middot; policy ${R.policy} &middot; standing price $${R.standing_price_usd_per_ton[0]} to $${R.standing_price_usd_per_ton[1]} per wet ton</div>
<h3>Requests</h3><div>${R.dairies.length} dairies asked for <b>${fmt(req)} t</b>; <b>${fmt(met)} t</b> matched within ${R.params.max_road_miles} road miles.</div>
<h3>County share</h3><div>Silage share ${R.county_silage_share_pct.raw}% raw &rarr; <b>${R.county_silage_share_pct.after_prior}%</b> after prior (${R.prior.source}).</div>
<h3>Which fields could be chopped</h3><table>`;
for(const k of ['fields','abstained','harvested_by_date','silage_bound','too_small','browned_down_or_no_view','candidate'])h+=`<tr><td>${k.replaceAll('_',' ')}</td><td>${fmt(fu[k])}</td></tr>`;
h+=`</table><h3>Checks</h3>`;
for(const c of R.checks)h+=`<div class="${c.status}">[${c.status}] <b>${c.check}</b>${c.observed_pct!=null?' ('+c.observed_pct+'%)':''}: ${c.why}</div>`;
h+=`<h3>Dairies</h3><table><tr><th>dairy</th><th>asked</th><th>matched</th><th>fields</th><th>$/t</th></tr>`;
for(const d of [...R.dairies].sort((a,b)=>b.request_tons-a.request_tons))h+=`<tr><td>${d.dairy_id}</td><td>${fmt(d.request_tons)}</td><td>${fmt(d.matched_tons)}</td><td>${d.n_fields}</td><td>${d.delivered_usd_per_ton[0]} to ${d.delivered_usd_per_ton[1]}</td></tr>`;
h+=`</table><h3>Sources</h3>`+Object.values(R.sources).map(s=>`<div class="muted">${s}</div>`).join('');
document.getElementById('side').innerHTML=h;
</script></body></html>"""


def build(result_path: Path) -> Path:
    r = json.loads(result_path.read_text())
    slim = {k: v for k, v in r.items() if k != "assignments"}
    html = (PAGE.replace("__TITLE__", f"{r['fips']} {r['as_of']}")
            .replace("__RESULT__", json.dumps(slim, default=str))
            .replace("__GEOJSON__", json.dumps(sm.to_geojson(r), default=str)))
    out = result_path.with_name(result_path.name.replace("match_", "map_")).with_suffix(".html")
    out.write_text(html)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:] or sorted(sm.OUT_DIR.glob("match_*.json")):
        print(build(Path(p)))
