const topojson = require("/Users/krishi1211/hack-the-harvest/FieldVision/app/vendor/topojson/topojson-client.min.js");
const fs = require("fs");
const t = JSON.parse(fs.readFileSync("/Users/krishi1211/hack-the-harvest/FieldVision/app/vendor/us-atlas/counties-10m.json"));
fs.writeFileSync("counties.geojson", JSON.stringify(topojson.feature(t, t.objects.counties)));
fs.writeFileSync("states.geojson", JSON.stringify(topojson.feature(t, t.objects.states)));
console.log("ok");
