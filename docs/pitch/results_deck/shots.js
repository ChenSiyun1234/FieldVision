// App screenshots for the deck: drives the installed Chrome, waits for each view to finish loading.
const puppeteer = require("puppeteer-core");
const path = require("path");
const OUT = path.join(__dirname, "shots");
const BASE = "http://127.0.0.1:8765";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

async function shot(browser, name, url, { width = 1440, height = 900, waitFor, wait = 1500, timeout = 120000, before } = {}) {
  const page = await browser.newPage();
  await page.setViewport({ width, height, deviceScaleFactor: 2 });
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
  await page.goto(BASE + url, { waitUntil: "networkidle2", timeout });
  if (waitFor) await page.waitForFunction(waitFor, { timeout, polling: 500 });
  if (before) await page.evaluate(before);
  await new Promise((r) => setTimeout(r, wait));
  await page.screenshot({ path: path.join(OUT, name + ".png") });
  console.log("saved", name);
  await page.close();
}

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new", args: ["--hide-scrollbars"] });
  const only = process.argv.slice(2);
  const jobs = {
    landing: () => shot(browser, "landing", "/?role="),
    usmap_us: () => shot(browser, "usmap_us", "/?role=usda&tab=usmap&fips=&year=", {
      waitFor: "document.querySelector('.usmap-panel h3') && document.querySelectorAll('.usmap svg path').length > 40", wait: 3000 }),
    usmap_md: () => shot(browser, "usmap_md", "/?role=usda&tab=usmap&fips=24033&year=2022", {
      waitFor: "document.querySelector('.usmap-panel h3')?.textContent.includes('field calls')", wait: 1500,
      before: "document.querySelector('.usmap-panel button.ghost')?.click()" }),
    usmap_frederick: () => shot(browser, "usmap_frederick", "/?role=usda&tab=usmap&fips=24021&year=2022", {
      waitFor: "document.querySelector('.usmap-panel h3')?.textContent.includes('field calls')", wait: 5000 }),
    dashboard: () => shot(browser, "dashboard", "/?role=usda&tab=dashboard&fips=24021&year=2022", { height: 1000,
      waitFor: "document.querySelectorAll('.tile').length >= 8", wait: 1500 }),
    dashboard_whatif: () => shot(browser, "dashboard_whatif", "/?role=usda&tab=dashboard&fips=24021&year=2022", { height: 1000,
      waitFor: "document.querySelectorAll('.tile').length >= 8", wait: 1500,
      before: "(() => { const h=[...document.querySelectorAll('.page h2')].find(x=>x.textContent.startsWith('Our silage')); h.scrollIntoView(); })()" }),
    dairy_map: () => shot(browser, "dairy_map", "/?role=dairy&tab=map&fips=24021&year=2022", { height: 1000,
      waitFor: "document.querySelectorAll('.leaflet-interactive').length > 3", wait: 4000 }),
    dairy_chat: () => shot(browser, "dairy_chat", "/?role=dairy&tab=chat&fips=24021&year=2022&ask=" + encodeURIComponent("Where can I get 2,000 tons of silage within 15 miles, and when is silage usually cut here?"), {
      height: 1100, timeout: 180000, waitFor: "document.querySelector('.msg .meta')", wait: 4000,
      before: "document.querySelector('.msg.user').scrollIntoView({block: 'start'})" }),
    usda_chat: () => shot(browser, "usda_chat", "/?role=usda&tab=chat&fips=24021&year=2022&ask=" + encodeURIComponent("Where should enumerators go with 5 visits, and what does that mean for grain supply?"), {
      height: 1100, timeout: 180000, waitFor: "document.querySelector('.msg .meta')", wait: 2000,
      before: "document.querySelector('.msg.user').scrollIntoView({block: 'start'})" }),
    run: () => shot(browser, "run", "/?role=usda&tab=runs&fips=24033&year=2022", { height: 1100,
      before: "", waitFor: "document.querySelector('.page')", wait: 1500 }),
  };
  for (const [name, job] of Object.entries(jobs)) {
    if (only.length && !only.includes(name)) continue;
    try { await job(); } catch (e) { console.log("FAILED", name, e.message.split("\n")[0]); }
  }
  await browser.close();
})();
