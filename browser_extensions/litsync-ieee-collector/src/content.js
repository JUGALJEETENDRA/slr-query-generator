(() => {
  "use strict";
  if (globalThis.__lsIEEERunner) return;
  globalThis.__lsIEEERunner = true;
  const C = LitSyncIEEE, D = C.dom, documentToken = crypto.randomUUID();
  let busy = false, dead = false;
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  async function rpc(type, extra = {}) {
    const response = await chrome.runtime.sendMessage({type, document: documentToken, ...extra});
    if (!response?.ok) throw Error(response?.error || "Extension disconnected");
    return response.data;
  }
  async function run(job) {
    const id = job.id;
    const step = async (stage, extra = {}) => { job = await rpc("STEP", {id, stage, ...extra}); };
    async function wait(check, label, allowExport = false) {
      const end = Date.now() + 60000;
      while (Date.now() < end) {
        await rpc("CHECK", {id});
        const reason = D.interruption(document, allowExport);
        if (reason) throw Error(reason);
        const value = check();
        if (value) return value;
        await sleep(500);
      }
      throw Error(`Timed out waiting for ${label}. Check IEEE and Resume.`);
    }
    if (job.stage === "submitting_query") {
      const input = await wait(() => D.command(document), "IEEE Command Search");
      D.fill(input, job.query);
      const search = await wait(() => {
        const b = D.button(input.closest("form"), "Search");
        return b && !b.disabled ? b : null;
      }, "Search button");
      if (input.value !== job.query) throw Error("Exact query no longer matches the search field");
      await step("waiting_for_results"); // Durable before native submission/full navigation.
      search.click();
    }
    if (job.stage === "waiting_for_results") {
      const r = await wait(() => C.sameQuery(location.href, job.query) && D.results(document), "matching rendered results", true);
      // Require a second stable snapshot; do not act on a transient heading.
      await sleep(500);
      const stable = D.results(document);
      if (!stable || stable.total !== r.total || !C.sameQuery(location.href, job.query)) throw Error("IEEE results changed while loading");
      const sort = [...document.querySelectorAll("button")].find(e => D.visible(e) && /Sort By|Relevance|Newest|Oldest/.test(D.text(e)));
      await step("results_ready", {total: r.total, sort: D.text(sort)});
    }
    if (["results_ready", "opening_export", "waiting_for_export_dialog"].includes(job.stage)) {
      if (!C.sameQuery(location.href, job.query)) throw Error("IEEE query changed; Resume the saved query");
      if (D.selected(document)) throw Error("Results are selected in IEEE. Clear them manually, then Resume. The collector never clicks selection controls.");
      if (job.stage === "results_ready") await step("opening_export");
      if (job.stage === "opening_export") {
        const reason = D.interruption(document, true); if (reason) throw Error(reason);
        if (!D.exportDialog(document)) {
          const exportButton = await wait(() => D.button(document, "Export"), "Export button", true);
          await rpc("CHECK", {id}); exportButton.click();
        }
        await step("waiting_for_export_dialog");
      }
      const download = await wait(() => D.verifyDialog(document), "native no-selection CSV dialog", true);
      await step("downloading"); // At-most-once durable intent before the only Download click.
      // Revalidate after the storage round-trip. Never reconstruct or fetch CSV ourselves.
      if (!C.sameQuery(location.href, job.query) || D.verifyDialog(document) !== download) throw Error("IEEE export changed before Download");
      download.click();
    }
  }
  async function tick() {
    if (busy || dead) return;
    busy = true; let job;
    try { job = await rpc("CLAIM"); if (job) await run(job); }
    catch (error) {
      if (/Extension context invalidated|Receiving end does not exist/.test(error.message)) dead = true;
      if (job) await rpc("ERROR", {id: job.id, reason: error.message}).catch(() => {});
    } finally { busy = false; }
  }
  setInterval(tick, 1500);
  tick(); // Only an explicitly authorized active job can be claimed.
})();
