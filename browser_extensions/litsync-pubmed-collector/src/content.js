(() => {
  "use strict";
  if (globalThis.__lsPubMedRunner) return;
  globalThis.__lsPubMedRunner = true;
  const C = LitSyncPubMed, D = C.dom, documentToken = crypto.randomUUID();
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
    async function wait(check, label, allowSave = false) {
      const end = Date.now() + 60000;
      while (Date.now() < end) {
        await rpc("CHECK", {id});
        const reason = D.interruption(document, allowSave); if (reason) throw Error(reason);
        const value = check(); if (value) return value;
        await sleep(500);
      }
      throw Error(`Timed out waiting for ${label}. Check PubMed and Resume.`);
    }
    if (job.stage === "submitting_query") {
      const input = await wait(() => D.queryInput(document), "PubMed search field", true);
      D.fill(input, job.query);
      const search = await wait(() => { const b = D.button(input.closest("form"), "Search"); return b && !b.disabled ? b : null; }, "Search", true);
      if (input.value !== job.query) throw Error("Exact query changed before Search");
      await step("waiting_for_results"); search.click();
    }
    if (job.stage === "waiting_for_results") {
      const r = await wait(() => C.sameQuery(location.href, job.query) && D.results(document, job.query), "matching PubMed results", true);
      await sleep(500);
      if (D.results(document, job.query)?.total !== r.total || !C.sameQuery(location.href, job.query)) throw Error("PubMed results changed while loading");
      await step("results_ready", {total: r.total, sort: D.text(document.querySelector('#id_sort option:checked'))});
    }
    if (job.stage === "results_ready") await step("opening_save");
    if (job.stage === "opening_save") {
      if (!C.sameQuery(location.href, job.query)) throw Error("PubMed query changed");
      if (!D.saveDialog(document)) {
        const save = await wait(() => D.button(document, "Save"), "Save button");
        await rpc("CHECK", {id}); save.click();
      }
      await step("waiting_for_save_dialog");
    }
    const stages = ["waiting_for_save_dialog", "setting_all_results", "setting_csv", "creating_file"];
    if (stages.includes(job.stage)) {
      const panel = await wait(() => D.saveDialog(document), "Save citations to file", true);
      if (job.stage === "waiting_for_save_dialog") await step("setting_all_results");
      if (job.stage === "setting_all_results") {
        D.choose(D.selection(panel), "All results");
        await step("setting_csv");
      }
      if (job.stage === "setting_csv") {
        D.choose(D.format(panel), "CSV");
        await step("creating_file");
      }
      if (!C.sameQuery(location.href, job.query)) throw Error("PubMed query changed before Create file");
      const create = D.verifyExport(document, job.query, job.resultsCount);
      await step("downloading"); // Persist intent before the single native submission.
      if (!C.sameQuery(location.href, job.query) || D.verifyExport(document, job.query, job.resultsCount) !== create) throw Error("PubMed export changed before Create file");
      create.click();
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
  setInterval(tick, 1500); tick();
})();
