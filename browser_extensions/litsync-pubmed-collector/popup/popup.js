"use strict";
const $ = id => document.getElementById(id);
let view = null, pending = false;
const labels = {submitting_query: "Running PubMed search", waiting_for_results: "Waiting for PubMed results", results_ready: "PubMed results ready", opening_save: "Preparing CSV export", waiting_for_save_dialog: "Waiting for Save citations to file", setting_all_results: "Selecting All results", setting_csv: "Selecting CSV", creating_file: "Creating PubMed CSV", downloading: "Downloading PubMed CSV", complete: "Complete — PubMed CSV downloaded", paused: "Action required in PubMed", failed: "Stopped"};
function render() {
  if (!view) return;
  const j = view.job, ctx = view.context;
  const radios = [...document.querySelectorAll('input[name="version"]')];
  for (const radio of radios) radio.disabled = Boolean(j && j.stage !== "complete") || !ctx?.queries[radio.value];
  let selected = radios.find(r => r.checked);
  if (j && j.stage !== "complete") { selected = radios.find(r => r.value === j.version); selected.checked = true; }
  else if (!ctx?.queries[selected.value]) { const available = radios.find(r => ctx?.queries[r.value]); if (available) { selected = available; available.checked = true; } }
  $("sync").textContent = ctx ? "Query synced ✓" : "Open LitSync and generate or display your PubMed queries";
  $("query").textContent = j && j.stage !== "complete" ? j.query : ctx?.queries[selected.value] || "No PubMed query synced";
  $("start").disabled = pending || !ctx?.queries[selected.value] || Boolean(j && j.stage !== "complete");
  $("status").textContent = labels[j?.stage] || "Idle";
  $("counts").textContent = j?.resultsCount != null ? `Results available: ${j.resultsCount.toLocaleString()}` : "";
  $("reason").textContent = j?.reason || (j?.downloadState === "in_progress" ? "PubMed download started; waiting for browser completion." : "");
  $("resume").hidden = !j || j.stage === "complete" || (j.active && j.stage !== "downloading");
  $("resume").textContent = j?.attempt ? "Check download" : "Resume";
  $("pause").hidden = !j?.active || j.stage === "complete" || Boolean(j.attempt);
  $("discard").hidden = !j;
  $("logs").textContent = [...(view.logs || [])].reverse().map(l => `${new Date(l.at).toLocaleTimeString()} ${l.event}`).join("\n");
  for (const id of ["resume", "pause", "discard"]) $(id).disabled = pending;
}
async function request(type) {
  const response = await chrome.runtime.sendMessage({type, version: document.querySelector('input[name="version"]:checked').value});
  if (!response.ok) throw Error(response.error);
  view = response.data; render();
}
for (const [id, type] of [["start", "START"], ["resume", "RESUME"], ["pause", "PAUSE"], ["discard", "DISCARD"]]) {
  $(id).addEventListener("click", async () => {
    if (pending) return; pending = true; render();
    let failure;
    try { await request(type); } catch (error) { failure = error.message; }
    finally { pending = false; render(); if (failure) $("reason").textContent = failure; }
  });
}
document.querySelectorAll('input[name="version"]').forEach(r => r.addEventListener("change", render));
chrome.storage.onChanged.addListener(changes => { if (changes[LitSyncPubMed.KEY]) { view = changes[LitSyncPubMed.KEY].newValue; render(); } });
request("VIEW").catch(error => { $("reason").textContent = error.message; }); // Opening the popup only reads.
