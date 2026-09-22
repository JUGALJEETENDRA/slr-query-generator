"use strict";
importScripts("core.js");
const C = LitSyncPubMed;
let queue = Promise.resolve();
const serial = fn => { const p = queue.then(fn); queue = p.catch(() => {}); return p; };
const read = async () => (await chrome.storage.local.get(C.KEY))[C.KEY] || {context: null, job: null, logs: []};
async function save(s, event) {
  if (s.job) s.job.updatedAt = Date.now();
  if (event) s.logs = [...(s.logs || []), {at: Date.now(), event, stage: s.job?.stage}].slice(-100);
  await chrome.storage.local.set({[C.KEY]: s});
}
function pause(s, reason) {
  if (!s.job) return;
  s.job.resumeStage = s.job.stage;
  s.job.stage = "paused"; s.job.active = false; s.job.reason = reason; s.job.owner = null;
}
async function fingerprint(query) {
  const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(query));
  return [...new Uint8Array(hash)].map(n => n.toString(16).padStart(2, "0")).join("");
}
function popup(sender) { return sender.url === chrome.runtime.getURL("popup/popup.html"); }
function owner(s, m, sender) {
  const j = s.job;
  if (!j?.active || j.id !== m.id || j.tabId !== sender.tab?.id || !C.isPubMed(sender.url)
      || j.owner !== `${sender.documentId || "legacy"}:${m.document}`) throw Error("This document does not own the active job");
  return j;
}
async function reconcile(s) {
  const j = s.job;
  if (!j?.attempt) return;
  const items = j.downloadId != null ? await chrome.downloads.search({id: j.downloadId})
    : (await chrome.downloads.search({startedAfter: new Date(j.attempt.at - 1000).toISOString()})).filter(d => C.downloadMatches(d, j.attempt));
  if (items.length !== 1) {
    pause(s, items.length ? "Multiple possible downloads. Inspect browser Downloads; no retry was triggered." : "CSV download is unconfirmed. Inspect browser Downloads. Resume will only check again; Discard explicitly before retrying.");
    return;
  }
  const d = items[0]; j.downloadId = d.id;
  if (d.state === "complete" && C.csv(d)) {
    j.stage = "complete"; j.active = false; j.reason = null; j.downloadState = "complete"; j.filename = d.filename;
  } else if (d.state === "interrupted") pause(s, "PubMed download was interrupted. Check browser Downloads; it will not be requested twice.");
  else if (d.state === "complete") pause(s, "PubMed download completed but could not be identified as CSV.");
  else { j.stage = "downloading"; j.downloadState = "in_progress"; j.reason = null; }
}
async function route(m, sender) {
  const s = await read();
  if (["SYNC", "CLEAR"].includes(m.type)) {
    if (!C.local(sender.url) || sender.frameId !== 0) throw Error("Untrusted query source");
    s.context = m.type === "CLEAR" ? null : C.context(m.data);
    await save(s); return {};
  }
  if (["VIEW", "START", "RESUME", "DISCARD", "PAUSE"].includes(m.type)) {
    if (!popup(sender)) throw Error("Explicit popup action required");
    if (m.type === "VIEW") return s; // No browser or storage mutation.
    if (m.type === "DISCARD") { s.job = null; await save(s, "Checkpoint explicitly discarded; website and downloads unchanged"); return s; }
    if (m.type === "PAUSE") { pause(s, "Stopped by user"); await save(s, "Paused"); return s; }
    if (m.type === "START") {
      if (s.job && s.job.stage !== "complete") throw Error("An incomplete run exists. Resume or explicitly Discard it.");
      const query = s.context?.queries[m.version];
      if (!query) throw Error("Sync this query version from LitSync first");
      const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
      if (!C.searchSurface(tab?.url)) throw Error("Open PubMed's search page, then click Start");
      s.job = C.newJob(s.context, m.version, await fingerprint(query), tab.id);
      s.job.litSyncFingerprint = s.context.fingerprints[m.version];
      await save(s, "Explicit Start authorized exact query submission"); return s;
    }
    if (!s.job || s.job.stage === "complete") return s;
    if (s.job.attempt) { await reconcile(s); await save(s, "Checked existing download without clicking Download again"); return s; }
    let tab;
    try { tab = await chrome.tabs.get(s.job.tabId); } catch { /* closed tab */ }
    if (!tab || !(C.sameQuery(tab.url, s.job.query) || C.searchSurface(tab.url))) {
      tab = await chrome.tabs.create({url: s.job.resultsUrl || C.HOME, active: true});
    }
    if (s.job.resultsUrl && !C.sameQuery(tab.url, s.job.query)) {
      tab = await chrome.tabs.update(tab.id, {url: s.job.resultsUrl, active: true});
    }
    s.job.tabId = tab.id; s.job.owner = null; s.job.reason = null; s.job.active = true;
    s.job.stage = C.sameQuery(tab.url, s.job.query) || s.job.resultsUrl ? "waiting_for_results" : "submitting_query";
    await save(s, "Explicit Resume"); return s;
  }
  if (m.type === "CLAIM") {
    const j = s.job;
    if (!j?.active || j.tabId !== sender.tab?.id || !C.isPubMed(sender.url) || sender.frameId !== 0 || j.attempt) return null;
    const document = `${sender.documentId || "legacy"}:${m.document}`;
    if (j.owner && j.owner !== document && !(j.stage === "waiting_for_results" && sender.documentId && j.ownerDocument !== sender.documentId)) {
      pause(s, "PubMed document or extension restarted. Resume the preserved run."); await save(s, "Owner changed; explicit Resume required"); return null;
    }
    j.owner = document; j.ownerDocument = sender.documentId; await save(s); return j;
  }
  const j = owner(s, m, sender);
  if (m.type === "CHECK") return j;
  if (m.type === "ERROR") { pause(s, String(m.reason).slice(0,500)); await save(s, j.reason); return {}; }
  const transitions = {
    submitting_query: ["waiting_for_results"], waiting_for_results: ["results_ready"],
    results_ready: ["opening_save"], opening_save: ["waiting_for_save_dialog"],
    waiting_for_save_dialog: ["setting_all_results"], setting_all_results: ["setting_csv"],
    setting_csv: ["creating_file"], creating_file: ["downloading"]
  };
  if (m.type !== "STEP" || !transitions[j.stage]?.includes(m.stage)) throw Error("Invalid or duplicate state transition");
  if (m.stage === "results_ready") {
    if (!C.sameQuery(sender.url, j.query) || !Number.isSafeInteger(m.total) || m.total < 1) throw Error("Results do not match this job");
    j.resultsCount = m.total; j.resultsUrl = sender.url; j.sort = String(m.sort || "unknown").slice(0,100);
  }
  if (m.stage === "downloading") {
    if (j.attempt) throw Error("Download already requested");
    j.attempt = {at: Date.now(), id: crypto.randomUUID()};
  }
  j.stage = m.stage; await save(s, m.stage);
  if (m.stage === "downloading") await chrome.alarms.create(`pubmed-download-${j.id}`, {delayInMinutes: 0.5});
  return j;
}
chrome.runtime.onMessage.addListener((m, sender, reply) => {
  serial(() => route(m, sender)).then(data => reply({ok: true, data}), error => reply({ok: false, error: error.message}));
  return true;
});
chrome.downloads.onCreated.addListener(item => serial(async () => {
  const s = await read(), j = s.job;
  if (!j?.attempt || j.stage === "complete" || !C.downloadMatches(item, j.attempt)) return;
  if (j.downloadId != null && j.downloadId !== item.id) {
    pause(s, "Multiple PubMed downloads detected. Inspect browser Downloads."); await save(s, j.reason); return;
  }
  else { j.downloadId = item.id; j.downloadState = item.state; }
  await reconcile(s); await save(s, "Native download observed");
}).catch(() => {}));
chrome.downloads.onChanged.addListener(delta => serial(async () => {
  const s = await read();
  if (s.job?.downloadId !== delta.id || s.job.stage === "complete") return;
  await reconcile(s); await save(s, "Native download updated");
}).catch(() => {}));
chrome.alarms.onAlarm.addListener(alarm => serial(async () => {
  const s = await read();
  if (alarm.name !== `pubmed-download-${s.job?.id}` || !s.job?.attempt || s.job.stage === "complete") return;
  await reconcile(s); await save(s, "Download confirmation check");
}).catch(() => {}));
const suspended = reason => serial(async () => {
  const s = await read(); if (s.job && s.job.stage !== "complete") { pause(s, reason); await save(s, reason); }
}).catch(() => {});
chrome.runtime.onStartup.addListener(() => suspended("Browser restarted. Resume the existing run."));
chrome.runtime.onInstalled.addListener(() => suspended("Extension installed or updated. Resume the existing run."));
chrome.tabs.onRemoved.addListener(id => serial(async () => {
  const s = await read(); if (s.job?.tabId === id && s.job.stage !== "complete") { pause(s, "PubMed tab closed. Resume to restore it."); await save(s); }
}).catch(() => {}));
