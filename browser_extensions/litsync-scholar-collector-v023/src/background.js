"use strict";
importScripts("core.js");
const Core = globalThis.LitSyncScholarCore;

const MAX_LOGS = 200;

async function getStored() {
  return chrome.storage.local.get([Core.STORAGE_KEYS.context, Core.STORAGE_KEYS.run, Core.STORAGE_KEYS.logs]);
}

async function appendLog(level, event, details = {}) {
  const stored = await chrome.storage.local.get(Core.STORAGE_KEYS.logs);
  const logs = Array.isArray(stored[Core.STORAGE_KEYS.logs]) ? stored[Core.STORAGE_KEYS.logs] : [];
  logs.push({at: new Date().toISOString(), level, event, details});
  if (logs.length > MAX_LOGS) logs.splice(0, logs.length - MAX_LOGS);
  await chrome.storage.local.set({[Core.STORAGE_KEYS.logs]: logs});
}

async function currentRun() {
  const stored = await chrome.storage.local.get(Core.STORAGE_KEYS.run);
  return stored[Core.STORAGE_KEYS.run] || null;
}

async function setRun(run) {
  const next = {...run, updatedAt: new Date().toISOString()};
  await chrome.storage.local.set({[Core.STORAGE_KEYS.run]: next});
  return next;
}

async function patchRun(patch, sender) {
  const run = await currentRun();
  if (!run) throw new Error("No Scholar collection run exists");
  if (sender && sender.tab && run.tabId != null && sender.tab.id !== run.tabId) {
    throw new Error("This Scholar tab does not own the active LitSync Scholar run");
  }
  return setRun({...run, ...patch});
}



async function verifyScholarCsvRequest(run, request) {
  const expectedRows = Core.expectedExportRows(run);
  const rawUrl = String(request && request.url || "").trim();
  if (!rawUrl) throw new Error("Scholar export request URL was not captured");

  let parsed;
  try { parsed = new URL(rawUrl); } catch (_error) { throw new Error("Scholar export request URL is invalid"); }
  if (!["scholar.google.com", "scholar.googleusercontent.com"].includes(parsed.hostname)) {
    throw new Error("Refusing to fetch a non-Scholar export URL");
  }
  if (!parsed.pathname.includes("citations")) throw new Error("Refusing to fetch a non-citation Scholar URL");

  const method = String(request && request.method || "GET").toUpperCase() === "POST" ? "POST" : "GET";
  const body = method === "POST" ? String(request && request.body || "") : undefined;
  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer = controller ? setTimeout(() => controller.abort(), 30000) : null;
  try {
    const response = await fetch(parsed.href, {
      method,
      credentials: "include",
      cache: "no-store",
      redirect: "follow",
      headers: method === "POST" ? {"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"} : undefined,
      body,
      signal: controller ? controller.signal : undefined
    });
    if (!response || !response.ok) throw new Error(`Scholar CSV pre-download fetch failed with HTTP ${response && response.status}`);
    const text = await response.text();
    const inspection = Core.inspectScholarCsv(text, expectedRows);
    return {inspection, text};
  } finally {
    if (timer) clearTimeout(timer);
  }
}
async function verifyScholarCsvUrl(run, url) {
  const expectedRows = Core.expectedExportRows(run);
  if (!url) throw new Error("Scholar download URL was not captured");
  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer = controller ? setTimeout(() => controller.abort(), 30000) : null;
  try {
    const response = await fetch(url, {
      method: "GET",
      credentials: "include",
      cache: "no-store",
      redirect: "follow",
      signal: controller ? controller.signal : undefined
    });
    if (!response || !response.ok) throw new Error(`Scholar CSV verification fetch failed with HTTP ${response && response.status}`);
    const text = await response.text();
    const inspection = Core.inspectScholarCsv(text, expectedRows);
    return {inspection, text};
  } finally {
    if (timer) clearTimeout(timer);
  }
}


function localPathToFileUrl(filename) {
  const raw = String(filename || "").trim();
  if (!raw) throw new Error("Downloaded CSV filename was not available");
  const normalized = raw.replace(/\\/g, "/");
  const encodeParts = parts => parts.map(part => encodeURIComponent(part)).join("/");

  // Windows drive path, e.g. C:\\Users\\name\\Downloads\\citations.csv
  if (/^[A-Za-z]:\//.test(normalized)) {
    const drive = normalized.slice(0, 2);
    const rest = normalized.slice(3).split("/");
    return `file:///${drive}/${encodeParts(rest)}`;
  }

  // UNC path, e.g. \\\\server\\share\\citations.csv
  if (normalized.startsWith("//")) {
    const parts = normalized.slice(2).split("/");
    const host = parts.shift();
    if (!host) throw new Error("Downloaded CSV UNC path was invalid");
    return `file://${host}/${encodeParts(parts)}`;
  }

  // POSIX path.
  if (normalized.startsWith("/")) {
    return `file:///${encodeParts(normalized.slice(1).split("/"))}`;
  }
  throw new Error("Downloaded CSV path was not absolute");
}

async function fileSchemeAccessAllowed() {
  const api = chrome.extension && chrome.extension.isAllowedFileSchemeAccess;
  if (typeof api !== "function") return true; // Older Chromium builds: try the read and surface its actual error.
  return new Promise(resolve => {
    let settled = false;
    const finish = value => {
      if (settled) return;
      settled = true;
      resolve(Boolean(value));
    };
    try {
      const maybe = api.call(chrome.extension, finish);
      if (maybe && typeof maybe.then === "function") maybe.then(finish).catch(() => finish(false));
    } catch (_error) {
      finish(false);
    }
  });
}

async function fetchFileTextWithRetry(fileUrl, attempts = 10, delayMs = 300) {
  let lastError = null;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(fileUrl, {method: "GET", cache: "no-store"});
      if (!response || !response.ok) {
        throw new Error(`Downloaded CSV file read failed with HTTP ${response && response.status}`);
      }
      return await response.text();
    } catch (error) {
      lastError = error;
      if (attempt < attempts) await new Promise(resolve => setTimeout(resolve, delayMs));
    }
  }
  throw lastError || new Error("Downloaded CSV file could not be read");
}

async function verifyDownloadedCsvFile(run, downloadId) {
  if (!chrome.downloads || typeof chrome.downloads.search !== "function") {
    throw new Error("The browser does not expose the downloaded file metadata needed for CSV verification");
  }
  const items = await chrome.downloads.search({id: downloadId});
  const item = Array.isArray(items) ? items[0] : null;
  if (!item || !item.filename) throw new Error("The completed Scholar CSV download could not be located on disk");
  if (item.state && item.state !== "complete") throw new Error(`Scholar CSV download is not complete (${item.state})`);

  const allowed = await fileSchemeAccessAllowed();
  if (!allowed) {
    const error = new Error("Edge has not granted this unpacked extension access to local file URLs");
    error.code = "FILE_SCHEME_ACCESS_DISABLED";
    throw error;
  }

  const fileUrl = localPathToFileUrl(item.filename);
  const text = await fetchFileTextWithRetry(fileUrl);
  const expectedRows = Core.expectedExportRows(run);
  const inspection = Core.inspectScholarCsv(text, expectedRows);
  return {inspection, text, item};
}

function csvVerificationPatch(inspection) {
  return {
    exportCsvVerified: Boolean(inspection && inspection.ok),
    exportCsvRowCount: inspection && Number.isInteger(inspection.rowCount) ? inspection.rowCount : null,
    exportCsvExpectedRows: inspection ? inspection.expectedRows : null,
    exportCsvHeader: inspection && Array.isArray(inspection.header) ? inspection.header : [],
    exportCsvVerificationError: inspection && inspection.errors ? inspection.errors.join("; ") : ""
  };
}

async function finalizeExportInspection(run, inspection, downloadId) {
  const unresolvedCount = Core.unresolvedIds(run).length;
  const saved = Core.progress(run).saved;
  const target = run.targetTotal ?? Core.RESULT_CAP;
  const labelVerified = Core.uniqueStrings(run.labelVerifiedIds || []).length;
  const labelIntegrityOk = Core.labelIntegritySatisfied(run);
  const csvOk = Boolean(inspection && inspection.ok);
  const complete = unresolvedCount === 0
    && saved >= target
    && labelIntegrityOk
    && csvOk
    && run.exportMode !== "partial";

  const rowSummary = inspection ? `${inspection.rowCount}/${inspection.expectedRows}` : "unverified";
  const failure = inspection && inspection.errors && inspection.errors.length
    ? inspection.errors.join("; ")
    : `checkpointed=${saved}/${target}, verified-in-label=${labelVerified}/${target}, unresolved=${unresolvedCount}`;
  const updated = await setRun({
    ...run,
    ...csvVerificationPatch(inspection),
    status: complete ? Core.STATUS.COMPLETE : Core.STATUS.INCOMPLETE,
    stage: Core.STAGE.DONE,
    message: complete
      ? `Complete: ${saved}/${target} Scholar records saved, ${labelVerified}/${target} verified in the LitSync label, and the downloaded CSV was verified at exactly ${rowSummary} data rows.`
      : `Export is not complete: ${failure}. CSV rows=${rowSummary}. Click Resume to retry without recollecting.`
  });
  await appendLog(complete ? "info" : "warn", complete ? "run_complete" : "run_export_incomplete", {
    runId: run.runId, downloadId, saved, target, labelVerified, unresolvedCount, labelIntegrityOk, csvOk,
    rowCount: inspection && inspection.rowCount, expectedRows: inspection && inspection.expectedRows
  });
  return updated;
}

function resetExportVerification(run) {
  return {
    ...run,
    exportDownloadId: null,
    exportDownloadUrl: "",
    downloadedFilename: "",
    exportCsvVerified: false,
    exportCsvRowCount: null,
    exportCsvExpectedRows: Core.expectedExportRows(run),
    exportCsvHeader: [],
    exportCsvVerificationError: ""
  };
}

async function saveVerifiedCsvText(run, text, sourceUrl = "") {
  const expectedRows = Core.expectedExportRows(run);
  const inspection = Core.inspectScholarCsv(text, expectedRows);
  if (!inspection.ok) return {ok: false, inspection, run};
  const filename = Core.sanitizeFilename(`scholar_${run.label}.csv`);
  const dataUrl = `data:text/csv;charset=utf-8,${encodeURIComponent(String(text || ""))}`;
  const downloadId = await chrome.downloads.download({url: dataUrl, filename, saveAs: false});
  const updated = await setRun({
    ...run,
    stage: Core.STAGE.EXPORTING,
    exportStartedAt: Date.now(),
    exportDownloadId: downloadId,
    exportDownloadUrl: String(sourceUrl || ""),
    downloadedFilename: filename,
    ...csvVerificationPatch(inspection),
    message: `Validated ${inspection.rowCount}/${expectedRows} CSV rows. Saving the verified CSV…`
  });
  return {ok: true, inspection, run: updated, downloadId};
}

async function ensureRunTab(run) {
  if (run.tabId != null) {
    try {
      const tab = await chrome.tabs.get(run.tabId);
      if (tab) return tab;
    } catch (_error) {}
  }
  const tab = await chrome.tabs.create({url: run.lastUrl || "https://scholar.google.com/", active: true});
  await setRun({...run, tabId: tab.id, message: "Scholar tab reopened. Resuming…"});
  return tab;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    const type = message && message.type;
    if (type === "LSSC_SAVE_CONTEXT") {
      if (!Core.validateContext(message.context)) throw new Error("Rejected invalid Scholar query context");
      await chrome.storage.local.set({[Core.STORAGE_KEYS.context]: message.context});
      await appendLog("info", "query_context_saved", {fingerprint: message.context.query_fingerprint});
      return {ok: true};
    }
    if (type === "LSSC_CLEAR_CONTEXT") {
      await chrome.storage.local.remove(Core.STORAGE_KEYS.context);
      return {ok: true};
    }
    if (type === "LSSC_GET_SNAPSHOT") {
      const stored = await getStored();
      return {
        ok: true,
        context: stored[Core.STORAGE_KEYS.context] || null,
        run: stored[Core.STORAGE_KEYS.run] || null,
        logs: stored[Core.STORAGE_KEYS.logs] || [],
        senderTabId: sender && sender.tab ? sender.tab.id : null
      };
    }
    if (type === "LSSC_START") {
      const stored = await getStored();
      const context = stored[Core.STORAGE_KEYS.context];
      if (!Core.validateContext(context)) throw new Error("Generate a LitSync query first; no synced Google Scholar query was found.");
      const existing = stored[Core.STORAGE_KEYS.run];
      if (existing && !Core.fullExportIntegritySatisfied(existing)) {
        // Never overwrite a checkpointed run (paused, attention, incomplete,
        // legacy-complete, or running) just because Start was clicked again.
        // Resume/Reset are the explicit actions for that run.
        await ensureRunTab(existing);
        return {ok: true, run: existing, reused: true};
      }
      const run = Core.createRun(context);
      const searchUrl = Core.buildSearchUrl(run.query);
      const tab = await chrome.tabs.create({url: searchUrl, active: true});
      run.tabId = tab.id;
      run.stage = Core.STAGE.SEARCH_LOADING;
      run.lastUrl = searchUrl;
      run.message = "Opening the exact synced Google Scholar search…";
      await setRun(run);
      await appendLog("info", "run_started", {runId: run.runId, label: run.label});
      await new Promise(resolve => setTimeout(resolve, 500));
      try { await chrome.tabs.sendMessage(tab.id, {type: "LSSC_WAKE"}); } catch (_error) {}
      return {ok: true, run};
    }
    if (type === "LSSC_PATCH_RUN") {
      const run = await patchRun(message.patch || {}, sender);
      return {ok: true, run};
    }
    if (type === "LSSC_PAUSE") {
      const run = await currentRun();
      if (!run) return {ok: true, run: null};
      const updated = await setRun({...run, status: Core.STATUS.PAUSED, message: "Paused. Click Resume when ready."});
      await appendLog("info", "run_paused", {runId: run.runId});
      return {ok: true, run: updated};
    }
    if (type === "LSSC_RESUME") {
      const run = await currentRun();
      if (!run) throw new Error("No Scholar run to resume");
      const needsIntegrityAudit = !Core.labelIntegritySatisfied(run);
      const completionVerified = Core.fullExportIntegritySatisfied(run);
      if (run.status === Core.STATUS.COMPLETE && completionVerified) return {ok: true, run};

      // If collection + label integrity already passed and a native CSV exists,
      // Resume verifies that exact downloaded file first. This avoids another
      // Scholar export and never recollects papers just because verification
      // needed a retry (for example after enabling file:// access).
      if ([Core.STATUS.NEEDS_ATTENTION, Core.STATUS.INCOMPLETE].includes(run.status)
          && run.stage === Core.STAGE.DONE
          && run.exportDownloadId != null
          && run.exportCsvVerified !== true
          && Core.labelIntegritySatisfied(run)) {
        try {
          const verified = await verifyDownloadedCsvFile(run, run.exportDownloadId);
          const verifiedRun = await setRun({
            ...run,
            downloadedFilename: verified.item && verified.item.filename || run.downloadedFilename || "",
            ...csvVerificationPatch(verified.inspection)
          });
          await appendLog(verified.inspection.ok ? "info" : "error", "resume_existing_download_validated", {
            runId: run.runId,
            downloadId: run.exportDownloadId,
            rowCount: verified.inspection.rowCount,
            expectedRows: verified.inspection.expectedRows,
            errors: verified.inspection.errors
          });
          const finalized = await finalizeExportInspection(verifiedRun, verified.inspection, run.exportDownloadId);
          return {ok: true, run: finalized, verifiedExistingDownload: true};
        } catch (error) {
          const rawError = String(error && error.message || error);
          if (error && error.code === "FILE_SCHEME_ACCESS_DISABLED") {
            const blocked = await setRun({
              ...run,
              status: Core.STATUS.NEEDS_ATTENTION,
              stage: Core.STAGE.DONE,
              exportCsvVerified: false,
              exportCsvVerificationError: rawError,
              message: "The CSV is already downloaded. In Edge extension Details, enable 'Allow access to file URLs', then click Resume again. It will verify the existing CSV without recollecting or re-exporting."
            });
            await appendLog("warn", "resume_existing_download_file_access_blocked", {runId: run.runId, downloadId: run.exportDownloadId});
            return {ok: true, run: blocked, needsFileAccess: true};
          }
          await appendLog("warn", "resume_existing_download_read_failed", {runId: run.runId, downloadId: run.exportDownloadId, error: rawError});
          // The file may have been moved/deleted. Fall through to the existing
          // safe re-export path; saved papers and label membership stay intact.
        }
      }

      let resumePatch = {status: Core.STATUS.RUNNING, message: "Resuming from the last checkpoint…"};
      if (run.status === Core.STATUS.COMPLETE && !completionVerified) {
        resumePatch = {
          ...resumePatch,
          stage: Core.STAGE.OPENING_LIBRARY,
          integrityVersion: 3,
          labelAuditIds: [],
          labelAuditPages: [],
          labelAuditComplete: false,
          labelVerifiedIds: [],
          labelVerifiedCount: null,
          labelMissingIds: [],
          labelUnexpectedIds: [],
          labelRepairIds: [],
          labelRepairAttemptedIds: [],
          exportDownloadId: null,
          exportDownloadUrl: "",
          exportStartedAt: null,
          exportFormatChosen: false,
          exportScopeConfirmed: false,
          exportCsvVerified: false,
          exportCsvRowCount: null,
          exportCsvExpectedRows: Core.expectedExportRows(run),
          exportCsvHeader: [],
          exportCsvVerificationError: "",
          message: needsIntegrityAudit
            ? "Revalidating the actual Scholar label before accepting the previous completion…"
            : "The previous completion did not verify the downloaded CSV contents. Re-exporting and validating the exact row count…"
        };
      }
      if (run.status === Core.STATUS.INCOMPLETE && Core.unresolvedIds(run).length) {
        const unresolved = Object.fromEntries(Object.entries(run.unresolved || {}).map(([cid, details]) => [cid, {...details, finalized: false}]));
        resumePatch = {
          ...resumePatch,
          unresolved,
          stage: Core.STAGE.RETRYING_FAILURES,
          retryRound: Number(run.retryRound || 0) + 1,
          exportMode: "full",
          exportDownloadId: null,
          exportStartedAt: null,
          message: `Retrying ${Core.unresolvedIds({unresolved}, {includeFinalized: false}).length} unresolved Scholar record(s) from the incomplete export…`
        };
      } else if (run.status === Core.STATUS.INCOMPLETE && !Core.labelIntegritySatisfied(run)) {
        resumePatch = {
          ...resumePatch,
          stage: Core.STAGE.OPENING_LIBRARY,
          labelAuditIds: [],
          labelAuditPages: [],
          labelAuditComplete: false,
          labelVerifiedIds: [],
          labelVerifiedCount: null,
          labelMissingIds: [],
          labelUnexpectedIds: [],
          labelRepairIds: [],
          labelRepairAttemptedIds: [],
          exportMode: "full",
          exportDownloadId: null,
          exportDownloadUrl: "",
          exportStartedAt: null,
          exportFormatChosen: false,
          exportScopeConfirmed: false,
          exportCsvVerified: false,
          exportCsvRowCount: null,
          exportCsvExpectedRows: Core.expectedExportRows(run),
          exportCsvHeader: [],
          exportCsvVerificationError: "",
          message: "Re-auditing the actual LitSync label after an integrity-incomplete CSV export…"
        };
      } else if ([Core.STATUS.INCOMPLETE, Core.STATUS.NEEDS_ATTENTION].includes(run.status)
          && Core.labelIntegritySatisfied(run)
          && !Core.fullExportIntegritySatisfied(run)
          && run.stage === Core.STAGE.DONE) {
        resumePatch = {
          ...resumePatch,
          stage: Core.STAGE.OPENING_LIBRARY,
          exportMode: "full",
          exportDownloadId: null,
          exportDownloadUrl: "",
          exportStartedAt: null,
          exportFormatChosen: false,
          exportScopeConfirmed: false,
          exportCsvVerified: false,
          exportCsvRowCount: null,
          exportCsvExpectedRows: Core.expectedExportRows(run),
          exportCsvHeader: [],
          exportCsvVerificationError: "",
          message: "The label is intact, but the prior CSV was not verified. Re-exporting the full label and validating its rows…"
        };
      }
      let updated = await setRun({...run, ...resumePatch});
      const tab = await ensureRunTab(updated);

      // Raw googleusercontent export pages may be rendered as text/plain, where
      // a content script is not guaranteed to run. Validate that URL from the
      // service worker itself. Valid full CSV is saved; BibTeX/short CSV is sent
      // back through Scholar's native export UI without touching saved records.
      const tabUrl = String(tab && tab.url || "");
      if (tabUrl.includes("scholar.googleusercontent.com/") && [Core.STAGE.EXPORTING, Core.STAGE.EXPORT_MENU].includes(updated.stage)) {
        try {
          const verified = await verifyScholarCsvUrl(updated, tabUrl);
          if (verified.inspection.ok) {
            const saved = await saveVerifiedCsvText(updated, verified.text, tabUrl);
            await appendLog("info", "resume_rendered_csv_validated", {runId: run.runId, rowCount: verified.inspection.rowCount, expectedRows: verified.inspection.expectedRows});
            return {ok: true, run: saved.run};
          }
          updated = await setRun({
            ...resetExportVerification(updated),
            stage: Core.STAGE.OPENING_LIBRARY,
            exportStartedAt: null,
            exportFormatChosen: false,
            exportScopeConfirmed: false,
            exportCsvVerificationError: verified.inspection.errors.join("; "),
            message: `The rendered Scholar export was not the required full CSV (${verified.inspection.errors.join("; ")}). Returning to My Library to export again…`
          });
          await appendLog("warn", "resume_rendered_export_rejected", {runId: run.runId, tabUrl, inspection: verified.inspection});
        } catch (error) {
          updated = await setRun({
            ...resetExportVerification(updated),
            stage: Core.STAGE.OPENING_LIBRARY,
            exportStartedAt: null,
            exportFormatChosen: false,
            exportScopeConfirmed: false,
            exportCsvVerificationError: String(error && error.message || error),
            message: "Could not verify the rendered Scholar export page. Returning to My Library to retry the native CSV export…"
          });
          await appendLog("warn", "resume_rendered_export_fetch_failed", {runId: run.runId, tabUrl, error: String(error && error.message || error)});
        }
        await chrome.tabs.update(tab.id, {url: "https://scholar.google.com/scholar?scilib=1&hl=en", active: true});
        return {ok: true, run: updated};
      }

      try { await chrome.tabs.update(tab.id, {active: true}); } catch (_error) {}
      await appendLog("info", "run_resumed", {runId: run.runId});
      try {
        await chrome.tabs.sendMessage(tab.id, {type: "LSSC_WAKE"});
      } catch (error) {
        // Reloading an unpacked extension invalidates content scripts that were
        // injected by the previous build. A tab reload safely injects v0.2.1
        // and preserves every checkpoint in chrome.storage.local.
        try {
          await chrome.tabs.reload(tab.id);
          await appendLog("info", "run_tab_reloaded_for_fresh_content_script", {runId: run.runId, tabId: tab.id});
        } catch (reloadError) {
          await appendLog("warn", "run_wake_failed", {runId: run.runId, error: String(error && error.message || error), reloadError: String(reloadError && reloadError.message || reloadError)});
        }
      }
      return {ok: true, run: updated};
    }
    if (type === "LSSC_RESET") {
      const run = await currentRun();
      if (run) await appendLog("info", "run_reset", {runId: run.runId});
      await chrome.storage.local.remove(Core.STORAGE_KEYS.run);
      return {ok: true};
    }
    if (type === "LSSC_LOG") {
      await appendLog(message.level || "info", message.event || "event", message.details || {});
      return {ok: true};
    }
    if (type === "LSSC_EXPORT_TRIGGERED") {
      const before = await currentRun();
      if (!before) throw new Error("No Scholar run exists for export");
      const run = await patchRun({
        stage: Core.STAGE.EXPORTING,
        exportStartedAt: Date.now(),
        exportDownloadId: null,
        exportDownloadUrl: "",
        downloadedFilename: "",
        exportCsvVerified: false,
        exportCsvRowCount: null,
        exportCsvExpectedRows: Core.expectedExportRows(before),
        exportCsvHeader: [],
        exportCsvVerificationError: "",
        message: "CSV export clicked. Waiting for the browser download, then validating its contents…"
      }, sender);
      await appendLog("info", "export_triggered", {runId: run.runId});
      return {ok: true, run};
    }
    if (type === "LSSC_VERIFY_EXPORT_REQUEST") {
      const run = await currentRun();
      if (!run) throw new Error("No Scholar run exists for export verification");
      if (sender && sender.tab && run.tabId != null && sender.tab.id !== run.tabId) {
        throw new Error("This Scholar tab does not own the active LitSync Scholar run");
      }
      if (![Core.STAGE.EXPORTING, Core.STAGE.EXPORT_MENU].includes(run.stage)) {
        throw new Error("Scholar export request arrived outside the export stage");
      }
      try {
        const verified = await verifyScholarCsvRequest(run, message.request || {});
        if (!verified.inspection.ok) {
          await appendLog("warn", "predownload_csv_validation_rejected", {
            runId: run.runId,
            rowCount: verified.inspection.rowCount,
            expectedRows: verified.inspection.expectedRows,
            errors: verified.inspection.errors
          });
          return {ok: false, fallback: true, error: verified.inspection.errors.join("; ")};
        }
        const saved = await saveVerifiedCsvText(run, verified.text, message.request && message.request.url || "");
        await appendLog("info", "predownload_csv_validated_and_saved", {
          runId: run.runId,
          downloadId: saved.downloadId,
          rowCount: verified.inspection.rowCount,
          expectedRows: verified.inspection.expectedRows
        });
        return {ok: true, captured: true, run: saved.run};
      } catch (error) {
        await appendLog("warn", "predownload_csv_capture_failed", {
          runId: run.runId,
          error: String(error && error.message || error)
        });
        return {ok: false, fallback: true, error: String(error && error.message || error)};
      }
    }
    if (type === "LSSC_RENDERED_CSV") {
      const run = await currentRun();
      if (!run) throw new Error("No Scholar run exists for rendered CSV");
      if (![Core.STAGE.EXPORTING, Core.STAGE.EXPORT_MENU].includes(run.stage)) throw new Error("Rendered CSV arrived outside the Scholar export stage");
      const expectedRows = Core.expectedExportRows(run);
      const inspection = Core.inspectScholarCsv(message.text || "", expectedRows);
      if (!inspection.ok) {
        const updated = await setRun({
          ...run,
          status: Core.STATUS.NEEDS_ATTENTION,
          stage: Core.STAGE.DONE,
          ...csvVerificationPatch(inspection),
          message: `Scholar rendered an export page, but CSV validation failed: ${inspection.errors.join("; ")}`
        });
        await appendLog("error", "rendered_csv_validation_failed", {runId: run.runId, inspection});
        return {ok: false, error: updated.message, run: updated};
      }
      const saved = await saveVerifiedCsvText(run, message.text || "", message.pageUrl || "");
      await appendLog("info", "rendered_csv_validated_and_downloaded", {runId: run.runId, downloadId: saved.downloadId, rowCount: inspection.rowCount, expectedRows});
      return {ok: true, run: saved.run};
    }
    return {ok: false, error: "Unknown message"};
  })().then(sendResponse).catch(async error => {
    try { await appendLog("error", "background_error", {message: String(error && error.message || error)}); } catch (_ignored) {}
    sendResponse({ok: false, error: String(error && error.message || error)});
  });
  return true;
});

chrome.downloads.onCreated.addListener(async item => {
  const run = await currentRun();
  if (!run || run.stage !== Core.STAGE.EXPORTING || !run.exportStartedAt) return;
  if (run.exportDownloadId != null) return; // rendered-CSV path already owns its download
  const url = String(item.finalUrl || item.url || "");
  const filename = String(item.filename || "");
  const mime = String(item.mime || "").toLowerCase();
  const recent = Date.now() - Number(run.exportStartedAt) < 120000;
  const scholarSource = url.includes("scholar.google.com") || url.includes("scholar.googleusercontent.com");
  const exportish = /(?:\/|\?)citations(?:\?|$)/i.test(url)
    || /view_op=export_citations/i.test(url)
    || filename.toLowerCase().endsWith(".csv")
    || mime.includes("csv");
  if (!recent || !scholarSource || !exportish) return;
  await setRun({
    ...run,
    exportDownloadId: item.id,
    exportDownloadUrl: url,
    downloadedFilename: filename,
    exportCsvExpectedRows: Core.expectedExportRows(run),
    exportCsvVerified: false
  });
  await appendLog("info", "download_created", {id: item.id, filename, url, mime});
});

chrome.downloads.onChanged.addListener(async delta => {
  if (!delta || !delta.id || !delta.state) return;
  let run = await currentRun();
  if (!run || run.exportDownloadId !== delta.id) return;
  if (delta.state.current === "complete") {
    let inspection = null;
    if (run.exportCsvVerified === true) {
      inspection = {
        ok: true,
        rowCount: run.exportCsvRowCount,
        expectedRows: run.exportCsvExpectedRows,
        header: run.exportCsvHeader || [],
        errors: []
      };
    } else {
      try {
        // Verify the bytes that Edge actually wrote to disk. Scholar export URLs
        // are short-lived/one-use and can legitimately return 404 if fetched
        // again after the native download has already consumed them.
        const verified = await verifyDownloadedCsvFile(run, delta.id);
        inspection = verified.inspection;
        run = await setRun({
          ...run,
          downloadedFilename: verified.item && verified.item.filename || run.downloadedFilename || "",
          ...csvVerificationPatch(inspection)
        });
        await appendLog(inspection.ok ? "info" : "error", "downloaded_csv_file_validated", {
          runId: run.runId,
          downloadId: delta.id,
          rowCount: inspection.rowCount,
          expectedRows: inspection.expectedRows,
          header: inspection.header,
          errors: inspection.errors
        });
      } catch (error) {
        const rawError = String(error && error.message || error);
        const fileAccessDisabled = error && error.code === "FILE_SCHEME_ACCESS_DISABLED";
        const message = fileAccessDisabled
          ? "CSV downloaded successfully, but Edge has not granted this unpacked extension access to local file URLs. Open this extension's Details, enable 'Allow access to file URLs', then click Resume. The existing CSV will be verified; no recollection is needed."
          : `CSV downloaded successfully, but the exact downloaded file could not be read for verification: ${rawError}. Click Resume to retry verification of the existing file; no recollection is needed.`;
        await setRun({
          ...run,
          status: Core.STATUS.NEEDS_ATTENTION,
          stage: Core.STAGE.DONE,
          exportCsvVerified: false,
          exportCsvVerificationError: rawError,
          message
        });
        await appendLog("error", "downloaded_csv_file_verification_failed", {runId: run.runId, downloadId: delta.id, error: rawError, fileAccessDisabled});
        return;
      }
    }

    await finalizeExportInspection(run, inspection, delta.id);
  } else if (delta.state.current === "interrupted") {
    await setRun({...run, status: Core.STATUS.NEEDS_ATTENTION, stage: Core.STAGE.DONE, message: "The Scholar CSV download was interrupted. Click Resume to retry the export without recollecting."});
    await appendLog("error", "download_interrupted", {runId: run.runId, downloadId: delta.id});
  }
});
