(function (root, factory) {
  const api = factory();
  root.LitSyncScholarCore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const SCHEMA_VERSION = 1;
  const RESULT_CAP = 1000;
  const STORAGE_KEYS = Object.freeze({
    context: "lsscScholarQueryContext",
    run: "lsscScholarRun",
    logs: "lsscScholarLogs"
  });

  const STATUS = Object.freeze({
    IDLE: "idle",
    RUNNING: "running",
    PAUSED: "paused",
    NEEDS_ATTENTION: "needs_attention",
    COMPLETE: "complete",
    FAILED: "failed",
    INCOMPLETE: "incomplete"
  });

  const STAGE = Object.freeze({
    STARTING: "starting",
    SEARCH_LOADING: "search_loading",
    COLLECTING: "collecting",
    RETRYING_FAILURES: "retrying_failures",
    OPENING_LIBRARY: "opening_library",
    OPENING_LABEL: "opening_label",
    VERIFYING_LABEL: "verifying_label",
    INDEXING_LIBRARY_IDS: "indexing_library_ids",
    REPAIRING_LABEL: "repairing_label",
    LABEL_LOADING: "label_loading",
    EXPORT_MENU: "export_menu",
    EXPORTING: "exporting",
    DONE: "done"
  });

  function normalizeWhitespace(value) {
    return String(value == null ? "" : value).replace(/\u00a0/g, " ").replace(/\s+/g, " ").trim();
  }

  function normalizeText(value) {
    return normalizeWhitespace(value).toLowerCase();
  }

  function parseReportedTotal(text) {
    const source = normalizeWhitespace(text);
    if (!source) return null;
    const match = source.match(/(?:about\s+)?([0-9][0-9\s,.'’\u202f\u00a0]*)\s+results?\b/i);
    if (!match) return null;
    const digits = match[1].replace(/\D/g, "");
    if (!digits) return null;
    const value = Number(digits);
    return Number.isSafeInteger(value) && value >= 0 ? value : null;
  }

  function targetForTotal(reportedTotal, cap = RESULT_CAP) {
    if (!Number.isInteger(cap) || cap < 1) throw new Error("cap must be a positive integer");
    if (reportedTotal == null) return cap;
    if (!Number.isInteger(reportedTotal) || reportedTotal < 0) throw new Error("reportedTotal must be null or a non-negative integer");
    return Math.min(reportedTotal, cap);
  }



  function provisionalTarget(existingTarget, latestReportedTotal, cap = RESULT_CAP) {
    const observed = targetForTotal(latestReportedTotal, cap);
    if (!Number.isInteger(existingTarget) || existingTarget < 0) return observed;
    return Math.min(cap, Math.max(existingTarget, observed));
  }

  function finalAccessibleTarget(discoveredCount, cap = RESULT_CAP) {
    if (!Number.isInteger(cap) || cap < 1) throw new Error("cap must be a positive integer");
    if (!Number.isInteger(discoveredCount) || discoveredCount < 0) throw new Error("discoveredCount must be a non-negative integer");
    return Math.min(discoveredCount, cap);
  }

  function confirmedEndOfResults({currentStart = 0, rowCount = 0, latestReportedTotal = null, discoveredCount = 0, nextStart = null, cap = RESULT_CAP, pageSize = 10} = {}) {
    const current = Number.isInteger(currentStart) && currentStart >= 0 ? currentStart : 0;
    const rows = Number.isInteger(rowCount) && rowCount >= 0 ? rowCount : 0;
    const discovered = Number.isInteger(discoveredCount) && discoveredCount >= 0 ? discoveredCount : 0;
    const forwardMissingOrStale = nextStart == null || !Number.isInteger(nextStart) || nextStart <= current;
    if (!forwardMissingOrStale) return false;
    if (discovered >= cap) return true;
    if (rows > 0 && rows < pageSize) return true;
    if (Number.isInteger(latestReportedTotal) && latestReportedTotal >= 0) {
      const boundedLatest = Math.min(latestReportedTotal, cap);
      if (current + rows >= boundedLatest) return true;
    }
    return false;
  }

  function throttleCooldownMs(level) {
    const sequence = [60000, 120000, 300000, 600000, 900000, 1800000];
    const index = Math.max(0, Math.min(sequence.length - 1, Number(level || 1) - 1));
    return sequence[index];
  }

  function extractStartOffset(url) {
    try {
      const value = new URL(url).searchParams.get("start");
      const parsed = value == null ? 0 : Number(value);
      return Number.isInteger(parsed) && parsed >= 0 ? parsed : 0;
    } catch (_error) {
      return 0;
    }
  }

  function buildSearchUrl(query, start = 0) {
    const url = new URL("https://scholar.google.com/scholar");
    url.searchParams.set("q", normalizeWhitespace(query));
    url.searchParams.set("hl", "en");
    if (Number.isInteger(start) && start > 0) url.searchParams.set("start", String(start));
    return url.href;
  }

  function isScholarSearchUrl(url) {
    try {
      const parsed = new URL(url);
      return parsed.hostname === "scholar.google.com" && parsed.pathname === "/scholar" && parsed.searchParams.has("q");
    } catch (_error) {
      return false;
    }
  }

  function isScholarLibraryUrl(url) {
    try {
      const parsed = new URL(url);
      return parsed.hostname === "scholar.google.com" && (
        (parsed.pathname === "/scholar" && parsed.searchParams.get("scilib") != null) ||
        (parsed.pathname === "/citations" && parsed.searchParams.get("view_op") === "search_library")
      );
    } catch (_error) {
      return false;
    }
  }

  function looksLikeBlocker(url, bodyText) {
    const text = normalizeText(bodyText);
    const path = (() => {
      try { return new URL(url).pathname.toLowerCase(); } catch (_error) { return ""; }
    })();
    if (path.includes("/sorry/")) return true;
    return [
      "unusual traffic",
      "automated queries",
      "verify you are human",
      "not a robot",
      "our systems have detected",
      "please show you're not a robot",
      "please show you are not a robot"
    ].some(token => text.includes(token));
  }


  function classifyScholarOperationFailure(text) {
    const value = normalizeText(text);
    if (!value) return null;
    const temporary = [
      "the system can't perform the operation now",
      "the system cannot perform the operation now",
      "try again later",
      "please try again later",
      "temporarily unable to perform"
    ];
    return temporary.some(token => value.includes(token)) ? "temporary_operation_failure" : null;
  }

  function unresolvedIds(run, options = {}) {
    const unresolved = run && run.unresolved && typeof run.unresolved === "object" ? run.unresolved : {};
    const includeFinalized = options.includeFinalized !== false;
    return Object.keys(unresolved).filter(cid => includeFinalized || !unresolved[cid]?.finalized);
  }

  function recordUnresolved(unresolved, cid, details = {}) {
    const previous = unresolved && unresolved[cid] && typeof unresolved[cid] === "object" ? unresolved[cid] : {};
    return {
      ...(unresolved && typeof unresolved === "object" ? unresolved : {}),
      [cid]: {
        ...previous,
        ...details,
        failures: Number(previous.failures || 0) + 1,
        firstFailedAt: previous.firstFailedAt || details.at || new Date().toISOString(),
        lastFailedAt: details.at || new Date().toISOString()
      }
    };
  }

  function validateContext(value) {
    if (!value || typeof value !== "object") return false;
    if (value.schema_version !== SCHEMA_VERSION) return false;
    if (!normalizeWhitespace(value.query)) return false;
    if (!normalizeWhitespace(value.research_question)) return false;
    if (!normalizeWhitespace(value.active_query_version)) return false;
    const fp = normalizeWhitespace(value.query_fingerprint);
    return /^[a-f0-9]{64}$/i.test(fp);
  }

  function makeLabelName(fingerprint, now = new Date()) {
    const fp = normalizeWhitespace(fingerprint).replace(/[^a-f0-9]/gi, "").toLowerCase().slice(0, 8) || "query";
    const y = now.getUTCFullYear();
    const m = String(now.getUTCMonth() + 1).padStart(2, "0");
    const d = String(now.getUTCDate()).padStart(2, "0");
    return `litsync_${y}${m}${d}_${fp}`;
  }

  function makeRunLabelName(fingerprint, now = new Date()) {
    // A label identifies ONE collection run, not merely one query/day. Keep
    // it short enough that Scholar's My Library sidebar does not replace the
    // tail with an ellipsis; the sidebar text is also our safest navigation
    // handle during the final audit/export flow. Milliseconds since UTC
    // midnight encoded in base36 stay unique for normal human-started runs.
    const fp = normalizeWhitespace(fingerprint).replace(/[^a-f0-9]/gi, "").toLowerCase().slice(0, 4) || "qry";
    const y = String(now.getUTCFullYear()).slice(-2);
    const m = String(now.getUTCMonth() + 1).padStart(2, "0");
    const d = String(now.getUTCDate()).padStart(2, "0");
    const dayMs = now.getUTCHours() * 3600000 + now.getUTCMinutes() * 60000 + now.getUTCSeconds() * 1000 + now.getUTCMilliseconds();
    const tick = dayMs.toString(36).padStart(6, "0");
    return `litsync_${y}${m}${d}_${fp}_${tick}`;
  }

  function uniqueStrings(values) {
    const seen = new Set();
    const output = [];
    for (const raw of Array.isArray(values) ? values : []) {
      const value = String(raw == null ? "" : raw).trim();
      if (!value || seen.has(value)) continue;
      seen.add(value);
      output.push(value);
    }
    return output;
  }

  function appendUnique(values, value) {
    return uniqueStrings([...(Array.isArray(values) ? values : []), value]);
  }

  function differenceStrings(expected, actual) {
    const actualSet = new Set(uniqueStrings(actual));
    return uniqueStrings(expected).filter(value => !actualSet.has(value));
  }

  function sameUniqueSet(left, right) {
    const a = uniqueStrings(left);
    const b = uniqueStrings(right);
    if (a.length !== b.length) return false;
    const bSet = new Set(b);
    return a.every(value => bSet.has(value));
  }

  function expectedLibraryIds(run) {
    if (!run || !run.libraryIdsBySearchId || typeof run.libraryIdsBySearchId !== "object") return [];
    return uniqueStrings(run.savedIds || [])
      .map(cid => normalizeWhitespace(run.libraryIdsBySearchId[cid]))
      .filter(Boolean);
  }

  function labelIntegritySatisfied(run) {
    if (!run || !Number.isInteger(run.targetTotal) || run.targetTotal < 0) return false;
    const saved = uniqueStrings(run.savedIds);
    const verified = uniqueStrings(run.labelVerifiedIds);
    if (!run.labelAuditComplete || saved.length !== run.targetTotal || unresolvedIds(run).length !== 0) return false;

    if (Number(run.integrityVersion || 0) >= 3) {
      // Search-result data-cid and My Library data-cid are different Scholar
      // namespaces. Search rows expose the corresponding My Library ID in
      // data-lid after the item is saved. Compare those library IDs instead.
      const expected = expectedLibraryIds(run);
      return expected.length === run.targetTotal
        && verified.length === run.targetTotal
        && sameUniqueSet(expected, verified);
    }

    // Legacy compatibility only; full completion additionally requires v3.
    return verified.length === run.targetTotal && sameUniqueSet(saved, verified);
  }

  function simpleHashStrings(values) {
    let h = 2166136261;
    for (const value of uniqueStrings(values)) {
      const text = `${value}|`;
      for (let i = 0; i < text.length; i += 1) {
        h ^= text.charCodeAt(i);
        h = Math.imul(h, 16777619);
      }
    }
    return (h >>> 0).toString(16).padStart(8, "0");
  }

  function pageIdentity(url, cids) {
    return `${extractStartOffset(url)}:${simpleHashStrings(cids)}`;
  }

  function createRun(context, options = {}) {
    if (!validateContext(context)) throw new Error("Invalid Google Scholar query context");
    const now = new Date(options.now || Date.now());
    const runId = options.runId || `scholar-${now.getTime()}-${context.query_fingerprint.slice(0, 8)}`;
    return {
      schemaVersion: SCHEMA_VERSION,
      runId,
      query: normalizeWhitespace(context.query),
      queryFingerprint: context.query_fingerprint,
      researchQuestion: normalizeWhitespace(context.research_question),
      queryVersion: normalizeWhitespace(context.active_query_version),
      label: makeRunLabelName(context.query_fingerprint, now),
      status: STATUS.RUNNING,
      stage: STAGE.STARTING,
      tabId: null,
      reportedTotal: null,
      initialReportedTotal: null,
      latestReportedTotal: null,
      accessibleTotal: null,
      targetTotal: null,
      savedIds: [],
      discoveredIds: [],
      pageIdentities: [],
      pagesVisited: [],
      resultLocations: {},
      libraryIdsBySearchId: {},
      unresolved: {},
      labelUrl: "",
      labelAuditIds: [],
      labelAuditPages: [],
      labelAuditComplete: false,
      labelVerifiedIds: [],
      labelVerifiedCount: null,
      labelMissingIds: [],
      labelUnexpectedIds: [],
      labelRepairIds: [],
      labelRepairAttemptedIds: [],
      labelRepairRound: 0,
      integrityVersion: 3,
      consecutiveFailures: 0,
      cooldownCount: 0,
      throttleLevel: 0,
      throttleEvents: 0,
      throttleUntil: null,
      retryStartedAt: null,
      duplicateCount: 0,
      exportStartedAt: null,
      exportDownloadId: null,
      exportDownloadUrl: "",
      downloadedFilename: "",
      exportCsvVerified: false,
      exportCsvRowCount: null,
      exportCsvExpectedRows: null,
      exportCsvHeader: [],
      exportCsvVerificationError: "",
      exportMode: "full",
      retryRound: 0,
      lastUrl: "",
      message: "Opening Google Scholar…",
      createdAt: now.toISOString(),
      updatedAt: now.toISOString()
    };
  }

  function progress(run) {
    const saved = uniqueStrings(run && run.savedIds).length;
    const target = Number.isInteger(run && run.targetTotal) ? run.targetTotal : RESULT_CAP;
    return {
      saved,
      target,
      remaining: Math.max(0, target - saved),
      percent: target === 0 ? 100 : Math.min(100, Math.round((saved / target) * 100))
    };
  }

  function parseCsvRecords(text) {
    const source = String(text == null ? "" : text).replace(/^\uFEFF/, "");
    const rows = [];
    let row = [];
    let field = "";
    let inQuotes = false;

    for (let i = 0; i < source.length; i += 1) {
      const ch = source[i];
      if (inQuotes) {
        if (ch === '"') {
          if (source[i + 1] === '"') {
            field += '"';
            i += 1;
          } else {
            inQuotes = false;
          }
        } else {
          field += ch;
        }
        continue;
      }

      if (ch === '"' && field.length === 0) {
        inQuotes = true;
      } else if (ch === ',') {
        row.push(field);
        field = "";
      } else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && source[i + 1] === '\n') i += 1;
        row.push(field);
        field = "";
        if (row.some(value => String(value).trim() !== "")) rows.push(row);
        row = [];
      } else {
        field += ch;
      }
    }

    if (inQuotes) throw new Error("CSV ended inside a quoted field");
    if (field.length || row.length) {
      row.push(field);
      if (row.some(value => String(value).trim() !== "")) rows.push(row);
    }
    return rows;
  }

  function inspectScholarCsv(text, expectedRows = null) {
    const source = String(text == null ? "" : text).replace(/^\uFEFF/, "");
    const trimmed = source.trim();
    const result = {
      ok: false,
      likelyCsv: false,
      bibtexLike: /^@[a-z][a-z0-9_-]*\s*\{/i.test(trimmed),
      rowCount: 0,
      expectedRows: Number.isInteger(expectedRows) && expectedRows >= 0 ? expectedRows : null,
      header: [],
      errors: []
    };
    if (!trimmed) {
      result.errors.push("CSV body is empty");
      return result;
    }
    if (result.bibtexLike) {
      result.errors.push("Export body is BibTeX, not CSV");
      return result;
    }

    let rows;
    try {
      rows = parseCsvRecords(source);
    } catch (error) {
      result.errors.push(String(error && error.message || error));
      return result;
    }
    if (!rows.length) {
      result.errors.push("CSV has no rows");
      return result;
    }

    result.header = rows[0].map(value => normalizeWhitespace(value));
    const normalizedHeader = result.header.map(value => normalizeText(value));
    const required = ["authors", "title", "year"];
    const missing = required.filter(name => !normalizedHeader.includes(name));
    if (missing.length) result.errors.push(`CSV header is missing: ${missing.join(", ")}`);

    const dataRows = rows.slice(1).filter(record => record.some(value => normalizeWhitespace(value) !== ""));
    result.rowCount = dataRows.length;
    const headerWidth = result.header.length;
    const malformedCount = headerWidth > 0 ? dataRows.filter(record => record.length !== headerWidth).length : dataRows.length;
    if (malformedCount) result.errors.push(`${malformedCount} CSV row(s) do not match the header column count`);

    result.likelyCsv = missing.length === 0 && headerWidth >= 3 && malformedCount === 0;
    if (result.expectedRows != null && result.rowCount !== result.expectedRows) {
      result.errors.push(`CSV row count ${result.rowCount} does not match expected ${result.expectedRows}`);
    }
    result.ok = result.likelyCsv && result.errors.length === 0;
    return result;
  }

  function expectedExportRows(run) {
    if (!run) return null;
    if (run.exportMode === "partial") return uniqueStrings(run.labelVerifiedIds || []).length;
    return Number.isInteger(run.targetTotal) && run.targetTotal >= 0 ? run.targetTotal : null;
  }

  function fullExportIntegritySatisfied(run) {
    if (!run || run.exportMode === "partial") return false;
    const expected = expectedExportRows(run);
    if (expected === 0) {
      return uniqueStrings(run.savedIds || []).length === 0 && unresolvedIds(run).length === 0;
    }
    return Number(run.integrityVersion || 0) >= 3
      && labelIntegritySatisfied(run)
      && run.exportCsvVerified === true
      && Number.isInteger(expected)
      && run.exportCsvRowCount === expected;
  }

  function sanitizeFilename(value) {
    const cleaned = normalizeWhitespace(value).replace(/[<>:"/\\|?*\x00-\x1F]/g, "_").replace(/\.+$/g, "");
    return cleaned || "citations.csv";
  }

  return Object.freeze({
    SCHEMA_VERSION,
    RESULT_CAP,
    STORAGE_KEYS,
    STATUS,
    STAGE,
    normalizeWhitespace,
    normalizeText,
    parseReportedTotal,
    targetForTotal,
    provisionalTarget,
    finalAccessibleTarget,
    confirmedEndOfResults,
    throttleCooldownMs,
    extractStartOffset,
    buildSearchUrl,
    isScholarSearchUrl,
    isScholarLibraryUrl,
    looksLikeBlocker,
    classifyScholarOperationFailure,
    unresolvedIds,
    recordUnresolved,
    validateContext,
    makeLabelName,
    makeRunLabelName,
    uniqueStrings,
    appendUnique,
    differenceStrings,
    sameUniqueSet,
    expectedLibraryIds,
    labelIntegritySatisfied,
    simpleHashStrings,
    pageIdentity,
    createRun,
    progress,
    parseCsvRecords,
    inspectScholarCsv,
    expectedExportRows,
    fullExportIntegritySatisfied,
    sanitizeFilename
  });
});
