(() => {
  "use strict";
  const KEY = "litsync_ieee_native_v1";
  const LIMIT = 1000;
  const COMMAND = "https://ieeexplore.ieee.org/search/advanced/command";
  const versions = ["balanced", "high_recall"];
  function context(data) {
    if (data?.schema_version !== 1) throw Error("Unsupported query context");
    const queries = {}, fingerprints = {};
    for (const version of versions) {
      const query = data.query_versions?.[version]?.ieee_xplore;
      if (typeof query === "string" && query.trim()) {
        queries[version] = query; // Never trim or rewrite the supplied query.
        fingerprints[version] = data.query_fingerprints?.[version] || "";
      }
    }
    if (!Object.keys(queries).length) throw Error("No IEEE queries supplied");
    return {queries, fingerprints, receivedAt: Date.now()};
  }
  function range(text) {
    const m = String(text).match(/Showing\s+([\d,]+)\s*[-–]\s*([\d,]+)\s+of\s+([\d,]+)\s+results/i);
    if (!m) {
      const single = String(text).match(/Showing\s+1\s+of\s+1\s+result\b/i);
      return single ? {first: 1, last: 1, total: 1} : null;
    }
    const [first, last, total] = m.slice(1).map(s => Number(s.replaceAll(",", "")));
    return first > 0 && last >= first && total >= last ? {first, last, total} : null;
  }
  function isIEEE(url) { try { return new URL(url).origin === "https://ieeexplore.ieee.org"; } catch { return false; } }
  function local(url) { try { const u = new URL(url); return u.protocol === "http:" && ["localhost", "127.0.0.1"].includes(u.hostname); } catch { return false; } }
  function sameQuery(url, query) {
    try {
      const u = new URL(url), actual = u.searchParams.get("queryText");
      // Command Search itself adds a single outer pair (observed live). Input is unchanged.
      return isIEEE(url) && u.pathname === "/search/searchresult.jsp" && (actual === query || actual === `(${query})`);
    } catch { return false; }
  }
  function nativeMode(text, total = LIMIT) {
    const offer = text.match(/If no results are selected,\s*up to\s*([\d,]+) results? will be included\./i);
    return /Download Results/i.test(text) && /CSV file/i.test(text) && offer && Number(offer[1].replaceAll(",", "")) === Math.min(total, LIMIT)
      && !/You have selected\s+[\d,]+\s+results/i.test(text);
  }
  function csv(item) { return /\.csv$/i.test(item.filename || "") || /^(text\/csv|application\/csv)(;|$)/i.test(item.mime || ""); }
  function downloadMatches(item, attempt) {
    const time = Date.parse(item.startTime);
    const origin = [item.url, item.finalUrl, item.referrer].some(url => isIEEE(String(url || "").replace(/^blob:/, "")));
    return origin && time >= attempt.at - 1000 && time <= attempt.at + 120000;
  }
  function newJob(ctx, version, fingerprint, tabId) {
    if (!versions.includes(version) || !ctx?.queries[version]) throw Error("Selected query is not synced");
    return {id: crypto.randomUUID(), query: ctx.queries[version], version, fingerprint,
      tabId, stage: "submitting_query", active: true, createdAt: Date.now(), updatedAt: Date.now(),
      resultsCount: null, resultsUrl: null, sort: null, attempt: null, owner: null};
  }
  globalThis.LitSyncIEEE = {KEY, LIMIT, COMMAND, versions, context, range, isIEEE, local, sameQuery, nativeMode, csv, downloadMatches, newJob};
})();
