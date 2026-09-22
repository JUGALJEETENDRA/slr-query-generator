(() => {
  "use strict";
  const KEY = "litsync_pubmed_native_v1", HOME = "https://pubmed.ncbi.nlm.nih.gov/";
  const versions = ["balanced", "high_recall"];
  function context(data) {
    if (data?.schema_version !== 1) throw Error("Unsupported query context");
    const queries = {}, fingerprints = {};
    for (const version of versions) {
      const query = data.query_versions?.[version]?.pubmed;
      if (typeof query === "string" && query.trim()) {
        queries[version] = query;
        fingerprints[version] = data.query_fingerprints?.[version] || "";
      }
    }
    if (!Object.keys(queries).length) throw Error("No PubMed queries supplied");
    return {queries, fingerprints, receivedAt: Date.now()};
  }
  function resultCount(text) {
    const m = String(text).trim().match(/^([\d,]+)\s+results?$/i);
    if (!m) return null;
    const n = Number(m[1].replaceAll(",", ""));
    return Number.isSafeInteger(n) && n >= 0 ? n : null;
  }
  function isPubMed(url) { try { return new URL(url).origin === new URL(HOME).origin; } catch { return false; } }
  function searchSurface(url) { try { return isPubMed(url) && new URL(url).pathname === "/"; } catch { return false; } }
  function local(url) { try { const u = new URL(url); return u.protocol === "http:" && ["localhost", "127.0.0.1"].includes(u.hostname); } catch { return false; } }
  function sameQuery(url, query) {
    try { return searchSurface(url) && new URL(url).searchParams.get("term") === query; } catch { return false; }
  }
  function csv(item) { return /\.csv$/i.test(item.filename || "") || /^(text\/csv|application\/csv)(;|$)/i.test(item.mime || ""); }
  function downloadMatches(item, attempt) {
    const time = Date.parse(item.startTime);
    const origin = [item.url, item.finalUrl, item.referrer].some(url => isPubMed(String(url || "").replace(/^blob:/, "")));
    return origin && time >= attempt.at - 1000 && time <= attempt.at + 120000;
  }
  function newJob(ctx, version, fingerprint, tabId) {
    if (!versions.includes(version) || !ctx?.queries[version]) throw Error("Selected query is not synced");
    return {id: crypto.randomUUID(), query: ctx.queries[version], version, fingerprint, tabId,
      stage: "submitting_query", active: true, createdAt: Date.now(), updatedAt: Date.now(),
      resultsCount: null, resultsUrl: null, sort: null, attempt: null, owner: null};
  }
  globalThis.LitSyncPubMed = {KEY, HOME, versions, context, resultCount, isPubMed, searchSurface, local, sameQuery, csv, downloadMatches, newJob};
})();
