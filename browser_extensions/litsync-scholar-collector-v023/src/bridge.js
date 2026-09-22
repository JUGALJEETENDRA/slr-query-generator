(() => {
  "use strict";
  const Core = globalThis.LitSyncScholarCore;
  if (!Core) return;

  function allowedOrigin(origin) {
    try {
      const url = new URL(origin);
      return url.protocol === "http:" && (url.hostname === "localhost" || url.hostname === "127.0.0.1");
    } catch (_error) {
      return false;
    }
  }

  function announce() {
    window.postMessage({
      type: "LITSYNC_SCHOLAR_EXTENSION_READY",
      schema_version: Core.SCHEMA_VERSION
    }, window.location.origin);
  }

  window.addEventListener("message", async event => {
    if (event.source !== window || event.origin !== window.location.origin || !allowedOrigin(event.origin)) return;
    const data = event.data || {};
    if (data.type === "LITSYNC_SCHOLAR_EXTENSION_PROBE") {
      announce();
      return;
    }
    if (data.type === "LITSYNC_SCHOLAR_QUERY_CONTEXT") {
      const context = {
        schema_version: data.schema_version,
        research_question: data.research_question,
        active_query_version: data.active_query_version,
        query_fingerprint: data.query_fingerprint,
        query: data.query
      };
      if (!Core.validateContext(context)) return;
      await chrome.runtime.sendMessage({type: "LSSC_SAVE_CONTEXT", context});
      announce();
    }
    if (data.type === "LITSYNC_QUERY_CONTEXT_CLEAR") {
      await chrome.runtime.sendMessage({type: "LSSC_CLEAR_CONTEXT"});
    }
  });

  document.addEventListener("DOMContentLoaded", announce, {once: true});
  announce();
})();
