(() => {
  "use strict";
  if (globalThis.__lsPubMedBridge) return;
  globalThis.__lsPubMedBridge = true;
  const announce = () => window.postMessage({type: "LITSYNC_EXTENSION_READY", schema_version: 1, collector: "pubmed_native"}, location.origin);
  window.addEventListener("message", event => {
    if (event.source !== window || event.origin !== location.origin || !LitSyncPubMed.local(event.origin)) return;
    const data = event.data;
    if (data?.type === "LITSYNC_EXTENSION_PROBE") announce();
    if (data?.type === "LITSYNC_QUERY_CONTEXT") {
      try { LitSyncPubMed.context(data); chrome.runtime.sendMessage({type: "SYNC", data}).catch(() => {}); } catch { /* unrelated/invalid context */ }
    }
    if (data?.type === "LITSYNC_QUERY_CONTEXT_CLEAR") chrome.runtime.sendMessage({type: "CLEAR"}).catch(() => {});
  });
  document.addEventListener("DOMContentLoaded", announce, {once: true});
  announce();
})();
