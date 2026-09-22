(() => {
  "use strict";
  if (globalThis.__lsIEEEBridge) return;
  globalThis.__lsIEEEBridge = true;
  const announce = () => window.postMessage({type: "LITSYNC_EXTENSION_READY", schema_version: 1, collector: "ieee_native"}, location.origin);
  window.addEventListener("message", event => {
    if (event.source !== window || event.origin !== location.origin || !LitSyncIEEE.local(event.origin)) return;
    const data = event.data;
    if (data?.type === "LITSYNC_EXTENSION_PROBE") announce();
    if (data?.type === "LITSYNC_QUERY_CONTEXT") {
      try { LitSyncIEEE.context(data); chrome.runtime.sendMessage({type: "SYNC", data}).catch(() => {}); } catch { /* unrelated/invalid context */ }
    }
    if (data?.type === "LITSYNC_QUERY_CONTEXT_CLEAR") chrome.runtime.sendMessage({type: "CLEAR"}).catch(() => {});
  });
  document.addEventListener("DOMContentLoaded", announce, {once: true});
  announce();
})();
