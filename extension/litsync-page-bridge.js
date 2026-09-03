(() => {
  const schemaVersion = 1;
  const extensionIsActive = () => Boolean(chrome.runtime?.id);

  function storeContext(context) {
    if (!extensionIsActive()) return;
    chrome.runtime.sendMessage({ type: 'LITSYNC_STORE_QUERY_CONTEXT', context });
  }

  function announceReady() {
    window.postMessage({ type: 'LITSYNC_EXTENSION_READY', schema_version: schemaVersion }, window.location.origin);
  }

  window.addEventListener('message', (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = event.data;

    if (data?.type === 'LITSYNC_EXTENSION_PROBE' && data.schema_version === schemaVersion) {
      announceReady();
      return;
    }

    if (data?.type === 'LITSYNC_QUERY_CONTEXT_CLEAR' && data.schema_version === schemaVersion) {
      storeContext(null);
      return;
    }

    if (data?.type !== 'LITSYNC_QUERY_CONTEXT' || data.schema_version !== schemaVersion) return;
    if (typeof data.research_question !== 'string' || typeof data.active_query_version !== 'string') return;
    if (typeof data.query_fingerprint !== 'string' || typeof data.queries?.scopus !== 'string') return;

    storeContext({
        researchQuestion: data.research_question,
        queryVersion: data.active_query_version,
        queryFingerprint: data.query_fingerprint,
        scopusQuery: data.queries.scopus,
        highRecallScopusQuery: data.high_recall_scopus_query || '',
        highRecallQueryFingerprint: data.high_recall_query_fingerprint || '',
        webOfScienceQuery: data.queries.web_of_science || '',
        receivedAt: new Date().toISOString()
    });
  });
})();
