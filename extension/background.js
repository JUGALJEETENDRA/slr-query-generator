const SCOPUS_ADVANCED_SEARCH_URL = 'https://www.scopus.com/search/form.uri?display=advanced';
const WEB_OF_SCIENCE_ADVANCED_SEARCH_URL = 'https://www.webofscience.com/wos/woscc/advanced-search';
const WEB_OF_SCIENCE_HOSTS = ['https://www.webofscience.com/', 'https://webofscience.com/'];

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === 'LITSYNC_STORE_QUERY_CONTEXT' || message?.type === 'LITSYNC_STORE_SCOPUS_QUERY') {
    chrome.storage.local.set({
      litsyncQueryContext: message.context,
      // Kept for the existing Scopus content script and users updating in place.
      litsyncScopusContext: message.context
    }).then(() => {
      sendResponse({ ok: true });
    }).catch((error) => {
      sendResponse({ ok: false, error: error.message });
    });
    return true;
  }

  if (message?.type === 'LITSYNC_OPEN_WEB_OF_SCIENCE') {
    chrome.storage.local.set({ litsyncWebOfScienceAutomation: { state: 'opening', message: 'Opening Web of Science Advanced Search.' } }).then(() =>
      chrome.tabs.create({ url: WEB_OF_SCIENCE_ADVANCED_SEARCH_URL })
    ).then(() => sendResponse({ ok: true })).catch((error) => sendResponse({ ok: false, error: error.message }));
    return true;
  }

  if (message?.type === 'LITSYNC_OPEN_SCOPUS') {
    chrome.storage.local.set({ litsyncScopusAutomation: { state: 'opening', message: 'Opening Scopus Advanced Search.' } }).then(() =>
      chrome.tabs.create({ url: SCOPUS_ADVANCED_SEARCH_URL })
    ).then(() => {
      sendResponse({ ok: true });
    }).catch((error) => {
      sendResponse({ ok: false, error: error.message });
    });
    return true;
  }

  if (message?.type === 'LITSYNC_START_SCOPUS_AUTOMATION') {
    chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
      if (!tab?.id || !tab.url?.startsWith('https://www.scopus.com/')) {
        throw new Error('Open the Scopus Advanced Search tab before starting the workflow.');
      }
      return chrome.tabs.sendMessage(tab.id, { type: 'LITSYNC_START_SCOPUS_AUTOMATION' });
    }).then(() => {
      sendResponse({ ok: true });
    }).catch((error) => {
      sendResponse({ ok: false, error: error.message });
    });
    return true;
  }

  if (message?.type === 'LITSYNC_START_WEB_OF_SCIENCE_AUTOMATION') {
    chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
      if (!tab?.id || !WEB_OF_SCIENCE_HOSTS.some((host) => tab.url?.startsWith(host))) {
        throw new Error('Open the Web of Science Advanced Search tab before starting the workflow.');
      }
      return chrome.tabs.sendMessage(tab.id, { type: 'LITSYNC_START_WEB_OF_SCIENCE_AUTOMATION' });
    }).then(() => sendResponse({ ok: true })).catch((error) => sendResponse({ ok: false, error: error.message }));
    return true;
  }
});
