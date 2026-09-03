(() => {
  const ADVANCED_SEARCH_PATH = '/wos/woscc/advanced-search';
  const SEARCHED_KEY = 'litsyncWebOfScienceSearchedFingerprint';

  const active = () => Boolean(chrome.runtime?.id);
  const visible = (element) => Boolean(element && element.getClientRects().length);
  const text = (element) => (element?.innerText || element?.textContent || '').replace(/\s+/g, ' ').trim();

  function update(state, message) {
    if (!active()) return Promise.resolve();
    return chrome.storage.local.set({
      litsyncWebOfScienceAutomation: { state, message, updatedAt: new Date().toISOString() }
    });
  }

  // Angular-based versions of Web of Science use native inputs, while some
  // institutional variants use a contenteditable search field.
  function setFieldValue(field, value) {
    if (field.isContentEditable || (!(field instanceof HTMLInputElement) && !(field instanceof HTMLTextAreaElement))) {
      field.focus();
      field.textContent = value;
      field.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: value }));
      field.dispatchEvent(new Event('change', { bubbles: true }));
      return;
    }
    const prototype = field instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set;
    if (setter) setter.call(field, value);
    else field.value = value;
    field.dispatchEvent(new Event('input', { bubbles: true }));
    field.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function queryField() {
    const selectors = [
      'textarea[aria-label*="query" i]', 'textarea[placeholder*="query" i]',
      'input[aria-label*="query" i]', 'input[placeholder*="query" i]',
      'textarea', 'input[type="text"]', 'input:not([type])',
      '[contenteditable="true"]', '[role="textbox"]'
    ];
    for (const selector of selectors) {
      const candidate = [...document.querySelectorAll(selector)].find((element) => {
        if (!visible(element) || element.disabled || element.readOnly) return false;
        const box = element.getBoundingClientRect();
        return box.width > 250 && box.height > 20;
      });
      if (candidate) return candidate;
    }
    return null;
  }

  function searchButton() {
    return [...document.querySelectorAll('button, input[type="submit"], [role="button"]')].find((element) =>
      visible(element) && !element.disabled && /^(search|run search)$/i.test(text(element) || element.value || '')
    );
  }

  async function runSearch(context) {
    const field = queryField();
    if (!field) {
      await update('waiting-for-query-field', 'Waiting for Web of Science Advanced Search to finish loading.');
      return false;
    }
    setFieldValue(field, context.webOfScienceQuery);
    const button = searchButton();
    if (!button) {
      await update('waiting-for-search-button', 'Query entered; waiting for the Web of Science Search button.');
      return false;
    }
    sessionStorage.setItem(SEARCHED_KEY, context.queryFingerprint);
    await update('searching', 'Query entered. Running Web of Science search.');
    button.click();
    return true;
  }

  async function automate(manualStart = false) {
    if (!active()) return;
    const { litsyncQueryContext, litsyncScopusContext } = await chrome.storage.local.get([
      'litsyncQueryContext', 'litsyncScopusContext'
    ]);
    const context = litsyncQueryContext || litsyncScopusContext;
    if (!context?.webOfScienceQuery || !context?.queryFingerprint) {
      await update('waiting-for-query', 'Open LitSync and generate a Web of Science query first.');
      return { ok: false, error: 'Generate a Web of Science query in LitSync first.' };
    }
    if (location.pathname === ADVANCED_SEARCH_PATH) {
      if (manualStart && sessionStorage.getItem(SEARCHED_KEY) !== context.queryFingerprint) {
        const started = await runSearch(context);
        return started
          ? { ok: true }
          : { ok: false, error: 'Web of Science search controls are still loading. Try again in a moment.' };
      }
      return { ok: true };
    }
    if (sessionStorage.getItem(SEARCHED_KEY) === context.queryFingerprint) {
      await update('search-complete', 'Web of Science results are ready. Use Export in Web of Science to download your selected records.');
    }
    return { ok: true };
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type !== 'LITSYNC_START_WEB_OF_SCIENCE_AUTOMATION') return;
    automate(true).then(sendResponse).catch((error) => sendResponse({ ok: false, error: error.message }));
    return true;
  });

  chrome.storage.local.get(['litsyncQueryContext', 'litsyncScopusContext']).then(({ litsyncQueryContext, litsyncScopusContext }) => {
    const context = litsyncQueryContext || litsyncScopusContext;
    if (context?.queryFingerprint && sessionStorage.getItem(SEARCHED_KEY) === context.queryFingerprint) automate();
  });
})();
