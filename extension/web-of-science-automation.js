(() => {
  const ADVANCED_SEARCH_PATH = '/wos/woscc/advanced-search';
  const SEARCHED_KEY = 'litsyncWebOfScienceSearchedFingerprint';
  const EXPORT_ATTEMPTED_KEY = 'litsyncWebOfScienceExportAttemptedFingerprint';
  const EXPORT_CLICKED_KEY = 'litsyncWebOfScienceExportClickedFingerprint';
  let exportAttempts = 0;

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

  function findText(pattern, selector = 'button, a, [role="button"], label, option, span, div') {
    return [...document.querySelectorAll(selector)].find((element) => visible(element) && pattern.test(text(element)));
  }

  function exportControl() {
    return findText(/^export$/i)?.closest('button, a, [role="button"]')
      || [...document.querySelectorAll('button, a, [role="button"]')].find((element) =>
        visible(element) && /export/i.test(text(element) || element.getAttribute('aria-label') || element.getAttribute('title') || ''));
  }

  function exportDialog() {
    const heading = findText(/^export records to excel$/i, 'h1, h2, h3, [role="heading"], div');
    for (let container = heading; container; container = container.parentElement) {
      if ([...container.querySelectorAll('button, [role="button"]')].some((element) => /^export$/i.test(text(element)))) return container;
    }
    return null;
  }

  function setRecordRange(dialog) {
    const inputs = [...dialog.querySelectorAll('input')].filter((element) => visible(element) && !element.disabled);
    const from = inputs.find((element) => /from/i.test(element.getAttribute('aria-label') || element.placeholder || '')) || inputs.find((element) => element.value === '1');
    const to = inputs.find((element) => /to/i.test(element.getAttribute('aria-label') || element.placeholder || '')) || inputs.find((element) => element !== from && /^\d[\d,]*$/.test(element.value));
    if (!from || !to) return false;
    setFieldValue(from, '1');
    const availableText = text(dialog).match(/\b(\d[\d,]*)\s*$/);
    const available = Number.parseInt((to.value || availableText?.[1] || '1000').replaceAll(',', ''), 10);
    setFieldValue(to, String(Math.min(1000, Number.isFinite(available) ? available : 1000)));
    const rangeLabel = findText(/^records from:/i, 'label, span, div');
    const rangeRadio = rangeLabel?.closest('label, div')?.querySelector('input[type="radio"]');
    if (rangeRadio && !rangeRadio.checked) (rangeLabel.closest('label') || rangeLabel).click();
    return true;
  }

  function selectFullRecord(dialog) {
    const select = [...dialog.querySelectorAll('select')].find((element) => visible(element) && !element.disabled);
    if (select) {
      const option = [...select.options].find((item) => /^full record$/i.test(item.textContent.trim()));
      if (option) {
        select.value = option.value;
        select.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
      }
    }
    const fullRecord = findText(/^full record$/i, 'button, [role="option"], option, li, span, div');
    if (fullRecord) {
      fullRecord.click();
      return true;
    }
    const contentControl = findText(/^record content:?/i, 'label, span, div')?.parentElement?.querySelector('button, [role="button"]');
    contentControl?.click();
    return Boolean(contentControl);
  }

  async function confirmExcelExport(context, dialog) {
    if (exportAttempts >= 3) return false;
    const button = [...dialog.querySelectorAll('button, [role="button"], input[type="submit"]')]
      .find((element) => visible(element) && !element.disabled && /^export$/i.test(text(element) || element.value || ''));
    if (!button) return false;
    exportAttempts += 1;
    sessionStorage.setItem(EXPORT_CLICKED_KEY, context.queryFingerprint);
    button.click();
    await update('export-requested', 'Excel export requested for the first 1,000 Web of Science records. Check Chrome downloads.');
    return true;
  }

  async function exportExcel(context) {
    if (sessionStorage.getItem(EXPORT_CLICKED_KEY) === context.queryFingerprint) return true;
    const dialog = exportDialog();
    if (dialog) {
      setRecordRange(dialog);
      selectFullRecord(dialog);
      if (await confirmExcelExport(context, dialog)) {
        sessionStorage.setItem(EXPORT_ATTEMPTED_KEY, context.queryFingerprint);
        return true;
      }
      return false;
    }
    if (sessionStorage.getItem(EXPORT_ATTEMPTED_KEY) === context.queryFingerprint) return false;
    const button = exportControl();
    if (!button) {
      await update('waiting-for-export', 'Waiting for Web of Science results and the Export control.');
      return false;
    }
    sessionStorage.setItem(EXPORT_ATTEMPTED_KEY, context.queryFingerprint);
    button.click();
    await update('opening-export', 'Opening Web of Science Excel export options.');
    return false;
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
      await exportExcel(context);
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
    if (context?.queryFingerprint && sessionStorage.getItem(SEARCHED_KEY) === context.queryFingerprint) {
      automate();
      window.setTimeout(automate, 1500);
      window.setInterval(automate, 1000);
    }
  });
})();
