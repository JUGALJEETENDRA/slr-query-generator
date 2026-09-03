(() => {
  const ADVANCED_SEARCH = '/search/form.uri';
  const SEARCHED_KEY = 'litsyncScopusSearchedFingerprint';
  const EXPORT_ATTEMPTED_KEY = 'litsyncScopusExportAttemptedFingerprint';
  const FINAL_EXPORT_CLICK_KEY = 'litsyncScopusFinalExportFingerprintV2';
  const PENDING_HIGH_RECALL_KEY = 'litsyncScopusPendingHighRecallFingerprint';
  const CSV_OPTIONS_SETTLE_MS = 2000;
  let finalExportAttempts = 0;

  const visible = (element) => Boolean(element && element.getClientRects().length);
  const text = (element) => (element?.innerText || element?.textContent || '').replace(/\s+/g, ' ').trim();
  const extensionIsActive = () => Boolean(chrome.runtime?.id);

  function deepQueryAll(selector, root = document) {
    const matches = [...root.querySelectorAll(selector)];
    for (const element of root.querySelectorAll('*')) {
      if (element.shadowRoot) matches.push(...deepQueryAll(selector, element.shadowRoot));
    }
    return matches;
  }

  function parentContainer(element) {
    return element?.parentElement || element?.getRootNode?.().host || null;
  }

  function findByText(selector, pattern) {
    return deepQueryAll(selector).find((element) => visible(element) && pattern.test(text(element)));
  }

  function update(state, message) {
    if (!extensionIsActive()) return Promise.resolve();
    return chrome.storage.local.set({ litsyncScopusAutomation: { state, message, updatedAt: new Date().toISOString() } });
  }

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

  function selectExportOption(labelText) {
    const label = findByText('label', new RegExp(`^${labelText}$`, 'i'));
    const container = label?.closest('label, div, li') || label;
    const input = container?.querySelector('input[type="checkbox"], input[type="radio"]');
    if (input?.checked) return true;
    if (label) {
      label.click();
      return true;
    }
    const control = findByText('button, [role="checkbox"], [role="radio"]', new RegExp(`^${labelText}$`, 'i'));
    if (control?.getAttribute('aria-checked') !== 'true') {
      control?.click();
      return Boolean(control);
    }
    return Boolean(control);
  }

  function turnOffTruncation() {
    const label = findByText('label', /^truncate to optimize for excel$/i);
    const container = label?.closest('label, div') || label;
    const input = container?.querySelector('input[type="checkbox"]');
    if (input?.checked) label.click();
    const switchControl = container?.querySelector('[role="switch"][aria-checked="true"]');
    if (switchControl) switchControl.click();
  }

  function selectExportRange() {
    const documents = findByText('label, div, span', /^documents\s+1\s*[–-]/i);
    const radio = documents?.closest('label, div')?.querySelector('input[type="radio"]')
      || documents?.parentElement?.querySelector('input[type="radio"]');
    if (radio && !radio.checked) (documents?.closest('label') || documents)?.click();

    const resultText = [...document.querySelectorAll('body *')]
      .map(text)
      .find((value) => /^\d[\d,]*\s+documents?\s+found$/i.test(value));
    const available = Number.parseInt((resultText?.match(/[\d,]+/)?.[0] || '20000').replaceAll(',', ''), 10);
    const maximum = Math.min(20000, Number.isFinite(available) ? available : 20000);
    const end = [...document.querySelectorAll('input[placeholder="To" i], input[aria-label*="to" i]')]
      .find((field) => visible(field) && !field.disabled);
    if (end) setFieldValue(end, String(maximum));
  }

  function csvDialogContainer() {
    const heading = findByText(
      'h1, h2, h3, [role="heading"], div',
      /^export\s+(?:\d[\d,]*\s+)?documents?\s+to\s+csv$/i
    );
    for (let container = heading; container; container = parentContainer(container)) {
      if (deepQueryAll('button, [role="button"], input[type="submit"]', container)
        .some((element) => /^export$/i.test(text(element) || element.value || ''))) return container;
    }
    return null;
  }

  function finalExportButton() {
    const dialog = csvDialogContainer();
    if (!dialog) return null;
    return deepQueryAll('button, [role="button"], input[type="submit"]', dialog)
      .filter((element) => visible(element) && !element.disabled && /^export$/i.test(text(element) || element.value || ''))
      .sort((a, b) => b.getBoundingClientRect().top - a.getBoundingClientRect().top)[0] || null;
  }

  async function clickFinalExport(context) {
    if (finalExportAttempts >= 3) return false;
    // Scopus finishes rendering its footer immediately after the options are
    // selected; wait briefly, then target Export inside this dialog only.
    await new Promise((resolve) => window.setTimeout(resolve, 250));
    const button = finalExportButton();
    if (!button) return false;
    finalExportAttempts += 1;
    sessionStorage.setItem(FINAL_EXPORT_CLICK_KEY, context.queryFingerprint);
    button.focus();
    button.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, composed: true }));
    button.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, composed: true }));
    button.click();
    button.dispatchEvent(new MouseEvent('click', { bubbles: true, composed: true }));
    await update('export-requested', 'CSV export requested. Check Chrome downloads.');
    window.setTimeout(() => {
      if (csvDialogContainer()) automate();
    }, 1000);
    return true;
  }

  async function configureCsvDialog(context) {
    const heading = findByText(
      '[role="dialog"], div, h1, h2, h3',
      /^export\s+(?:\d[\d,]*\s+)?documents?\s+to\s+csv$/i
    );
    let dialog = heading;
    while (dialog && ![...dialog.querySelectorAll('button, [role="button"]')]
      .some((element) => /^export$/i.test(text(element)))) {
      dialog = dialog.parentElement;
    }
    if (!dialog) return false;

    // LitSync requires title/abstract screening and cross-source deduplication.
    selectExportRange();
    selectExportOption('Citation information');
    selectExportOption('Abstract & keywords');
    selectExportOption('Abstract');
    selectExportOption('Author keywords');
    selectExportOption('Indexed keywords');
    selectExportOption('DOI');
    turnOffTruncation();

    if (!await clickFinalExport(context)) return false;
    sessionStorage.setItem(EXPORT_ATTEMPTED_KEY, context.queryFingerprint);
    return true;
  }

  async function chooseCsvFromMenu() {
    const csv = deepQueryAll('button, a, label, div, li, span, [role="menuitem"], [role="option"]')
      .filter((element) => visible(element) && /^csv$/i.test(text(element)))
      .sort((a, b) => b.getBoundingClientRect().top - a.getBoundingClientRect().top)[0];
    if (!csv) return false;
    const target = csv.closest('button, a, [role="menuitem"], [role="option"], li') || csv;
    target.focus?.();
    target.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, composed: true }));
    target.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, composed: true }));
    target.click();
    await update('opening-csv-options', 'Opening Scopus CSV export options.');
    // Give Scopus time to replace the file-type menu with the CSV-options
    // modal before selecting the required export fields.
    window.setTimeout(automate, CSV_OPTIONS_SETTLE_MS);
    return true;
  }

  function resultsExportControl() {
    const exact = findByText('button, a, [role="button"], [role="menuitem"], span', /^export$/i);
    if (exact) return exact.closest('button, a, [role="button"], [role="menuitem"]') || parentContainer(exact);
    const labelled = deepQueryAll('button, a, [role="button"], [role="menuitem"]')
      .find((element) => visible(element)
        && /export/i.test(text(element) || element.getAttribute('aria-label') || element.getAttribute('title') || ''));
    return labelled || null;
  }

  function queryField() {
    const selectors = [
      'textarea[aria-label*="query" i]',
      'textarea[placeholder*="query" i]',
      'textarea',
      'input[aria-label*="query" i]',
      'input[placeholder*="query" i]',
      'input[type="text"]',
      'input:not([type])',
      '[contenteditable="true"]',
      '[role="textbox"]',
      '[aria-multiline="true"]'
    ];
    const roots = [document];
    for (const element of document.querySelectorAll('*')) {
      if (element.shadowRoot) roots.push(element.shadowRoot);
    }
    for (const root of roots) for (const selector of selectors) {
      const field = [...root.querySelectorAll(selector)].find((candidate) => {
        if (!visible(candidate) || candidate.disabled || candidate.readOnly) return false;
        const rect = candidate.getBoundingClientRect();
        return rect.width > 250 && rect.height > 20;
      });
      if (field) return field;
    }
    return null;
  }

  function noResultsFound() {
    // Scopus has used both phrasings across its result-page variants. Restrict
    // this check to visible page text so hidden templates cannot trigger a retry.
    const pageText = text(document.body);
    return /(?:no\s+(?:documents?|results?)\s+(?:matching\s+your\s+keywords\s+)?(?:were\s+)?found|your\s+search\s+returned\s+no\s+results)/i.test(pageText);
  }

  async function switchToHighRecall(context) {
    if (context.queryVersion !== 'balanced'
      || !context.highRecallScopusQuery
      || context.highRecallScopusQuery === context.scopusQuery
      || sessionStorage.getItem(SEARCHED_KEY) === context.highRecallQueryFingerprint) return false;

    const fallbackFingerprint = context.highRecallQueryFingerprint || `${context.queryFingerprint}:high-recall`;
    await chrome.storage.local.set({
      litsyncScopusContext: {
        ...context,
        queryVersion: 'high_recall',
        queryFingerprint: fallbackFingerprint,
        scopusQuery: context.highRecallScopusQuery,
        switchedFromBalanced: true
      }
    });
    sessionStorage.removeItem(EXPORT_ATTEMPTED_KEY);
    sessionStorage.removeItem(FINAL_EXPORT_CLICK_KEY);
    sessionStorage.setItem(PENDING_HIGH_RECALL_KEY, fallbackFingerprint);
    finalExportAttempts = 0;
    await update('retrying-high-recall', 'Balanced query returned no results. Retrying with the High Recall query.');
    window.location.assign(`${location.origin}${ADVANCED_SEARCH}`);
    return true;
  }

  async function runAdvancedSearch(context) {
    const field = queryField();
    if (!field) {
      await update('waiting-for-query-field', 'Waiting for Scopus Advanced Search to finish loading.');
      return false;
    }
    setFieldValue(field, context.scopusQuery);
    const search = findByText('button, [role="button"]', /^search$/i);
    if (!search) {
      await update('waiting-for-search-button', 'Query entered; waiting for the Scopus Search button.');
      return false;
    }
    sessionStorage.setItem(SEARCHED_KEY, context.queryFingerprint);
    await update('searching', 'Query entered. Running Scopus search.');
    search.click();
    return true;
  }

  async function exportCsv(context) {
    // A CSV dialog may already be open from an earlier click. Configure it
    // even when this query was previously marked as attempted.
    if (await configureCsvDialog(context)) return;
    if (await clickFinalExport(context)) return;
    // Likewise, continue from an already-open file-type menu after a reload.
    if (await chooseCsvFromMenu()) {
      return;
    }
    if (sessionStorage.getItem(EXPORT_ATTEMPTED_KEY) === context.queryFingerprint) return;
    const exportButton = resultsExportControl();
    if (!exportButton) {
      await update('waiting-for-export', 'Waiting for Scopus results and the Export control.');
      return;
    }
    // Mark this run before opening the dialog so DOM updates do not trigger
    // duplicate export dialogs while Scopus is rendering its modal.
    sessionStorage.setItem(EXPORT_ATTEMPTED_KEY, context.queryFingerprint);
    exportButton.focus?.();
    exportButton.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, composed: true }));
    exportButton.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, composed: true }));
    exportButton.click();
    await update('opening-export', 'Opening Scopus export options.');

    const started = Date.now();
    const timer = window.setInterval(async () => {
      if (await configureCsvDialog(context)) {
        window.clearInterval(timer);
        return;
      }
      if (await chooseCsvFromMenu()) {
        // Selecting CSV starts an asynchronous modal transition. Stop this
        // loop so it cannot keep clicking CSV and close/reopen the menu;
        // chooseCsvFromMenu schedules the next automation pass itself.
        window.clearInterval(timer);
        return;
      } else if (Date.now() - started > 30000) {
        window.clearInterval(timer);
        await update('export-needs-confirmation', 'Scopus opened export options. Choose CSV and confirm Export to download.');
      }
    }, 500);
  }

  async function automate(manualStart = false) {
    // Chrome leaves an old content script in already-open tabs after an
    // extension reload. That script no longer has access to chrome.runtime.
    if (!extensionIsActive()) return;
    const { litsyncScopusContext: context } = await chrome.storage.local.get('litsyncScopusContext');
    if (!context?.scopusQuery || !context?.queryFingerprint) {
      await update('waiting-for-query', 'Open LitSync and generate a query first.');
      return;
    }
    if (location.pathname === ADVANCED_SEARCH) {
      if (manualStart && sessionStorage.getItem(SEARCHED_KEY) !== context.queryFingerprint) await runAdvancedSearch(context);
      return;
    }
    if (/\/results\//.test(location.pathname) || /\/search\//.test(location.pathname)) {
      if (noResultsFound() && await switchToHighRecall(context)) return;
      await exportCsv(context);
    }
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type !== 'LITSYNC_START_SCOPUS_AUTOMATION') return;
    automate(true).then(() => sendResponse({ ok: true })).catch((error) => sendResponse({ ok: false, error: error.message }));
    return true;
  });

  // A workflow started by the user continues automatically after Scopus
  // navigates from Advanced Search to its results page.
  chrome.storage.local.get('litsyncScopusContext').then(({ litsyncScopusContext: context }) => {
    if (!context?.queryFingerprint) return;
    if (location.pathname === ADVANCED_SEARCH
      && sessionStorage.getItem(PENDING_HIGH_RECALL_KEY) === context.queryFingerprint) {
      sessionStorage.removeItem(PENDING_HIGH_RECALL_KEY);
      automate(true);
      return;
    }
    if (sessionStorage.getItem(SEARCHED_KEY) === context.queryFingerprint
      && (/\/results\//.test(location.pathname) || /\/search\//.test(location.pathname))) {
      automate();
      // The zero-results panel is sometimes added after document_idle.
      // Recheck once its client-side rendering has settled.
      window.setTimeout(automate, 1500);
    }
  });
})();
