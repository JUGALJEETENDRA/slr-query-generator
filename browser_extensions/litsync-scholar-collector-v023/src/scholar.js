(() => {
  "use strict";
  const Core = globalThis.LitSyncScholarCore;
  if (!Core) return;

  const ACTION_DELAY_MS = 1500;
  const PAGE_DELAY_MS = 2200;
  const WAIT_TIMEOUT_MS = 10000;
  const TRANSIENT_RETRY_DELAY_MS = 12000;
  let driverActive = false;

  const SELECTORS = Object.freeze({
    resultRows: ".gs_r[data-cid]",
    resultTitle: ".gs_rt",
    saveButton: ".gs_or_sav",
    alreadySaved: ".gs_or_sav_add",
    labelButton: ".gs_or_lbl",
    resultsMeta: "#gs_ab_md",
    searchInput: "#gs_hdr_tsi, input[name='q'], input[aria-label='Search']",
    searchButton: "#gs_hdr_tsb, button[aria-label='Search'], button[type='submit']",
    labelDialog: "#gs_md_albl-d",
    labelDialogBody: "#gs_md_albl-d-bdy",
    labelDone: "#gs_lbd_apl",
    labelCreate: "#gs_lbd_new",
    labelNewInput: "#gs_lbd_new-input",
    labelNewCheckbox: "#gs_lbd_new_in .gs_in_cb",
    labelOptions: "#gs_lbl_op .gs_in_cb",
    nextLink: "#gs_n a, a[aria-label='Next'], a[aria-label='Next page']"
  });

  function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

  async function snapshot() {
    const response = await chrome.runtime.sendMessage({type: "LSSC_GET_SNAPSHOT"});
    if (!response || !response.ok) throw new Error(response && response.error || "Cannot read Scholar run state");
    return response;
  }

  async function patch(patchValue) {
    const response = await chrome.runtime.sendMessage({type: "LSSC_PATCH_RUN", patch: patchValue});
    if (!response || !response.ok) throw new Error(response && response.error || "Cannot checkpoint Scholar run");
    return response.run;
  }

  async function log(event, details = {}, level = "info") {
    try { await chrome.runtime.sendMessage({type: "LSSC_LOG", event, details, level}); } catch (_error) {}
  }

  function visible(element) {
    if (!element) return false;
    let cursor = element;
    for (let depth = 0; cursor && depth < 32; depth += 1, cursor = cursor.parentElement) {
      try {
        if (cursor.hidden === true) return false;
        if (cursor.getAttribute && cursor.getAttribute("aria-hidden") === "true") return false;
        const style = getComputedStyle(cursor);
        if (style && (style.display === "none" || style.visibility === "hidden" || style.visibility === "collapse" || style.opacity === "0")) return false;
      } catch (_error) {}
    }
    try {
      if (typeof element.getClientRects === "function" && element.getClientRects().length === 0) return false;
    } catch (_error) {}
    return true;
  }

  async function waitFor(getter, timeout = WAIT_TIMEOUT_MS, interval = 100) {
    const started = Date.now();
    while (Date.now() - started < timeout) {
      const value = getter();
      if (value) return value;
      await sleep(interval);
    }
    return null;
  }

  function setNativeValue(input, value) {
    const descriptor = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(input), "value");
    if (descriptor && descriptor.set) descriptor.set.call(input, value);
    else input.value = value;
    input.dispatchEvent(new Event("input", {bubbles: true}));
    input.dispatchEvent(new Event("change", {bubbles: true}));
  }

  function clickElement(element) {
    if (!element) return false;
    element.scrollIntoView({block: "center", inline: "nearest"});
    const tag = String(element.tagName || "").toUpperCase();
    if (tag === "A" || tag === "BUTTON" || tag === "INPUT") {
      element.click();
    } else {
      element.dispatchEvent(new MouseEvent("mousedown", {bubbles: true, cancelable: true, view: window}));
      element.dispatchEvent(new MouseEvent("mouseup", {bubbles: true, cancelable: true, view: window}));
      element.click();
    }
    return true;
  }

  function findByText(root, text, selectors = "a,button,label,span,div") {
    const wanted = Core.normalizeText(text);
    const nodes = Array.from((root || document).querySelectorAll(selectors));
    return nodes.find(node => Core.normalizeText(node.textContent) === wanted) || null;
  }

  function findVisibleByText(root, text, selectors = "a,button,label,span,div") {
    const wanted = Core.normalizeText(text);
    const nodes = Array.from((root || document).querySelectorAll(selectors));
    return nodes.find(node => visible(node) && Core.normalizeText(node.textContent) === wanted) || null;
  }

  function textContains(node, text) {
    if (!node) return false;
    return Core.normalizeText(node.textContent || "").includes(Core.normalizeText(text));
  }

  function findVisibleContainingText(root, text, selectors = "a,button,label,span,div") {
    const wanted = Core.normalizeText(text);
    const nodes = Array.from((root || document).querySelectorAll(selectors));
    return nodes.find(node => visible(node) && Core.normalizeText(node.textContent || "").includes(wanted)) || null;
  }

  function currentHostname() {
    try { return new URL(location.href).hostname.toLowerCase(); } catch (_error) { return ""; }
  }

  function isGoogleusercontentExportPage() {
    return currentHostname() === "scholar.googleusercontent.com";
  }

  function controlLabel(node) {
    if (!node) return "";
    const candidates = [
      node.textContent,
      node.getAttribute && node.getAttribute("aria-label"),
      node.getAttribute && node.getAttribute("title"),
      node.value
    ];
    return Core.normalizeWhitespace(candidates.filter(Boolean).join(" "));
  }

  function isCsvControl(node) {
    const label = Core.normalizeText(controlLabel(node));
    return label === "csv" || /(?:^|\s)csv(?:$|\s)/.test(label);
  }

  function looksLikeExportFormatMenu(menu) {
    if (!menu || !menu.querySelectorAll || !visible(menu)) return false;
    const labels = Array.from(menu.querySelectorAll('a,button,[role="menuitem"]'))
      .filter(visible)
      .map(controlLabel)
      .map(Core.normalizeText);
    const known = ["bibtex", "endnote", "refman", "csv"];
    return known.filter(name => labels.some(label => label === name || label.includes(name))).length >= 2;
  }

  function findCsvFormatControl(root = document) {
    if (!root || !root.querySelectorAll) return null;

    // Use the control Scholar visibly labels CSV. Numeric data-a/cit_fmt values
    // are undocumented implementation details and are intentionally not used to
    // choose the format. Hidden/stale menus are rejected by visible().
    const visibleMenus = Array.from(root.querySelectorAll(
      '#gs_res_ab_exp-d,[role="menu"],.gs_md_d,.gs_md_ul'
    )).filter(looksLikeExportFormatMenu);
    for (const menu of visibleMenus) {
      const candidate = Array.from(menu.querySelectorAll('a,button,[role="menuitem"]'))
        .find(node => visible(node) && isCsvControl(node));
      if (candidate) return candidate;
    }

    return Array.from(root.querySelectorAll('a,button,[role="menuitem"]'))
      .find(node => visible(node) && isCsvControl(node)) || null;
  }

  function hasExportSubmit(root) {
    if (!root || !root.querySelectorAll) return false;
    return Boolean(findVisibleByText(root, "EXPORT", "button,a,input[type='submit']")
      || Array.from(root.querySelectorAll("button,input[type='submit']")).find(node => visible(node) && Core.normalizeText(node.value || node.textContent) === "export"));
  }

  function looksLikeExportArticlesDialog(root) {
    if (!root || !root.querySelectorAll) return false;
    return textContains(root, "Export articles")
      && textContains(root, "Export all articles with this label")
      && hasExportSubmit(root);
  }

  function closestExportDialog(node) {
    // Scholar nests several generic modal wrappers (.gs_md_w/.gs_md_d). The
    // first modal-looking ancestor is not necessarily the whole dialog. Walk
    // upward until an ancestor actually contains BOTH export scopes + EXPORT.
    let cursor = node;
    for (let depth = 0; cursor && depth < 16; depth += 1, cursor = cursor.parentElement) {
      if (looksLikeExportArticlesDialog(cursor)) return cursor;
    }
    return null;
  }

  function findExportArticlesDialog() {
    const titleCandidates = Array.from(document.querySelectorAll("h1,h2,h3,div,span"))
      .filter(node => visible(node) && Core.normalizeText(node.textContent || "") === "export articles");
    for (const title of titleCandidates) {
      const dialog = closestExportDialog(title);
      if (dialog && visible(dialog)) return dialog;
    }

    // Current Scholar builds use custom radio wrappers (.gs_in_ra) and the
    // descriptive label may be easier to locate than the modal title.
    const scopeCandidates = Array.from(document.querySelectorAll("label,.gs_in_ra,span,div"))
      .filter(node => visible(node) && textContains(node, "Export all articles with this label"));
    for (const scopeText of scopeCandidates) {
      const dialog = closestExportDialog(scopeText);
      if (dialog && visible(dialog)) return dialog;
    }

    // Last-resort scan of Scholar modal wrappers. This avoids accidentally
    // selecting a header-only .gs_md_w ancestor.
    const modalCandidates = Array.from(document.querySelectorAll("[role='dialog'],.gs_md_d,.gs_md_w,.gs_md_wnw"));
    return modalCandidates.find(node => visible(node) && looksLikeExportArticlesDialog(node)) || null;
  }

  function associatedRadio(control) {
    if (!control) return null;
    if (control.matches && control.matches("input[type='radio']")) return control;

    const nested = control.querySelector && control.querySelector("input[type='radio']");
    if (nested) return nested;

    if (control.matches && control.matches("label")) {
      const forId = control.getAttribute && control.getAttribute("for");
      if (forId && document.getElementById) {
        const byId = document.getElementById(forId);
        if (byId && byId.matches && byId.matches("input[type='radio']")) return byId;
      }
    }

    const wrapper = control.closest && control.closest(".gs_in_ra");
    if (wrapper) {
      const inside = wrapper.querySelector && wrapper.querySelector("input[type='radio']");
      if (inside) return inside;
    }
    return null;
  }

  function exportScopeControl(dialog) {
    if (!dialog) return null;
    const target = "Export all articles with this label";

    // Prefer the actual <label>; Scholar normally keeps the radio input hidden
    // inside a .gs_in_ra wrapper, so visibility of the input itself is NOT a
    // requirement.
    const labels = Array.from(dialog.querySelectorAll("label"));
    const label = labels.find(node => visible(node) && textContains(node, target));
    if (label) return label;

    const wrappers = Array.from(dialog.querySelectorAll(".gs_in_ra"));
    const wrapper = wrappers.find(node => visible(node) && textContains(node, target));
    if (wrapper) return wrapper;

    const textNode = findVisibleContainingText(dialog, target, "span,div");
    if (textNode) {
      const enclosingLabel = textNode.closest && textNode.closest("label");
      if (enclosingLabel) return enclosingLabel;
      const enclosingWrapper = textNode.closest && textNode.closest(".gs_in_ra");
      if (enclosingWrapper) return enclosingWrapper;
    }

    // Fallback by radio order. Do not filter by visibility: Scholar visually
    // renders custom radios while the native <input type=radio> may be hidden.
    const radios = Array.from(dialog.querySelectorAll("input[type='radio'],[role='radio']"));
    if (radios.length >= 2) return radios[1];

    const customRadios = Array.from(dialog.querySelectorAll(".gs_in_ra"));
    return customRadios.length >= 2 ? customRadios[1] : null;
  }

  function radioSelected(control) {
    if (!control) return false;
    const radio = associatedRadio(control);
    if (radio) return Boolean(radio.checked) || (radio.getAttribute && radio.getAttribute("aria-checked") === "true");
    if (control.getAttribute && control.getAttribute("aria-checked") === "true") return true;
    if (control.classList && control.classList.contains("gs_sel")) return true;
    return false;
  }

  function forceSelectRadio(control) {
    const radio = associatedRadio(control);
    if (!radio) return false;
    try { radio.click(); } catch (_error) {}
    if (!radio.checked) {
      try { radio.checked = true; } catch (_error) {}
      try { radio.dispatchEvent(new Event("input", {bubbles: true})); } catch (_error) {}
      try { radio.dispatchEvent(new Event("change", {bubbles: true})); } catch (_error) {}
    }
    return Boolean(radio.checked);
  }

  function exportRequestFromControl(control) {
    if (!control) return null;

    const candidateUrls = [];
    const addCandidate = value => {
      const text = value == null ? "" : String(value).trim();
      if (text) candidateUrls.push(text);
    };
    try { addCandidate(control.href); } catch (_error) {}
    for (const attr of ["href", "data-href", "data-url", "data-export-url", "data-download-url", "data-action-url", "formaction", "onclick"]) {
      try { addCandidate(control.getAttribute && control.getAttribute(attr)); } catch (_error) {}
    }
    const expandedCandidates = [...candidateUrls];
    for (const raw of candidateUrls) {
      const decoded = raw.replace(/\\x26/g, "&").replace(/&amp;/g, "&");
      const match = decoded.match(/(?:https?:\/\/[^\s'"<>]+|\/citations\?[^\s'"<>]+)/i);
      if (match) expandedCandidates.push(match[0]);
    }
    for (const raw of expandedCandidates) {
      try {
        const url = new URL(raw, location.href);
        if (["scholar.google.com", "scholar.googleusercontent.com"].includes(url.hostname) && url.pathname.includes("citations")) {
          return {url: url.href, method: "GET", body: ""};
        }
      } catch (_error) {}
    }

    const form = control.closest && control.closest("form");
    if (!form) return null;
    const actionRaw = (control.getAttribute && control.getAttribute("formaction"))
      || (form.getAttribute && form.getAttribute("action")) || form.action || location.href;
    let action;
    try { action = new URL(actionRaw, location.href); } catch (_error) { return null; }
    if (!["scholar.google.com", "scholar.googleusercontent.com"].includes(action.hostname) || !action.pathname.includes("citations")) return null;

    const params = new URLSearchParams();
    const controls = Array.from(form.elements || (form.querySelectorAll ? form.querySelectorAll("input,select,textarea,button") : []));
    for (const field of controls) {
      if (!field || field.disabled || !field.name) continue;
      const type = String(field.type || "").toLowerCase();
      if ((type === "radio" || type === "checkbox") && !field.checked) continue;
      if ((type === "submit" || type === "button" || type === "image" || type === "reset") && field !== control) continue;
      if (field.tagName === "SELECT" && field.multiple && field.options) {
        for (const option of Array.from(field.options)) if (option.selected) params.append(field.name, option.value);
      } else {
        params.append(field.name, field.value == null ? "" : String(field.value));
      }
    }
    if (control.name && !params.has(control.name)) params.append(control.name, control.value == null ? "" : String(control.value));

    const method = String((control.getAttribute && control.getAttribute("formmethod"))
      || (form.getAttribute && form.getAttribute("method")) || form.method || "GET").toUpperCase() === "POST" ? "POST" : "GET";
    if (method === "GET") {
      for (const [key, value] of params.entries()) action.searchParams.append(key, value);
      return {url: action.href, method: "GET", body: ""};
    }
    return {url: action.href, method: "POST", body: params.toString()};
  }

  async function tryVerifiedExportRequest(control, run) {
    const request = exportRequestFromControl(control);
    if (!request) return false;
    const response = await chrome.runtime.sendMessage({type: "LSSC_VERIFY_EXPORT_REQUEST", request});
    if (response && response.ok && response.captured) {
      await log("scholar_csv_predownload_verified", {
        label: run.label,
        method: request.method,
        host: (() => { try { return new URL(request.url).hostname; } catch (_error) { return ""; } })()
      });
      return true;
    }
    await log("scholar_csv_predownload_fallback", {
      label: run.label,
      error: response && response.error || "No verifiable export request was captured"
    }, "warn");
    return false;
  }

  async function clickCsvFormat(run, {finalAfterCsv = false} = {}) {
    const csv = await waitFor(() => findCsvFormatControl(document), WAIT_TIMEOUT_MS);
    if (!csv) return attention("Scholar's export menu opened, but the CSV format control was not found.");

    const dataA = csv.getAttribute && csv.getAttribute("data-a");
    await log("scholar_csv_format_control", {
      text: Core.normalizeWhitespace(csv.textContent || ""),
      dataA: dataA || null,
      tag: csv.tagName || null,
      finalAfterCsv
    });

    let updated = await patch({
      stage: finalAfterCsv ? Core.STAGE.EXPORTING : Core.STAGE.EXPORT_MENU,
      exportFormatChosen: true,
      message: finalAfterCsv
        ? `CSV format selected for all ${Core.progress(run).saved} labeled articles. Waiting for the browser download…`
        : "CSV format selected. Choosing 'Export all articles with this label'…",
      lastUrl: location.href
    });

    // Always use Scholar's native, visible CSV control. Do not navigate to an
    // inferred export URL: in format-first builds that would bypass the later
    // "Export all articles with this label" scope confirmation.
    if (finalAfterCsv) {
      await chrome.runtime.sendMessage({type: "LSSC_EXPORT_TRIGGERED"});
      if (await tryVerifiedExportRequest(csv, updated)) return;
    }
    clickElement(csv);

    if (finalAfterCsv) {
      await log("scholar_csv_final_click", {label: run.label, dataA: dataA || null, native: true, verifierFallback: true});
      return;
    }

    const dialog = await waitFor(findExportArticlesDialog, WAIT_TIMEOUT_MS);
    if (!dialog) {
      return attention("Scholar's CSV option was clicked, but the 'Export articles' confirmation dialog did not appear.", {
        csvDataA: dataA || null
      });
    }
    updated = (await snapshot()).run || updated;
    return completeExportArticlesDialog(updated, dialog);
  }

  async function completeExportArticlesDialog(run, existingDialog = null) {
    const dialog = existingDialog || await waitFor(findExportArticlesDialog, WAIT_TIMEOUT_MS);
    if (!dialog) return attention("Scholar opened the export flow, but the 'Export articles' confirmation dialog did not appear.");

    const scope = exportScopeControl(dialog);
    if (!scope) {
      await log("scholar_export_scope_not_found", {
        dialogText: Core.normalizeWhitespace(dialog.textContent || "").slice(0, 1200),
        radioCount: dialog.querySelectorAll ? dialog.querySelectorAll("input[type='radio'],[role='radio'],.gs_in_ra").length : null
      }, "error");
      return attention("Scholar's export dialog opened, but 'Export all articles with this label' could not be selected.");
    }

    if (!radioSelected(scope)) clickElement(scope);
    let selected = await waitFor(() => {
      const current = exportScopeControl(dialog);
      return current && radioSelected(current) ? current : null;
    }, 1500);

    if (!selected) {
      forceSelectRadio(scope);
      selected = await waitFor(() => {
        const current = exportScopeControl(dialog);
        return current && radioSelected(current) ? current : null;
      }, 1500);
    }

    if (!selected) {
      const label = Array.from(dialog.querySelectorAll("label"))
        .find(node => visible(node) && textContains(node, "Export all articles with this label"));
      if (label) clickElement(label);
      selected = await waitFor(() => {
        const current = exportScopeControl(dialog);
        return current && radioSelected(current) ? current : null;
      }, 1500);
    }

    const finalScope = exportScopeControl(dialog);
    if (!finalScope || !radioSelected(finalScope)) {
      return attention("Scholar's export dialog did not accept the 'Export all articles with this label' selection.");
    }

    const submit = findVisibleByText(dialog, "EXPORT", "button,a,input[type='submit']")
      || Array.from(dialog.querySelectorAll("button,input[type='submit']")).find(visible);
    if (!submit) return attention("Scholar's export dialog has no EXPORT button.");

    const formatAlreadyChosen = Boolean(run.exportFormatChosen);
    if (formatAlreadyChosen) {
      await patch({
        stage: Core.STAGE.EXPORTING,
        exportScopeConfirmed: true,
        message: `Exporting all ${Core.progress(run).saved} articles in label ${run.label} as CSV…`,
        lastUrl: location.href
      });
      await chrome.runtime.sendMessage({type: "LSSC_EXPORT_TRIGGERED"});
      const current = (await snapshot()).run || run;
      if (await tryVerifiedExportRequest(submit, current)) {
        await log("scholar_export_confirmation_captured", {
          label: run.label, saved: Core.progress(run).saved, target: run.targetTotal,
          exportMode: run.exportMode || "full", formatAlreadyChosen: true
        });
        return;
      }
      clickElement(submit);
      await log("scholar_export_confirmation_submitted", {
        label: run.label, saved: Core.progress(run).saved, target: run.targetTotal,
        exportMode: run.exportMode || "full", formatAlreadyChosen: true, verifierFallback: true
      });
      return;
    }

    // Some Scholar builds show the scope dialog before the format menu. In
    // that ordering this click only confirms the scope; CSV must be chosen
    // afterwards. Do NOT mark EXPORTING until the actual CSV control is clicked.
    let updated = await patch({
      stage: Core.STAGE.EXPORT_MENU,
      exportScopeConfirmed: true,
      message: "All-label export scope confirmed. Selecting CSV format…",
      lastUrl: location.href
    });
    clickElement(submit);
    await log("scholar_export_scope_confirmed", {
      label: run.label, saved: Core.progress(run).saved, target: run.targetTotal
    });

    const csv = await waitFor(() => findCsvFormatControl(document), 3000);
    if (csv) {
      updated = (await snapshot()).run || updated;
      return clickCsvFormat(updated, {finalAfterCsv: true});
    }
    // A navigation may be in progress. pageshow/drive will continue from the
    // durable exportScopeConfirmed checkpoint.
  }

  function blockerPresent() {
    return Core.looksLikeBlocker(location.href, document.body ? document.body.innerText : "");
  }

  function temporaryOperationFailure() {
    return Core.classifyScholarOperationFailure(document.body ? document.body.innerText : "");
  }

  function rowTitle(row) {
    const title = row && row.querySelector(SELECTORS.resultTitle);
    return Core.normalizeWhitespace(title && title.textContent || "");
  }

  async function attention(message, details = {}) {
    await log("needs_attention", {message, ...details}, "warn");
    await patch({status: Core.STATUS.NEEDS_ATTENTION, message, lastUrl: location.href});
  }

  async function fail(message, details = {}) {
    await log("run_failed", {message, ...details}, "error");
    await patch({status: Core.STATUS.FAILED, message, lastUrl: location.href});
  }

  function signedInEnoughToSave() {
    const text = Core.normalizeText(document.body ? document.body.innerText : "");
    if (text.includes("my library")) return true;
    return Boolean(Array.from(document.querySelectorAll("a")).find(a => Core.normalizeText(a.textContent) === "my library"));
  }

  async function submitSearch(run) {
    if (blockerPresent()) return attention("Google Scholar is asking for human verification. Complete it in this tab, then click Resume.");
    const searchUrl = Core.buildSearchUrl(run.query);
    await patch({
      stage: Core.STAGE.SEARCH_LOADING,
      message: "Opening the exact synced Google Scholar search…",
      lastUrl: searchUrl
    });
    if (location.href !== searchUrl) location.replace(searchUrl);
  }

  function parseTotalFromPage() {
    const meta = document.querySelector(SELECTORS.resultsMeta);
    const candidates = [meta && meta.textContent, document.body && document.body.innerText].filter(Boolean);
    for (const value of candidates) {
      const parsed = Core.parseReportedTotal(value);
      if (parsed != null) return parsed;
    }
    return null;
  }

  function rowIds(rows) {
    return Core.uniqueStrings(rows.map(row => row.getAttribute("data-cid")));
  }

  function rowLibraryId(row) {
    return row && row.getAttribute ? Core.normalizeWhitespace(row.getAttribute("data-lid")) : "";
  }

  function rowAlreadySaved(row) {
    // Current Scholar marks saved search results with a non-empty data-lid on
    // the result row while keeping the visible control text as "Save".
    const lid = rowLibraryId(row);
    if (lid) return true;
    if (row.querySelector(SELECTORS.alreadySaved)) return true;
    const save = row.querySelector(SELECTORS.saveButton);
    if (!save) return Boolean(row.querySelector(SELECTORS.labelButton));
    const label = Core.normalizeText(save.getAttribute("aria-label") || save.textContent || "");
    return label === "saved" || save.getAttribute("aria-pressed") === "true";
  }

  function isChecked(control) {
    if (!control) return false;
    if (control.matches && control.matches("input[type='checkbox']")) return Boolean(control.checked);
    const aria = control.getAttribute && control.getAttribute("aria-checked");
    if (aria === "true") return true;
    if (aria === "false") return false;
    if (control.classList && (control.classList.contains("gs_sel") || control.classList.contains("gs_in_cb_sel"))) return true;
    const input = control.querySelector && control.querySelector("input[type='checkbox']");
    return input ? Boolean(input.checked) : false;
  }

  function findLabelControl(body, label) {
    const wanted = Core.normalizeText(label);
    const textNodes = Array.from(body.querySelectorAll("label,.gs_lbl,span,a,div"))
      .filter(node => Core.normalizeText(node.textContent) === wanted);
    for (const node of textNodes) {
      let cursor = node;
      for (let depth = 0; cursor && depth < 5; depth += 1, cursor = cursor.parentElement) {
        const direct = cursor.matches && cursor.matches("input[type='checkbox'],[role='checkbox'],.gs_in_cb") ? cursor : null;
        if (direct) return direct;
        const child = cursor.querySelector && cursor.querySelector("input[type='checkbox'],[role='checkbox'],.gs_in_cb");
        if (child) return child;
      }
    }
    return null;
  }

  async function ensureLabel(dialog, label) {
    const body = await waitFor(() => {
      const node = dialog.querySelector(SELECTORS.labelDialogBody) || document.querySelector(SELECTORS.labelDialogBody);
      return node && visible(node) ? node : null;
    });
    if (!body) throw new Error("Scholar label dialog body did not load");

    let control = findLabelControl(body, label);
    if (control) {
      if (!isChecked(control)) clickElement(control);
      return "existing";
    }

    const create = findByText(body, "Create new", "a,button,span")
      || document.querySelector("#gs_lbd_new a, #gs_lbd_new button")
      || document.querySelector(SELECTORS.labelCreate);
    if (!create) throw new Error("Scholar did not expose the Create new label control");
    clickElement(create);
    const input = await waitFor(() => document.querySelector(SELECTORS.labelNewInput));
    if (!input) throw new Error("Scholar did not expose the new-label input");
    setNativeValue(input, label);
    const checkbox = document.querySelector(SELECTORS.labelNewCheckbox);
    if (checkbox && !isChecked(checkbox)) clickElement(checkbox);
    return "created";
  }

  async function openLabelDialogForRow(row) {
    const preSaved = rowAlreadySaved(row);
    if (preSaved) {
      // On current Scholar builds the same Save/star control becomes the
      // entry point to the label dialog after an item is already in My Library.
      // Older builds exposed a separate .gs_or_lbl control. Support both.
      const labelButton = row.querySelector(SELECTORS.labelButton) || row.querySelector(SELECTORS.saveButton);
      if (!labelButton) {
        throw new Error("This result is already saved, but Scholar exposed no control for editing its labels");
      }
      clickElement(labelButton);
    } else {
      const save = row.querySelector(SELECTORS.saveButton);
      if (!save) throw new Error("Scholar result has no Save control");
      clickElement(save);
    }
    const dialog = await waitFor(() => {
      const node = document.querySelector(SELECTORS.labelDialog);
      return node && visible(node) ? node : null;
    });
    if (!dialog) throw new Error("Scholar Save/Label dialog did not open");
    return {dialog, preSaved};
  }

  async function saveAndLabel(row, run) {
    const cid = row.getAttribute("data-cid");
    if (!cid) throw new Error("Scholar row has no data-cid");
    for (let attempt = 1; attempt <= 2; attempt += 1) {
      try {
        const {dialog, preSaved} = await openLabelDialogForRow(row);
        const labelMode = await ensureLabel(dialog, run.label);
        const done = document.querySelector(SELECTORS.labelDone) || findByText(dialog, "Done", "button,a");
        if (!done) throw new Error("Scholar label dialog has no Done control");
        clickElement(done);
        const closed = await waitFor(() => {
          const node = document.querySelector(SELECTORS.labelDialog);
          return !node || !visible(node);
        }, WAIT_TIMEOUT_MS);
        if (!closed) throw new Error("Scholar label dialog did not close after Done");
        const transientAfterDone = temporaryOperationFailure();
        if (transientAfterDone) throw new Error("Scholar temporary operation failure after save");
        await log("paper_saved", {cid, attempt, preSaved, labelMode});
        return {ok: true, cid, preSaved, labelMode};
      } catch (error) {
        const transient = temporaryOperationFailure();
        await log("paper_save_attempt_failed", {
          cid, attempt, transient, error: String(error && error.message || error)
        }, "warn");
        const cancel = document.querySelector("#gs_md_albl-d-x");
        if (cancel) clickElement(cancel);
        if (blockerPresent()) {
          await attention("Google Scholar interrupted the run with human verification. Complete it, then click Resume.", {cid});
          return {ok: false, cid, blocked: true};
        }
        if (attempt < 2) await sleep(transient ? TRANSIENT_RETRY_DELAY_MS : ACTION_DELAY_MS);
        else return {ok: false, cid, transient: Boolean(transient), error: String(error && error.message || error)};
      }
    }
    return {ok: false, cid, error: "Unknown save failure"};
  }

  async function applyThrottleCooldown(run, cid, phase = "collection") {
    const throttleLevel = Number(run.throttleLevel || 0) + 1;
    const throttleEvents = Number(run.throttleEvents || 0) + 1;
    const delayMs = Core.throttleCooldownMs(throttleLevel);
    const throttleUntil = new Date(Date.now() + delayMs).toISOString();
    const seconds = Math.round(delayMs / 1000);
    let updated = await patch({
      throttleLevel,
      throttleEvents,
      throttleUntil,
      consecutiveFailures: 0,
      message: `Google Scholar is temporarily throttling saves. Cooling down ${seconds}s before retrying the SAME record (${cid}); no paper is being skipped.`,
      lastUrl: location.href
    });
    await log("scholar_throttle_cooldown", {cid, phase, throttleLevel, throttleEvents, delayMs}, "warn");
    await sleep(delayMs);
    updated = (await snapshot()).run;
    if (!updated || updated.status !== Core.STATUS.RUNNING) return updated;
    return patch({
      throttleUntil: null,
      message: `Cooldown finished. Retrying Scholar record ${cid}…`,
      lastUrl: location.href
    });
  }

  function findNextLink() {
    const aria = document.querySelector("a[aria-label='Next'], a[aria-label='Next page']");
    if (aria) return aria;
    const byText = Array.from(document.querySelectorAll("a")).find(a => Core.normalizeText(a.textContent) === "next");
    if (byText) return byText;
    const currentStart = Core.extractStartOffset(location.href);
    const forward = Array.from(document.querySelectorAll("#gs_n a[href]"))
      .map(anchor => ({anchor, start: Core.extractStartOffset(anchor.href)}))
      .filter(item => item.start > currentStart)
      .sort((a, b) => a.start - b.start);
    return forward.length ? forward[0].anchor : null;
  }

  async function goToLibrary(run, options = {}) {
    const unresolvedCount = Core.unresolvedIds(run).length;
    const partial = options.partial === true || unresolvedCount > 0 || Core.progress(run).saved < (run.targetTotal ?? Core.RESULT_CAP);
    const exportMode = partial ? "partial" : "full";
    const message = partial
      ? `Traversal/retries finished with ${unresolvedCount} unresolved record(s). Opening My Library to verify the actual LitSync label before any export…`
      : `Checkpointed ${Core.progress(run).saved}/${run.targetTotal}. Opening My Library to verify actual label membership before export…`;
    await patch({
      stage: Core.STAGE.OPENING_LIBRARY,
      exportMode,
      exportFormatChosen: false,
      exportScopeConfirmed: false,
      exportStartedAt: null,
      exportDownloadId: null,
      labelAuditIds: [],
      labelAuditPages: [],
      labelAuditComplete: false,
      labelVerifiedIds: [],
      labelVerifiedCount: null,
      labelMissingIds: [],
      labelUnexpectedIds: [],
      message,
      lastUrl: location.href
    });
    location.href = "https://scholar.google.com/scholar?scilib=1&hl=en";
  }

  async function collectPage(run) {
    if (blockerPresent()) return attention("Google Scholar is asking for human verification. Complete it in this tab, then click Resume.");
    if (!signedInEnoughToSave()) return attention("Sign in to the Google account you use for Google Scholar, then click Resume. The extension will continue from this page.");

    const rows = Array.from(document.querySelectorAll(SELECTORS.resultRows));
    const latestParsedTotal = parseTotalFromPage();
    const reportedTotal = latestParsedTotal != null ? latestParsedTotal : run.reportedTotal;
    const initialReportedTotal = run.initialReportedTotal != null
      ? run.initialReportedTotal
      : (run.reportedTotal != null ? run.reportedTotal : reportedTotal);
    const targetTotal = Core.provisionalTarget(run.targetTotal, reportedTotal);

    if (reportedTotal === 0 || targetTotal === 0) {
      await patch({
        reportedTotal: 0,
        initialReportedTotal: initialReportedTotal ?? 0,
        latestReportedTotal: 0,
        accessibleTotal: 0,
        targetTotal: 0,
        status: Core.STATUS.COMPLETE,
        stage: Core.STAGE.DONE,
        message: "Google Scholar returned 0 results; nothing to export.",
        lastUrl: location.href
      });
      return;
    }

    if (!rows.length) {
      return attention("No Scholar result rows were found. If the page is still loading, wait a moment and click Resume; otherwise Scholar may have changed its page layout.");
    }

    const ids = rowIds(rows);
    const identity = Core.pageIdentity(location.href, ids);
    const discoveredIds = Core.uniqueStrings([...(run.discoveredIds || []), ...ids]);
    const pageStart = Core.extractStartOffset(location.href);
    const resultLocations = {...(run.resultLocations || {})};
    const libraryIdsBySearchId = {...(run.libraryIdsBySearchId || {})};
    for (const row of rows) {
      const cid = row.getAttribute("data-cid");
      if (!cid) continue;
      if (!Number.isInteger(resultLocations[cid])) resultLocations[cid] = pageStart;
      const lid = rowLibraryId(row);
      if (lid) libraryIdsBySearchId[cid] = lid;
    }
    let nextRun = await patch({
      reportedTotal,
      initialReportedTotal,
      latestReportedTotal: reportedTotal,
      targetTotal,
      stage: Core.STAGE.COLLECTING,
      discoveredIds,
      resultLocations,
      libraryIdsBySearchId,
      pageIdentities: Core.appendUnique(run.pageIdentities || [], identity),
      pagesVisited: Array.from(new Set([...(run.pagesVisited || []), Core.extractStartOffset(location.href)])),
      message: `Collecting Scholar results: ${Core.progress({...run, targetTotal}).saved}/${targetTotal}`,
      lastUrl: location.href
    });

    for (const row of rows) {
      nextRun = (await snapshot()).run;
      if (!nextRun || nextRun.status !== Core.STATUS.RUNNING) return;
      const currentProgress = Core.progress(nextRun);
      if (currentProgress.saved >= nextRun.targetTotal) return goToLibrary(nextRun);
      const cid = row.getAttribute("data-cid");
      if (!cid || (nextRun.savedIds || []).includes(cid)) continue;
      if (nextRun.unresolved && nextRun.unresolved[cid]?.skippedDuringTraversal) continue;

      let result = await saveAndLabel(row, nextRun);
      while (!result.ok && result.transient && !result.blocked) {
        nextRun = await applyThrottleCooldown(nextRun, cid, "collection");
        if (!nextRun || nextRun.status !== Core.STATUS.RUNNING) return;
        result = await saveAndLabel(row, nextRun);
      }

      if (!result.ok) {
        if (result.blocked) return;
        const now = new Date().toISOString();
        const unresolved = Core.recordUnresolved(nextRun.unresolved || {}, cid, {
          error: result.error || "save failed",
          at: now,
          title: rowTitle(row),
          pageUrl: location.href,
          startOffset: Core.extractStartOffset(location.href),
          transient: false,
          skippedDuringTraversal: true,
          finalized: false
        });
        nextRun = await patch({
          unresolved,
          consecutiveFailures: Number(nextRun.consecutiveFailures || 0) + 1,
          message: `Recorded one non-throttle Scholar save failure (${cid}) and continued automatically. ${Core.unresolvedIds({unresolved}).length} unresolved so far.`,
          lastUrl: location.href
        });
        await log("paper_skipped_after_retries", {cid, title: rowTitle(row), transient: false}, "warn");
        continue;
      }

      const refreshedLibraryId = rowLibraryId(row);
      const refreshedLibraryIds = {...(nextRun.libraryIdsBySearchId || {})};
      if (refreshedLibraryId) refreshedLibraryIds[cid] = refreshedLibraryId;
      nextRun = await patch({
        savedIds: Core.appendUnique(nextRun.savedIds || [], cid),
        libraryIdsBySearchId: refreshedLibraryIds,
        unresolved: Object.fromEntries(Object.entries(nextRun.unresolved || {}).filter(([key]) => key !== cid)),
        consecutiveFailures: 0,
        throttleLevel: 0,
        throttleUntil: null,
        message: `Saved ${Core.progress({...nextRun, savedIds: Core.appendUnique(nextRun.savedIds || [], cid)}).saved}/${nextRun.targetTotal}`,
        lastUrl: location.href
      });
      await sleep(ACTION_DELAY_MS);
    }

    nextRun = (await snapshot()).run;
    if (!nextRun || nextRun.status !== Core.STATUS.RUNNING) return;
    if (Core.progress(nextRun).saved >= nextRun.targetTotal) return goToLibrary(nextRun);

    const next = findNextLink();
    const currentStart = Core.extractStartOffset(location.href);
    const nextStart = next && next.href ? Core.extractStartOffset(next.href) : null;
    const latestTotal = parseTotalFromPage() ?? nextRun.latestReportedTotal ?? nextRun.reportedTotal;
    const uniqueDiscovered = Core.uniqueStrings(nextRun.discoveredIds || []).length;
    const atConfirmedEnd = Core.confirmedEndOfResults({
      currentStart,
      rowCount: rows.length,
      latestReportedTotal: latestTotal,
      discoveredCount: uniqueDiscovered,
      nextStart
    });

    if (atConfirmedEnd) {
      const finalTarget = Core.finalAccessibleTarget(uniqueDiscovered);
      nextRun = await patch({
        reportedTotal: latestTotal,
        latestReportedTotal: latestTotal,
        accessibleTotal: finalTarget,
        targetTotal: finalTarget,
        message: initialReportedTotal != null && latestTotal != null && initialReportedTotal !== latestTotal
          ? `Reached Scholar's actual end: ${finalTarget} unique results were accessible (initial estimate ${initialReportedTotal}, latest ${latestTotal}).`
          : `Reached Scholar's actual end: ${finalTarget} unique results were accessible.`,
        lastUrl: location.href
      });
      await log("actual_end_of_results", {
        currentStart, rowCount: rows.length, uniqueDiscovered, initialReportedTotal, latestTotal, finalTarget, nextStart
      });

      const pending = Core.unresolvedIds(nextRun, {includeFinalized: false});
      if (pending.length) {
        await patch({
          stage: Core.STAGE.RETRYING_FAILURES,
          retryStartedAt: nextRun.retryStartedAt || new Date().toISOString(),
          message: `Reached the actual end of Scholar results. Retrying ${pending.length} unresolved record(s) only…`,
          lastUrl: location.href
        });
        setTimeout(drive, 100);
        return;
      }
      if (Core.progress(nextRun).saved >= finalTarget) return goToLibrary(nextRun);
      const remaining = Core.unresolvedIds(nextRun);
      if (remaining.length) return goToLibrary(nextRun, {partial: true});
      return attention(`Scholar traversal ended at ${finalTarget} accessible results, but only ${Core.progress(nextRun).saved} are saved and no unresolved IDs explain the gap.`, {finalTarget});
    }

    if (!next) {
      return attention("Scholar's Next control disappeared before the current page looked like the true end of the result set. Stopping to avoid silently truncating collection.", {currentStart, rowCount: rows.length, latestTotal});
    }
    if (nextStart == null || nextStart <= currentStart) {
      return attention("Scholar's Next link did not advance to a later result page and this page does not look like the real final page; stopping to avoid an infinite loop.", {currentStart, nextStart, rowCount: rows.length, latestTotal});
    }
    await patch({message: `Page complete. Moving to results ${nextStart + 1} onward…`, lastUrl: location.href});
    await sleep(PAGE_DELAY_MS);
    clickElement(next);
  }

  async function retryFailures(run) {
    if (blockerPresent()) return attention("Google Scholar is asking for human verification. Complete it, then click Resume.");
    const pending = Core.unresolvedIds(run, {includeFinalized: false});
    if (!pending.length) {
      const remaining = Core.unresolvedIds(run);
      if (remaining.length) {
        return goToLibrary(run, {partial: true});
      }
      if (Core.progress(run).saved >= run.targetTotal) return goToLibrary(run);
      return attention(`Retry queue is empty, but only ${Core.progress(run).saved}/${run.targetTotal} records are saved.`, {});
    }

    const cid = pending[0];
    const failure = (run.unresolved || {})[cid] || {};
    const targetUrl = failure.pageUrl || Core.buildSearchUrl(run.query, Number(failure.startOffset || 0));
    if (Core.extractStartOffset(location.href) !== Core.extractStartOffset(targetUrl) || !Core.isScholarSearchUrl(location.href)) {
      await patch({message: `Final retry for ${cid}: reopening its Scholar result page…`, lastUrl: targetUrl});
      location.href = targetUrl;
      return;
    }

    const rows = Array.from(document.querySelectorAll(SELECTORS.resultRows));
    const row = rows.find(item => item.getAttribute("data-cid") === cid);
    if (!row) {
      const unresolved = {...(run.unresolved || {})};
      unresolved[cid] = {...failure, finalized: true, finalError: "Result ID not found on its recorded retry page", finalAttemptedAt: new Date().toISOString()};
      await patch({unresolved, message: `Final retry could not find ${cid} on its recorded page; moving to the next failed record.`});
      await log("paper_final_retry_not_found", {cid, targetUrl}, "warn");
      setTimeout(drive, 100);
      return;
    }

    let result = await saveAndLabel(row, run);
    while (!result.ok && result.transient && !result.blocked) {
      run = await applyThrottleCooldown((await snapshot()).run, cid, "final_retry");
      if (!run || run.status !== Core.STATUS.RUNNING) return;
      result = await saveAndLabel(row, run);
    }
    if (result.blocked) return;
    if (result.ok) {
      const current = (await snapshot()).run || run;
      const unresolved = Object.fromEntries(Object.entries(current.unresolved || {}).filter(([key]) => key !== cid));
      const savedIds = Core.appendUnique(current.savedIds || [], cid);
      const updated = await patch({
        savedIds, unresolved, consecutiveFailures: 0, throttleLevel: 0, throttleUntil: null,
        message: `Recovered failed Scholar result ${cid}. ${Core.unresolvedIds({unresolved}, {includeFinalized: false}).length} retry item(s) remain.`,
        lastUrl: location.href
      });
      await log("paper_recovered_on_final_retry", {cid, title: rowTitle(row)});
      if (!Core.unresolvedIds(updated, {includeFinalized: false}).length) {
        if (Core.unresolvedIds(updated).length) return goToLibrary(updated, {partial: true});
        if (Core.progress(updated).saved >= updated.targetTotal) return goToLibrary(updated);
      }
      setTimeout(drive, ACTION_DELAY_MS);
      return;
    }

    const current = (await snapshot()).run || run;
    const unresolved = {...(current.unresolved || {})};
    unresolved[cid] = {
      ...failure,
      finalized: true,
      finalError: result.error || "final retry failed",
      finalTransient: false,
      finalAttemptedAt: new Date().toISOString()
    };
    await patch({unresolved, message: `Final non-throttle retry failed for ${cid}; recorded and moving on.`});
    await log("paper_final_retry_failed", {cid, transient: false, error: result.error || "final retry failed"}, "warn");
    setTimeout(drive, 100);
  }

  function currentScilib() {
    try { return new URL(location.href).searchParams.get("scilib"); } catch (_error) { return null; }
  }

  function onSpecificLibraryLabelPage() {
    const value = currentScilib();
    return Boolean(value && value !== "1");
  }

  function labelPageRows() {
    return Array.from(document.querySelectorAll(SELECTORS.resultRows));
  }

  function repairOffsets(run) {
    const missing = Core.uniqueStrings(run.labelRepairIds || run.labelMissingIds || []);
    const locations = run.resultLocations && typeof run.resultLocations === "object" ? run.resultLocations : {};
    const mapped = Core.uniqueStrings(missing.map(cid => Number.isInteger(locations[cid]) ? String(locations[cid]) : ""))
      .map(Number).filter(Number.isInteger).sort((a, b) => a - b);
    if (mapped.length === missing.length && mapped.length) return Array.from(new Set(mapped));
    const visited = Array.from(new Set((run.pagesVisited || []).filter(value => Number.isInteger(value) && value >= 0))).sort((a, b) => a - b);
    return visited.length ? visited : [0];
  }

  async function indexLibraryIds(run) {
    if (blockerPresent()) return attention("Google Scholar is asking for human verification. Complete it in this tab, then click Resume.");
    if (!Core.isScholarSearchUrl(location.href)) {
      location.href = Core.buildSearchUrl(run.query, 0);
      return;
    }

    const rows = Array.from(document.querySelectorAll(SELECTORS.resultRows));
    if (!rows.length) return attention("No Scholar result rows were found while rebuilding Library IDs. Wait for the page to finish loading, then click Resume.");

    const expectedSearchIds = new Set(Core.uniqueStrings(run.savedIds || []));
    const libraryIdsBySearchId = {...(run.libraryIdsBySearchId || {})};
    for (const row of rows) {
      const cid = row.getAttribute("data-cid");
      if (!cid || !expectedSearchIds.has(cid)) continue;
      const lid = rowLibraryId(row);
      if (lid) libraryIdsBySearchId[cid] = lid;
    }

    const indexed = Core.uniqueStrings(run.savedIds || []).filter(cid => libraryIdsBySearchId[cid]).length;
    const currentStart = Core.extractStartOffset(location.href);
    const next = findNextLink();
    const nextStart = next && next.href ? Core.extractStartOffset(next.href) : null;

    let updated = await patch({
      stage: Core.STAGE.INDEXING_LIBRARY_IDS,
      integrityVersion: 3,
      libraryIdsBySearchId,
      message: `Rebuilding Scholar Library IDs: ${indexed}/${expectedSearchIds.size} checkpointed records mapped…`,
      lastUrl: location.href
    });

    if (indexed === expectedSearchIds.size) {
      updated = await patch({
        stage: Core.STAGE.OPENING_LIBRARY,
        labelAuditIds: [],
        labelAuditPages: [],
        labelAuditComplete: false,
        labelVerifiedIds: [],
        labelVerifiedCount: null,
        labelMissingIds: [],
        labelUnexpectedIds: [],
        message: `Mapped all ${indexed}/${expectedSearchIds.size} search records to their Scholar Library IDs. Re-auditing the LitSync label…`,
        lastUrl: location.href
      });
      return goToLibrary(updated, {partial: updated.exportMode === "partial"});
    }

    if (!next || nextStart == null || nextStart <= currentStart) {
      return attention(`Could map only ${indexed}/${expectedSearchIds.size} checkpointed records to Scholar Library IDs. Automatic repair stopped safely; no records were deleted or reset.`, {indexed, expected: expectedSearchIds.size});
    }
    await sleep(PAGE_DELAY_MS);
    clickElement(next);
  }

  async function auditRunLabel(run) {
    if (blockerPresent()) return attention("Google Scholar is asking for human verification. Complete it, then click Resume.");
    if (!Core.isScholarLibraryUrl(location.href) || !onSpecificLibraryLabelPage()) {
      return openRunLabel(run);
    }

    const rows = labelPageRows();
    const ids = rowIds(rows);
    const currentStart = Core.extractStartOffset(location.href);
    const auditIds = Core.uniqueStrings([...(run.labelAuditIds || []), ...ids]);
    const auditPages = Array.from(new Set([...(run.labelAuditPages || []), currentStart])).sort((a, b) => a - b);
    const next = findNextLink();
    const nextStart = next && next.href ? Core.extractStartOffset(next.href) : null;
    const atEnd = rows.length < 10 || !next || nextStart == null || nextStart <= currentStart;

    let updated = await patch({
      stage: Core.STAGE.VERIFYING_LABEL,
      labelAuditIds: auditIds,
      labelAuditPages: auditPages,
      labelVerifiedCount: auditIds.length,
      message: `Auditing LitSync label membership: ${auditIds.length}/${run.targetTotal} unique Scholar IDs verified so far…`,
      lastUrl: location.href
    });

    if (!atEnd) {
      await sleep(900);
      clickElement(next);
      return;
    }

    const savedSearchIds = Core.uniqueStrings(updated.savedIds || []);
    const expected = Core.expectedLibraryIds(updated);
    const requiredCount = updated.exportMode === "partial" ? savedSearchIds.length : updated.targetTotal;

    // v0.1.9 compared search-result data-cid values directly with My Library
    // data-cid values. Scholar does not use the same ID namespace there. The
    // corresponding Library ID is exposed as data-lid on a saved search row.
    // Backfill that mapping once for legacy checkpoints before comparing.
    if (expected.length !== savedSearchIds.length) {
      updated = await patch({
        stage: Core.STAGE.INDEXING_LIBRARY_IDS,
        integrityVersion: 3,
        labelAuditComplete: false,
        message: `Scholar uses different IDs on search pages and My Library. Mapping ${savedSearchIds.length} checkpointed results to their Library IDs before repair…`,
        lastUrl: location.href
      });
      location.href = Core.buildSearchUrl(updated.query, 0);
      return;
    }

    const missing = Core.differenceStrings(expected, auditIds);
    const unexpected = Core.differenceStrings(auditIds, expected);
    const exact = missing.length === 0 && unexpected.length === 0 && auditIds.length === requiredCount && expected.length === requiredCount;

    const missingSet = new Set(missing);
    const missingSearchIds = savedSearchIds.filter(cid => missingSet.has(updated.libraryIdsBySearchId && updated.libraryIdsBySearchId[cid]));

    updated = await patch({
      labelAuditComplete: true,
      labelVerifiedIds: auditIds,
      labelVerifiedCount: auditIds.length,
      labelMissingIds: missingSearchIds,
      labelUnexpectedIds: unexpected,
      integrityVersion: 3,
      message: exact
        ? `Label integrity PASS: ${auditIds.length}/${requiredCount} expected Scholar Library records are actually present in ${updated.label}.`
        : `Label integrity mismatch: checkpointed=${expected.length}, actually-in-label=${auditIds.length}, missing=${missing.length}, unexpected=${unexpected.length}.`,
      lastUrl: location.href
    });

    await log(exact ? "label_integrity_pass" : "label_integrity_mismatch", {
      target: updated.targetTotal,
      requiredCount,
      checkpointed: expected.length,
      verifiedInLabel: auditIds.length,
      missingCount: missing.length,
      unexpectedCount: unexpected.length,
      missingSample: missing.slice(0, 20),
      unexpectedSample: unexpected.slice(0, 20),
      pagesAudited: auditPages.length
    }, exact ? "info" : "warn");

    if (exact) {
      await patch({
        stage: Core.STAGE.LABEL_LOADING,
        labelRepairIds: [],
        labelRepairAttemptedIds: [],
        message: `Integrity verified ${auditIds.length}/${requiredCount}. Starting native Scholar CSV export…`,
        lastUrl: location.href
      });
      setTimeout(drive, 100);
      return;
    }

    if (unexpected.length) {
      return attention(`The LitSync label contains ${unexpected.length} unexpected Scholar ID(s). Refusing automatic export so the CSV cannot silently include the wrong records.`, {
        unexpected: unexpected.slice(0, 30), verifiedInLabel: auditIds.length, target: updated.targetTotal
      });
    }

    if (!missing.length) {
      return attention(`The LitSync label count (${auditIds.length}) does not equal the strict target (${updated.targetTotal}) even though no ID-level difference was found. Export stopped for integrity.`, {
        verifiedInLabel: auditIds.length, target: updated.targetTotal
      });
    }

    const nextRound = Number(updated.labelRepairRound || 0) + 1;
    if (nextRound > 2) {
      return attention(`After two repair rounds, ${missing.length} Scholar record(s) are still absent from the LitSync label. Nothing was exported as complete.`, {
        missing: missing.slice(0, 50), verifiedInLabel: auditIds.length, target: updated.targetTotal
      });
    }

    const repairing = await patch({
      stage: Core.STAGE.REPAIRING_LABEL,
      labelRepairRound: nextRound,
      labelRepairIds: missingSearchIds,
      labelRepairAttemptedIds: [],
      labelRepairFoundIds: [],
      message: `Repair round ${nextRound}: ${missingSearchIds.length} checkpointed Scholar record(s) are missing from the label. Revisiting only their original search pages…`,
      lastUrl: location.href
    });
    const offsets = repairOffsets(repairing);
    location.href = Core.buildSearchUrl(repairing.query, offsets[0] || 0);
  }

  async function repairLabelMembership(run) {
    if (blockerPresent()) return attention("Google Scholar is asking for human verification. Complete it, then click Resume.");
    if (!Core.isScholarSearchUrl(location.href)) {
      const offsets = repairOffsets(run);
      location.href = Core.buildSearchUrl(run.query, offsets[0] || 0);
      return;
    }

    const offsets = repairOffsets(run);
    const currentStart = Core.extractStartOffset(location.href);
    if (!offsets.includes(currentStart)) {
      const nextOffset = offsets.find(value => value > currentStart) ?? offsets[0] ?? 0;
      location.href = Core.buildSearchUrl(run.query, nextOffset);
      return;
    }

    const rows = Array.from(document.querySelectorAll(SELECTORS.resultRows));
    const missingSet = new Set(Core.uniqueStrings(run.labelRepairIds || []));
    let attempted = Core.uniqueStrings(run.labelRepairAttemptedIds || []);
    let found = Core.uniqueStrings(run.labelRepairFoundIds || []);

    for (const row of rows) {
      let current = (await snapshot()).run || run;
      if (!current || current.status !== Core.STATUS.RUNNING) return;
      const cid = row.getAttribute("data-cid");
      if (!cid || !missingSet.has(cid) || attempted.includes(cid)) continue;
      found = Core.appendUnique(found, cid);

      let result = await saveAndLabel(row, current);
      while (!result.ok && result.transient && !result.blocked) {
        current = await applyThrottleCooldown((await snapshot()).run || current, cid, "label_repair");
        if (!current || current.status !== Core.STATUS.RUNNING) return;
        result = await saveAndLabel(row, current);
      }
      if (result.blocked) return;

      attempted = Core.appendUnique(attempted, cid);
      const patchValue = {
        labelRepairAttemptedIds: attempted,
        labelRepairFoundIds: found,
        message: result.ok
          ? `Repair attempted for missing label ID ${cid}. ${attempted.length}/${missingSet.size} repair candidates processed; final membership will be re-audited.`
          : `Repair attempt for ${cid} failed without a throttle signal. It remains missing and will be caught by the post-repair label audit.`,
        lastUrl: location.href
      };
      await patch(patchValue);
      await log(result.ok ? "label_repair_attempted" : "label_repair_attempt_failed", {
        cid, pageStart: currentStart, round: current.labelRepairRound, error: result.error || null
      }, result.ok ? "info" : "warn");
      await sleep(ACTION_DELAY_MS);
    }

    const fresh = (await snapshot()).run || run;
    attempted = Core.uniqueStrings(fresh.labelRepairAttemptedIds || attempted);
    if (attempted.length >= missingSet.size) {
      await patch({
        stage: Core.STAGE.OPENING_LIBRARY,
        labelAuditIds: [], labelAuditPages: [], labelAuditComplete: false,
        labelVerifiedIds: [], labelVerifiedCount: null,
        message: `Repair attempts finished for all ${missingSet.size} missing IDs. Re-auditing the actual LitSync label from page 1…`,
        lastUrl: location.href
      });
      location.href = "https://scholar.google.com/scholar?scilib=1&hl=en";
      return;
    }

    const index = offsets.indexOf(currentStart);
    const nextOffset = index >= 0 && index + 1 < offsets.length ? offsets[index + 1] : null;
    if (nextOffset != null) {
      await patch({message: `Repair scan: moving to original Scholar result page starting at ${nextOffset + 1}…`, lastUrl: location.href});
      await sleep(700);
      location.href = Core.buildSearchUrl(fresh.query, nextOffset);
      return;
    }

    await patch({
      stage: Core.STAGE.OPENING_LIBRARY,
      labelAuditIds: [], labelAuditPages: [], labelAuditComplete: false,
      labelVerifiedIds: [], labelVerifiedCount: null,
      message: `Repair scan finished. Found ${Core.uniqueStrings(fresh.labelRepairFoundIds || found).length}/${missingSet.size} missing IDs in the current Scholar result ordering. Re-auditing the label now…`,
      lastUrl: location.href
    });
    location.href = "https://scholar.google.com/scholar?scilib=1&hl=en";
  }

  function canonicalRunLabelText(value) {
    return String(value == null ? "" : value)
      .replace(/[\u200B-\u200D\u2060\uFEFF]/g, "")
      .replace(/\s+/g, "")
      .replace(/(?:\.{3}|…)+$/g, "")
      .trim();
  }

  function runLabelDestination(href) {
    try {
      const url = new URL(href, location.href);
      const scilib = url.searchParams.get("scilib");
      if (url.origin !== "https://scholar.google.com" || url.pathname !== "/scholar"
          || !/^\d+$/.test(scilib || "") || ["1", "5", "6"].includes(scilib)
          || url.searchParams.has("q") || url.searchParams.has("cluster")) return null;
      // A label destination must always begin an unfiltered, first-page audit.
      const clean = new URL("https://scholar.google.com/scholar");
      clean.searchParams.set("scilib", scilib);
      clean.searchParams.set("hl", "en");
      if (url.searchParams.has("authuser")) clean.searchParams.set("authuser", url.searchParams.get("authuser"));
      return {scilib, href: clean.href};
    } catch (_error) { return null; }
  }

  function findRunLabelLink(label) {
    const wanted = canonicalRunLabelText(label);
    const links = Array.from(document.querySelectorAll("a"))
      .filter(a => runLabelDestination(a.href));

    function valuesFor(a) {
      const nodes = [a, ...Array.from(a.querySelectorAll ? a.querySelectorAll("[title],[aria-label],[data-label],[data-name]") : [])];
      return [a.textContent, a.innerText, ...nodes.flatMap(node =>
        ["title", "aria-label", "data-label", "data-name"].map(name => node.getAttribute(name))
      )].map(canonicalRunLabelText).filter(Boolean);
    }

    // Exact match after removing display-only whitespace, zero-width chars,
    // and a trailing visual ellipsis. Scholar can wrap a long sidebar label
    // across nested spans, so plain normalizeWhitespace() is not sufficient.
    function uniqueDestination(matches) {
      // Live Scholar renders both a responsive menu and a desktop sidebar.
      // Their duplicate anchors are one label when the destination is the same.
      const destinations = new Set(matches.map(a => runLabelDestination(a.href).href));
      return destinations.size === 1 ? (matches.find(visible) || matches[0]) : null;
    }
    const exact = links.filter(a => valuesFor(a).includes(wanted));
    if (exact.length) return uniqueDestination(exact);

    // Legacy v0.2.3 labels can be visually shortened by Scholar. Treat a
    // sufficiently-long UNIQUE textual prefix as the same run label whether
    // the DOM exposes literal dots, a Unicode ellipsis, CSS clipping, or
    // whitespace inserted between nested spans. We never accept a short or
    // ambiguous prefix; the later strict ID-set audit is still authoritative.
    const matches = links.filter(a => valuesFor(a).some(candidate => {
      if (candidate.length < 18) return false;
      return wanted.startsWith(candidate);
    }));
    return uniqueDestination(matches);
  }

  function runLabelDiagnostics(label) {
    const nodes = Array.from(document.querySelectorAll("a,button,[role], [data-label], [data-name]"));
    const candidates = nodes.filter(node => {
      const text = [node.textContent, node.getAttribute("title"), node.getAttribute("aria-label"),
        node.getAttribute("data-label"), node.getAttribute("data-name")].join(" ");
      return /litsync/i.test(text) || runLabelDestination(node.href);
    });
    const describe = node => ({
      tag: node.tagName, text: Core.normalizeWhitespace(node.innerText || node.textContent || "").slice(0, 1000),
      rawText: String(node.textContent || "").slice(0, 1000),
      href: node.getAttribute("href"), role: node.getAttribute("role"),
      ariaLabel: node.getAttribute("aria-label"), title: node.getAttribute("title"),
      id: node.id || "", className: node.className || "",
      data: Object.fromEntries(Array.from(node.attributes || []).filter(a => a.name.startsWith("data-")).map(a => [a.name, a.value.slice(0, 1000)])),
      parent: node.parentElement ? {tag: node.parentElement.tagName, id: node.parentElement.id, className: node.parentElement.className} : null,
      children: Array.from(node.children || []).slice(0, 10).map(child => ({tag: child.tagName, text: String(child.textContent || "").slice(0, 300), title: child.getAttribute("title"), ariaLabel: child.getAttribute("aria-label")})),
      visible: visible(node)
    });
    return {label, url: location.href, readyState: document.readyState,
      candidateCount: candidates.length, omitted: Math.max(0, candidates.length - 100),
      candidates: candidates.slice(0, 100).map(describe)};
  }

  async function openRunLabel(run) {
    if (blockerPresent()) return attention("Google Scholar is asking for human verification. Complete it, then click Resume.");
    const reference = run.labelReference;
    let destination = reference && reference.name === run.label
      ? runLabelDestination(reference.href) : null;
    // labelUrl was already stored by older builds only after discovery.
    if (!reference && run.labelUrl) destination = runLabelDestination(run.labelUrl);
    if (!destination) {
      const labelLink = await waitFor(() => findRunLabelLink(run.label));
      destination = labelLink && runLabelDestination(labelLink.href);
    }
    // A pause or owner-run change during sidebar rendering must stop navigation.
    const current = (await snapshot()).run;
    if (!current || current.status !== Core.STATUS.RUNNING || current.runId !== run.runId || current.label !== run.label) return;
    if (!destination) {
      return attention(`Could not uniquely identify the LitSync label '${run.label}' after waiting for My Library. See Recent diagnostics for the actual label controls. The checkpoint is preserved; click Resume after checking the original Scholar account.`, runLabelDiagnostics(run.label));
    }
    const labelUrl = destination.href;
    await patch({
      stage: Core.STAGE.VERIFYING_LABEL,
      labelUrl,
      labelReference: {name: run.label, scilib: destination.scilib, href: labelUrl},
      labelAuditIds: [],
      labelAuditPages: [],
      labelAuditComplete: false,
      labelVerifiedIds: [],
      labelVerifiedCount: null,
      labelMissingIds: [],
      labelUnexpectedIds: [],
      message: `Opening Scholar label ${run.label} for a full membership audit…`,
      lastUrl: location.href
    });
    await log("run_label_destination_resolved", {label: run.label, scilib: destination.scilib, labelUrl});
    // Navigate the persisted destination; the matching menu copy may be hidden.
    location.href = labelUrl;
  }

  async function exportLabel(run) {
    if (blockerPresent()) return attention("Google Scholar is asking for human verification. Complete it, then click Resume.");
    if (run.exportMode !== "partial" && !Core.labelIntegritySatisfied(run)) {
      return attention(`Refusing to export: ${Core.progress(run).saved}/${run.targetTotal} are checkpointed but only ${Core.uniqueStrings(run.labelVerifiedIds || []).length}/${run.targetTotal} have been verified inside the LitSync label.`);
    }

    const existingDialog = findExportArticlesDialog();
    if (existingDialog) return completeExportArticlesDialog(run, existingDialog);

    // Scope-first Scholar variant: the all-label scope was already confirmed,
    // so the next visible format choice is the final download action.
    if (run.exportScopeConfirmed && !run.exportFormatChosen) {
      const csv = await waitFor(() => findCsvFormatControl(document), 2500);
      if (csv) return clickCsvFormat(run, {finalAfterCsv: true});
    }

    // Format-first Scholar variant: CSV was selected and the scope dialog may
    // still be loading. Never reopen Export all and accidentally reset format.
    if (run.exportFormatChosen && !run.exportScopeConfirmed) {
      const dialog = await waitFor(findExportArticlesDialog, WAIT_TIMEOUT_MS);
      if (dialog) return completeExportArticlesDialog(run, dialog);
    }

    const exportAll = findVisibleByText(document, "Export all", "a,button");
    if (!exportAll) return attention("The Scholar label page loaded, but 'Export all' was not found. Scholar may have changed its Library layout.");
    await patch({
      stage: Core.STAGE.EXPORT_MENU,
      exportFormatChosen: false,
      exportScopeConfirmed: false,
      message: "Opening Scholar's Export all flow…",
      lastUrl: location.href
    });
    clickElement(exportAll);

    // Scholar has used both orderings over time/builds:
    //   Export all -> CSV -> scope dialog -> EXPORT
    //   Export all -> scope dialog -> EXPORT -> CSV
    // Detect whichever UI is actually visible instead of assuming one order.
    const nextStep = await waitFor(() => {
      const dialog = findExportArticlesDialog();
      if (dialog) return {kind: "scope", node: dialog};
      const csv = findCsvFormatControl(document);
      if (csv) return {kind: "format", node: csv};
      return null;
    }, WAIT_TIMEOUT_MS);

    if (!nextStep) return attention("Scholar's Export all flow opened, but neither the CSV format menu nor the export-scope dialog appeared.");
    const current = (await snapshot()).run || run;
    if (nextStep.kind === "scope") return completeExportArticlesDialog(current, nextStep.node);
    return clickCsvFormat(current, {finalAfterCsv: false});
  }

  async function recoverNonCsvExportPage(run, inspection = null) {
    const retryCount = Number(run.wrongExportRetries || 0) + 1;
    const body = document.body ? (document.body.textContent || document.body.innerText || "") : "";
    const sample = Core.normalizeWhitespace(body).slice(0, 240);
    await log("scholar_non_csv_export_page", {
      retryCount, url: location.href, sample,
      errors: inspection && inspection.errors || []
    }, "warn");

    if (retryCount > 2) {
      return attention(`Scholar repeatedly opened a non-CSV citation page during export. The ${Core.progress(run).saved} saved records remain intact; export was stopped to avoid a loop.`, {retryCount, sample});
    }

    await patch({
      stage: Core.STAGE.OPENING_LIBRARY,
      wrongExportRetries: retryCount,
      exportFormatChosen: false,
      exportScopeConfirmed: false,
      exportStartedAt: null,
      exportDownloadId: null,
      exportDownloadUrl: "",
      exportCsvVerified: false,
      exportCsvRowCount: null,
      exportCsvExpectedRows: Core.expectedExportRows(run),
      exportCsvHeader: [],
      exportCsvVerificationError: inspection && inspection.errors ? inspection.errors.join("; ") : "Wrong export format",
      message: "Scholar opened a non-CSV citation page. Returning to My Library and selecting the native visible CSV control…",
      lastUrl: location.href
    });
    location.replace("https://scholar.google.com/scholar?scilib=1&hl=en");
  }

  async function handleGoogleusercontentExportPage(run) {
    const body = document.body ? (document.body.textContent || document.body.innerText || "") : "";
    const expectedRows = Core.expectedExportRows(run);
    const inspection = Core.inspectScholarCsv(body, expectedRows);
    await log("scholar_rendered_export_page_inspected", {
      url: location.href,
      rowCount: inspection.rowCount,
      expectedRows,
      likelyCsv: inspection.likelyCsv,
      bibtexLike: inspection.bibtexLike,
      errors: inspection.errors
    }, inspection.ok ? "info" : "warn");

    if (inspection.ok) {
      const response = await chrome.runtime.sendMessage({
        type: "LSSC_RENDERED_CSV",
        text: body,
        pageUrl: location.href
      });
      if (!response || !response.ok) {
        return attention("Scholar rendered a valid CSV page, but the extension could not save the verified CSV.", {
          error: response && response.error || "Unknown background error"
        });
      }
      return;
    }

    if (inspection.bibtexLike) return recoverNonCsvExportPage(run, inspection);
    return attention("Scholar opened an export page, but its contents were neither a valid full CSV nor recognized BibTeX. No completion was recorded.", {
      expectedRows,
      observedRows: inspection.rowCount,
      errors: inspection.errors
    });
  }

  async function drive() {
    if (driverActive) return;
    driverActive = true;
    try {
      const snap = await snapshot();
      const run = snap.run;
      if (!run || run.status !== Core.STATUS.RUNNING) return;
      if (run.tabId != null && snap.senderTabId != null && run.tabId !== snap.senderTabId) return;
      if (isGoogleusercontentExportPage()) {
        if ([Core.STAGE.EXPORTING, Core.STAGE.EXPORT_MENU].includes(run.stage)) return handleGoogleusercontentExportPage(run);
        return attention("The active Scholar run navigated to a citation-export page unexpectedly. Click Resume to return to My Library.", {url: location.href, stage: run.stage});
      }
      if (blockerPresent()) return attention("Google Scholar is asking for human verification. Complete it in this tab, then click Resume.");

      if (run.stage === Core.STAGE.STARTING) return submitSearch(run);
      if (run.stage === Core.STAGE.RETRYING_FAILURES) return retryFailures(run);
      if (run.stage === Core.STAGE.INDEXING_LIBRARY_IDS) return indexLibraryIds(run);
      if (run.stage === Core.STAGE.VERIFYING_LABEL) return auditRunLabel(run);
      if (run.stage === Core.STAGE.REPAIRING_LABEL) return repairLabelMembership(run);
      if (Core.isScholarSearchUrl(location.href) && [Core.STAGE.SEARCH_LOADING, Core.STAGE.COLLECTING].includes(run.stage)) return collectPage(run);

      if (run.stage === Core.STAGE.OPENING_LIBRARY) {
        if (!Core.isScholarLibraryUrl(location.href)) {
          const link = Array.from(document.querySelectorAll("a")).find(a => Core.normalizeText(a.textContent) === "my library");
          if (link) clickElement(link);
          else location.href = "https://scholar.google.com/scholar?scilib=1&hl=en";
          return;
        }
        await patch({stage: Core.STAGE.OPENING_LABEL, message: "My Library opened. Selecting the LitSync label…", lastUrl: location.href});
        return openRunLabel((await snapshot()).run);
      }

      if (run.stage === Core.STAGE.OPENING_LABEL) return openRunLabel(run);
      if (run.stage === Core.STAGE.LABEL_LOADING) return exportLabel(run);
      if (run.stage === Core.STAGE.EXPORT_MENU) return exportLabel(run);
      if (run.stage === Core.STAGE.EXPORTING) {
        // v0.1.3 marked EXPORTING immediately after clicking CSV, before Scholar's
        // final 'Export articles' confirmation. Recover that durable checkpoint
        // instead of leaving an upgraded run permanently waiting for a download.
        if (!run.exportDownloadId) {
          const age = Date.now() - Number(run.exportStartedAt || 0);
          if (age >= 15000) {
            const dialog = findExportArticlesDialog();
            if (dialog) {
              const recovered = await patch({
                stage: Core.STAGE.EXPORT_MENU,
                message: "Resuming Scholar's final export confirmation…",
                lastUrl: location.href
              });
              return completeExportArticlesDialog(recovered, dialog);
            }
            if (Core.isScholarLibraryUrl(location.href)) {
              const recovered = await patch({
                stage: Core.STAGE.LABEL_LOADING,
                exportFormatChosen: false,
                exportScopeConfirmed: false,
                exportStartedAt: null,
                message: "No CSV download started. Reopening the label export flow…",
                lastUrl: location.href
              });
              return exportLabel(recovered);
            }
            const recovered = await patch({
              stage: Core.STAGE.OPENING_LIBRARY,
              message: "No CSV download started. Returning to My Library to retry export…",
              lastUrl: location.href
            });
            return goToLibrary(recovered, {partial: recovered.exportMode === "partial"});
          }
        }
        return;
      }

      if (Core.isScholarSearchUrl(location.href)) return collectPage(run);
      if (Core.isScholarLibraryUrl(location.href) && [Core.STAGE.OPENING_LIBRARY, Core.STAGE.OPENING_LABEL, Core.STAGE.VERIFYING_LABEL, Core.STAGE.LABEL_LOADING, Core.STAGE.EXPORT_MENU].includes(run.stage)) {
        await patch({stage: Core.STAGE.OPENING_LABEL, message: "Continuing export from My Library…", lastUrl: location.href});
        return openRunLabel((await snapshot()).run);
      }

      return attention("The active Scholar run is on an unexpected Scholar page. Return to the run's search results and click Resume.", {stage: run.stage, url: location.href});
    } catch (error) {
      try { await fail(`Scholar extension error: ${String(error && error.message || error)}`, {stack: String(error && error.stack || "")}); } catch (_ignored) {}
    } finally {
      driverActive = false;
    }
  }

  chrome.runtime.onMessage.addListener(message => {
    if (message && message.type === "LSSC_WAKE") drive();
  });
  window.addEventListener("pageshow", () => setTimeout(drive, 100));
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) setTimeout(drive, 100);
  });
  setTimeout(drive, 100);

  globalThis.__LSSC_TEST_HOOKS__ = Object.freeze({
    SELECTORS,
    parseTotalFromPage,
    rowAlreadySaved,
    findNextLink,
    findExportArticlesDialog,
    findCsvFormatControl,
    isGoogleusercontentExportPage,
    exportScopeControl,
    exportRequestFromControl,
    tryVerifiedExportRequest,
    radioSelected,
    currentScilib,
    onSpecificLibraryLabelPage,
    findRunLabelLink,
    runLabelDestination,
    runLabelDiagnostics,
    openRunLabel,
    repairOffsets,
    rowLibraryId,
    indexLibraryIds,
    auditRunLabel,
    repairLabelMembership,
    completeExportArticlesDialog,
    handleGoogleusercontentExportPage
  });
})();
