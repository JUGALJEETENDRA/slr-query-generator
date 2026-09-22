(() => {
  "use strict";
  const C = LitSyncPubMed;
  const text = e => (e?.textContent || "").replace(/\s+/g, " ").trim();
  const visible = e => Boolean(e && !e.closest('[hidden],[inert],[aria-hidden="true"]') && e.getClientRects().length && getComputedStyle(e).visibility !== "hidden");
  function unique(nodes, name) {
    const items = [...nodes].filter(visible);
    if (items.length > 1) throw Error(`Ambiguous ${name}`);
    return items[0] || null;
  }
  const button = (root, label) => unique([...root.querySelectorAll('button,[role="button"]')].filter(e => text(e) === label || e.getAttribute("aria-label") === label), label);
  const queryInput = doc => unique(doc.querySelectorAll('input[type="search"][name="term"]'), "PubMed search field")
    || unique(doc.querySelectorAll('input#id_term'), "PubMed search field");
  function fill(input, value) {
    const win = input.ownerDocument.defaultView;
    Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, "value").set.call(input, value);
    input.dispatchEvent(new win.Event("input", {bubbles: true}));
    input.dispatchEvent(new win.Event("change", {bubbles: true}));
    if (input.value !== value) throw Error("Search field did not preserve the exact query");
  }
  function saveDialog(doc) {
    return unique([...doc.querySelectorAll('[role="dialog"],#save-action-panel')].filter(e => [...e.querySelectorAll('h2,[role="heading"]')].some(h => text(h) === "Save citations to file")), "Save citations dialog");
  }
  function selectControl(panel, label, name) {
    const associated = [...panel.querySelectorAll("label")].filter(e => text(e).replace(/:$/, "") === label)
      .map(e => e.control || (e.htmlFor && panel.querySelector(`[id="${e.htmlFor}"]`))).filter(Boolean);
    return unique(associated, label) || unique(panel.querySelectorAll(`select[name="${name}"]`), label);
  }
  const selection = panel => selectControl(panel, "Selection", "results-selection");
  const format = panel => selectControl(panel, "Format", "results-format");
  function option(control, label) {
    const options = [...(control?.options || [])].filter(e => text(e) === label && !e.disabled);
    if (options.length !== 1) throw Error(`PubMed does not offer ${label}; check the native restriction`);
    return options[0];
  }
  function choose(control, label) {
    if (!control || control.disabled) throw Error(`PubMed ${label} control is unavailable`);
    const selected = option(control, label);
    control.value = selected.value;
    control.dispatchEvent(new control.ownerDocument.defaultView.Event("change", {bubbles: true}));
    if (control.selectedOptions.length !== 1 || text(control.selectedOptions[0]) !== label) throw Error(`PubMed did not retain ${label}`);
  }
  function results(doc, query) {
    if (queryInput(doc)?.value !== query) return null;
    const counts = [...doc.querySelectorAll('.results-amount')].filter(visible).map(e => C.resultCount(text(e))).filter(n => n != null);
    if (!counts.length || counts.some(n => n !== counts[0])) return null;
    if (counts[0] === 0) throw Error("No PubMed results. No CSV was requested.");
    const cards = [...doc.querySelectorAll('article.full-docsum')].filter(visible);
    if (!cards.length || cards.length > counts[0] || !button(doc, "Save")) return null;
    return {total: counts[0]};
  }
  function interruption(doc, allowSave = false) {
    const body = doc.body?.innerText || "";
    if (/unusual traffic|access denied|session (?:has )?expired|verify (?:that )?you are human|captcha|temporarily unavailable|too many requests/i.test(body)
      || [...doc.querySelectorAll('input[type="password"],iframe[title*="challenge" i],iframe[title*="captcha" i]')].some(visible)) return "Action required in PubMed: login, access or verification interruption";
    const unexpected = [...doc.querySelectorAll('[role="dialog"]')].filter(visible).find(e => !(allowSave && e === saveDialog(doc)));
    if (unexpected) return `Action required in PubMed: unexpected dialog (${text(unexpected).slice(0,120)})`;
    const warning = [...doc.querySelectorAll('[role="alert"],.selection-validation-message')].find(e => visible(e) && text(e));
    if (warning) return `Action required in PubMed: ${text(warning).slice(0,250)}`;
    const panel = saveDialog(doc);
    if (panel && /(?:maximum|limited to|only the first|cannot (?:save|export)|exceed|limit of)/i.test(text(panel))) return `Action required in PubMed: ${text(panel).slice(0,300)}`;
    return null;
  }
  function verifyExport(doc, query, expected) {
    const reason = interruption(doc, true); if (reason) throw Error(reason);
    const panel = saveDialog(doc);
    if (!panel) throw Error("Native Save citations to file panel closed");
    if (results(doc, query)?.total !== expected) throw Error("PubMed results changed before export");
    for (const [control, label] of [[selection(panel), "All results"], [format(panel), "CSV"]]) {
      if (!control || text(control.selectedOptions[0]) !== label) throw Error(`PubMed export is not set to ${label}`);
    }
    const create = button(panel, "Create file");
    if (!create || create.disabled) throw Error("PubMed Create file is unavailable");
    return create;
  }
  C.dom = {text, visible, button, queryInput, fill, saveDialog, selection, format, option, choose, results, interruption, verifyExport};
})();
