(() => {
  "use strict";
  const C = globalThis.LitSyncIEEE;
  const text = e => (e?.textContent || "").replace(/\s+/g, " ").trim();
  const visible = e => Boolean(e && !e.closest('[hidden],[inert],[aria-hidden="true"]') && e.getClientRects().length && getComputedStyle(e).visibility !== "hidden");
  function unique(items, label) {
    const found = [...items].filter(visible);
    if (found.length > 1) throw Error(`Ambiguous ${label}`);
    return found[0] || null;
  }
  const button = (root, name) => unique([...root.querySelectorAll('button,[role="button"]')].filter(e => text(e) === name || e.getAttribute("aria-label") === name), name);
  const dialogs = doc => [...doc.querySelectorAll('[role="dialog"],ngb-modal-window')].filter(visible);
  function exportDialog(doc) {
    return unique(dialogs(doc).filter(e => [...e.querySelectorAll('h1,h2,[role="heading"]')].some(h => text(h) === "Download Results")), "Download Results dialog");
  }
  function interruption(doc, allowExport = false) {
    const body = doc.body?.innerText || "";
    if (dialogs(doc).some(e => [...e.querySelectorAll('input[type="password"]')].some(visible))) return "Action required in IEEE: sign in manually in the export dialog, then Resume";
    if (/session (?:has )?expired|unusual traffic|verify (?:that )?you are human|access denied|captcha|sign in to continue|log in to continue/i.test(body)
      || [...doc.querySelectorAll('iframe[title*="challenge" i],iframe[title*="captcha" i]')].some(visible)
      || (!C.range(doc.body?.textContent) && !command(doc) && [...doc.querySelectorAll('input[type="password"]')].some(visible))) return "Action required in IEEE: login, access or verification interruption";
    const modal = dialogs(doc).find(e => !(allowExport && e === exportDialog(doc)));
    if (modal) return `Action required in IEEE: close the unexpected dialog (${text(modal).slice(0,100)})`;
    return null;
  }
  function selected(doc) {
    // Observation only. There is deliberately no function that clicks a checkbox.
    return [...doc.querySelectorAll('xpl-results-item input[type="checkbox"],input[aria-label="Select search result"],label.results-actions-selectall input')].some(e => e.checked || e.indeterminate);
  }
  function results(doc) {
    const modal = exportDialog(doc);
    // ng-bootstrap sets LayoutWrapper aria-hidden while Export is open. Its rendered
    // result count remains valid read-only evidence; never click a covered control.
    const headings = [...doc.querySelectorAll('h1,[role="heading"][aria-level="1"]')].filter(e => visible(e)
      || (modal && !modal.contains(e) && e.getClientRects().length && getComputedStyle(e).visibility !== "hidden"));
    const r = headings.map(e => C.range(text(e))).find(Boolean);
    if (!r) return null;
    const cards = [...doc.querySelectorAll("xpl-results-item")].filter(e => e.querySelector('h3 a[href*="/document/"]'));
    if (cards.length !== r.last - r.first + 1 || (!modal && !button(doc, "Export"))) return null;
    return r;
  }
  function command(doc) {
    return unique(doc.querySelectorAll('textarea[aria-label="Enter Search Text"]'), "Command Search field")
      || unique(doc.querySelectorAll('textarea#cmdTextArea[name="queryText"]'), "Command Search field");
  }
  function fill(input, query) {
    Object.getOwnPropertyDescriptor(input.ownerDocument.defaultView.HTMLTextAreaElement.prototype, "value").set.call(input, query);
    input.dispatchEvent(new input.ownerDocument.defaultView.Event("input", {bubbles: true}));
    input.dispatchEvent(new input.ownerDocument.defaultView.Event("change", {bubbles: true}));
    if (input.value !== query) throw Error("IEEE field changed the exact query");
  }
  function verifyDialog(doc) {
    const modal = exportDialog(doc);
    if (!modal) return null;
    if (selected(doc)) throw Error("Results are selected in IEEE. Clear the selection manually, then Resume.");
    const r = results(doc);
    if (!r || !C.nativeMode(text(modal), r.total)) throw Error("IEEE did not confirm the expected no-selection CSV export count (up to 1,000 results)");
    const download = button(modal, "Download");
    if (!download || download.disabled) return null;
    return download;
  }
  C.dom = {text, visible, button, exportDialog, interruption, selected, results, command, fill, verifyDialog};
})();
