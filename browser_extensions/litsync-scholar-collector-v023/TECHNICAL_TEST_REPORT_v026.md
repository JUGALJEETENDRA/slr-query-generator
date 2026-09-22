# LitSync Scholar Collector 0.2.6 — live DOM fix and in-place update

## Proven root cause (Edge, 2026-09-05)

Read-only DevTools inspection on the existing signed-in My Library document established:

- Current label name supplied by the preserved run: `litsync_20260905_d2ee15f2_141237814`.
- Both live controls have tag `A` and raw text `litsync_20260905_d2ee15f2_1412378...`.
- Both have href `/scholar?scilib=1030&hl=en&as_sdt=0,5`.
- Responsive menu copy: `class="gs_md_li" role="menuitemradio" tabindex="-1"`, under `DIV.gs_res_ab_dd_sec`, under `DIV.gs_res_ab_dd_bdy`.
- Sidebar copy: only the href attribute, under `LI.gs_ind`, under `UL.gs_bdy_sb_sec`.
- Both contain an empty `SPAN.gs_notf`. Neither control exposes an id, title, aria-label, or data-label attribute. The stable navigation identifier is the href's `scilib=1030`.
- The document was fully loaded (`readyState: complete`). Responsive layout hides sidebar controls visually but keeps both copies in the DOM.

Reproduction of the existing finder on the live DOM returned:

```json
{"exact":0,"prefixMatches":2,"result":"null"}
```

`findRunLabelLink()` counted matching DOM elements instead of distinct Scholar label destinations. The truncated text cannot exactly match the stored full name. Its two prefix matches refer to one label, but `matches.length === 1` fails, so `openRunLabel()` emits the misleading not-visible message. The earlier leaf-only inspection missed these anchors because each has an empty child span; the subsequent anchor and attribute inspection established the actual structure.

The actual replacement finder, run in an isolated read-only DevTools expression against the same live document, returned:

```json
{"fixedFinderFound":true,"text":"litsync_20260905_d2ee15f2_1412378...","destination":{"scilib":"1030","href":"https://scholar.google.com/scholar?scilib=1030&hl=en"}}
```

No live navigation, extension reload, Resume, collection, label creation, or storage writes were performed. The real 58/58 label audit and CSV export remain to be run after updating. COMPLETE is not claimed.

## Exact behavior changed

Only the run-label discovery/opening block and test hooks in `src/scholar.js` changed at runtime.

- Matching copies are grouped by validated Scholar label destination; different destinations remain ambiguous, even with identical full names.
- Existing whitespace, zero-width character and legacy truncation support is retained. A longer different full name is no longer accepted by reverse-prefix matching. Full descendant accessible attributes can identify a clipped control.
- Name-only checkpoints wait up to the existing 10-second timeout for label controls.
- On first discovery, `labelReference: {name, scilib, href}` and the existing `labelUrl` are checkpointed before navigating. Later resumes use the stable destination. Older `labelUrl` checkpoints migrate in place.
- Navigation starts a full, unfiltered first-page audit using the resolved href, including when the duplicate menu control is hidden. No hardcoded run-specific ID is present in runtime code.
- Pausing or changing the run while waiting stops navigation and failure-state updates.
- Failed discovery logs candidate tag, normalized/raw text, href, role, aria-label, title, class, id, data attributes, parent and child summaries, visibility, URL and document readiness. Output is bounded to 100 candidates, with an omitted count and bounded text fields; no page HTML dump.

Save logic, query generation, pagination, throttling, result reconciliation, ID mapping, audit logic, CSV/export logic, storage keys, permissions and extension identity are unchanged. No uninstall or storage migration is required. The manifest version alone changes from 0.2.5 to 0.2.6; package metadata is synchronized to 0.2.6.

## Exact extension files changed/added

- `src/scholar.js`
- `manifest.json`
- `package.json`
- `tests/label-integrity.test.cjs`
- `tests/background.test.cjs`
- `tests/test_static_extension.py`
- `tests/test_fixture_contract.py`
- `tests/fixtures/library_labels_live_20260905.html` (new bounded live DOM fixture)
- `TECHNICAL_TEST_REPORT_v026.md` (this file)

Existing ambiguity test fixtures now assign distinct default hrefs to distinct labels. Previously the mock assigned the same href to every anchor, which failed to model the distinction between two labels and two copies of one label. Their rejection expectations remain intact; a separate regression tests the actual same-href duplicate case.

## Verification

Before changes: 87/87 Node tests, 15/15 Python tests, 5/5 runtime JS syntax checks.
After changes: 102/102 Node tests, 16/16 Python tests, 5/5 runtime JS syntax checks.
15 additional Node regressions cover duplicate destinations, distinct-destination ambiguity, exact current selection among old labels, nested text/accessible attributes, longer-name rejection, URL validation, rendering delay, persisted references, legacy labelUrl migration, missing-label diagnostics, pause handling, and the preserved 58/58 background Resume path. Existing truncation/whitespace/zero-width and strict audit/CSV tests remain passing.

The in-place ZIP is built with `manifest.json` at its root, without the old accidental nested extension copy or Python caches. Extraction smoke verification checks every archived byte against the source, manifest resource paths, version, unchanged identity/permissions, and JavaScript syntax in the extracted package. The full Node and Python suites are also run from the extracted copy.

## Manual update and verification

ZIP: `C:\Users\Harshil\litsync-extension-lab\litsync-site-integration\browser_extensions\litsync-scholar-collector-v026-IN-PLACE.zip`

1. Extract the ZIP contents directly over `C:\Users\Harshil\litsync-extension-lab\litsync-site-integration\browser_extensions\litsync-scholar-collector-v023` and replace existing files. The manifest must be immediately inside that folder, not a new nested folder.
2. At `edge://extensions`, click Reload on the existing LitSync Scholar Collector entry. Do not remove/uninstall it or load a second copy. Confirm version 0.2.6.
3. Reload the existing Scholar My Library tab so it receives the new content script.
4. Open the popup. Confirm Target 58, Saved 58 and Unresolved 0. Click Resume, not Start or Reset.
5. Resume should resolve scilib 1030, persist it, and audit the existing label. Wait for Verified in label 58/58.
6. Continue the existing export flow and downloaded-CSV verification. COMPLETE is valid only when Verified CSV rows is also 58/58 and all existing integrity gates pass.
7. If it stops, copy Recent diagnostics. A discovery failure now records what the actual DOM exposed rather than only saying the label is not visible.
