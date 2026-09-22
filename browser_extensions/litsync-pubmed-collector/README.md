# LitSync PubMed Collector 0.1.0

Standalone Edge/Chrome MV3 extension. The only collection workflow is:

**Exact LitSync query → PubMed Search → Save → All results → CSV → Create file.**

PubMed creates the CSV. There are no metadata/API requests, article visits, page loops, translators, CSV construction.

## Install

In Edge open `edge://extensions`, enable Developer mode and select **Load unpacked**:

`C:\Users\Harshil\litsync-extension-lab\litsync-site-integration\browser_extensions\litsync-pubmed-collector`

For Chrome use `chrome://extensions`. Refresh both LitSync and PubMed after loading. ZIP users should extract into a new dedicated folder, then load the extracted folder containing manifest.json. Do not replace the Scholar or IEEE extension directories.

## First use

1. On local LitSync, generate/display the desired query versions.
2. Open https://pubmed.ncbi.nlm.nih.gov/ in a normal browser tab.
3. Open **LitSync PubMed Collector**, select Balanced or High Recall and verify the displayed exact query.
4. Click **Start collection** explicitly. Opening the popup only reads state.
5. The collector submits Search, verifies the query, result count and rendered result cards, opens Save, sets and verifies All results and CSV, then clicks Create file once.
6. Completion requires a matching completed CSV browser download. The popup shows results available; it does not claim a verified CSV row count.

First validate with a small LitSync query. If PubMed or Edge blocks export, resolve that condition manually. Do not bypass it. See VALIDATION.md for the successful installed-extension run: 351 native CSV records matched the PubMed result count and completion popup.

## Query sync

`src/bridge.js` receives the existing `LITSYNC_QUERY_CONTEXT` schema on localhost/127.0.0.1. It accepts `query_versions.balanced.pubmed` and `query_versions.high_recall.pubmed` unchanged, using the existing probe/ready handshake. The worker validates sender origin and frame. No LitSync source or query-generation changes are necessary. Shared message format does not imply shared runtime state.

Each job stores the literal query, variant, SHA-256 fingerprint and supplied LitSync fingerprint. The results URL's decoded `term` and visible search field must match the exact query. No normalization of field tags, parentheses or query terms is performed.

## Architecture and files

- `manifest.json`: independent MV3 manifest, storage/tabs/downloads/alarms permissions, host access limited to PubMed and local LitSync.
- `src/core.js`: pure query, result-count, job and download-correlation helpers.
- `src/bridge.js`: local-site query context receiver.
- `src/background.js`: serialized durable job transitions, popup authorization, owner guard, restart/closed-tab recovery and browser download observation.
- `src/dom.js`: centralized live-DOM semantic controls, native select option verification and interruption checks.
- `src/content.js`: one document controller; only Search, Save and Create file are clicked.
- `popup/popup.html`, `popup/popup.css`, `popup/popup.js`: simple variants/query/status/Start/Resume/Stop/Discard UI and collapsed logs.
- `tests/helpers.cjs`, `tests/collector.test.cjs`, `tests/fixtures/search.html`, `tests/fixtures/save.html`: simulated MV3 environment and reduced fixtures from the real PubMed DOM.
- `scripts/validate.cjs`, `scripts/package.py`, `package.json`: independent development and packaging commands.
- `README.md`, `VALIDATION.md`: usage and evidence.

## Checkpoint and duplicate safety

Own storage key: `litsync_pubmed_native_v1`, inside this extension's storage.

Stages: submitting_query, waiting_for_results, results_ready, opening_save, waiting_for_save_dialog, setting_all_results, setting_csv, creating_file, downloading, complete, paused.

The worker serializes mutations and accepts Start only from the extension popup. An incomplete job blocks a second Start. Document ownership and a content-script guard suppress duplicate execution. Full document navigation continues only a previously authorized waiting-results job. Browser startup or replacement of the same document's controller pauses for explicit Resume.

Before Create file, Resume reuses matching results or restores the recorded URL in a new tab when necessary. An already-open Save panel is reused without a second Save click. Native selections are set and verified again safely. Discard removes only this checkpoint; it never deletes files or changes another extension's state.

## Downloads and limitations

Download intent is persisted before Create file. A matching download must have PubMed source URL/blob origin/referrer evidence and a start time within the attempt window. Completion requires browser state `complete` plus CSV filename/MIME evidence. The download ID and filename are retained. A 30-second alarm checks missing initiation.

Once Create file may have been clicked, Resume only checks for the existing download; it never submits Create file again. Missing/interrupted/ambiguous downloads pause. Inspect browser Downloads and explicitly Discard before starting a fresh attempt. Avoid unrelated concurrent PubMed downloads because they can make correlation ambiguous.

The extension does not parse downloaded file contents or claim row equality. Final live acceptance must independently count native CSV data rows. PubMed warnings/restrictions are reported as seen; there is no assumed export limit and no API/scraping fallback. Empty results or a query redirected away from the normal results page stop safely.

## Development and package

With Node, Python 3 and development dependencies installed:

```text
npm test
npm run validate
npm run package
```

No runtime dependencies or transpilation. ZIP output: `../litsync-pubmed-collector-v0.1.0.zip`, with a SHA-256 companion. Packaging verifies all included bytes and excludes tests/dev dependencies. Existing collectors are not modified.
