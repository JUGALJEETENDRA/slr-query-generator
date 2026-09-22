# LitSync IEEE Collector 0.1.0

Standalone Edge/Chrome Manifest V3 extension. It submits an exact LitSync query through IEEE Command Search, waits for rendered results, opens IEEE Export, verifies native **no-selection** CSV mode, and clicks Download once. It does not click result checkboxes, paginate, scrape metadata, construct CSV, change sort,.

## Install and first test

1. In Edge, open `edge://extensions`, enable Developer mode, and choose **Load unpacked**.
2. Select this exact folder (the one containing manifest.json):

   `C:\Users\Harshil\litsync-extension-lab\litsync-site-integration\browser_extensions\litsync-ieee-collector`

3. Reload the LitSync local website and IEEE Command Search after installation so their content scripts are present. Generate/display the queries on LitSync. No Scholar files or checkpoint changes are needed.
4. Sign into IEEE normally. Open `https://ieeexplore.ieee.org/search/advanced/command` in a desktop-width window (IEEE hides Export in its narrow/mobile layout).
5. Open **LitSync IEEE Collector**. Verify Query synced, choose Balanced or High Recall, and inspect the exact query. Opening this popup does not start collection.
6. For the first test, use a LitSync-generated query with a small result count. Click **Start collection**. Leave the tab open. Expect native search, Export, then Download with **zero selection clicks**.
7. If IEEE requests authentication or presents an unexpected dialog, handle it manually and click **Resume**. If results were already selected, clear them yourself before Resume; the extension never toggles them.
8. Confirm the popup says **Complete — IEEE CSV downloaded**, and check the browser Downloads list. The popup reports available results and the requested upper bound, not a verified CSV row count.

This installation has separate storage. ZIP users should extract the ZIP to its own folder and load that folder.

## Architecture and files

- `manifest.json`: independent MV3 registration; storage, tabs, downloads, and alarms. Host access only IEEE and local LitSync origins.
- `src/core.js`: exact-query validation, fingerprints, result parsing, native export cap verification and download correlation.
- `src/bridge.js`: receives the existing local-origin `LITSYNC_QUERY_CONTEXT` handshake. Keeps exact `query_versions.balanced.ieee_xplore` and `query_versions.high_recall.ieee_xplore` strings. The message schema is shared by the standalone collectors. No website change is needed. Sync never starts a run.
- `src/background.js`: serialized job mutations, explicit popup authorization, document ownership, durable state, restart recovery, native download observation.
- `src/dom.js`: centralized semantic selectors based on the live IEEE DOM. Includes native Command Search, result heading/cards, Export, Download Results, and scoped Download. Selection controls are only read to reject selected-mode export.
- `src/content.js`: one guarded controller per document. Its only click sites are Search, Export, and Download.
- `popup/*`: query variants, exact query, status, Resume/Stop/Discard, collapsed logs. No developer diagnostics in the main flow.
- `tests/*`: reduced live DOM fixtures and Node tests exercising the actual runtime scripts against a simulated MV3 API.
- `scripts/*`: syntax/manifest checks and byte-verified ZIP packaging.

## State and resume

Storage uses only `chrome.storage.local.litsync_ieee_native_v1` in this extension's own storage. Job data includes UUID, exact query, query version, SHA-256 of exact query, supplied LitSync fingerprint, tab/document owner, result count, result URL, visible sort text, stage, timestamps, and download attempt/ID.

Stages: submitting_query → waiting_for_results → results_ready → opening_export → waiting_for_export_dialog → downloading → complete; interruptions pause with a reason. The service worker serializes changes. Duplicate Start is rejected while an incomplete checkpoint exists. Content-script reinjection has a document-local guard. Native full navigation may continue the already authorized waiting-results job. Browser startup or detected replacement of the owning script pauses for Resume.

Resume reuses an existing matching results tab; if closed or unrelated, it restores the recorded URL in a new tab. It does not rewrite the query or change sort parameters. A new Start requires IEEE Command Search. Discard only removes this extension's checkpoint; it does not change the website or delete downloaded files.

The exact submitted string is preserved. Live Command Search adds outer parentheses to the resulting URL; verification accepts either the exact string or that specific site-added wrapper, without rewriting the input.

## Native export and download verification

With no checked results, the native dialog must explicitly offer CSV and say `If no results are selected, up to N results will be included.` N must equal the smaller of the observed total and the internal `LIMIT = 1000`. This variable wording was confirmed live for a one-result search. Selected-mode text or a mismatching limit stops the job.

Download intent is saved before the native click. Downloads events/search correlate an IEEE URL, blob origin or referrer with a tight start-time window. Completion requires the browser's completed state plus CSV filename or MIME evidence. An alarm checks missing initiation. Resume after a download intent **only checks for the existing download**; it never clicks Download again. If unconfirmed/interrupted/ambiguous, inspect browser Downloads and explicitly Discard before starting a new attempt. Avoid concurrent manual IEEE CSV downloads during a run.

The extension does not read downloaded file contents. Native CSV row counts can be independently checked, but the popup never claims the requested cap is a verified row count. No sort control is changed; visible sort text is recorded when available.

## Development

Use Node with npm and Python 3. Install development dependencies with `npm install` when needed. There are no runtime dependencies or build transpilation.

```text
npm test
npm run validate
npm run package
```

The ZIP is written alongside this folder as `litsync-ieee-collector-v0.1.0.zip`, with a SHA-256 companion. Only manifest, runtime, popup and README are packaged; tests/dev dependencies are excluded.

See `VALIDATION.md` for the final automated and live validation record and any remaining limitations.
