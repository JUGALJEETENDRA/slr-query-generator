# Validation — 2026-09-07

## Automated

- Standalone collector: **31 tests passed**, zero failed. Tests run the actual core, DOM helpers, background, popup and content controller using reduced live IEEE fixtures and a simulated MV3 API.
- Workflow tests for 1, 75 and 2,739 results assert **zero selection or pagination clicks**, one Search click, one native Download click and observed completion. Duplicate content-script injection is included.
- Other checks cover exact Balanced/High Recall routing, explicit Start, side-effect-free popup, strict native export cap, selected-mode rejection, login/CAPTCHA/unexpected modal handling, full navigation, reload owner replacement, matching-tab resume, stale tab restoration, duplicate Start/download suppression, missing/interrupted/non-CSV download handling, origin checks and Discard.
- Existing parent repository: **27 test files passed; 1,510 tests passed, one skipped**. Other runtime files were not changed by this implementation.
- Frozen Scholar extension: **102 tests passed**, zero failed; no source changes.
- Runtime JavaScript syntax and manifest references/permission scope passed.
- `git diff --check` passed in both the site integration repository and parent repository (existing line-ending warnings only).

## Real Edge native workflow

These tests were performed through Computer Use on the actual IEEE website, in order. They validate IEEE's native UI, not by themselves the installed extension's popup/service-worker integration.

| Test | Available results | Native dialog upper bound | Downloaded CSV data rows | File |
| --- | ---: | ---: | ---: | --- |
| Exact healthcare article title | 1 | 1 | 1 | `C:\Users\Harshil\Downloads\export2026.09.06-16.16.46.csv` |
| Federated learning AND intrusion detection, Document Title fields | 317 | 317 | 317 | `C:\Users\Harshil\Downloads\export2026.09.06-16.21.31.csv` |
| Exact currently displayed LitSync IEEE query | 2,587 | 1,000 | 1,000 | `C:\Users\Harshil\Downloads\export2026.09.06-16.25.03.csv` |

All three used Command Search → Export → Download. **Zero result-selection clicks and zero pagination clicks occurred in these three tests.** No PDFs or article-detail pages were opened. Native downloaded files were counted with PowerShell Import-Csv, so embedded CSV newlines do not inflate counts. The new extension itself does not parse downloaded CSV contents.

Test 1 initially encountered a native sign-in form inside Download Results. Automation stopped; the user signed in manually; the test then continued normally. No authentication workaround was used.

The current LitSync query returns 2,587, unlike the older 2,739-result query from the earlier investigation. The current visible query was read verbatim from LitSync and submitted unchanged; the difference was not hidden or normalized away.

## Live discoveries incorporated

- `textarea[aria-label="Enter Search Text"]`, with fallback `textarea#cmdTextArea[name=queryText]`, belongs to `form[name=command-search-form]`. Search is scoped to that form.
- IEEE adds an outer parenthesis pair to the submitted query in the resulting URL. The input string is not changed by the collector.
- One result uses `Showing 1 of 1 result`; ordinary pages use `Showing 1-25 of N results`.
- Native no-selection export says **up to 1**, **up to 317**, or **up to 1,000** according to total. Verification compares this number to min(total, 1000).
- Narrow/mobile layout hides Export. Tests use the desktop-width tab. A missing Export control safely times out.
- Authentication may be demanded inside the export dialog, even when results are visible.
- Native ng-bootstrap Export sets `aria-hidden=true` on `#LayoutWrapper`. The rendered result summary remains valid read-only evidence behind the dialog. A focused regression covers this; covered Export controls remain non-actionable.

## Installed-extension validation

The user loaded the standalone unpacked extension and clicked Start. The extension submitted the newly synced query, reached **2,727 results**, opened native Export automatically, and left **zero checked results**. It stopped before Download because the modal hid the background result summary from accessibility. This real issue was fixed in the centralized read-only result helper and reproduced in the regression fixtures.

The user then reloaded the extension, refreshed the same results page, and resumed the preserved checkpoint. **The installed collector automatically generated and downloaded `C:\Users\Harshil\Downloads\export2026.09.06-16.31.24.csv`: 2,543,870 bytes and 1,000 verified CSV data rows.** Computer Use did not click Export or Download during this installed-extension run. The website remained on the same 2,727-result search with zero checked result boxes. The resulting URL query was compared with LitSync's visible exact IEEE query and matched, allowing only IEEE's observed outer wrapper.

The user operates the toolbar because Computer Use cannot operate the protected Edge extension-management page or toolbar popup. The user confirmed the final popup status: **Complete — IEEE CSV downloaded**. Actual installed-extension download and the 1,000-row file contents were independently verified. This completes the installed-extension Start, interruption/reload/Resume, native export, download, and completion-display validation.

## Deliberate limits

- No CSV-content inspection in the extension; browser completion plus CSV filename/MIME is required. The popup does not claim exactly 1,000 data rows.
- No split queries, pagination, selection management, custom metadata collection or CSV construction.
- Download matching requires IEEE source/referrer evidence and a bounded start time. A concurrent unrelated IEEE download can be ambiguous; the collector stops rather than automatically retrying.
- After durable Download intent, Resume only checks for a download. If it remains unconfirmed, inspect Downloads and explicitly Discard before a fresh attempt.
- Query sync uses the current localhost/127.0.0.1 handshake. Remote LitSync origins are not enabled.

No existing source file was modified. All new runtime, tests and documentation are confined to `browser_extensions/litsync-ieee-collector/`; package artifacts live alongside that directory. Frozen Scholar runtime/storage were not modified.
