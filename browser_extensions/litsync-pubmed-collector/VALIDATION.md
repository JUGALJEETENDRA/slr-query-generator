# PubMed native collector validation — 2026-09-07

## Status

**Automated implementation tests: PASS (38). Installed-extension live acceptance: PASS.** The user subsequently ran the collector successfully and supplied its completion screenshot and native CSV. Earlier blocked attempts are recorded below as historical evidence.

Verified `C:\Users\Harshil\Downloads\csv-deeplearni-set (1).csv`: 351 data rows matched the visible PubMed total and popup count (351); 351 unique numeric PMIDs, no malformed rows or empty titles. SHA-256: `b1836c62d372fc99c9e8f90fc7cca76874ded06415b65005691662829e9315bc`. Two DOI fields are empty; native CSV has no abstract column.

## Live evidence

Inspected the actual signed-in PubMed tab and existing LitSync query card. Stable controls:

| Purpose | Live evidence |
| --- | --- |
| Query | Visible `input[type=search][name=term]`, ID `id_term`, label Search: |
| Search | Exact Search button in the input's form |
| Total | `.results-amount` with numeric span and results text |
| Cards | `article.full-docsum` |
| Save | Button Save; `aria-controls=save-action-panel` |
| Panel | `#save-action-panel[role=dialog]`, heading Save citations to file |
| Selection | Associated Selection: label; select `name=results-selection` |
| All results | Native option value `all-results` |
| Format | Associated Format: label; select `name=results-format` |
| CSV | Native option value `csv` |
| Download | Create file submit button scoped to native panel |

PubMed inserts a hidden temporary form containing a duplicate `id_term`. The helper filters visible inputs; the duplicate is reproduced in tests. Native data attributes expose warning thresholds, but no limit is hardcoded into the collector: visible warnings stop execution.

### Small native test

Query `(36246896[PMID] OR 29234807[PMID])` returned **2 results**. Save opened normally; All results and CSV were selected and their actual selected-option states were verified. Clicking Create file led to Edge's `ERR_BLOCKED_BY_CLIENT` page. No new CSV download was observed. Automation stopped without bypassing the browser block.

### Exact real-query retry requested by the user

After the user asked to retry, the exact current query was read from LitSync's PubMed card and entered into PubMed:

```text
("deep learning"[tiab] OR "neural networks"[tiab] OR "artificial intelligence"[tiab] OR "machine learning"[tiab]) AND ("early detection"[tiab] OR "early diagnosis"[tiab] OR "screening"[tiab] OR "diagnosis"[tiab]) AND ("diabetic retinopathy"[tiab] OR "diabetic eye disease"[tiab]) AND ("retinal images"[tiab] OR "fundus photographs"[tiab] OR "fundus images"[tiab] OR "retinal scans"[tiab])
```

RQ: How can deep learning improve the early detection of diabetic retinopathy from retinal images?

PubMed returned **578 results**, preserving the query in the search input and URL. Save → All results → CSV was configured and verified. Create file again led to **“pubmed.ncbi.nlm.nih.gov is blocked — ERR_BLOCKED_BY_CLIENT.”** No new CSV appeared in Downloads. The existing `csv-deeplearni-set.csv`, timestamp 14:02:37, predates this retry and is not evidence of its success.

No result checkboxes, article links or pagination controls were clicked in these tests. No metadata APIs or custom CSV generation were used. A medium successful export and installed-extension run remain unverified because the native export prerequisite failed.

## Automated coverage

38 tests pass. Coverage includes popup read-only opening, explicit Start, exact Balanced/High Recall strings, query insertion, result parsing/readiness, all native selectors and select states, hidden duplicate inputs, failed option activation, strict All results/CSV checks, restrictions/interstitials/login, exact-query protection, reload/resume, existing-open-panel reuse, duplicate Start/Create suppression, stale-tab recovery, Discard, download source/time/CSV completion and final complete state.

Actual content-controller fixture tests assert one Search, one Save and one Create file, or reuse an existing panel on Resume. Both assert zero checkbox/pagination clicks and disallow fetch. No CSV-content row verification is implemented, so the popup reports only available results and observed download completion.

## Regression and package checks

- New PubMed collector: **38 passed**.
- Existing IEEE collector: **31 passed**.
- Existing frozen Scholar collector: **102 passed**.
- Parent repository: **27 test files passed, 1,510 tests passed, one existing skip**.
- Runtime JavaScript syntax and MV3 manifest references/permissions: PASS.
- ZIP: 11 runtime/documentation files, member list and every byte verified, SHA-256 companion generated.
- `git diff --check`: PASS (existing line-ending warning only).
- SHA-256 comparison of 139 protected existing files: **zero changed**. Baseline saved at `artifacts/pubmed-native-validation-20260907/protected-before.json`.
- No existing product source files were modified. New extension folder: `browser_extensions/litsync-pubmed-collector/`. ZIP: `browser_extensions/litsync-pubmed-collector-v0.1.0.zip`.

Subsequent installed-extension validation succeeded as documented at the top of this report. The 351-row native CSV reconciled with PubMed and the completion popup. Resume and duplicate safety are covered by automated tests; a separate installed-extension interruption/resume trial was not recorded.
