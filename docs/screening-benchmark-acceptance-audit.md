# LitSync Screening Benchmark: Independent Pre-Fix Acceptance Audit

Audit date: 2026-07-28  
Audit phase: Phase 1 only; no remediation performed  
Repository: `C:\Users\Harshil\Downloads\newresearchtool`

## Executive conclusion

The subsystem's basic contracts, metrics, frozen-gold/run-population separation,
Phase 3A engine identity, and offline dependency boundary are substantially real.
The existing focused tests pass. However, the subsystem is not release-gate ready.

The audit proves four blocker classes:

1. An unvalidated benchmark version can escape the registry root, including to an
   absolute drive path.
2. Published reports have no completion manifest, hashes, or trustworthy completed
   state; a registry entry remains valid after its result is changed.
3. Publication is a sequence of per-file moves into an existing directory. It can
   mix evaluations or leave a partial output that has no authoritative completion
   boundary.
4. A failure after registry replacement can raise to the caller while leaving the
   failed publication registered and accepted.

Phase 2 must not start until these findings are reviewed.

## Environment

```text
Python 3.13.4
git version 2.49.0.windows.1
Windows PowerShell 5.1.26100.8894
Microsoft Windows NT 10.0.26200.0
```

## Initial worktree snapshot

Captured before audit tests and experiments:

```text
 M litsync_app/app.py
 M litsync_app/integrations/gemini_web_v24_prompt.py
 M litsync_app/integrations/gemini_web_v24_screening.py
 M litsync_app/prisma.py
 M litsync_app/screening/bulk.py
 M litsync_app/validation/gold.py
 M tests/unit/test_existing_website_integration.py
 M tests/unit/test_gemini_web_v24.py
 M tests/unit/test_gold_validation.py
 M web/slr_query_generator.html
?? delete_blockchain_cache.cmd
?? docs/
?? examples/
?? litsync_app/benchmarking/
?? tests/unit/test_benchmarking.py
```

Initial `git diff --stat`:

```text
 litsync_app/app.py                                 |  93 ++-
 litsync_app/integrations/gemini_web_v24_prompt.py  | 119 +++-
 .../integrations/gemini_web_v24_screening.py       | 483 +++++++++++++--
 litsync_app/prisma.py                              |   3 +
 litsync_app/screening/bulk.py                      |   3 +
 litsync_app/validation/gold.py                     | 200 ++++++-
 tests/unit/test_existing_website_integration.py    | 470 ++++++++++++++-
 tests/unit/test_gemini_web_v24.py                  | 649 ++++++++++++++++++++-
 tests/unit/test_gold_validation.py                 | 167 +++++-
 web/slr_query_generator.html                       | 173 +++++-
 10 files changed, 2239 insertions(+), 121 deletions(-)
```

The pre-existing line-ending warnings from Git are not audit changes.

## Existing focused tests

Executed before any retained audit file was created:

```powershell
python -m pytest tests/unit/test_benchmarking.py -q --basetemp .pytest_tmp_benchmark_audit
```

Result: exit `0`, `14 passed in 2.35s`.

```powershell
python -m pytest tests/unit/test_gemini_web_v24.py -q --basetemp .pytest_tmp_gemini_audit
```

Result: exit `0`, `59 passed in 1.90s`.

These tests are unchanged. `tests/unit/test_benchmarking.py:9,273,396,414,503`
invoke `main()` directly; they do not run the CLI as subprocesses. They contain no
publication-boundary failure injection or completion-marker coverage.

## BLOCKER findings

### B-1: benchmark version permits registry-root escape and arbitrary drive paths

Expected: benchmark IDs, versions, and derived registry paths remain beneath the
explicit registry root.

Observed:

- `BenchmarkSpec.benchmark_version` has only `min_length=1`
  (`litsync_app/benchmarking/contracts.py:67-76`).
- `_registry_path` directly interpolates it
  (`litsync_app/benchmarking/registry.py:14-15`).
- No post-resolution containment check exists.

Probe results:

```text
benchmark_version="../../escape"
accepted=True
resolved=C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\escape.json

benchmark_version="C:\escape"
accepted=True
resolved=C:\escape.json
```

Reproducible CLI case:

```powershell
python -m litsync_app.benchmarking.cli evaluate `
  --spec C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\pathprobe\benchmark-traversal.json `
  --job-id cold-job `
  --artifacts-root C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\base\outputs `
  --registry-dir C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\pathprobe\registry\deep `
  --output-dir C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\pathprobe\output `
  --json
```

Observed exit: `0`, verdict `PASS`. The registry was written outside the supplied
registry root at:

```text
C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\pathprobe\registry\escaped-registry.json
```

This is a filesystem write primitive controlled by specification content.

### B-2: no trustworthy completed-publication contract or artifact integrity

Expected: a completed result is established by a final completion marker that
records every artifact hash and byte size; registry readers validate it.

Observed:

- No `_COMPLETE.json` implementation exists anywhere in the package.
- Successful output contains only `benchmark-result.json`,
  `benchmark-report.html`, and `benchmark-errors.csv`.
- Registry v1 stores only `first_recorded_result`
  (`registry.py:57-68,125-136`).
- `check_registry` validates only the specification fingerprint
  (`registry.py:18-29`); it never checks that the result exists or is unchanged.

After a valid PASS publication, the audit appended `tampered` to the registered
`benchmark-result.json` and ran:

```powershell
python -m litsync_app.benchmarking.cli validate-spec `
  --spec C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\base\benchmark.json `
  --registry-dir "C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\eval registry" `
  --json
```

Observed exit: `0`, status `valid`. Registry validation silently accepted the
changed result. Missing-result behavior is equally unchecked by the same code path.

### B-3: output publication is non-transactional and permits mixed directories

Expected: publish one complete directory atomically, or idempotently accept only a
fully validated identical publication.

Observed:

- `publish_staged_report` creates/reuses the destination and moves files one at a
  time (`report.py:22-42`).
- Registered publication repeats the same per-file loop
  (`registry.py:105-123`).
- Existing contents are neither rejected nor reconciled.

The audit placed `older-evaluation.txt` in a destination, then evaluated a PASS
run into it. Exit was `0`; the final directory contained:

```text
benchmark-errors.csv
benchmark-report.html
benchmark-result.json
older-evaluation.txt
```

Injected failure on the second artifact move produced:

```text
failure=OSError('injected after one artifact')
output_files=.benchmark-publication-incomplete.json,benchmark-result.json
registry_exists=False
staged_remaining=benchmark-errors.csv,benchmark-report.html
```

There is no reader capable of distinguishing this state from user-visible output,
and no final marker defines completion.

### B-4: a failed publication can permanently register and lock an evaluation

Expected: failed publication does not register a benchmark identity.

Observed ordering is: publish files, replace registry, delete the temporary
incomplete marker (`registry.py:117-139`). An injected exception from marker
deletion after registry replacement produced:

```text
failure=RuntimeError('injected after registry replacement')
registry_exists=True
marker_exists=True
check_registry=accepted
```

The caller observes failure, but the identity is registered and accepted.
Documentation says failed publications do not register
(`docs/screening-benchmark.md:70-75`), so the implementation directly contradicts
the documented rule.

## HIGH findings

### H-1: registry discards all evaluation history after the first result

Two distinct valid cold PASS jobs were published using the same benchmark identity
and registry. The second command returned exit `0`, but the registry remained:

```json
{
  "registry_schema_version": "litsync-benchmark-registry-v1",
  "benchmark_id": "domain-neutral-fixture",
  "benchmark_version": "1.0.0",
  "benchmark_spec_fingerprint": "611d23593c26cd880a5e78e4381bd101158dafa4e9c2bdc3bb54157486fd0b40",
  "first_recorded_result": "...\\pass output with spaces\\benchmark-result.json"
}
```

`registry.py:124-136` writes only when the file does not exist. No append-only
logical collection, publication kind, job IDs, verdict, marker path, or marker
hash is preserved.

### H-2: every report CSV is vulnerable to spreadsheet-formula injection

`csv.DictWriter.writerows` serializes values unchanged in all result/comparison
paths (`report.py:138-145,170-176,239-242`). A hostile title was emitted as:

```csv
0,KEEP,KEEP,FRESH,"  =2+2,quoted"
```

The first non-whitespace character is `=`. JSON preservation is correct, but CSV
has no spreadsheet-safety transform.

### H-3: structurally malformed evidence crashes the CLI with exit 1

The loader verifies only that `Evidence_JSON` is a list
(`loader.py:267-272`). The evaluator assumes every member is a mapping and calls
`span.get` (`evaluator.py:139-158`).

With `Evidence_JSON=["not-an-object"]`, real subprocess evaluation produced exit
`1`, no JSON result, and this traceback:

```text
AttributeError: 'str' object has no attribute 'get'
```

This violates fail-closed provenance and the documented exit-code/JSON contract.

### H-4: contradictory missing-abstract provenance can receive COLD PASS

The loader reconciles resumed, cache, and fresh counts, but never compares
`missing_abstract_count` with `missing_abstract_source_row_ids`
(`loader.py:341-353`).

Audit fixture:

```text
missing_abstract_count=99
missing_abstract_source_row_ids=[]
fresh_primary_papers=5
fresh_primary_source_row_ids=[0,1,2,3,4]
```

Real CLI with `--enforce-gate` returned exit `0`, provenance `COLD`, verdict
`PASS`. This disproves the documentation's claim that exact missing-abstract
counts and partitions are reconciled (`docs/screening-benchmark.md:57-60`).

### H-5: invalid cross-dataset comparison still calculates quality deltas

`compare_results` identifies different source datasets and screened populations
(`comparison.py:130-141`), but the condition suppressing pairwise calculations
omits both reasons (`comparison.py:142-152`).

With a candidate summary carrying a different dataset fingerprint:

```json
{
  "valid": false,
  "reasons": [
    "invalid provenance prevents improvement claims",
    "runs use different source datasets"
  ],
  "metric_delta_count": 32,
  "pairwise_count": 1
}
```

CLI exited `2`, but still published those deltas. Quality comparison across an
invalid/different dataset must not be calculated.

### H-6: untrusted path and identifier validation is incomplete

`_safe_job_id` strips input before validation and accepts Windows-reserved or
ambiguous names (`loader.py:146-150`):

```text
"CON" -> accepted
"CON.txt" -> accepted
"." -> accepted
".." -> accepted
"name." -> accepted
" name" -> normalized and accepted as "name"
"name " -> normalized and accepted as "name"
```

Separators, drive prefixes, colons, absolute Unix paths, and control characters
were correctly rejected.

Benchmark IDs accept `CON` and trailing-dot names. Benchmark versions have no
allowlist or length limit. Gold files may be absolute, and `../base/gold.csv`
validated successfully because `load_gold` resolves without containment
(`loader.py:88-100`).

### H-7: the CLI is not runnable from another directory without external setup

From a Windows path containing spaces outside the repository root:

```powershell
python -m litsync_app.benchmarking.cli validate-spec --spec <absolute-path> --json
```

Observed exit `1`:

```text
ModuleNotFoundError: No module named 'litsync_app'
```

The repository has no `pyproject.toml`, `setup.py`, or `setup.cfg`. Setting
`PYTHONPATH` to the repository made the same command return exit `0`, proving path
arguments work but standalone module discovery does not.

### H-8: `--json` does not cover argparse usage failures

Command:

```powershell
python -m litsync_app.benchmarking.cli validate-spec --json
```

Observed:

```json
{"exit": 2, "stdout": "", "stderr": "usage: ... the following arguments are required: --spec"}
```

Normal and handled-invalid commands emit clean JSON, but a command explicitly
using `--json` cannot parse stdout for invalid usage.

### H-9: the CLI cannot express two benchmark identities or two resolved-gold populations

`compare` accepts one `--spec`, loads one gold file once, and applies both to every
job (`cli.py:107-121`). Therefore the requested subprocess cases for incompatible
benchmark identity and different resolved-gold populations cannot be constructed
through the public CLI.

The pure comparison function correctly rejected a directly constructed
different-resolved-population pair and suppressed deltas:

```text
valid=False
reasons=['runs use different resolved-gold populations']
metric_deltas=0
pairwise=0
```

This is an unsupported CLI requirement, not a defect in that pure comparison
guard.

## MEDIUM findings

### M-1: source-row IDs are not fully string preserving

Although CSV is read with `dtype=str`, `source_row_id` converts strings ending in
`.0` through floating-point parsing (`provenance.py:26-33`):

```text
"001"  -> "001"
"1.0"  -> "1"
"01.0" -> "1"
"1.00" -> "1.00"
```

This can conflate distinct source identifiers and is inconsistent across decimal
spellings.

### M-2: failed lock acquisition leaks staged directories

An existing lock correctly caused exit `2` and was not removed. However, because
CLI staging occurs before lock acquisition, the failure left:

```text
.locked-output.benchmark-stage-79537e1d89534506a8fac8d372b3303f
```

The directory is not interpreted by current readers, but it is not cleaned or
reported as an interrupted transaction.

### M-3: comparison registration uses `PASS` as a generic comparison verdict

Every valid comparison calls `publish_completed_evaluation` with
`BenchmarkVerdict.PASS` (`cli.py:126-134`), even though a comparison has no PASS
gate verdict and may compare valid FAIL/PROVISIONAL runs. Registry v1 happens not
to persist the argument, masking the semantic mismatch.

### M-4: current tests materially under-cover claimed acceptance behavior

The 14 benchmark tests cover core synthetic evaluation but not:

- real subprocess execution;
- alternate working directories;
- identifier/path traversal;
- publication interruption;
- completion or artifact hashes;
- registry history;
- mixed output directories;
- CSV formulas;
- malformed evidence members;
- contradictory missing-abstract counts.

Passing totals therefore do not substantiate the publication and standalone-CLI
claims.

### M-5: documentation overstates atomicity and exact reconciliation

`docs/screening-benchmark.md:57-60,70-75,99-102` claims exact provenance
reconciliation, failed-publication non-registration, and atomic reports. The
missing-abstract, crash-window, mixed-output, and partial-publication probes
disprove those claims. Individual files use temporary replacement; the report
bundle is not atomic.

## LOW findings

### L-1: generated HTML contains mojibake punctuation literals

`report.py:48-50` contains `â€“` and `â€”` rather than an en dash and em dash.
The generated report also uses a mojibake middle-dot literal near the job and
population text. This is presentation-only but contradicts polished UTF-8 output.

## VERIFIED behavior

### V-1: Phase 3A execution identity and reliability mechanisms are present

Production constants:

```text
litsync_app/integrations/gemini_web_v24_screening.py:34
  gemini-web-v2.4-protocol-v3
litsync_app/integrations/gemini_web_v24_screening.py:35
  gemini-web-v2.4-assessment-prompt-v5
litsync_app/integrations/gemini_web_v24_screening.py:36
  gemini-web-v2.4-assessment-v5
```

No production source or current unit test contains
`scoped_exclusion_reject` or `methodological_study_reject`. Current verifier flags
contain only validation errors and unresolved criterion IDs
(`screening.py:1353-1363`), not Phase 3B route predictions or contributing IDs.
Current checkpoint identity uses protocol and prompt v5, with no
`Assessment_Cache_Version` row requirement.

Adaptive batching (`screening.py:246`), compact transport schema
(`screening.py:216`), exact criterion/evidence validation
(`screening.py:826`), bounded retry/replay (`screening.py:665-786`), and
safe-MAYBE outcomes remain covered by the 59 passing tests.

A complete-repository search does find Phase 3B route names and v6 identities in
persisted historical CSVs and `.codex_test_outputs`. Those are audit artifacts,
not active production rules, and should not be rewritten as part of restoration.
Generic method-neutral prompt language remains from earlier Phase 2C behavior and
is explicitly covered by `test_phase2c_*`; it is not by itself proof of an active
Phase 3B route.

### V-2: offline architecture boundary is real

AST-derived package import graph:

```text
__init__    -> contracts
cli         -> comparison, contracts, errors, evaluator, loader, registry, report
comparison  -> contracts, errors
contracts   -> pydantic
evaluator   -> contracts, loader, release_gate
loader      -> contracts, errors, provenance, pandas, pydantic
provenance  -> pandas
registry    -> contracts, errors, provenance
release_gate -> contracts
report      -> contracts, provenance
```

No benchmarking module imports browser automation, Gemini execution, FastAPI
state, application state, Gold Validation, network clients, or socket/URL
libraries. Production searches found no blockchain, healthcare, finance,
supply-chain, title-specific, dataset-specific, or row-specific rule.

### V-3: strict contracts and core population rules work

- `StrictModel` rejects unexpected fields.
- Gold/run CSV reads use `dtype=str`, `keep_default_na=False`, and
  `encoding="utf-8-sig"` (`loader.py:94-101,235-240`).
- Duplicate/blank run IDs and duplicate/blank gold IDs fail.
- Gold coverage must exactly match `gold_selected_source_row_ids`
  (`loader.py:129-136`).
- Gold IDs may be a subset of run IDs; extra non-gold rows are valid.
- `UNSURE` is separated from resolved labels (`loader.py:137-143`).
- The existing explicit 100-run/60-gold test passes.

### V-4: canonicalization and fingerprint separation are substantively correct

`provenance.py:36-109` uses normalized string line endings, UTF-8, sorted compact
JSON keys, SHA-256, string-preserving logical CSV content, and distinct
`fingerprint_kind` payloads.

Independent probes:

```json
{
  "canonical_equal_across_key_order_and_line_endings": true,
  "source_cell_mutation_changes_source_fingerprint": true,
  "raw_input_mutation_changes_input_fingerprint": true,
  "output_object_key_order_is_canonical": true
}
```

`screening_output_fingerprint` is absent from `BenchmarkSpec` and appears only in
run/result provenance. Exact raw-input equality among summary, PRISMA, and spec is
enforced (`loader.py:326-338`).

### V-5: audited quality metrics are correct on the hand fixture

Hand fixture confusion matrix:

```text
gold KEEP:   model KEEP=1, MAYBE=1, REJECT=0
gold REJECT: model KEEP=0, MAYBE=0, REJECT=1
UNSURE row: excluded
```

Observed values:

```text
KEEP+MAYBE recall                  2/2 = 1.0
false-reject rate                  0/2 = 0.0
definitive KEEP precision          1/1 = 1.0
definitive REJECT precision        1/1 = 1.0
specificity                        1/1 = 1.0
manual review burden, full run     1/5 = 0.2
manual review burden, resolved     1/3 = 0.333333
```

Wilson intervals match the standard 95% formula; zero denominator yields
`value=null` and `confidence_interval=null`. Metrics include numerator,
denominator, value, interval/null, population, definition version, and source
fields. Operational values are read directly from authoritative summary fields
(`evaluator.py:274-301`), not reconstructed from JSONL.

### V-6: ordinary provenance classifications and principal contradictions work

Existing tests prove:

- no resume/cache -> `COLD`;
- cache only -> `WARM_CACHE`;
- some resumed -> `PARTIALLY_RESUMED`;
- all resumed -> `FULLY_RESUMED`;
- mixed/missing fingerprint data -> `INVALID_PROVENANCE`;
- overlapping/incomplete partitions fail;
- resumed/cache/fresh count disagreements fail;
- row prompt/protocol disagreement and PRISMA input mismatch fail.

The missing-abstract count omission is the exception recorded as H-4.

### V-7: HTML and deterministic JSON protections work

Existing hostile HTML test passes and all dynamic report fields are escaped with
`html.escape`. Static inspection found no external scripts, stylesheets, images,
fonts, URLs, or network assets. Canonical result/comparison JSON is deterministic.
No raw Gemini response or hidden-reasoning field is part of the result contracts.
CSV quoting handles commas, quotes, CRLF, and embedded newlines; formula execution
is the distinct unresolved defect H-2.

### V-8: registry locking basics work

- `validate-spec` did not create the supplied registry directory.
- Lock creation uses Windows-compatible `os.open(..., O_CREAT | O_EXCL | O_WRONLY)`
  (`registry.py:46-52,94-100`).
- Owned locks are released in `finally`.
- An existing/stale lock returned exit `2` and remained present.
- Corrupt JSON registry content returned exit `2` with clean JSON:
  `{"error":"benchmark registry entry is unreadable","status":"invalid"}`.
- Valid PASS, FAIL, and PROVISIONAL evaluations create the v1 identity entry;
  INVALID does not.

Artifact completeness and history remain blockers despite these lower-level
locking properties.

## Suspected risks tested and disproven

1. **Forbidden runtime dependencies:** disproven; import graph is offline and
   state-independent.
2. **Domain or benchmark-row production heuristics:** disproven.
3. **Unexpected Pydantic fields accepted:** disproven.
4. **Object key order changes canonical fingerprints:** disproven.
5. **Meaningful source/input cell mutation leaves fingerprints unchanged:**
   disproven.
6. **Gold must equal the entire run population:** disproven; subset behavior works.
7. **UNSURE contaminates resolved quality denominators:** disproven.
8. **Operational metrics silently reconstructed from JSONL:** disproven.
9. **Different resolved-gold populations receive quality deltas in the pure
   comparison function:** disproven; the function suppresses them. The public CLI
   cannot construct this case.
10. **Stale lock silently deleted:** disproven.
11. **Normal `--json` success/handled-invalid output contains logs:** disproven.
    The separate argparse and unhandled-exception defects are H-8 and H-3.
12. **Windows paths containing spaces fail when run from repository root:**
    disproven; PASS evaluation returned exit `0`.

## CLI subprocess acceptance matrix

All paths below were isolated under `.audit_tmp` and are removed after this report.

| Case | Observed exit | Machine-readable stdout | Result |
|---|---:|---|---|
| `validate-spec` valid | 0 | Yes | valid; registry untouched |
| `validate-spec` changed content/stale fingerprint | 2 | Yes | invalid |
| `evaluate` PASS + `--enforce-gate` | 0 | Yes | COLD/PASS |
| `evaluate` FAIL + `--enforce-gate` | 3 | Yes | COLD/FAIL |
| `evaluate` FAIL without enforcement | 0 | Yes | COLD/FAIL |
| `evaluate` PROVISIONAL + enforcement | 3 | Yes | WARM_CACHE/PROVISIONAL |
| `evaluate` INVALID | 2 | Yes | INVALID_PROVENANCE/INVALID |
| `compare` valid | 0 | Yes | valid |
| `compare` different dataset | 2 | Yes | invalid, but incorrectly includes 32 deltas |
| different resolved-gold population | N/A via CLI | N/A | one-spec interface cannot express it |
| invalid usage with `--json` | 2 | No | stdout empty; usage on stderr |
| malformed evidence list member | 1 | No | uncaught traceback |
| traversal job ID `../escape` | 2 | Yes | correctly rejected |
| reserved job ID `CON` | accepted by validator | N/A | later failure depends on artifacts |
| traversal benchmark version | 0 | Yes | PASS and registry-root escape |
| traversal gold path | 0 | Yes | accepted |
| non-empty mixed output | 0 | Yes | old and new files mixed |
| existing lock | 2 | Yes | lock preserved; staging leaked |
| corrupt registry | 2 | Yes | fails closed |
| contradictory missing-abstract count | 0 | Yes | incorrectly COLD/PASS |
| different CWD, no `PYTHONPATH` | 1 | No | module not found |
| different CWD with repository `PYTHONPATH` | 0 | Yes | valid |

Representative exact successful evaluation command:

```powershell
python -m litsync_app.benchmarking.cli evaluate `
  --spec "C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\base\benchmark.json" `
  --job-id cold-job `
  --artifacts-root "C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\base\outputs" `
  --registry-dir "C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\eval registry" `
  --output-dir "C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\pass output with spaces" `
  --json --enforce-gate
```

Representative valid comparison command:

```powershell
python -m litsync_app.benchmarking.cli compare `
  --spec "C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\base\benchmark.json" `
  --job-id cold-job --job-id second-job `
  --artifacts-root "C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\base\outputs" `
  --registry-dir "C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\compare-registry" `
  --output-dir "C:\Users\Harshil\Downloads\newresearchtool\.audit_tmp\compare-output" `
  --json
```

## Historical persisted-artifact audit

No historical file was changed. No real benchmark was created or registered.

All four jobs have 100 screened CSV rows and PRISMA
`records_selected=100`. Their PRISMA input fingerprint is:

```text
0ea9bc8c9706d2fe9085fa449461aa7e95e6cd89f1c2284ecfb20a3777b82640
```

All four PRISMA records have `finalized=false` and `csv_counts_match=null`.
All four diagnostic summaries lack:

- `job_id`
- `run_status`
- `run_selected_count`
- `run_selected_source_row_ids`
- exact resumed/cache/fresh/missing-abstract row-ID partitions
- `source_dataset_fingerprint`
- `screening_input_fingerprint`
- `screening_output_fingerprint`

Consequently none can satisfy the current strict loader, and none may receive a
release PASS.

| Job | Persisted identity/evidence | Audit provenance conclusion |
|---|---|---|
| `1bc24287-684a-42ce-9e74-247363a16ad5` | protocol v3, prompt/cache v5; `primary_papers_requested=33`; no exact reuse partitions | `INVALID_PROVENANCE`; neither cold nor reuse status is provable |
| `c68f05c5-6832-4358-ac6c-8574e54ceacf` | protocol v3, prompt/cache v5; 100 rows but only 8 primary papers requested; route counts include one historical `scoped_exclusion_reject` | `INVALID_PROVENANCE`; definitely must not be called cold, but the exact 92 reused row IDs are absent |
| `2289eb04-c9e6-47f0-90bc-cd76ee92be24` | protocol v3, prompt/cache v6; 100 primary papers requested, cache count 0 | `INVALID_PROVENANCE`; fresh execution is suggested, but exact row provenance is absent and v6 conflicts with restored v5 release identity |
| `4c38836d-cc5d-451d-b110-d2e6e57910ed` | protocol v3, assessment cache v4, no prompt identity, no primary-paper count | `INVALID_PROVENANCE`; historical identity and reuse are unprovable |

The historical `scoped_exclusion_reject` and v6 values are persisted provenance,
not active production behavior.

### Candidate gold file

Inspected:

```text
C:\Users\Harshil\Downloads\newresearchtool\outputs\gold_validation\27d0f7fdfc30dc48_labels.csv
SHA-256: 749def3b24475dc9d4db1fcc1fc1c96330607cf3e397bb0508b4d2c607966796
```

Observed current file:

```text
rows=60
KEEP=0
REJECT=0
UNSURE=0
blank Gold_Decision=60
```

Its report binds it to job `1bc24287-684a-42ce-9e74-247363a16ad5`,
validation set `27d0f7fdfc30dc48`, protocol `9a2fc25a6203f968`, and prompt v5,
but the report claims `resolved_labels=59`. That claim does not match the current
all-blank CSV and cannot substitute for frozen labels.

A deliberately incomplete audit-only specification omitted the two fingerprints
that cannot be proved:

- `source_dataset_fingerprint`
- `screening_input_fingerprint`

`validate-spec --json` returned exit `2` with both fields reported missing. No
values were invented. Even if those identities became available, the current
blank gold labels would fail the loader's `KEEP|REJECT|UNSURE` rule.

## Documentation discrepancies

1. Exact missing-abstract reconciliation is claimed but not implemented
   (`docs/screening-benchmark.md:57-60`; H-4).
2. Failed publication non-registration is claimed but false in the post-registry
   crash window (`docs:70-75`; B-4).
3. Reports are called atomic, but only individual files are atomically replaced;
   the bundle can be partial or mixed (`docs:99-102`; B-3).
4. Comparisons are said to require identical datasets, but invalid cross-dataset
   comparisons still publish metric deltas (`docs:104-108`; H-5).
5. Independent release compatibility across assessment versions is not surfaced
   in comparison output; the comparison contract omits per-run gate verdicts and
   assessment identities.
6. The documented `python -m` commands work at repository root but the package is
   not installable/discoverable from another directory without `PYTHONPATH`.

## Requirements currently unsupported

- Hashed `_COMPLETE.json` publication marker.
- Artifact size/hash verification and completion readers.
- Deterministic pending-publication recovery.
- Orphaned-complete-publication detection/recovery.
- Non-mixing atomic directory publication.
- Append-only registry evaluation/comparison history.
- Idempotent completed-publication references.
- Fully hardened ASCII/Windows identifiers and path containment.
- CSV formula-injection protection.
- CLI comparison of two independently specified benchmark/gold populations.
- JSON stdout for parser-level usage errors.
- Standalone invocation away from the repository without external path setup.

## Phase 1 stop statement

This audit created only:

```text
docs/screening-benchmark-acceptance-audit.md
```

No production code, tests, existing documentation, examples, fixtures, benchmark
artifacts, persisted jobs, or candidate gold files were modified. All CLI
experiments used isolated audit temporary paths. No real benchmark specification
was completed or registered. No Gemini Web execution occurred. No commit, push,
reset, clean, checkout, or broad restore was performed.

Phase 2 corrections have not begun and require explicit approval.

---

## Phase 2A remediation record

Remediation date: 2026-07-29  
Authorization: audited BLOCKER and correctness-critical HIGH findings only

This section is append-only. The pre-fix findings above were not rewritten or
weakened.

### Proven defects fixed

- **B-1 / H-6 identifier and registry-path escape:** benchmark IDs, benchmark
  versions, and job IDs now share a strict 128-character ASCII validator that
  rejects whitespace normalization, trailing dots, path syntax, Unicode
  lookalikes, and Windows device names. Registry paths are resolved and checked
  beneath the selected registry root
  (`contracts.py:20`, `registry.py:25`, `loader.py:160`).
- **B-2 completion and artifact integrity:** every publication now ends with
  `_COMPLETE.json`, which records immutable identity plus every required
  non-marker artifact's relative path, SHA-256, and byte size. Readers reject
  missing, extra, unsafe, size-mismatched, and hash-mismatched artifacts
  (`report.py:91-246`).
- **B-3 mixed/partial publication:** reports are staged in a unique sibling
  directory and renamed as a complete directory. A non-empty destination is
  accepted only when its completion marker and immutable artifact manifest are
  identical; otherwise publication fails without mixing files
  (`report.py:255-296`).
- **B-4 commit ordering:** registry v2 writes pending state before output
  publication and treats the final atomic promotion to
  `completed_publications` as the commit point. Pre-promotion failures have no
  completed reference. Identical retries recover matching pending output or
  return an already committed publication without duplicate history
  (`registry.py:207-306`).
- **H-1 registry history:** registry v2 retains an append-only logical list of
  completed evaluation/comparison references. Each reference records kind,
  ordered job IDs, verdict, completion-marker path, and marker hash
  (`registry.py:72-83`, `registry.py:181-204`).
- **H-2 CSV formula injection:** every serialized string cell is protected when
  its first non-whitespace character is `=`, `+`, `-`, or `@`; JSON values remain
  unprefixed (`report.py:299-319`).
- **H-3 malformed persisted evidence:** evidence members must be objects, nested
  summary/PRISMA structures are type-checked, and evaluation no longer calls
  mapping methods on arbitrary values. Handled malformed artifacts produce
  `INVALID`, JSON output, and exit 2 without a traceback
  (`loader.py:285-566`, `evaluator.py:145-165`).
- **H-4 contradictory provenance:** `Execution_Origin` and
  `Direct_Handling_Reason` are persisted on final rows. Current-run origins are
  assigned at the production boundary; checkpoint-restored rows become
  `resumed`. Summary partitions and reason mappings are derived from final rows
  and the loader fails closed on any disagreement. Missing-abstract IDs remain
  an independent set and may overlap every origin
  (`gemini_web_v24_screening.py:1073-1178,1238-1285,1504-1572`;
  `loader.py:445-566`).
- **H-5 invalid comparison calculations:** any compatibility or provenance
  reason now suppresses pairwise calculations, deltas, transitions, matrices,
  and correction/regression claims (`comparison.py:110-174`).
- **M-1 exact source identifiers:** persisted CSV identifiers are retained as
  strings without decimal conversion, including leading zeros
  (`provenance.py:26-27`).
- **M-3 comparison registry semantics:** valid comparison publications use
  publication kind `comparison` and verdict `COMPARISON_VALID`, rather than an
  evaluation PASS surrogate (`cli.py:121-166`).
- **M-4 / M-5 acceptance coverage and documentation:** focused tests now cover
  subprocess exits, provenance contradictions, hostile identifiers, completion
  integrity, registry recovery/history, failure boundaries, mixed output,
  malformed artifacts, CSV injection, and invalid comparisons. The public
  benchmark documentation now describes row-authoritative provenance,
  completion markers, and registry v2.

### Transaction-state regression evidence

Regression tests cover:

- failure before the initial pending write;
- failure immediately after the pending write with no output;
- failure after complete output publication while the registry remains pending;
- failure immediately after durable completed-history promotion;
- deterministic same-publication retry for every state;
- matching pending state with valid marker/hash promotion;
- foreign pending-publication rejection without mutation;
- completed-reference corruption and legacy-registry rejection;
- timestamp-independent idempotence preserving the original marker and
  `created_at`.

All recovery cases finish with one completed history reference and no pending
entry.

### CLI subprocess regression

The subprocess regression invokes the unchanged public entry point:

```text
python -m litsync_app.benchmarking.cli validate-spec ... --json
python -m litsync_app.benchmarking.cli evaluate ... --enforce-gate --json
```

It uses absolute paths under an isolated Windows directory containing spaces.
Observed exit codes:

```text
validate-spec valid: 0
evaluate PASS with --enforce-gate: 0
evaluate FAIL with --enforce-gate: 3
evaluate PROVISIONAL with --enforce-gate: 3
evaluate INVALID malformed provenance: 2
```

Each stdout value parsed as exactly one JSON document; handled INVALID output
contained no traceback and did not create a registry.

### Validation commands and results

```powershell
python -m pytest tests/unit/test_benchmarking.py -q --basetemp .pytest_tmp_benchmark_phase2a_final
```

Result: exit `0`, `65 passed in 18.56s`.

```powershell
python -m pytest tests/unit/test_gemini_web_v24.py -q --basetemp .pytest_tmp_gemini_phase2a_final
```

Result: exit `0`, `61 passed in 2.20s`.

```powershell
python -m pytest -q --basetemp .pytest_tmp_full_phase2a_final
```

Result: exit `0`, `298 passed in 55.75s`.

```powershell
python -m pytest tests/unit/test_benchmarking.py::test_cli_subprocess_phase2a_exit_codes_and_json_contract -q --basetemp .pytest_tmp_cli_phase2a_final3
```

Result: exit `0`, `1 passed in 10.23s`.

```powershell
python -m compileall -q litsync_app tests
```

Result: exit `0`; no compilation errors.

```powershell
git diff --check
```

Result: exit `0`; only pre-existing Git LF-to-CRLF warnings were emitted.

### Phase 2A files

Screening provenance:

- `litsync_app/integrations/gemini_web_v24_screening.py`

Benchmark implementation:

- `litsync_app/benchmarking/contracts.py`
- `litsync_app/benchmarking/provenance.py`
- `litsync_app/benchmarking/loader.py`
- `litsync_app/benchmarking/evaluator.py`
- `litsync_app/benchmarking/comparison.py`
- `litsync_app/benchmarking/report.py`
- `litsync_app/benchmarking/registry.py`
- `litsync_app/benchmarking/cli.py`

Tests:

- `tests/unit/test_benchmarking.py`
- `tests/unit/test_gemini_web_v24.py`

Documentation/example:

- `docs/screening-benchmark.md`
- `docs/screening-benchmark-acceptance-audit.md`
- `examples/benchmarking/benchmark.example.json`

### Deferred or unsupported findings

- H-7: standalone off-repository module discovery remains unsupported without
  configuring `PYTHONPATH` or packaging the project.
- H-8: argparse-level usage failures still use argparse stderr rather than the
  command's JSON error envelope.
- H-9: the public compare CLI still accepts one specification and therefore
  cannot express two unrelated benchmark identities or frozen-gold files.
- M-2: abandoned staging-directory cleanup outside the corrected transaction
  paths remains deferred.
- General documentation polish and unrelated presentation work remain deferred.

No Gemini Web execution, real benchmark registration, commit, push, reset,
clean, checkout, or broad restore occurred during Phase 2A.
