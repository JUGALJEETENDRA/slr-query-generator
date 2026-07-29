# LitSync Screening Benchmark and Release Gate

The benchmarking command is an offline reader of completed screening artifacts. It
does not start Gemini Web, import browser automation, inspect server memory, or use
Gold Validation UI state.

## Immutable identities and populations

A benchmark specification has two separate populations:

- `run_selected_source_row_ids` is the complete population that the persisted run
  must contain, once each and in the recorded order.
- `gold_selected_source_row_ids` is the frozen-gold subset. Every gold row must
  occur once in the run, but additional run rows are valid.

Confusion and quality metrics use resolved (`KEEP` or `REJECT`) gold rows only.
Operational and reliability metrics use the complete run. `UNSURE` rows are
reported separately. The two review-burden metrics are deliberately distinct:

- `manual_review_burden_full_run`: model `MAYBE` rows / all completed run rows.
- `manual_review_burden_resolved_gold`: model `MAYBE` rows among resolved gold /
  resolved gold rows.

A threshold must name the exact metric it gates.

The five fingerprints are not interchangeable:

- `source_dataset_fingerprint` hashes canonical logical CSV content: ordered
  columns and rows, string-preserved values, normalized line endings, and the
  canonical payload version.
- `screening_input_fingerprint` is SHA-256 of the exact uploaded CSV bytes. For
  current supported runs this must equal PRISMA `input_fingerprint`, which is
  produced from `sha256(Path(csv_path).read_bytes())`.
- `screening_output_fingerprint` hashes the canonical completed screening CSV.
  It is run-specific and belongs only to loaded-run/result provenance, never to a
  benchmark specification.
- `gold_file_fingerprint` is SHA-256 of the exact frozen-gold file bytes.
- `benchmark_spec_fingerprint` hashes canonical immutable specification content
  after omitting that fingerprint field.

Canonical JSON is UTF-8, key-sorted, compact JSON with normalized line endings
and SHA-256. A historical producer whose raw-input identity cannot be proven
equivalent is `INVALID_PROVENANCE`; filenames and row counts are not evidence of
identity.

## Required persisted artifacts

Under `--artifacts-root`, a completed job named `JOB` must provide:

```text
runs/screened-JOB.csv
cache/gemini_web_v24/diagnostics/JOB.summary.json
cache/gemini_web_v24/diagnostics/JOB.jsonl
prisma/JOB.json
```

Every completed CSV row records its current-run `Execution_Origin` as exactly one
of `resumed`, `assessment_cache_hit`, `fresh_primary`, or
`directly_handled_without_primary`. Direct rows also record one stable
`Direct_Handling_Reason`; all other rows leave that field empty. Summary counts,
ID lists, and reason mappings are derived from and reconciled against those final
rows. Missing-abstract IDs are a separate descriptive set and may overlap any
execution origin. Ambiguous, mixed, missing, or contradictory provenance is
invalid.

Provenance is classified as `COLD`, `WARM_CACHE`, `PARTIALLY_RESUMED`,
`FULLY_RESUMED`, or `INVALID_PROVENANCE`. A valid cold run passing all gates is
`PASS`; cold threshold failures and any partial/full resume are `FAIL`; an
otherwise passing warm-cache run is `PROVISIONAL`; invalid or unprovable runs are
`INVALID`.

## Registry and publication

`validate-spec` is read-only. A successfully published valid completed
evaluation locks its benchmark ID/version fingerprint whether its verdict is
`PASS`, `FAIL`, or `PROVISIONAL`. Invalid, aborted, and failed publications do
not create completed registry references. Registry v2 retains an append-only
history of completed publications and a recoverable pending transaction. Each
publication is an atomic directory containing `_COMPLETE.json`, which records
the required artifact paths, hashes, and byte sizes. Completed references are
accepted only while their marker and artifacts remain intact. Reusing a locked
ID/version with different immutable content is rejected.

## Commands

```powershell
python -m litsync_app.benchmarking.cli validate-spec `
  --spec examples/benchmarking/benchmark.example.json `
  --registry-dir outputs/benchmarks/_registry --json

python -m litsync_app.benchmarking.cli evaluate `
  --spec benchmark.json --job-id JOB --artifacts-root outputs `
  --registry-dir outputs/benchmarks/_registry --output-dir benchmark/JOB `
  --enforce-gate --json

python -m litsync_app.benchmarking.cli compare `
  --spec benchmark.json --job-id OLD --job-id NEW --artifacts-root outputs `
  --registry-dir outputs/benchmarks/_registry --output-dir benchmark/comparison `
  --json
```

Exit code `0` means the command completed successfully, `2` means invalid
specification/artifacts/provenance, and `3` means `--enforce-gate` rejected a
`FAIL` or `PROVISIONAL` result.

Evaluation writes `benchmark-result.json`, `benchmark-report.html`,
`benchmark-errors.csv`, and `_COMPLETE.json`. Comparison writes
`benchmark-comparison.json` plus the HTML, CSV, and completion marker. Reports
are atomic, self-contained, escaped, and omit raw model responses and hidden
reasoning. CSV string cells beginning with spreadsheet formula prefixes after
leading whitespace are apostrophe-prefixed for CSV serialization only.

Comparisons require the same immutable benchmark, frozen gold, source dataset,
run/gold selections, protocol, and metric definitions. Assessment-version
differences are descriptive only and release compatibility is evaluated
independently. Quality deltas are suppressed for different resolved-gold
populations. Correction/regression language is reserved for rows freshly
assessed in both runs; reused rows are visibly labeled observed transitions.
