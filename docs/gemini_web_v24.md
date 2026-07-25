# Gemini Web v2.4 Candidate

`gemini-web-batched-v2.4` is an opt-in candidate. Frozen v2.3 remains the default
Gemini Web baseline until comparative validation supports promotion.

## Domain neutrality

The protocol compiler receives only the researcher’s question, context, and
optional inclusion/exclusion criteria. Gemini derives the study-specific
concepts and semantic boundaries. Python validates schema, evidence references,
criterion consistency, and the final decision policy; it does not contain
domain keyword lists, acronym dictionaries, paper exceptions, expected label
ratios, or hidden ontologies.

Generated equivalents and near-neighbour concepts are advisory interpretation
context. They do not become automatic exclusions unless logically required by
the question or explicitly supplied by the researcher.

## Decision policy

- `KEEP`: every required inclusion is `MET` with substantive supporting
  title/abstract evidence, and no exclusion applies.
- `REJECT`: title/abstract evidence explicitly establishes a conflicting
  required relationship or an exclusion.
- `MAYBE`: a requirement is unmentioned, insufficient, incidental, unresolved,
  contradictory, or unavailable because of bounded technical failure.

Validated high-confidence, low-risk assessments finish after one primary call.
Risky or uncertain rows receive a prediction-blind assessment in a clean chat.
Disagreement or unavailable verification resolves conservatively to `MAYBE`.

## Caches and recovery

v2.4 uses independent protocol, validated-assessment, checkpoint, and diagnostic
namespaces under `outputs/cache/gemini_web_v24/`. Assessment cache identity
includes the workflow version, protocol ID, and normalized title/abstract
fingerprint. Transport fallbacks and unresolved verification are never cached
or cemented by resume.

Browser chat and persistent context rotation are bounded and configurable:

- `GEMINI_WEB_V24_MAX_CHAT_SUBMISSIONS` (default `5`)
- `GEMINI_WEB_V24_MAX_BROWSER_SUBMISSIONS` (default `10`)
- `GEMINI_WEB_V24_RECOVERY_BACKOFF_MS` (default `2000`, maximum `10000`)

Diagnostics remain metadata-only. Raw response capture remains disabled by
default and uses the existing explicit temporary debug mechanism.

## Comparison

Run a blinded v2.3/v2.4 comparison with researcher-owned inputs:

```powershell
python -m evaluation.gemini_web_v24_benchmark `
  --papers path\papers.csv `
  --gold path\gold.csv `
  --question "Your research question" `
  --output-root .codex_test_outputs\gemini-web-v24-comparison
```

The runner performs fresh sequential jobs and reports quality, runtime, calls,
verification, retries, fallbacks, manual-review burden, and repeatability. It
contains no built-in research topic or expected decision distribution.

An independently adjudicated 20–30-paper adversarial fixture can be blinded
without embedding its topic or labels in production:

```powershell
python -m evaluation.gemini_web_v24_adversarial `
  --adjudicated path\human_adjudicated.csv `
  --papers-out .codex_test_outputs\adversarial\papers.csv `
  --gold-out .codex_test_outputs\adversarial\gold.csv
```

The source must provide `Title`, `Abstract`, `Gold_Decision`,
`Gold_Rationale`, and `Challenge_Type`, with at least five independently
defined challenge types. No decision ratio is required or optimized.
