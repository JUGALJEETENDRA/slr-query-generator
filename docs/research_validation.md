# LitSync Research Validation

LitSync Research Validation tests the frozen `gemini-web-batched-v2.3` workflow on a new systematic-review topic. It does not tune model behavior, target decision counts, or treat generated decisions as truth.

## Researcher Inputs

- A research question and optional context, inclusion criteria, and exclusion criteria.
- A CSV paper collection and explicit title/abstract column names. Year and DOI are optional.
- Exactly two independent reviewer identifiers.
- A declared manual-review capacity, defaulting to 30%.

Each paper must have a title and abstract. Input columns representing gold labels, expected outcomes, reviewer notes, sampling strata, or model decisions are rejected.

## Lifecycle

1. `init` fingerprints the corpus and review specification, selects a uniform probability sample, selects the 60-paper core before any model call, and seals the private preregistration.
   It also reports whether that core can mathematically satisfy the configured Wilson-interval trust gate under perfect observed performance.
2. `run` executes the same 100-paper sample twice with unique jobs, `resume=False`, raw capture disabled, frozen Gemini Web v2.3, and one immutable protocol.
3. The framework selects up to 30 supplemental papers outside the core using model-independent failure coverage: repeat disagreement, direct contradiction, critic or verification failure, transport failure, invalid evidence, or non-substantive support.
4. `export-review` creates two independently ordered reviewer packs. They contain review criteria and bibliographic text, but no Gemini decision, confidence, route, evidence, protocol identifier, or selection reason.
5. `import-review` verifies reviewer identity, opaque row IDs, paper text, complete labels, rationales, and confidence. Human `ABSTAIN` is reviewer uncertainty; gold `MAYBE` means title/abstract evidence is genuinely insufficient.
6. `export-adjudication` creates a prediction-blind pack only for disagreements and abstentions.
7. `import-adjudication` resolves every disputed paper and atomically locks a private gold sidecar.
8. `report` exposes model decisions only after gold lock, calculates paper-level and aggregate evidence, updates the private cross-study registry, and issues a trust verdict.
9. Researchers review the generated root-cause worksheet and use `import-root-causes`; only confirmed failure signatures can support a cross-domain weakness claim.

Before dual review and adjudication, the only valid verdict is `INSUFFICIENT_EVIDENCE`.

## CLI

```powershell
python -m evaluation.research_validation init `
  --corpus papers.csv `
  --question "Your research question" `
  --title-column "Title" `
  --abstract-column "Abstract" `
  --reviewer reviewer-a --reviewer reviewer-b

python -m evaluation.research_validation run --study STUDY_ID
python -m evaluation.research_validation export-review --study STUDY_ID
python -m evaluation.research_validation import-review --study STUDY_ID --reviewer reviewer-a --file reviewer-a-completed.csv
python -m evaluation.research_validation import-review --study STUDY_ID --reviewer reviewer-b --file reviewer-b-completed.csv
python -m evaluation.research_validation export-adjudication --study STUDY_ID
python -m evaluation.research_validation import-adjudication --study STUDY_ID --file adjudication-completed.csv
python -m evaluation.research_validation report --study STUDY_ID
python -m evaluation.research_validation import-root-causes --study STUDY_ID --file root-cause-confirmation.csv
python -m evaluation.research_validation status --study STUDY_ID
python -m evaluation.research_validation compare --baseline baseline-report.json --candidate candidate-report.json
```

The web page exposes the same lifecycle in the **Research Validation** panel.

## Blinding and Leakage Guarantees

- Core sampling happens before model execution and is sealed by corpus, input-file, and preregistration fingerprints.
- Reviewer packs use reviewer-specific opaque row IDs and independent ordering.
- Gold files and reviewer linkage live under `private/research_validation/<study_id>/`; public reports and blinded worksheets live under `outputs/research_validation/<study_id>/`.
- Production screening receives only the review specification and paper text.
- Runtime checks reject evaluation columns in screening input, changed paper text, changed reviewer identity, changed sample membership, altered preregistration, protocol drift, resumed decisions, raw-response capture, and non-metadata diagnostics.
- No expected label distribution, domain dictionary, acronym list, ontology, or historical gold is used to choose papers or make decisions.

## Evidence and Trust Logic

The report includes:

- Human evidence: coverage, raw agreement, Cohen's kappa, abstentions, disagreements, and adjudication completion.
- Screening quality: three-class confusion, `KEEP|MAYBE` relevant recall, false-REJECT rate, definitive-KEEP precision, definitive accuracy, gold-MAYBE overcommitment, and Wilson 95% intervals from the uniform core.
- Manual burden: MAYBE rate, critic rate, and verification-fallback rate.
- Repeatability: exact agreement, full transition matrix, direct `KEEP` to `REJECT` contradictions, repeated MAYBEs, and unstable rows.
- Evidence and verification: validation status, cited evidence, criterion assessments, scope-support classes, critic routes, and verification outcomes.
- Operations: completion, wall time, throughput inputs, retries, transport fallbacks, detector outcomes, recovery actions, and browser degradation signals.
- Root cause: one audit row per unsafe disagreement, unstable decision, or verification problem, with a deterministic preliminary category and a separate researcher-confirmation worksheet.

Default gates require 60 resolved core labels, at least 95% coverage, complete adjudication, kappa at least 0.70, relevant recall at least 0.95, false-REJECT rate at most 0.05, definitive-KEEP precision at least 0.85, repeatability at least 0.90, no direct contradictions, no diagnostic-supplement unsafe errors, structurally valid decisions, transport fallback at most 5%, and no leakage violation.

- `TRUST`: every point gate passes and intervals do not cross safety thresholds.
- `CONDITIONAL`: no observed unsafe or operational failure, but uncertainty remains or manual burden exceeds capacity.
- `REJECT`: observed unsafe errors, direct contradictions, leakage, or operational failure.
- `INSUFFICIENT_EVIDENCE`: human review is incomplete or insufficiently reliable.

Wide intervals never become `TRUST` merely because point estimates look attractive.

## Cross-Domain and Version Evidence

Failures are normalized by decision transition, criterion role, scope support, critic route, verification outcome, and confirmed category. A weakness is cross-domain only when the same signature appears in at least two independently dual-reviewed studies. Historical single-reviewer controls remain context only.

Confirmed `gold_adjudication_error` findings qualify interpretation without rewriting immutable human gold or changing historical confusion metrics. Browser evidence is registered separately as late-session degradation, no-container timeout recovery, or exhausted transport fallback. Cross-domain recurrence is counted by distinct review-domain fingerprints, never by repeated studies in one topic.

Version comparison requires the same study ID, corpus fingerprint, preregistration, protocol, locked gold, sampled rows, and repeat structure. It reports paired paper transitions and metrics; it does not optimize label distributions.

## Main Limitation

This framework can rigorously evaluate title-and-abstract screening, but it cannot replace full-text eligibility adjudication. A finite human sample can also leave rare unsafe failures statistically unresolved; that uncertainty is reported rather than hidden.
