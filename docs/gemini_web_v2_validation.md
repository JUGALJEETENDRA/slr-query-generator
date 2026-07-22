# Gemini Web v2 validation suite

This suite validates the committed Gemini Web v2 candidate without changing production screening behavior. It runs
one fresh clear control, the same hard near-miss control twice, and one natural 100-paper cross-domain control.

## Privacy and blinding

- Screening fixtures contain only `Title` and `Abstract`.
- Gold decisions and adjudication rationale remain in separate sidecars and are never sent to Gemini.
- Production diagnostics are checked against the metadata-only field allowlist.
- `GEMINI_WEB_CAPTURE_RAW_DEBUG` must be disabled or the suite refuses to run.
- Generated fixtures and reports live under `.codex_test_outputs/`; screened exports and normal diagnostics retain the
  existing `outputs/runs/` and `outputs/cache/gemini_web/diagnostics/` contracts.

Dataset-specific row selections and labels are evaluation fixtures only. They are not imported by production
screening and do not create keyword or topic rules.

## Prepare without running

```powershell
python -m evaluation.gemini_web_v2_validation prepare
```

The command prints the timestamped artifact root and manifest. Review the blinded fixture and gold sidecar separately
before starting the browser run.

## Run all four controls

Close any dedicated LitSync Gemini browser window, confirm the saved browser profile is signed in, then run:

```powershell
python -m evaluation.gemini_web_v2_validation run
```

Individual cases can be rerun without reusing paper decisions:

```powershell
python -m evaluation.gemini_web_v2_validation run --manifest "PATH_TO_MANIFEST" --case hard_control_a_40 --case hard_control_b_40
python -m evaluation.gemini_web_v2_validation score-suite --manifest "PATH_TO_MANIFEST"
```

Every case uses a unique job ID and `resume=False`. The aggregate report includes paper-level failures, group-level
confusion, repeatability transitions, repeated unresolved MAYBEs, detector outcomes, and a recommendation. A
production change is supported only when a hard-control safety failure or transport failure repeats.

## Gemini Web v2.1 candidate

Version `gemini-web-batched-v2.1` applies only the two weaknesses reproduced by the v2 suite:

- Definitive `KEEP` decisions cannot use an acronym as the sole bridge to a required concept unless the supplied
  title or abstract defines a criterion-aligned expansion or states the concept independently.
- A chat rotates after six completed submissions, and the persistent browser context recycles after twelve. A
  final-sweep no-container timeout triggers one bounded recovery and retries only the affected batch.

Operational limits remain configurable with `GEMINI_WEB_MAX_CHAT_SUBMISSIONS`,
`GEMINI_WEB_MAX_BROWSER_SUBMISSIONS`, and `GEMINI_WEB_RECOVERY_BACKOFF_MS`. Defaults are `6`, `12`, and `2000`;
backoff is bounded to ten seconds. Batch size, sequential execution, one retry, safe transport `MAYBE`, checkpoints,
and resume semantics are unchanged.

## Gemini Web v2.2 candidate

Version `gemini-web-batched-v2.2` treats routed definitive decisions as provisional. A low-risk REJECT remains
single-pass when an explicit exclusion criterion is met. A REJECT based only on required inclusions marked
`NOT_MET` receives one independent blinded critic pass in a clean chat. Agreement retains the decision; disagreement,
critic uncertainty, malformed verification, or unavailable verification resolves safely to `MAYBE`.

Workflow and protocol-cache versions are separate, so orchestration changes no longer regenerate a compatible
immutable protocol. v2.1 decision checkpoints are intentionally incompatible, while validated v2.1 protocol artifacts
are migrated and reused. Output rows expose `Critic_Route` and `Verification_Status`; summaries report route and
agreement counts without adding prompts or response text to diagnostics.
## Gemini Web v2.3 frozen safety-first candidate

Version `gemini-web-batched-v2.3` adds a Gemini-Web-only `scope_support` classification to every criterion.
Required inclusions can be `MET` only when cited title/abstract evidence represents substantive study scope;
incidental, background, definitional, list-level, or insufficient support must remain `UNCLEAR`. The protocol cache
remains compatible, while v2.2 decision checkpoints are intentionally not resumed.

The final fresh 100-paper validation completed all rows with 71 `KEEP`, 29 `MAYBE`, and 0 `REJECT` in 358.8
seconds with one retry. It produced zero false `KEEP`, positive `REJECT`, invalid, unverified definitive, or
timeout-fallback rows. Compared with v2.2, nine decisions changed from `KEEP` to `MAYBE` and two changed from
`MAYBE` to `KEEP`.

The original acceptance report records failure against a predeclared ceiling of 27 MAYBEs. That artifact remains
unchanged. Subsequent paper-level review found every added MAYBE had explicit `INCIDENTAL` or `INSUFFICIENT`
support, so the two-row ceiling miss did not justify another production patch. v2.3 is therefore frozen as the
safety-first Gemini Web candidate rather than optimized toward a target decision distribution.

Known limitations:

- The workflow is calibrated for safe manual review rather than maximum automatic resolution.
- Industrial applications that do not explicitly establish the protocol relationship can become `MAYBE`.
- Verification failure or invalid critic structure generated a substantial portion of the MAYBE set.
- Rejection decisiveness is weak: the 100-paper run produced no REJECTs, and the hard 40 produced 20 KEEP and
  20 MAYBE with all routed non-KEEP rows resolving as verification fallbacks.

Do not start v2.4 from the current blockchain or digital-twin controls. Reconsider calibration only if a fresh,
blinded review topic independently reproduces the same over-cautious boundary or verification-fallback behavior.
