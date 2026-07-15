# LitSync local-AI-first screening

LitSync now asks local models to understand the review question, understand each paper,
compare the two, and return criterion-level evidence. Deterministic code validates JSON,
criterion consistency, and exact title/abstract evidence references; it does not judge research relevance.

## Local semantic-boundary three-layer runtime

The website's local screening path uses two small installed models and no 8B/14B model:

- Before paper screening, Qwen3 4B interprets the research question, optional Research
  Context, and explicit criteria. The original RQ remains the verbatim authoritative
  semantic anchor; a model paraphrase can never silently add mandatory scope. The model
  also creates a few advisory, question-specific semantic near-miss checks. The long
  context is not repeated for every paper, and no code vocabulary is generated.
- Layer 1 runs `qwen2.5:3b` over batches of four papers. Evidence-backed, low-risk
  KEEP decisions finish here. Every REJECT plus every MAYBE, invalid, or risk-flagged
  decision enters the deep-review queue so a small-model false REJECT cannot be final.
- After the complete quick pass is checkpointed, LitSync unloads 3B and loads
  `qwen3:4b-instruct-2507-q4_K_M` once. Layer 2 reviews queued papers in batches of four.
- Layer 3 keeps the same Qwen 4B model resident and gives it a fresh adjudication prompt.
  It challenges Layer-2 MAYBE/invalid results and every risk-flagged definitive result
  without paying for another model swap.

The final critic is prediction-blind: it receives the paper, protocol, and reason it was
sent for review, but not Layer 2's decision, rationale, or criterion verdicts. This avoids
anchoring on the earlier model output and forces an independent replacement assessment.
If the critic output is structurally or evidentially invalid, it cannot replace an
already validated deep assessment; LitSync retains the valid result and records the
failed critic attempt. This fallback compares validity only, never research relevance.
Every layer distinguishes a paper's own contribution or explicit review scope from a
background mention, and distinguishes a required relationship from co-occurring topics.
There are no domain keywords, ontologies, target decision ratios, automatic model
downgrades, or 14B escalation in this path. Output records contain `Layer_Trace_JSON`
so each final decision shows which layers actually ran.

`LOCAL_TRIAGE_MODEL`, `LOCAL_DEEP_MODEL`, `LOCAL_EDGE_MODEL`, and the three
`LOCAL_*_BATCH_SIZE` values are developer benchmark overrides. The beginner website
exposes none of them.

Inspect hardware without starting the web server:

```powershell
python -m local_ai
```

The lightweight HTTP equivalent is `GET /status`.

Install the two throughput models once (downloads can take longer than two minutes):

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

`LOCAL_EDGE_MODEL` can still point to a different local model for developer comparisons,
but the website default does not require or load Phi.

## Screening contract

`POST /screen` accepts `question`, `title`, `abstract`, optional `inclusion_criteria` and
`exclusion_criteria`, optional `research_context`, plus tier/resource overrides.
`POST /screen_csv` accepts the same
review inputs as multipart form fields. Both return schema version 2.

CSV outputs retain source fields and add `Decision`, `Reason`, `Confidence`,
`Protocol_ID`, `Criteria_JSON`, `Evidence_JSON`, `Validation_Status`, model/tier fields,
timing, `Decision_Risk`, and `Layer_Metrics_JSON`. Only `validated` assessments can produce definitive KEEP or REJECT decisions;
unrecoverable model or evidence failures become MAYBE.

CSV jobs checkpoint every `LOCAL_CHECKPOINT_INTERVAL` rows. Resume accepts only final
three-layer results belonging to the same question, criteria, and prompt version; an
unfinished intermediate layer is safely rerun (normally from its inference cache).
The checkpoint key also includes the uploaded CSV hash, model names, and batch sizes.
Quick-test and full runs of the identical file can share work, while a different
file or legacy output can never be resumed accidentally. Progress, results, downloads,
and finalization are tied to the job ID returned by `POST /screen_csv`.

## Evaluation

Human labels belong under `benchmark/gold/`; never use generated decisions as gold labels.
`evaluation.local_ai_benchmark` reports the release gates for relevant-paper recall,
false rejects, definitive-KEEP precision, evidence validity, and invalid definitive output.

External Gemini adapters require `requirements-external.txt` and
`ENABLE_EXTERNAL_ENGINES=true`. They are lazy-loaded and are not part of the local core.

## Real throughput benchmark

Run this yourself because a 1,000-paper model benchmark intentionally exceeds the
two-minute Codex command limit:

```powershell
python benchmark_local_screening.py benchmark_digital_twin_holdout_full.json --limit 1000 --save-rows outputs/benchmark_batched_1000.csv --baseline "C:\Users\Harshil\Downloads\screened (21).csv"
```

The JSON summary reports elapsed screening metadata, models used, batch calls,
structural validity, exact evidence validity, disagreements, and critic reversals.
The baseline comparison is diagnostic only and is never treated as human gold truth.

### Measured limit on the RTX 3050 6 GB laptop

The clean 100-paper `local-resident-three-layer-v3.4` run completed in 423.61 seconds:
177.6 seconds of 3B triage, 145.3 seconds of 4B deep review, 57.0 seconds of 4B edge
criticism, and protocol/setup overhead. It made 25 triage, 9 deep, and 4 edge calls
with zero batch retries. Linear projection is about 70.6 minutes per 1,000 papers.
This is a throughput measurement, not a gold-quality claim. The 30-minute target is
not achievable by this sequential three-layer Ollama path on the measured hardware;
meeting it requires a different inference runtime/hardware or a materially different
screening strategy, not weaker evidence validation.

## Quality rationale and current evidence limit

Version `local-semantic-boundary-v3.12` changes prompts and the protocol contract only.
It does not change the 4/4/4 batch sizes, phase order, routing rules, context limits,
model residency, website, or external/Gemini path. Previous outputs and caches are
excluded by the new version identity.

The design borrows the useful part of multi-reviewer systems such as LatteReview—an
independent later reviewer—while keeping LitSync fully local and preserving its existing
fast/deep/edge call budget. ASReview's active-learning approach is not included because
it requires iterative human labels and would be a different architecture. Recent
title/abstract-screening research reports boundary ambiguity, keyword overemphasis, and
incorrect topic inference as common LLM disagreement modes; semantic boundaries and a
prediction-blind critic directly target those modes without putting domain knowledge in
deterministic code.

On the uncached 100-paper diagnostic, v3.11 took 512.06 seconds, escalated 65 papers, and
produced 79 KEEP / 19 MAYBE / 2 REJECT. It caught the two known topic-substitution cases
that the earlier run had confidently kept, but only 85 outputs validated because invalid
critic responses displaced valid deep results. The v3.12 validity fallback then achieved
16/16 structurally valid and 100% exact evidence on the uncached 16-paper stress slice in
135.12 seconds, with 11 escalations and two batch retries. The full v3.11 run was about
21% slower than the earlier v3.4 100-paper run because semantic caution sent 65 rather
than 35 papers to 4B; model sizes, 4/4/4 batches, and routing rules themselves are
unchanged. This is an observed quality/throughput tradeoff, not hidden as a speed win.

These diagnostics are not gold truth. Recall, false-REJECT rate, and KEEP precision
remain unknown until the blinded human validation set is labeled.

References:

- [Understanding LLMs in Title-Abstract Screening: From Disagreements to Recommendations](https://arxiv.org/abs/2606.17588)
- [LatteReview local/Ollama multi-agent review framework](https://github.com/PouriaRouzrokh/LatteReview)
- [ASReview privacy-first active-learning framework](https://github.com/asreview/asreview)
