# LitSync local-AI-first screening

LitSync now asks local models to understand the review question, understand each paper,
compare the two, and return criterion-level evidence. Deterministic code validates JSON,
criterion consistency, and exact title/abstract evidence references; it does not judge research relevance.

## Local resident three-layer runtime

The website's local screening path uses two small installed models and no 8B/14B model:

- Before paper screening, Qwen3 4B compiles the research question, optional Research
  Context, and explicit criteria into one compact screening guide. The long context is
  not repeated for every paper.
- Layer 1 runs `qwen2.5:3b` over batches of four papers. Evidence-backed, low-risk
  KEEP decisions finish here. Every REJECT plus every MAYBE, invalid, or risk-flagged
  decision enters the deep-review queue so a small-model false REJECT cannot be final.
- After the complete quick pass is checkpointed, LitSync unloads 3B and loads
  `qwen3:4b-instruct-2507-q4_K_M` once. Layer 2 reviews queued papers in batches of four.
- Layer 3 keeps the same Qwen 4B model resident and gives it a fresh adversarial prompt.
  It challenges Layer-2 MAYBE/invalid results and every risk-flagged definitive result
  without paying for another model swap.

The final critic does not see the deep-review prompt. It is explicitly instructed to
falsify the proposed decision and returns a complete replacement assessment.
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
