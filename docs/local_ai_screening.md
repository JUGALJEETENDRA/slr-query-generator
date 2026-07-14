# LitSync local-AI-first screening

LitSync now asks local models to understand the review question, understand each paper,
compare the two, and return criterion-level evidence. Deterministic code validates JSON,
criterion consistency, and exact title/abstract evidence references; it does not judge research relevance.

## Local cross-model three-layer runtime

The website's local screening path uses three small installed models and no 8B/14B model:

- Before paper screening, Qwen3 4B compiles the research question, optional Research
  Context, and explicit criteria into one compact screening guide. The long context is
  not repeated for every paper.
- Layer 1 runs `qwen2.5:3b` over batches of eight papers. Evidence-backed, low-risk
  KEEP and REJECT decisions finish here. MAYBE, invalid, and risk-flagged decisions
  enter the deep-review queue.
- After the complete quick pass is checkpointed, LitSync unloads 3B and loads
  `qwen3:4b-instruct-2507-q4_K_M` once. Layer 2 reviews queued papers in batches of four.
- Layer 3 unloads Qwen and loads `phi4-mini:3.8b-q4_K_M` once. This independent model
  challenges every Layer-2 REJECT/MAYBE/invalid result and every risk-flagged KEEP.

The final critic belongs to a different model family so it is less likely to repeat
Qwen's exact misunderstanding. It returns a complete replacement assessment.
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
ollama pull phi4-mini:3.8b-q4_K_M
```

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
