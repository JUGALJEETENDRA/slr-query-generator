# API reference

The live OpenAPI interface at `http://127.0.0.1:8000/docs` is the authoritative schema. This page is a practical route map. JSON requests use `Content-Type: application/json`; uploads use `multipart/form-data`.

## Health and queries

| Method and route | Request | Purpose |
| --- | --- | --- |
| `GET /status` | — | Local runtime, Ollama, model, and collection readiness. |
| `GET /debug/hardware-profile` | optional `calibrate` query parameter | Hardware and resolved runtime profile. |
| `POST /generate` | `{"question":"...","processing_engine":"local"}` | Generate a query bundle. Engine is `local` or `gemini_web`. |

## Import and screening

| Method and route | Request | Purpose |
| --- | --- | --- |
| `POST /litsync` | multipart `files` (CSV/XLS/XLSX) | Normalize and deduplicate; returns `import_id`, counts, download URL, and PRISMA links. |
| `POST /screen_csv` | multipart `file`, `question`; optional criteria, context, engine, `max_rows`, `resume`, `import_id` | Start asynchronous screening. |
| `GET /progress?job_id=<id>` | — | Job progress and PRISMA snapshot. |
| `GET /screening_results?job_id=<id>` | — | Completed results or current job state. |
| `GET /maybe_papers?job_id=<id>` | — | Records marked `MAYBE`. |
| `PATCH /screening_jobs/{job_id}/records/{source_row_index}` | `{"decision":"KEEP","exclusion_reason":""}` | Save a manual decision. |
| `POST /screening-jobs/{job_id}/exports` | — | Create screening exports. |
| `GET /screening-jobs/{job_id}/exports/{export_name}` | — | Download a generated export. |
| `POST /finalize` | `{"job_id":"..."}` | Finalize after all `MAYBE` records are resolved. |

`/screen_csv` returns a `job_id` immediately. Poll `/progress` or `/screening_results` rather than assuming screening is complete.

## Validation and PRISMA

| Method and route | Request | Purpose |
| --- | --- | --- |
| `POST /gold_validation/sample` | `{"question":"...","job_id":"...","sample_size":60}` | Create a blinded sample from a completed job. |
| `POST /gold_validation/evaluate` | multipart `job_id`, `file` | Evaluate completed gold-label CSV. |
| `GET /prisma/{workflow_id}` | — | PRISMA JSON manifest. |
| `GET /prisma/{workflow_id}.csv` | — | PRISMA CSV. |
| `GET /prisma/{workflow_id}.svg` | — | PRISMA flow diagram. |

## Experimental collection

| Method and route | Request | Purpose |
| --- | --- | --- |
| `POST /experimental/collection-runs` | `{"research_question":"...","queries":{"pubmed":"..."},"limit":100}` | Create a collection run. |
| `GET /experimental/collection-runs/{run_id}` | — | Read run and source status. |
| `POST /experimental/collection-runs/{run_id}/sources/{source}/launch` | — | Launch a supported browser collection step. |
| `POST /experimental/collection-runs/{run_id}/sources/{source}/upload` | multipart `file` | Import a native source export. |
| `POST /experimental/collection-runs/{run_id}/sources/{source}/skip` | — | Mark a source skipped. |
| `POST /experimental/collection-runs/{run_id}/finalize` | — | Combine, deduplicate, and publish collection artifacts. |

Supported source keys: `pubmed`, `google_scholar`, `scopus`, `web_of_science`, and `ieee_xplore`. PubMed is the only native automated browser adapter; the other sources return guided visible-export instructions.
