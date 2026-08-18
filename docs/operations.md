# Operations and configuration

## Prerequisites

- Python 3.10 or later.
- Ollama for local query generation and screening.
- Chromium installed with `python -m playwright install chromium` for visible PubMed collection.

Run `python run_litsync.py`. LitSync binds to `127.0.0.1:8000`, so it is reachable only from the local computer. Use `/status` to verify Ollama connectivity and the configured model.

## Environment variables

Create `.env` from `.env.example` only when needed. Never commit it.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama address. |
| `LOCAL_AI_MODEL` | `qwen3.5:4b` | Primary local screening model. |
| `LOCAL_AI_REVIEW_MODEL` | `qwen3.5:9b` | Review-model override where enabled. |
| `LOCAL_QUERY_MODEL` | automatic selection | Explicit local query-generation model. |
| `LOCAL_TRIAGE_MODEL` | `qwen2.5:3b` | Triage model for configurable profiles. |
| `LOCAL_DEEP_MODEL` | `qwen3:4b-instruct-2507-q4_K_M` | Deep-review model for configurable profiles. |
| `LOCAL_EDGE_MODEL` | value of `LOCAL_DEEP_MODEL` | Edge-case review model. |
| `LOCAL_AI_TIMEOUT_SECONDS` | `120` | Local-model request timeout. |
| `LOCAL_AI_CACHE_PATH` | `outputs/cache/local_ai` | Local screening cache. |
| `GEMINI_WEB_PROFILE_DIR` | `browser_profiles/gemini` | Persistent Gemini Web browser profile. |
| `GEMINI_WEB_READY_TIMEOUT_MS` | `120000` | Browser readiness timeout. |
| `GEMINI_WEB_RESPONSE_TIMEOUT_MS` | `120000` | Browser response timeout. |

`MODEL_TIER`, `RESOURCE_PROFILE`, and `*_FAST_MODEL` / `*_STRONG_MODEL` are compatibility settings. Consult `/status` for the current resolved local runtime.

## Data and backup

`uploads/`, `outputs/`, and `private/` are created automatically and ignored by Git. Back up `outputs/` for finished reviews. To resume experimental collection, preserve the relevant `private/experimental_collection/` run directory; it contains checkpoints, queries, raw exports, and normalized records.

Do not share `.env` or browser-profile directories: they may contain authorized credentials or signed-in sessions.

## Troubleshooting

- **Ollama needs attention:** start Ollama, run `ollama list`, and pull the model shown in `required_models` or change the matching variable and restart LitSync.
- **Chromium fails to launch:** rerun `python -m playwright install chromium` inside the environment used by LitSync.
- **Collection needs attention:** finish the visible source search and upload its native export. LitSync does not bypass CAPTCHA, authentication, or source restrictions.
- **Screening cannot start:** another job may be active; inspect `/progress?job_id=<job_id>`.
- **Finalization fails:** resolve all `MAYBE` decisions and retry.
