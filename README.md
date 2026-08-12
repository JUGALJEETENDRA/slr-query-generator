# LitSync

LitSync is a local-first systematic-review application. It generates database
queries, imports and deduplicates paper records, screens titles and abstracts,
supports human validation workflows, and exports PRISMA 2020 artifacts.

## Production capabilities

- Local AI screening through one lightweight Qwen model in Ollama
- Gemini Web browser screening with validated batched decisions
- Agentic research-question generation and paper collection through Skyvern
- CSV/XLS/XLSX normalization and deduplication
- Blinded gold-label validation for completed screening jobs
- PRISMA JSON, CSV, and SVG exports

## Setup

1. Create and activate a Python virtual environment outside the repository or
   in `.venv/`.
2. Install `requirements.txt`.
3. Run `python -m playwright install chromium`.
4. Install and start Ollama, then run `ollama pull qwen3.5:4b`.
5. Copy `.env.example` to `.env` and configure only the integrations you use.
6. Run `python run_litsync.py` or `start.bat` on Windows.

The application listens on `http://127.0.0.1:8000`. Runtime uploads, outputs,
browser profiles, and private state are created on demand and are ignored by
Git.

## Agentic collection

Set `SKYVERN_API_KEY` and the applicable `SKYVERN_CREDENTIAL_*` vault IDs.
LitSync proposes and critiques research questions, generates database-specific
queries, collects records, deduplicates them, runs local screening, and
checkpoints recoverable state in `private/agentic_runs.sqlite3`. It stops for
login, MFA, CAPTCHA, subscription, or access-control blockers rather than
attempting to bypass them.

## Development

Install `requirements-dev.txt`, then run:

```powershell
python -m pytest
```

Production code lives under `litsync_app/`; `run_litsync.py` is the launcher and
`web/` contains the static application UI.
