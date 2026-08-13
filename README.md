# LitSync

LitSync is a local-first systematic-review application. It generates database
queries, imports and deduplicates paper records, screens titles and abstracts,
supports human validation workflows, and exports PRISMA 2020 artifacts.

## Production capabilities

- Local AI screening through one lightweight Qwen model in Ollama
- Gemini Web browser screening with validated batched decisions
- Isolated browser-assisted paper collection with native export audit trails
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

## Experimental collection

After generating database queries, create an experimental collection run.
PubMed has a native CSV browser adapter. Scopus, Web of Science, Google Scholar,
and IEEE Xplore use guided open/copy/export/upload workflows because subscription
layouts or access rules make unattended collection unsafe or unreliable. Exact queries, raw
exports, hashes, normalized records, and a resumable SQLite checkpoint are kept
under `private/experimental_collection/`; combined and deduplicated downloads
are published under `outputs/experimental-collection/`.

PubMed's automated CSV contains strong citation metadata but not abstracts;
upload a native NBIB export when abstract-complete PubMed records are required.

## Development

Install `requirements-dev.txt`, then run:

```powershell
python -m pytest
```

Production code lives under `litsync_app/`; `run_litsync.py` is the launcher and
`web/` contains the static application UI.
