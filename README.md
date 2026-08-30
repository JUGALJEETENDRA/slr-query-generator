# LitSync

LitSync is a local-first application for systematic literature reviews. It turns a research question into database-specific queries, helps collect and normalize records, screens titles and abstracts, supports blinded validation, and produces traceable PRISMA 2020 artifacts.

It runs at `http://127.0.0.1:8000`; review data, browser profiles, and generated artifacts stay in local runtime directories.

## What it does

- Generate balanced and high-recall database queries.
- Import CSV, XLS, and XLSX files, normalize records, and remove duplicates.
- Screen CSV records with local Ollama models or the guided Gemini Web workflow.
- Review `MAYBE` records, create blinded validation samples, and finalize results.
- Export screening CSVs plus PRISMA JSON, CSV, and SVG flow diagrams.
- Run auditable experimental collection: PubMed has a visible Playwright-assisted native CSV export; Google Scholar, Scopus, Web of Science, and IEEE Xplore use guided native-export uploads.

## Quick start

Prerequisites: Python 3.10+ and, for local AI features, [Ollama](https://ollama.com/). Playwright is needed only for browser-assisted collection.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m playwright install chromium
ollama pull qwen3.5:4b
python run_litsync.py
```

On Windows, `start.bat` is an alternative launcher after Python and dependencies are installed. Open `http://127.0.0.1:8000` after startup. Interactive API documentation is at `/docs`.

If you change a configured model name, pull that model in Ollama before screening. Check local readiness at `http://127.0.0.1:8000/status`.

## Typical workflow

1. Enter a research question and generate database queries.
2. Collect native exports with the experimental collection workflow or upload existing exports for normalization and deduplication.
3. Start title/abstract screening from a CSV with the research question and criteria.
4. Monitor the job, review unresolved `MAYBE` records, and save manual decisions.
5. Optionally complete a blinded gold-validation sample.
6. Generate screening exports, finalize the review, and download PRISMA artifacts.

The collection workflow never attempts to bypass authentication, CAPTCHA, or database access controls. For subscription services, use the visible site to export native results, then upload them to LitSync.

## Configuration and local data

Copy `.env.example` to `.env` only when you need to override defaults or use an authorized integration. `OLLAMA_BASE_URL` defaults to `http://localhost:11434`; local screening defaults to `qwen3.5:4b`.

Runtime content is ignored by Git:

- `uploads/` — temporary inputs.
- `outputs/` — cleaned datasets, screening runs, exports, and PRISMA files.
- `private/` — collection checkpoints, raw exports, browser profiles, and blinded validation material.

## Documentation

- [Workflow guide](docs/workflow.md) — collection, import, screening, validation, and exports.
- [Operations and configuration](docs/operations.md) — setup, model settings, local data, and troubleshooting.
- [API reference](docs/api.md) — route summary and request formats. The live OpenAPI UI is at `/docs`.

## Development

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

Application code is in `litsync_app/`, the launcher is `run_litsync.py`, and the static UI is in `web/`.
