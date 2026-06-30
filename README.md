# SLR Query Generator

The Query Generator supports two modes:

- **Local** runs the existing Ollama/Qwen extraction and query pipeline.
- **Gemini Website** generates the base Boolean strategy through the reusable
  Playwright browser profile, then formats it for Google Scholar, Scopus, Web of
  Science, IEEE Xplore, and PubMed. It does not require an API key.

The CSV screener has two modes:

- **Local AI** sends every paper to the existing Ollama/Qwen semantic screener.
- **Hybrid** embeds the research question once with `nomic-embed-text`, filters titles
  by cosine similarity, then screens the surviving papers in batches through the
  Gemini website with Playwright. It does not use a Gemini API key or Google SDK.

## Setup

```powershell
pip install -r requirements.txt
playwright install chromium
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Start Ollama, then run `python server.py` (or `start.bat`). The first Hybrid run opens
a Chromium window. Sign in to Gemini there if required, then run screening again.
The login is reused from `%USERPROFILE%\.slr-query-generator\gemini-profile` by
default. Set `GEMINI_PROFILE_DIR` to use a different location.

Hybrid defaults to a cosine threshold of `0.35` and batches of 10. Both values can
be changed in the web UI or CLI:

```powershell
python bulk_screen.py papers.csv "Your research question" --mode hybrid --threshold 0.35 --batch-size 10
```

Every run writes `screened.csv`, `included_studies.csv`, `excluded_studies.csv`,
`maybe_studies.csv`, and `summary.csv` to `outputs/`. Category files are created even
when empty, so download links remain predictable.
