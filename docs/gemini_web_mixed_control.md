# Gemini Web mixed-control check

This diagnostic checks whether Gemini Web distinguishes clearly relevant papers from clear negatives. It does not
expect or enforce a particular decision distribution in normal screening.

## Prepared files

- Screening input: `.benchmarks/gemini_web_mixed_control_40.csv`
- Blinded labels: `.benchmarks/gemini_web_mixed_control_40.gold.csv`

The screening input contains only `Title` and `Abstract`. The gold sidecar is never uploaded to Gemini.

## Real browser run

Start the application:

```powershell
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

At `http://127.0.0.1:8000`, upload `.benchmarks/gemini_web_mixed_control_40.csv`, select Gemini Web, and use:

> How can blockchain technology improve transparency, traceability, trust, and security in supply chain management?

Leave optional inclusion and exclusion criteria empty. Record the displayed wall-clock runtime and retry count, then
download the completed screening CSV.

## Score the export

```powershell
python -m evaluation.gemini_web_mixed_control score --screened "PATH_TO_DOWNLOADED_SCREENING.csv" --gold ".benchmarks\gemini_web_mixed_control_40.gold.csv" --runtime-seconds DISPLAYED_RUNTIME --retry-count DISPLAYED_RETRIES --report-out ".benchmarks\gemini_web_mixed_control_40.report.json"
```

The check passes when every row is present and structurally valid, no clear negative is kept, and no clear positive
is rejected. MAYBE is allowed for negative controls when the abstract cannot safely establish a definitive rejection.
Any false KEEP is reported with its rationale and cited evidence for review.

Rebuild the fixture, if needed, with:

```powershell
python -m evaluation.gemini_web_mixed_control build --positive "uploads\6a19d17c-2649-4f01-a886-4de75edc7748-LitSync_Clean_Dataset_2026-06-13 (1) (2).csv" --negative "data\holdout\LitSync_Clean_Dataset_2026-07-09.csv" --papers-out ".benchmarks\gemini_web_mixed_control_40.csv" --gold-out ".benchmarks\gemini_web_mixed_control_40.gold.csv" --per-group 20
```
