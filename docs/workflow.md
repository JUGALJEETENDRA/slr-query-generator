# LitSync workflow guide

## Create queries

Enter the research question in LitSync and select Local AI or Gemini Web when available. Preserve generated queries when collecting records so the audit trail reflects the actual search.

## Collect or import records

Use one of two entry points:

- **Experimental collection:** create a collection run from generated queries. PubMed opens a visible browser and requests a native CSV export. Google Scholar, Scopus, Web of Science, and IEEE Xplore provide guided steps; export on the visible site, then upload the native file.
- **Import and deduplicate:** upload CSV, XLS, or XLSX exports. LitSync combines, normalizes, and deduplicates them, then writes `clean_dataset.csv` under `outputs/imports/<import_id>/`.

Experimental collection accepts CSV, Excel, RIS, NBIB, and PubMed text exports. Finalizing a run publishes the clean dataset, provenance-preserving combined dataset, and audit manifest under `outputs/experimental-collection/<run_id>/`.

PubMed's automated CSV may omit abstracts. For abstract-complete screening, export and upload PubMed text or NBIB.

## Screen records

Upload a CSV with the research question, research context, and inclusion/exclusion criteria as appropriate. A screening job runs asynchronously and returns a `job_id`; use the progress and results views until it finishes. Only one screening job runs at a time.

Supported engines:

- `local` (default): local Ollama-based screening.
- `gemini_web`: browser-mediated Gemini Web workflow.

Use `max_rows` for a small protocol check before screening a large dataset.

## Resolve and validate decisions

Records are classified as `KEEP`, `MAYBE`, or `REJECT`. Resolve every `MAYBE` record before finalization. For quality control, create a blinded gold-validation sample from a completed job, label it outside LitSync, and upload the completed CSV for evaluation.

## Export and archive

Generate screening exports and finalize the job. The output includes the canonical screened CSV, included/excluded files, and PRISMA JSON, CSV, and SVG. Archive the exact queries, native exports, clean CSV, screening result, and PRISMA manifest together for reproducibility.
