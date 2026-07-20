# PRISMA 2020 title/abstract flow

LitSync uses the current general PRISMA standard, **PRISMA 2020**, and adapts the
official databases/registers flow to the stages the application actually records.
The reference is the [official PRISMA 2020 flow-diagram page](https://www.prisma-statement.org/prisma-2020-flow-diagram).

LitSync records identification, actual deduplication and preprocessing removals,
title/abstract screening, manual resolution of uncertain records, and provisional
inclusion after title/abstract screening. It does not perform report retrieval,
full-text eligibility assessment, synthesis, or study/report disambiguation. Those
boxes are therefore omitted from the SVG and represented as `null` with
`full_text_stage_status="not_performed"` in the reproducibility manifest.

`prisma_flow.Prisma2020Manifest` is the only count authority. Screening engines
write decisions through the shared screening session, and API/UI consumers receive
snapshots from that service. Browser code must not derive or repair counts.

Counting invariants:

- `records_screened = KEEP + MAYBE + REJECT`;
- `REJECT` alone is excluded;
- `MAYBE` is awaiting manual review and is neither included nor excluded;
- `KEEP` is provisional inclusion after title/abstract screening;
- pre-screen automation removals never include screening decisions;
- non-classified exclusion reasons are reported as `reason not classified`.

The JSON state is persisted under `outputs/prisma/`. Finalization re-reads the
screened, included, and excluded CSV files and sets `csv_counts_match=true` only
when their row counts equal the server-owned decisions. SVG, JSON, and flattened
CSV exports are available from `/prisma/{workflow_id}` and its `.svg`/`.csv`
variants. The SVG attribution is “Adapted from PRISMA 2020, CC BY 4.0.”
