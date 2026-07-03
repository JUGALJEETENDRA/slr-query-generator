# LitSync Query Generator Improvement Baseline

Date: 2026-06-28

## Scope

This document captures the current query-generation pipeline before behavioral changes. The goal for the next iteration is a single benchmark-driven improvement, not an architecture redesign.

## Active Query Pipeline

The active Boolean query path is implemented by `server.py` in the `/generate` endpoint.

1. `extractor.py`
   - Uses a two-pass LLM extraction gateway.
   - Pass 1 isolates a literal comparator baseline.
   - Pass 2 slices the research question into `primary_paradigm`, `comparator_baseline`, `domain_context`, and `outcome_variables`.

2. `schema.py`
   - Defines `SLRExtractionContract` for extraction output.
   - Defines `SLRQueryContext` as the stable downstream five-facet schema: `technology`, `domain`, `comparison`, `context`, and `outcomes`.

3. `generator.py`
   - Expands extracted seed terms with LLM-generated bibliographic variants.
   - Runs facet-by-facet and filters terms labeled `RELATED_CONCEPT`.
   - Preserves the original modular facet arrays.

4. `acronym_expander.py`
   - Deterministically expands known acronyms across all facets.
   - Uses a fixed acronym map and regex word-boundary matching.

5. `classifier.py`
   - Classifies the current context into a primary domain using deterministic keyword rules.
   - The selected domain controls later ontology expansion.

6. `registries.py`
   - Injects implicit academic outcome layers using deterministic outcome rules.
   - Context-registry macro discipline injection is currently disabled, reducing broad contextual spill.

7. `ontology_expander.py`
   - Applies frozen domain-specific ontology packs.
   - Currently enforces a facet boundary guard: ontology terms are only added when the destination facet matches the source facet.

8. `comparator_registry.py`
   - Expands known comparator dualities and special comparison packs.
   - Adds terms only to the `comparison` facet.

9. `validator.py`
   - Applies final blacksets and negative ontology rules.
   - Removes known universal noise, outcome-only technology leaks, and active domain-denied terms.

10. `compiler.py`
    - Compiles non-empty facets into quoted OR blocks joined by AND.
    - Platform wrappers are added in `server.py` for Scopus, Web of Science, IEEE Xplore, and PubMed.

## Baseline Verification

Compilation command:

```powershell
python -m py_compile acronym_expander.py classifier.py compiler.py comparator_registry.py extractor.py generator.py ontology_expander.py registries.py schema.py server.py validator.py
```

Result: passed.

Fresh live benchmark status:

- `/generate` benchmark runner requires API server at `localhost:8000`.
- Query expansion requires Ollama-compatible model service at `localhost:11434`.
- Both ports were unreachable during this baseline pass, so a fresh live benchmark could not be produced.

Available historical benchmark artifacts:

- `archive/term_telemetry.json`
- `archive/manual_audit_workspace.json`
- `archive/benchmark_questions.json`
- `outputs/comparator_results.csv`

## Historical Telemetry Summary

From `archive/term_telemetry.json`:

| Category | Count |
| --- | ---: |
| pending_manual_audit | 650 |
| semantic_generalization | 69 |
| metric_inflation | 38 |
| related_concept | 17 |
| canonical_realization | 17 |
| comparator_leak | 8 |

By source stage:

| Source | Total | semantic_generalization | metric_inflation | related_concept | comparator_leak |
| --- | ---: | ---: | ---: | ---: | ---: |
| extractor.py | 240 | 31 | 14 | 7 | 3 |
| generator.py | 411 | 38 | 19 | 10 | 1 |
| ontology_expander.py | 98 | 0 | 1 | 0 | 4 |
| registries.py | 50 | 0 | 4 | 0 | 0 |

From `archive/manual_audit_workspace.json`:

| Manual Class | Count |
| --- | ---: |
| EXACT_SYNONYM | 32 |
| SEMANTIC_GENERALIZATION | 28 |
| NEAR_SYNONYM | 19 |
| RELATED_CONCEPT | 12 |
| CANONICAL_REALIZATION | 8 |
| METRIC_INFLATION | 1 |

From `outputs/comparator_results.csv` screening benchmark artifact:

- Rows: 999
- Decisions: REJECT 974, KEEP 14, MAYBE 11
- Average `technology_match`: 0.271
- Average `task_match`: 0.108
- Average `subject_match`: 0.209

## Current Failure Patterns

The dominant audited quality problem is semantic generalization. It appears mainly in:

- `generator.py`, where LLM expansion sometimes climbs from a specific concept to a broad parent discipline or system class.
- `extractor.py`, where initial facet extraction can include broad domain terms such as macro disciplines or generic systems.

Secondary issues:

- Metric inflation, mostly from LLM expansion and deterministic registry outcome additions.
- Related-concept contamination, mostly from LLM expansion and some extractor outputs.
- Comparator leakage is comparatively rare but high severity; ontology and extraction both contribute some cases.

## Recommended First Improvement Target

First target: reduce semantic generalization in `generator.py` output.

Reasoning:

- It is the largest non-pending historical failure category.
- `generator.py` is the largest contributor by count.
- The change can be isolated to one module.
- It does not require changing extraction, ontology routing, comparator handling, or compilation.

Candidate implementation shape for the next iteration:

- Add a deterministic post-filter in `generator.py` for broad macro-discipline and generic structural terms after LLM expansion.
- Keep it facet-local and conservative.
- Do not remove exact seed terms.
- Benchmark before and after using the same telemetry harness.

## Next Iteration Gate

Before implementing the first behavior change:

1. Start the local Ollama-compatible service on `localhost:11434`.
2. Start the FastAPI server on `localhost:8000`.
3. Run the archived benchmark harness or restore the active benchmark source if available.
4. Save the fresh baseline summary and telemetry.
5. Implement only the semantic-generalization filter.
6. Compile and rerun the same benchmark.
7. Keep the change only if semantic-generalization count drops without increasing comparator leakage, related-concept contamination, or compilation failures.
