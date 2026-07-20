# Corpus-grounded Boolean query generation

Date: 2026-07-20

## Research position

LitSync's experimental contribution is a latency-bounded, local-first, cross-domain Boolean query
generator that admits expansion terms only when they are supported by retrieved academic metadata.
It combines, but does not claim to reproduce, three research lines:

- AutoBool optimizes Boolean formulation against retrieval rewards, but its released models target
  biomedical PubMed search: https://aclanthology.org/2026.eacl-long.68/
- Corpus-Steered Query Expansion grounds expansion in initially retrieved documents and reports its
  largest benefits where the LLM lacks knowledge: https://aclanthology.org/2024.eacl-short.34/
- MuGI and Query2doc use generated pseudo-references for expansion, without LitSync's Boolean
  validity, source-provenance, or cross-domain constraints:
  https://aclanthology.org/2024.findings-emnlp.103/ and
  https://aclanthology.org/2023.emnlp-main.585/

The related-work search did not identify a prior system combining all four properties: cross-domain
SLR Boolean generation, corpus-verified term admission, local 4B inference, and a hard interactive
deadline. This is a working research gap, not a publication novelty claim; it must be re-audited
immediately before submission.

## Implemented method

1. Remove embedded review/interrogative framing, retain its topical scope, and split paragraph-style
   questions into lossless spans using general grammatical relations and coordination.
2. Generate schema-validated 2–4 concept groups without visible reasoning.
3. Ask the installed Qwen 3 4B model to assign roles and return verbatim `source_spans`, while
   Semantic Scholar retrieves eight title/abstract records in parallel.
4. Merge recurring corpus phrases deterministically, even when local inference fails or times out.
5. Require adjacent terms to occur in at least two papers and record their supporting paper IDs.
6. Reject nonliteral model terms unless they are mechanically verifiable acronyms or small
   source-linked spelling corrections; rebuild drafts whenever the model drops literal coverage.
7. Compile exact and prefix terms using platform-specific syntax and return term provenance,
   `generation_status`, `literal_coverage`, repaired spans, correction provenance, and a degraded-mode
   warning in `concepts`.
8. Reserve the final three seconds for grounding, validation, and compilation; never retry a failed
   model call.

## Evaluation caution

The 120-case XDQ manifest is suitable immediately for syntax, concept-preservation, contamination,
and latency experiments. Retrieval recall, F3, and WSS@95 are not valid until the hard cases receive
human-curated or published-review gold papers. Top API results are candidates, not relevance labels.
