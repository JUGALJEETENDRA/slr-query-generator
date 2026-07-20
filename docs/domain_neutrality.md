# LitSync domain-neutrality policy

Production behavior must work from supplied evidence rather than a hand-written subject
ontology. A query term may enter output only as a literal source span, deterministic morphology,
a mechanically verified source acronym, a source-linked typo correction, validated model output,
or a corpus term with supporting paper IDs.

Production modules may use general grammar, evidence validation, Boolean syntax, platform
formatting, operational limits, and generic roles. They must not import benchmark fixtures,
archived interfaces, or historical ontology experiments. Screening decisions follow the user's
protocol and cited paper evidence; diagnostic sampling distributions never feed back into them.

Topic names are valid data in research questions, retrieved papers, and test or benchmark
fixtures. Their presence there does not authorize runtime branches, forced synonyms, registries,
or decision rules.

## Historical baseline

`experiments.legacy_query_pipeline` reproduces the former registry-based workflow for research
comparison. Its hand-written domain packs are excluded from the server and default registry. It
is enabled only by an explicit benchmark flag or historical strategy ID. `litsync_workflow` is a
deprecated compatibility alias.
