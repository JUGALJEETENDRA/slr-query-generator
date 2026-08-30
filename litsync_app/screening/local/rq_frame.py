from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Any

from litsync_app.query.generator import decompose_literal_question

from .contracts import (
    RQAllowedVariant, RQConceptGroup, RQ_FRAME_VERSION,
    ScreeningRQFrame,
)


def question_fingerprint(question: str) -> str:
    return sha256(str(question or "").strip().encode("utf-8")).hexdigest()


def _group_id(label: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    return (slug or f"concept_{index + 1}")[:80]


def _fallback(
    question: str, inclusion: str, exclusion: str, context: str,
    failures: list[str] | None = None, frame_version: str = RQ_FRAME_VERSION,
) -> ScreeningRQFrame:
    draft = decompose_literal_question(question)
    groups = [
        RQConceptGroup(
            id=_group_id(group.label, index), label=group.label, role=group.role,
            required=True, group_relationship="AND", source_spans=group.source_spans,
        )
        for index, group in enumerate(draft.groups)
    ]
    if not groups:
        groups = [RQConceptGroup(
            id="research_question", label="Research question", role="other",
            required=True, source_spans=[question.strip()],
        )]
    required = list(dict.fromkeys(span for group in groups for span in group.source_spans))
    return ScreeningRQFrame(
        frame_version=frame_version,
        question=question.strip(), question_fingerprint=question_fingerprint(question),
        groups=groups, required_concepts=required,
        research_context=context, inclusion_criteria=inclusion, exclusion_criteria=exclusion,
        ambiguities=list(draft.uncertain_terms),
        forbidden_broadening_warnings=[
            "Do not substitute a related technology, population, setting, task, or outcome for the requested one.",
            "Do not treat background discussion or separately mentioned concepts as the required relationship.",
            "Allowed variants are interpretation aids and must not broaden eligibility.",
        ],
        source="parser_fallback", status="fallback",
        generation_status="parser_fallback", validation_failures=list(failures or []),
        provenance={"method": "lossless_source_span_parser"},
    ).with_identity()


def build_screening_rq_frame(
    question: str,
    *,
    inclusion: str = "",
    exclusion: str = "",
    context: str = "",
    submitted: str | dict[str, Any] | None = None,
    frame_version: str = RQ_FRAME_VERSION,
) -> ScreeningRQFrame:
    question = str(question or "").strip()
    if not question:
        raise ValueError("research question is required")
    if not submitted:
        return _fallback(question, inclusion, exclusion, context, frame_version=frame_version)
    try:
        concepts = json.loads(submitted) if isinstance(submitted, str) else dict(submitted)
        supplied_question = str(concepts.get("question") or question).strip()
        supplied_fingerprint = str(concepts.get("question_fingerprint") or question_fingerprint(supplied_question))
        if supplied_question != question or supplied_fingerprint != question_fingerprint(question):
            raise ValueError("submitted RQ structure does not match the current research question")
        raw_groups = concepts.get("groups") or []
        if not raw_groups:
            raise ValueError("submitted RQ structure has no concept groups")
        groups: list[RQConceptGroup] = []
        by_label: dict[str, str] = {}
        used_ids: set[str] = set()
        for index, raw in enumerate(raw_groups):
            spans = [str(value).strip() for value in raw.get("source_spans") or [] if str(value).strip()]
            if not spans or any(span.casefold() not in question.casefold() for span in spans):
                raise ValueError(f"concept group {index + 1} has a source span absent from the RQ")
            group_id = _group_id(str(raw.get("label") or ""), index)
            if group_id in used_ids:
                group_id = f"{group_id[:70]}_{index + 1}"
            used_ids.add(group_id)
            group = RQConceptGroup(
                id=group_id, label=str(raw.get("label") or f"Concept {index + 1}"),
                role=raw.get("role", "other"), required=True,
                group_relationship="AND", source_spans=spans,
            )
            groups.append(group)
            by_label[group.label.casefold()] = group.id
        variants: list[RQAllowedVariant] = []
        for detail in concepts.get("term_details") or []:
            group_id = by_label.get(str(detail.get("group") or "").casefold())
            if not group_id:
                raise ValueError("term provenance references an unknown concept group")
            variants.append(RQAllowedVariant(
                term=str(detail.get("term") or ""), group_id=group_id,
                source=detail.get("source"),
                supporting_paper_ids=[str(value) for value in detail.get("supporting_paper_ids") or []],
                advisory_only=True,
            ))
        if any(item.source == "corpus" and not item.supporting_paper_ids for item in variants):
            raise ValueError("corpus-derived variants require supporting paper IDs")
        required = list(dict.fromkeys(span for group in groups for span in group.source_spans))
        literal = {value.casefold() for value in required}
        advisory = list(dict.fromkeys(item.term for item in variants if item.term.casefold() not in literal))
        return ScreeningRQFrame(
            frame_version=frame_version,
            question=question, question_fingerprint=question_fingerprint(question),
            groups=groups, required_concepts=required, advisory_concepts=advisory,
            allowed_variants=variants, research_context=context,
            inclusion_criteria=inclusion, exclusion_criteria=exclusion,
            ambiguities=[str(value) for value in concepts.get("uncertain_terms") or []],
            forbidden_broadening_warnings=[
                "Do not substitute a related technology, population, setting, task, or outcome for the requested one.",
                "Do not treat background discussion or separately mentioned concepts as the required relationship.",
                "Allowed variants are interpretation aids and must not broaden eligibility.",
            ],
            source="generated_query", status="validated",
            generation_model=str(concepts.get("model") or ""),
            generation_status=str(concepts.get("generation_status") or ""),
            generation_fallback_reason=str(concepts.get("fallback_reason") or ""),
            provenance={
                "term_details": concepts.get("term_details") or [],
                "grounded_terms": concepts.get("grounded_terms") or [],
                "grounding_papers": concepts.get("grounding_papers") or [],
                "literal_coverage": concepts.get("literal_coverage"),
            },
        ).with_identity()
    except Exception as exc:
        return _fallback(
            question, inclusion, exclusion, context, [str(exc)], frame_version=frame_version
        )
