from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import Field, model_validator

from .contracts import (
    CriterionAssessment,
    CriterionRelation,
    EvidenceCitation,
    EvidenceSource,
    StrictModel,
)


EVIDENCE_VALIDATION_VERSION = "local-v2-evidence-v1"

EvidenceIssueCode = Literal[
    "SOURCE_TEXT_MISSING",
    "EVIDENCE_ID_NOT_FOUND",
    "SOURCE_MISMATCH",
    "QUOTE_NOT_FOUND_IN_UNIT",
]

_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n[ \t]*\n+")
_WHITESPACE_RE = re.compile(r"\s+")
_DECISIVE_RELATIONS = {"DIRECT_SUPPORT", "DIRECT_CONTRADICTION"}


class EvidenceUnit(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=40)
    source: EvidenceSource
    text: str = Field(min_length=1, max_length=12000)
    start: int = Field(ge=0)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_offsets(self) -> "EvidenceUnit":
        if self.end <= self.start:
            raise ValueError("evidence unit end must be greater than start")
        if self.end - self.start != len(self.text):
            raise ValueError("evidence unit offsets must span the exact unit text")
        return self


class EvidenceMatch(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=40)
    source: EvidenceSource
    quote: str = Field(min_length=1, max_length=1200)
    unit_text: str = Field(min_length=1, max_length=12000)
    normalized_quote: str = Field(min_length=1, max_length=1200)
    normalized_start: int = Field(ge=0)
    normalized_end: int = Field(ge=1)
    occurrence_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_offsets(self) -> "EvidenceMatch":
        if self.normalized_end <= self.normalized_start:
            raise ValueError("evidence match end must be greater than start")
        if self.normalized_end - self.normalized_start != len(
            self.normalized_quote
        ):
            raise ValueError("evidence match offsets must span the normalized quote")
        return self


class EvidenceIssue(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=40)
    source: EvidenceSource
    code: EvidenceIssueCode
    message: str = Field(min_length=1, max_length=600)


class CitationValidation(StrictModel):
    citation: EvidenceCitation
    valid: bool
    match: EvidenceMatch | None = None
    issues: list[EvidenceIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_consistency(self) -> "CitationValidation":
        if self.valid and (self.match is None or self.issues):
            raise ValueError("valid citation validation requires a match and no issues")
        if not self.valid and (self.match is not None or not self.issues):
            raise ValueError("invalid citation validation requires issues and no match")
        return self


class AssessmentEvidenceValidation(StrictModel):
    criterion_id: str = Field(min_length=1, max_length=80)
    original_relation: CriterionRelation
    sanitized_assessment: CriterionAssessment
    citation_results: list[CitationValidation] = Field(default_factory=list)
    valid_evidence_ids: list[str] = Field(default_factory=list)
    issues: list[EvidenceIssue] = Field(default_factory=list)
    safe_downgrade: bool = False


class EvidenceBatchValidation(StrictModel):
    validation_version: Literal[EVIDENCE_VALIDATION_VERSION] = (
        EVIDENCE_VALIDATION_VERSION
    )
    evidence_units: list[EvidenceUnit] = Field(default_factory=list)
    assessments: list[CriterionAssessment] = Field(default_factory=list)
    assessment_results: list[AssessmentEvidenceValidation] = Field(
        default_factory=list
    )
    issues: list[EvidenceIssue] = Field(default_factory=list)
    issue_count: int = Field(ge=0)
    invalid_citation_count: int = Field(ge=0)
    safe_downgrade_count: int = Field(ge=0)
    all_citations_valid: bool


def normalize_evidence_text(value: str | None) -> str:
    """Normalize representation-only variation while preserving exact wording.

    Unicode compatibility normalization and whitespace collapse make line-wrapped CSV/PDF
    text comparable. Case, punctuation, token order, and wording remain unchanged.
    """

    normalized = unicodedata.normalize("NFKC", value or "")
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _trimmed_segment(text: str, start: int, end: int) -> tuple[str, int, int] | None:
    raw = text[start:end]
    if not raw or not raw.strip():
        return None
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw) - len(raw.rstrip())
    exact_start = start + leading
    exact_end = end - trailing if trailing else end
    return text[exact_start:exact_end], exact_start, exact_end


def _source_units(source: EvidenceSource, text: str | None) -> list[EvidenceUnit]:
    value = str(text or "")
    segments: list[tuple[str, int, int]] = []
    segment_start = 0
    for match in _SPLIT_RE.finditer(value):
        segment = _trimmed_segment(value, segment_start, match.start())
        if segment is not None:
            segments.append(segment)
        segment_start = match.end()
    tail = _trimmed_segment(value, segment_start, len(value))
    if tail is not None:
        segments.append(tail)

    return [
        EvidenceUnit(
            evidence_id=f"{source}_{index:03d}",
            source=source,
            text=unit_text,
            start=start,
            end=end,
        )
        for index, (unit_text, start, end) in enumerate(segments, start=1)
    ]


def build_evidence_units(
    title: str | None,
    abstract: str | None,
) -> list[EvidenceUnit]:
    """Build stable, exact title/abstract units that models can cite by id."""

    return _source_units("title", title) + _source_units("abstract", abstract)


def evidence_lookup(
    title: str | None,
    abstract: str | None,
) -> dict[str, EvidenceUnit]:
    return {
        unit.evidence_id: unit for unit in build_evidence_units(title, abstract)
    }


def _issue(
    citation: EvidenceCitation,
    code: EvidenceIssueCode,
    message: str,
) -> EvidenceIssue:
    return EvidenceIssue(
        evidence_id=citation.evidence_id,
        source=citation.source,
        code=code,
        message=message,
    )


def _validate_citation_against_units(
    citation: EvidenceCitation,
    *,
    title: str | None,
    abstract: str | None,
    unit_by_id: dict[str, EvidenceUnit],
) -> CitationValidation:
    source_value = title if citation.source == "title" else abstract
    if not normalize_evidence_text(source_value):
        issue = _issue(
            citation,
            "SOURCE_TEXT_MISSING",
            f"The declared {citation.source} source is empty or unavailable.",
        )
        return CitationValidation(citation=citation, valid=False, issues=[issue])

    unit = unit_by_id.get(citation.evidence_id)
    if unit is None:
        issue = _issue(
            citation,
            "EVIDENCE_ID_NOT_FOUND",
            f"Evidence id {citation.evidence_id!r} does not resolve to a source unit.",
        )
        return CitationValidation(citation=citation, valid=False, issues=[issue])

    if unit.source != citation.source:
        issue = _issue(
            citation,
            "SOURCE_MISMATCH",
            (
                f"Evidence id {citation.evidence_id!r} resolves to {unit.source}, "
                f"not declared source {citation.source}."
            ),
        )
        return CitationValidation(citation=citation, valid=False, issues=[issue])

    normalized_unit = normalize_evidence_text(unit.text)
    normalized_quote = normalize_evidence_text(citation.quote)
    start = normalized_unit.find(normalized_quote)
    if start < 0:
        issue = _issue(
            citation,
            "QUOTE_NOT_FOUND_IN_UNIT",
            (
                "The normalized quotation is not an exact contiguous substring of "
                f"evidence unit {citation.evidence_id!r}."
            ),
        )
        return CitationValidation(citation=citation, valid=False, issues=[issue])

    match = EvidenceMatch(
        evidence_id=citation.evidence_id,
        source=citation.source,
        quote=citation.quote,
        unit_text=unit.text,
        normalized_quote=normalized_quote,
        normalized_start=start,
        normalized_end=start + len(normalized_quote),
        occurrence_count=normalized_unit.count(normalized_quote),
    )
    return CitationValidation(citation=citation, valid=True, match=match)


def validate_citation(
    citation: EvidenceCitation,
    *,
    title: str | None,
    abstract: str | None,
) -> CitationValidation:
    """Validate one citation against its stable unit, source, and exact wording."""

    return _validate_citation_against_units(
        citation,
        title=title,
        abstract=abstract,
        unit_by_id=evidence_lookup(title, abstract),
    )


def _assemble_assessment_result(
    assessment: CriterionAssessment,
    citation_results: list[CitationValidation],
) -> AssessmentEvidenceValidation:
    valid_citations = [
        result.citation for result in citation_results if result.valid
    ]
    issues = [issue for result in citation_results for issue in result.issues]
    safe_downgrade = (
        assessment.relation in _DECISIVE_RELATIONS and not valid_citations
    )

    if safe_downgrade:
        sanitized = assessment.model_copy(
            update={
                "relation": "MISSING_OR_UNCLEAR",
                "rationale": (
                    "The decisive relation was downgraded because no citation "
                    "matched its evidence unit and declared source exactly."
                ),
                "evidence": [],
            }
        )
    else:
        sanitized = assessment.model_copy(update={"evidence": valid_citations})

    return AssessmentEvidenceValidation(
        criterion_id=assessment.criterion_id,
        original_relation=assessment.relation,
        sanitized_assessment=sanitized,
        citation_results=citation_results,
        valid_evidence_ids=[item.evidence_id for item in valid_citations],
        issues=issues,
        safe_downgrade=safe_downgrade,
    )


def validate_assessment_evidence(
    assessment: CriterionAssessment,
    *,
    title: str | None,
    abstract: str | None,
) -> AssessmentEvidenceValidation:
    """Validate and sanitize all evidence attached to one criterion assessment."""

    unit_by_id = evidence_lookup(title, abstract)
    citation_results = [
        _validate_citation_against_units(
            citation,
            title=title,
            abstract=abstract,
            unit_by_id=unit_by_id,
        )
        for citation in assessment.evidence
    ]
    return _assemble_assessment_result(assessment, citation_results)


def validate_assessments_evidence(
    assessments: list[CriterionAssessment],
    *,
    title: str | None,
    abstract: str | None,
) -> EvidenceBatchValidation:
    """Validate a paper's assessments without changing their input ordering."""

    units = build_evidence_units(title, abstract)
    unit_by_id = {unit.evidence_id: unit for unit in units}
    assessment_results: list[AssessmentEvidenceValidation] = []

    for assessment in assessments:
        citation_results = [
            _validate_citation_against_units(
                citation,
                title=title,
                abstract=abstract,
                unit_by_id=unit_by_id,
            )
            for citation in assessment.evidence
        ]
        assessment_results.append(
            _assemble_assessment_result(assessment, citation_results)
        )

    issues = [
        issue
        for assessment_result in assessment_results
        for issue in assessment_result.issues
    ]
    invalid_citation_count = sum(
        not citation_result.valid
        for assessment_result in assessment_results
        for citation_result in assessment_result.citation_results
    )
    safe_downgrade_count = sum(
        assessment_result.safe_downgrade
        for assessment_result in assessment_results
    )

    return EvidenceBatchValidation(
        evidence_units=units,
        assessments=[
            assessment_result.sanitized_assessment
            for assessment_result in assessment_results
        ],
        assessment_results=assessment_results,
        issues=issues,
        issue_count=len(issues),
        invalid_citation_count=invalid_citation_count,
        safe_downgrade_count=safe_downgrade_count,
        all_citations_valid=invalid_citation_count == 0,
    )
