from __future__ import annotations

from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "2.0"
PROMPT_VERSION = "local-semantic-boundary-v3.12"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


RQ_FRAME_VERSION = "local-rq-frame-v1"
RQ_FRAME_VERSION_V2 = "local-rq-frame-v2"


class RQConceptGroup(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=80)
    role: Literal["technology", "population", "domain", "comparison", "outcome", "context", "other"]
    required: bool = True
    group_relationship: Literal["AND", "OR", "ADVISORY"] = "AND"
    term_relationship: Literal["OR"] = "OR"
    source_spans: list[str] = Field(min_length=1, max_length=8)


class RQAllowedVariant(StrictModel):
    term: str = Field(min_length=1, max_length=160)
    group_id: str = Field(min_length=1, max_length=80)
    source: Literal[
        "literal", "morphology", "source_acronym", "typo_correction",
        "validated_model", "corpus",
    ]
    supporting_paper_ids: list[str] = Field(default_factory=list, max_length=12)
    advisory_only: bool = True


class ScreeningRQFrame(StrictModel):
    frame_version: str = RQ_FRAME_VERSION
    frame_id: str = ""
    question: str = Field(min_length=3)
    question_fingerprint: str = Field(min_length=16, max_length=64)
    groups: list[RQConceptGroup] = Field(min_length=1, max_length=8)
    required_concepts: list[str] = Field(default_factory=list, max_length=32)
    advisory_concepts: list[str] = Field(default_factory=list, max_length=32)
    allowed_variants: list[RQAllowedVariant] = Field(default_factory=list, max_length=64)
    research_context: str = Field(default="", max_length=4000)
    inclusion_criteria: str = Field(default="", max_length=4000)
    exclusion_criteria: str = Field(default="", max_length=4000)
    ambiguities: list[str] = Field(default_factory=list, max_length=12)
    forbidden_broadening_warnings: list[str] = Field(default_factory=list, max_length=8)
    source: Literal["generated_query", "parser_fallback"]
    status: Literal["validated", "fallback"]
    generation_model: str = ""
    generation_status: str = ""
    generation_fallback_reason: str = ""
    validation_failures: list[str] = Field(default_factory=list, max_length=16)
    provenance: dict[str, Any] = Field(default_factory=dict)

    def with_identity(self) -> "ScreeningRQFrame":
        payload = self.model_dump(exclude={"frame_id"}, mode="json")
        digest = sha256(str(sorted(payload.items())).encode("utf-8")).hexdigest()[:16]
        return self.model_copy(update={"frame_id": digest})

    def compact_prompt_payload(self, *, triage: bool = False) -> dict[str, Any]:
        payload = {
            "frame_id": self.frame_id,
            "original_question": self.question,
            "groups": [
                {
                    "id": group.id, "role": group.role, "required": group.required,
                    "relationship": group.group_relationship,
                    "source_spans": group.source_spans,
                }
                for group in self.groups
            ],
            "forbidden_broadening_warnings": self.forbidden_broadening_warnings[:3 if triage else 8],
        }
        if self.frame_version == RQ_FRAME_VERSION:
            # Preserve the v4.0 prompt contract for reproducible comparisons.
            payload["required_concepts"] = self.required_concepts
        else:
            payload["required_relationship"] = {
                "between_groups": "AND",
                "within_each_group": "OR",
                "instruction": "Preserve the relationship expressed by the original question.",
            }
        if not triage:
            payload.update({
                "advisory_concepts": self.advisory_concepts,
                "allowed_variants": [item.model_dump(mode="json") for item in self.allowed_variants],
                "ambiguities": self.ambiguities,
                "provenance_status": {
                    "source": self.source, "status": self.status,
                    "generation_model": self.generation_model,
                    "generation_status": self.generation_status,
                },
            })
        return payload


class ReviewCriterion(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    kind: Literal["inclusion", "exclusion"]
    description: str = Field(min_length=3, max_length=600)
    required: bool = True
    expected_evidence: str = Field(default="", max_length=600)
    source: Literal["research_question", "user"] = "research_question"

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        value = "_".join(value.strip().lower().split())
        if not value.replace("_", "").replace("-", "").isalnum():
            raise ValueError("criterion id must contain only letters, numbers, underscores, or hyphens")
        return value


class ReviewProtocol(StrictModel):
    schema_version: str = SCHEMA_VERSION
    protocol_id: str = ""
    research_question: str = Field(min_length=3)
    research_context: str = Field(default="", max_length=4000)
    objective: str = Field(min_length=3, max_length=1200)
    scope_interpretation: str = Field(min_length=3, max_length=1600)
    criteria: list[ReviewCriterion] = Field(min_length=1)
    expected_relationships: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    semantic_boundaries: list[str] = Field(default_factory=list, max_length=6)
    prompt_version: str = PROMPT_VERSION
    model: str = ""

    @model_validator(mode="after")
    def unique_criterion_ids(self):
        ids = [criterion.id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion ids must be unique")
        if not any(c.kind == "inclusion" and c.required for c in self.criteria):
            raise ValueError("protocol requires at least one required inclusion criterion")
        return self

    def with_identity(self) -> "ReviewProtocol":
        payload = self.model_dump(exclude={"protocol_id"}, mode="json")
        digest = sha256(str(sorted(payload.items())).encode("utf-8")).hexdigest()[:16]
        return self.model_copy(update={"protocol_id": digest})


class EvidenceSpan(StrictModel):
    source: Literal["title", "abstract"]
    evidence_id: str = Field(min_length=1, max_length=40)


class CriterionEvidence(StrictModel):
    criterion_id: str
    verdict: Literal["MET", "NOT_MET", "UNCLEAR"]
    rationale: str = Field(min_length=1, max_length=500)
    evidence: list[EvidenceSpan] = Field(default_factory=list, max_length=2)


class PaperEvidence(StrictModel):
    schema_version: str = SCHEMA_VERSION
    summary: str = Field(min_length=1, max_length=600)
    criteria: list[CriterionEvidence]
    contradictions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class PaperAssessment(PaperEvidence):
    decision: Literal["KEEP", "MAYBE", "REJECT"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=500)
    uncertainty: list[str] = Field(default_factory=list)


class ValidationReport(StrictModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    exact_quote_count: int = 0
    decisive_evidence_count: int = 0
    rq_group_coverage: dict[str, bool] = Field(default_factory=dict)


def safe_maybe(reason: str) -> PaperAssessment:
    concise_reason = (str(reason).strip() or "Assessment could not be validated.")[:500]
    return PaperAssessment(
        summary="Assessment could not be validated.",
        criteria=[],
        contradictions=[],
        missing_information=[concise_reason],
        decision="MAYBE",
        confidence=0.0,
        reason=concise_reason,
        uncertainty=[concise_reason],
    )
