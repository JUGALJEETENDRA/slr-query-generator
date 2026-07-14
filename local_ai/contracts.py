from __future__ import annotations

from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "2.0"
PROMPT_VERSION = "local-cross-model-three-layer-v3"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
