from __future__ import annotations

import json
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "local-v2-contract-v1"
POLICY_VERSION = "local-v2-policy-v1"

CriterionRole = Literal["REQUIRED_INCLUSION", "EXCLUSION_TRIGGER"]
CriterionRelation = Literal[
    "DIRECT_SUPPORT",
    "DIRECT_CONTRADICTION",
    "MISSING_OR_UNCLEAR",
    "NOT_APPLICABLE",
]
FinalDecision = Literal["KEEP", "MAYBE", "REJECT"]
EvidenceSource = Literal["title", "abstract"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _normalize_criterion_id(value: str) -> str:
    normalized = "_".join(str(value).strip().lower().split())
    if not normalized:
        raise ValueError("criterion id must not be empty")
    if not normalized.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            "criterion id must contain only letters, numbers, underscores, or hyphens"
        )
    return normalized


class EvidenceCitation(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=40)
    source: EvidenceSource
    quote: str = Field(min_length=1, max_length=1200)

    @field_validator("evidence_id", "quote")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped


class ProtocolCriterion(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    role: CriterionRole
    description: str = Field(min_length=3, max_length=800)
    expected_evidence: str | None = Field(default=None, max_length=800)
    resolution_required: bool

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        return _normalize_criterion_id(value)

    @field_validator("description")
    @classmethod
    def strip_description(cls, value: str) -> str:
        return value.strip()

    @field_validator("expected_evidence")
    @classmethod
    def strip_expected_evidence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip()


class ScreeningProtocolV2(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    protocol_id: str = ""
    research_question: str = Field(min_length=3)
    research_context: str = Field(default="", max_length=4000)
    criteria: list[ProtocolCriterion] = Field(min_length=1)
    model: str = ""

    @field_validator("research_question")
    @classmethod
    def strip_research_question(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_protocol_structure(self) -> "ScreeningProtocolV2":
        criterion_ids = [criterion.id for criterion in self.criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("criterion ids must be unique")

        required_inclusions = [
            criterion
            for criterion in self.criteria
            if criterion.role == "REQUIRED_INCLUSION"
        ]
        if not required_inclusions:
            raise ValueError("protocol requires at least one REQUIRED_INCLUSION criterion")

        if any(not criterion.resolution_required for criterion in required_inclusions):
            raise ValueError(
                "every REQUIRED_INCLUSION criterion must have resolution_required=True"
            )
        return self

    def with_identity(self) -> "ScreeningProtocolV2":
        payload = self.model_dump(exclude={"protocol_id"}, mode="json")
        canonical = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return self.model_copy(update={"protocol_id": digest})


class CriterionAssessment(StrictModel):
    criterion_id: str = Field(min_length=1, max_length=80)
    relation: CriterionRelation
    rationale: str = Field(min_length=1, max_length=800)
    evidence: list[EvidenceCitation] = Field(default_factory=list, max_length=2)

    @field_validator("criterion_id")
    @classmethod
    def normalize_criterion_id(cls, value: str) -> str:
        return _normalize_criterion_id(value)

    @field_validator("rationale")
    @classmethod
    def strip_rationale(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("rationale must not be empty")
        return stripped

    @model_validator(mode="after")
    def validate_relation_evidence(self) -> "CriterionAssessment":
        evidence_ids = [citation.evidence_id for citation in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence ids must be unique within an assessment")

        if self.relation in {"DIRECT_SUPPORT", "DIRECT_CONTRADICTION"} and not self.evidence:
            raise ValueError(
                f"{self.relation} requires at least one evidence citation"
            )

        if self.relation == "NOT_APPLICABLE" and self.evidence:
            raise ValueError("NOT_APPLICABLE cannot contain evidence")

        return self


class PolicyResult(StrictModel):
    policy_version: Literal[POLICY_VERSION] = POLICY_VERSION
    decision: FinalDecision
    reason: str = Field(min_length=1, max_length=1200)
    decisive_criterion_ids: list[str] = Field(default_factory=list)
    unresolved_criterion_ids: list[str] = Field(default_factory=list)
    policy_errors: list[str] = Field(default_factory=list)
    safe_fallback: bool = False
