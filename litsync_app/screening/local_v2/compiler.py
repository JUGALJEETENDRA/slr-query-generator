from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from .contracts import (
    CriterionRole,
    ProtocolCriterion,
    ScreeningProtocolV2,
    StrictModel,
)


COMPILER_VERSION = "local-v2-compiler-v1"

CompilationIssueCode = Literal[
    "INVALID_DRAFT",
    "INVALID_CRITERION_ID",
    "DUPLICATE_CRITERION_ID",
    "NO_REQUIRED_INCLUSION",
]
CompilationWarningCode = Literal[
    "REQUIRED_RESOLUTION_FORCED",
]

_NON_WORD_RE = re.compile(r"[^\w-]+", flags=re.UNICODE)
_UNDERSCORE_RE = re.compile(r"_+")


class CriterionDraft(StrictModel):
    label: str = Field(min_length=1, max_length=160)
    role: CriterionRole
    description: str = Field(min_length=3, max_length=800)
    expected_evidence: str | None = Field(default=None, max_length=800)
    criterion_id: str | None = Field(default=None, max_length=80)
    resolution_required: bool | None = None

    @field_validator("label", "description")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped

    @field_validator("expected_evidence", "criterion_id")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ProtocolDraft(StrictModel):
    research_question: str = Field(min_length=3, max_length=4000)
    research_context: str = Field(default="", max_length=4000)
    criteria: list[CriterionDraft] = Field(min_length=1)
    model: str = Field(default="", max_length=200)

    @field_validator("research_question")
    @classmethod
    def strip_research_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("research question must not be empty")
        return stripped

    @field_validator("research_context", "model")
    @classmethod
    def strip_optional_fields(cls, value: str) -> str:
        return value.strip()


class CompilationIssue(StrictModel):
    code: CompilationIssueCode
    message: str = Field(min_length=1, max_length=1200)
    criterion_index: int | None = Field(default=None, ge=0)
    criterion_id: str | None = Field(default=None, max_length=80)


class CompilationWarning(StrictModel):
    code: CompilationWarningCode
    message: str = Field(min_length=1, max_length=1200)
    criterion_index: int = Field(ge=0)
    criterion_id: str = Field(min_length=1, max_length=80)


class ProtocolCompilationResult(StrictModel):
    compiler_version: Literal[COMPILER_VERSION] = COMPILER_VERSION
    success: bool
    protocol: ScreeningProtocolV2 | None = None
    issues: list[CompilationIssue] = Field(default_factory=list)
    warnings: list[CompilationWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_consistency(self) -> "ProtocolCompilationResult":
        if self.success and (self.protocol is None or self.issues):
            raise ValueError("successful compilation requires a protocol and no issues")
        if not self.success and (self.protocol is not None or not self.issues):
            raise ValueError("failed compilation requires issues and no protocol")
        return self


def _slugify_label(label: str, index: int) -> str:
    normalized = unicodedata.normalize("NFKC", label).strip().casefold()
    normalized = _NON_WORD_RE.sub("_", normalized)
    normalized = _UNDERSCORE_RE.sub("_", normalized).strip("_-")
    return (normalized or f"criterion_{index + 1}")[:80]


def _validation_issues(exc: ValidationError) -> list[CompilationIssue]:
    issues: list[CompilationIssue] = []
    for error in exc.errors(include_url=False):
        location = error.get("loc") or ()
        criterion_index = None
        if len(location) >= 2 and location[0] == "criteria" and isinstance(location[1], int):
            criterion_index = location[1]
        rendered_location = ".".join(str(part) for part in location) or "draft"
        issues.append(
            CompilationIssue(
                code="INVALID_DRAFT",
                message=f"{rendered_location}: {error.get('msg', 'invalid value')}",
                criterion_index=criterion_index,
            )
        )
    return issues


def compile_protocol_draft(
    draft: ProtocolDraft | Mapping[str, Any],
) -> ProtocolCompilationResult:
    """Compile a structured draft into a canonical, identity-bearing protocol.

    This function performs no semantic inference and uses no domain-specific rules. It
    only validates, canonicalizes, and enforces the deterministic safety invariants
    required by ``ScreeningProtocolV2``.
    """

    try:
        candidate = draft if isinstance(draft, ProtocolDraft) else ProtocolDraft.model_validate(draft)
    except (ValidationError, TypeError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            issues = _validation_issues(exc)
        else:
            issues = [
                CompilationIssue(
                    code="INVALID_DRAFT",
                    message=f"draft: {str(exc) or exc.__class__.__name__}",
                )
            ]
        return ProtocolCompilationResult(success=False, issues=issues)

    criteria: list[ProtocolCriterion] = []
    issues: list[CompilationIssue] = []
    warnings: list[CompilationWarning] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(candidate.criteria):
        criterion_id = item.criterion_id or _slugify_label(item.label, index)
        resolution_required = item.resolution_required
        if item.role == "REQUIRED_INCLUSION":
            if resolution_required is False:
                warnings.append(
                    CompilationWarning(
                        code="REQUIRED_RESOLUTION_FORCED",
                        message=(
                            "Required inclusion criteria must be resolved; "
                            "resolution_required was forced to true."
                        ),
                        criterion_index=index,
                        criterion_id=criterion_id,
                    )
                )
            resolution_required = True
        elif resolution_required is None:
            resolution_required = False

        try:
            criterion = ProtocolCriterion(
                id=criterion_id,
                role=item.role,
                description=item.description,
                expected_evidence=item.expected_evidence,
                resolution_required=bool(resolution_required),
            )
        except ValidationError as exc:
            message = "; ".join(
                error.get("msg", "invalid criterion id")
                for error in exc.errors(include_url=False)
            )
            issues.append(
                CompilationIssue(
                    code="INVALID_CRITERION_ID",
                    message=message,
                    criterion_index=index,
                    criterion_id=criterion_id,
                )
            )
            continue

        if criterion.id in seen_ids:
            issues.append(
                CompilationIssue(
                    code="DUPLICATE_CRITERION_ID",
                    message=f"criterion id {criterion.id!r} is duplicated",
                    criterion_index=index,
                    criterion_id=criterion.id,
                )
            )
            continue

        seen_ids.add(criterion.id)
        criteria.append(criterion)

    if not any(item.role == "REQUIRED_INCLUSION" for item in criteria):
        issues.append(
            CompilationIssue(
                code="NO_REQUIRED_INCLUSION",
                message="at least one valid REQUIRED_INCLUSION criterion is required",
            )
        )

    if issues:
        return ProtocolCompilationResult(
            success=False,
            issues=issues,
            warnings=warnings,
        )

    protocol = ScreeningProtocolV2(
        research_question=candidate.research_question,
        research_context=candidate.research_context,
        criteria=criteria,
        model=candidate.model,
    ).with_identity()
    return ProtocolCompilationResult(
        success=True,
        protocol=protocol,
        warnings=warnings,
    )
