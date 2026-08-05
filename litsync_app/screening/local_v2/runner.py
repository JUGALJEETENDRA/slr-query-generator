from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .compiler import ProtocolCompilationResult, ProtocolDraft, compile_protocol_draft
from .contracts import PolicyResult, ScreeningProtocolV2, StrictModel
from .orchestrator import (
    LocalModelPlan,
    LocalV2OrchestrationResult,
    StructuredAssessmentEngine,
    orchestrate_local_v2_assessment,
)


RUNNER_VERSION = "local-v2-runner-v1"

PaperTextStatus = Literal[
    "TITLE_AND_ABSTRACT",
    "TITLE_ONLY",
    "ABSTRACT_ONLY",
    "NO_SCREENABLE_TEXT",
]
PaperRunStatus = Literal["COMPLETED", "NO_SCREENABLE_TEXT"]
EndToEndRunStatus = Literal[
    "COMPLETED",
    "NO_SCREENABLE_TEXT",
    "PROTOCOL_COMPILATION_FAILED",
]


class LocalV2Paper(StrictModel):
    paper_id: str = Field(min_length=1, max_length=200)
    title: str | None = Field(default=None, max_length=4000)
    abstract: str | None = Field(default=None, max_length=100000)

    @field_validator("paper_id")
    @classmethod
    def strip_paper_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("paper_id must not be empty")
        return stripped

    @field_validator("title", "abstract")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class LocalV2PaperRunResult(StrictModel):
    runner_version: Literal[RUNNER_VERSION] = RUNNER_VERSION
    status: PaperRunStatus
    protocol_id: str = Field(min_length=1, max_length=80)
    paper: LocalV2Paper
    text_status: PaperTextStatus
    orchestration: LocalV2OrchestrationResult | None = None
    final_policy: PolicyResult
    model_call_count: int = Field(ge=0, le=3)
    elapsed_seconds: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    safe_fallback: bool = False

    @model_validator(mode="after")
    def validate_consistency(self) -> "LocalV2PaperRunResult":
        expected_text_status = _paper_text_status(self.paper)
        if self.text_status != expected_text_status:
            raise ValueError("text_status must match the normalized paper text")

        if self.status == "NO_SCREENABLE_TEXT":
            if self.text_status != "NO_SCREENABLE_TEXT":
                raise ValueError("NO_SCREENABLE_TEXT status requires no title or abstract")
            if self.orchestration is not None:
                raise ValueError("missing-text result cannot contain orchestration")
            if self.model_call_count != 0 or self.elapsed_seconds != 0:
                raise ValueError("missing-text result cannot report model work")
            if self.final_policy.decision != "MAYBE" or not self.safe_fallback:
                raise ValueError("missing-text result must be a safe MAYBE")
            return self

        if self.text_status == "NO_SCREENABLE_TEXT":
            raise ValueError("completed run requires screenable title or abstract text")
        if self.orchestration is None:
            raise ValueError("completed run requires orchestration")
        if self.orchestration.protocol_id != self.protocol_id:
            raise ValueError("orchestration protocol_id must match the runner protocol")
        if self.orchestration.paper_id != self.paper.paper_id:
            raise ValueError("orchestration paper_id must match the runner paper")
        if self.final_policy != self.orchestration.final_policy:
            raise ValueError("final_policy must match the orchestration result")

        expected_calls = _orchestration_model_call_count(self.orchestration)
        if self.model_call_count != expected_calls:
            raise ValueError("model_call_count must match attempted orchestration stages")
        expected_elapsed = _orchestration_elapsed_seconds(self.orchestration)
        if abs(self.elapsed_seconds - expected_elapsed) > 0.0001:
            raise ValueError("elapsed_seconds must equal the summed stage durations")
        if (self.final_policy.safe_fallback or self.orchestration.safe_fallback) and not self.safe_fallback:
            raise ValueError("safe fallback state cannot be hidden")
        return self


class LocalV2EndToEndResult(StrictModel):
    runner_version: Literal[RUNNER_VERSION] = RUNNER_VERSION
    status: EndToEndRunStatus
    compilation: ProtocolCompilationResult
    paper: LocalV2Paper
    paper_result: LocalV2PaperRunResult | None = None
    final_policy: PolicyResult
    safe_fallback: bool = False

    @model_validator(mode="after")
    def validate_consistency(self) -> "LocalV2EndToEndResult":
        if not self.compilation.success:
            if self.status != "PROTOCOL_COMPILATION_FAILED":
                raise ValueError("failed compilation requires compilation-failed status")
            if self.paper_result is not None:
                raise ValueError("failed compilation cannot contain a paper result")
            if self.final_policy.decision != "MAYBE" or not self.safe_fallback:
                raise ValueError("failed compilation must return a safe MAYBE")
            return self

        if self.compilation.protocol is None:
            raise ValueError("successful compilation requires a protocol")
        if self.paper_result is None:
            raise ValueError("successful compilation requires a paper result")
        if self.paper_result.paper != self.paper:
            raise ValueError("paper_result must correspond to the requested paper")
        if self.paper_result.protocol_id != self.compilation.protocol.protocol_id:
            raise ValueError("paper_result protocol_id must match compiled protocol")
        if self.status != self.paper_result.status:
            raise ValueError("end-to-end status must match the paper result")
        if self.final_policy != self.paper_result.final_policy:
            raise ValueError("end-to-end final_policy must match the paper result")
        if self.paper_result.safe_fallback and not self.safe_fallback:
            raise ValueError("safe fallback state cannot be hidden")
        return self


def _safe_maybe(reason: str, *errors: str) -> PolicyResult:
    return PolicyResult(
        decision="MAYBE",
        reason=reason[:1200],
        policy_errors=[str(item)[:1200] for item in errors if str(item).strip()],
        safe_fallback=True,
    )


def _paper_text_status(paper: LocalV2Paper) -> PaperTextStatus:
    if paper.title and paper.abstract:
        return "TITLE_AND_ABSTRACT"
    if paper.title:
        return "TITLE_ONLY"
    if paper.abstract:
        return "ABSTRACT_ONLY"
    return "NO_SCREENABLE_TEXT"


def _text_warnings(text_status: PaperTextStatus) -> list[str]:
    if text_status == "TITLE_ONLY":
        return ["Abstract is unavailable; screening used title evidence only."]
    if text_status == "ABSTRACT_ONLY":
        return ["Title is unavailable; screening used abstract evidence only."]
    if text_status == "NO_SCREENABLE_TEXT":
        return ["Neither title nor abstract is available for semantic screening."]
    return []


def _orchestration_model_call_count(result: LocalV2OrchestrationResult) -> int:
    return 1 + int(result.review_used) + int(result.validator_used)


def _orchestration_elapsed_seconds(result: LocalV2OrchestrationResult) -> float:
    stages = [result.primary, result.reviewer, result.validator]
    return round(sum(stage.elapsed_seconds for stage in stages if stage is not None), 4)


def _validate_protocol_identity(protocol: ScreeningProtocolV2) -> None:
    if not protocol.protocol_id:
        raise ValueError("protocol must have a deterministic protocol_id")
    canonical = protocol.model_copy(update={"protocol_id": ""}).with_identity()
    if canonical.protocol_id != protocol.protocol_id:
        raise ValueError("protocol_id does not match the current protocol contents")


def _normalize_paper(paper: LocalV2Paper | Mapping[str, Any]) -> LocalV2Paper:
    if isinstance(paper, LocalV2Paper):
        return paper
    return LocalV2Paper.model_validate(paper)


def run_compiled_local_v2_paper(
    engine: StructuredAssessmentEngine,
    protocol: ScreeningProtocolV2,
    *,
    paper: LocalV2Paper | Mapping[str, Any],
    model_plan: LocalModelPlan | None = None,
) -> LocalV2PaperRunResult:
    """Run one paper through the complete Local AI v2 semantic stack.

    The caller can compile a protocol once and reuse this function for every paper in a
    dataset. Python only validates identity, handles missing text, and records execution
    diagnostics. Semantic assessment remains delegated to the local models through the
    orchestrator.
    """

    _validate_protocol_identity(protocol)
    normalized_paper = _normalize_paper(paper)
    text_status = _paper_text_status(normalized_paper)
    warnings = _text_warnings(text_status)

    if text_status == "NO_SCREENABLE_TEXT":
        final_policy = _safe_maybe(
            "Neither title nor abstract is available, so the paper cannot be safely "
            "resolved and the final decision is MAYBE.",
            "no screenable title or abstract text",
        )
        return LocalV2PaperRunResult(
            status="NO_SCREENABLE_TEXT",
            protocol_id=protocol.protocol_id,
            paper=normalized_paper,
            text_status=text_status,
            final_policy=final_policy,
            model_call_count=0,
            elapsed_seconds=0.0,
            warnings=warnings,
            safe_fallback=True,
        )

    orchestration = orchestrate_local_v2_assessment(
        engine,
        protocol,
        paper_id=normalized_paper.paper_id,
        title=normalized_paper.title,
        abstract=normalized_paper.abstract,
        model_plan=model_plan,
    )
    final_policy = orchestration.final_policy
    return LocalV2PaperRunResult(
        status="COMPLETED",
        protocol_id=protocol.protocol_id,
        paper=normalized_paper,
        text_status=text_status,
        orchestration=orchestration,
        final_policy=final_policy,
        model_call_count=_orchestration_model_call_count(orchestration),
        elapsed_seconds=_orchestration_elapsed_seconds(orchestration),
        warnings=warnings,
        safe_fallback=orchestration.safe_fallback or final_policy.safe_fallback,
    )


def compile_and_run_local_v2_paper(
    engine: StructuredAssessmentEngine,
    draft: ProtocolDraft | Mapping[str, Any],
    *,
    paper: LocalV2Paper | Mapping[str, Any],
    model_plan: LocalModelPlan | None = None,
) -> LocalV2EndToEndResult:
    """Compile one protocol draft and screen one paper as an isolated end-to-end run.

    This convenience path is intended for smoke tests and single-paper execution. Bulk
    callers should compile once with ``compile_protocol_draft`` and then reuse
    ``run_compiled_local_v2_paper`` so protocol compilation is not repeated per paper.
    """

    normalized_paper = _normalize_paper(paper)
    compilation = compile_protocol_draft(draft)
    if not compilation.success or compilation.protocol is None:
        errors = [f"{issue.code}: {issue.message}" for issue in compilation.issues]
        final_policy = _safe_maybe(
            "The screening protocol could not be compiled safely, so no model decision "
            "was attempted and the final decision is MAYBE.",
            *errors,
        )
        return LocalV2EndToEndResult(
            status="PROTOCOL_COMPILATION_FAILED",
            compilation=compilation,
            paper=normalized_paper,
            final_policy=final_policy,
            safe_fallback=True,
        )

    paper_result = run_compiled_local_v2_paper(
        engine,
        compilation.protocol,
        paper=normalized_paper,
        model_plan=model_plan,
    )
    return LocalV2EndToEndResult(
        status=paper_result.status,
        compilation=compilation,
        paper=normalized_paper,
        paper_result=paper_result,
        final_policy=paper_result.final_policy,
        safe_fallback=paper_result.safe_fallback,
    )
