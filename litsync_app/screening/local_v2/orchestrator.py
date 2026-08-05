from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol

from pydantic import Field, field_validator, model_validator

from .assessor import (
    ModelAssessmentEnvelope,
    ModelAssessmentParseResult,
    build_assessment_prompt,
    model_assessment_envelope_schema,
    parse_model_assessment_response,
)
from .contracts import PolicyResult, ScreeningProtocolV2, StrictModel
from .evidence import EvidenceBatchValidation, validate_assessments_evidence
from .policy import derive_policy_decision


ORCHESTRATOR_VERSION = "local-v2-orchestrator-v2"
DEFAULT_PRIMARY_MODEL = "qwen3.5:4b"
DEFAULT_REVIEW_MODEL = "qwen3:8b"
DEFAULT_VALIDATOR_MODEL = "qwen3.5:4b"

AssessmentStage = Literal["primary", "reviewer", "validator"]
OrchestrationRoute = Literal[
    "PRIMARY_KEEP_FAST_PATH",
    "REVIEW_KEEP",
    "REVIEW_MAYBE",
    "REVIEW_FAILURE_SAFE_MAYBE",
    "PRIMARY_REJECT_REVIEW_DISAGREEMENT",
    "REJECTION_CONFIRMED",
    "REJECTION_UNCONFIRMED",
    "VALIDATOR_FAILURE_SAFE_MAYBE",
]

_PRIMARY_ROLE = "You are the primary local semantic screener."
_ROLE_PROMPTS = {
    "primary": _PRIMARY_ROLE,
    "reviewer": (
        "You are the deep local exclusion reviewer. Independently assess every "
        "criterion from the supplied protocol and paper. You are blind to any prior "
        "model output. Treat exclusion as high risk and require explicit evidence."
    ),
    "validator": (
        "You are the independent local semantic safety validator. Reassess every "
        "criterion from the supplied protocol and paper without access to prior model "
        "outputs. A rejection is valid only when explicit evidence establishes it."
    ),
}


class StructuredAssessmentEngine(Protocol):
    def generate(
        self,
        model: str,
        prompt: str,
        schema: type[ModelAssessmentEnvelope],
        *,
        timeout_seconds: float | None = None,
    ) -> Any: ...


class LocalModelPlan(StrictModel):
    primary_model: str = Field(default=DEFAULT_PRIMARY_MODEL, min_length=1, max_length=200)
    review_model: str = Field(default=DEFAULT_REVIEW_MODEL, min_length=1, max_length=200)
    validator_model: str = Field(
        default=DEFAULT_VALIDATOR_MODEL,
        min_length=1,
        max_length=200,
    )
    primary_timeout_seconds: float = Field(default=120.0, gt=0, le=900)
    review_timeout_seconds: float = Field(default=180.0, gt=0, le=900)
    validator_timeout_seconds: float = Field(default=120.0, gt=0, le=900)

    @field_validator("primary_model", "review_model", "validator_model")
    @classmethod
    def strip_model_names(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("model name must not be empty")
        return stripped


class AssessmentStageOutcome(StrictModel):
    stage: AssessmentStage
    model: str = Field(min_length=1, max_length=200)
    attempted: bool = True
    usable: bool = False
    elapsed_seconds: float = Field(default=0.0, ge=0)
    generation_error: str | None = Field(default=None, max_length=1200)
    parse_result: ModelAssessmentParseResult | None = None
    evidence_result: EvidenceBatchValidation | None = None
    policy_result: PolicyResult

    @model_validator(mode="after")
    def validate_consistency(self) -> "AssessmentStageOutcome":
        if self.usable:
            if self.generation_error is not None:
                raise ValueError("usable stage cannot contain a generation error")
            if self.parse_result is None or not self.parse_result.success:
                raise ValueError("usable stage requires a successful parse result")
            if self.evidence_result is None:
                raise ValueError("usable stage requires evidence validation")
        return self


class LocalV2OrchestrationResult(StrictModel):
    orchestrator_version: Literal[ORCHESTRATOR_VERSION] = ORCHESTRATOR_VERSION
    protocol_id: str = Field(min_length=1, max_length=80)
    paper_id: str = Field(min_length=1, max_length=200)
    model_plan: LocalModelPlan
    route: OrchestrationRoute
    primary: AssessmentStageOutcome
    reviewer: AssessmentStageOutcome | None = None
    validator: AssessmentStageOutcome | None = None
    final_policy: PolicyResult
    review_used: bool = False
    validator_used: bool = False
    safe_fallback: bool = False

    @model_validator(mode="after")
    def validate_consistency(self) -> "LocalV2OrchestrationResult":
        if self.review_used != (self.reviewer is not None):
            raise ValueError("review_used must match reviewer presence")
        if self.validator_used != (self.validator is not None):
            raise ValueError("validator_used must match validator presence")
        if self.validator is not None and self.reviewer is None:
            raise ValueError("validator cannot run without reviewer")
        if self.final_policy.decision == "REJECT" and self.route != "REJECTION_CONFIRMED":
            raise ValueError("REJECT is only allowed after confirmed rejection")
        return self


def _safe_maybe(reason: str, *errors: str) -> PolicyResult:
    return PolicyResult(
        decision="MAYBE",
        reason=reason[:1200],
        policy_errors=[str(item)[:1200] for item in errors if str(item).strip()],
        safe_fallback=True,
    )


def _role_prompt(
    stage: AssessmentStage,
    protocol: ScreeningProtocolV2,
    *,
    paper_id: str,
    title: str | None,
    abstract: str | None,
) -> str:
    prompt = build_assessment_prompt(
        protocol,
        paper_id=paper_id,
        title=title,
        abstract=abstract,
    )
    return prompt.replace(_PRIMARY_ROLE, _ROLE_PROMPTS[stage], 1)


def _unwrap_generation_value(result: Any) -> str | Mapping[str, Any]:
    value = getattr(result, "value", result)
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        return value
    raise TypeError("structured engine returned neither a mapping nor a string")


def _elapsed_seconds(result: Any) -> float:
    try:
        return max(0.0, float(getattr(result, "elapsed_seconds", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _run_stage(
    engine: StructuredAssessmentEngine,
    *,
    stage: AssessmentStage,
    model: str,
    timeout_seconds: float,
    protocol: ScreeningProtocolV2,
    paper_id: str,
    title: str | None,
    abstract: str | None,
) -> AssessmentStageOutcome:
    prompt = _role_prompt(
        stage,
        protocol,
        paper_id=paper_id,
        title=title,
        abstract=abstract,
    )
    try:
        schema = model_assessment_envelope_schema(protocol)
        generation = engine.generate(
            model,
            prompt,
            schema,
            timeout_seconds=timeout_seconds,
        )
        raw = _unwrap_generation_value(generation)
    except Exception as exc:
        message = f"{stage} generation failed: {str(exc) or exc.__class__.__name__}"
        return AssessmentStageOutcome(
            stage=stage,
            model=model,
            generation_error=message[:1200],
            policy_result=_safe_maybe(
                "The local model stage failed, so screening returned a safe MAYBE.",
                message,
            ),
        )

    parsed = parse_model_assessment_response(
        raw,
        protocol=protocol,
        paper_id=paper_id,
    )
    if not parsed.success:
        errors = [f"{issue.code}: {issue.message}" for issue in parsed.issues]
        return AssessmentStageOutcome(
            stage=stage,
            model=model,
            elapsed_seconds=_elapsed_seconds(generation),
            parse_result=parsed,
            policy_result=_safe_maybe(
                "The local model output was structurally unsafe, so screening returned "
                "a safe MAYBE.",
                *errors,
            ),
        )

    evidence = validate_assessments_evidence(
        parsed.assessments,
        title=title,
        abstract=abstract,
    )
    policy = derive_policy_decision(protocol, evidence.assessments)
    return AssessmentStageOutcome(
        stage=stage,
        model=model,
        usable=True,
        elapsed_seconds=_elapsed_seconds(generation),
        parse_result=parsed,
        evidence_result=evidence,
        policy_result=policy,
    )


def _primary_needs_review(primary: AssessmentStageOutcome) -> bool:
    if not primary.usable:
        return True
    if primary.policy_result.decision != "KEEP":
        return True
    evidence = primary.evidence_result
    return bool(evidence and evidence.safe_downgrade_count)


def _confirmed_ids(
    reviewer: AssessmentStageOutcome,
    validator: AssessmentStageOutcome,
) -> list[str]:
    if (
        not reviewer.usable
        or reviewer.policy_result.decision != "REJECT"
        or not validator.usable
        or validator.policy_result.decision != "REJECT"
    ):
        return []
    validator_ids = set(validator.policy_result.decisive_criterion_ids)
    return [
        criterion_id
        for criterion_id in reviewer.policy_result.decisive_criterion_ids
        if criterion_id in validator_ids
    ]


def _result(
    *,
    protocol: ScreeningProtocolV2,
    paper_id: str,
    plan: LocalModelPlan,
    route: OrchestrationRoute,
    primary: AssessmentStageOutcome,
    reviewer: AssessmentStageOutcome | None,
    validator: AssessmentStageOutcome | None,
    final_policy: PolicyResult,
    safe_fallback: bool = False,
) -> LocalV2OrchestrationResult:
    return LocalV2OrchestrationResult(
        protocol_id=protocol.protocol_id,
        paper_id=paper_id,
        model_plan=plan,
        route=route,
        primary=primary,
        reviewer=reviewer,
        validator=validator,
        final_policy=final_policy,
        review_used=reviewer is not None,
        validator_used=validator is not None,
        safe_fallback=safe_fallback or final_policy.safe_fallback,
    )


def orchestrate_local_v2_assessment(
    engine: StructuredAssessmentEngine,
    protocol: ScreeningProtocolV2,
    *,
    paper_id: str,
    title: str | None,
    abstract: str | None,
    model_plan: LocalModelPlan | None = None,
) -> LocalV2OrchestrationResult:
    """Run recall-first local screening with blind review of risky outcomes.

    The 4B primary model performs the ordinary semantic workload. A blind 8B reviewer
    runs only when the primary stage is unresolved, unsafe, or proposes exclusion. Any
    reviewer rejection must then be independently confirmed by the 4B safety validator
    on at least one identical decisive criterion. Technical or structural failures can
    never produce REJECT.
    """

    normalized_paper_id = str(paper_id or "").strip()
    if not normalized_paper_id:
        raise ValueError("paper_id is required")
    if not protocol.protocol_id:
        raise ValueError("protocol must have a deterministic protocol_id")

    plan = model_plan or LocalModelPlan()
    primary = _run_stage(
        engine,
        stage="primary",
        model=plan.primary_model,
        timeout_seconds=plan.primary_timeout_seconds,
        protocol=protocol,
        paper_id=normalized_paper_id,
        title=title,
        abstract=abstract,
    )

    if not _primary_needs_review(primary):
        return _result(
            protocol=protocol,
            paper_id=normalized_paper_id,
            plan=plan,
            route="PRIMARY_KEEP_FAST_PATH",
            primary=primary,
            reviewer=None,
            validator=None,
            final_policy=primary.policy_result,
        )

    reviewer = _run_stage(
        engine,
        stage="reviewer",
        model=plan.review_model,
        timeout_seconds=plan.review_timeout_seconds,
        protocol=protocol,
        paper_id=normalized_paper_id,
        title=title,
        abstract=abstract,
    )

    if not reviewer.usable:
        return _result(
            protocol=protocol,
            paper_id=normalized_paper_id,
            plan=plan,
            route="REVIEW_FAILURE_SAFE_MAYBE",
            primary=primary,
            reviewer=reviewer,
            validator=None,
            final_policy=_safe_maybe(
                "The risky primary result could not be independently reviewed, so the "
                "final decision is MAYBE.",
                *(reviewer.policy_result.policy_errors or ["reviewer unusable"]),
            ),
            safe_fallback=True,
        )

    if (
        reviewer.evidence_result is not None
        and reviewer.evidence_result.safe_downgrade_count
    ):
        return _result(
            protocol=protocol,
            paper_id=normalized_paper_id,
            plan=plan,
            route="REVIEW_MAYBE",
            primary=primary,
            reviewer=reviewer,
            validator=None,
            final_policy=_safe_maybe(
                "The blind reviewer attempted a decisive relation without valid exact "
                "evidence, so the final decision is MAYBE.",
                "reviewer decisive evidence was downgraded",
            ),
            safe_fallback=True,
        )

    if reviewer.policy_result.decision == "KEEP":
        if primary.usable and primary.policy_result.decision == "REJECT":
            return _result(
                protocol=protocol,
                paper_id=normalized_paper_id,
                plan=plan,
                route="PRIMARY_REJECT_REVIEW_DISAGREEMENT",
                primary=primary,
                reviewer=reviewer,
                validator=None,
                final_policy=_safe_maybe(
                    "The primary model proposed exclusion but the blind 8B reviewer "
                    "did not confirm it, so the final decision is MAYBE.",
                    "primary and reviewer disagreed on rejection",
                ),
                safe_fallback=True,
            )
        return _result(
            protocol=protocol,
            paper_id=normalized_paper_id,
            plan=plan,
            route="REVIEW_KEEP",
            primary=primary,
            reviewer=reviewer,
            validator=None,
            final_policy=reviewer.policy_result,
        )

    if reviewer.policy_result.decision == "MAYBE":
        return _result(
            protocol=protocol,
            paper_id=normalized_paper_id,
            plan=plan,
            route="REVIEW_MAYBE",
            primary=primary,
            reviewer=reviewer,
            validator=None,
            final_policy=reviewer.policy_result,
        )

    validator = _run_stage(
        engine,
        stage="validator",
        model=plan.validator_model,
        timeout_seconds=plan.validator_timeout_seconds,
        protocol=protocol,
        paper_id=normalized_paper_id,
        title=title,
        abstract=abstract,
    )

    if not validator.usable:
        return _result(
            protocol=protocol,
            paper_id=normalized_paper_id,
            plan=plan,
            route="VALIDATOR_FAILURE_SAFE_MAYBE",
            primary=primary,
            reviewer=reviewer,
            validator=validator,
            final_policy=_safe_maybe(
                "The 8B reviewer proposed exclusion, but the safety validator failed, "
                "so the final decision is MAYBE.",
                *(validator.policy_result.policy_errors or ["validator unusable"]),
            ),
            safe_fallback=True,
        )

    confirmed = _confirmed_ids(reviewer, validator)
    if confirmed:
        return _result(
            protocol=protocol,
            paper_id=normalized_paper_id,
            plan=plan,
            route="REJECTION_CONFIRMED",
            primary=primary,
            reviewer=reviewer,
            validator=validator,
            final_policy=PolicyResult(
                decision="REJECT",
                reason=(
                    "Blind reviewer and independent safety validator confirmed explicit "
                    "exclusion evidence for: " + ", ".join(confirmed) + "."
                ),
                decisive_criterion_ids=confirmed,
            ),
        )

    return _result(
        protocol=protocol,
        paper_id=normalized_paper_id,
        plan=plan,
        route="REJECTION_UNCONFIRMED",
        primary=primary,
        reviewer=reviewer,
        validator=validator,
        final_policy=_safe_maybe(
            "The reviewer proposed exclusion, but the independent validator did not "
            "confirm the same decisive criterion, so the final decision is MAYBE.",
            "rejection lacked criterion-level two-model confirmation",
        ),
        safe_fallback=True,
    )
