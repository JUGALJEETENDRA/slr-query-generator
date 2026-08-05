from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from litsync_app.screening.local_v2 import (
    ModelAssessmentEnvelope,
    PolicyResult,
    ProtocolCriterion,
    ScreeningProtocolV2,
)
from litsync_app.screening.local_v2.orchestrator import (
    DEFAULT_PRIMARY_MODEL,
    DEFAULT_REVIEW_MODEL,
    DEFAULT_VALIDATOR_MODEL,
    LocalModelPlan,
    LocalV2OrchestrationResult,
    orchestrate_local_v2_assessment,
)


TITLE = "Large language models automate abstract screening."
ABSTRACT = (
    "We evaluate a large language model for title and abstract screening in systematic "
    "reviews. The system prioritizes candidate studies for human review."
)


@dataclass
class FakeGeneration:
    value: Any
    elapsed_seconds: float = 0.25


class FakeEngine:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "schema": schema,
                "timeout_seconds": timeout_seconds,
            }
        )
        if not self.responses:
            raise AssertionError("unexpected model call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, FakeGeneration):
            return response
        return FakeGeneration(response)


def protocol() -> ScreeningProtocolV2:
    return ScreeningProtocolV2(
        research_question="Can LLMs automate systematic-review screening?",
        criteria=[
            ProtocolCriterion(
                id="uses_llm",
                role="REQUIRED_INCLUSION",
                description="The paper studies a large language model.",
                expected_evidence="Explicit mention of an LLM or large language model.",
                resolution_required=True,
            ),
            ProtocolCriterion(
                id="screening_task",
                role="REQUIRED_INCLUSION",
                description="The model is applied to study screening.",
                expected_evidence="Title or abstract screening is explicitly described.",
                resolution_required=True,
            ),
            ProtocolCriterion(
                id="non_review_task",
                role="EXCLUSION_TRIGGER",
                description="The work is explicitly unrelated to literature-review screening.",
                expected_evidence="An explicit incompatible task or setting.",
                resolution_required=False,
            ),
        ],
        model="qwen3.5:4b",
    ).with_identity()


def citation(*, quote: str, evidence_id: str = "abstract_001") -> list[dict[str, str]]:
    return [
        {
            "evidence_id": evidence_id,
            "source": "abstract" if evidence_id.startswith("abstract") else "title",
            "quote": quote,
        }
    ]


def assessment(
    criterion_id: str,
    relation: str,
    *,
    quote: str | None = None,
    evidence_id: str = "abstract_001",
    rationale: str | None = None,
) -> dict[str, Any]:
    evidence = citation(quote=quote, evidence_id=evidence_id) if quote is not None else []
    return {
        "criterion_id": criterion_id,
        "relation": relation,
        "rationale": rationale or f"Assessment for {criterion_id}.",
        "evidence": evidence,
    }


def envelope(
    proto: ScreeningProtocolV2,
    *,
    uses_llm: str = "DIRECT_SUPPORT",
    screening_task: str = "DIRECT_SUPPORT",
    non_review_task: str = "MISSING_OR_UNCLEAR",
    bad_quote_for: str | None = None,
    paper_id: str = "p-1",
) -> dict[str, Any]:
    support_quote = "We evaluate a large language model for title and abstract screening"
    screening_quote = "title and abstract screening in systematic reviews"
    exclusion_quote = "The system prioritizes candidate studies for human review"

    def quote_for(criterion_id: str, relation: str, valid_quote: str) -> str | None:
        if relation not in {"DIRECT_SUPPORT", "DIRECT_CONTRADICTION"}:
            return None
        if criterion_id == bad_quote_for:
            return "words that are not in the paper"
        return valid_quote

    return {
        "protocol_id": proto.protocol_id,
        "paper_id": paper_id,
        "assessments": [
            assessment(
                "uses_llm",
                uses_llm,
                quote=quote_for("uses_llm", uses_llm, support_quote),
            ),
            assessment(
                "screening_task",
                screening_task,
                quote=quote_for("screening_task", screening_task, screening_quote),
            ),
            assessment(
                "non_review_task",
                non_review_task,
                quote=quote_for("non_review_task", non_review_task, exclusion_quote),
                evidence_id="abstract_002",
            ),
        ],
    }


def run(engine: FakeEngine, *, model_plan: LocalModelPlan | None = None):
    return orchestrate_local_v2_assessment(
        engine,
        protocol(),
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
        model_plan=model_plan,
    )


def test_default_model_plan_freezes_selected_local_models():
    plan = LocalModelPlan()
    assert plan.primary_model == DEFAULT_PRIMARY_MODEL == "qwen3.5:4b"
    assert plan.review_model == DEFAULT_REVIEW_MODEL == "qwen3:8b"
    assert plan.validator_model == DEFAULT_VALIDATOR_MODEL == "qwen3.5:4b"


def test_primary_keep_uses_fast_path_and_only_one_model_call():
    proto = protocol()
    engine = FakeEngine([envelope(proto)])

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.final_policy.decision == "KEEP"
    assert result.route == "PRIMARY_KEEP_FAST_PATH"
    assert not result.review_used
    assert not result.validator_used
    assert [call["model"] for call in engine.calls] == ["qwen3.5:4b"]


def test_primary_maybe_is_rescued_by_blind_8b_keep():
    proto = protocol()
    engine = FakeEngine(
        [
            envelope(proto, screening_task="MISSING_OR_UNCLEAR"),
            envelope(proto),
        ]
    )

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.final_policy.decision == "KEEP"
    assert result.route == "REVIEW_KEEP"
    assert result.review_used
    assert not result.validator_used
    assert [call["model"] for call in engine.calls] == ["qwen3.5:4b", "qwen3:8b"]


def test_primary_generation_failure_can_be_rescued_by_8b_keep():
    proto = protocol()
    engine = FakeEngine([RuntimeError("ollama unavailable"), envelope(proto)])

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert not result.primary.usable
    assert result.primary.generation_error
    assert result.final_policy.decision == "KEEP"
    assert result.route == "REVIEW_KEEP"


def test_primary_parse_failure_can_be_rescued_by_8b_keep():
    proto = protocol()
    engine = FakeEngine(["not json", envelope(proto)])

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert not result.primary.usable
    assert result.primary.parse_result is not None
    assert result.primary.parse_result.safe_fallback
    assert result.final_policy.decision == "KEEP"


def test_primary_reject_and_reviewer_keep_becomes_safe_maybe():
    proto = protocol()
    engine = FakeEngine(
        [
            envelope(proto, non_review_task="DIRECT_SUPPORT"),
            envelope(proto),
        ]
    )

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.primary.policy_result.decision == "REJECT"
    assert result.reviewer.policy_result.decision == "KEEP"
    assert result.final_policy.decision == "MAYBE"
    assert result.route == "PRIMARY_REJECT_REVIEW_DISAGREEMENT"
    assert result.safe_fallback


def test_primary_reject_and_reviewer_maybe_stays_maybe():
    proto = protocol()
    engine = FakeEngine(
        [
            envelope(proto, non_review_task="DIRECT_SUPPORT"),
            envelope(proto, screening_task="MISSING_OR_UNCLEAR"),
        ]
    )

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.final_policy.decision == "MAYBE"
    assert result.route == "REVIEW_MAYBE"
    assert not result.validator_used


def test_reviewer_reject_requires_validator_and_same_decisive_criterion():
    proto = protocol()
    engine = FakeEngine(
        [
            envelope(proto, screening_task="MISSING_OR_UNCLEAR"),
            envelope(proto, non_review_task="DIRECT_SUPPORT"),
            envelope(proto, non_review_task="DIRECT_SUPPORT"),
        ]
    )

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.final_policy.decision == "REJECT"
    assert result.route == "REJECTION_CONFIRMED"
    assert result.final_policy.decisive_criterion_ids == ["non_review_task"]
    assert result.validator_used
    assert [call["model"] for call in engine.calls] == [
        "qwen3.5:4b",
        "qwen3:8b",
        "qwen3.5:4b",
    ]


def test_reviewer_and_validator_rejecting_different_criteria_cannot_reject():
    proto = protocol()
    engine = FakeEngine(
        [
            envelope(proto, screening_task="MISSING_OR_UNCLEAR"),
            envelope(proto, non_review_task="DIRECT_SUPPORT"),
            envelope(proto, uses_llm="DIRECT_CONTRADICTION"),
        ]
    )

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.final_policy.decision == "MAYBE"
    assert result.route == "REJECTION_UNCONFIRMED"
    assert result.safe_fallback


def test_reviewer_reject_validator_keep_cannot_reject():
    proto = protocol()
    engine = FakeEngine(
        [
            envelope(proto, screening_task="MISSING_OR_UNCLEAR"),
            envelope(proto, non_review_task="DIRECT_SUPPORT"),
            envelope(proto),
        ]
    )

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.final_policy.decision == "MAYBE"
    assert result.route == "REJECTION_UNCONFIRMED"


def test_validator_generation_failure_is_safe_maybe():
    proto = protocol()
    engine = FakeEngine(
        [
            envelope(proto, screening_task="MISSING_OR_UNCLEAR"),
            envelope(proto, non_review_task="DIRECT_SUPPORT"),
            TimeoutError("validator timeout"),
        ]
    )

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.final_policy.decision == "MAYBE"
    assert result.route == "VALIDATOR_FAILURE_SAFE_MAYBE"
    assert result.safe_fallback


def test_reviewer_generation_failure_is_safe_maybe():
    proto = protocol()
    engine = FakeEngine(
        [
            envelope(proto, screening_task="MISSING_OR_UNCLEAR"),
            TimeoutError("review timeout"),
        ]
    )

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.final_policy.decision == "MAYBE"
    assert result.route == "REVIEW_FAILURE_SAFE_MAYBE"
    assert result.safe_fallback


def test_reviewer_parse_failure_is_safe_maybe():
    proto = protocol()
    engine = FakeEngine(
        [envelope(proto, screening_task="MISSING_OR_UNCLEAR"), "broken output"]
    )

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.final_policy.decision == "MAYBE"
    assert result.route == "REVIEW_FAILURE_SAFE_MAYBE"


def test_invalid_primary_decisive_citation_triggers_review_instead_of_reject():
    proto = protocol()
    engine = FakeEngine(
        [
            envelope(
                proto,
                non_review_task="DIRECT_SUPPORT",
                bad_quote_for="non_review_task",
            ),
            envelope(proto),
        ]
    )

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.primary.evidence_result.safe_downgrade_count == 1
    assert result.review_used
    assert result.final_policy.decision == "KEEP"


def test_invalid_reviewer_rejection_evidence_downgrades_to_maybe_without_validator():
    proto = protocol()
    engine = FakeEngine(
        [
            envelope(proto, screening_task="MISSING_OR_UNCLEAR"),
            envelope(
                proto,
                non_review_task="DIRECT_SUPPORT",
                bad_quote_for="non_review_task",
            ),
        ]
    )

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.reviewer.evidence_result.safe_downgrade_count == 1
    assert result.reviewer.policy_result.decision == "KEEP"
    assert result.final_policy.decision == "MAYBE"
    assert not result.validator_used


def test_invalid_validator_rejection_evidence_cannot_confirm_reject():
    proto = protocol()
    engine = FakeEngine(
        [
            envelope(proto, screening_task="MISSING_OR_UNCLEAR"),
            envelope(proto, non_review_task="DIRECT_SUPPORT"),
            envelope(
                proto,
                non_review_task="DIRECT_SUPPORT",
                bad_quote_for="non_review_task",
            ),
        ]
    )

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.validator.policy_result.decision == "KEEP"
    assert result.final_policy.decision == "MAYBE"
    assert result.route == "REJECTION_UNCONFIRMED"


def test_reviewer_prompt_is_blind_and_role_specific():
    proto = protocol()
    primary = envelope(
        proto,
        screening_task="MISSING_OR_UNCLEAR",
    )
    primary["assessments"][1]["rationale"] = "SECRET_PRIMARY_RATIONALE"
    engine = FakeEngine([primary, envelope(proto)])

    orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    reviewer_prompt = engine.calls[1]["prompt"]
    assert "deep local exclusion reviewer" in reviewer_prompt
    assert "blind to any prior model output" in reviewer_prompt
    assert "SECRET_PRIMARY_RATIONALE" not in reviewer_prompt


def test_validator_prompt_is_blind_and_role_specific():
    proto = protocol()
    engine = FakeEngine(
        [
            envelope(proto, screening_task="MISSING_OR_UNCLEAR"),
            envelope(proto, non_review_task="DIRECT_SUPPORT"),
            envelope(proto),
        ]
    )

    orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    validator_prompt = engine.calls[2]["prompt"]
    assert "independent local semantic safety validator" in validator_prompt
    assert "without access to prior model outputs" in validator_prompt


def test_all_calls_use_strict_assessment_envelope_schema():
    proto = protocol()
    engine = FakeEngine(
        [
            envelope(proto, screening_task="MISSING_OR_UNCLEAR"),
            envelope(proto, non_review_task="DIRECT_SUPPORT"),
            envelope(proto, non_review_task="DIRECT_SUPPORT"),
        ]
    )

    orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert all(call["schema"] is ModelAssessmentEnvelope for call in engine.calls)


def test_model_plan_controls_models_and_timeouts():
    proto = protocol()
    plan = LocalModelPlan(
        primary_model="primary-local",
        review_model="deep-local",
        validator_model="validator-local",
        primary_timeout_seconds=11,
        review_timeout_seconds=22,
        validator_timeout_seconds=33,
    )
    engine = FakeEngine(
        [
            envelope(proto, screening_task="MISSING_OR_UNCLEAR"),
            envelope(proto, non_review_task="DIRECT_SUPPORT"),
            envelope(proto, non_review_task="DIRECT_SUPPORT"),
        ]
    )

    orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
        model_plan=plan,
    )

    assert [call["model"] for call in engine.calls] == [
        "primary-local",
        "deep-local",
        "validator-local",
    ]
    assert [call["timeout_seconds"] for call in engine.calls] == [11, 22, 33]


def test_elapsed_seconds_are_preserved_from_engine_results():
    proto = protocol()
    engine = FakeEngine([FakeGeneration(envelope(proto), elapsed_seconds=1.75)])

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.primary.elapsed_seconds == 1.75


def test_invalid_paper_id_is_rejected_before_model_call():
    engine = FakeEngine([])
    with pytest.raises(ValueError, match="paper_id is required"):
        orchestrate_local_v2_assessment(
            engine,
            protocol(),
            paper_id="  ",
            title=TITLE,
            abstract=ABSTRACT,
        )
    assert engine.calls == []


def test_protocol_without_identity_is_rejected_before_model_call():
    proto = protocol().model_copy(update={"protocol_id": ""})
    engine = FakeEngine([])
    with pytest.raises(ValueError, match="deterministic protocol_id"):
        orchestrate_local_v2_assessment(
            engine,
            proto,
            paper_id="p-1",
            title=TITLE,
            abstract=ABSTRACT,
        )
    assert engine.calls == []


def test_generation_value_must_be_mapping_or_string_and_fails_safe():
    proto = protocol()
    engine = FakeEngine([FakeGeneration(123), envelope(proto)])

    result = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    assert result.primary.generation_error
    assert result.final_policy.decision == "KEEP"


def test_manual_unconfirmed_reject_result_is_invalid():
    proto = protocol()
    engine = FakeEngine([envelope(proto)])
    valid = orchestrate_local_v2_assessment(
        engine,
        proto,
        paper_id="p-1",
        title=TITLE,
        abstract=ABSTRACT,
    )

    with pytest.raises(ValidationError, match="REJECT is only allowed"):
        LocalV2OrchestrationResult(
            protocol_id=proto.protocol_id,
            paper_id="p-1",
            model_plan=LocalModelPlan(),
            route="REVIEW_KEEP",
            primary=valid.primary,
            final_policy=PolicyResult(
                decision="REJECT",
                reason="unsafe manual rejection",
                decisive_criterion_ids=["non_review_task"],
            ),
        )


def test_model_plan_rejects_nonpositive_timeout():
    with pytest.raises(ValidationError):
        LocalModelPlan(primary_timeout_seconds=0)


def test_model_plan_strips_model_names_and_rejects_blank_name():
    plan = LocalModelPlan(primary_model="  qwen3.5:4b  ")
    assert plan.primary_model == "qwen3.5:4b"
    with pytest.raises(ValidationError, match="model name must not be empty"):
        LocalModelPlan(review_model="   ")
