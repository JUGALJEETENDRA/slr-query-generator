from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from litsync_app.screening.local_v2 import (
    RUNNER_VERSION,
    LocalModelPlan,
    LocalV2EndToEndResult,
    LocalV2Paper,
    LocalV2PaperRunResult,
    ModelAssessmentEnvelope,
    ProtocolCriterion,
    ScreeningProtocolV2,
    compile_and_run_local_v2_paper,
    compile_protocol_draft,
    run_compiled_local_v2_paper,
)


TITLE = "Large language models automate abstract screening."
ABSTRACT = (
    "We evaluate a large language model for title and abstract screening in systematic "
    "reviews."
)
REJECT_ABSTRACT = ABSTRACT + " The work is unrelated to literature-review screening."


@dataclass
class FakeGeneration:
    value: Any
    elapsed_seconds: float = 0.25


class FakeEngine:
    def __init__(self, responses: list[Any] | None = None):
        self.responses = list(responses or [])
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
                expected_evidence="Explicit mention of a large language model.",
                resolution_required=True,
            ),
            ProtocolCriterion(
                id="unrelated_task",
                role="EXCLUSION_TRIGGER",
                description="The work is explicitly unrelated to review screening.",
                expected_evidence="An explicit incompatible task or setting.",
                resolution_required=False,
            ),
        ],
        model="qwen3.5:4b",
    ).with_identity()


def draft() -> dict[str, Any]:
    return {
        "research_question": "Can LLMs automate systematic-review screening?",
        "research_context": "Title and abstract screening.",
        "model": "qwen3.5:4b",
        "criteria": [
            {
                "label": "Uses LLM",
                "role": "REQUIRED_INCLUSION",
                "description": "The paper studies a large language model.",
                "expected_evidence": "Explicit mention of a large language model.",
            },
            {
                "label": "Unrelated task",
                "role": "EXCLUSION_TRIGGER",
                "description": "The work is explicitly unrelated to review screening.",
                "expected_evidence": "An explicit incompatible task or setting.",
            },
        ],
    }


def citation(evidence_id: str, source: str, quote: str) -> list[dict[str, str]]:
    return [{"evidence_id": evidence_id, "source": source, "quote": quote}]


def envelope(
    proto: ScreeningProtocolV2,
    *,
    paper_id: str = "p-1",
    uses_llm: str = "DIRECT_SUPPORT",
    unrelated_task: str = "MISSING_OR_UNCLEAR",
    support_source: str = "abstract",
    rejection: bool = False,
) -> dict[str, Any]:
    if support_source == "title":
        support_evidence = citation(
            "title_001",
            "title",
            "Large language models automate abstract screening",
        )
    else:
        support_evidence = citation(
            "abstract_001",
            "abstract",
            "We evaluate a large language model for title and abstract screening",
        )

    uses_evidence = (
        support_evidence
        if uses_llm in {"DIRECT_SUPPORT", "DIRECT_CONTRADICTION"}
        else []
    )
    exclusion_evidence = (
        citation(
            "abstract_002",
            "abstract",
            "The work is unrelated to literature-review screening",
        )
        if rejection and unrelated_task == "DIRECT_SUPPORT"
        else []
    )
    return {
        "protocol_id": proto.protocol_id,
        "paper_id": paper_id,
        "assessments": [
            {
                "criterion_id": "uses_llm",
                "relation": uses_llm,
                "rationale": "The paper explicitly describes its model.",
                "evidence": uses_evidence,
            },
            {
                "criterion_id": "unrelated_task",
                "relation": unrelated_task,
                "rationale": "The exclusion trigger was assessed independently.",
                "evidence": exclusion_evidence,
            },
        ],
    }


def paper(**updates: Any) -> dict[str, Any]:
    value = {"paper_id": "p-1", "title": TITLE, "abstract": ABSTRACT}
    value.update(updates)
    return value


def test_runner_version_is_frozen():
    assert RUNNER_VERSION == "local-v2-runner-v1"


def test_paper_normalizes_boundary_whitespace():
    value = LocalV2Paper(paper_id="  p-1  ", title="  Title  ", abstract="  Abstract  ")
    assert value.paper_id == "p-1"
    assert value.title == "Title"
    assert value.abstract == "Abstract"


def test_paper_blank_optional_text_becomes_none():
    value = LocalV2Paper(paper_id="p-1", title="  ", abstract="\n\t")
    assert value.title is None
    assert value.abstract is None


def test_paper_rejects_blank_id():
    with pytest.raises(ValidationError):
        LocalV2Paper(paper_id="   ")


def test_paper_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        LocalV2Paper.model_validate({"paper_id": "p-1", "unknown": True})


def test_compile_and_run_primary_keep_is_complete_end_to_end():
    compiled = compile_protocol_draft(draft())
    assert compiled.success and compiled.protocol is not None
    engine = FakeEngine([envelope(compiled.protocol)])

    result = compile_and_run_local_v2_paper(engine, draft(), paper=paper())

    assert result.status == "COMPLETED"
    assert result.compilation.success
    assert result.paper_result is not None
    assert result.final_policy.decision == "KEEP"
    assert result.paper_result.model_call_count == 1
    assert result.paper_result.elapsed_seconds == 0.25
    assert len(engine.calls) == 1


def test_compiled_runner_uses_primary_fast_path():
    proto = protocol()
    engine = FakeEngine([envelope(proto)])

    result = run_compiled_local_v2_paper(engine, proto, paper=paper())

    assert result.status == "COMPLETED"
    assert result.text_status == "TITLE_AND_ABSTRACT"
    assert result.orchestration is not None
    assert result.orchestration.route == "PRIMARY_KEEP_FAST_PATH"
    assert result.model_call_count == 1
    assert not result.safe_fallback


def test_title_only_paper_runs_local_model_and_records_warning():
    proto = protocol()
    engine = FakeEngine([envelope(proto, support_source="title")])

    result = run_compiled_local_v2_paper(
        engine,
        proto,
        paper=paper(abstract=None),
    )

    assert result.text_status == "TITLE_ONLY"
    assert result.final_policy.decision == "KEEP"
    assert result.model_call_count == 1
    assert result.warnings == [
        "Abstract is unavailable; screening used title evidence only."
    ]
    assert '"source":"title"' in engine.calls[0]["prompt"]


def test_abstract_only_paper_runs_local_model_and_records_warning():
    proto = protocol()
    engine = FakeEngine([envelope(proto)])

    result = run_compiled_local_v2_paper(
        engine,
        proto,
        paper=paper(title=None),
    )

    assert result.text_status == "ABSTRACT_ONLY"
    assert result.final_policy.decision == "KEEP"
    assert result.model_call_count == 1
    assert result.warnings == [
        "Title is unavailable; screening used abstract evidence only."
    ]


def test_no_screenable_text_returns_safe_maybe_without_model_call():
    proto = protocol()
    engine = FakeEngine()

    result = run_compiled_local_v2_paper(
        engine,
        proto,
        paper=paper(title=" ", abstract=None),
    )

    assert result.status == "NO_SCREENABLE_TEXT"
    assert result.text_status == "NO_SCREENABLE_TEXT"
    assert result.final_policy.decision == "MAYBE"
    assert result.final_policy.safe_fallback
    assert result.safe_fallback
    assert result.model_call_count == 0
    assert result.elapsed_seconds == 0
    assert result.orchestration is None
    assert not engine.calls


def test_end_to_end_no_text_preserves_successful_compilation():
    engine = FakeEngine()

    result = compile_and_run_local_v2_paper(
        engine,
        draft(),
        paper=paper(title=None, abstract="  "),
    )

    assert result.status == "NO_SCREENABLE_TEXT"
    assert result.compilation.success
    assert result.paper_result is not None
    assert result.final_policy.decision == "MAYBE"
    assert result.safe_fallback
    assert not engine.calls


def test_protocol_compilation_failure_never_calls_model():
    invalid = draft()
    invalid["criteria"] = [
        {
            "label": "Only exclusion",
            "role": "EXCLUSION_TRIGGER",
            "description": "An exclusion without required inclusion.",
        }
    ]
    engine = FakeEngine()

    result = compile_and_run_local_v2_paper(engine, invalid, paper=paper())

    assert result.status == "PROTOCOL_COMPILATION_FAILED"
    assert not result.compilation.success
    assert result.paper_result is None
    assert result.final_policy.decision == "MAYBE"
    assert result.safe_fallback
    assert any("NO_REQUIRED_INCLUSION" in item for item in result.final_policy.policy_errors)
    assert not engine.calls


def test_invalid_draft_structure_is_exposed_as_safe_policy_error():
    engine = FakeEngine()

    result = compile_and_run_local_v2_paper(
        engine,
        {"research_question": "RQ", "criteria": "not-a-list"},
        paper=paper(),
    )

    assert result.status == "PROTOCOL_COMPILATION_FAILED"
    assert any("INVALID_DRAFT" in item for item in result.final_policy.policy_errors)
    assert not engine.calls


def test_custom_model_plan_reaches_orchestrator_unchanged():
    proto = protocol()
    plan = LocalModelPlan(
        primary_model="local-primary",
        review_model="local-review",
        validator_model="local-validator",
        primary_timeout_seconds=11,
        review_timeout_seconds=22,
        validator_timeout_seconds=33,
    )
    engine = FakeEngine([envelope(proto)])

    result = run_compiled_local_v2_paper(
        engine,
        proto,
        paper=paper(),
        model_plan=plan,
    )

    assert result.orchestration is not None
    assert result.orchestration.model_plan == plan
    assert engine.calls[0]["model"] == "local-primary"
    assert engine.calls[0]["timeout_seconds"] == 11


def test_blind_review_keep_reports_two_model_calls_and_summed_time():
    proto = protocol()
    engine = FakeEngine(
        [
            FakeGeneration(envelope(proto, uses_llm="MISSING_OR_UNCLEAR"), 0.4),
            FakeGeneration(envelope(proto), 0.6),
        ]
    )

    result = run_compiled_local_v2_paper(engine, proto, paper=paper())

    assert result.final_policy.decision == "KEEP"
    assert result.orchestration is not None
    assert result.orchestration.route == "REVIEW_KEEP"
    assert result.model_call_count == 2
    assert result.elapsed_seconds == 1.0
    assert [call["model"] for call in engine.calls] == ["qwen3.5:4b", "qwen3:8b"]


def test_confirmed_rejection_reports_three_model_calls():
    proto = protocol()
    engine = FakeEngine(
        [
            envelope(proto, uses_llm="MISSING_OR_UNCLEAR"),
            envelope(proto, unrelated_task="DIRECT_SUPPORT", rejection=True),
            envelope(proto, unrelated_task="DIRECT_SUPPORT", rejection=True),
        ]
    )

    result = run_compiled_local_v2_paper(
        engine,
        proto,
        paper=paper(abstract=REJECT_ABSTRACT),
    )

    assert result.final_policy.decision == "REJECT"
    assert result.orchestration is not None
    assert result.orchestration.route == "REJECTION_CONFIRMED"
    assert result.model_call_count == 3
    assert result.elapsed_seconds == 0.75
    assert not result.safe_fallback


def test_two_generation_failures_return_safe_maybe_with_two_attempts():
    proto = protocol()
    engine = FakeEngine([RuntimeError("primary down"), TimeoutError("review down")])

    result = run_compiled_local_v2_paper(engine, proto, paper=paper())

    assert result.final_policy.decision == "MAYBE"
    assert result.safe_fallback
    assert result.orchestration is not None
    assert result.orchestration.route == "REVIEW_FAILURE_SAFE_MAYBE"
    assert result.model_call_count == 2


def test_stale_protocol_identity_is_rejected_before_model_call():
    proto = protocol().model_copy(update={"research_question": "Changed question"})
    engine = FakeEngine()

    with pytest.raises(ValueError, match="protocol_id does not match"):
        run_compiled_local_v2_paper(engine, proto, paper=paper())

    assert not engine.calls


def test_missing_protocol_identity_is_rejected_before_model_call():
    proto = protocol().model_copy(update={"protocol_id": ""})
    engine = FakeEngine()

    with pytest.raises(ValueError, match="deterministic protocol_id"):
        run_compiled_local_v2_paper(engine, proto, paper=paper())

    assert not engine.calls


def test_compiled_protocol_can_be_reused_across_papers_without_recompilation():
    compiled = compile_protocol_draft(draft())
    assert compiled.success and compiled.protocol is not None
    proto = compiled.protocol
    engine = FakeEngine(
        [
            envelope(proto, paper_id="p-1"),
            envelope(proto, paper_id="p-2"),
        ]
    )

    first = run_compiled_local_v2_paper(engine, proto, paper=paper(paper_id="p-1"))
    second = run_compiled_local_v2_paper(engine, proto, paper=paper(paper_id="p-2"))

    assert first.protocol_id == second.protocol_id == proto.protocol_id
    assert first.final_policy.decision == second.final_policy.decision == "KEEP"
    assert len(engine.calls) == 2


def test_paper_object_is_accepted_without_reconstruction_side_effects():
    proto = protocol()
    value = LocalV2Paper(paper_id="p-1", title=TITLE, abstract=ABSTRACT)
    engine = FakeEngine([envelope(proto)])

    result = run_compiled_local_v2_paper(engine, proto, paper=value)

    assert result.paper is value


def test_engine_calls_use_the_strict_assessment_schema():
    proto = protocol()
    engine = FakeEngine([envelope(proto)])

    run_compiled_local_v2_paper(engine, proto, paper=paper())

    assert issubclass(engine.calls[0]["schema"], ModelAssessmentEnvelope)


def test_paper_result_round_trips_through_json_validation():
    proto = protocol()
    engine = FakeEngine([envelope(proto)])
    result = run_compiled_local_v2_paper(engine, proto, paper=paper())

    restored = LocalV2PaperRunResult.model_validate_json(result.model_dump_json())

    assert restored == result


def test_end_to_end_result_round_trips_through_json_validation():
    compiled = compile_protocol_draft(draft())
    assert compiled.success and compiled.protocol is not None
    engine = FakeEngine([envelope(compiled.protocol)])
    result = compile_and_run_local_v2_paper(engine, draft(), paper=paper())

    restored = LocalV2EndToEndResult.model_validate_json(result.model_dump_json())

    assert restored == result


def test_paper_result_cannot_hide_safe_fallback_state():
    proto = protocol()
    engine = FakeEngine([RuntimeError("primary down"), TimeoutError("review down")])
    result = run_compiled_local_v2_paper(engine, proto, paper=paper())
    payload = result.model_dump(mode="json")
    payload["safe_fallback"] = False

    with pytest.raises(ValidationError, match="cannot be hidden"):
        LocalV2PaperRunResult.model_validate(payload)


def test_missing_text_result_cannot_claim_model_work():
    proto = protocol()
    result = run_compiled_local_v2_paper(
        FakeEngine(),
        proto,
        paper=paper(title=None, abstract=None),
    )
    payload = result.model_dump(mode="json")
    payload["model_call_count"] = 1

    with pytest.raises(ValidationError, match="cannot report model work"):
        LocalV2PaperRunResult.model_validate(payload)


def test_end_to_end_result_rejects_mismatched_paper_result():
    compiled = compile_protocol_draft(draft())
    assert compiled.success and compiled.protocol is not None
    engine = FakeEngine([envelope(compiled.protocol)])
    result = compile_and_run_local_v2_paper(engine, draft(), paper=paper())
    payload = result.model_dump(mode="json")
    payload["paper"]["paper_id"] = "other-paper"

    with pytest.raises(ValidationError, match="must correspond"):
        LocalV2EndToEndResult.model_validate(payload)


def test_mapping_with_unknown_paper_field_fails_before_any_model_call():
    proto = protocol()
    engine = FakeEngine()

    with pytest.raises(ValidationError):
        run_compiled_local_v2_paper(
            engine,
            proto,
            paper={"paper_id": "p-1", "title": TITLE, "extra": "forbidden"},
        )

    assert not engine.calls
