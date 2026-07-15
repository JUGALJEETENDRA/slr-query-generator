from __future__ import annotations

import json

import pytest

from local_ai.cache import cache_key
from external_ai.orchestrator import PROMPT_VERSION
from local_ai.contracts import PaperAssessment, ReviewProtocol
from local_ai.evidence import build_evidence_units
from local_ai.engine import GenerationResult, LocalAIError, LocalAIMemoryError, LocalAIOutputError
from local_ai.hardware import HardwareSnapshot, classify_tier, resolve_runtime_profile
from external_ai.orchestrator import ExternalAIScreeningOrchestrator
from local_ai.validator import validate_assessment


def _hardware(ram=24.0):
    return HardwareSnapshot(
        total_ram_gb=ram,
        available_ram_gb=ram - 4,
        cpu_cores=8,
        platform="test",
        installed_models={
            "qwen2.5:3b": 1_900_000_000,
            "qwen3:8b": 5_200_000_000,
            "qwen3:14b": 9_300_000_000,
        },
    )


def _protocol(model="qwen3:8b"):
    return {
        "research_question": "Does the method address the target task?",
        "objective": "Identify direct evidence for the target task.",
        "scope_interpretation": "Include papers applying the method to the task.",
        "criteria": [
            {
                "id": "direct_task_evidence",
                "kind": "inclusion",
                "description": "The paper applies the method to the target task.",
                "required": True,
                "expected_evidence": "A direct application statement.",
                "source": "research_question",
            }
        ],
        "expected_relationships": ["method applied to task"],
        "ambiguities": [],
        "model": model,
    }


def _assessment(decision="KEEP", verdict="MET", evidence_id="abstract_001", confidence=0.9):
    return {
        "summary": "The paper directly applies the method.",
        "criteria": [
            {
                "criterion_id": "direct_task_evidence",
                "verdict": verdict,
                "rationale": "The abstract states the direct application.",
                "evidence": [{"source": "abstract", "evidence_id": evidence_id}] if evidence_id else [],
            }
        ],
        "contradictions": [],
        "missing_information": [],
        "decision": decision,
        "confidence": confidence,
        "reason": "Direct evidence matches the review protocol.",
        "uncertainty": [] if decision != "MAYBE" else ["Scope is not fully established."],
    }


class QueueEngine:
    def __init__(self, values):
        self.values = list(values)
        self.models = []
        self.unloaded = []

    def generate(self, model, prompt, schema):
        self.models.append(model)
        value = self.values.pop(0)
        return GenerationResult(value=value, model=model, elapsed_seconds=0.01)

    def unload(self, model):
        self.unloaded.append(model)


def test_hardware_tiers_and_manual_override(monkeypatch):
    assert classify_tier(8) == "compact"
    assert classify_tier(16) == "balanced"
    assert classify_tier(24) == "performance"
    monkeypatch.delenv("FAST_MODEL", raising=False)
    monkeypatch.delenv("STRONG_MODEL", raising=False)
    profile = resolve_runtime_profile("balanced", "maximum", _hardware(8))
    assert profile.resolved_tier == "balanced"
    assert profile.fast_model == "qwen3:8b"
    assert profile.strong_model == "qwen3:8b"
    assert profile.resource_profile == "maximum"
    assert profile.num_ctx == 4096


def test_evidence_id_and_decision_consistency_validation():
    protocol = ReviewProtocol.model_validate(_protocol()).with_identity()
    title = "A direct study"
    abstract = "This study applies the method to the target task in practice."
    valid = PaperAssessment.model_validate(_assessment())
    report = validate_assessment(valid, protocol, title, abstract)
    assert report.valid is True
    unsupported = PaperAssessment.model_validate(_assessment(evidence_id="invented_evidence"))
    report = validate_assessment(unsupported, protocol, title, abstract)
    assert report.valid is False
    assert any("unknown evidence id" in error for error in report.errors)


def test_reject_without_affirmative_evidence_is_invalid():
    protocol = ReviewProtocol.model_validate(_protocol()).with_identity()
    assessment = PaperAssessment.model_validate(
        _assessment(decision="REJECT", verdict="UNCLEAR", evidence_id="", confidence=0.9)
    )
    report = validate_assessment(assessment, protocol, "Title", "Abstract")
    assert report.valid is False
    assert any("REJECT lacks" in error for error in report.errors)


def test_affirmative_exclusion_semantics_are_domain_neutral():
    protocol = ReviewProtocol.model_validate({
        "research_question": "Which glimmers support a norvax process?",
        "objective": "Find direct norvax applications.",
        "scope_interpretation": "Include applied glimmer studies.",
        "criteria": [
            {
                "id": "required_application", "kind": "inclusion", "required": True,
                "description": "The glimmer is applied to a norvax process.",
                "expected_evidence": "Direct application evidence.", "source": "research_question",
            },
            {
                "id": "different_context", "kind": "exclusion", "required": True,
                "description": "The glimmer is used only in an orbital habitat.",
                "expected_evidence": "Affirmative orbital-only evidence.", "source": "research_question",
            },
        ],
    }).with_identity()
    keep = PaperAssessment.model_validate({
        "summary": "Applied norvax work.",
        "criteria": [
            {"criterion_id": "required_application", "verdict": "MET", "rationale": "Direct.",
             "evidence": [{"source": "abstract", "evidence_id": "abstract_001"}]},
            {"criterion_id": "different_context", "verdict": "NOT_MET", "rationale": "Not orbital.",
             "evidence": []},
        ],
        "contradictions": [], "missing_information": [], "decision": "KEEP",
        "confidence": 0.9, "reason": "Required application is present.", "uncertainty": [],
    })
    keep_report = validate_assessment(
        keep, protocol, "Norvax glimmer", "The glimmer is applied to a norvax process."
    )
    assert keep_report.valid is True

    reject_payload = keep.model_dump(mode="json")
    reject_payload["criteria"][1].update({
        "verdict": "MET", "rationale": "Orbital-only context.",
        "evidence": [{"source": "abstract", "evidence_id": "abstract_001"}],
    })
    reject_payload.update({"decision": "REJECT", "reason": "Affirmative exclusion applies."})
    reject = PaperAssessment.model_validate(reject_payload)
    reject_report = validate_assessment(
        reject, protocol, "Orbital glimmer", "The glimmer is used only in an orbital habitat."
    )
    assert reject_report.valid is True


@pytest.mark.parametrize(
    ("question", "title", "abstract"),
    [
        ("Which velons support a tessera process?", "Tessera velon", "The velon is applied to a tessera process."),
        ("Which quorils support a pavo process?", "Pavo quoril", "The quoril is applied to a pavo process."),
    ],
)
def test_identical_contract_logic_works_with_unrelated_wording(question, title, abstract):
    payload = _protocol()
    payload.update({"research_question": question, "objective": "Find direct applications."})
    protocol = ReviewProtocol.model_validate(payload).with_identity()
    assessment = PaperAssessment.model_validate(_assessment())
    assert validate_assessment(assessment, protocol, title, abstract).valid is True


def test_evidence_units_preserve_exact_source_text_and_stable_ids():
    title = "  A title with spacing  "
    abstract = "First sentence.  Second sentence!\nThird sentence?"
    units = build_evidence_units(title, abstract)
    assert [unit["evidence_id"] for unit in units] == [
        "title_001", "abstract_001", "abstract_002", "abstract_003"
    ]
    assert all(unit["text"] in (title if unit["source"] == "title" else abstract) for unit in units)


def test_external_cache_namespace_cannot_reuse_legacy_contracts():
    assert PROMPT_VERSION == "external-gemini-v3"
    assert cache_key("paper", PROMPT_VERSION) != cache_key("paper", "local-ai-first-v1")
    assert cache_key("paper", PROMPT_VERSION) != cache_key("paper", "local-ai-first-v2")


def test_compact_tier_uses_protocol_critic_and_two_paper_passes(tmp_path):
    profile = resolve_runtime_profile("compact", "balanced", _hardware(8))
    evidence = {
        "summary": "The paper applies the method.",
        "criteria": _assessment()["criteria"],
        "contradictions": [],
        "missing_information": [],
    }
    engine = QueueEngine([_protocol("qwen2.5:3b"), _protocol("qwen2.5:3b"), evidence, _assessment()])
    orchestrator = ExternalAIScreeningOrchestrator(profile, engine=engine)
    orchestrator.cache.directory = tmp_path
    protocol = orchestrator.compile_protocol("Does the method address the target task?")
    envelope = orchestrator.assess_fast(
        protocol,
        "A direct study",
        "This study applies the method to the target task in practice.",
    )
    assert envelope.validation.valid is True
    assert engine.models == ["qwen2.5:3b"] * 4


def test_inferred_exclusion_is_repaired_into_positive_rq_scope(tmp_path):
    invalid = _protocol()
    invalid["criteria"].append({
        "id": "inverse_scope",
        "kind": "exclusion",
        "description": "The paper must not omit the target task.",
        "required": True,
        "expected_evidence": "Missing target-task evidence.",
        "source": "research_question",
    })
    corrected = _protocol()
    profile = resolve_runtime_profile("balanced", "balanced", _hardware(16))
    engine = QueueEngine([invalid, corrected, corrected])
    orchestrator = ExternalAIScreeningOrchestrator(profile, engine=engine)
    orchestrator.cache.directory = tmp_path
    protocol = orchestrator.compile_protocol("Does the method address the target task?")
    assert all(
        criterion.kind == "inclusion" for criterion in protocol.criteria
        if criterion.source == "research_question"
    )
    assert engine.models == ["qwen3:8b"] * 3


def test_malformed_protocol_json_retries_same_8b_without_downgrade(tmp_path):
    class MalformedProtocolEngine(QueueEngine):
        def generate(self, model, prompt, schema):
            self.models.append(model)
            value = self.values.pop(0)
            if isinstance(value, Exception):
                raise value
            return GenerationResult(value=value, model=model, elapsed_seconds=0.01)

    profile = resolve_runtime_profile("performance", "balanced", _hardware(24))
    engine = MalformedProtocolEngine([
        LocalAIOutputError("malformed protocol JSON"), _protocol(), _protocol()
    ])
    orchestrator = ExternalAIScreeningOrchestrator(profile, engine=engine)
    orchestrator.cache.directory = tmp_path
    protocol = orchestrator.compile_protocol("Does the method address the target task?")
    assert protocol.model == "qwen3:8b"
    assert orchestrator.runtime_downgrades == []
    assert engine.models == ["qwen3:8b"] * 3


def test_explicit_user_exclusion_remains_authoritative(tmp_path):
    protocol_payload = _protocol()
    protocol_payload["criteria"].append({
        "id": "orbital_only",
        "kind": "exclusion",
        "description": "The work is conducted exclusively in an orbital habitat.",
        "required": True,
        "expected_evidence": "Affirmative evidence of an orbital-only setting.",
        "source": "user",
    })
    profile = resolve_runtime_profile("balanced", "balanced", _hardware(16))
    engine = QueueEngine([protocol_payload, protocol_payload])
    orchestrator = ExternalAIScreeningOrchestrator(profile, engine=engine)
    orchestrator.cache.directory = tmp_path
    protocol = orchestrator.compile_protocol(
        "Does the method address the target task?",
        exclusion_criteria="Exclude work conducted exclusively in an orbital habitat",
    )
    exclusions = [criterion for criterion in protocol.criteria if criterion.kind == "exclusion"]
    assert len(exclusions) == 1
    assert exclusions[0].source == "user"


def test_performance_tier_keeps_valid_maybe_without_repair(tmp_path):
    profile = resolve_runtime_profile("performance", "maximum", _hardware(24))
    maybe = _assessment(decision="MAYBE", verdict="UNCLEAR", evidence_id="", confidence=0.55)
    engine = QueueEngine([_protocol("qwen3:8b"), _protocol("qwen3:8b"), maybe])
    orchestrator = ExternalAIScreeningOrchestrator(profile, engine=engine)
    orchestrator.cache.directory = tmp_path
    protocol = orchestrator.compile_protocol("Does the method address the target task?")
    fast = orchestrator.assess_fast(
        protocol, "A direct study", "This study applies the method to the target task in practice."
    )
    assert fast.assessment.decision == "MAYBE"
    assert fast.validation.valid is True
    assert fast.needs_escalation() is False
    assert engine.models == ["qwen3:8b"] * 3
    assert engine.unloaded == []


def test_invalid_assessment_gets_one_8b_repair(tmp_path):
    profile = resolve_runtime_profile("performance", "maximum", _hardware(24))
    invalid = _assessment(decision="KEEP", verdict="MET", evidence_id="unknown")
    engine = QueueEngine([
        _protocol("qwen3:8b"), _protocol("qwen3:8b"), invalid, _assessment()
    ])
    orchestrator = ExternalAIScreeningOrchestrator(profile, engine=engine)
    orchestrator.cache.directory = tmp_path
    result = orchestrator.screen_paper(
        "Does the method address the target task?", "A direct study",
        "This study applies the method to the target task in practice.",
    )
    assert result["decision"] == "KEEP"
    assert result["validation_status"] == "validated"
    assert result["attempts"] == 2
    assert engine.models == ["qwen3:8b"] * 4


def test_malformed_8b_output_repairs_with_8b_without_tier_downgrade(tmp_path):
    class MalformedOnceEngine(QueueEngine):
        def generate(self, model, prompt, schema):
            self.models.append(model)
            value = self.values.pop(0)
            if isinstance(value, Exception):
                raise value
            return GenerationResult(value=value, model=model, elapsed_seconds=0.01)

    profile = resolve_runtime_profile("performance", "maximum", _hardware(24))
    engine = MalformedOnceEngine([
        _protocol(), _protocol(), LocalAIOutputError("malformed JSON"), _assessment()
    ])
    orchestrator = ExternalAIScreeningOrchestrator(profile, engine=engine)
    orchestrator.cache.directory = tmp_path
    result = orchestrator.screen_paper(
        "Does the method address the target task?", "A direct study",
        "This study applies the method to the target task in practice.",
    )
    assert result["decision"] == "KEEP"
    assert result["model"] == "qwen3:8b"
    assert result["runtime_downgrades"] == []
    assert orchestrator.profile.resolved_tier == "performance"
    assert engine.models == ["qwen3:8b"] * 4


def test_malformed_assessment_gets_exactly_one_fresh_repair(tmp_path):
    profile = resolve_runtime_profile("balanced", "balanced", _hardware(16))
    malformed = {"summary": "Incomplete output", "criteria": []}
    engine = QueueEngine([_protocol(), _protocol(), malformed, _assessment()])
    orchestrator = ExternalAIScreeningOrchestrator(profile, engine=engine)
    orchestrator.cache.directory = tmp_path
    result = orchestrator.screen_paper(
        "Does the method address the target task?", "A direct study",
        "This study applies the method to the target task in practice.",
    )
    assert result["decision"] == "KEEP"
    assert result["attempts"] == 2
    assert len(engine.models) == 4


def test_public_result_contains_versioned_audit_payload(tmp_path):
    profile = resolve_runtime_profile("balanced", "balanced", _hardware(16))
    engine = QueueEngine([_protocol("qwen3:8b"), _protocol("qwen3:8b"), _assessment()])
    orchestrator = ExternalAIScreeningOrchestrator(profile, engine=engine)
    orchestrator.cache.directory = tmp_path
    result = orchestrator.screen_paper(
        "Does the method address the target task?",
        "A direct study",
        "This study applies the method to the target task in practice.",
    )
    assert result["schema_version"] == "2.0"
    assert result["decision"] == "KEEP"
    assert result["validation_status"] == "validated"
    assert result["evidence"][0]["quote"] in "This study applies the method to the target task in practice."


def test_cached_assessment_reports_zero_current_time_and_historical_time(tmp_path):
    profile = resolve_runtime_profile("balanced", "balanced", _hardware(16))
    engine = QueueEngine([_protocol(), _protocol(), _assessment()])
    orchestrator = ExternalAIScreeningOrchestrator(profile, engine=engine)
    orchestrator.cache.directory = tmp_path
    protocol = orchestrator.compile_protocol("Does the method address the target task?")
    title = "A direct study"
    abstract = "This study applies the method to the target task in practice."
    fresh = orchestrator.assess_fast(protocol, title, abstract)
    cached = orchestrator.assess_fast(protocol, title, abstract)
    assert fresh.cache_hit is False
    assert cached.cache_hit is True
    assert cached.elapsed_seconds == 0.0
    assert cached.original_elapsed_seconds == pytest.approx(fresh.original_elapsed_seconds)
    assert cached.model == "qwen3:8b"


def test_model_failure_can_only_produce_unresolved_maybe(tmp_path):
    class BrokenEngine:
        def generate(self, model, prompt, schema):
            raise LocalAIError("model unavailable")

    profile = resolve_runtime_profile("compact", "eco", _hardware(8))
    orchestrator = ExternalAIScreeningOrchestrator(profile, engine=BrokenEngine())
    orchestrator.cache.directory = tmp_path
    result = orchestrator.screen_paper("A valid research question?", "Title", "Abstract")
    assert result["decision"] == "MAYBE"
    assert result["validation_status"] == "unresolved"
    assert result["confidence"] == 0.0


def test_authoritative_user_criteria_cannot_be_silently_dropped(tmp_path):
    profile = resolve_runtime_profile("balanced", "balanced", _hardware(16))
    engine = QueueEngine([_protocol("qwen3:8b"), _protocol("qwen3:8b")])
    orchestrator = ExternalAIScreeningOrchestrator(profile, engine=engine)
    orchestrator.cache.directory = tmp_path
    try:
        orchestrator.compile_protocol(
            "Does the method address the target task?",
            inclusion_criteria="Must evaluate real-world data",
        )
    except ValueError as exc:
        assert "authoritative user criteria" in str(exc)
    else:
        raise AssertionError("protocol compilation accepted a dropped user criterion")


def test_fast_model_oom_skips_to_a_smaller_distinct_model(tmp_path):
    class OOMOnceEngine:
        def __init__(self):
            self.calls = 0
        def generate(self, model, prompt, schema):
            self.calls += 1
            if self.calls == 1:
                raise LocalAIMemoryError("out of memory")
            return GenerationResult(
                value=_protocol("qwen2.5:3b"), model=model, elapsed_seconds=0.01
            )

    profile = resolve_runtime_profile("performance", "maximum", _hardware(24))
    orchestrator = ExternalAIScreeningOrchestrator(profile, engine=OOMOnceEngine())
    orchestrator.cache.directory = tmp_path
    protocol = orchestrator.compile_protocol("Does the method address the target task?")
    assert protocol.model == "qwen2.5:3b"
    assert orchestrator.profile.resolved_tier == "compact"
    assert orchestrator.profile.fast_model == "qwen2.5:3b"
    assert orchestrator.runtime_downgrades
