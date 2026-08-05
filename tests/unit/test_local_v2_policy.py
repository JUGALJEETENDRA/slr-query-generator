from __future__ import annotations

import inspect

from litsync_app.screening.local_v2.contracts import (
    CriterionAssessment,
    EvidenceCitation,
    ProtocolCriterion,
    ScreeningProtocolV2,
)
from litsync_app.screening.local_v2.policy import derive_policy_decision


def _citation(evidence_id: str) -> EvidenceCitation:
    return EvidenceCitation(
        evidence_id=evidence_id,
        source="abstract",
        quote=f"Exact evidence for {evidence_id}.",
    )


def _criterion(
    criterion_id: str,
    role: str = "REQUIRED_INCLUSION",
    *,
    resolution_required: bool = True,
) -> ProtocolCriterion:
    return ProtocolCriterion(
        id=criterion_id,
        role=role,
        description=f"Criterion {criterion_id} must be evaluated.",
        resolution_required=resolution_required,
    )


def _protocol(*criteria: ProtocolCriterion) -> ScreeningProtocolV2:
    return ScreeningProtocolV2(
        research_question="Does the evidence satisfy this review protocol?",
        criteria=list(criteria) or [_criterion("core")],
    ).with_identity()


def _assessment(criterion_id: str, relation: str) -> CriterionAssessment:
    evidence = []
    if relation in {"DIRECT_SUPPORT", "DIRECT_CONTRADICTION"}:
        evidence = [_citation(f"{criterion_id}_evidence")]
    return CriterionAssessment(
        criterion_id=criterion_id,
        relation=relation,
        rationale=f"The relation is {relation}.",
        evidence=evidence,
    )


def test_all_required_inclusion_criteria_supported_returns_keep():
    protocol = _protocol(_criterion("population"), _criterion("method"))
    result = derive_policy_decision(
        protocol,
        [_assessment("population", "DIRECT_SUPPORT"), _assessment("method", "DIRECT_SUPPORT")],
    )
    assert result.decision == "KEEP"
    assert result.decisive_criterion_ids == []
    assert result.unresolved_criterion_ids == []
    assert not result.safe_fallback


def test_required_inclusion_directly_contradicted_returns_reject():
    protocol = _protocol(_criterion("method"))
    result = derive_policy_decision(protocol, [_assessment("method", "DIRECT_CONTRADICTION")])
    assert result.decision == "REJECT"
    assert result.decisive_criterion_ids == ["method"]


def test_exclusion_trigger_directly_supported_returns_reject():
    protocol = _protocol(
        _criterion("core"),
        _criterion("editorial", "EXCLUSION_TRIGGER", resolution_required=False),
    )
    result = derive_policy_decision(
        protocol,
        [_assessment("core", "DIRECT_SUPPORT"), _assessment("editorial", "DIRECT_SUPPORT")],
    )
    assert result.decision == "REJECT"
    assert result.decisive_criterion_ids == ["editorial"]


def test_required_inclusion_missing_or_unclear_returns_maybe():
    protocol = _protocol(_criterion("core"))
    result = derive_policy_decision(protocol, [_assessment("core", "MISSING_OR_UNCLEAR")])
    assert result.decision == "MAYBE"
    assert result.unresolved_criterion_ids == ["core"]


def test_required_inclusion_not_applicable_returns_maybe():
    protocol = _protocol(_criterion("core"))
    result = derive_policy_decision(protocol, [_assessment("core", "NOT_APPLICABLE")])
    assert result.decision == "MAYBE"
    assert result.unresolved_criterion_ids == ["core"]


def test_unresolved_non_required_exclusion_does_not_block_keep():
    protocol = _protocol(
        _criterion("core"),
        _criterion("optional_exclusion", "EXCLUSION_TRIGGER", resolution_required=False),
    )
    result = derive_policy_decision(
        protocol,
        [_assessment("core", "DIRECT_SUPPORT"), _assessment("optional_exclusion", "MISSING_OR_UNCLEAR")],
    )
    assert result.decision == "KEEP"
    assert result.unresolved_criterion_ids == []


def test_unresolved_resolution_required_exclusion_returns_maybe():
    protocol = _protocol(
        _criterion("core"),
        _criterion("required_exclusion", "EXCLUSION_TRIGGER", resolution_required=True),
    )
    result = derive_policy_decision(
        protocol,
        [_assessment("core", "DIRECT_SUPPORT"), _assessment("required_exclusion", "NOT_APPLICABLE")],
    )
    assert result.decision == "MAYBE"
    assert result.unresolved_criterion_ids == ["required_exclusion"]


def test_explicit_rejection_takes_precedence_over_unresolved_criteria():
    protocol = _protocol(_criterion("first"), _criterion("second"))
    result = derive_policy_decision(
        protocol,
        [_assessment("first", "MISSING_OR_UNCLEAR"), _assessment("second", "DIRECT_CONTRADICTION")],
    )
    assert result.decision == "REJECT"
    assert result.decisive_criterion_ids == ["second"]
    assert result.unresolved_criterion_ids == ["first"]


def test_missing_assessment_returns_safe_maybe():
    protocol = _protocol(_criterion("first"), _criterion("second"))
    result = derive_policy_decision(protocol, [_assessment("first", "DIRECT_SUPPORT")])
    assert result.decision == "MAYBE"
    assert result.safe_fallback
    assert any("missing assessment" in error for error in result.policy_errors)


def test_unknown_assessment_returns_safe_maybe():
    protocol = _protocol(_criterion("core"))
    result = derive_policy_decision(
        protocol,
        [_assessment("core", "DIRECT_SUPPORT"), _assessment("unknown", "DIRECT_SUPPORT")],
    )
    assert result.decision == "MAYBE"
    assert result.safe_fallback
    assert any("unknown assessment" in error for error in result.policy_errors)


def test_duplicate_assessment_returns_safe_maybe():
    protocol = _protocol(_criterion("core"))
    result = derive_policy_decision(
        protocol,
        [_assessment("core", "DIRECT_SUPPORT"), _assessment("core", "DIRECT_SUPPORT")],
    )
    assert result.decision == "MAYBE"
    assert result.safe_fallback
    assert any("duplicate assessment" in error for error in result.policy_errors)


def test_policy_result_criterion_ids_remain_in_protocol_order():
    protocol = _protocol(
        _criterion("first"),
        _criterion("second"),
        _criterion("third", "EXCLUSION_TRIGGER", resolution_required=True),
        _criterion("fourth", "EXCLUSION_TRIGGER", resolution_required=False),
    )
    result = derive_policy_decision(
        protocol,
        [
            _assessment("fourth", "DIRECT_SUPPORT"),
            _assessment("third", "MISSING_OR_UNCLEAR"),
            _assessment("second", "DIRECT_CONTRADICTION"),
            _assessment("first", "MISSING_OR_UNCLEAR"),
        ],
    )
    assert result.decisive_criterion_ids == ["second", "fourth"]
    assert result.unresolved_criterion_ids == ["first", "third"]


def test_input_assessment_order_does_not_change_decision_or_output_ordering():
    protocol = _protocol(_criterion("first"), _criterion("second"))
    forward = [_assessment("first", "DIRECT_SUPPORT"), _assessment("second", "DIRECT_CONTRADICTION")]
    reverse = list(reversed(forward))
    assert derive_policy_decision(protocol, forward) == derive_policy_decision(protocol, reverse)


def test_policy_api_neither_accepts_nor_requires_a_model_proposed_overall_label():
    signature = inspect.signature(derive_policy_decision)
    assert list(signature.parameters) == ["protocol", "assessments"]


def test_duplicate_unknown_assessment_is_reported_as_duplicate_and_unknown():
    protocol = _protocol(_criterion("core"))
    result = derive_policy_decision(
        protocol,
        [
            _assessment("core", "DIRECT_SUPPORT"),
            _assessment("unknown", "DIRECT_SUPPORT"),
            _assessment("unknown", "DIRECT_SUPPORT"),
        ],
    )
    assert result.decision == "MAYBE"
    assert result.safe_fallback
    assert any("duplicate assessment for criterion: unknown" == error for error in result.policy_errors)
    assert any("unknown assessment criterion: unknown" == error for error in result.policy_errors)
