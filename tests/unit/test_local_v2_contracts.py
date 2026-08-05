from __future__ import annotations

import pytest
from pydantic import ValidationError

from litsync_app.screening.local_v2.contracts import (
    CriterionAssessment,
    EvidenceCitation,
    ProtocolCriterion,
    ScreeningProtocolV2,
)


def _evidence(evidence_id: str = "abstract_001") -> EvidenceCitation:
    return EvidenceCitation(
        evidence_id=evidence_id,
        source="abstract",
        quote="The model directly screened citation abstracts.",
    )


def _required(
    criterion_id: str = "core_relationship",
    *,
    description: str = "The required relationship is present.",
) -> ProtocolCriterion:
    return ProtocolCriterion(
        id=criterion_id,
        role="REQUIRED_INCLUSION",
        description=description,
        expected_evidence="Direct supporting evidence.",
        resolution_required=True,
    )


def _protocol(**updates) -> ScreeningProtocolV2:
    payload = {
        "research_question": "Does the study satisfy the review relationship?",
        "criteria": [_required()],
        "model": "test-model",
    }
    payload.update(updates)
    return ScreeningProtocolV2.model_validate(payload)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            EvidenceCitation,
            {
                "evidence_id": "abstract_001",
                "source": "abstract",
                "quote": "Direct evidence.",
                "unexpected": True,
            },
        ),
        (
            ProtocolCriterion,
            {
                "id": "criterion",
                "role": "REQUIRED_INCLUSION",
                "description": "A valid criterion.",
                "resolution_required": True,
                "unexpected": True,
            },
        ),
    ],
)
def test_strict_models_reject_extra_fields(model, payload):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate(payload)


def test_criterion_ids_are_normalized_consistently():
    criterion = _required("  Core Relationship  ")
    assessment = CriterionAssessment(
        criterion_id="  Core Relationship  ",
        relation="MISSING_OR_UNCLEAR",
        rationale="The record does not resolve the relationship.",
    )
    assert criterion.id == "core_relationship"
    assert assessment.criterion_id == "core_relationship"


@pytest.mark.parametrize("criterion_id", ["bad/id", "bad.id", "bad:id", "bad id!"])
def test_invalid_criterion_ids_are_rejected(criterion_id):
    with pytest.raises(ValidationError, match="letters, numbers, underscores, or hyphens"):
        _required(criterion_id)


def test_duplicate_protocol_criterion_ids_are_rejected_after_normalization():
    with pytest.raises(ValidationError, match="criterion ids must be unique"):
        _protocol(criteria=[_required("Core Relationship"), _required("core_relationship")])


def test_protocol_requires_at_least_one_required_inclusion():
    exclusion = ProtocolCriterion(
        id="editorial",
        role="EXCLUSION_TRIGGER",
        description="The record is an editorial.",
        resolution_required=False,
    )
    with pytest.raises(ValidationError, match="at least one REQUIRED_INCLUSION"):
        _protocol(criteria=[exclusion])


def test_required_inclusion_must_require_resolution():
    with pytest.raises(ValidationError, match="resolution_required=True"):
        _protocol(
            criteria=[
                ProtocolCriterion(
                    id="core",
                    role="REQUIRED_INCLUSION",
                    description="The core relationship is present.",
                    resolution_required=False,
                )
            ]
        )


def test_protocol_identity_is_deterministic_and_16_characters():
    first = _protocol().with_identity()
    second = _protocol().with_identity()
    assert first.protocol_id == second.protocol_id
    assert len(first.protocol_id) == 16
    assert first.protocol_id


def test_protocol_identity_ignores_existing_identity_but_changes_with_behavior():
    original = _protocol().with_identity()
    copied = original.model_copy(update={"protocol_id": "manually-overwritten"}).with_identity()
    changed = _protocol(
        criteria=[_required(description="A materially different relationship is required.")]
    ).with_identity()
    assert copied.protocol_id == original.protocol_id
    assert changed.protocol_id != original.protocol_id


@pytest.mark.parametrize("relation", ["DIRECT_SUPPORT", "DIRECT_CONTRADICTION"])
def test_decisive_relations_require_evidence(relation):
    with pytest.raises(ValidationError, match="requires at least one evidence citation"):
        CriterionAssessment(
            criterion_id="core",
            relation=relation,
            rationale="A decisive relationship was asserted.",
        )


def test_duplicate_evidence_ids_are_rejected():
    with pytest.raises(ValidationError, match="evidence ids must be unique"):
        CriterionAssessment(
            criterion_id="core",
            relation="DIRECT_SUPPORT",
            rationale="The relationship is directly supported.",
            evidence=[_evidence(), _evidence()],
        )


def test_not_applicable_rejects_evidence():
    with pytest.raises(ValidationError, match="NOT_APPLICABLE cannot contain evidence"):
        CriterionAssessment(
            criterion_id="core",
            relation="NOT_APPLICABLE",
            rationale="The criterion does not apply.",
            evidence=[_evidence()],
        )


def test_valid_missing_or_unclear_without_evidence_is_accepted():
    assessment = CriterionAssessment(
        criterion_id="core",
        relation="MISSING_OR_UNCLEAR",
        rationale="The abstract does not resolve the required relationship.",
    )
    assert assessment.evidence == []
