from __future__ import annotations

from litsync_app.screening.local_v2.compiler import (
    COMPILER_VERSION,
    CriterionDraft,
    ProtocolDraft,
    compile_protocol_draft,
)


def _required(**updates):
    payload = {
        "label": "Screening task",
        "role": "REQUIRED_INCLUSION",
        "description": "The study performs the required screening task.",
        "expected_evidence": "A direct statement describing the implemented task.",
    }
    payload.update(updates)
    return payload


def _exclusion(**updates):
    payload = {
        "label": "Future work only",
        "role": "EXCLUSION_TRIGGER",
        "description": "The method is proposed only as future work.",
        "expected_evidence": "A direct statement that implementation is deferred.",
    }
    payload.update(updates)
    return payload


def _draft(**updates):
    payload = {
        "research_question": "Do studies implement the required screening task?",
        "research_context": "Title and abstract screening.",
        "criteria": [_required(), _exclusion()],
        "model": "compiler-test",
    }
    payload.update(updates)
    return payload


def test_valid_draft_compiles_to_identity_bearing_protocol():
    result = compile_protocol_draft(_draft())
    assert result.success is True
    assert result.compiler_version == COMPILER_VERSION
    assert result.protocol is not None
    assert len(result.protocol.protocol_id) == 16
    assert result.issues == []


def test_compilation_is_deterministic():
    first = compile_protocol_draft(_draft())
    second = compile_protocol_draft(_draft())
    assert first == second
    assert first.protocol.protocol_id == second.protocol.protocol_id


def test_compiler_preserves_criterion_order():
    result = compile_protocol_draft(
        _draft(criteria=[_exclusion(label="First exclusion"), _required(label="Required core")])
    )
    assert [item.id for item in result.protocol.criteria] == [
        "first_exclusion",
        "required_core",
    ]


def test_ids_are_generated_from_labels_without_domain_logic():
    result = compile_protocol_draft(
        _draft(criteria=[_required(label="Population / Setting + Task")])
    )
    assert result.protocol.criteria[0].id == "population_setting_task"


def test_explicit_ids_are_normalized_by_protocol_contract():
    result = compile_protocol_draft(
        _draft(criteria=[_required(criterion_id="  Core Relationship  ")])
    )
    assert result.protocol.criteria[0].id == "core_relationship"


def test_unicode_label_generates_a_valid_id():
    result = compile_protocol_draft(
        _draft(criteria=[_required(label="Évaluation clinique")])
    )
    assert result.success is True
    assert result.protocol.criteria[0].id == "évaluation_clinique"


def test_duplicate_generated_ids_fail_safely():
    result = compile_protocol_draft(
        _draft(criteria=[_required(label="Core task"), _required(label="Core task")])
    )
    assert result.success is False
    assert result.protocol is None
    assert [item.code for item in result.issues] == ["DUPLICATE_CRITERION_ID"]


def test_duplicate_explicit_ids_fail_after_normalization():
    result = compile_protocol_draft(
        _draft(
            criteria=[
                _required(label="One", criterion_id="Core Task"),
                _required(label="Two", criterion_id="core_task"),
            ]
        )
    )
    assert result.success is False
    assert result.issues[0].code == "DUPLICATE_CRITERION_ID"


def test_protocol_requires_a_required_inclusion():
    result = compile_protocol_draft(_draft(criteria=[_exclusion()]))
    assert result.success is False
    assert result.protocol is None
    assert result.issues[-1].code == "NO_REQUIRED_INCLUSION"


def test_required_inclusion_defaults_resolution_to_true():
    result = compile_protocol_draft(_draft(criteria=[_required()]))
    assert result.protocol.criteria[0].resolution_required is True
    assert result.warnings == []


def test_required_false_is_forced_true_with_warning():
    result = compile_protocol_draft(
        _draft(criteria=[_required(resolution_required=False)])
    )
    assert result.success is True
    assert result.protocol.criteria[0].resolution_required is True
    assert [item.code for item in result.warnings] == [
        "REQUIRED_RESOLUTION_FORCED"
    ]


def test_exclusion_trigger_defaults_resolution_to_false():
    result = compile_protocol_draft(_draft())
    assert result.protocol.criteria[1].resolution_required is False


def test_exclusion_trigger_can_explicitly_require_resolution():
    result = compile_protocol_draft(
        _draft(criteria=[_required(), _exclusion(resolution_required=True)])
    )
    assert result.protocol.criteria[1].resolution_required is True


def test_blank_expected_evidence_is_canonicalized_to_none():
    result = compile_protocol_draft(
        _draft(criteria=[_required(expected_evidence="   ")])
    )
    assert result.protocol.criteria[0].expected_evidence is None


def test_structured_model_instances_are_accepted():
    draft = ProtocolDraft(
        research_question="Does the study implement the target task?",
        criteria=[CriterionDraft.model_validate(_required())],
    )
    result = compile_protocol_draft(draft)
    assert result.success is True


def test_extra_top_level_fields_fail_without_raising():
    payload = _draft()
    payload["unexpected"] = True
    result = compile_protocol_draft(payload)
    assert result.success is False
    assert result.protocol is None
    assert result.issues[0].code == "INVALID_DRAFT"
    assert "unexpected" in result.issues[0].message


def test_extra_criterion_fields_fail_without_raising():
    criterion = _required()
    criterion["unexpected"] = True
    result = compile_protocol_draft(_draft(criteria=[criterion]))
    assert result.success is False
    assert result.issues[0].criterion_index == 0
    assert result.issues[0].code == "INVALID_DRAFT"


def test_blank_question_fails_without_raising():
    result = compile_protocol_draft(_draft(research_question="   "))
    assert result.success is False
    assert result.protocol is None
    assert result.issues[0].code == "INVALID_DRAFT"


def test_empty_criteria_fail_without_raising():
    result = compile_protocol_draft(_draft(criteria=[]))
    assert result.success is False
    assert result.protocol is None
    assert result.issues[0].code == "INVALID_DRAFT"


def test_invalid_explicit_criterion_id_fails_safely():
    result = compile_protocol_draft(
        _draft(criteria=[_required(criterion_id="bad/id")])
    )
    assert result.success is False
    assert result.protocol is None
    assert result.issues[0].code == "INVALID_CRITERION_ID"
    assert result.issues[0].criterion_index == 0


def test_behavior_change_changes_protocol_identity():
    original = compile_protocol_draft(_draft())
    changed = compile_protocol_draft(
        _draft(criteria=[_required(description="A different requirement is enforced.")])
    )
    assert original.protocol.protocol_id != changed.protocol.protocol_id


def test_arbitrary_domains_compile_identically_by_structure():
    medicine = compile_protocol_draft(
        _draft(criteria=[_required(label="Clinical population")])
    )
    agriculture = compile_protocol_draft(
        _draft(criteria=[_required(label="Crop population")])
    )
    assert medicine.success is True
    assert agriculture.success is True
    assert medicine.issues == agriculture.issues == []
