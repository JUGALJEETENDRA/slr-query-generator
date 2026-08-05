from __future__ import annotations

import json

import pytest

from litsync_app.screening.local_v2.assessor import (
    ASSESSOR_VERSION,
    build_assessment_prompt,
    model_assessment_envelope_schema,
    parse_model_assessment_response,
)
from litsync_app.screening.local_v2.contracts import (
    ProtocolCriterion,
    ScreeningProtocolV2,
)


def _criterion(
    criterion_id: str,
    role: str = "REQUIRED_INCLUSION",
    *,
    resolution_required: bool | None = None,
) -> ProtocolCriterion:
    if resolution_required is None:
        resolution_required = role == "REQUIRED_INCLUSION"
    return ProtocolCriterion(
        id=criterion_id,
        role=role,
        description=f"Resolve {criterion_id} semantically.",
        expected_evidence="Direct title or abstract evidence.",
        resolution_required=resolution_required,
    )


def _protocol() -> ScreeningProtocolV2:
    return ScreeningProtocolV2(
        protocol_id="abc123def4567890",
        research_question="Do local language models screen review citations?",
        research_context="Title and abstract screening only.",
        criteria=[
            _criterion("model_type"),
            _criterion("screening_task"),
            _criterion("protocol_only", "EXCLUSION_TRIGGER"),
        ],
        model="qwen3.5:4b",
    )


def _citation(evidence_id: str, source: str, quote: str) -> dict:
    return {"evidence_id": evidence_id, "source": source, "quote": quote}


def _valid_payload() -> dict:
    return {
        "protocol_id": "abc123def4567890",
        "paper_id": "p-001",
        "assessments": [
            {
                "criterion_id": "model_type",
                "relation": "DIRECT_SUPPORT",
                "rationale": "The abstract names a local language model.",
                "evidence": [
                    _citation(
                        "abstract_001",
                        "abstract",
                        "A local language model screened citations.",
                    )
                ],
            },
            {
                "criterion_id": "screening_task",
                "relation": "DIRECT_SUPPORT",
                "rationale": "The model screened review citations.",
                "evidence": [
                    _citation(
                        "abstract_001",
                        "abstract",
                        "screened citations",
                    )
                ],
            },
            {
                "criterion_id": "protocol_only",
                "relation": "DIRECT_CONTRADICTION",
                "rationale": "The paper reports an implemented experiment.",
                "evidence": [
                    _citation(
                        "abstract_002",
                        "abstract",
                        "We evaluated the implementation on 500 records.",
                    )
                ],
            },
        ],
    }


def test_assessor_version_is_frozen():
    assert ASSESSOR_VERSION == "local-v2-assessor-v2"


def test_protocol_bound_schema_requires_exactly_one_item_per_criterion():
    schema = model_assessment_envelope_schema(_protocol())
    assessments_schema = schema.model_json_schema()["properties"]["assessments"]

    assert assessments_schema["minItems"] == 3
    assert assessments_schema["maxItems"] == 3
    assert schema is model_assessment_envelope_schema(_protocol())


def test_prompt_output_shape_names_every_criterion_in_protocol_order():
    prompt = build_assessment_prompt(
        _protocol(),
        paper_id="p-001",
        title="Title.",
        abstract="Abstract.",
    )
    output_shape = json.loads(prompt.split("REQUIRED_OUTPUT_SHAPE:\n", 1)[1])

    assert [item["criterion_id"] for item in output_shape["assessments"]] == [
        "model_type",
        "screening_task",
        "protocol_only",
    ]
    assert "exactly 3 assessments in protocol order" in prompt


def test_prompt_assigns_semantic_work_to_local_model():
    prompt = build_assessment_prompt(
        _protocol(),
        paper_id="p-001",
        title="Local screening study",
        abstract="A local language model screened citations.",
    )
    assert "primary local semantic screener" in prompt
    assert "Perform the semantic reasoning yourself" in prompt
    assert "do not rely on keyword overlap alone" in prompt


def test_prompt_contains_every_protocol_criterion_in_order():
    prompt = build_assessment_prompt(
        _protocol(), paper_id="p-001", title="Title.", abstract="Abstract."
    )
    positions = [prompt.index(identifier) for identifier in ["model_type", "screening_task", "protocol_only"]]
    assert positions == sorted(positions)


def test_prompt_contains_stable_evidence_unit_ids_and_text():
    prompt = build_assessment_prompt(
        _protocol(),
        paper_id="p-001",
        title="First title sentence. Second title sentence.",
        abstract="First abstract sentence. Second abstract sentence.",
    )
    assert '"evidence_id":"title_001"' in prompt
    assert '"evidence_id":"title_002"' in prompt
    assert '"evidence_id":"abstract_001"' in prompt
    assert "Second abstract sentence." in prompt


def test_prompt_forbids_overall_decision_and_outside_knowledge():
    prompt = build_assessment_prompt(
        _protocol(), paper_id="p-001", title="Title.", abstract="Abstract."
    )
    assert "Do not produce an overall KEEP, MAYBE, or REJECT decision" in prompt
    assert "Do not use outside knowledge" in prompt


def test_prompt_explains_absence_is_not_direct_contradiction():
    prompt = build_assessment_prompt(
        _protocol(), paper_id="p-001", title="Title.", abstract="Abstract."
    )
    assert "absence of information is MISSING_OR_UNCLEAR" in prompt


def test_prompt_handles_missing_abstract_without_inventing_units():
    prompt = build_assessment_prompt(
        _protocol(), paper_id="p-001", title="Only title evidence.", abstract=None
    )
    assert '"evidence_id":"title_001"' in prompt
    assert '"source":"abstract"' not in prompt.split("PAPER_JSON:", 1)[1].split("REQUIRED_OUTPUT_SHAPE:", 1)[0]


def test_prompt_requires_paper_id():
    with pytest.raises(ValueError, match="paper_id is required"):
        build_assessment_prompt(_protocol(), paper_id=" ", title="T", abstract="A")


def test_valid_payload_is_policy_ready_and_protocol_ordered():
    payload = _valid_payload()
    payload["assessments"] = list(reversed(payload["assessments"]))
    result = parse_model_assessment_response(payload, protocol=_protocol(), paper_id="p-001")
    assert result.success is True
    assert result.safe_fallback is False
    assert [item.criterion_id for item in result.assessments] == [
        "model_type",
        "screening_task",
        "protocol_only",
    ]
    assert result.parsed_assessments == result.assessments
    assert result.issues == []


def test_json_string_is_accepted():
    result = parse_model_assessment_response(
        json.dumps(_valid_payload()), protocol=_protocol(), paper_id="p-001"
    )
    assert result.success is True


def test_fenced_json_is_accepted_without_semantic_repair():
    raw = "```json\n" + json.dumps(_valid_payload()) + "\n```"
    result = parse_model_assessment_response(raw, protocol=_protocol(), paper_id="p-001")
    assert result.success is True


def test_leading_commentary_before_json_is_accepted():
    raw = "Here is the requested object:\n" + json.dumps(_valid_payload())
    result = parse_model_assessment_response(raw, protocol=_protocol(), paper_id="p-001")
    assert result.success is True


def test_empty_response_returns_safe_failure():
    result = parse_model_assessment_response("  ", protocol=_protocol(), paper_id="p-001")
    assert result.success is False
    assert result.safe_fallback is True
    assert result.assessments == []
    assert [issue.code for issue in result.issues] == ["EMPTY_RESPONSE"]


def test_non_json_response_returns_safe_failure():
    result = parse_model_assessment_response("not json", protocol=_protocol(), paper_id="p-001")
    assert result.success is False
    assert result.assessments == []
    assert [issue.code for issue in result.issues] == ["NO_JSON_OBJECT"]


def test_protocol_id_mismatch_never_returns_policy_ready_assessments():
    payload = _valid_payload()
    payload["protocol_id"] = "wrong"
    result = parse_model_assessment_response(payload, protocol=_protocol(), paper_id="p-001")
    assert result.success is False
    assert result.assessments == []
    assert len(result.parsed_assessments) == 3
    assert "PROTOCOL_ID_MISMATCH" in [issue.code for issue in result.issues]


def test_paper_id_mismatch_never_returns_policy_ready_assessments():
    payload = _valid_payload()
    payload["paper_id"] = "wrong"
    result = parse_model_assessment_response(payload, protocol=_protocol(), paper_id="p-001")
    assert result.success is False
    assert result.assessments == []
    assert "PAPER_ID_MISMATCH" in [issue.code for issue in result.issues]


def test_unexpected_overall_decision_field_is_rejected():
    payload = _valid_payload()
    payload["decision"] = "REJECT"
    result = parse_model_assessment_response(payload, protocol=_protocol(), paper_id="p-001")
    assert result.success is False
    assert result.assessments == []
    assert "INVALID_ENVELOPE" in [issue.code for issue in result.issues]


def test_assessments_must_be_a_list():
    payload = _valid_payload()
    payload["assessments"] = {"criterion_id": "model_type"}
    result = parse_model_assessment_response(payload, protocol=_protocol(), paper_id="p-001")
    assert result.success is False
    assert [issue.code for issue in result.issues] == ["ASSESSMENTS_NOT_LIST"]


def test_missing_criterion_is_safe_failure():
    payload = _valid_payload()
    payload["assessments"] = payload["assessments"][:-1]
    result = parse_model_assessment_response(payload, protocol=_protocol(), paper_id="p-001")
    assert result.success is False
    assert result.assessments == []
    assert "MISSING_CRITERION" in [issue.code for issue in result.issues]


def test_duplicate_criterion_is_safe_failure():
    payload = _valid_payload()
    payload["assessments"].append(dict(payload["assessments"][0]))
    result = parse_model_assessment_response(payload, protocol=_protocol(), paper_id="p-001")
    codes = [issue.code for issue in result.issues]
    assert result.success is False
    assert result.assessments == []
    assert "DUPLICATE_CRITERION" in codes


def test_unknown_criterion_is_safe_failure():
    payload = _valid_payload()
    payload["assessments"].append(
        {
            "criterion_id": "invented_rule",
            "relation": "MISSING_OR_UNCLEAR",
            "rationale": "Invented by the model.",
            "evidence": [],
        }
    )
    result = parse_model_assessment_response(payload, protocol=_protocol(), paper_id="p-001")
    assert result.success is False
    assert result.assessments == []
    assert "UNKNOWN_CRITERION" in [issue.code for issue in result.issues]


def test_invalid_relation_is_reported_and_causes_missing_criterion():
    payload = _valid_payload()
    payload["assessments"][0]["relation"] = "PROBABLY"
    result = parse_model_assessment_response(payload, protocol=_protocol(), paper_id="p-001")
    codes = [issue.code for issue in result.issues]
    assert result.success is False
    assert result.assessments == []
    assert "INVALID_ASSESSMENT" in codes
    assert "MISSING_CRITERION" in codes


def test_decisive_relation_without_evidence_is_never_repaired():
    payload = _valid_payload()
    payload["assessments"][0]["evidence"] = []
    result = parse_model_assessment_response(payload, protocol=_protocol(), paper_id="p-001")
    assert result.success is False
    assert result.assessments == []
    assert len(result.parsed_assessments) == 2
    assert "INVALID_ASSESSMENT" in [issue.code for issue in result.issues]


def test_invalid_citation_shape_is_not_silently_fixed():
    payload = _valid_payload()
    payload["assessments"][0]["evidence"][0]["unexpected"] = True
    result = parse_model_assessment_response(payload, protocol=_protocol(), paper_id="p-001")
    assert result.success is False
    assert result.assessments == []
    assert "INVALID_ASSESSMENT" in [issue.code for issue in result.issues]


def test_more_than_two_citations_is_invalid():
    payload = _valid_payload()
    payload["assessments"][0]["evidence"] = [
        _citation("abstract_001", "abstract", "one"),
        _citation("abstract_002", "abstract", "two"),
        _citation("abstract_003", "abstract", "three"),
    ]
    result = parse_model_assessment_response(payload, protocol=_protocol(), paper_id="p-001")
    assert result.success is False
    assert "INVALID_ASSESSMENT" in [issue.code for issue in result.issues]


def test_missing_or_unclear_without_evidence_is_valid():
    payload = _valid_payload()
    payload["assessments"][0] = {
        "criterion_id": "model_type",
        "relation": "MISSING_OR_UNCLEAR",
        "rationale": "The abstract does not identify the model type.",
        "evidence": [],
    }
    result = parse_model_assessment_response(payload, protocol=_protocol(), paper_id="p-001")
    assert result.success is True
    assert result.assessments[0].relation == "MISSING_OR_UNCLEAR"


def test_parser_requires_requested_paper_id():
    with pytest.raises(ValueError, match="paper_id is required"):
        parse_model_assessment_response(_valid_payload(), protocol=_protocol(), paper_id="")
