import json
import re
from pathlib import Path

import pandas as pd
import pytest

import litsync_app.integrations.gemini_web_v24_screening as v24_screening
from litsync_app.screening.bulk import ScreeningProgress, ScreeningSession
from litsync_app.integrations.gemini_web_v24_automation import GeminiWebV24Config
from litsync_app.integrations.gemini_web_v24_prompt import (
    V24Paper,
    assessment_protocol_projection,
    authoritative_criterion_entries,
    build_primary_prompt,
    build_protocol_prompt,
    build_verification_prompt,
)
from litsync_app.integrations.gemini_web_v24_screening import (
    GEMINI_WEB_V24_CACHE_VERSION,
    GEMINI_WEB_V24_ASSESSMENT_PROMPT_VERSION,
    GEMINI_WEB_V24_PROTOCOL_VERSION,
    GEMINI_WEB_V24_VERSION,
    V24_STRUCTURED_OUTPUT_FAILURE,
    V24_TRANSPORT_FAILURE,
    V24Assessment,
    V24CompactAssessmentBatch,
    V24Diagnostics,
    V24Protocol,
    _compile_protocol,
    _execute_batch_with_degraded_retry,
    _resume_rows,
    _validate_and_decide,
    _validate_protocol_sources,
    _verification_route,
    plan_assessment_batch_size,
    screen_csv_with_gemini_web_v24,
)
from litsync_app.screening.engines import GEMINI_WEB_V24_ENGINE, normalize_processing_engine


def _protocol_payload():
    return {
        "protocol_version": GEMINI_WEB_V24_PROTOCOL_VERSION,
        "protocol_id": "",
        "research_question": "How does a requested approach address a requested task?",
        "objective": "Identify studies that substantively apply the requested approach to the requested task.",
        "population_or_subject": ["The subject specified by the researcher"],
        "methods_or_interventions": ["The approach specified by the researcher"],
        "target_tasks_or_outcomes": ["The task specified by the researcher"],
        "application_context": [],
        "required_inclusion_criteria": [{
            "id": "required_relationship",
            "kind": "inclusion",
            "description": "The paper substantively studies the requested relationship.",
            "required": True,
            "expected_evidence": "Title or abstract evidence of the objective, method, analysis, or findings.",
            "source": "research_question",
            "authoritative_text": "",
            "is_composite_relationship": True,
        }],
        "exclusion_boundaries": [],
        "ambiguities": ["An incidental mention does not establish the relationship."],
        "synonyms_and_equivalent_concepts": [],
        "near_neighbor_but_out_of_scope_concepts": [],
    }


def _protocol():
    return V24Protocol.model_validate(_protocol_payload()).with_identity()


def _protocol_with_criterion_count(count: int) -> V24Protocol:
    payload = _protocol_payload()
    payload["required_inclusion_criteria"] = [
        {
            **payload["required_inclusion_criteria"][0],
            "id": f"criterion_{index}",
            "is_composite_relationship": index == 0,
        }
        for index in range(min(count, 20))
    ]
    payload["exclusion_boundaries"] = [{
        **payload["required_inclusion_criteria"][0],
        "id": f"criterion_{index}",
        "kind": "exclusion",
        "is_composite_relationship": False,
    } for index in range(20, count)]
    return V24Protocol.model_validate(payload).with_identity()


def _assessment(
    paper_id="0",
    *,
    verdict="MET",
    support="SUBSTANTIVE",
    relationship="SUPPORTS",
    decision="KEEP",
    risk="LOW",
    evidence=True,
):
    return V24Assessment.model_validate({
        "paper_id": paper_id,
        "decision": decision,
        "confidence": 0.93,
        "decision_risk": risk,
        "reason": "The supplied title directly establishes the requested relationship.",
        "criterion_assessments": [{
            "criterion_id": "required_relationship",
            "verdict": verdict,
            "scope_support": support,
            "evidence_relationship": relationship,
            "rationale": "The title supplies the decisive relationship evidence.",
            "evidence": (
                [{"source": "title", "evidence_id": "title_001"}] if evidence else []
            ),
        }],
    })


def _compact_assessment(value: V24Assessment) -> dict:
    return {
        "p": value.paper_id,
        "d": value.decision,
        "f": value.confidence,
        "k": value.decision_risk,
        "r": value.reason,
        "c": [{
            "c": criterion.criterion_id,
            "v": criterion.verdict,
            "u": criterion.scope_support,
            "l": criterion.evidence_relationship,
            "r": criterion.rationale,
            "e": [
                {"s": reference.source, "e": reference.evidence_id}
                for reference in criterion.evidence
            ],
        } for criterion in value.criterion_assessments],
    }


@pytest.mark.parametrize(
    ("criterion_count", "expected_size", "over_budget"),
    [
        (1, 5, False),
        (3, 5, False),
        (4, 4, False),
        (6, 3, False),
        (10, 2, False),
        (13, 1, False),
        (26, 1, True),
    ],
)
def test_phase3a_adaptive_batch_boundaries(
    criterion_count, expected_size, over_budget,
):
    protocol = _protocol_with_criterion_count(criterion_count)
    assert plan_assessment_batch_size(
        protocol, stage="primary"
    ) == (expected_size, over_budget)
    assert plan_assessment_batch_size(
        protocol, stage="verification"
    ) == (expected_size, over_budget)


def test_phase3a_protocol_projection_preserves_all_ten_criteria():
    protocol = _protocol_with_criterion_count(10)
    projection = assessment_protocol_projection(protocol.model_dump(mode="json"))
    assert len(projection["criteria"]) == 10
    assert set(projection["criteria"][0]) == {
        "id", "kind", "description", "required", "expected_evidence",
        "source", "authoritative_text", "is_composite_relationship",
    }
    assert [item["id"] for item in projection["criteria"]] == [
        criterion.id for criterion in protocol.criteria
    ]
    assert "objective" not in projection
    assert "synonyms_and_equivalent_concepts" not in projection


def test_phase3a_compact_contract_is_strict_and_expandable():
    compact = _compact_assessment(_assessment())
    parsed = V24CompactAssessmentBatch.model_validate({"items": [compact]})
    assert parsed.items[0].expand() == _assessment()
    with pytest.raises(ValueError):
        V24CompactAssessmentBatch.model_validate({
            "items": [{**compact, "d": "INCLUDE"}],
        })
    with pytest.raises(ValueError):
        V24CompactAssessmentBatch.model_validate({
            "items": [{**compact, "unexpected": True}],
        })


def test_domain_neutral_deterministic_keep_reject_and_maybe():
    protocol = _protocol()
    paper = V24Paper("0", "A substantive requested relationship study", "The method is evaluated.")

    keep = _validate_and_decide(_assessment(), protocol, paper)
    assert keep["decision"] == "KEEP"
    assert keep["validation_status"] == "validated"

    reject = _validate_and_decide(_assessment(
        verdict="NOT_MET", relationship="CONFLICTS", decision="REJECT",
    ), protocol, paper)
    assert reject["decision"] == "REJECT"
    assert reject["validation_status"] == "validated"

    unclear = _validate_and_decide(_assessment(
        verdict="UNCLEAR", support="INSUFFICIENT", relationship="INSUFFICIENT",
        decision="MAYBE", risk="BORDERLINE", evidence=False,
    ), protocol, paper)
    assert unclear["decision"] == "MAYBE"
    assert unclear["validation_status"] == "validated"


def test_valid_insufficient_maybe_does_not_trigger_verification():
    protocol = _protocol()
    result = _validate_and_decide(
        _assessment(
            verdict="UNCLEAR",
            support="INSUFFICIENT",
            relationship="INSUFFICIENT",
            decision="MAYBE",
            risk="LOW",
            evidence=False,
        ),
        protocol,
        V24Paper(
            "0",
            "Broad related study",
            "The abstract does not establish the required relationship.",
        ),
    )

    assert result["decision"] == "MAYBE"
    assert result["decision_risk"] == "BORDERLINE"
    assert result["validation_status"] == "validated"
    assert _verification_route(result, protocol) == ""


def test_high_risk_substantive_unresolved_maybe_is_verified():
    protocol = _protocol()
    result = _validate_and_decide(
        _assessment(
            verdict="UNCLEAR",
            support="SUBSTANTIVE",
            relationship="INSUFFICIENT",
            decision="MAYBE",
            risk="HIGH",
            evidence=True,
        ),
        protocol,
        V24Paper(
            "0",
            "Substantive related study",
            "The paper evaluates part of the requested relationship.",
        ),
    )

    assert result["decision"] == "MAYBE"
    assert result["decision_risk"] == "HIGH"
    assert result["validation_status"] == "validated"
    assert _verification_route(result, protocol) == "borderline_primary"


def test_unmentioned_requirement_cannot_be_invented_as_not_met():
    result = _validate_and_decide(
        _assessment(
            verdict="NOT_MET", relationship="CONFLICTS", decision="REJECT", evidence=False,
        ),
        _protocol(),
        V24Paper("0", "A broad study", "The required relationship is not discussed."),
    )
    assert result["decision"] == "MAYBE"
    assert result["validation_status"] == "unresolved"
    assert any("affirmative conflicting evidence" in error for error in result["validation_errors"])


def test_incidental_or_hallucinated_evidence_cannot_support_keep():
    protocol = _protocol()
    paper = V24Paper("0", "A broad study", "The requested topic appears only as background.")
    incidental = _validate_and_decide(_assessment(
        support="INCIDENTAL", relationship="INCIDENTAL", decision="KEEP",
    ), protocol, paper)
    assert incidental["decision"] == "MAYBE"
    assert incidental["validation_status"] == "unresolved"

    payload = _assessment().model_dump(mode="json")
    payload["criterion_assessments"][0]["evidence"][0]["evidence_id"] = "abstract_999"
    hallucinated = _validate_and_decide(V24Assessment.model_validate(payload), protocol, paper)
    assert hallucinated["decision"] == "MAYBE"
    assert any("invalid evidence reference" in error for error in hallucinated["validation_errors"])


def test_inferred_exclusion_uncertainty_does_not_become_hidden_rejection():
    payload = _protocol_payload()
    payload["exclusion_boundaries"] = [{
        "id": "inferred_boundary",
        "kind": "exclusion",
        "description": "A logically incompatible focus.",
        "required": True,
        "expected_evidence": "Evidence of the incompatible focus.",
        "source": "research_question",
    }]
    protocol = V24Protocol.model_validate(payload).with_identity()
    assessment = _assessment().model_dump(mode="json")
    assessment["criterion_assessments"].append({
        "criterion_id": "inferred_boundary",
        "verdict": "UNCLEAR",
        "scope_support": "INSUFFICIENT",
        "evidence_relationship": "INSUFFICIENT",
        "rationale": "The inferred boundary is not indicated.",
        "evidence": [],
    })
    result = _validate_and_decide(
        V24Assessment.model_validate(assessment),
        protocol,
        V24Paper("0", "A substantive study", "The requested relationship is evaluated."),
    )
    assert result["decision"] == "KEEP"
    assert result["validation_status"] == "validated"
    assert _verification_route(result, protocol) == ""

    payload["exclusion_boundaries"][0]["source"] = "user"
    user_protocol = V24Protocol.model_validate(payload).with_identity()
    user_result = _validate_and_decide(
        V24Assessment.model_validate(assessment),
        user_protocol,
        V24Paper("0", "A substantive study", "The requested relationship is evaluated."),
    )
    assert user_result["decision"] == "MAYBE"


def test_unknown_or_missing_criterion_id_is_unresolved():
    payload = _assessment().model_dump(mode="json")
    payload["criterion_assessments"][0]["criterion_id"] = "invented_criterion"
    result = _validate_and_decide(
        V24Assessment.model_validate(payload),
        _protocol(),
        V24Paper("0", "A study", "An abstract."),
    )
    assert result["decision"] == "MAYBE"
    assert result["validation_status"] == "unresolved"


@pytest.mark.parametrize(
    ("title", "abstract"),
    [
        (
            "Clinical decision support evaluation",
            "The requested relationship is the objective and is evaluated on clinical records.",
        ),
        (
            "Automated defect analysis",
            "The requested relationship is implemented and evaluated in a software workflow.",
        ),
        (
            "Adaptive learning intervention",
            "The requested relationship is tested with learners in a controlled study.",
        ),
        (
            "Operational planning system",
            "The requested relationship is evaluated in an organizational process.",
        ),
    ],
)
def test_policy_is_identical_across_unrelated_domains(title, abstract):
    result = _validate_and_decide(
        _assessment(),
        _protocol(),
        V24Paper("0", title, abstract),
    )
    assert result["decision"] == "KEEP"
    assert result["validation_status"] == "validated"


def test_verifier_is_prediction_blind():
    prompt = build_verification_prompt(
        protocol=_protocol().model_dump(mode="json"),
        papers=[V24Paper("0", "Distinct title", "Distinct abstract")],
        flags={"0": {
            "validation_errors": ["criterion tension"],
            "unresolved_criterion_ids": ["required_relationship"],
        }},
        schema={"type": "object"},
    )
    assert '"decision": "KEEP"' not in prompt
    assert "DISTINCTIVE_PRIMARY_RATIONALE" not in prompt
    assert "criterion tension" in prompt


def test_prompt_defines_exclusion_polarity_without_domain_rules():
    prompt = build_primary_prompt(
        protocol=_protocol().model_dump(mode="json"),
        papers=[V24Paper("0", "Title", "Abstract")],
        schema={"type": "object"},
    )
    assert "exclusion MET means the" in prompt
    assert "disqualifying condition itself is present" in prompt
    assert "Never mark an exclusion MET because the paper avoids" in prompt


def _protocol_with_user_exclusion(authoritative_text):
    payload = _protocol_payload()
    payload["exclusion_boundaries"] = [{
        "id": "user_study_role_exclusion",
        "kind": "exclusion",
        "description": "The source paper has the explicitly excluded study role.",
        "required": True,
        "expected_evidence": "The source paper affirmatively identifies its study role.",
        "source": "user",
        "authoritative_text": authoritative_text,
        "is_composite_relationship": False,
    }]
    return V24Protocol.model_validate(payload).with_identity()


def _assessment_with_user_exclusion(*, exclusion_verdict, decision):
    payload = _assessment(decision=decision).model_dump(mode="json")
    payload["criterion_assessments"].append({
        "criterion_id": "user_study_role_exclusion",
        "verdict": exclusion_verdict,
        "scope_support": (
            "SUBSTANTIVE" if exclusion_verdict == "MET" else "INSUFFICIENT"
        ),
        "evidence_relationship": (
            "SUPPORTS" if exclusion_verdict == "MET" else "INSUFFICIENT"
        ),
        "rationale": "The source paper's own study role is directly stated.",
        "evidence": (
            [{"source": "title", "evidence_id": "title_001"}]
            if exclusion_verdict == "MET"
            else []
        ),
    })
    return V24Assessment.model_validate(payload)


def test_phase2c_compilation_preserves_every_authoritative_user_criterion():
    inclusions = [
        "Must measure learner retention",
        "Must compare two lesson plans",
    ]
    exclusions = [
        "Exclude secondary syntheses",
        "Exclude opinion pieces",
    ]
    payload = _protocol_payload()
    payload["required_inclusion_criteria"].extend([
        {
            "id": f"user_inclusion_{index}",
            "kind": "inclusion",
            "description": text,
            "required": True,
            "expected_evidence": "Affirmative evidence satisfying the user requirement.",
            "source": "user",
            "authoritative_text": text,
            "is_composite_relationship": False,
        }
        for index, text in enumerate(inclusions)
    ])
    payload["exclusion_boundaries"] = [
        {
            "id": f"user_exclusion_{index}",
            "kind": "exclusion",
            "description": "The source has the excluded study characteristic.",
            "required": True,
            "expected_evidence": "Affirmative evidence of the excluded characteristic.",
            "source": "user",
            "authoritative_text": text,
            "is_composite_relationship": False,
        }
        for index, text in enumerate(exclusions)
    ]

    class ProtocolBrowser:
        def set_attempt_context(self, **kwargs):
            return None

        def submit_prompt_and_get_response(self, prompt):
            return json.dumps(payload)

        def recover_transport_failure(self):
            return None

    protocol = _compile_protocol(
        ProtocolBrowser(),
        question="How does an adaptive lesson affect learner retention?",
        context="",
        inclusion=(
            "1. Must measure learner retention;\n"
            "- Must compare two lesson plans"
        ),
        exclusion=(
            "* Exclude secondary syntheses;\n"
            "2) Exclude opinion pieces"
        ),
    )
    assert protocol.protocol_version == "gemini-web-v2.4-protocol-v3"
    assert GEMINI_WEB_V24_ASSESSMENT_PROMPT_VERSION == (
        "gemini-web-v2.4-assessment-prompt-v5"
    )
    assert GEMINI_WEB_V24_CACHE_VERSION == "gemini-web-v2.4-assessment-v5"
    assert [
        criterion.authoritative_text
        for criterion in protocol.required_inclusion_criteria
        if criterion.source == "user"
    ] == inclusions
    assert [
        criterion.authoritative_text
        for criterion in protocol.exclusion_boundaries
        if criterion.source == "user"
    ] == exclusions


def test_phase2c_protocol_validation_rejects_omitted_explicit_exclusion():
    with pytest.raises(ValueError, match="authoritative user criteria"):
        _validate_protocol_sources(
            _protocol(),
            "",
            "Exclude narrative syntheses",
        )


def test_phase2c_protocol_validation_rejects_changed_user_polarity():
    payload = _protocol_payload()
    payload["exclusion_boundaries"] = [{
        "id": "wrong_polarity",
        "kind": "exclusion",
        "description": "The requested measurement is present.",
        "required": True,
        "expected_evidence": "Evidence of the measurement.",
        "source": "user",
        "authoritative_text": "Must measure habitat recovery",
        "is_composite_relationship": False,
    }]
    with pytest.raises(ValueError, match="polarity"):
        _validate_protocol_sources(
            V24Protocol.model_validate(payload).with_identity(),
            "Must measure habitat recovery",
            "",
        )


def test_phase2c_protocol_validation_rejects_merged_or_weakened_user_criteria():
    payload = _protocol_payload()
    payload["required_inclusion_criteria"].append({
        "id": "merged_requirement",
        "kind": "inclusion",
        "description": "The paper broadly discusses both requirements.",
        "required": True,
        "expected_evidence": "Evidence related to either requirement.",
        "source": "user",
        "authoritative_text": "Must discuss attendance and satisfaction",
        "is_composite_relationship": False,
    })
    with pytest.raises(ValueError, match="merged, weakened"):
        _validate_protocol_sources(
            V24Protocol.model_validate(payload).with_identity(),
            "Must measure attendance; Must measure satisfaction",
            "",
        )


def test_phase2c_composite_relationship_remains_alongside_user_criteria():
    payload = _protocol_payload()
    payload["required_inclusion_criteria"].append({
        "id": "user_measurement",
        "kind": "inclusion",
        "description": "The paper reports the requested measurement.",
        "required": True,
        "expected_evidence": "A reported measurement.",
        "source": "user",
        "authoritative_text": "Must report a measured outcome",
        "is_composite_relationship": False,
    })
    protocol = V24Protocol.model_validate(payload).with_identity()
    _validate_protocol_sources(protocol, "Must report a measured outcome", "")
    assert any(
        criterion.source == "research_question"
        and criterion.is_composite_relationship
        for criterion in protocol.required_inclusion_criteria
    )
    assert any(
        criterion.source == "user"
        and not criterion.is_composite_relationship
        for criterion in protocol.required_inclusion_criteria
    )


def test_phase2c_protocol_validation_rejects_missing_composite_relationship():
    payload = _protocol_payload()
    payload["required_inclusion_criteria"][0]["is_composite_relationship"] = False
    with pytest.raises(ValueError, match="composite research relationship"):
        _validate_protocol_sources(
            V24Protocol.model_validate(payload).with_identity(),
            "",
            "",
        )


def test_phase2c_disconnected_method_context_and_outcome_remain_unclear():
    paper = V24Paper(
        "synthetic-education",
        "Adaptive lesson planning",
        (
            "A forecasting method is introduced. Rural classrooms are described. "
            "Attendance is discussed as background."
        ),
    )
    prompt = build_primary_prompt(
        protocol=_protocol().model_dump(mode="json"),
        papers=[paper],
        schema={"type": "object"},
    )
    assert "Separate mentions across unrelated evidence units" in prompt
    result = _validate_and_decide(
        _assessment(
            verdict="UNCLEAR",
            support="INSUFFICIENT",
            relationship="INSUFFICIENT",
            decision="MAYBE",
            risk="HIGH",
            evidence=False,
        ),
        _protocol(),
        paper,
    )
    assert result["decision"] == "MAYBE"


def test_phase2c_requested_method_in_broad_list_remains_incidental():
    paper = V24Paper(
        "synthetic-ecology",
        "Digital methods for wetland monitoring",
        "The overview lists sensors, statistical tools, and the requested method.",
    )
    prompt = build_primary_prompt(
        protocol=_protocol().model_dump(mode="json"),
        papers=[paper],
        schema={"type": "object"},
    )
    assert "requested method appearing only in a broad list is INCIDENTAL" in prompt
    incidental = _validate_and_decide(
        _assessment(
            verdict="UNCLEAR",
            support="INCIDENTAL",
            relationship="INCIDENTAL",
            decision="MAYBE",
            risk="HIGH",
        ),
        _protocol(),
        paper,
    )
    assert incidental["decision"] == "MAYBE"


def test_phase2c_case_setting_cannot_transfer_an_unrelated_outcome():
    paper = V24Paper(
        "synthetic-software",
        "Compiler construction framework",
        (
            "The framework improves software build speed. "
            "A municipal service is used only as a demonstration setting."
        ),
    )
    prompt = build_primary_prompt(
        protocol=_protocol().model_dump(mode="json"),
        papers=[paper],
        schema={"type": "object"},
    )
    assert "case-study setting cannot transfer an outcome" in prompt


def test_phase2c_review_cannot_inherit_primary_status_from_summarized_studies():
    protocol = _protocol_with_user_exclusion("Exclude secondary syntheses")
    paper = V24Paper(
        "synthetic-review",
        "Review of adaptive assessment experiments",
        "This review summarizes controlled experiments reported by prior studies.",
    )
    prompt = build_primary_prompt(
        protocol=protocol.model_dump(mode="json"),
        papers=[paper],
        schema={"type": "object"},
    )
    assert "what the source paper itself does" in prompt
    result = _validate_and_decide(
        _assessment_with_user_exclusion(
            exclusion_verdict="MET",
            decision="REJECT",
        ),
        protocol,
        paper,
    )
    assert result["decision"] == "REJECT"


def test_phase2c_editorial_describing_experiments_remains_editorial():
    protocol = _protocol_with_user_exclusion("Exclude editorials")
    paper = V24Paper(
        "synthetic-editorial",
        "Editorial on public-service experiments",
        "This editorial describes measurements produced by studies in the collection.",
    )
    prompt = build_primary_prompt(
        protocol=protocol.model_dump(mode="json"),
        papers=[paper],
        schema={"type": "object"},
    )
    assert "does not become a primary study by describing eligible work elsewhere" in prompt
    result = _validate_and_decide(
        _assessment_with_user_exclusion(
            exclusion_verdict="MET",
            decision="REJECT",
        ),
        protocol,
        paper,
    )
    assert result["decision"] == "REJECT"


def test_phase2c_original_analytical_model_with_results_can_be_primary():
    paper = V24Paper(
        "synthetic-policy",
        "Analytical comparison of permit-allocation policies",
        "The study develops competing models and reports comparative welfare results.",
    )
    prompt = build_primary_prompt(
        protocol=_protocol().model_dump(mode="json"),
        papers=[paper],
        schema={"type": "object"},
    )
    assert "Original analytical" in prompt
    assert _validate_and_decide(
        _assessment(),
        _protocol(),
        paper,
    )["decision"] == "KEEP"


def test_phase2c_game_theoretic_and_simulation_results_can_be_primary():
    paper = V24Paper(
        "synthetic-manufacturing",
        "Strategic production scheduling analysis",
        (
            "An original game-theoretic model is compared through simulation, "
            "with reported cost and throughput results."
        ),
    )
    prompt = build_primary_prompt(
        protocol=_protocol().model_dump(mode="json"),
        papers=[paper],
        schema={"type": "object"},
    )
    assert "game-theoretic, simulation" in prompt
    assert _validate_and_decide(
        _assessment(),
        _protocol(),
        paper,
    )["decision"] == "KEEP"


def test_phase2c_unevaluated_proposed_framework_remains_insufficient():
    paper = V24Paper(
        "synthetic-framework",
        "Proposed civic-participation framework",
        "The framework is proposed with desired accuracy and efficiency properties.",
    )
    prompt = build_primary_prompt(
        protocol=_protocol().model_dump(mode="json"),
        papers=[paper],
        schema={"type": "object"},
    )
    assert "Merely proposing a model, framework, or desired performance property" in prompt
    result = _validate_and_decide(
        _assessment(
            verdict="UNCLEAR",
            support="INSUFFICIENT",
            relationship="INSUFFICIENT",
            decision="MAYBE",
            risk="HIGH",
            evidence=False,
        ),
        _protocol(),
        paper,
    )
    assert result["decision"] == "MAYBE"


def test_phase2c_measured_mediator_or_causal_pathway_supports_outcome():
    paper = V24Paper(
        "synthetic-mediator",
        "Observed pathways in teacher-support programs",
        (
            "The analysis estimates instructional confidence as a mediator and "
            "reports its measured pathway to learner persistence."
        ),
    )
    prompt = build_primary_prompt(
        protocol=_protocol().model_dump(mode="json"),
        papers=[paper],
        schema={"type": "object"},
    )
    assert "mediator" in prompt
    assert "causal pathway" in prompt
    assert _validate_and_decide(
        _assessment(),
        _protocol(),
        paper,
    )["decision"] == "KEEP"


def test_phase2c_explicit_exclusion_precedes_satisfied_inclusions():
    protocol = _protocol_with_user_exclusion("Exclude opinion articles")
    paper = V24Paper(
        "synthetic-precedence",
        "Opinion article describing a relevant evaluation",
        "The source is an opinion article and describes the requested evaluation.",
    )
    result = _validate_and_decide(
        _assessment_with_user_exclusion(
            exclusion_verdict="MET",
            decision="REJECT",
        ),
        protocol,
        paper,
    )
    assert result["decision"] == "REJECT"
    assert result["validation_status"] == "validated"


class FakeV24Browser:
    instances = []

    def __init__(self, config):
        self.prompts = []
        self.new_chats = 0
        self.recoveries = []
        FakeV24Browser.instances.append(self)

    def __enter__(self):
        self.start_new_job_chat()
        return self

    def __exit__(self, *args):
        return None

    def set_attempt_context(self, **kwargs):
        return None

    def start_new_job_chat(self):
        self.new_chats += 1

    def note_recovery(self, action):
        self.recoveries.append(action)

    def recover_transport_failure(self, **kwargs):
        self.recoveries.append("transport")

    def submit_prompt_and_get_response(self, prompt):
        self.prompts.append(prompt)
        if "Compile an immutable systematic-review screening protocol" in prompt:
            return json.dumps(_protocol_payload())
        identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
        return json.dumps({
            "items": [
                _compact_assessment(_assessment(paper_id=paper_id))
                for paper_id in identifiers
            ]
        })


def _run(
    tmp_path,
    *,
    job_id,
    output_name,
    browser_factory=FakeV24Browser,
    frame=None,
    resume=False,
    input_fingerprint="same-input",
):
    if frame is None:
        frame = pd.DataFrame({
            "Title": ["A substantive requested relationship study"],
            "Abstract": ["The requested approach is evaluated for the requested task."],
            "Existing": ["preserved"],
        })
    progress = ScreeningProgress()
    session = ScreeningSession()
    assert progress.start_job(job_id)
    progress.begin_screening(job_id, 1, GEMINI_WEB_V24_VERSION)
    output = tmp_path / "runs" / output_name
    summary = screen_csv_with_gemini_web_v24(
        frame=frame,
        valid=frame,
        title_col="Title",
        abstract_col="Abstract",
        research_question="How does a requested approach address a requested task?",
        research_context="",
        inclusion_criteria="",
        exclusion_criteria="",
        output_path=str(output),
        job_id=job_id,
        input_fingerprint=input_fingerprint,
        resume=resume,
        limit=0,
        progress=progress,
        screening_session=session,
        browser_factory=browser_factory,
    )
    return summary, pd.read_csv(output)


def test_end_to_end_one_paper_then_validated_cache(tmp_path):
    FakeV24Browser.instances.clear()
    first, saved = _run(tmp_path, job_id="v24-first", output_name="first.csv")
    assert first["architecture_version"] == GEMINI_WEB_V24_VERSION
    assert first["keep"] == 1
    assert first["verification_count"] == 0
    assert first["primary_batches_submitted"] == 1
    assert first["primary_papers_requested"] == 1
    assert first["primary_structured_failures"] == 0
    assert first["primary_technical_fallbacks"] == 0
    assert first["verification_batches_submitted"] == 0
    assert first["verification_papers_requested"] == 0
    assert first["job_id"] == "v24-first"
    assert first["run_status"] == "complete"
    assert first["run_selected_count"] == 1
    assert first["run_selected_source_row_ids"] == ["0"]
    assert first["resumed_source_row_ids"] == []
    assert first["assessment_cache_hit_source_row_ids"] == []
    assert first["fresh_primary_source_row_ids"] == ["0"]
    assert first["directly_handled_without_primary_source_row_ids"] == []
    assert first["direct_handling_reasons"] == {}
    assert first["missing_abstract_source_row_ids"] == []
    assert first["screening_input_fingerprint"] == "same-input"
    assert re.fullmatch(r"[0-9a-f]{64}", first["source_dataset_fingerprint"])
    assert re.fullmatch(r"[0-9a-f]{64}", first["screening_output_fingerprint"])
    persisted_summary = json.loads(
        (
            tmp_path / "cache" / "gemini_web_v24" / "diagnostics"
            / "v24-first.summary.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted_summary["screening_output_fingerprint"] == (
        first["screening_output_fingerprint"]
    )
    assert saved.loc[0, "Existing"] == "preserved"
    assert saved.loc[0, "Route_Used"] == "primary"
    assert saved.loc[0, "Validation_Status"] == "validated"
    assert saved.loc[0, "Prompt_Version"] == GEMINI_WEB_V24_ASSESSMENT_PROMPT_VERSION
    assert saved.loc[0, "Execution_Origin"] == "fresh_primary"
    assert pd.isna(saved.loc[0, "Direct_Handling_Reason"])
    assert len(FakeV24Browser.instances) == 1
    second, cached = _run(tmp_path, job_id="v24-cache", output_name="cached.csv")
    assert second["cache_hit_count"] == 1
    assert second["assessment_cache_hit_source_row_ids"] == ["0"]
    assert second["fresh_primary_source_row_ids"] == []
    assert cached.loc[0, "Execution_Origin"] == "assessment_cache_hit"
    assert cached.loc[0, "Route_Used"] == "validated_cache"
    assert bool(cached.loc[0, "Cache_Hit"])
    assert len(FakeV24Browser.instances) == 1


def test_phase3a_v4_assessment_cache_is_not_reused(tmp_path):
    FakeV24Browser.instances.clear()
    first, _ = _run(
        tmp_path, job_id="v5-cache-first", output_name="v5-cache-first.csv",
        input_fingerprint="v5-cache-isolation",
    )
    assert first["cache_hit_count"] == 0
    cache_file = next(
        (tmp_path / "cache" / "gemini_web_v24" / "assessments").glob("*.json")
    )
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    payload["cache_version"] = "gemini-web-v2.4-assessment-v4"
    payload.pop("assessment_prompt_version", None)
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    second, saved = _run(
        tmp_path, job_id="v5-cache-second", output_name="v5-cache-second.csv",
        input_fingerprint="v5-cache-isolation",
    )
    assert second["cache_hit_count"] == 0
    assert saved.loc[0, "Route_Used"] == "primary"
    assert len(FakeV24Browser.instances) == 2


def test_phase3a_checkpoint_identity_includes_assessment_prompt_version(monkeypatch):
    current = v24_screening._contract_key("input", "protocol")
    monkeypatch.setattr(
        v24_screening,
        "GEMINI_WEB_V24_ASSESSMENT_PROMPT_VERSION",
        "gemini-web-v2.4-assessment-prompt-v4",
    )
    assert v24_screening._contract_key("input", "protocol") != current


def _five_paper_frame():
    return pd.DataFrame({
        "Title": [f"Substantive relationship study {index}" for index in range(5)],
        "Abstract": [
            "The requested approach is evaluated for the requested task."
            for _ in range(5)
        ],
    })


def _valid_batch_response(identifiers):
    return json.dumps({
        "items": [
            _compact_assessment(_assessment(paper_id=paper_id))
            for paper_id in identifiers
        ]
    })


def _diagnostic_events(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_phase3a_attempt_diagnostics_capture_sizes_and_contract_counts(tmp_path):
    browser = FakeV24Browser(GeminiWebV24Config())
    diagnostics_path = tmp_path / "attempts.jsonl"
    diagnostics = V24Diagnostics(diagnostics_path)
    assessed, failures = _execute_batch_with_degraded_retry(
        browser,
        _protocol_with_criterion_count(1),
        [V24Paper("0", "Title", "Abstract")],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
        batch_id="primary-0001",
    )
    assert set(assessed) == {"0"}
    assert failures == {}
    event = next(
        item for item in _diagnostic_events(diagnostics_path)
        if item["event"] == "gemini_web_assessment_attempt"
    )
    assert event["stage"] == "v24_primary"
    assert event["batch_id"] == "primary-0001"
    assert event["subgroup_id"] == ""
    assert event["criterion_count"] == 1
    assert event["expected_criterion_object_count"] == 1
    assert event["paper_count"] == 1
    assert event["prompt_utf8_bytes"] > 0
    assert event["response_utf8_bytes"] > 0
    assert event["parsed_item_count"] == 1
    assert event["failure_class"] == ""


def test_timed_out_degraded_subgroup_replays_exactly_once_and_recovers(tmp_path):
    class ReplaySuccessBrowser:
        def __init__(self):
            self.batch_sizes = []
            self.paper_ids = []
            self.prompts = []
            self.events = []

        def set_attempt_context(self, **kwargs):
            return None

        def submit_prompt_and_get_response(self, prompt):
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.batch_sizes.append(len(identifiers))
            self.paper_ids.append(identifiers)
            self.prompts.append(prompt)
            self.events.append(("call", identifiers))
            call = len(self.batch_sizes)
            if call <= 2:
                return '{"items": []}'
            if call == 3:
                raise TimeoutError("response timed out")
            return _valid_batch_response(identifiers)

        def start_new_job_chat(self):
            self.events.append(("chat", []))

        def note_recovery(self, action):
            self.events.append(("note", action))

        def recover_transport_failure(self, **kwargs):
            self.events.append(("transport", kwargs))

    browser = ReplaySuccessBrowser()
    diagnostics_path = tmp_path / "replay-success-diagnostics.jsonl"
    diagnostics = V24Diagnostics(diagnostics_path)
    assessed, failures = _execute_batch_with_degraded_retry(
        browser,
        _protocol(),
        [
            V24Paper(str(index), f"Title {index}", f"Abstract {index}")
            for index in range(5)
        ],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
    )

    assert browser.batch_sizes == [5, 5, 2, 2, 3]
    assert browser.paper_ids[2] == browser.paper_ids[3] == ["0", "1"]
    assert browser.prompts[2] == browser.prompts[3]
    first_two_calls = [
        index for index, event in enumerate(browser.events)
        if event == ("call", ["0", "1"])
    ]
    transport_recovery = next(
        index for index, event in enumerate(browser.events)
        if event[0] == "transport"
    )
    assert first_two_calls[0] < transport_recovery < first_two_calls[1]
    assert set(assessed) == {"0", "1", "2", "3", "4"}
    assert failures == {}
    assert diagnostics.degraded_subgroup_replay_count == 1
    assert diagnostics.degraded_subgroup_replay_success_count == 1
    assert diagnostics.degraded_subgroup_replay_exhaustion_count == 0
    assert diagnostics.papers_recovered_through_replay == 2

    subgroup_events = [
        event for event in _diagnostic_events(diagnostics_path)
        if event["event"] == "gemini_web_degraded_subgroup"
    ]
    assert [event["outcome"] for event in subgroup_events] == [
        "transport_failure",
        "transport_recovery",
        "transport_replay_succeeded",
    ]
    assert all(event["paper_ids"] == ["0", "1"] for event in subgroup_events)
    assert all(event["paper_count"] == 2 for event in subgroup_events)


def test_timed_out_degraded_subgroup_exhausts_without_recursive_split(tmp_path):
    class ReplayExhaustionBrowser:
        def __init__(self):
            self.batch_sizes = []

        def set_attempt_context(self, **kwargs):
            return None

        def submit_prompt_and_get_response(self, prompt):
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.batch_sizes.append(len(identifiers))
            if len(self.batch_sizes) <= 2:
                return '{"items": []}'
            if identifiers == ["0", "1"]:
                raise TimeoutError("response timed out")
            return _valid_batch_response(identifiers)

        def start_new_job_chat(self):
            return None

        def note_recovery(self, action):
            return None

        def recover_transport_failure(self, **kwargs):
            return None

    browser = ReplayExhaustionBrowser()
    diagnostics_path = tmp_path / "replay-exhaustion-diagnostics.jsonl"
    diagnostics = V24Diagnostics(diagnostics_path)
    assessed, failures = _execute_batch_with_degraded_retry(
        browser,
        _protocol(),
        [
            V24Paper(str(index), f"Title {index}", f"Abstract {index}")
            for index in range(5)
        ],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
    )

    assert browser.batch_sizes == [5, 5, 2, 2, 3]
    assert set(assessed) == {"2", "3", "4"}
    assert set(failures) == {"0", "1"}
    assert {failure_class for _, failure_class in failures.values()} == {
        V24_TRANSPORT_FAILURE
    }
    assert diagnostics.fallback_count == 1
    assert diagnostics.degraded_subgroup_replay_count == 1
    assert diagnostics.degraded_subgroup_replay_success_count == 0
    assert diagnostics.degraded_subgroup_replay_exhaustion_count == 1
    assert diagnostics.papers_recovered_through_replay == 0
    outcomes = [
        event["outcome"] for event in _diagnostic_events(diagnostics_path)
        if event["event"] == "gemini_web_degraded_subgroup"
    ]
    assert outcomes == [
        "transport_failure",
        "transport_recovery",
        "transport_replay_exhausted",
    ]


def test_structured_degraded_subgroup_failure_is_terminal_without_replay(tmp_path):
    class StructuredTerminalBrowser:
        def __init__(self):
            self.batch_sizes = []

        def set_attempt_context(self, **kwargs):
            return None

        def submit_prompt_and_get_response(self, prompt):
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.batch_sizes.append(len(identifiers))
            if len(self.batch_sizes) <= 3:
                return '{"items": []}'
            return _valid_batch_response(identifiers)

        def start_new_job_chat(self):
            return None

        def note_recovery(self, action):
            return None

        def recover_transport_failure(self, **kwargs):
            return None

    browser = StructuredTerminalBrowser()
    diagnostics_path = tmp_path / "structured-terminal-diagnostics.jsonl"
    diagnostics = V24Diagnostics(diagnostics_path)
    assessed, failures = _execute_batch_with_degraded_retry(
        browser,
        _protocol(),
        [
            V24Paper(str(index), f"Title {index}", f"Abstract {index}")
            for index in range(5)
        ],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
    )

    assert browser.batch_sizes == [5, 5, 2, 3]
    assert set(assessed) == {"2", "3", "4"}
    assert set(failures) == {"0", "1"}
    assert {failure_class for _, failure_class in failures.values()} == {
        V24_STRUCTURED_OUTPUT_FAILURE
    }
    assert diagnostics.degraded_subgroup_replay_count == 0
    subgroup_events = [
        event for event in _diagnostic_events(diagnostics_path)
        if event["event"] == "gemini_web_degraded_subgroup"
    ]
    assert len(subgroup_events) == 1
    assert subgroup_events[0]["outcome"] == "structured_output_terminal"
    assert subgroup_events[0]["paper_ids"] == ["0", "1"]


def test_successful_degraded_subgroup_replay_is_cached_end_to_end(tmp_path):
    class ReplayCachingBrowser(FakeV24Browser):
        def __init__(self, config):
            super().__init__(config)
            self.batch_sizes = []

        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable systematic-review screening protocol" in prompt:
                return json.dumps(_protocol_payload())
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.batch_sizes.append(len(identifiers))
            call = len(self.batch_sizes)
            if call <= 2:
                return '{"items": []}'
            if call == 3:
                raise TimeoutError("response timed out")
            return _valid_batch_response(identifiers)

    FakeV24Browser.instances.clear()
    first, saved = _run(
        tmp_path,
        job_id="v24-subgroup-replay-cache",
        output_name="subgroup-replay-cache.csv",
        browser_factory=ReplayCachingBrowser,
        frame=_five_paper_frame(),
        input_fingerprint="subgroup-replay-cache-input",
    )

    first_browser = FakeV24Browser.instances[0]
    assert first_browser.batch_sizes == [5, 5, 2, 2, 3]
    assert first["keep"] == 5
    assert first["technical_fallback_count"] == 0
    assert first["timeout_fallback_count"] == 0
    assert first["degraded_subgroup_replay_count"] == 1
    assert first["degraded_subgroup_replay_success_count"] == 1
    assert first["degraded_subgroup_replay_exhaustion_count"] == 0
    assert first["papers_recovered_through_replay"] == 2
    assert set(saved["Route_Used"]) == {"primary"}

    second, cached = _run(
        tmp_path,
        job_id="v24-subgroup-replay-cache-second",
        output_name="subgroup-replay-cache-second.csv",
        browser_factory=ReplayCachingBrowser,
        frame=_five_paper_frame(),
        input_fingerprint="subgroup-replay-cache-input",
    )
    assert second["cache_hit_count"] == 5
    assert set(cached["Route_Used"]) == {"validated_cache"}
    assert len(FakeV24Browser.instances) == 1


def test_exhausted_subgroup_replay_is_resume_eligible_without_reprocessing_sibling(
    tmp_path,
):
    class ReplayResumeBrowser(FakeV24Browser):
        recover_failed = False

        def __init__(self, config):
            super().__init__(config)
            self.batch_sizes = []
            self.paper_ids = []

        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable systematic-review screening protocol" in prompt:
                return json.dumps(_protocol_payload())
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.batch_sizes.append(len(identifiers))
            self.paper_ids.append(identifiers)
            if self.recover_failed:
                return _valid_batch_response(identifiers)
            if len(self.batch_sizes) <= 2:
                return '{"items": []}'
            if identifiers == ["0", "1"]:
                raise TimeoutError("response timed out")
            return _valid_batch_response(identifiers)

    FakeV24Browser.instances.clear()
    ReplayResumeBrowser.recover_failed = False
    first, saved = _run(
        tmp_path,
        job_id="v24-subgroup-replay-resume",
        output_name="subgroup-replay-resume.csv",
        browser_factory=ReplayResumeBrowser,
        frame=_five_paper_frame(),
        input_fingerprint="subgroup-replay-resume-input",
    )

    first_browser = FakeV24Browser.instances[0]
    assert first_browser.batch_sizes == [5, 5, 2, 2, 3]
    assert first["keep"] == 3
    assert first["maybe"] == 2
    assert first["timeout_fallback_count"] == 2
    assert first["degraded_subgroup_replay_exhaustion_count"] == 1
    failed = saved[saved["Decision"] == "MAYBE"]
    assert set(failed["Source_Row_Index"]) == {0, 1}
    assert set(failed["Failure_Class"]) == {V24_TRANSPORT_FAILURE}

    ReplayResumeBrowser.recover_failed = True
    second, resumed = _run(
        tmp_path,
        job_id="v24-subgroup-replay-resume-second",
        output_name="subgroup-replay-resume-second.csv",
        browser_factory=ReplayResumeBrowser,
        frame=_five_paper_frame(),
        resume=True,
        input_fingerprint="subgroup-replay-resume-input",
    )

    second_browser = FakeV24Browser.instances[1]
    assert second_browser.batch_sizes == [2]
    assert second_browser.paper_ids == [["0", "1"]]
    assert second["resumed_count"] == 3
    assert len(second["fresh_primary_source_row_ids"]) == 2
    assert second["keep"] == 5
    assert second["technical_fallback_count"] == 0
    assert set(resumed["Decision"]) == {"KEEP"}
    assert set(
        resumed.loc[resumed["Source_Row_Index"].isin([2, 3, 4]), "Execution_Origin"]
    ) == {"resumed"}
    assert set(
        resumed.loc[resumed["Source_Row_Index"].isin([0, 1]), "Execution_Origin"]
    ) == {"fresh_primary"}


def test_invalid_five_paper_batch_recovers_as_two_plus_three_and_caches(tmp_path):
    class RecoveringStructuredBrowser(FakeV24Browser):
        def __init__(self, config):
            super().__init__(config)
            self.batch_sizes = []

        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable systematic-review screening protocol" in prompt:
                return json.dumps(_protocol_payload())
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.batch_sizes.append(len(identifiers))
            if len(self.batch_sizes) <= 2:
                return '{"items": []}'
            return _valid_batch_response(identifiers)

    FakeV24Browser.instances.clear()
    first, saved = _run(
        tmp_path,
        job_id="v24-degraded-structured",
        output_name="degraded-structured.csv",
        browser_factory=RecoveringStructuredBrowser,
        frame=_five_paper_frame(),
        input_fingerprint="degraded-structured-input",
    )

    browser = FakeV24Browser.instances[0]
    assert browser.batch_sizes == [5, 5, 2, 3]
    assert first["keep"] == 5
    assert first["technical_fallback_count"] == 0
    assert first["structured_output_fallback_count"] == 0
    assert set(saved["Route_Used"]) == {"primary"}
    assert set(saved["Failure_Class"].fillna("")) == {""}
    assert "v24_primary_degraded_retry_structured_clean_chat" in browser.recoveries

    second, cached = _run(
        tmp_path,
        job_id="v24-degraded-structured-cache",
        output_name="degraded-structured-cache.csv",
        browser_factory=RecoveringStructuredBrowser,
        frame=_five_paper_frame(),
        input_fingerprint="degraded-structured-input",
    )
    assert second["cache_hit_count"] == 5
    assert set(cached["Route_Used"]) == {"validated_cache"}
    assert len(FakeV24Browser.instances) == 1


def test_timed_out_five_paper_batch_recovers_as_two_plus_three(tmp_path):
    class RecoveringTimeoutBrowser:
        def __init__(self):
            self.batch_sizes = []
            self.recoveries = []

        def set_attempt_context(self, **kwargs):
            return None

        def submit_prompt_and_get_response(self, prompt):
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            self.batch_sizes.append(len(identifiers))
            if len(self.batch_sizes) <= 2:
                raise TimeoutError("response timed out")
            return _valid_batch_response(identifiers)

        def start_new_job_chat(self):
            self.recoveries.append("chat")

        def note_recovery(self, action):
            self.recoveries.append(action)

        def recover_transport_failure(self, **kwargs):
            self.recoveries.append("transport")

    browser = RecoveringTimeoutBrowser()
    diagnostics = V24Diagnostics(tmp_path / "timeout-diagnostics.jsonl")
    assessed, failures = _execute_batch_with_degraded_retry(
        browser,
        _protocol(),
        [
            V24Paper(str(index), f"Title {index}", f"Abstract {index}")
            for index in range(5)
        ],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
    )

    assert set(assessed) == {"0", "1", "2", "3", "4"}
    assert failures == {}
    assert browser.batch_sizes == [5, 5, 2, 3]
    assert "v24_primary_degraded_retry_transport_recovery" in browser.recoveries
    assert diagnostics.fallback_count == 0


def test_failed_two_paper_subgroup_is_isolated_and_reprocessed_on_resume(tmp_path):
    class PartialStructuredBrowser(FakeV24Browser):
        recover_two = False

        def __init__(self, config):
            super().__init__(config)
            self.batch_sizes = []

        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable systematic-review screening protocol" in prompt:
                return json.dumps(_protocol_payload())
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            size = len(identifiers)
            self.batch_sizes.append(size)
            if size == 5 or (size == 2 and not self.recover_two):
                return '{"items": []}'
            return _valid_batch_response(identifiers)

    FakeV24Browser.instances.clear()
    PartialStructuredBrowser.recover_two = False
    first, saved = _run(
        tmp_path,
        job_id="v24-partial-degraded",
        output_name="partial-degraded.csv",
        browser_factory=PartialStructuredBrowser,
        frame=_five_paper_frame(),
        input_fingerprint="partial-degraded-input",
    )

    first_browser = FakeV24Browser.instances[0]
    assert first_browser.batch_sizes == [5, 5, 2, 3]
    assert first["keep"] == 3
    assert first["maybe"] == 2
    assert first["technical_fallback_count"] == 2
    assert first["structured_output_fallback_count"] == 2
    failed = saved[saved["Decision"] == "MAYBE"]
    assert set(failed["Route_Used"]) == {"technical_failure"}
    assert set(failed["Failure_Class"]) == {V24_STRUCTURED_OUTPUT_FAILURE}

    PartialStructuredBrowser.recover_two = True
    second, resumed = _run(
        tmp_path,
        job_id="v24-partial-degraded-resume",
        output_name="partial-degraded-resume.csv",
        browser_factory=PartialStructuredBrowser,
        frame=_five_paper_frame(),
        resume=True,
        input_fingerprint="partial-degraded-input",
    )

    second_browser = FakeV24Browser.instances[1]
    assert second_browser.batch_sizes == [2]
    assert second["resumed_count"] == 3
    assert len(second["fresh_primary_source_row_ids"]) == 2
    assert second["keep"] == 5
    assert second["technical_fallback_count"] == 0
    assert set(resumed["Decision"]) == {"KEEP"}
    assert set(
        resumed.loc[resumed["Source_Row_Index"].isin([2, 3, 4]), "Execution_Origin"]
    ) == {"resumed"}


def test_end_to_end_valid_insufficient_maybe_skips_verifier(tmp_path):
    class InsufficientMaybeBrowser(FakeV24Browser):
        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable systematic-review screening protocol" in prompt:
                return json.dumps(_protocol_payload())
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            return json.dumps({
                "items": [
                    _compact_assessment(_assessment(
                        paper_id=paper_id,
                        verdict="UNCLEAR",
                        support="INSUFFICIENT",
                        relationship="INSUFFICIENT",
                        decision="MAYBE",
                        risk="BORDERLINE",
                        evidence=False,
                    ))
                    for paper_id in identifiers
                ]
            })

    InsufficientMaybeBrowser.instances.clear()
    summary, saved = _run(
        tmp_path,
        job_id="v24-insufficient-maybe",
        output_name="insufficient-maybe.csv",
        browser_factory=InsufficientMaybeBrowser,
    )

    browser = InsufficientMaybeBrowser.instances[0]
    assert summary["maybe"] == 1
    assert summary["verification_count"] == 0
    assert len(browser.prompts) == 2
    assert not any("independent prediction-blind verifier" in prompt for prompt in browser.prompts)
    assert "v24_primary_to_verification_clean_chat" not in browser.recoveries
    assert saved.loc[0, "Decision"] == "MAYBE"
    assert saved.loc[0, "Decision_Risk"] == "BORDERLINE"
    assert saved.loc[0, "Route_Used"] == "primary"
    assert saved.loc[0, "Verification_Status"] == "not_required"


def test_missing_abstract_is_safe_maybe_without_paper_assessment_call(tmp_path):
    FakeV24Browser.instances.clear()
    frame = pd.DataFrame({"Title": ["Title-only record"], "Abstract": [""]})
    progress = ScreeningProgress()
    session = ScreeningSession()
    assert progress.start_job("v24-missing-abstract")
    progress.begin_screening("v24-missing-abstract", 1, GEMINI_WEB_V24_VERSION)
    output = tmp_path / "runs" / "missing.csv"
    summary = screen_csv_with_gemini_web_v24(
        frame=frame,
        valid=frame,
        title_col="Title",
        abstract_col="Abstract",
        research_question="How does a requested approach address a requested task?",
        research_context="",
        inclusion_criteria="",
        exclusion_criteria="",
        output_path=str(output),
        job_id="v24-missing-abstract",
        input_fingerprint="missing-abstract",
        resume=False,
        limit=0,
        progress=progress,
        screening_session=session,
        browser_factory=FakeV24Browser,
    )
    saved = pd.read_csv(output)
    assert summary["maybe"] == 1
    assert summary["directly_handled_without_primary_count"] == 1
    assert summary["directly_handled_without_primary_source_row_ids"] == ["0"]
    assert summary["direct_handling_reasons"] == {"0": "missing_abstract"}
    assert summary["missing_abstract_source_row_ids"] == ["0"]
    assert saved.loc[0, "Route_Used"] == "missing_abstract"
    assert saved.loc[0, "Execution_Origin"] == "directly_handled_without_primary"
    assert saved.loc[0, "Direct_Handling_Reason"] == "missing_abstract"
    assert saved.loc[0, "Verification_Status"] == "not_required"
    assert saved.loc[0, "Validation_Status"] == "validated"
    assert len(FakeV24Browser.instances[0].prompts) == 1


def test_current_run_resume_origin_overrides_prior_direct_origin_and_missing_overlaps(
    tmp_path,
):
    frame = pd.DataFrame({"Title": ["Title-only record"], "Abstract": [""]})
    first, _ = _run(
        tmp_path,
        job_id="v24-direct-first",
        output_name="direct-first.csv",
        frame=frame,
        input_fingerprint="direct-resume",
    )
    assert first["directly_handled_without_primary_source_row_ids"] == ["0"]
    second, _ = _run(
        tmp_path,
        job_id="v24-direct-resumed",
        output_name="direct-resumed.csv",
        frame=frame,
        resume=True,
        input_fingerprint="direct-resume",
    )
    saved = pd.read_csv(
        tmp_path / "runs" / "direct-resumed.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert second["resumed_source_row_ids"] == ["0"]
    assert second["directly_handled_without_primary_source_row_ids"] == []
    assert second["direct_handling_reasons"] == {}
    assert second["missing_abstract_source_row_ids"] == ["0"]
    assert saved.loc[0, "Execution_Origin"] == "resumed"
    assert saved.loc[0, "Direct_Handling_Reason"] == ""


def test_source_row_index_with_leading_zero_is_persisted_exactly(tmp_path):
    frame = pd.DataFrame(
        {
            "Title": ["A substantive requested relationship study"],
            "Abstract": ["The requested approach is evaluated."],
        },
        index=["001"],
    )
    summary, _ = _run(
        tmp_path,
        job_id="v24-leading-zero",
        output_name="leading-zero.csv",
        frame=frame,
    )
    saved = pd.read_csv(
        tmp_path / "runs" / "leading-zero.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert summary["run_selected_source_row_ids"] == ["001"]
    assert summary["fresh_primary_source_row_ids"] == ["001"]
    assert saved.loc[0, "Source_Row_Index"] == "001"


def test_keep_reject_verification_conflict_resolves_to_maybe(tmp_path):
    class ConflictBrowser(FakeV24Browser):
        def submit_prompt_and_get_response(self, prompt):
            self.prompts.append(prompt)
            if "Compile an immutable systematic-review screening protocol" in prompt:
                return json.dumps(_protocol_payload())
            identifiers = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
            verification = "independent prediction-blind verifier" in prompt
            items = []
            for paper_id in identifiers:
                if verification:
                    item = _assessment(
                        paper_id=paper_id,
                        verdict="NOT_MET",
                        relationship="CONFLICTS",
                        decision="REJECT",
                    )
                else:
                    item = _assessment(paper_id=paper_id, risk="BORDERLINE")
                items.append(_compact_assessment(item))
            return json.dumps({"items": items})

    summary, saved = _run(
        tmp_path,
        job_id="v24-conflict",
        output_name="conflict.csv",
        browser_factory=ConflictBrowser,
    )
    assert summary["maybe"] == 1
    assert saved.loc[0, "Decision"] == "MAYBE"
    assert saved.loc[0, "Verification_Status"] == "disagreed"
    assert saved.loc[0, "Route_Used"] == "risky_definitive"
    assert not bool(saved.loc[0, "Cache_Hit"])


@pytest.mark.parametrize(
    "failure_class",
    [V24_TRANSPORT_FAILURE, V24_STRUCTURED_OUTPUT_FAILURE],
)
def test_technical_fallback_checkpoint_is_requeued(tmp_path, failure_class):
    checkpoint = tmp_path / "checkpoint.csv"
    pd.DataFrame([{
        "Source_Row_Index": 0,
        "Protocol_ID": "protocol",
        "Prompt_Version": GEMINI_WEB_V24_ASSESSMENT_PROMPT_VERSION,
        "Decision": "MAYBE",
        "Validation_Status": "unresolved",
        "Verification_Status": "failed",
        "Failure_Class": failure_class,
        "Criteria_JSON": "[]",
        "Evidence_JSON": "[]",
    }]).to_csv(checkpoint, index=False)
    assert _resume_rows(checkpoint, "protocol", {"0"}) == {}


def test_malformed_single_paper_batch_gets_one_repair_then_safe_failure(tmp_path):
    class MalformedBrowser:
        def __init__(self):
            self.calls = 0
            self.new_chats = 0

        def set_attempt_context(self, **kwargs):
            return None

        def submit_prompt_and_get_response(self, prompt):
            self.calls += 1
            return '{"items": []}'

        def start_new_job_chat(self):
            self.new_chats += 1

        def recover_transport_failure(self, **kwargs):
            return None

    browser = MalformedBrowser()
    diagnostics = V24Diagnostics(tmp_path / "diagnostics.jsonl")
    assessed, failures = _execute_batch_with_degraded_retry(
        browser,
        _protocol(),
        [V24Paper("0", "Title", "Abstract")],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
    )
    assert assessed == {}
    reason, failure_class = failures["0"]
    assert "failed after one retry" in reason
    assert failure_class == V24_STRUCTURED_OUTPUT_FAILURE
    assert browser.calls == 2
    assert browser.new_chats == 1
    assert diagnostics.retry_count == 2
    assert diagnostics.fallback_count == 1


def test_exhausted_timeout_is_classified_as_transport_failure(tmp_path):
    class TimeoutBrowser:
        def __init__(self):
            self.calls = 0
            self.recoveries = 0

        def set_attempt_context(self, **kwargs):
            return None

        def submit_prompt_and_get_response(self, prompt):
            self.calls += 1
            raise TimeoutError("response timed out")

        def start_new_job_chat(self):
            return None

        def recover_transport_failure(self, **kwargs):
            self.recoveries += 1

    browser = TimeoutBrowser()
    diagnostics = V24Diagnostics(tmp_path / "timeout-failure-diagnostics.jsonl")
    assessed, failures = _execute_batch_with_degraded_retry(
        browser,
        _protocol(),
        [V24Paper("0", "Title", "Abstract")],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
    )

    assert assessed == {}
    assert failures["0"][1] == V24_TRANSPORT_FAILURE
    assert browser.calls == 2
    assert browser.recoveries == 2
    assert diagnostics.retry_count == 2
    assert diagnostics.fallback_count == 1


def test_v24_lifecycle_defaults_are_bounded_and_independent(monkeypatch):
    monkeypatch.delenv("GEMINI_WEB_V24_MAX_CHAT_SUBMISSIONS", raising=False)
    monkeypatch.delenv("GEMINI_WEB_V24_MAX_BROWSER_SUBMISSIONS", raising=False)
    config = GeminiWebV24Config()
    assert config.max_chat_submissions == 5
    assert config.max_browser_submissions == 10
    transport = config.transport_config()
    assert transport.max_chat_submissions == 5
    assert transport.max_browser_submissions == 10


def test_v24_engine_is_explicit_and_obsolete_engine_falls_back_to_local():
    assert normalize_processing_engine("gemini_web_v24") == GEMINI_WEB_V24_ENGINE
    assert normalize_processing_engine("gemini-web-v2.4") == GEMINI_WEB_V24_ENGINE
    assert normalize_processing_engine("gemini_web") == "local"
