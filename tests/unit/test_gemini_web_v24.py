import json
import re
from pathlib import Path

import pandas as pd
import pytest

import litsync_app.integrations.gemini_web_v24_screening as v24_screening
from litsync_app.integrations.gemini_browser import (
    GeminiWebAutomation,
    GeminiWebConfig,
    _ResponseSnapshot,
)
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
    V24RuntimeMetrics,
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
    runtime_metrics=None,
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
        runtime_metrics=runtime_metrics,
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
    attempts = [
        event for event in _diagnostic_events(diagnostics.path)
        if event["event"] == "gemini_web_assessment_attempt"
    ]
    assert [event["retry_number"] for event in attempts] == [0, 1]
    assert all(event["exception_type"] == "LocalAIOutputError" for event in attempts)
    assert all(
        0 < len(event["exception_message"]) <= 300
        for event in attempts
    )
    assert all(event["response_empty"] is False for event in attempts)
    assert all(event["syntactically_valid_json"] is True for event in attempts)
    assert all(event["response_utf8_bytes"] > 0 for event in attempts)
    assert all(event["parsed_item_count"] == 0 for event in attempts)
    assert all(
        event["structured_failure_code"] == "schema_validation_failed"
        for event in attempts
    )
    assert all(event["parser_total_candidate_count"] == 3 for event in attempts)
    assert all(
        event["parser_json_decodable_candidate_count"] == 3
        for event in attempts
    )
    assert all(event["parser_dictionary_candidate_count"] == 2 for event in attempts)
    assert all(
        event["parser_schema_validation_failure_count"] == 2
        for event in attempts
    )
    assert all(event["parser_validation_error_count"] == 1 for event in attempts)
    assert all(event["parser_validation_error_types"] == ["too_short"] for event in attempts)
    assert all(event["parser_validation_error_locations"] == [["items"]] for event in attempts)
    assert all(
        event["parser_validation_error_messages"]
        == ["Value does not meet the minimum length"]
        for event in attempts
    )
    assert all(event["parser_candidate_source"] == "complete_response" for event in attempts)
    assert all(
        event["parser_full_response_json_decodable"] is True
        for event in attempts
    )
    assert all(
        event["parser_full_response_top_level_type"] == "dict"
        for event in attempts
    )
    assert all(
        event["parser_full_response_schema_valid"] is False
        for event in attempts
    )
    assert all(
        event["parser_full_response_raw_decode_succeeded"] is True
        for event in attempts
    )
    assert all(
        event["parser_full_response_raw_decode_consumed_ratio"] == 1.0
        for event in attempts
    )


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


class _SequenceClock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def test_runtime_metrics_aggregate_success_failure_and_nearest_rank_percentiles():
    metrics = V24RuntimeMetrics()
    for duration in range(1, 21):
        metrics.record(
            "primary", duration, family="gemini_calls", success=duration != 20,
            stage="primary", attempt_type="initial", paper_count=5,
            batch_id="primary-0001", retry_number=0,
        )

    summary = metrics.serialize()
    assert summary["gemini_calls"]["primary"] == {
        "count": 20,
        "total_seconds": 210.0,
        "mean_seconds": 10.5,
        "p50_seconds": 10.0,
        "p95_seconds": 19.0,
        "min_seconds": 1.0,
        "max_seconds": 20.0,
        "success_count": 19,
        "failure_count": 1,
    }
    assert summary["dimensions"]["by_batch_id"]["primary-0001"]["count"] == 20


def test_runtime_observer_records_failure_without_suppressing_real_error():
    metrics = V24RuntimeMetrics(clock=_SequenceClock(10.0, 12.5))

    with pytest.raises(RuntimeError, match="screening failed"):
        with metrics.observe("result_validation"):
            raise RuntimeError("screening failed")

    aggregate = metrics.serialize()["categories"]["result_validation"]
    assert aggregate["total_seconds"] == 2.5
    assert aggregate["failure_count"] == 1


def test_runtime_call_classification_separates_repair_retry_and_subgroup_replay(tmp_path):
    protocol = V24Protocol.model_validate(_protocol_payload()).with_identity()
    papers = [
        V24Paper(
            paper_id=str(index), title=f"Paper {index}",
            abstract="The requested approach is evaluated for the requested task.",
        )
        for index in range(5)
    ]

    class ClassifiedBrowser(FakeV24Browser):
        def __init__(self):
            self.calls_by_ids = {}
            self.prompts = []
            self.recoveries = []
            self.new_chats = 0

        def submit_prompt_and_get_response(self, prompt):
            identifiers = list(dict.fromkeys(
                re.findall(r'"paper_id":\s*"([^"]+)"', prompt)
            ))
            key = tuple(identifiers)
            self.calls_by_ids[key] = self.calls_by_ids.get(key, 0) + 1
            if len(identifiers) == 5:
                return "{}"
            if identifiers == ["0", "1"] and self.calls_by_ids[key] == 1:
                raise TimeoutError("transport timeout")
            return _valid_batch_response(identifiers)

    metrics = V24RuntimeMetrics()
    browser = ClassifiedBrowser()
    diagnostics = V24Diagnostics(tmp_path / "classified.jsonl", metrics)
    assessed, failures = _execute_batch_with_degraded_retry(
        browser, protocol, papers, verification=False, flags=None,
        diagnostics=diagnostics, batch_id="primary-0001",
        runtime_metrics=metrics,
    )

    assert not failures
    assert set(assessed) == {str(index) for index in range(5)}
    calls = metrics.serialize()["gemini_calls"]
    assert calls["primary"]["count"] == 5
    assert calls["structured_repair"]["count"] == 4
    assert calls["bounded_retry"]["count"] == 4
    assert calls["degraded_subgroup_attempt"]["count"] == 3
    assert calls["subgroup_transport_replay"]["count"] == 1


def test_browser_transport_timers_cover_response_wait_and_recovery(monkeypatch):
    events = []
    ticks = iter(float(value) for value in range(30))
    monkeypatch.setattr(
        "litsync_app.integrations.gemini_browser.time.perf_counter",
        lambda: next(ticks),
    )
    browser = GeminiWebAutomation(GeminiWebConfig(diagnostic_sink=events.append))

    class Box:
        def click(self):
            return None

        def fill(self, _prompt):
            return None

    monkeypatch.setattr(browser, "_prepare_for_submission", lambda: None)
    monkeypatch.setattr(browser, "_require_page", lambda: object())
    monkeypatch.setattr(browser, "_response_snapshots", lambda: ())
    monkeypatch.setattr(browser, "_find_prompt_box", Box)
    monkeypatch.setattr(browser, "_submit_prompt", lambda: None)
    monkeypatch.setattr(
        browser, "_wait_for_new_response", lambda _before: '{"items": []}'
    )
    monkeypatch.setattr(browser, "_record_raw_response", lambda _response: None)
    monkeypatch.setattr(browser, "note_recovery", lambda _action: None)
    monkeypatch.setattr(browser, "recover_job_chat", lambda: None)

    assert browser.submit_prompt_and_get_response("prompt") == '{"items": []}'
    browser.recover_transport_failure()

    runtime_events = {
        event["runtime_metric"]: event
        for event in events
        if event.get("event") == "gemini_web_runtime"
    }
    assert runtime_events["response_wait"]["duration_seconds"] > 0
    assert runtime_events["browser_recovery"]["duration_seconds"] > 0
    attempt = next(
        event for event in events if event.get("event") == "gemini_web_attempt"
    )
    assert attempt["attempt_duration_ms"] > 0


def test_runtime_summary_persists_for_cache_only_and_resumed_runs(
    tmp_path, monkeypatch,
):
    FakeV24Browser.instances.clear()
    cleared_jobs = []
    original_clear = ScreeningProgress.clear_prisma_timing_observer

    def clear_observer(self, job_id):
        cleared_jobs.append(job_id)
        return original_clear(self, job_id)

    monkeypatch.setattr(
        ScreeningProgress, "clear_prisma_timing_observer", clear_observer
    )
    _run(tmp_path, job_id="runtime-fresh", output_name="runtime-fresh.csv")
    cached, _ = _run(
        tmp_path, job_id="runtime-cache", output_name="runtime-cache.csv"
    )
    resumed, _ = _run(
        tmp_path, job_id="runtime-resumed", output_name="runtime-resumed.csv",
        resume=True,
    )

    runtime = cached["runtime_metrics"]
    assert runtime["schema_version"] == "gemini-web-v2.4-runtime-v1"
    assert runtime["gemini_calls"]["primary"]["count"] == 0
    assert runtime["categories"]["assessment_cache_load"]["count"] == 1
    assert runtime["categories"]["total_job"]["count"] == 1
    assert runtime["categories"]["local_processing_total"]["total_seconds"] == max(
        0.0,
        runtime["categories"]["total_job"]["total_seconds"]
        - runtime["categories"]["gemini_browser_total"]["total_seconds"],
    )
    assert runtime["definitions"]["local_processing_total"].startswith(
        "Residual non-browser wall time"
    )
    persisted = json.loads(
        (
            tmp_path / "cache" / "gemini_web_v24" / "diagnostics"
            / "runtime-cache.summary.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted["runtime_metrics"] == runtime
    assert resumed["resumed_count"] == 1
    assert resumed["runtime_metrics"]["gemini_calls"]["primary"]["count"] == 0
    assert cleared_jobs == ["runtime-fresh", "runtime-cache", "runtime-resumed"]


def test_metrics_clock_failure_does_not_change_direct_handling_decision(tmp_path):
    class BrokenClock:
        def __call__(self):
            raise RuntimeError("metrics clock failed")

    frame = pd.DataFrame({"Title": ["Title only"], "Abstract": [""]})
    summary, saved = _run(
        tmp_path, job_id="runtime-broken-clock",
        output_name="runtime-broken-clock.csv", frame=frame,
        runtime_metrics=V24RuntimeMetrics(clock=BrokenClock()),
    )

    assert saved.loc[0, "Decision"] == "MAYBE"
    assert summary["directly_handled_without_primary_count"] == 1
    assert summary["runtime_metrics"]["status"] == "partial"
    assert summary["runtime_metrics"]["instrumentation_errors"]


def test_failed_browser_entry_skips_exit_and_clears_prisma_observer(
    tmp_path, monkeypatch,
):
    cache_root = tmp_path / "cache" / "gemini_web_v24"
    _, protocol_path = v24_screening._load_protocol(
        cache_root,
        "How does a requested approach address a requested task?",
        "",
        "",
        "",
    )
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(_protocol().model_dump_json(), encoding="utf-8")

    startup_error = RuntimeError("browser startup failed")

    class FailingEntryBrowser:
        instances = []

        def __init__(self, _config):
            self.exit_calls = 0
            self.submission_calls = 0
            self.__class__.instances.append(self)

        def __enter__(self):
            raise startup_error

        def __exit__(self, *_args):
            self.exit_calls += 1

        def submit_prompt_and_get_response(self, _prompt):
            self.submission_calls += 1
            raise AssertionError("Gemini submission must not occur")

    cleared_jobs = []
    original_clear = ScreeningProgress.clear_prisma_timing_observer

    def clear_observer(self, job_id):
        cleared_jobs.append(job_id)
        return original_clear(self, job_id)

    monkeypatch.setattr(
        ScreeningProgress, "clear_prisma_timing_observer", clear_observer
    )

    with pytest.raises(RuntimeError) as captured:
        _run(
            tmp_path,
            job_id="failed-browser-entry",
            output_name="failed-browser-entry.csv",
            browser_factory=FailingEntryBrowser,
            input_fingerprint="failed-browser-entry-input",
        )

    browser = FailingEntryBrowser.instances[0]
    assert captured.value is startup_error
    assert browser.exit_calls == 0
    assert browser.submission_calls == 0
    assert cleared_jobs == ["failed-browser-entry"]


class _AdvancingMonotonicClock:
    def __init__(self):
        self.seconds = 0.0

    def monotonic(self):
        return self.seconds

    def sleep(self, seconds):
        self.seconds += seconds


def _watchdog_browser(monkeypatch, clock, **config_overrides):
    config_values = {
        "response_timeout_ms": 120000,
        "no_container_timeout_ms": 60000,
        "poll_interval_ms": 1000,
        **config_overrides,
    }
    config = GeminiWebConfig(**config_values)
    browser = GeminiWebAutomation(config)
    monkeypatch.setattr(
        "litsync_app.integrations.gemini_browser.time.monotonic",
        clock.monotonic,
    )
    monkeypatch.setattr(
        "litsync_app.integrations.gemini_browser.time.sleep",
        clock.sleep,
    )
    return browser


def test_stalled_generation_without_container_times_out_at_watchdog(monkeypatch):
    clock = _AdvancingMonotonicClock()
    browser = _watchdog_browser(monkeypatch, clock)
    monkeypatch.setattr(browser, "_response_snapshots", lambda: ())
    monkeypatch.setattr(browser, "_is_generating", lambda: True)

    with pytest.raises(TimeoutError) as captured:
        browser._wait_for_new_response(())

    assert type(captured.value) is TimeoutError
    assert clock.seconds == 60
    assert browser._last_wait_metadata == {
        "response_selector": "",
        "response_container_count": 0,
        "response_state": "no_new_response",
        "generation_detected": True,
        "timeout_stage": "stalled_generation_no_container",
    }
    assert browser._submission_count == 0


@pytest.mark.parametrize("reset_state", ["partial_container", "generation_stopped"])
def test_stalled_generation_watchdog_resets_when_state_breaks(
    monkeypatch, reset_state,
):
    clock = _AdvancingMonotonicClock()
    browser = _watchdog_browser(monkeypatch, clock)
    partial = _ResponseSnapshot("model-response", 1, 0, "partial")

    monkeypatch.setattr(
        browser,
        "_response_snapshots",
        lambda: (
            (partial,)
            if reset_state == "partial_container" and 40 <= clock.seconds < 41
            else ()
        ),
    )
    monkeypatch.setattr(
        browser,
        "_is_generating",
        lambda: not (
            reset_state == "generation_stopped" and 40 <= clock.seconds < 41
        ),
    )

    with pytest.raises(TimeoutError):
        browser._wait_for_new_response(())

    assert clock.seconds == 101
    assert (
        browser._last_wait_metadata["timeout_stage"]
        == "stalled_generation_no_container"
    )


def test_nonmatching_state_keeps_global_response_timeout(monkeypatch):
    clock = _AdvancingMonotonicClock()
    browser = _watchdog_browser(
        monkeypatch,
        clock,
        response_timeout_ms=70000,
    )
    monkeypatch.setattr(browser, "_response_snapshots", lambda: ())
    monkeypatch.setattr(browser, "_is_generating", lambda: False)

    with pytest.raises(TimeoutError):
        browser._wait_for_new_response(())

    assert clock.seconds == 70
    assert browser._last_wait_metadata["timeout_stage"] == "timeout_final_sweep"


def test_stalled_generation_timeout_reuses_no_container_recycle(monkeypatch):
    browser = GeminiWebAutomation(GeminiWebConfig())
    browser._last_wait_metadata = {
        "timeout_stage": "stalled_generation_no_container",
        "response_state": "no_new_response",
        "response_container_count": 0,
        "generation_detected": True,
    }
    recoveries = []
    monkeypatch.setattr(
        browser,
        "recycle_browser_context",
        lambda action: recoveries.append(action),
    )
    monkeypatch.setattr(
        browser,
        "recover_job_chat",
        lambda: pytest.fail("new-chat recovery must not be used"),
    )

    browser.recover_transport_failure()

    assert recoveries == ["browser_recycle_after_no_container_timeout"]
    assert browser._submission_count == 0


def test_no_container_timeout_setting_is_bounded_and_clamped(monkeypatch):
    monkeypatch.setenv("GEMINI_WEB_NO_CONTAINER_TIMEOUT_MS", "1000")
    assert GeminiWebConfig().no_container_timeout_ms == 30000

    monkeypatch.setenv("GEMINI_WEB_NO_CONTAINER_TIMEOUT_MS", "999999")
    assert GeminiWebConfig().no_container_timeout_ms == 120000

    assert GeminiWebConfig(
        response_timeout_ms=45000,
        no_container_timeout_ms=60000,
    ).no_container_timeout_ms == 45000


def test_complete_json_capture_reports_stable_return_reason(monkeypatch):
    clock = _AdvancingMonotonicClock()
    browser = _watchdog_browser(
        monkeypatch,
        clock,
        response_timeout_ms=10000,
        response_stable_ms=750,
    )
    response = '{"items": []}'
    snapshot = _ResponseSnapshot("model-response", 1, 0, response)
    monkeypatch.setattr(browser, "_response_snapshots", lambda: (snapshot,))
    monkeypatch.setattr(browser, "_is_generating", lambda: False)

    assert browser._wait_for_new_response(()) == response
    metadata = browser.last_response_capture_metadata
    assert metadata["response_return_reason"] == "complete_json_stable"
    assert metadata["response_complete_json_at_capture"] is True
    assert metadata["response_generation_detected_at_capture"] is False
    assert metadata["response_stable_duration_ms"] >= 750
    assert metadata["response_utf8_bytes_at_capture"] == len(response.encode("utf-8"))
    assert response not in str(metadata)


def test_incomplete_capture_reports_existing_four_second_fallback(monkeypatch):
    clock = _AdvancingMonotonicClock()
    browser = _watchdog_browser(
        monkeypatch,
        clock,
        response_timeout_ms=10000,
        response_stable_ms=750,
    )
    response = "PRIVATE INCOMPLETE RESPONSE"
    snapshot = _ResponseSnapshot("model-response", 1, 0, response)
    monkeypatch.setattr(browser, "_response_snapshots", lambda: (snapshot,))
    monkeypatch.setattr(browser, "_is_generating", lambda: False)

    assert browser._wait_for_new_response(()) == response
    metadata = browser.last_response_capture_metadata
    assert (
        metadata["response_return_reason"]
        == "incomplete_response_stable_generation_stopped"
    )
    assert metadata["response_complete_json_at_capture"] is False
    assert metadata["response_stable_duration_ms"] >= 4000
    assert response not in str(metadata)


def test_timeout_final_sweep_capture_reports_return_reason(monkeypatch):
    browser = GeminiWebAutomation(
        GeminiWebConfig(response_timeout_ms=0, poll_interval_ms=0)
    )
    response = '{"items": []}'
    snapshot = _ResponseSnapshot("model-response", 1, 0, response)
    monkeypatch.setattr(browser, "_response_snapshots", lambda: (snapshot,))
    monkeypatch.setattr(browser, "_is_generating", lambda: False)

    assert browser._wait_for_new_response(()) == response
    assert (
        browser.last_response_capture_metadata["response_return_reason"]
        == "timeout_final_sweep_complete"
    )


def _prepare_submit_test_browser(monkeypatch, *, diagnostic_sink=None):
    browser = GeminiWebAutomation(
        GeminiWebConfig(diagnostic_sink=diagnostic_sink)
    )

    class Box:
        def click(self):
            return None

        def fill(self, _prompt):
            return None

    monkeypatch.setattr(browser, "_prepare_for_submission", lambda: None)
    monkeypatch.setattr(browser, "_require_page", lambda: object())
    monkeypatch.setattr(browser, "_response_snapshots", lambda: ())
    monkeypatch.setattr(browser, "_find_prompt_box", Box)
    monkeypatch.setattr(browser, "_submit_prompt", lambda: None)
    monkeypatch.setattr(browser, "_record_raw_response", lambda _response: None)
    return browser


def test_success_capture_metadata_is_cleared_when_next_call_times_out(monkeypatch):
    browser = _prepare_submit_test_browser(monkeypatch)
    calls = 0

    def wait(_before):
        nonlocal calls
        calls += 1
        if calls == 1:
            browser._last_response_capture_metadata = {
                "response_return_reason": "complete_json_stable",
                "response_utf8_bytes_at_capture": 14,
            }
            return '{"items": []}'
        raise TimeoutError("timed out")

    monkeypatch.setattr(browser, "_wait_for_new_response", wait)
    assert browser.submit_prompt_and_get_response("first") == '{"items": []}'
    assert browser.last_response_capture_metadata
    with pytest.raises(TimeoutError):
        browser.submit_prompt_and_get_response("second")
    assert browser.last_response_capture_metadata == {}


def test_success_capture_metadata_is_cleared_when_next_call_has_browser_error(
    monkeypatch,
):
    browser = _prepare_submit_test_browser(monkeypatch)

    def successful_wait(_before):
        browser._last_response_capture_metadata = {
            "response_return_reason": "complete_json_stable",
        }
        return '{"items": []}'

    monkeypatch.setattr(browser, "_wait_for_new_response", successful_wait)
    assert browser.submit_prompt_and_get_response("first") == '{"items": []}'
    monkeypatch.setattr(
        browser,
        "_prepare_for_submission",
        lambda: (_ for _ in ()).throw(RuntimeError("browser failed")),
    )
    with pytest.raises(RuntimeError, match="browser failed"):
        browser.submit_prompt_and_get_response("second")
    assert browser.last_response_capture_metadata == {}


def test_selected_capture_is_cleared_if_current_submission_later_fails(
    monkeypatch,
):
    browser = _prepare_submit_test_browser(monkeypatch)

    def successful_wait(_before):
        browser._last_response_capture_metadata = {
            "response_return_reason": "complete_json_stable",
        }
        browser._last_wait_metadata = {
            "response_state": "new_response",
            "response_return_reason": "complete_json_stable",
        }
        return '{"items": []}'

    monkeypatch.setattr(browser, "_wait_for_new_response", successful_wait)
    monkeypatch.setattr(
        browser,
        "_record_raw_response",
        lambda _response: (_ for _ in ()).throw(OSError("capture write failed")),
    )
    with pytest.raises(RuntimeError, match="browser interaction failed"):
        browser.submit_prompt_and_get_response("prompt")
    assert browser.last_response_capture_metadata == {}
    assert "response_return_reason" not in browser._last_wait_metadata
    assert browser._last_wait_metadata["response_state"] == "new_response"


def test_consecutive_successes_publish_only_their_own_defensive_metadata(
    monkeypatch,
):
    browser = _prepare_submit_test_browser(monkeypatch)
    captures = iter([
        {
            "response_return_reason": "complete_json_stable",
            "response_utf8_bytes_at_capture": 14,
        },
        {
            "response_return_reason": (
                "incomplete_response_stable_generation_stopped"
            ),
            "response_utf8_bytes_at_capture": 9,
        },
    ])

    def wait(_before):
        metadata = next(captures)
        browser._last_response_capture_metadata = metadata
        return "response"

    monkeypatch.setattr(browser, "_wait_for_new_response", wait)
    browser.submit_prompt_and_get_response("first")
    first = browser.last_response_capture_metadata
    first["response_return_reason"] = "mutated"
    assert (
        browser.last_response_capture_metadata["response_return_reason"]
        == "complete_json_stable"
    )
    browser.submit_prompt_and_get_response("second")
    assert browser.last_response_capture_metadata == {
        "response_return_reason": (
            "incomplete_response_stable_generation_stopped"
        ),
        "response_utf8_bytes_at_capture": 9,
    }


def test_broken_browser_diagnostic_sink_does_not_alter_return(monkeypatch):
    def broken_sink(_event):
        raise RuntimeError("diagnostics failed")

    browser = _prepare_submit_test_browser(
        monkeypatch,
        diagnostic_sink=broken_sink,
    )
    monkeypatch.setattr(
        browser,
        "_wait_for_new_response",
        lambda _before: '{"items": []}',
    )
    assert browser.submit_prompt_and_get_response("prompt") == '{"items": []}'


def test_structured_diagnostic_failure_preserves_original_parse_failure(
    tmp_path, monkeypatch,
):
    class InvalidBrowser:
        def set_attempt_context(self, **_kwargs):
            return None

        def submit_prompt_and_get_response(self, _prompt):
            return '{"items": []}'

    metadata_builder = v24_screening._structured_failure_diagnostic_metadata
    monkeypatch.setattr(
        v24_screening,
        "_structured_failure_diagnostic_metadata",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("diagnostics failed")),
    )
    diagnostics = V24Diagnostics(tmp_path / "inspection-failure.jsonl")
    assessed, reason, failure_class = v24_screening._execute_batch(
        InvalidBrowser(),
        _protocol(),
        [V24Paper("0", "Title", "Abstract")],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
        max_attempts=1,
    )

    assert assessed == {}
    assert failure_class == V24_STRUCTURED_OUTPUT_FAILURE
    assert "response was not valid structured JSON" in reason
    event = next(
        item for item in _diagnostic_events(diagnostics.path)
        if item["event"] == "gemini_web_assessment_attempt"
    )
    assert event["exception_type"] == ""
    assert event["syntactically_valid_json"] is None

    bounded = metadata_builder(
        "", v24_screening.LocalAIOutputError("x" * 500)
    )
    assert len(bounded["exception_message"]) == 300
    assert bounded["response_empty"] is True
    assert bounded["syntactically_valid_json"] is None


def test_execute_batch_parses_once_without_extra_submission_retry_or_repair(
    tmp_path, monkeypatch,
):
    class InvalidBrowser:
        def __init__(self):
            self.calls = 0
            self.new_chats = 0

        def set_attempt_context(self, **_kwargs):
            return None

        def submit_prompt_and_get_response(self, _prompt):
            self.calls += 1
            return '{"items": []}'

        def start_new_job_chat(self):
            self.new_chats += 1

    real_parser = v24_screening.parse_structured_model_output
    parser_calls = []

    def counting_parser(*args, **kwargs):
        parser_calls.append(args[0])
        return real_parser(*args, **kwargs)

    monkeypatch.setattr(
        v24_screening, "parse_structured_model_output", counting_parser
    )
    browser = InvalidBrowser()
    diagnostics = V24Diagnostics(tmp_path / "single-parse.jsonl")
    assessed, reason, failure_class = v24_screening._execute_batch(
        browser,
        _protocol(),
        [V24Paper("0", "Title", "Abstract")],
        verification=False,
        flags=None,
        diagnostics=diagnostics,
        max_attempts=1,
    )

    assert assessed == {}
    assert "response was not valid structured JSON" in reason
    assert failure_class == V24_STRUCTURED_OUTPUT_FAILURE
    assert len(parser_calls) == 1
    assert browser.calls == 1
    assert browser.new_chats == 0


def test_browser_capture_fields_reach_assessment_event_and_missing_property_is_safe(
    tmp_path,
):
    class CaptureBrowser:
        def __init__(self, expose_capture):
            self.expose_capture = expose_capture

        def set_attempt_context(self, **_kwargs):
            return None

        def submit_prompt_and_get_response(self, _prompt):
            if self.expose_capture:
                self.last_response_capture_metadata = {
                    "response_return_reason": "complete_json_stable",
                    "response_complete_json_at_capture": True,
                    "response_generation_detected_at_capture": False,
                    "response_stable_duration_ms": 1000,
                    "response_utf8_bytes_at_capture": 13,
                    "response_selector_at_capture": "model-response",
                    "response_container_count_at_capture": 1,
                }
            return '{"items": []}'

    for expose_capture in (True, False):
        diagnostics = V24Diagnostics(
            tmp_path / f"capture-{expose_capture}.jsonl"
        )
        browser = CaptureBrowser(expose_capture)
        assessed, _, failure_class = v24_screening._execute_batch(
            browser,
            _protocol(),
            [V24Paper("0", "Title", "Abstract")],
            verification=False,
            flags=None,
            diagnostics=diagnostics,
            max_attempts=1,
        )
        assert assessed == {}
        assert failure_class == V24_STRUCTURED_OUTPUT_FAILURE
        event = next(
            item for item in _diagnostic_events(diagnostics.path)
            if item["event"] == "gemini_web_assessment_attempt"
        )
        if expose_capture:
            assert event["response_return_reason"] == "complete_json_stable"
            assert event["response_complete_json_at_capture"] is True
            assert event["response_selector_at_capture"] == "model-response"
        else:
            assert event["response_return_reason"] == ""
            assert event["response_complete_json_at_capture"] is None
