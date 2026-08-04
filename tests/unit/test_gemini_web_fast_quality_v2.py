from __future__ import annotations

import json

import pandas as pd

from litsync_app.integrations.gemini_web_fast_prompt import (
    ARCHITECTURE_VERSION,
    PROMPT_VERSION,
    CriterionGroup,
    RubricCriterion,
    ScreeningRubric,
    batch_prompt,
    fallback_rubric,
)
from litsync_app.integrations.gemini_web_fast_screening import (
    _clean_cell,
    _merge,
    _row,
    _validate_batch,
    _validate_evidence,
)


def _paper() -> dict[str, str]:
    return {
        "paper_id": "0",
        "title": "Privacy-preserving federated learning for hospitals",
        "abstract": (
            "We evaluate a federated learning framework with differential privacy "
            "across five hospitals and report accuracy and privacy results."
        ),
    }


def _rubric_any() -> ScreeningRubric:
    return ScreeningRubric(
        review_summary="Primary evaluated federated-learning studies.",
        inclusion_criteria=[
            RubricCriterion(
                criterion_id="I1",
                text="The study evaluates privacy.",
                role="ALTERNATIVE",
                group_id="G1",
                source_text="privacy",
            ),
            RubricCriterion(
                criterion_id="I2",
                text="The study evaluates robustness.",
                role="ALTERNATIVE",
                group_id="G1",
                source_text="robustness",
            ),
            RubricCriterion(
                criterion_id="I3",
                text="The study reports quantitative evaluation.",
                role="MANDATORY",
                source_text="quantitative evaluation",
            ),
        ],
        exclusion_criteria=[
            RubricCriterion(
                criterion_id="E1",
                text="Exclude reviews and surveys.",
                role="EXCLUSION",
                source_text="Exclude reviews and surveys.",
            )
        ],
        criterion_groups=[
            CriterionGroup(
                group_id="G1",
                operator="ANY",
                member_ids=["I1", "I2"],
                minimum_required=1,
                description="At least one relevant technical objective.",
            )
        ],
        original_logic_preserved=True,
    )


def _rubric_optional_domains() -> ScreeningRubric:
    return ScreeningRubric(
        review_summary="Eligible studies in any listed application domain.",
        inclusion_criteria=[
            RubricCriterion(
                criterion_id="I1",
                text="The study is an evaluated primary study.",
                role="MANDATORY",
            ),
            RubricCriterion(
                criterion_id="I2",
                text="Healthcare is an example domain.",
                role="OPTIONAL",
            ),
            RubricCriterion(
                criterion_id="I3",
                text="Finance is an example domain.",
                role="OPTIONAL",
            ),
            RubricCriterion(
                criterion_id="I4",
                text="Transportation is an example domain.",
                role="OPTIONAL",
            ),
        ],
        exclusion_criteria=[],
        original_logic_preserved=True,
    )


def _item(
    *,
    decision: str,
    inclusion: dict[str, str],
    exclusion: dict[str, str] | None = None,
    evidence_quote: str = "",
    confidence: float = 0.95,
) -> str:
    return json.dumps(
        {
            "items": [
                {
                    "paper_id": "0",
                    "decision": decision,
                    "confidence": confidence,
                    "reason": f"{decision} based on supplied title and abstract.",
                    "evidence_quote": evidence_quote,
                    "inclusion_assessments": [
                        {"criterion_id": key, "status": value}
                        for key, value in inclusion.items()
                    ],
                    "exclusion_assessments": [
                        {"criterion_id": key, "status": value}
                        for key, value in (exclusion or {}).items()
                    ],
                    "risk_flags": [],
                }
            ]
        }
    )


def _assessment(
    decision: str,
    *,
    evidence_valid: bool = True,
    semantic_valid: bool = True,
    structural_valid: bool = True,
    technical_failure: bool = False,
    failure_class: str = "",
    confidence: float = 0.9,
    evidence_quote: str = (
        "We evaluate a federated learning framework with differential privacy"
    ),
    evidence_source: str = "abstract",
) -> dict:
    return {
        "paper_id": "0",
        "decision": decision if semantic_valid else "MAYBE",
        "model_decision": decision,
        "confidence": confidence,
        "reason": f"{decision} assessment",
        "evidence_quote": evidence_quote,
        "inclusion_assessments": [],
        "exclusion_assessments": [],
        "risk_flags": [],
        "validation_errors": [],
        "validation_warnings": [],
        "validation_status": (
            "valid"
            if structural_valid and semantic_valid and evidence_valid
            else "semantic_valid_evidence_invalid"
        ),
        "failure_class": failure_class,
        "structural_valid": structural_valid,
        "semantic_valid": semantic_valid,
        "evidence_valid": evidence_valid,
        "evidence_source": evidence_source if evidence_valid else "none",
        "evidence_failure_reason": "" if evidence_valid else "quote_not_found_in_source",
        "technical_failure": technical_failure,
        "valid": structural_valid and semantic_valid and evidence_valid,
    }


def test_prompt_version_is_v3_and_architecture_remains_focused_v1():
    assert PROMPT_VERSION == "gemini-web-fast-prompt-v3"
    assert ARCHITECTURE_VERSION == "gemini-web-fast-v1"


def test_fallback_rubric_preserves_complete_inclusion_block():
    original = (
        "The study must evaluate at least one of privacy, robustness, "
        "communication efficiency, or predictive performance."
    )
    rubric = fallback_rubric(original, "Exclude reviews.")
    assert len(rubric.inclusion_criteria) == 1
    assert rubric.inclusion_criteria[0].text == original
    assert rubric.inclusion_criteria[0].role == "UNRESOLVED"
    assert rubric.criterion_groups == []
    assert rubric.original_logic_preserved is False


def test_any_group_does_not_turn_unmet_alternatives_into_mandatory_failures():
    raw = _item(
        decision="KEEP",
        inclusion={"I1": "MET", "I2": "NOT_MET", "I3": "MET"},
        exclusion={"E1": "NOT_MET"},
        evidence_quote=(
            "We evaluate a federated learning framework with differential privacy"
        ),
    )
    accepted, unresolved, failures = _validate_batch(raw, [_paper()], _rubric_any())
    assessment = accepted["0"]

    assert not unresolved
    assert not failures
    assert assessment["model_decision"] == "KEEP"
    assert assessment["decision"] == "KEEP"
    assert assessment["semantic_valid"] is True
    assert assessment["evidence_valid"] is True
    assert assessment["valid"] is True
    assert "keep_criterion_contradiction" not in assessment["validation_errors"]
    assert "keep_failed_criterion_group" not in assessment["validation_errors"]


def test_failed_any_group_prevents_direct_keep():
    raw = _item(
        decision="KEEP",
        inclusion={"I1": "NOT_MET", "I2": "NOT_MET", "I3": "MET"},
        exclusion={"E1": "NOT_MET"},
        evidence_quote=(
            "We evaluate a federated learning framework with differential privacy"
        ),
    )
    accepted, _, _ = _validate_batch(raw, [_paper()], _rubric_any())
    assessment = accepted["0"]

    assert assessment["decision"] == "MAYBE"
    assert assessment["semantic_valid"] is False
    assert "keep_failed_criterion_group" in assessment["validation_errors"]


def test_explicit_mandatory_failure_prevents_direct_keep():
    raw = _item(
        decision="KEEP",
        inclusion={"I1": "MET", "I2": "NOT_MET", "I3": "NOT_MET"},
        exclusion={"E1": "NOT_MET"},
        evidence_quote=(
            "We evaluate a federated learning framework with differential privacy"
        ),
    )
    accepted, _, _ = _validate_batch(raw, [_paper()], _rubric_any())
    assessment = accepted["0"]

    assert assessment["decision"] == "MAYBE"
    assert assessment["semantic_valid"] is False
    assert "keep_failed_mandatory_criterion" in assessment["validation_errors"]


def test_met_exclusion_prevents_direct_keep():
    raw = _item(
        decision="KEEP",
        inclusion={"I1": "MET", "I2": "NOT_MET", "I3": "MET"},
        exclusion={"E1": "MET"},
        evidence_quote=(
            "We evaluate a federated learning framework with differential privacy"
        ),
    )
    accepted, _, _ = _validate_batch(raw, [_paper()], _rubric_any())
    assessment = accepted["0"]

    assert assessment["decision"] == "MAYBE"
    assert assessment["semantic_valid"] is False
    assert "keep_met_exclusion_criterion" in assessment["validation_errors"]


def test_unmet_optional_examples_do_not_invalidate_keep():
    rubric = _rubric_optional_domains()
    raw = _item(
        decision="KEEP",
        inclusion={
            "I1": "MET",
            "I2": "MET",
            "I3": "NOT_MET",
            "I4": "NOT_MET",
        },
        evidence_quote=(
            "We evaluate a federated learning framework with differential privacy"
        ),
    )
    accepted, _, _ = _validate_batch(raw, [_paper()], rubric)
    assessment = accepted["0"]

    assert assessment["decision"] == "KEEP"
    assert assessment["semantic_valid"] is True
    assert assessment["valid"] is True


def test_exact_evidence_accepts_contiguous_source_span():
    valid, source, failure = _validate_evidence(
        "federated learning framework with differential privacy",
        title=_paper()["title"],
        abstract=_paper()["abstract"],
    )
    assert (valid, source, failure) == (True, "abstract", "")


def test_exact_evidence_allows_quote_wrappers_and_whitespace_normalization():
    valid, source, failure = _validate_evidence(
        '“federated learning framework with differential privacy”',
        title=_paper()["title"],
        abstract=(
            "We evaluate a federated learning framework\n"
            "with differential privacy across five hospitals."
        ),
    )
    assert (valid, source, failure) == (True, "abstract", "")


def test_exact_evidence_rejects_ascii_and_unicode_ellipses():
    assert _validate_evidence(
        "federated learning...privacy",
        title=_paper()["title"],
        abstract=_paper()["abstract"],
    ) == (False, "none", "ellipsis_not_allowed")

    assert _validate_evidence(
        "federated learning…privacy",
        title=_paper()["title"],
        abstract=_paper()["abstract"],
    ) == (False, "none", "ellipsis_not_allowed")


def test_exact_evidence_rejects_joined_fragments_and_paraphrases():
    valid, source, failure = _validate_evidence(
        "federated learning across hospitals privacy results",
        title=_paper()["title"],
        abstract=_paper()["abstract"],
    )
    assert valid is False
    assert source == "none"
    assert failure == "quote_not_found_in_source"


def test_primary_bad_evidence_and_valid_agreeing_verifier_recovers_keep():
    primary = _assessment("KEEP", evidence_valid=False, evidence_quote="invented")
    verifier = _assessment("KEEP", evidence_valid=True, confidence=0.88)

    decision, confidence, reason, agreement = _merge(primary, verifier)

    assert decision == "KEEP"
    assert confidence == 0.88
    assert reason == "KEEP assessment"
    assert agreement == "agreement_recovered_by_verifier_evidence"


def test_primary_bad_evidence_and_valid_agreeing_verifier_recovers_reject():
    primary = _assessment("REJECT", evidence_valid=False, evidence_quote="invented")
    verifier = _assessment(
        "REJECT",
        evidence_valid=True,
        evidence_quote="report accuracy and privacy results",
        confidence=0.86,
    )

    decision, confidence, _, agreement = _merge(primary, verifier)

    assert decision == "REJECT"
    assert confidence == 0.86
    assert agreement == "agreement_recovered_by_verifier_evidence"


def test_disagreement_remains_maybe():
    primary = _assessment("KEEP")
    verifier = _assessment("REJECT")

    decision, confidence, _, agreement = _merge(primary, verifier)

    assert decision == "MAYBE"
    assert confidence == 0.0
    assert agreement == "disagreement"


def test_technical_primary_plus_one_valid_verifier_cannot_become_definitive():
    primary = _assessment(
        "KEEP",
        evidence_valid=False,
        semantic_valid=False,
        structural_valid=False,
        technical_failure=True,
        failure_class="schema_invalid",
    )
    verifier = _assessment("KEEP")

    decision, confidence, _, agreement = _merge(primary, verifier)

    assert decision == "MAYBE"
    assert confidence == 0.0
    assert agreement == "primary_validation_failed"


def test_final_row_uses_valid_verifier_evidence_when_recovering():
    paper = {
        "paper_id": "0",
        "order": 0,
        "title": _paper()["title"],
        "abstract": _paper()["abstract"],
        "original": {
            "Title": _paper()["title"],
            "Abstract": _paper()["abstract"],
        },
    }
    primary = _assessment("KEEP", evidence_valid=False, evidence_quote="invented")
    verifier = _assessment(
        "KEEP",
        evidence_valid=True,
        evidence_quote=(
            "We evaluate a federated learning framework with differential privacy"
        ),
        evidence_source="abstract",
    )

    row = _row(
        paper,
        primary,
        verifier,
        origin="fresh_verification",
        protocol_id="protocol-test",
    )

    assert row["Decision"] == "KEEP"
    assert row["Validation_Status"] == "validated"
    assert row["Failure_Class"] == ""
    assert row["Agreement_Status"] == "agreement_recovered_by_verifier_evidence"
    assert row["Evidence_Quote"] == verifier["evidence_quote"]
    assert row["Final_Evidence_Source"] == "verifier"
    assert row["Final_Evidence_Location"] == "abstract"
    assert row["Primary_Evidence_Valid"] is False
    assert row["Verifier_Evidence_Valid"] is True


def test_primary_and_verifier_prompts_apply_the_same_exact_evidence_rule():
    rubric = fallback_rubric(
        "Include relevant evaluated studies.",
        "Exclude reviews.",
    )
    kwargs = {
        "question": "Which methods improve the system?",
        "context": "Primary evaluated studies.",
        "inclusion": "Include relevant evaluated studies.",
        "exclusion": "Exclude reviews.",
        "rubric": rubric,
        "papers": [_paper()],
    }

    primary_prompt = batch_prompt(**kwargs, verification=False)
    verifier_prompt = batch_prompt(**kwargs, verification=True)

    required_phrases = [
        "exactly one continuous, nonempty span verbatim",
        "Never:",
        "insert three dots",
        'insert "..."',
        'Unicode ellipsis character "…"',
        "combine separate fragments",
    ]

    for phrase in required_phrases:
        assert phrase in primary_prompt
        assert phrase in verifier_prompt

    assert "prediction-blind independent review" not in primary_prompt
    assert "prediction-blind independent review" in verifier_prompt


def test_clean_cell_normalizes_real_nulls_but_preserves_literal_nan_text():
    assert _clean_cell(None) == ""
    assert _clean_cell(float("nan")) == ""
    assert _clean_cell(pd.NA) == ""
    assert _clean_cell("") == ""
    assert _clean_cell("   ") == ""
    assert _clean_cell("nan") == "nan"
    assert _clean_cell("  actual text  ") == "actual text"
