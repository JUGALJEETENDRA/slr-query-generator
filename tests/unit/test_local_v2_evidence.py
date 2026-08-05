from __future__ import annotations

from litsync_app.screening.local_v2.contracts import (
    CriterionAssessment,
    EvidenceCitation,
    ProtocolCriterion,
    ScreeningProtocolV2,
)
from litsync_app.screening.local_v2.evidence import (
    EVIDENCE_VALIDATION_VERSION,
    build_evidence_units,
    evidence_lookup,
    normalize_evidence_text,
    validate_assessment_evidence,
    validate_assessments_evidence,
    validate_citation,
)
from litsync_app.screening.local_v2.policy import derive_policy_decision


def _citation(
    evidence_id: str = "abstract_001",
    *,
    source: str = "abstract",
    quote: str = "The model screened titles and abstracts.",
) -> EvidenceCitation:
    return EvidenceCitation(evidence_id=evidence_id, source=source, quote=quote)


def _assessment(
    criterion_id: str,
    relation: str,
    *citations: EvidenceCitation,
) -> CriterionAssessment:
    return CriterionAssessment(
        criterion_id=criterion_id,
        relation=relation,
        rationale=f"The relation is {relation}.",
        evidence=list(citations),
    )


def _single_required_protocol(criterion_id: str) -> ScreeningProtocolV2:
    return ScreeningProtocolV2(
        research_question="Does the paper meet the required review criterion?",
        criteria=[
            ProtocolCriterion(
                id=criterion_id,
                role="REQUIRED_INCLUSION",
                description="The paper must satisfy the required relationship.",
                resolution_required=True,
            )
        ],
    ).with_identity()


def test_validation_version_is_frozen():
    assert EVIDENCE_VALIDATION_VERSION == "local-v2-evidence-v1"


def test_normalization_collapses_unicode_and_whitespace_representation_only():
    assert normalize_evidence_text("  Fullwidth Ａ\n\t test  ") == "Fullwidth A test"


def test_build_evidence_units_assigns_stable_source_scoped_ids():
    units = build_evidence_units(
        "A screening title",
        "First abstract sentence. Second abstract sentence!",
    )
    assert [(unit.evidence_id, unit.source, unit.text) for unit in units] == [
        ("title_001", "title", "A screening title"),
        ("abstract_001", "abstract", "First abstract sentence."),
        ("abstract_002", "abstract", "Second abstract sentence!"),
    ]


def test_evidence_units_preserve_exact_trimmed_source_text_and_offsets():
    abstract = "  First sentence.\n\n  Second sentence?  "
    units = build_evidence_units("", abstract)
    assert [unit.text for unit in units] == ["First sentence.", "Second sentence?"]
    for unit in units:
        assert abstract[unit.start : unit.end] == unit.text


def test_evidence_lookup_resolves_units_by_stable_id():
    lookup = evidence_lookup("Title", "One. Two.")
    assert lookup["title_001"].text == "Title"
    assert lookup["abstract_002"].text == "Two."


def test_exact_title_quote_is_valid_with_normalized_offsets():
    result = validate_citation(
        _citation(
            "title_001",
            source="title",
            quote="Automated abstract screening",
        ),
        title="A study of Automated\nabstract screening systems",
        abstract="",
    )
    assert result.valid
    assert result.match is not None
    assert result.match.normalized_quote == "Automated abstract screening"
    assert result.match.normalized_start == 11
    assert result.match.normalized_end == 39
    assert result.match.occurrence_count == 1
    assert result.issues == []


def test_exact_abstract_quote_is_valid_across_line_wrapping_inside_unit():
    result = validate_citation(
        _citation(quote="screened titles and abstracts"),
        title="Example",
        abstract="The model screened titles\n   and abstracts before retrieval.",
    )
    assert result.valid
    assert result.match is not None


def test_case_change_is_not_treated_as_an_exact_quote():
    result = validate_citation(
        _citation(quote="the model screened titles and abstracts."),
        title="",
        abstract="The model screened titles and abstracts.",
    )
    assert not result.valid
    assert result.match is None
    assert [issue.code for issue in result.issues] == ["QUOTE_NOT_FOUND_IN_UNIT"]


def test_punctuation_change_is_not_treated_as_an_exact_quote():
    result = validate_citation(
        _citation(quote="The model screened titles, and abstracts."),
        title="",
        abstract="The model screened titles and abstracts.",
    )
    assert not result.valid
    assert [issue.code for issue in result.issues] == ["QUOTE_NOT_FOUND_IN_UNIT"]


def test_unknown_evidence_id_is_rejected_even_when_quote_exists_in_source():
    result = validate_citation(
        _citation("abstract_999"),
        title="",
        abstract="The model screened titles and abstracts.",
    )
    assert not result.valid
    assert [issue.code for issue in result.issues] == ["EVIDENCE_ID_NOT_FOUND"]


def test_citation_source_must_match_the_resolved_evidence_unit():
    result = validate_citation(
        _citation(
            "title_001",
            source="abstract",
            quote="The model screened titles and abstracts.",
        ),
        title="The model screened titles and abstracts.",
        abstract="A separate abstract.",
    )
    assert not result.valid
    assert [issue.code for issue in result.issues] == ["SOURCE_MISMATCH"]


def test_quote_must_exist_inside_declared_unit_not_merely_elsewhere_in_source():
    result = validate_citation(
        _citation(
            "abstract_001",
            quote="The model screened titles and abstracts.",
        ),
        title="",
        abstract="Background sentence. The model screened titles and abstracts.",
    )
    assert not result.valid
    assert [issue.code for issue in result.issues] == ["QUOTE_NOT_FOUND_IN_UNIT"]


def test_missing_declared_source_is_reported_separately():
    result = validate_citation(
        _citation("abstract_001", quote="Evidence"),
        title="Evidence",
        abstract="   \n\t ",
    )
    assert not result.valid
    assert [issue.code for issue in result.issues] == ["SOURCE_TEXT_MISSING"]


def test_repeated_exact_quote_inside_unit_reports_occurrence_count():
    result = validate_citation(
        _citation(quote="Exact phrase"),
        title="",
        abstract="Exact phrase and Exact phrase remain in one unit",
    )
    assert result.valid
    assert result.match is not None
    assert result.match.occurrence_count == 2
    assert result.match.normalized_start == 0


def test_decisive_relation_is_preserved_when_exact_evidence_exists():
    assessment = _assessment(
        "screening_task",
        "DIRECT_SUPPORT",
        _citation(quote="The model screened titles and abstracts."),
    )
    result = validate_assessment_evidence(
        assessment,
        title="",
        abstract="The model screened titles and abstracts.",
    )
    assert result.sanitized_assessment.relation == "DIRECT_SUPPORT"
    assert [item.evidence_id for item in result.sanitized_assessment.evidence] == [
        "abstract_001"
    ]
    assert not result.safe_downgrade
    assert result.valid_evidence_ids == ["abstract_001"]


def test_fabricated_decisive_support_is_downgraded_to_missing_or_unclear():
    assessment = _assessment(
        "screening_task",
        "DIRECT_SUPPORT",
        _citation(quote="A fabricated sentence."),
    )
    result = validate_assessment_evidence(
        assessment,
        title="",
        abstract="The actual abstract says something else.",
    )
    assert result.sanitized_assessment.relation == "MISSING_OR_UNCLEAR"
    assert result.sanitized_assessment.evidence == []
    assert result.safe_downgrade
    assert result.valid_evidence_ids == []
    assert any(issue.code == "QUOTE_NOT_FOUND_IN_UNIT" for issue in result.issues)


def test_fabricated_decisive_contradiction_is_also_safely_downgraded():
    assessment = _assessment(
        "empirical_evidence",
        "DIRECT_CONTRADICTION",
        _citation(quote="No empirical evaluation was conducted."),
    )
    result = validate_assessment_evidence(
        assessment,
        title="",
        abstract="We report an empirical evaluation on three datasets.",
    )
    assert result.sanitized_assessment.relation == "MISSING_OR_UNCLEAR"
    assert result.safe_downgrade


def test_one_valid_quote_preserves_relation_and_invalid_quote_is_removed():
    assessment = _assessment(
        "screening_task",
        "DIRECT_SUPPORT",
        _citation(
            "abstract_001",
            quote="We screened titles and abstracts.",
        ),
        _citation(
            "abstract_002",
            quote="This sentence is fabricated.",
        ),
    )
    result = validate_assessment_evidence(
        assessment,
        title="",
        abstract="We screened titles and abstracts. A second real sentence.",
    )
    assert result.sanitized_assessment.relation == "DIRECT_SUPPORT"
    assert [item.evidence_id for item in result.sanitized_assessment.evidence] == [
        "abstract_001"
    ]
    assert result.valid_evidence_ids == ["abstract_001"]
    assert not result.safe_downgrade
    assert any(issue.evidence_id == "abstract_002" for issue in result.issues)


def test_invalid_evidence_on_missing_relation_is_removed_without_relation_change():
    assessment = _assessment(
        "review_context",
        "MISSING_OR_UNCLEAR",
        _citation(quote="Fabricated context."),
    )
    result = validate_assessment_evidence(
        assessment,
        title="",
        abstract="No review context is stated.",
    )
    assert result.sanitized_assessment.relation == "MISSING_OR_UNCLEAR"
    assert result.sanitized_assessment.evidence == []
    assert not result.safe_downgrade


def test_same_evidence_unit_can_support_multiple_criteria():
    citation = _citation(
        "abstract_001",
        quote="The language model screened titles and abstracts.",
    )
    first = _assessment("model_type", "DIRECT_SUPPORT", citation)
    second = _assessment("screening_task", "DIRECT_SUPPORT", citation)
    result = validate_assessments_evidence(
        [first, second],
        title="",
        abstract="The language model screened titles and abstracts.",
    )
    assert [item.relation for item in result.assessments] == [
        "DIRECT_SUPPORT",
        "DIRECT_SUPPORT",
    ]
    assert result.safe_downgrade_count == 0


def test_batch_preserves_assessment_and_evidence_order():
    assessments = [
        _assessment(
            "first",
            "DIRECT_SUPPORT",
            _citation("abstract_002", quote="Second sentence."),
            _citation("abstract_001", quote="First sentence."),
        ),
        _assessment("second", "MISSING_OR_UNCLEAR"),
    ]
    result = validate_assessments_evidence(
        assessments,
        title="",
        abstract="First sentence. Second sentence.",
    )
    assert [item.criterion_id for item in result.assessments] == ["first", "second"]
    assert [item.evidence_id for item in result.assessments[0].evidence] == [
        "abstract_002",
        "abstract_001",
    ]


def test_validation_does_not_mutate_input_assessment():
    assessment = _assessment(
        "screening_task",
        "DIRECT_SUPPORT",
        _citation(quote="Fabricated evidence."),
    )
    original_dump = assessment.model_dump()
    validate_assessment_evidence(
        assessment,
        title="",
        abstract="Actual evidence.",
    )
    assert assessment.model_dump() == original_dump
    assert assessment.relation == "DIRECT_SUPPORT"


def test_word_reordering_does_not_pass_exact_matching():
    result = validate_citation(
        _citation(quote="abstracts and titles screened model the"),
        title="",
        abstract="The model screened titles and abstracts.",
    )
    assert not result.valid


def test_batch_summary_counts_issues_and_invalid_citations():
    assessment = _assessment(
        "screening_task",
        "DIRECT_SUPPORT",
        _citation("abstract_001", quote="Exact evidence."),
        _citation("abstract_002", quote="Fabricated evidence."),
    )
    result = validate_assessments_evidence(
        [assessment],
        title="",
        abstract="Exact evidence. Real second sentence.",
    )
    assert result.validation_version == EVIDENCE_VALIDATION_VERSION
    assert result.issue_count == 1
    assert result.invalid_citation_count == 1
    assert not result.all_citations_valid


def test_empty_nondecisive_assessment_batch_is_vacuously_evidence_valid():
    result = validate_assessments_evidence(
        [_assessment("context", "MISSING_OR_UNCLEAR")],
        title="",
        abstract="",
    )
    assert result.all_citations_valid
    assert result.issue_count == 0
    assert result.invalid_citation_count == 0


def test_package_exports_public_evidence_api():
    import litsync_app.screening.local_v2 as local_v2

    assert local_v2.EVIDENCE_VALIDATION_VERSION == EVIDENCE_VALIDATION_VERSION
    assert local_v2.build_evidence_units is build_evidence_units
    assert local_v2.evidence_lookup is evidence_lookup
    assert local_v2.validate_citation is validate_citation
    assert local_v2.validate_assessment_evidence is validate_assessment_evidence
    assert local_v2.validate_assessments_evidence is validate_assessments_evidence


def test_invalid_decisive_evidence_cannot_drive_policy_rejection():
    assessment = _assessment(
        "empirical_evidence",
        "DIRECT_CONTRADICTION",
        _citation(quote="No empirical evaluation was conducted."),
    )
    validated = validate_assessments_evidence(
        [assessment],
        title="",
        abstract="We report an empirical evaluation on three datasets.",
    )
    decision = derive_policy_decision(
        _single_required_protocol("empirical_evidence"),
        validated.assessments,
    )
    assert validated.safe_downgrade_count == 1
    assert decision.decision == "MAYBE"
    assert decision.decisive_criterion_ids == []


def test_valid_decisive_evidence_can_reach_policy_rejection():
    assessment = _assessment(
        "empirical_evidence",
        "DIRECT_CONTRADICTION",
        _citation(quote="No empirical evaluation was conducted."),
    )
    validated = validate_assessments_evidence(
        [assessment],
        title="",
        abstract="No empirical evaluation was conducted.",
    )
    decision = derive_policy_decision(
        _single_required_protocol("empirical_evidence"),
        validated.assessments,
    )
    assert validated.safe_downgrade_count == 0
    assert decision.decision == "REJECT"
    assert decision.decisive_criterion_ids == ["empirical_evidence"]
