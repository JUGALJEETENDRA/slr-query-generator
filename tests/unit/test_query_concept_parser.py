from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from litsync_app.query import generator as query_module
from litsync_app.query.generator import (
    GeminiDirectConceptProposal,
    MAX_COMPILED_REQUIRED_GROUPS,
    MAX_COORDINATION_CANDIDATES,
    MAX_PARSER_CLAUSES,
    MAX_PARSER_CONCEPTS,
    MAX_QUESTION_CODEPOINTS,
    MAX_SOURCE_SPANS_PER_GROUP,
    MAX_TERMS_PER_GROUP,
    MAX_UNCERTAINTIES,
    StructuredQueryDraft,
    _deterministic_seed,
    generate_query_bundle,
)
from litsync_app.screening.local.engine import GenerationResult
from litsync_app.screening.local.hardware import HardwareSnapshot, RuntimeProfile


TEST_PROFILE = RuntimeProfile(
    requested_tier="auto",
    resolved_tier="balanced",
    resource_profile="balanced",
    fast_model="qwen3.5:4b",
    strong_model="qwen3.5:4b",
    num_ctx=4096,
    keep_alive="30m",
    concurrency=1,
    memory_reserve_ratio=0.2,
    downgrade_reasons=(),
    hardware=HardwareSnapshot(
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        cpu_cores=8,
        platform="Test",
        gpu_name="",
        gpu_vram_gb=0.0,
        installed_models={"qwen3.5:4b": 1},
    ),
    calibration={},
)


class ProposalEngine:
    def __init__(self, value=None):
        self.calls = []
        self.value = value or {}

    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        self.calls.append((model, prompt, schema, timeout_seconds))
        return GenerationResult(value=self.value, model=model, elapsed_seconds=0.01)


class MalformedProposalEngine:
    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        return GenerationResult(
            value={"unexpected": "shape"}, model=model, elapsed_seconds=0.01,
        )


def _gemini_bundle(question: str, *, required_groups=None, optional_groups=None):
    assert required_groups is None and optional_groups is None
    seed = _deterministic_seed(question)
    engine = ProposalEngine(seed.model_dump(mode="json"))
    bundle = generate_query_bundle(
        question, processing_engine="local", engine=engine, profile=TEST_PROFILE,
    )
    return bundle, engine


def test_question_length_is_validated_before_decomposition_without_truncation():
    accepted = "How is alpha used for beta?" + " " * (
        MAX_QUESTION_CODEPOINTS - len("How is alpha used for beta?")
    )
    assert len(accepted) == 4000
    assert _deterministic_seed(accepted).groups
    rejected = accepted + "x"
    with pytest.raises(ValueError, match="4,000 Unicode code-point limit"):
        _deterministic_seed(rejected)
    with pytest.raises(ValueError, match="4,000 Unicode code-point limit"):
        generate_query_bundle(rejected, engine=ProposalEngine())


def test_anchored_framing_produces_clean_concepts_and_preserves_population():
    question = (
        "How effective is mindfulness-based stress reduction in reducing anxiety "
        "and improving sleep quality among university students?"
    )
    draft = _deterministic_seed(question)
    assert [group.role for group in draft.groups] == [
        "intervention_or_method", "outcome", "population",
    ]
    assert draft.groups[0].canonical_text == "mindfulness-based stress reduction"
    assert draft.groups[1].canonical_text == "anxiety OR sleep quality"
    assert draft.groups[1].coordination == "alternatives"
    assert draft.groups[2].canonical_text == "university students"
    assert all("effective is" not in term.casefold() for group in draft.groups for term in group.terms)


def test_offsets_are_original_zero_based_end_exclusive_for_unicode_and_punctuation():
    question = "How effective is café-based learning in improving focus, among café students?"
    draft = _deterministic_seed(question)
    for group in draft.groups:
        for offset, span in zip(group.source_offsets, group.source_spans):
            assert question[offset["start"]:offset["end"]] == span == offset["text"]
    focus = next(group for group in draft.groups if group.role == "outcome")
    assert focus.canonical_text == "improving focus"
    assert focus.confidence == "low"
    assert focus.canonical_text in draft.uncertain_terms


def test_repeated_text_offsets_identify_the_linked_occurrence():
    question = "How does alpha affect alpha in alpha systems?"
    draft = _deterministic_seed(question)
    offsets = [
        offset for group in draft.groups for offset in group.source_offsets
        if "alpha" in offset["text"]
    ]
    assert len({(item["start"], item["end"]) for item in offsets}) >= 2
    assert all(question[item["start"]:item["end"]] == item["text"] for item in offsets)


def test_same_role_and_is_recall_safe_or_while_or_is_alternatives():
    and_question = "How does alpha improve writing quality and student engagement among learners?"
    and_outcome = next(
        group for group in _deterministic_seed(and_question).groups if group.role == "outcome"
    )
    assert and_outcome.coordination == "alternatives"
    assert and_outcome.canonical_text == "writing quality OR student engagement"
    or_question = "How does alpha affect anxiety or depression among learners?"
    or_outcome = next(
        group for group in _deterministic_seed(or_question).groups if group.role == "outcome"
    )
    assert or_outcome.coordination == "alternatives"
    assert or_outcome.canonical_text == "anxiety OR depression"


def test_ambiguous_and_shared_head_coordination_remain_unsplit():
    lexicalized = _deterministic_seed("How does research and development affect productivity?")
    subject = lexicalized.groups[0]
    assert subject.coordination == "ambiguous"
    assert subject.source_spans == ["research and development"]
    combined = _deterministic_seed("What is the combined effect of alpha and beta on gamma?")
    assert combined.groups[0].coordination == "shared_head"


def test_repeated_complete_clauses_are_explicitly_co_required():
    draft = _deterministic_seed(
        "How does alpha training improve accuracy, and beta tutoring reduce attrition?"
    )
    assert [group.canonical_text for group in draft.groups] == [
        "alpha training", "accuracy", "beta tutoring", "attrition",
    ]
    assert all(group.coordination == "co_required" for group in draft.groups)
    assert all(group.search_role == "required" for group in draft.groups)


def test_explicit_temporal_attachment_is_screening_only():
    draft = _deterministic_seed("How does alpha improve beta during 2020 and 2021?")
    temporal = draft.groups[-1]
    assert temporal.search_role == "screening_only"
    assert temporal.status_reason == "explicit_temporal_restriction"
    assert temporal.compiled is False


REGRESSION_QUESTION = (
    "Among small and medium-sized enterprises, how does remote work compared with "
    "office-based work affect employee productivity and job satisfaction during "
    "organizational change?"
)


def test_comparison_regression_recovers_clean_parser_owned_structure():
    draft = _deterministic_seed(REGRESSION_QUESTION)
    assert [group.role for group in draft.groups] == [
        "intervention_or_method", "comparison", "outcome", "population", "limitation",
    ]
    assert [group.search_role for group in draft.groups] == [
        "required", "required", "required", "required", "optional",
    ]
    assert [group.canonical_text for group in draft.groups] == [
        "remote work",
        "office-based work",
        "employee productivity OR job satisfaction",
        "small and medium-sized enterprises",
        "organizational change",
    ]
    population = draft.groups[3]
    assert population.source_spans == ["Among small and medium-sized enterprises"]
    assert population.coordination == "shared_head"
    assert all(
        REGRESSION_QUESTION[item["start"]:item["end"]] == item["text"]
        for group in draft.groups for item in group.source_offsets
    )
    assert all(
        "how does" not in term.casefold() and " affect " not in f" {term.casefold()} "
        for group in draft.groups for term in group.terms
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "Among coastal arrays, how does adaptive sampling versus fixed sampling "
            "influence detection accuracy and response latency during field deployment?",
            [
                "adaptive sampling", "fixed sampling",
                "detection accuracy OR response latency", "coastal arrays", "field deployment",
            ],
        ),
        (
            "Within learning cohorts, how does guided practice relative to independent study "
            "improve retention and transfer during 2022?",
            [
                "guided practice", "independent study", "retention OR transfer",
                "learning cohorts", "2022",
            ],
        ),
    ],
)
def test_comparison_grammar_generalizes_across_unrelated_domains(question, expected):
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == expected
    assert [group.role for group in draft.groups[:3]] == [
        "intervention_or_method", "comparison", "outcome",
    ]
    assert draft.groups[2].coordination == "alternatives"
    assert all(
        question[item["start"]:item["end"]] == item["text"]
        for group in draft.groups for item in group.source_offsets
    )


@pytest.mark.parametrize(
    "marker", ["compared with", "compared to", "versus", "rather than", "relative to"],
)
def test_comparison_markers_share_the_same_structural_decomposition(marker):
    question = f"How does alpha method {marker} beta method affect gamma rate and delta cost?"
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == [
        "alpha method", "beta method", "gamma rate OR delta cost",
    ]
    assert [group.role for group in draft.groups] == [
        "intervention_or_method", "comparison", "outcome",
    ]


def test_trailing_population_is_separate_from_comparator_and_outcomes():
    question = (
        "How does alpha method compared with beta method affect gamma rate "
        "and delta cost among epsilon groups?"
    )
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == [
        "alpha method", "beta method", "gamma rate OR delta cost", "epsilon groups",
    ]
    assert [group.role for group in draft.groups] == [
        "intervention_or_method", "comparison", "outcome", "population",
    ]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How does alpha compared with beta alter access and cost?", ["alpha", "beta", "access OR cost"]),
        ("How does alpha versus beta yield precision and recall?", ["alpha", "beta", "precision OR recall"]),
        ("How does alpha relative to beta lead to speed and accuracy?", ["alpha", "beta", "speed OR accuracy"]),
    ],
)
def test_unknown_comparison_predicate_uses_narrow_parallel_frame(question, expected):
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == expected
    assert [group.role for group in draft.groups] == [
        "intervention_or_method", "comparison", "outcome",
    ]
    assert draft.groups[2].coordination == "alternatives"
    assert all(group.compiled for group in draft.groups)


def test_shared_head_shape_is_preserved_without_a_known_head_registry():
    question = "Among compact and high-capacity cells, how does alpha affect beta?"
    draft = _deterministic_seed(question)
    attachment = next(group for group in draft.groups if group.role == "population")
    assert attachment.canonical_text == "compact and high-capacity cells"
    assert attachment.coordination == "shared_head"
    assert attachment.source_spans == ["Among compact and high-capacity cells"]


def test_inconclusive_leading_attachment_is_preserved_as_required_other():
    question = "Within alpha, how does beta affect gamma?"
    draft = _deterministic_seed(question)
    attachment = next(group for group in draft.groups if group.source_spans == ["Within alpha"])
    assert attachment.role == "other"
    assert attachment.search_role == "required"
    assert attachment.coordination == "ambiguous"
    assert attachment.canonical_text == "alpha"
    assert draft.uncertain_terms


def test_incomplete_comparison_frame_remains_one_ambiguous_span():
    question = "How does alpha method compared with beta method?"
    draft = _deterministic_seed(question)
    assert len(draft.groups) == 1
    group = draft.groups[0]
    assert group.canonical_text == "alpha method compared with beta method"
    assert group.source_spans == ["alpha method compared with beta method"]
    assert group.coordination == "ambiguous"
    assert group.status_reason == "incomplete_comparison_frame"


def test_malformed_gemini_output_does_not_use_parser_fallback():
    with pytest.raises(ValueError):
        generate_query_bundle(
            REGRESSION_QUESTION,
            processing_engine="gemini_web_v24",
            engine=MalformedProposalEngine(),
        )


def test_production_parser_rules_contain_no_regression_specific_vocabulary():
    source = inspect.getsource(query_module).casefold()
    forbidden = (
        REGRESSION_QUESTION.casefold(),
        "remote work",
        "office-based work",
        "employee productivity",
        "job satisfaction",
        "small and medium-sized enterprises",
        "organizational change",
    )
    assert all(value not in source for value in forbidden)


def test_hardening_examples_do_not_enter_production_rules_or_predicate_family():
    source = inspect.getsource(query_module).casefold()
    distinctive_phrases = (
        "learning to rank", "design for manufacturability", "quality by design",
        "access to care", "models with uncertainty", "learning from demonstrations",
        "clinical decision support systems", "policy impact assessments",
        "organizational change management",
        "large language models help automate systematic literature reviews",
        "during treatment and after discharge",
        "before baseline, during treatment, and after follow-up",
        "within hospitals and across regions",
        "supported decision making",
    )
    assert all(value not in source for value in distinctive_phrases)
    predicate_pattern = query_module.PREDICATE_PATTERN.pattern.casefold()
    newly_tested_unknown_verbs = ("alter", "yield", "relate", "lead", "perform", "lower")
    assert all(value not in predicate_pattern for value in newly_tested_unknown_verbs)


def test_preposition_role_uses_attachment_not_the_token_alone():
    chained = _deterministic_seed("How does alpha affect beta in older adults in hospitals?")
    assert [(group.role, group.source_spans[0]) for group in chained.groups][-2:] == [
        ("population", "older adults"), ("domain", "hospitals"),
    ]
    ambiguous = _deterministic_seed("How is alpha used in depression?")
    tail = ambiguous.groups[-1]
    assert tail.role == "other"
    assert tail.search_role == "required"
    assert tail.confidence == "low"


def test_explicit_original_acronym_enters_balanced_with_exact_provenance():
    question = "How is artificial intelligence (AI) used for defect detection?"
    bundle, _ = _gemini_bundle(question)
    technology = bundle.concepts["groups"][0]
    assert technology["source_spans"] == ["artificial intelligence", "AI"]
    assert '"AI"' in bundle.google_scholar
    detail = next(item for item in bundle.concepts["term_details"] if item["term"] == "AI")
    assert detail["source"] == "explicit_original_acronym"
    offset = detail["source_offsets"][0]
    assert question[offset["start"]:offset["end"]] == "AI"


@pytest.mark.parametrize(
    "literal",
    [
        "university students", "daily news", "bird species", "time series",
        "popular movies", "disease status", "camera lens", "research corpus",
        "statistical bias",
    ],
)
def test_automatic_number_morphology_is_disabled(literal):
    draft = _deterministic_seed(f"How does {literal} affect alpha quality?")
    subject = draft.groups[0]
    assert subject.canonical_text == literal
    assert subject.terms == [literal]


def test_only_source_present_hyphens_may_be_removed():
    hyphenated = _deterministic_seed("How does office-based work affect sleep quality?")
    assert hyphenated.groups[0].terms == ["office-based work", "office based work"]
    plain_terms = {term for group in hyphenated.groups for term in group.terms}
    assert "sleep-quality" not in plain_terms
    students = _deterministic_seed("How does training affect university students?")
    assert "university-students" not in {
        term for group in students.groups for term in group.terms
    }


def test_eight_required_groups_compile_and_omitted_gemini_groups_survive_in_order():
    question = (
        "How does alpha affect beta compared with gamma for delta among epsilon "
        "in zeta with eta under theta by iota?"
    )
    seed = _deterministic_seed(question)
    selected = [group for group in seed.groups if group.compiled]
    assert len(selected) == 8
    bundle, engine = _gemini_bundle(question)
    assert len(engine.calls) == 1
    assert engine.calls[0][2] is StructuredQueryDraft
    for version in ("balanced", "high_recall"):
        query = bundle.query_versions[version]["google_scholar"]
        assert query.count(" AND ") == 7
        blocks = query.split(" AND ")
        assert all(group.terms[0] in block for group, block in zip(selected, blocks))
    assert bundle.concepts["concept_counts"]["compiled_required"] == 8
    assert any(
        item["code"] == "more_than_five_required_groups"
        for item in bundle.concepts["parser_warnings"]
    )


def test_required_overflow_selection_is_deterministic_and_warning_only():
    question = (
        "How does alpha affect beta compared with gamma for delta among epsilon "
        "in zeta with eta under theta across kappa by iota?"
    )
    first = _deterministic_seed(question)
    second = _deterministic_seed(question)
    selected_first = [group.label for group in first.groups if group.compiled]
    selected_second = [group.label for group in second.groups if group.compiled]
    assert selected_first == selected_second
    assert len(selected_first) == MAX_COMPILED_REQUIRED_GROUPS
    overflow = [group for group in first.groups if group.search_role == "required" and not group.compiled]
    assert overflow and all(group.search_role == "required" for group in overflow)
    bundle, _ = _gemini_bundle(question)
    assert bundle.concepts["uncompiled_required_groups"]
    assert any(
        item["code"] == "required_group_compile_limit_exceeded"
        for item in bundle.concepts["parser_warnings"]
    )
    assert bundle.google_scholar.count(" AND ") == 7


@pytest.mark.parametrize(
    ("question", "subject", "outcome"),
    [
        ("How do support vector machines improve classification accuracy?", "support vector machines", "classification accuracy"),
        ("How do clinical decision support systems improve diagnostic accuracy?", "clinical decision support systems", "diagnostic accuracy"),
        ("How do policy impact assessments improve decision quality?", "policy impact assessments", "decision quality"),
        ("How does organizational change management affect employee engagement?", "organizational change management", "employee engagement"),
        ("How does training improve support quality?", "training", "support quality"),
        ("How does treatment improve impact awareness?", "treatment", "impact awareness"),
        ("How do methods support prediction?", "methods", "prediction"),
        ("How do systems detect changes?", "systems", "changes"),
    ],
)
def test_predicate_candidates_preserve_nominal_family_words(question, subject, outcome):
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == [subject, outcome]
    assert [group.role for group in draft.groups] == ["intervention_or_method", "outcome"]
    assert all(group.search_role == "required" and group.compiled for group in draft.groups)
    assert all(
        question[item["start"]:item["end"]] == item["text"]
        for group in draft.groups for item in group.source_offsets
    )


@pytest.mark.parametrize(
    ("question", "subject", "outcome"),
    [
        ("How does learning to rank improve search relevance?", "learning to rank", "search relevance"),
        ("How does design for manufacturability reduce production cost?", "design for manufacturability", "production cost"),
        ("How does quality by design improve manufacturing consistency?", "quality by design", "manufacturing consistency"),
        ("How does access to care affect health outcomes?", "access to care", "health outcomes"),
        ("How do models with uncertainty improve calibration?", "models with uncertainty", "calibration"),
        ("How does learning from demonstrations improve control performance?", "learning from demonstrations", "control performance"),
    ],
)
def test_fixed_subject_ranges_protect_lexicalized_prepositions(question, subject, outcome):
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == [subject, outcome]
    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == f'("{subject}") AND ("{outcome}")'
    assert all(group.compiled for group in draft.groups)


@pytest.mark.parametrize(
    "question",
    [
        "How does alpha compare with beta quality and safety?",
        "How does alpha compared with beta during deployment?",
        "How does alpha compare with beta and gamma?",
        "How does alpha compared with beta produce?",
        "How does a phrase compared with form remain incomplete?",
    ],
)
def test_incomplete_comparison_shapes_remain_ambiguous(question):
    draft = _deterministic_seed(question)
    assert len(draft.groups) == 1
    group = draft.groups[0]
    assert group.role == "other"
    assert group.search_role == "required"
    assert group.coordination == "ambiguous"
    assert group.compiled is True
    assert group.canonical_text in draft.uncertain_terms


@pytest.mark.parametrize(
    ("population", "canonical", "coordination"),
    [
        ("doctors and nurses", "doctors OR nurses", "alternatives"),
        ("teachers and school administrators", "teachers OR school administrators", "alternatives"),
        ("children and adolescents", "children OR adolescents", "alternatives"),
        ("urban and rural hospitals", "urban and rural hospitals", "shared_head"),
        ("small and medium-sized enterprises", "small and medium-sized enterprises", "shared_head"),
    ],
)
def test_population_coordination_uses_structural_shape(population, canonical, coordination):
    question = f"Among {population}, how does alpha affect beta?"
    draft = _deterministic_seed(question)
    group = next(item for item in draft.groups if item.role == "population")
    assert group.canonical_text == canonical
    assert group.coordination == coordination
    assert group.search_role == "required" and group.compiled
    offset = group.source_offsets[0]
    assert question[offset["start"]:offset["end"]] == f"Among {population}"


@pytest.mark.parametrize(
    ("prefix", "role", "search_role"),
    [
        ("Among accuracy and cost", "other", "required"),
        ("Within alpha systems", "other", "required"),
        ("In older cohorts", "other", "required"),
        ("Across distributed sites", "other", "required"),
        ("Under limited observations", "limitation", "optional"),
        ("With external oversight", "limitation", "optional"),
    ],
)
def test_leading_attachment_defaults_are_conservative(prefix, role, search_role):
    question = f"{prefix}, how does alpha affect beta?"
    draft = _deterministic_seed(question)
    group = next(item for item in draft.groups if item.source_spans == [prefix])
    assert group.canonical_text == prefix.split(" ", 1)[1]
    assert group.role == role and group.search_role == search_role
    assert group.compiled is (search_role == "required")
    offset = group.source_offsets[0]
    assert question[offset["start"]:offset["end"]] == prefix
    if group.confidence == "low":
        assert group.canonical_text in draft.uncertain_terms


def test_parser_uncertainty_survives_empty_gemini_uncertainty():
    question = "How does alpha compared with beta during deployment?"
    seed = _deterministic_seed(question)
    bundle, _ = _gemini_bundle(question)
    assert seed.uncertain_terms
    assert bundle.concepts["uncertain_terms"] == seed.uncertain_terms


def test_parser_bounds_and_no_four_group_assumption_are_explicit():
    assert MAX_QUESTION_CODEPOINTS == 4000
    assert MAX_PARSER_CLAUSES == 16
    assert MAX_COORDINATION_CANDIDATES == 32
    assert MAX_PARSER_CONCEPTS == 16
    assert MAX_SOURCE_SPANS_PER_GROUP == 8
    assert MAX_TERMS_PER_GROUP == 8
    assert MAX_UNCERTAINTIES == 16
    schema = GeminiDirectConceptProposal.model_json_schema()["properties"]
    assert schema["concepts"]["maxItems"] == 5
    source = inspect.getsource(query_module)
    forbidden = ("groups[:4]", "draft.groups[:4]", "max_length=4")
    assert all(value not in source for value in forbidden)


def _assert_exact_group_offsets(question, groups):
    for group in groups:
        for offset, span in zip(group.source_offsets, group.source_spans):
            assert question[offset["start"]:offset["end"]] == span


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "How does policy impact assessment compared with cost-benefit analysis "
            "improve decision quality and transparency?",
            ["policy impact assessment", "cost-benefit analysis", "decision quality OR transparency"],
        ),
        (
            "How do clinical decision support systems compared with rule-based systems "
            "improve accuracy and speed?",
            ["clinical decision support systems", "rule-based systems", "accuracy OR speed"],
        ),
        (
            "How does alpha method compared with decision support systems improve accuracy and speed?",
            ["alpha method", "decision support systems", "accuracy OR speed"],
        ),
        (
            "How does alpha method compared with policy impact assessments improve quality and cost?",
            ["alpha method", "policy impact assessments", "quality OR cost"],
        ),
    ],
)
def test_comparison_boundaries_ignore_predicate_family_nouns(question, expected):
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == expected
    assert [group.role for group in draft.groups] == [
        "intervention_or_method", "comparison", "outcome",
    ]
    assert [group.search_role for group in draft.groups] == ["required"] * 3
    assert [group.coordination for group in draft.groups] == [
        "single", "single", "alternatives",
    ]
    assert all(group.compiled for group in draft.groups)
    assert not draft.uncertain_terms
    _assert_exact_group_offsets(question, draft.groups)
    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar.count(" AND ") == 2
    assert all(f'"{term}"' in bundle.google_scholar for term in expected[:2])


@pytest.mark.parametrize(
    ("question", "subject", "outcomes", "population", "balanced"),
    [
        (
            "How does treatment reduce anxiety and improve sleep quality among students?",
            "treatment", "anxiety OR improve sleep quality", "students",
            '("treatment") AND ("anxiety" OR "improve sleep quality") AND ("students")',
        ),
        (
            "How does training improve accuracy and reduce cost?",
            "training", "accuracy OR reduce cost", None,
            '("training") AND ("accuracy" OR "reduce cost")',
        ),
        (
            "How do methods support prediction and improve accuracy?",
            "methods", "prediction OR improve accuracy", None,
            '("methods") AND ("prediction" OR "improve accuracy")',
        ),
        (
            "How do systems detect faults and predict failures?",
            "systems", "faults OR predict failures", None,
            '("systems") AND ("faults" OR "predict failures")',
        ),
    ],
)
def test_coordinated_predicates_keep_first_boundary_and_clean_outcomes(
    question, subject, outcomes, population, balanced,
):
    draft = _deterministic_seed(question)
    expected = [subject, outcomes] + ([population] if population else [])
    assert [group.canonical_text for group in draft.groups] == expected
    assert [group.role for group in draft.groups[:2]] == [
        "intervention_or_method", "outcome",
    ]
    assert draft.groups[1].coordination == "alternatives"
    assert draft.groups[1].confidence == "low"
    assert draft.groups[1].status_reason == "ambiguous_coordinated_predicate"
    assert draft.groups[1].canonical_text in draft.uncertain_terms
    assert all(group.search_role == "required" and group.compiled for group in draft.groups)
    _assert_exact_group_offsets(question, draft.groups)
    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == balanced


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "How do clinical decision support systems improve accuracy, and "
            "organizational change management reduces attrition?",
            ["clinical decision support systems", "accuracy", "organizational change management", "attrition"],
        ),
        (
            "How do support vector machines improve accuracy, and policy impact assessments reduce cost?",
            ["support vector machines", "accuracy", "policy impact assessments", "cost"],
        ),
    ],
)
def test_independent_clauses_reuse_selected_predicate_boundaries(question, expected):
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == expected
    assert [group.role for group in draft.groups] == [
        "intervention_or_method", "outcome", "intervention_or_method", "outcome",
    ]
    assert all(group.coordination == "co_required" for group in draft.groups)
    assert all(group.search_role == "required" and group.compiled for group in draft.groups)
    _assert_exact_group_offsets(question, draft.groups)
    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == " AND ".join(f'("{term}")' for term in expected)


@pytest.mark.parametrize(
    ("question", "subject", "outcome"),
    [
        ("How does policy improve access to care?", "policy", "access to care"),
        ("How does training reduce time to completion?", "training", "time to completion"),
        ("How does rehabilitation support return to work?", "rehabilitation", "return to work"),
        ("How does ownership improve cost of ownership?", "ownership", "cost of ownership"),
        ("How does design improve ease of use?", "design", "ease of use"),
        ("How does filtering improve robustness to noise?", "filtering", "robustness to noise"),
        ("How does defense improve resistance to attacks?", "defense", "resistance to attacks"),
    ],
)
def test_fixed_governed_ranges_preserve_internal_prepositions(question, subject, outcome):
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == [subject, outcome]
    assert [group.role for group in draft.groups] == ["intervention_or_method", "outcome"]
    assert all(group.coordination == "single" for group in draft.groups)
    assert all(group.search_role == "required" and group.compiled for group in draft.groups)
    _assert_exact_group_offsets(question, draft.groups)
    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == f'("{subject}") AND ("{outcome}")'


def test_governed_range_cross_combination_preserves_attachments_and_offsets():
    question = (
        "Among adults, how does telemedicine compared with in-person care affect "
        "access to care and quality of life during rollout?"
    )
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == [
        "telemedicine", "in-person care", "access to care OR quality of life",
        "adults", "rollout",
    ]
    assert [group.role for group in draft.groups] == [
        "intervention_or_method", "comparison", "outcome", "population", "limitation",
    ]
    assert [group.search_role for group in draft.groups] == [
        "required", "required", "required", "required", "optional",
    ]
    assert draft.groups[2].coordination == "alternatives"
    assert [group.compiled for group in draft.groups] == [True, True, True, True, False]
    _assert_exact_group_offsets(question, draft.groups)
    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == (
        '("telemedicine") AND ("in-person care") AND '
        '("access to care" OR "quality of life") AND ("adults")'
    )


@pytest.mark.parametrize(
    "population",
    [
        "decision support systems", "impact assessment teams",
        "change management professionals", "support groups",
    ],
)
def test_leading_set_shape_is_not_vetoed_by_predicate_family_nouns(population):
    question = f"Among {population}, how does training affect performance?"
    draft = _deterministic_seed(question)
    group = next(item for item in draft.groups if item.role == "population")
    assert group.canonical_text == population
    assert group.search_role == "required" and group.compiled
    assert group.coordination == "single"
    assert group.source_spans == [f"Among {population}"]
    _assert_exact_group_offsets(question, draft.groups)
    bundle, _ = _gemini_bundle(question)
    assert f'("{population}")' in bundle.google_scholar


@pytest.mark.parametrize(
    "question",
    [
        "How does model A outperform model B on accuracy and speed?",
        "How does procedure A supersede procedure B with cost and reliability?",
    ],
)
def test_unknown_question_predicates_fail_closed_without_preposition_boundaries(question):
    draft = _deterministic_seed(question)
    assert len(draft.groups) == 1
    group = draft.groups[0]
    assert group.role == "other" and group.search_role == "required"
    assert group.coordination == "ambiguous" and group.confidence == "low"
    assert group.compiled
    assert group.canonical_text in draft.uncertain_terms
    _assert_exact_group_offsets(question, draft.groups)
    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == f'("{group.canonical_text}")'


def test_repeated_governed_text_keeps_absolute_original_occurrence():
    question = "How does access improve access to care among access groups?"
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == [
        "access", "access to care", "access groups",
    ]
    _assert_exact_group_offsets(question, draft.groups)
    starts = [group.source_offsets[0]["start"] for group in draft.groups]
    assert starts == sorted(starts) and len(set(starts)) == 3


def test_boundary_functions_do_not_call_raw_predicate_pattern_search():
    boundary_functions = (
        query_module._comparison_frame,
        query_module._independently_governed_clauses,
        query_module._lossless_structural_draft,
        query_module._segment_governed_ranges,
    )
    for function in boundary_functions:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        raw_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "search"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "PREDICATE_PATTERN"
        ]
        assert not raw_calls, function.__name__
    assert "_select_governing_predicate" in inspect.getsource(
        query_module._comparison_frame,
    )
    assert "_select_governing_predicate" in inspect.getsource(
        query_module._independently_governed_clauses,
    )


@pytest.mark.parametrize(
    ("question", "subject", "outcome"),
    [
        ("What is the effect of training on support quality?", "training", "support quality"),
        ("What is the impact of policy on impact awareness?", "policy", "impact awareness"),
        ("What is the effect of intervention on change readiness?", "intervention", "change readiness"),
        ("What is the impact of a system on support vector performance?", "system", "support vector performance"),
    ],
)
def test_relation_owned_outcomes_never_delete_predicate_family_nouns(question, subject, outcome):
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == [subject, outcome]
    group = draft.groups[1]
    assert group.role == "outcome" and group.search_role == "required"
    assert group.coordination == "single" and group.confidence == "high"
    assert group.compiled and not draft.uncertain_terms
    _assert_exact_group_offsets(question, draft.groups)
    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == f'("{subject}") AND ("{outcome}")'


@pytest.mark.parametrize(
    ("question", "expected", "uncertain"),
    [
        (
            "Training improves accuracy and reduces cost.",
            "accuracy OR cost",
            False,
        ),
        (
            "Training improved accuracy and reduced cost.",
            "accuracy OR cost",
            False,
        ),
        (
            "Treatment reduces anxiety and improves sleep quality.",
            "anxiety OR sleep quality",
            False,
        ),
        (
            "Treatment reduced anxiety and improved sleep quality.",
            "anxiety OR sleep quality",
            False,
        ),
        (
            "How does training improve accuracy and reduce cost?",
            "accuracy OR reduce cost",
            True,
        ),
        (
            "Training improves awareness and changes in policy.",
            "awareness OR changes in policy",
            True,
        ),
        (
            "Training improves quality and changes over time.",
            "quality OR changes over time",
            True,
        ),
        (
            "Training improves quality and supported decision making.",
            "quality OR supported decision making",
            True,
        ),
        (
            "How does training improve accuracy and support quality?",
            "accuracy OR support quality",
            True,
        ),
        (
            "How does treatment improve safety and impact awareness?",
            "safety OR impact awareness",
            True,
        ),
        (
            "How does intervention improve morale and change readiness?",
            "morale OR change readiness",
            True,
        ),
        (
            "How does training improve prediction accuracy and support quality?",
            "prediction accuracy OR support quality",
            True,
        ),
        (
            "How does training improve support quality and impact awareness?",
            "support quality OR impact awareness",
            True,
        ),
    ],
)
def test_clause_frame_controls_finite_coordinated_predicate_cleanup(question, expected, uncertain):
    draft = _deterministic_seed(question)
    outcome = next(group for group in draft.groups if group.role == "outcome")
    assert outcome.canonical_text == expected
    assert outcome.search_role == "required" and outcome.compiled
    assert outcome.coordination == "alternatives"
    assert outcome.confidence == ("low" if uncertain else "high")
    assert (outcome.status_reason == "ambiguous_coordinated_predicate") is uncertain
    assert (outcome.canonical_text in draft.uncertain_terms) is uncertain
    _assert_exact_group_offsets(question, draft.groups)
    bundle, _ = _gemini_bundle(question)
    terms = expected.split(" OR ")
    expected_query = f'("{draft.groups[0].canonical_text}") AND (' + " OR ".join(f'"{term}"' for term in terms) + ")"
    assert bundle.google_scholar == expected_query


@pytest.mark.parametrize(
    "question",
    [
        "Training improves accuracy, and the system reduces cost.",
        "Training improves accuracy and the system reduces cost.",
    ],
)
def test_explicit_second_subject_remains_independently_governed(question):
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == [
        "Training", "accuracy", "the system", "cost",
    ]
    assert [group.role for group in draft.groups] == [
        "intervention_or_method", "outcome", "intervention_or_method", "outcome",
    ]
    assert all(
        group.coordination == "co_required"
        and group.search_role == "required"
        and group.compiled
        for group in draft.groups
    )
    assert not draft.uncertain_terms
    _assert_exact_group_offsets(question, draft.groups)

    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == (
        '("Training") AND ("accuracy") AND ("the system") AND ("cost")'
    )


@pytest.mark.parametrize(
    ("question", "expected", "uncertain"),
    [
        ("Treatment compared with placebo reduces anxiety and improves sleep quality.", "anxiety OR sleep quality", False),
        ("How does treatment compared with placebo reduce anxiety and improve sleep quality?", "anxiety OR improve sleep quality", True),
        ("How does method compared with baseline improve accuracy and support quality?", "accuracy OR support quality", True),
    ],
)
def test_comparison_outcomes_share_fail_closed_canonicalization(question, expected, uncertain):
    draft = _deterministic_seed(question)
    assert [group.role for group in draft.groups[:3]] == [
        "intervention_or_method", "comparison", "outcome",
    ]
    outcome = draft.groups[2]
    assert outcome.canonical_text == expected
    assert outcome.coordination == "alternatives" and outcome.compiled
    assert outcome.confidence == ("low" if uncertain else "high")
    assert (outcome.canonical_text in draft.uncertain_terms) is uncertain
    _assert_exact_group_offsets(question, draft.groups)


def test_anchored_effectiveness_parallel_gerunds_are_the_only_gerund_exception():
    positive = (
        "How effective is mindfulness-based stress reduction in reducing anxiety "
        "and improving sleep quality among university students?"
    )
    draft = _deterministic_seed(positive)
    outcome = next(group for group in draft.groups if group.role == "outcome")
    assert outcome.canonical_text == "anxiety OR sleep quality"
    assert outcome.status_reason == "anchored_effectiveness_parallel_gerunds"
    assert outcome.confidence == "high" and outcome.coordination == "alternatives"
    assert outcome.source_spans == ["reducing anxiety", "improving sleep quality"]
    _assert_exact_group_offsets(positive, draft.groups)
    body, frame_kind, _, _, clause_mode, auxiliary_inversion = query_module._question_frame(positive)
    outcome_relation, outcome_range = query_module._segment_ranges(positive, body)[1]
    structure = query_module._canonicalize_outcome(
        positive, outcome_range, frame_kind=frame_kind,
        owning_relation=outcome_relation, clause_mode=clause_mode,
        auxiliary_inversion=auxiliary_inversion, first_predicate=None,
    )
    assert [item.text(positive) for item in structure.component_ranges] == [
        "reducing anxiety", "improving sleep quality",
    ]
    assert all(positive[item.start:item.end] == item.text(positive) for item in structure.component_ranges)

    negatives = [
        "How effective is training in reducing errors and support quality?",
        "How effective is training in reducing errors?",
        "What is the impact of training on reducing errors and improving quality?",
        "How does training improve reducing errors and improving quality?",
    ]
    for question in negatives:
        result = _deterministic_seed(question)
        candidates = [group for group in result.groups if group.role == "outcome"]
        if candidates:
            candidate = candidates[0]
            assert candidate.canonical_text != "errors OR quality"
            if candidate.status_reason == "ambiguous_effectiveness_coordination":
                assert candidate.confidence == "low"
                assert candidate.canonical_text in result.uncertain_terms
        else:
            assert len(result.groups) == 1
            assert result.groups[0].coordination == "ambiguous"
            assert result.groups[0].canonical_text in result.uncertain_terms
        _assert_exact_group_offsets(question, result.groups)


@pytest.mark.parametrize(
    "question",
    [
        "How does model A outperform model B on prediction accuracy and response latency?",
        "How does procedure A supersede procedure B with operating cost and system reliability?",
        "How does alpha compare with beta prediction accuracy and response latency?",
    ],
)
def test_fail_closed_groups_preserve_complete_phrase_without_or_manufacturing(question):
    draft = _deterministic_seed(question)
    assert len(draft.groups) == 1
    group = draft.groups[0]
    expected = query_module._strip_question_scaffold(question)
    assert group.canonical_text == expected
    assert " and " in f" {group.canonical_text.casefold()} "
    assert " OR " not in group.canonical_text
    assert group.role == "other" and group.search_role == "required"
    assert group.coordination == "ambiguous" and group.confidence == "low"
    assert group.compiled and group.canonical_text in draft.uncertain_terms
    _assert_exact_group_offsets(question, draft.groups)
    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == f'("{expected}")'


@pytest.mark.parametrize(
    ("question", "attachment"),
    [
        (
            "How does intervention affect symptoms during and after treatment?",
            "during and after treatment",
        ),
        (
            "How does policy change outcomes before and after implementation?",
            "before and after implementation",
        ),
        (
            "How does training improve scores before, during, and after instruction?",
            "before, during, and after instruction",
        ),
        (
            "How does method affect differences within and across groups?",
            "within and across groups",
        ),
        (
            "How does intervention affect symptoms during treatment and after discharge?",
            "during treatment and after discharge",
        ),
        (
            "How does training improve scores before baseline, during treatment, and after follow-up?",
            "before baseline, during treatment, and after follow-up",
        ),
        (
            "How does method affect differences within hospitals and across regions?",
            "within hospitals and across regions",
        ),
    ],
)
def test_coordinated_attachment_markers_form_one_optional_structure(question, attachment):
    draft = _deterministic_seed(question)
    group = next(item for item in draft.groups if item.status_reason == "coordinated_attachment_chain")
    assert group.canonical_text == attachment
    assert group.role == "limitation" and group.search_role == "optional"
    assert group.coordination == "ambiguous" and group.confidence == "low"
    assert not group.compiled and group.canonical_text in draft.uncertain_terms
    assert all(item.canonical_text.casefold() not in {"and", "or"} for item in draft.groups)
    _assert_exact_group_offsets(question, draft.groups)
    bundle, _ = _gemini_bundle(question)
    assert attachment not in bundle.google_scholar


def test_bare_outcome_uncertainty_survives_empty_gemini_uncertainty():
    question = "How does training improve accuracy and reduce cost?"
    seed = _deterministic_seed(question)
    bundle, _ = _gemini_bundle(question)
    assert seed.uncertain_terms == ["accuracy OR reduce cost"]
    assert bundle.concepts["uncertain_terms"] == seed.uncertain_terms


def test_declarative_modal_is_removed_from_subject_without_authorizing_bare_cleanup():
    question = "Training can improve accuracy and reduce cost."
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == [
        "Training", "accuracy OR reduce cost",
    ]
    subject, outcome = draft.groups
    assert subject.role == "intervention_or_method"
    assert subject.search_role == "required" and subject.compiled
    assert subject.source_spans == ["Training"]
    assert outcome.role == "outcome" and outcome.search_role == "required"
    assert outcome.coordination == "alternatives"
    assert outcome.confidence == "low"
    assert outcome.status_reason == "ambiguous_coordinated_predicate"
    assert outcome.compiled
    assert draft.uncertain_terms == ["accuracy OR reduce cost"]
    _assert_exact_group_offsets(question, draft.groups)
    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == '("Training") AND ("accuracy" OR "reduce cost")'


def test_separate_complement_attachment_markers_form_one_optional_chain():
    question = "How does intervention affect symptoms during treatment and after discharge?"
    draft = _deterministic_seed(question)
    assert [group.canonical_text for group in draft.groups] == [
        "intervention", "symptoms", "during treatment and after discharge",
    ]
    attachment = draft.groups[2]
    assert attachment.role == "limitation"
    assert attachment.search_role == "optional"
    assert attachment.coordination == "ambiguous"
    assert attachment.confidence == "low"
    assert attachment.status_reason == "coordinated_attachment_chain"
    assert not attachment.compiled
    assert attachment.source_spans == ["during treatment and after discharge"]
    assert draft.uncertain_terms == ["during treatment and after discharge"]
    _assert_exact_group_offsets(question, draft.groups)
    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == '("intervention") AND ("symptoms")'


def test_main_litsync_auxiliary_first_rq_fails_closed():
    question = (
        "Can large language models help automate "
        "systematic literature reviews?"
    )

    body, frame_kind, _, _, clause_mode, auxiliary_inversion = (
        query_module._question_frame(question)
    )

    assert body.text(question) == (
        "large language models help automate "
        "systematic literature reviews"
    )
    assert frame_kind == "question"
    assert clause_mode == "interrogative"
    assert auxiliary_inversion is True

    draft = _deterministic_seed(question)

    assert len(draft.groups) == 1
    group = draft.groups[0]

    assert group.canonical_text == (
        "large language models help automate "
        "systematic literature reviews"
    )
    assert group.role == "other"
    assert group.search_role == "required"
    assert group.coordination == "ambiguous"
    assert group.confidence == "low"
    assert group.status_reason == "ambiguous_unestablished_question_predicate"
    assert group.compiled
    assert group.canonical_text in draft.uncertain_terms
    assert "Can" not in group.source_spans[0]
    _assert_exact_group_offsets(question, draft.groups)

    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == (
        '("large language models help automate '
        'systematic literature reviews")'
    )
    assert bundle.concepts["uncertain_terms"] == draft.uncertain_terms


@pytest.mark.parametrize(
    ("question", "subject", "outcome"),
    [
        (
            "Does training improve accuracy and reduce cost?",
            "training",
            "accuracy OR reduce cost",
        ),
        (
            "Can machine learning improve diagnosis and reduce cost?",
            "machine learning",
            "diagnosis OR reduce cost",
        ),
        (
            "Should treatment improve safety and support quality?",
            "treatment",
            "safety OR support quality",
        ),
    ],
)
def test_auxiliary_first_known_predicates_exclude_leading_auxiliary(
    question,
    subject,
    outcome,
):
    body, frame_kind, _, _, clause_mode, auxiliary_inversion = (
        query_module._question_frame(question)
    )

    assert frame_kind == "question"
    assert clause_mode == "interrogative"
    assert auxiliary_inversion is True
    assert body.text(question).casefold().startswith(subject.casefold())

    draft = _deterministic_seed(question)

    assert [group.canonical_text for group in draft.groups] == [
        subject,
        outcome,
    ]

    subject_group, outcome_group = draft.groups

    assert subject_group.role == "intervention_or_method"
    assert subject_group.search_role == "required"
    assert subject_group.confidence == "high"
    assert subject_group.compiled

    assert outcome_group.role == "outcome"
    assert outcome_group.search_role == "required"
    assert outcome_group.coordination == "alternatives"
    assert outcome_group.confidence == "low"
    assert outcome_group.status_reason == "ambiguous_coordinated_predicate"
    assert outcome_group.compiled
    assert outcome in draft.uncertain_terms

    leading_auxiliary = question.split(" ", 1)[0]
    assert all(
        not group.canonical_text.casefold().startswith(
            leading_auxiliary.casefold() + " "
        )
        for group in draft.groups
    )

    _assert_exact_group_offsets(question, draft.groups)

    bundle, _ = _gemini_bundle(question)
    terms = outcome.split(" OR ")
    assert bundle.google_scholar == (
        f'("{subject}") AND ('
        + " OR ".join(f'"{term}"' for term in terms)
        + ")"
    )


def test_auxiliary_first_progressive_does_not_authorize_finite_cleanup():
    question = "Is training improving accuracy and reducing cost?"

    body, frame_kind, _, _, clause_mode, auxiliary_inversion = (
        query_module._question_frame(question)
    )

    assert body.text(question) == (
        "training improving accuracy and reducing cost"
    )
    assert frame_kind == "question"
    assert clause_mode == "interrogative"
    assert auxiliary_inversion is True

    draft = _deterministic_seed(question)

    assert [group.canonical_text for group in draft.groups] == [
        "training",
        "accuracy OR reducing cost",
    ]

    outcome = draft.groups[1]
    assert outcome.role == "outcome"
    assert outcome.coordination == "alternatives"
    assert outcome.confidence == "low"
    assert outcome.status_reason == "ambiguous_coordinated_predicate"
    assert outcome.canonical_text in draft.uncertain_terms
    assert outcome.source_spans == [
        "accuracy",
        "reducing cost",
    ]
    _assert_exact_group_offsets(question, draft.groups)

    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == (
        '("training") AND ("accuracy" OR "reducing cost")'
    )


@pytest.mark.parametrize(
    ("question", "expected", "uncertain"),
    [
        (
            "Training improves accuracy, reduces cost, and increases speed.",
            "accuracy OR cost OR speed",
            False,
        ),
        (
            "Training improves accuracy, and reduces cost.",
            "accuracy OR cost",
            False,
        ),
        (
            "Training improves accuracy and reduces cost and increases speed.",
            "accuracy OR cost OR speed",
            False,
        ),
        (
            "How does training improve accuracy, reduce cost, and increase speed?",
            "accuracy OR reduce cost OR increase speed",
            True,
        ),
    ],
)
def test_serial_predicates_keep_the_first_governing_boundary(
    question,
    expected,
    uncertain,
):
    draft = _deterministic_seed(question)

    assert draft.groups[0].canonical_text.casefold() == "training"
    assert draft.groups[0].role == "intervention_or_method"

    outcome = draft.groups[1]
    assert outcome.canonical_text == expected
    assert outcome.role == "outcome"
    assert outcome.search_role == "required"
    assert outcome.coordination == "alternatives"
    assert outcome.confidence == ("low" if uncertain else "high")
    assert (
        outcome.status_reason == "ambiguous_coordinated_predicate"
    ) is uncertain
    assert (
        outcome.canonical_text in draft.uncertain_terms
    ) is uncertain
    assert outcome.compiled

    assert all(
        "improves accuracy" not in group.canonical_text.casefold()
        for group in draft.groups
    )
    _assert_exact_group_offsets(question, draft.groups)

    bundle, _ = _gemini_bundle(question)
    terms = expected.split(" OR ")
    assert bundle.google_scholar == (
        f'("{draft.groups[0].canonical_text}") AND ('
        + " OR ".join(f'"{term}"' for term in terms)
        + ")"
    )


def test_structural_outcome_group_preserves_aligned_component_ranges():
    question = "Training improves accuracy and reduces cost."
    draft = _deterministic_seed(question)

    outcome = next(
        group for group in draft.groups
        if group.role == "outcome"
    )

    assert outcome.canonical_text == "accuracy OR cost"
    assert outcome.source_spans == [
        "accuracy",
        "reduces cost",
    ]
    assert len(outcome.source_offsets) == 2

    assert question[
        outcome.source_offsets[0]["start"]:
        outcome.source_offsets[0]["end"]
    ] == "accuracy"

    assert question[
        outcome.source_offsets[1]["start"]:
        outcome.source_offsets[1]["end"]
    ] == "reduces cost"

    assert outcome.source_offsets[0]["end"] <= (
        outcome.source_offsets[1]["start"]
    )
    _assert_exact_group_offsets(question, draft.groups)


def test_term_details_use_narrow_aligned_outcome_components():
    question = "Training improves accuracy and reduces cost."
    bundle, _ = _gemini_bundle(question)

    details = {
        item["term"]: item
        for item in bundle.concepts["term_details"]
        if item["term"] in {"accuracy", "cost"}
    }

    assert set(details) == {"accuracy", "cost"}

    accuracy_offsets = details["accuracy"]["source_offsets"]
    cost_offsets = details["cost"]["source_offsets"]

    assert len(accuracy_offsets) == 1
    assert len(cost_offsets) == 1

    accuracy_offset = accuracy_offsets[0]
    cost_offset = cost_offsets[0]

    assert question[
        accuracy_offset["start"]:accuracy_offset["end"]
    ] == "accuracy"

    assert question[
        cost_offset["start"]:cost_offset["end"]
    ] == "reduces cost"

    assert (
        accuracy_offset["start"],
        accuracy_offset["end"],
    ) != (
        cost_offset["start"],
        cost_offset["end"],
    )

    assert bundle.google_scholar == (
        '("Training") AND ("accuracy" OR "cost")'
    )


def test_aligned_term_provenance_does_not_move_to_repeated_occurrence():
    question = (
        "Accuracy training improves accuracy "
        "and reduces accuracy cost."
    )

    bundle, _ = _gemini_bundle(question)

    outcome_group = next(
        group for group in bundle.concepts["groups"]
        if group["role"] == "outcome"
    )

    assert outcome_group["source_spans"] == [
        "accuracy",
        "reduces accuracy cost",
    ]

    outcome_details = [
        item
        for item in bundle.concepts["term_details"]
        if item["group"] == outcome_group["label"]
    ]

    canonical_accuracy = next(
        item for item in outcome_details
        if item["term"].casefold() == "accuracy"
    )

    assert len(canonical_accuracy["source_offsets"]) == 1
    linked = canonical_accuracy["source_offsets"][0]

    first_accuracy = question.casefold().find("accuracy")
    governed_accuracy = question.casefold().find(
        "accuracy",
        first_accuracy + 1,
    )

    assert linked["start"] == governed_accuracy
    assert question[linked["start"]:linked["end"]] == "accuracy"


@pytest.mark.parametrize(
    "text",
    [
        (
            "The intervention improves outcomes. "
            "It was evaluated in regional hospitals."
        ),
        (
            "The intervention improves outcomes.\n"
            "The evaluation included regional hospitals."
        ),
        (
            "This is background information rather than a research question. "
            "It contains several explanatory statements."
        ),
    ],
)
def test_multi_sentence_or_paragraph_input_fails_closed_as_one_phrase(text):
    draft = _deterministic_seed(text)

    assert len(draft.groups) == 1
    group = draft.groups[0]

    expected = query_module._clean_term(text)

    assert group.canonical_text == expected
    assert group.role == "other"
    assert group.search_role == "required"
    assert group.coordination == "ambiguous"
    assert group.confidence == "low"
    assert group.status_reason == "non_question_or_uncertain_input_shape"
    assert group.compiled
    assert draft.uncertain_terms == [expected]
    assert group.source_spans == [
        text.strip().rstrip("?.!,;:")
    ]
    _assert_exact_group_offsets(text, draft.groups)

    bundle, _ = _gemini_bundle(text)
    assert bundle.google_scholar == f'("{expected}")'
    assert bundle.concepts["uncertain_terms"] == [expected]


@pytest.mark.parametrize(
    "question",
    [
        "Hwo does training improve accuracy?",
        "Waht does training improve?",
        "Dsoe training improve accuracy?",
        "Cna training improve accuracy?",
    ],
)
def test_likely_misspelled_question_starter_fails_closed(question):
    draft = _deterministic_seed(question)

    assert len(draft.groups) == 1
    group = draft.groups[0]

    expected = query_module._clean_term(question)

    assert group.canonical_text == expected
    assert group.role == "other"
    assert group.search_role == "required"
    assert group.coordination == "ambiguous"
    assert group.confidence == "low"
    assert group.status_reason == "non_question_or_uncertain_input_shape"
    assert group.compiled
    assert draft.uncertain_terms == [expected]
    _assert_exact_group_offsets(question, draft.groups)

    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == f'("{expected}")'


def test_misspelled_concept_terms_remain_literal_without_autocorrection():
    question = "How does traning improve accurcy?"
    draft = _deterministic_seed(question)

    assert [group.canonical_text for group in draft.groups] == [
        "traning",
        "accurcy",
    ]
    assert [group.role for group in draft.groups] == [
        "intervention_or_method",
        "outcome",
    ]
    assert all(
        group.search_role == "required"
        and group.compiled
        for group in draft.groups
    )
    assert all(
        corrected not in {
            term.casefold()
            for group in draft.groups
            for term in group.terms
        }
        for corrected in ("training", "accuracy")
    )
    _assert_exact_group_offsets(question, draft.groups)

    bundle, _ = _gemini_bundle(question)
    assert bundle.google_scholar == (
        '("traning") AND ("accurcy")'
    )
