from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from litsync_app.query import generator as query_module
from litsync_app.query.generator import (
    AIQueryExpansionProposal,
    AIQueryTermGroup,
    GEMINI_AI_EXPANSION_PROMPT,
    _deterministic_seed,
    _validate_ai_expansion,
    generate_query_bundle,
)
from litsync_app.screening.engines import GEMINI_WEB_V24_ENGINE
from litsync_app.screening.local.engine import GenerationResult, LocalAIError


QUESTION = (
    "Alpha widgets for beta prediction in gamma systems under scarce observations"
)


def _proposal(required=None, optional=None, uncertain=None):
    return AIQueryExpansionProposal(
        required_groups=required or [],
        optional_groups=optional or [],
        uncertain_terms=uncertain or [],
    )


class ProposalEngine:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = []

    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        self.calls.append((model, prompt, schema, timeout_seconds))
        if self.error:
            raise self.error
        return GenerationResult(
            value=self.value,
            model=model,
            elapsed_seconds=0.01,
        )


def _run(value):
    engine = ProposalEngine(value)
    bundle = generate_query_bundle(
        QUESTION,
        processing_engine=GEMINI_WEB_V24_ENGINE,
        engine=engine,
    )
    return bundle, engine


def test_compact_schema_requires_all_three_top_level_fields():
    for value in ({}, {"required_groups": [], "optional_groups": []}):
        with pytest.raises(ValidationError):
            AIQueryExpansionProposal.model_validate(value)


def test_one_logical_generate_call_and_compact_prompt_contract():
    bundle, engine = _run({
        "required_groups": [
            {"group_label": "Technology", "terms": ["AW", "distributed widgets"]},
        ],
        "optional_groups": [],
        "uncertain_terms": [],
    })
    assert len(engine.calls) == 1
    assert engine.calls[0][2] is AIQueryExpansionProposal
    prompt = engine.calls[0][1]
    assert prompt.startswith(GEMINI_AI_EXPANSION_PROMPT)
    assert f"Research question:\n{QUESTION}" in prompt
    assert '"group_label": "Technology"' in prompt
    lowered = prompt.casefold()
    forbidden = (
        "search the web", "web search", "literature discovery", "academic source",
        "paper record", '"doi"', '"url"', "evidence snippet", "grounding",
        "support_ids", "digital twin", "predictive maintenance",
    )
    assert all(term not in lowered for term in forbidden)
    assert "systematic-review search strategist" in lowered
    assert "search-term expansions" in lowered
    assert bundle.concepts["generation_status"] == "ai_assisted_expansion"


def test_literals_roles_and_canonical_labels_are_immutable():
    seed = _deterministic_seed(QUESTION)
    proposal = _proposal(required=[
        AIQueryTermGroup(group_label=" technology ", terms=["distributed widgets"]),
    ], optional=[
        AIQueryTermGroup(group_label="Conditions and context", terms=["few samples"]),
    ])
    draft, roles, accepted, rejected, balanced, high = _validate_ai_expansion(
        seed, proposal,
    )
    for original in seed.groups:
        final = next(group for group in draft.groups if group.label == original.label)
        assert final.source_spans == original.source_spans
        assert final.terms[:len(original.terms)] == original.terms
    assert accepted[0]["group"] == "Technology"
    assert roles["technology"]["search_role"] == "required"
    assert roles["conditions and context"]["search_role"] == "optional"
    assert all(group.role != "context" for group in balanced)
    assert all(group.role != "context" for group in high)
    assert rejected[0]["reason"] == "context_not_compiled"


def test_mechanical_acronym_enters_both_versions_semantic_only_high_recall():
    bundle, _ = _run({
        "required_groups": [{
            "group_label": "Technology",
            "terms": ["AW", "distributed widgets"],
        }],
        "optional_groups": [],
        "uncertain_terms": [],
    })
    assert '"AW"' in bundle.query_versions["balanced"]["google_scholar"]
    assert "distributed widgets" not in bundle.query_versions["balanced"]["google_scholar"]
    assert '"AW"' in bundle.query_versions["high_recall"]["google_scholar"]
    assert "distributed widgets" in bundle.query_versions["high_recall"]["google_scholar"]
    proposals = {item["term"]: item for item in bundle.concepts["expansion_proposals"]}
    assert proposals["AW"]["compiled_versions"] == ["balanced", "high_recall"]
    assert proposals["distributed widgets"]["compiled_versions"] == ["high_recall"]


def test_context_never_enters_compiled_versions_or_triggers_success():
    bundle, _ = _run({
        "required_groups": [],
        "optional_groups": [{
            "group_label": "Conditions and context", "terms": ["few observations"],
        }],
        "uncertain_terms": [],
    })
    assert bundle.concepts["generation_status"] == "literal_fallback"
    assert "few observations" not in bundle.google_scholar
    assert "few observations" not in bundle.query_versions["high_recall"]["google_scholar"]
    assert bundle.concepts["expansion_proposals"][0]["reason"] == "context_not_compiled"


def test_terms_are_rejected_independently_for_unknown_duplicate_and_cross_group():
    seed = _deterministic_seed(QUESTION)
    proposal = _proposal(required=[
        AIQueryTermGroup(group_label="Unknown", terms=["novel phrase"]),
        AIQueryTermGroup(
            group_label="Technology",
            terms=["Alpha widgets", "gamma systems", "distributed widgets"],
        ),
    ])
    _, _, accepted, rejected, _, _ = _validate_ai_expansion(seed, proposal)
    assert [item["term"] for item in accepted] == ["distributed widgets"]
    reasons = {item["term"]: item["reason"] for item in rejected}
    assert reasons["novel phrase"] == "unknown_or_ambiguous_group"
    assert reasons["Alpha widgets"] == "duplicate_term"
    assert reasons["gamma systems"] == "scope_or_group_mismatch"


def test_limits_apply_after_normalization_and_preserve_literals_first():
    seed = _deterministic_seed(QUESTION)
    terms = ["Alpha widgets", "term one", "term two", "term three", "term four", "term five"]
    proposal = _proposal(required=[
        AIQueryTermGroup(group_label="Technology", terms=terms),
    ])
    draft, _, accepted, rejected, _, high = _validate_ai_expansion(seed, proposal)
    assert len(accepted) == 4
    assert next(group for group in draft.groups if group.label == "Technology").terms[:2] == [
        "Alpha widget", "Alpha widgets",
    ]
    assert len(next(group for group in high if group.label == "Technology").terms) == 6
    assert {item["reason"] for item in rejected} == {"duplicate_term", "group_addition_limit"}


def test_malformed_or_useless_output_returns_literal_fallback():
    engine = ProposalEngine(error=LocalAIError("offline"))
    bundle = generate_query_bundle(
        QUESTION, processing_engine=GEMINI_WEB_V24_ENGINE, engine=engine,
    )
    assert len(engine.calls) == 1
    assert bundle.concepts["generation_status"] == "literal_fallback"
    assert bundle.concepts["warning"]
    assert bundle.query_versions["balanced"] == bundle.query_versions["high_recall"]


def test_top_level_fields_are_exact_balanced_aliases_and_legacy_evidence_is_empty():
    bundle, _ = _run({
        "required_groups": [{
            "group_label": "Technology", "terms": ["distributed widgets"],
        }],
        "optional_groups": [],
        "uncertain_terms": [],
    })
    balanced = bundle.query_versions["balanced"]
    for database in (
        "google_scholar", "scopus", "web_of_science", "ieee_xplore", "pubmed",
    ):
        assert getattr(bundle, database) == balanced[database]
    assert bundle.concepts["evidence_label"] == "AI-assisted query expansion"
    assert bundle.concepts["evidence_limitation"] == (
        "Gemini-proposed terminology was not independently checked against academic literature."
    )
    assert bundle.concepts["grounded_terms"] == []
    assert bundle.concepts["grounding_papers"] == []
    assert bundle.concepts["gemini_reported_sources"] == []
    assert "query_stage_debug" not in bundle.concepts


def test_production_has_no_semantic_registry_or_removed_architecture_vocabulary():
    source = inspect.getsource(query_module)
    forbidden = (
        "QueryLiteratureDiscoveryEnvelope", "GeminiReportedSource",
        "QueryExpansionStageProposal", "generate_query_expansion",
        "query_stage_debug", "claimed_support_ids", "search_activation_status",
    )
    assert all(term not in source for term in forbidden)
