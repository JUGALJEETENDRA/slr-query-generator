from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from litsync_app.query import generator as query_module
from litsync_app.query.generator import (
    GEMINI_DIRECT_CONCEPT_PROMPT,
    GeminiDirectConceptProposal,
    _gemini_direct_schema,
    generate_query_bundle,
)
from litsync_app.screening.engines import GEMINI_WEB_V24_ENGINE
from litsync_app.screening.local.engine import GenerationResult, LocalAIError


QUESTION = "Alpha widgets predict beta failures in gamma systems"
VALID_PROPOSAL = {
    "concepts": [
        {
            "label": "Technology",
            "balanced_terms": ["alpha widgets"],
            "high_recall_terms": ["alpha widgets", "distributed widgets"],
        },
        {
            "label": "Outcome",
            "balanced_terms": ["beta failures"],
            "high_recall_terms": ["beta failures", "failure prediction"],
        },
        {
            "label": "Domain",
            "balanced_terms": ["gamma systems"],
            "high_recall_terms": ["gamma systems"],
        },
    ],
}


class ProposalEngine:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = []

    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        self.calls.append((model, prompt, schema, timeout_seconds))
        if self.error:
            raise self.error
        return GenerationResult(value=self.value, model=model, elapsed_seconds=0.01)


def _run(value=VALID_PROPOSAL, question=QUESTION):
    engine = ProposalEngine(value)
    bundle = generate_query_bundle(
        question,
        processing_engine=GEMINI_WEB_V24_ENGINE,
        engine=engine,
    )
    return bundle, engine


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"concepts": []},
        {"concepts": [VALID_PROPOSAL["concepts"][0]]},
        {
            "concepts": [
                VALID_PROPOSAL["concepts"][0],
                {**VALID_PROPOSAL["concepts"][1], "label": " technology "},
            ],
        },
        {
            "concepts": [
                {
                    "label": "One",
                    "balanced_terms": ["alpha", "Alpha"],
                    "high_recall_terms": ["alpha", "extra"],
                },
                VALID_PROPOSAL["concepts"][1],
            ],
        },
        {
            "concepts": [
                {
                    "label": "One",
                    "balanced_terms": ["alpha"],
                    "high_recall_terms": ["different"],
                },
                VALID_PROPOSAL["concepts"][1],
            ],
        },
        {
            "concepts": [
                {
                    "label": "One",
                    "balanced_terms": ["alpha"],
                    "high_recall_terms": ["alpha"],
                },
                {
                    "label": "Two",
                    "balanced_terms": ["beta"],
                    "high_recall_terms": ["beta"],
                },
            ],
        },
    ],
)
def test_direct_schema_rejects_malformed_or_non_broader_contracts(value):
    with pytest.raises(ValidationError):
        GeminiDirectConceptProposal.model_validate(value)


def test_raw_input_reaches_one_logical_generate_call_without_parser_groups(monkeypatch):
    monkeypatch.setattr(
        query_module,
        "_deterministic_seed",
        lambda _question: (_ for _ in ()).throw(
            AssertionError("Gemini must not invoke the deterministic parser")
        ),
    )
    bundle, engine = _run()
    assert len(engine.calls) == 1
    _, prompt, schema, timeout = engine.calls[0]
    assert issubclass(schema, GeminiDirectConceptProposal)
    assert prompt.startswith(GEMINI_DIRECT_CONCEPT_PROMPT)
    assert f"Research question or paragraph:\n{QUESTION}" in prompt
    assert "Parser-owned groups" not in prompt
    assert "source_spans" not in prompt
    assert "balanced_query" not in prompt
    assert "high_recall_query" not in prompt
    assert timeout == 120.0
    assert bundle.concepts["generation_status"] == "ai_assisted_expansion"


def test_complete_input_cannot_be_returned_as_one_term():
    schema = _gemini_direct_schema(QUESTION)
    invalid = {
        "concepts": [
            {
                "label": "Whole input",
                "balanced_terms": [QUESTION],
                "high_recall_terms": [QUESTION, "extra"],
            },
            VALID_PROPOSAL["concepts"][1],
        ],
    }
    with pytest.raises(ValidationError, match="complete input"):
        schema.model_validate(invalid)


def test_existing_compilers_build_both_versions_and_balanced_aliases():
    bundle, _ = _run()
    balanced = bundle.query_versions["balanced"]
    high_recall = bundle.query_versions["high_recall"]
    assert balanced["google_scholar"] == (
        '("alpha widgets") AND ("beta failures") AND ("gamma systems")'
    )
    assert high_recall["google_scholar"] == (
        '("alpha widgets" OR "distributed widgets") AND '
        '("beta failures" OR "failure prediction") AND ("gamma systems")'
    )
    assert balanced["scopus"].startswith("TITLE-ABS-KEY(")
    assert balanced["web_of_science"].startswith("TS=(")
    assert "[tiab]" in balanced["pubmed"]
    for database in (
        "google_scholar", "scopus", "web_of_science", "ieee_xplore", "pubmed",
    ):
        assert getattr(bundle, database) == balanced[database]


def test_all_terms_are_ai_origin_with_empty_offsets_and_legacy_evidence_is_empty():
    bundle, _ = _run()
    assert all(
        detail["source"] == "ai_assisted_query_expansion"
        and detail["source_offsets"] == []
        for detail in bundle.concepts["term_details"]
    )
    assert bundle.concepts["evidence_label"] == "AI-generated query"
    assert bundle.concepts["grounded_terms"] == []
    assert bundle.concepts["grounding_papers"] == []
    assert bundle.concepts["gemini_reported_sources"] == []


def test_gemini_failure_raises_instead_of_returning_parser_fallback():
    engine = ProposalEngine(error=LocalAIError("offline"))
    with pytest.raises(LocalAIError, match="offline"):
        generate_query_bundle(
            QUESTION,
            processing_engine=GEMINI_WEB_V24_ENGINE,
            engine=engine,
        )
    assert len(engine.calls) == 1


def test_main_litsync_question_compiles_three_required_concepts():
    question = "Can large language models help automate systematic literature reviews?"
    proposal = {
        "concepts": [
            {
                "label": "Language models",
                "balanced_terms": ["large language model", "large language models", "LLM"],
                "high_recall_terms": [
                    "large language model", "large language models", "LLM", "generative AI", "GPT",
                ],
            },
            {
                "label": "Systematic reviews",
                "balanced_terms": ["systematic literature review", "systematic review"],
                "high_recall_terms": [
                    "systematic literature review", "systematic review", "evidence synthesis",
                ],
            },
            {
                "label": "Automation",
                "balanced_terms": ["automation", "automate", "assisted"],
                "high_recall_terms": [
                    "automation", "automate", "assisted", "review automation", "study screening",
                ],
            },
        ],
    }
    bundle, _ = _run(proposal, question)
    balanced = bundle.google_scholar
    assert balanced.count(" AND ") == 2
    assert "large language model" in balanced
    assert "systematic literature review" in balanced
    assert "automation" in balanced
    assert question not in balanced
    assert bundle.query_versions["high_recall"]["google_scholar"].count(" AND ") == 2


def test_production_prompt_has_no_parser_or_evidence_architecture():
    prompt = GEMINI_DIRECT_CONCEPT_PROMPT.casefold()
    forbidden = (
        "parser-owned", "source span", "doi", "url", "paper", "evidence",
        "grounding", "support id", "balanced_query", "high_recall_query",
    )
    assert all(value not in prompt for value in forbidden)
    source = inspect.getsource(query_module._generate_direct_gemini_bundle)
    assert "_deterministic_seed" not in source
    assert "_validate_ai_expansion" not in source
    obsolete_symbols = (
        "AIQueryTermGroup",
        "AIQueryExpansionProposal",
        "GEMINI_AI_EXPANSION_PROMPT",
        "_validate_ai_expansion",
        "_resolve_group_label",
    )
    assert all(not hasattr(query_module, name) for name in obsolete_symbols)
