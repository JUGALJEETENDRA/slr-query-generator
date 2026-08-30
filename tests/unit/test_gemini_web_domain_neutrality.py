from __future__ import annotations

import inspect

from litsync_app.integrations import gemini_web_screening_prompt as prompt_module
from litsync_app.integrations import gemini_web_screening as screening_module
from litsync_app.integrations.gemini_web_screening_prompt import (
    PROMPT_VERSION,
    batch_prompt,
    fallback_rubric,
    protocol_prompt,
)


BENCHMARK_SHAPED_TERMS = (
    "federated learning",
    "healthcare",
    "finance",
    "internet of things",
    "iot",
    "transportation",
    "privacy, robustness",
    "communication efficiency",
    "predictive performance",
)


def _assert_no_benchmark_shaping(value: str) -> None:
    lowered = value.casefold()
    found = [
        term
        for term in BENCHMARK_SHAPED_TERMS
        if term.casefold() in lowered
    ]
    assert not found, (
        "Gemini Web Fast contains benchmark-shaped production text: "
        + ", ".join(found)
    )


def test_prompt_version_changes_after_domain_neutrality_correction():
    assert PROMPT_VERSION == "gemini-web-screening-prompt-v6"


def test_neutral_protocol_prompt_contains_no_benchmark_shaped_examples():
    rendered = protocol_prompt(
        question="How does the specified method affect the stated outcome?",
        context="Review primary studies in the researcher-defined scope.",
        inclusion="Include studies evaluating at least one stated outcome.",
        exclusion="Exclude records without primary evidence.",
    )

    _assert_no_benchmark_shaping(rendered)


def test_neutral_batch_prompt_contains_no_benchmark_shaped_examples():
    inclusion = "Include studies evaluating at least one stated outcome."
    exclusion = "Exclude records without primary evidence."
    rubric = fallback_rubric(inclusion, exclusion)

    rendered = batch_prompt(
        question="How does the specified method affect the stated outcome?",
        context="Review primary studies in the researcher-defined scope.",
        inclusion=inclusion,
        exclusion=exclusion,
        rubric=rubric,
        papers=[
            {
                "paper_id": "paper-1",
                "title": "A neutral study title",
                "abstract": "The study evaluates a specified method.",
            }
        ],
        verification=False,
    )

    _assert_no_benchmark_shaping(rendered)


def test_domain_terms_enter_prompts_only_from_user_supplied_inputs():
    supplied_phrase = "federated learning in healthcare"
    rendered = protocol_prompt(
        question=f"How is {supplied_phrase} evaluated?",
        context="The researcher explicitly supplied this domain.",
        inclusion=f"Include primary studies of {supplied_phrase}.",
        exclusion="Exclude non-primary records.",
    )

    assert supplied_phrase in rendered.casefold()


def test_fast_runtime_source_has_no_benchmark_specific_decision_shortcuts():
    source = "\n".join(
        (
            inspect.getsource(prompt_module),
            inspect.getsource(screening_module),
        )
    )

    _assert_no_benchmark_shaping(source)
