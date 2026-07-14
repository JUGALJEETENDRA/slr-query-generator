from __future__ import annotations

from typing import Any

from local_ai.hardware import resolve_runtime_profile
from local_ai.three_layer import ThreeLayerLocalOrchestrator


LOCAL_AI_FIRST = "local_ai_first"
DEFAULT_SCREENING_STRATEGY = LOCAL_AI_FIRST
PUBLIC_SCREENING_STRATEGIES = {LOCAL_AI_FIRST}


def normalize_screening_strategy(strategy: str | None) -> str:
    return LOCAL_AI_FIRST


def strategy_requires_rq_frame(strategy: str | None) -> bool:
    return False


def screen_candidate(
    *,
    title: str,
    abstract: str,
    research_question: str,
    strategy: str | None = None,
    model: str | None = None,
    mode: str = "local",
    inference_engine=None,
    inclusion_criteria: str = "",
    exclusion_criteria: str = "",
    research_context: str = "",
    model_tier: str | None = None,
    resource_profile: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    profile = resolve_runtime_profile(model_tier, resource_profile)
    local_only = inference_engine is None
    if local_only:
        orchestrator = ThreeLayerLocalOrchestrator(profile=profile)
    else:
        from external_ai.orchestrator import ExternalAIScreeningOrchestrator
        orchestrator = ExternalAIScreeningOrchestrator(
            profile=profile, inference_engine=inference_engine
        )
    result = orchestrator.screen_paper(
        research_question=research_question,
        title=title,
        abstract=abstract,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        **({"research_context": research_context} if local_only else {}),
    )
    result["metadata"] = {
        "screening_strategy": "local_three_layer" if local_only else LOCAL_AI_FIRST,
        "schema_version": result["schema_version"],
        "model_tier": result["model_tier"],
        "resource_profile": profile.resource_profile,
        "model": result["model"],
    }
    return result


def direct_ai_screen_paper(**kwargs):
    """Removed duplicate path; retained as a compatibility alias."""
    return screen_candidate(**kwargs)
