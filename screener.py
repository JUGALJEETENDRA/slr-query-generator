"""Compatibility facade for callers of the former rule-based screener."""

from local_ai.hardware import resolve_runtime_profile
from local_ai.three_layer import ThreeLayerLocalOrchestrator


def screen_paper(
    title,
    abstract,
    research_question,
    model=None,
    inference_engine=None,
    inclusion_criteria="",
    exclusion_criteria="",
    model_tier=None,
    resource_profile=None,
    **_,
):
    profile = resolve_runtime_profile(model_tier, resource_profile)
    if inference_engine is None:
        orchestrator = ThreeLayerLocalOrchestrator(profile)
    else:
        from external_ai.orchestrator import ExternalAIScreeningOrchestrator
        orchestrator = ExternalAIScreeningOrchestrator(
            profile, inference_engine=inference_engine
        )
    return orchestrator.screen_paper(
        research_question, title, abstract,
        inclusion_criteria, exclusion_criteria,
    )
