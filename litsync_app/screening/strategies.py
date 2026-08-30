from __future__ import annotations

from typing import Any

from litsync_app.screening.local.hardware import resolve_runtime_profile
from litsync_app.screening.local.three_layer import ThreeLayerLocalOrchestrator
from litsync_app.screening.local.rq_frame import build_screening_rq_frame


LOCAL_AI_FIRST = "local_ai_first"
DEFAULT_SCREENING_STRATEGY = LOCAL_AI_FIRST
PUBLIC_SCREENING_STRATEGIES = {LOCAL_AI_FIRST}


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
    local_profile: str = "baseline-v3.12",
    rq_structure_json=None,
    **_: Any,
) -> dict[str, Any]:
    profile = resolve_runtime_profile(model_tier, resource_profile)
    local_only = inference_engine is None
    if local_only:
        orchestrator = ThreeLayerLocalOrchestrator(
            profile=profile, screening_profile=local_profile
        )
        orchestrator.require_profile_models()
        rq_frame = (
            build_screening_rq_frame(
                research_question, inclusion=inclusion_criteria, exclusion=exclusion_criteria,
                context=research_context, submitted=rq_structure_json,
                frame_version=orchestrator.screening_profile.rq_frame_version,
            ) if orchestrator.screening_profile.structured_rq else None
        )
    else:
        from litsync_app.screening.external.orchestrator import ExternalAIScreeningOrchestrator
        orchestrator = ExternalAIScreeningOrchestrator(
            profile=profile, inference_engine=inference_engine
        )
    kwargs = dict(
        research_question=research_question, title=title, abstract=abstract,
        inclusion_criteria=inclusion_criteria, exclusion_criteria=exclusion_criteria,
        research_context=research_context,
    )
    if local_only:
        kwargs["rq_frame"] = rq_frame
    result = orchestrator.screen_paper(**kwargs)
    if local_only:
        active = rq_frame
        result.update({
            "rq_frame_id": active.frame_id if active else "",
            "rq_frame_version": active.frame_version if active else "",
            "rq_frame_source": active.source if active else "not_used_baseline",
            "rq_frame_status": active.status if active else "not_used_baseline",
            "rq_frame_validation_failures": active.validation_failures if active else [],
            "local_profile": orchestrator.screening_profile.name,
            "protocol_model": orchestrator.protocol_model,
            "deep_model": orchestrator.deep_model,
            "edge_model": orchestrator.edge_model,
        })
    result["metadata"] = {
        "screening_strategy": "local_three_layer" if local_only else LOCAL_AI_FIRST,
        "schema_version": result["schema_version"],
        "model_tier": result["model_tier"],
        "resource_profile": profile.resource_profile,
        "model": result["model"],
    }
    return result
