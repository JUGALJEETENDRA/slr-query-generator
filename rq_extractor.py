"""Compatibility facade exposing the AI-compiled review protocol."""

from local_ai.hardware import resolve_runtime_profile
from local_ai.three_layer import ThreeLayerLocalOrchestrator


def extract_rq(research_question, inclusion_criteria="", exclusion_criteria="", **_):
    orchestrator = ThreeLayerLocalOrchestrator(resolve_runtime_profile())
    protocol = orchestrator.compile_protocol(
        research_question, inclusion_criteria, exclusion_criteria
    )
    return {"protocol": protocol.model_dump(mode="json"), "protocol_id": protocol.protocol_id}
