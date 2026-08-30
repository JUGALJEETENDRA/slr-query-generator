"""Local Ollama screening implementation."""

from .contracts import PaperAssessment, PaperEvidence, ReviewProtocol, ScreeningRQFrame
from .hardware import RuntimeProfile, resolve_runtime_profile
from .profiles import LocalScreeningProfile, resolve_local_screening_profile
from .three_layer import ThreeLayerLocalOrchestrator

__all__ = [
    "LocalScreeningProfile",
    "PaperAssessment",
    "PaperEvidence",
    "ReviewProtocol",
    "RuntimeProfile",
    "ScreeningRQFrame",
    "ThreeLayerLocalOrchestrator",
    "resolve_local_screening_profile",
    "resolve_runtime_profile",
]
