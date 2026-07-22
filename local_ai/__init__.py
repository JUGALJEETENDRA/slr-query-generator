"""Hardware-adaptive, evidence-grounded local AI screening core."""

from .contracts import PaperAssessment, PaperEvidence, ReviewProtocol, ScreeningRQFrame
from .hardware import RuntimeProfile, resolve_runtime_profile
from .profiles import LocalScreeningProfile, resolve_local_screening_profile
from .three_layer import ThreeLayerLocalOrchestrator

__all__ = [
    "ThreeLayerLocalOrchestrator",
    "PaperAssessment",
    "PaperEvidence",
    "ReviewProtocol",
    "ScreeningRQFrame",
    "LocalScreeningProfile",
    "resolve_local_screening_profile",
    "RuntimeProfile",
    "resolve_runtime_profile",
]
