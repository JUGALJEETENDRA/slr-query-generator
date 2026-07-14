"""Hardware-adaptive, evidence-grounded local AI screening core."""

from .contracts import PaperAssessment, PaperEvidence, ReviewProtocol
from .hardware import RuntimeProfile, resolve_runtime_profile
from .three_layer import ThreeLayerLocalOrchestrator

__all__ = [
    "ThreeLayerLocalOrchestrator",
    "PaperAssessment",
    "PaperEvidence",
    "ReviewProtocol",
    "RuntimeProfile",
    "resolve_runtime_profile",
]
