"""Optional external-engine screening support, isolated from the local core."""

from .orchestrator import AssessmentEnvelope, ExternalAIScreeningOrchestrator

__all__ = ["AssessmentEnvelope", "ExternalAIScreeningOrchestrator"]
