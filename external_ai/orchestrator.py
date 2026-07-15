from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from local_ai.cache import JsonDiskCache, cache_key
from local_ai.contracts import (
    SCHEMA_VERSION,
    PaperAssessment,
    PaperEvidence,
    ReviewProtocol,
    ValidationReport,
    safe_maybe,
)
from local_ai.engine import (
    LocalAIError,
    LocalAIMemoryError,
    LocalAIOutputError,
)
from external_ai.engine import InjectedStructuredEngine
from local_ai.evidence import evidence_lookup
from local_ai.hardware import RuntimeProfile, TIERS, resolve_runtime_profile
from external_ai.prompts import (
    assessment_prompt,
    critic_prompt,
    evidence_prompt,
    protocol_critic_prompt,
    protocol_prompt,
    protocol_repair_prompt,
)
from local_ai.validator import validate_assessment


PROMPT_VERSION = "external-gemini-v3"


@dataclass
class AssessmentEnvelope:
    assessment: PaperAssessment
    validation: ValidationReport
    model: str
    tier: str
    elapsed_seconds: float
    escalated: bool = False
    attempts: int = 1
    cache_hit: bool = False
    original_elapsed_seconds: float = 0.0
    repairable: bool = True
    runtime_downgrades: list[str] = field(default_factory=list)

    def needs_escalation(self) -> bool:
        """Return true only for a validator/schema failure that one repair can fix."""
        return self.repairable and not self.validation.valid

    def to_public_result(self, protocol: ReviewProtocol, title: str, abstract: str) -> dict[str, Any]:
        assessment = self.assessment.model_dump(mode="json")
        units = evidence_lookup(title, abstract)
        public_criteria = []
        public_evidence = []
        for item in assessment["criteria"]:
            resolved = []
            for reference in item["evidence"]:
                unit = units.get(reference["evidence_id"])
                if unit is None or unit["source"] != reference["source"]:
                    continue
                span = {
                    "source": unit["source"], "quote": unit["text"],
                    "evidence_id": unit["evidence_id"],
                }
                resolved.append(span)
                public_evidence.append({"criterion_id": item["criterion_id"], **span})
            public_criteria.append({**item, "evidence": resolved})
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": assessment["decision"],
            "reason": assessment["reason"],
            "confidence": assessment["confidence"],
            "protocol_id": protocol.protocol_id,
            "criteria": public_criteria,
            "evidence": public_evidence,
            "summary": assessment["summary"],
            "uncertainty": assessment["uncertainty"],
            "missing_information": assessment["missing_information"],
            "contradictions": assessment["contradictions"],
            "validation_status": "validated" if self.validation.valid else "unresolved",
            "validation_errors": self.validation.errors,
            "validation_warnings": self.validation.warnings,
            "escalated": self.escalated,
            "model": self.model,
            "model_tier": self.tier,
            "prompt_version": PROMPT_VERSION,
            "attempts": self.attempts,
            "cache_hit": self.cache_hit,
            "processing_seconds": round(self.elapsed_seconds, 4),
            "original_processing_seconds": round(self.original_elapsed_seconds, 4),
            "runtime_downgrades": self.runtime_downgrades,
        }


class ExternalAIScreeningOrchestrator:
    def __init__(
        self,
        profile: RuntimeProfile | None = None,
        engine=None,
        inference_engine=None,
        cache: JsonDiskCache | None = None,
    ):
        self.profile = profile or resolve_runtime_profile()
        if engine is not None:
            self.engine = engine
        elif inference_engine is not None:
            self.engine = InjectedStructuredEngine(inference_engine)
        else:
            raise ValueError("External screening requires an injected inference engine")
        self.cache = cache or JsonDiskCache(os.getenv("LOCAL_AI_CACHE_PATH", "outputs/cache/local_ai"))
        self.runtime_downgrades: list[str] = list(self.profile.downgrade_reasons)
        self._consecutive_runtime_failures = 0

    def hardware_diagnostics(self, calibrate: bool = False) -> dict[str, Any]:
        data = self.profile.as_dict()
        if calibrate and hasattr(self.engine, "calibrate"):
            data["calibration"] = self.engine.calibrate()
        data["runtime_downgrades"] = list(self.runtime_downgrades)
        return data

    def compile_protocol(
        self,
        research_question: str,
        inclusion_criteria: str = "",
        exclusion_criteria: str = "",
        research_context: str = "",
    ) -> ReviewProtocol:
        # One 8B protocol contract is shared by balanced and performance tiers.
        # A larger model is available only through an explicit environment override.
        model = self.profile.fast_model
        key = cache_key(
            research_question, inclusion_criteria, exclusion_criteria, research_context,
            model, self.profile.resolved_tier, PROMPT_VERSION,
        )
        cached = self.cache.get("protocols", key)
        if cached:
            return ReviewProtocol.model_validate(cached)

        result = self._generate_protocol(
            model, protocol_prompt(
                research_question, inclusion_criteria, exclusion_criteria, research_context
            )
        )
        model = result.model
        protocol = self._validated_protocol(
            result.value, result.model, research_question, inclusion_criteria, exclusion_criteria
        )
        criticised = self._generate_protocol(
            model,
            protocol_critic_prompt(protocol, inclusion_criteria, exclusion_criteria),
        )
        protocol = self._validated_protocol(
            criticised.value, criticised.model, research_question, inclusion_criteria, exclusion_criteria
        )
        protocol = protocol.with_identity()
        self.cache.set("protocols", key, protocol.model_dump(mode="json"))
        return protocol

    def assess_fast(self, protocol: ReviewProtocol, title: str, abstract: str) -> AssessmentEnvelope:
        model = self.profile.fast_model
        key = cache_key(protocol.protocol_id, title, abstract, model, self.profile.resolved_tier, PROMPT_VERSION, "fast")
        cached = self.cache.get("assessments", key)
        if cached:
            assessment = PaperAssessment.model_validate(cached["assessment"])
            validation = validate_assessment(assessment, protocol, title, abstract)
            cached_model = str(cached.get("model") or model)
            return AssessmentEnvelope(
                assessment, validation, cached_model, self.profile.resolved_tier,
                0.0, cache_hit=True,
                original_elapsed_seconds=float(cached.get("elapsed_seconds") or 0.0),
                runtime_downgrades=list(self.runtime_downgrades),
            )

        started = time.perf_counter()
        repairable = True
        try:
            evidence = None
            attempts = 1
            if self.profile.resolved_tier == "compact":
                evidence_result = self._generate(model, evidence_prompt(protocol, title, abstract), PaperEvidence)
                evidence = PaperEvidence.model_validate(evidence_result.value)
                attempts += 1
            generated = self._generate(
                model, assessment_prompt(protocol, title, abstract, evidence), PaperAssessment
            )
            assessment = PaperAssessment.model_validate(generated.value)
            validation = validate_assessment(assessment, protocol, title, abstract)
            model = generated.model
        except LocalAIOutputError as exc:
            assessment = safe_maybe(f"Local AI returned malformed structured output: {exc}")
            validation = ValidationReport(valid=False, errors=[str(exc)])
            attempts = 1
            repairable = True
        except LocalAIError as exc:
            assessment = safe_maybe(f"Local AI fast assessment failed: {exc}")
            validation = ValidationReport(valid=False, errors=[str(exc)])
            attempts = 1
            repairable = False
        except (ValidationError, ValueError) as exc:
            assessment = safe_maybe(f"Local AI fast assessment failed: {exc}")
            validation = ValidationReport(valid=False, errors=[str(exc)])
            attempts = 1
            repairable = True
        elapsed = time.perf_counter() - started
        envelope = AssessmentEnvelope(
            assessment, validation, model, self.profile.resolved_tier, elapsed,
            attempts=attempts, original_elapsed_seconds=elapsed,
            repairable=repairable,
            runtime_downgrades=list(self.runtime_downgrades),
        )
        self.cache.set("assessments", key, {
            "assessment": assessment.model_dump(mode="json"),
            "elapsed_seconds": elapsed,
            "model": model,
        })
        return envelope

    def escalate(
        self,
        protocol: ReviewProtocol,
        title: str,
        abstract: str,
        envelope: AssessmentEnvelope,
    ) -> AssessmentEnvelope:
        model = self.profile.strong_model
        key = cache_key(
            protocol.protocol_id, title, abstract, model, PROMPT_VERSION,
            envelope.assessment.model_dump(mode="json"), "critic",
        )
        cached = self.cache.get("assessments", key)
        if cached:
            assessment = PaperAssessment.model_validate(cached["assessment"])
            validation = validate_assessment(assessment, protocol, title, abstract)
            cached_model = str(cached.get("model") or model)
            return AssessmentEnvelope(
                assessment, validation, cached_model, self.profile.resolved_tier,
                envelope.elapsed_seconds,
                escalated=True, attempts=envelope.attempts + 1, cache_hit=True,
                original_elapsed_seconds=(
                    envelope.original_elapsed_seconds
                    + float(cached.get("elapsed_seconds") or 0.0)
                ),
                runtime_downgrades=list(self.runtime_downgrades),
            )
        started = time.perf_counter()
        try:
            generated = self._generate(
                model,
                critic_prompt(
                    protocol, title, abstract, envelope.assessment,
                    envelope.validation.errors, envelope.validation.warnings,
                ),
                PaperAssessment,
            )
            candidate = PaperAssessment.model_validate(generated.value)
            validation = validate_assessment(candidate, protocol, title, abstract)
            model = generated.model
            if validation.valid:
                assessment = candidate
            else:
                assessment = safe_maybe(
                    "Assessment repair remained unsupported: " + "; ".join(validation.errors)
                )
        except (LocalAIError, ValidationError, ValueError) as exc:
            assessment = safe_maybe(f"Assessment repair failed: {exc}")
            validation = ValidationReport(valid=False, errors=[str(exc)])
        elapsed = time.perf_counter() - started
        final_validation = validate_assessment(assessment, protocol, title, abstract)
        if assessment.decision == "MAYBE" and not validation.valid:
            final_validation = ValidationReport(
                valid=False, errors=validation.errors, warnings=validation.warnings,
                exact_quote_count=validation.exact_quote_count,
                decisive_evidence_count=validation.decisive_evidence_count,
            )
        result = AssessmentEnvelope(
            assessment, final_validation, model, self.profile.resolved_tier,
            envelope.elapsed_seconds + elapsed, escalated=True,
            attempts=envelope.attempts + 1,
            original_elapsed_seconds=envelope.original_elapsed_seconds + elapsed,
            runtime_downgrades=list(self.runtime_downgrades),
        )
        self.cache.set("assessments", key, {
            "assessment": assessment.model_dump(mode="json"),
            "elapsed_seconds": elapsed,
            "model": model,
        })
        return result

    def screen_paper(
        self,
        research_question: str,
        title: str,
        abstract: str,
        inclusion_criteria: str = "",
        exclusion_criteria: str = "",
        research_context: str = "",
        protocol: ReviewProtocol | None = None,
    ) -> dict[str, Any]:
        try:
            protocol = protocol or self.compile_protocol(
                research_question, inclusion_criteria, exclusion_criteria, research_context
            )
            envelope = self.assess_fast(protocol, title, abstract)
            if envelope.needs_escalation():
                envelope = self.escalate(protocol, title, abstract, envelope)
            return envelope.to_public_result(protocol, title, abstract)
        except (LocalAIError, ValidationError, ValueError) as exc:
            fallback = safe_maybe(f"Screening unavailable: {exc}")
            return {
                "schema_version": SCHEMA_VERSION,
                "decision": "MAYBE", "reason": fallback.reason, "confidence": 0.0,
                "protocol_id": getattr(protocol, "protocol_id", ""),
                "criteria": [], "evidence": [], "summary": fallback.summary,
                "uncertainty": fallback.uncertainty,
                "missing_information": fallback.missing_information,
                "contradictions": [], "validation_status": "unresolved",
                "validation_errors": [str(exc)], "validation_warnings": [],
                "escalated": False, "model": self.profile.fast_model,
                "model_tier": self.profile.resolved_tier,
                "prompt_version": PROMPT_VERSION, "attempts": 0,
                "cache_hit": False, "processing_seconds": 0.0,
                "original_processing_seconds": 0.0,
                "runtime_downgrades": list(self.runtime_downgrades),
            }

    def prepare_strong_pass(self) -> None:
        if self.profile.strong_model != self.profile.fast_model and hasattr(self.engine, "unload"):
            self.engine.unload(self.profile.fast_model)

    def _generate(self, model: str, prompt: str, schema):
        try:
            result = self.engine.generate(model, prompt, schema)
            self._consecutive_runtime_failures = 0
            return result
        except LocalAIOutputError:
            # Output syntax is not a hardware failure. Keep the selected model and
            # let the orchestrator perform its single fresh-context repair.
            raise
        except LocalAIMemoryError as exc:
            self._downgrade_after_oom(model, str(exc))
            replacement = self.profile.fast_model if model != self.profile.strong_model else self.profile.strong_model
            return self.engine.generate(replacement, prompt, schema)
        except LocalAIError as first_error:
            self._consecutive_runtime_failures += 1
            failure_text = str(first_error).lower()
            load_failure = any(term in failure_text for term in ("not found", "failed to load"))
            repeated_timeout = (
                any(term in failure_text for term in ("timed out", "timeout"))
                and self._consecutive_runtime_failures >= 2
            )
            if load_failure or repeated_timeout:
                if self.profile.resolved_tier == "compact":
                    raise
                self._downgrade_after_oom(model, f"runtime fallback: {first_error}")
                self._consecutive_runtime_failures = 0
                return self.engine.generate(self.profile.fast_model, prompt, schema)
            raise

    def _generate_protocol(self, model: str, prompt: str):
        """Give each amortized protocol step one fresh retry for malformed JSON."""
        try:
            return self._generate(model, prompt, ReviewProtocol)
        except LocalAIOutputError:
            return self._generate(model, prompt, ReviewProtocol)

    def _downgrade_after_oom(self, model: str, reason: str) -> None:
        current_index = TIERS.index(self.profile.resolved_tier)
        if current_index == 0:
            raise LocalAIMemoryError(reason)
        lower_index = current_index - 1
        lower = TIERS[lower_index]
        candidate = resolve_runtime_profile(
            requested_tier=lower, resource_profile=self.profile.resource_profile
        )
        while lower_index > 0 and candidate.fast_model == model:
            lower_index -= 1
            lower = TIERS[lower_index]
            candidate = resolve_runtime_profile(
                requested_tier=lower, resource_profile=self.profile.resource_profile
            )
        self.runtime_downgrades.append(f"{reason}; downgraded from {model} to {lower}")
        self.profile = candidate

    def _validated_protocol(
        self,
        candidate: dict,
        model: str,
        research_question: str,
        inclusion: str,
        exclusion: str,
    ) -> ReviewProtocol:
        try:
            protocol = ReviewProtocol.model_validate(candidate).model_copy(
                update={"research_question": research_question, "model": model, "prompt_version": PROMPT_VERSION}
            )
            self._validate_protocol_structure(protocol)
            self._validate_user_criteria_sources(protocol, inclusion, exclusion)
            return protocol
        except (ValidationError, ValueError) as exc:
            repaired = self._generate(
                model,
                protocol_repair_prompt(candidate, str(exc), research_question, inclusion, exclusion),
                ReviewProtocol,
            )
            protocol = ReviewProtocol.model_validate(repaired.value).model_copy(
                update={"research_question": research_question, "model": model, "prompt_version": PROMPT_VERSION}
            )
            self._validate_protocol_structure(protocol)
            self._validate_user_criteria_sources(protocol, inclusion, exclusion)
            return protocol

    @staticmethod
    def _validate_protocol_structure(protocol: ReviewProtocol) -> None:
        inferred_exclusions = [
            criterion.id for criterion in protocol.criteria
            if criterion.source == "research_question" and criterion.kind == "exclusion"
        ]
        if inferred_exclusions:
            raise ValueError(
                "research-question scope must be represented as positive inclusion criteria; "
                "only explicit user exclusions may create exclusion criteria: "
                + ", ".join(inferred_exclusions)
            )

    @staticmethod
    def _validate_user_criteria_sources(protocol: ReviewProtocol, inclusion: str, exclusion: str) -> None:
        def entries(value: str) -> list[str]:
            return [item.strip(" -*\t") for line in value.splitlines() for item in line.split(";") if item.strip(" -*\t")]

        expected_inclusions = len(entries(inclusion))
        expected_exclusions = len(entries(exclusion))
        actual_inclusions = sum(c.source == "user" and c.kind == "inclusion" for c in protocol.criteria)
        actual_exclusions = sum(c.source == "user" and c.kind == "exclusion" for c in protocol.criteria)
        if actual_inclusions != expected_inclusions or actual_exclusions != expected_exclusions:
            raise ValueError("compiled protocol omitted one or more authoritative user criteria")
