from __future__ import annotations

from typing import Callable

from .contracts import (
    BenchmarkResult,
    BenchmarkSpec,
    BenchmarkVerdict,
    GateFailure,
    GateOutcome,
    ProvenanceClass,
    Threshold,
)


COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    "ge": lambda observed, expected: observed >= expected,
    "gt": lambda observed, expected: observed > expected,
    "le": lambda observed, expected: observed <= expected,
    "lt": lambda observed, expected: observed < expected,
    "eq": lambda observed, expected: observed == expected,
}


def _threshold_failure(
    name: str,
    threshold: Threshold,
    result: BenchmarkResult,
) -> GateFailure | None:
    metric = result.metrics.get(name)
    if metric is None:
        return GateFailure(rule=name, reason="required metric is missing")
    if metric.value is None:
        return GateFailure(
            rule=name,
            reason="required metric is undefined",
            numerator=metric.numerator,
            denominator=metric.denominator,
            population_scope=metric.population_scope,
            metric_definition_version=metric.metric_definition_version,
        )
    observed = float(metric.value)
    if COMPARATORS[threshold.comparator](observed, threshold.value):
        return None
    return GateFailure(
        rule=name,
        reason="release threshold was not satisfied",
        observed=observed,
        comparator=threshold.comparator,
        threshold=threshold.value,
        numerator=metric.numerator,
        denominator=metric.denominator,
        population_scope=metric.population_scope,
        metric_definition_version=metric.metric_definition_version,
    )


def evaluate_gate(spec: BenchmarkSpec, result: BenchmarkResult) -> GateOutcome:
    provenance = result.provenance
    if not provenance.valid or provenance.classification == ProvenanceClass.INVALID_PROVENANCE:
        return GateOutcome(
            verdict=BenchmarkVerdict.INVALID,
            failures=[
                GateFailure(rule="provenance", reason=reason)
                for reason in provenance.reasons
            ] or [GateFailure(rule="provenance", reason="run provenance is invalid")],
        )
    failures: list[GateFailure] = []
    expected_protocol = spec.expected_protocol_identity
    expected_assessment = spec.expected_assessment_identity
    identity_checks = {
        "protocol_id": (provenance.protocol_id, expected_protocol.protocol_id),
        "protocol_cache_version": (
            provenance.protocol_cache_version,
            expected_protocol.protocol_cache_version,
        ),
        "architecture_version": (
            provenance.architecture_version,
            expected_assessment.architecture_version,
        ),
        "assessment_prompt_version": (
            provenance.assessment_prompt_version,
            expected_assessment.assessment_prompt_version,
        ),
        "assessment_cache_version": (
            provenance.assessment_cache_version,
            expected_assessment.assessment_cache_version,
        ),
    }
    for name, (observed, expected) in identity_checks.items():
        if observed != expected:
            failures.append(GateFailure(
                rule=name,
                reason="run identity does not match the benchmark release identity",
                observed=observed,
                comparator="eq",
                threshold=expected,
            ))
    for group in (
        spec.release_thresholds.quality,
        spec.release_thresholds.reliability,
    ):
        for name, threshold in group.items():
            failure = _threshold_failure(name, threshold, result)
            if failure is not None:
                failures.append(failure)

    classification = provenance.classification
    if classification in {
        ProvenanceClass.PARTIALLY_RESUMED,
        ProvenanceClass.FULLY_RESUMED,
    }:
        failures.append(GateFailure(
            rule="cold_release",
            reason=f"checkpoint-resumed run cannot pass a cold release gate ({classification.value})",
        ))
        return GateOutcome(verdict=BenchmarkVerdict.FAIL, failures=failures)
    if failures:
        return GateOutcome(verdict=BenchmarkVerdict.FAIL, failures=failures)
    if classification == ProvenanceClass.WARM_CACHE:
        return GateOutcome(
            verdict=BenchmarkVerdict.PROVISIONAL,
            failures=[GateFailure(
                rule="cold_release",
                reason="assessment-cache reuse prevents a cold release verdict",
            )],
        )
    return GateOutcome(verdict=BenchmarkVerdict.PASS, failures=[])
