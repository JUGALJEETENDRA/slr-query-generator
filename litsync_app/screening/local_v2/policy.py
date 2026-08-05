from __future__ import annotations

from collections import Counter

from .contracts import CriterionAssessment, PolicyResult, ScreeningProtocolV2


def _safe_structural_fallback(errors: list[str]) -> PolicyResult:
    return PolicyResult(
        decision="MAYBE",
        reason="Assessment structure is invalid, so the policy returned a safe MAYBE.",
        policy_errors=errors,
        safe_fallback=True,
    )


def derive_policy_decision(
    protocol: ScreeningProtocolV2,
    assessments: list[CriterionAssessment],
) -> PolicyResult:
    """Derive a final decision only from criterion-level validated relations."""

    protocol_ids = [criterion.id for criterion in protocol.criteria]
    protocol_id_set = set(protocol_ids)
    assessment_ids = [assessment.criterion_id for assessment in assessments]
    counts = Counter(assessment_ids)

    errors: list[str] = []

    for criterion_id in protocol_ids:
        if counts.get(criterion_id, 0) == 0:
            errors.append(f"missing assessment for criterion: {criterion_id}")

    seen_duplicates: set[str] = set()
    for criterion_id in assessment_ids:
        if counts[criterion_id] > 1 and criterion_id not in seen_duplicates:
            errors.append(f"duplicate assessment for criterion: {criterion_id}")
            seen_duplicates.add(criterion_id)

    seen_unknown: set[str] = set()
    for criterion_id in assessment_ids:
        if criterion_id not in protocol_id_set and criterion_id not in seen_unknown:
            errors.append(f"unknown assessment criterion: {criterion_id}")
            seen_unknown.add(criterion_id)

    if errors:
        return _safe_structural_fallback(errors)

    assessment_by_id = {
        assessment.criterion_id: assessment for assessment in assessments
    }
    decisive_ids: list[str] = []
    unresolved_ids: list[str] = []

    for criterion in protocol.criteria:
        assessment = assessment_by_id[criterion.id]

        if criterion.role == "REQUIRED_INCLUSION":
            if assessment.relation == "DIRECT_CONTRADICTION":
                decisive_ids.append(criterion.id)
            elif assessment.relation in {"MISSING_OR_UNCLEAR", "NOT_APPLICABLE"}:
                unresolved_ids.append(criterion.id)
            continue

        if assessment.relation == "DIRECT_SUPPORT":
            decisive_ids.append(criterion.id)
        elif (
            assessment.relation in {"MISSING_OR_UNCLEAR", "NOT_APPLICABLE"}
            and criterion.resolution_required
        ):
            unresolved_ids.append(criterion.id)

    if decisive_ids:
        return PolicyResult(
            decision="REJECT",
            reason=(
                "One or more criteria contain direct evidence that requires exclusion: "
                + ", ".join(decisive_ids)
                + "."
            ),
            decisive_criterion_ids=decisive_ids,
            unresolved_criterion_ids=unresolved_ids,
        )

    if unresolved_ids:
        return PolicyResult(
            decision="MAYBE",
            reason=(
                "No decisive exclusion was established, but required criteria remain "
                "unresolved: "
                + ", ".join(unresolved_ids)
                + "."
            ),
            unresolved_criterion_ids=unresolved_ids,
        )

    return PolicyResult(
        decision="KEEP",
        reason="All required inclusion criteria are supported and no exclusion trigger is present.",
    )
