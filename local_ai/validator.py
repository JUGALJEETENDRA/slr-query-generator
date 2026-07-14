from __future__ import annotations

from .contracts import PaperAssessment, ReviewProtocol, ValidationReport
from .evidence import evidence_lookup


def validate_assessment(
    assessment: PaperAssessment,
    protocol: ReviewProtocol,
    title: str,
    abstract: str,
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    protocol_by_id = {criterion.id: criterion for criterion in protocol.criteria}
    seen: set[str] = set()
    exact_quotes = 0
    evidence_by_id: dict[str, int] = {}
    source_units = evidence_lookup(title, abstract)

    for item in assessment.criteria:
        if item.criterion_id not in protocol_by_id:
            errors.append(f"unknown criterion id: {item.criterion_id}")
            continue
        if item.criterion_id in seen:
            errors.append(f"duplicate criterion assessment: {item.criterion_id}")
        seen.add(item.criterion_id)
        valid_count = 0
        for span in item.evidence:
            unit = source_units.get(span.evidence_id)
            if unit is None:
                errors.append(f"unknown evidence id for {item.criterion_id}: {span.evidence_id}")
            elif unit["source"] != span.source:
                errors.append(f"evidence source mismatch for {item.criterion_id}: {span.evidence_id}")
            else:
                exact_quotes += 1
                valid_count += 1
        evidence_by_id[item.criterion_id] = valid_count

    missing = set(protocol_by_id) - seen
    if missing:
        errors.append("missing criterion assessments: " + ", ".join(sorted(missing)))

    verdicts = {item.criterion_id: item.verdict for item in assessment.criteria}
    required_inclusions = [
        c.id for c in protocol.criteria if c.kind == "inclusion" and c.required
    ]
    met_exclusions = [
        c.id for c in protocol.criteria
        if c.kind == "exclusion" and verdicts.get(c.id) == "MET"
    ]
    unmet_required = [
        criterion_id for criterion_id in required_inclusions
        if verdicts.get(criterion_id) == "NOT_MET"
    ]
    unclear_required = [
        criterion_id for criterion_id in required_inclusions
        if verdicts.get(criterion_id) == "UNCLEAR"
    ]

    decisive_ids: list[str] = []
    if assessment.decision == "KEEP":
        not_met = [cid for cid in required_inclusions if verdicts.get(cid) != "MET"]
        if not_met:
            errors.append("KEEP lacks MET verdicts for required criteria: " + ", ".join(not_met))
        if met_exclusions:
            errors.append("KEEP conflicts with met exclusions: " + ", ".join(met_exclusions))
        decisive_ids = required_inclusions
    elif assessment.decision == "REJECT":
        decisive_ids = met_exclusions + unmet_required
        if not decisive_ids:
            errors.append("REJECT lacks a met exclusion or evidence-backed unmet required criterion")
    else:
        if not (unclear_required or assessment.uncertainty or assessment.missing_information):
            warnings.append("MAYBE does not identify its uncertainty")

    unsupported = [cid for cid in decisive_ids if evidence_by_id.get(cid, 0) == 0]
    if unsupported:
        errors.append("decisive criteria lack exact source evidence: " + ", ".join(unsupported))
    if assessment.decision in {"KEEP", "REJECT"} and assessment.confidence < 0.75:
        warnings.append("definitive decision has confidence below 0.75")

    return ValidationReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        exact_quote_count=exact_quotes,
        decisive_evidence_count=sum(evidence_by_id.get(cid, 0) for cid in decisive_ids),
    )
