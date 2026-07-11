from domain_vocabulary import join_terms


def build_evidence_ledger(
    required_dimensions,
    dimension_scores,
    dimension_terms,
    relation,
    negative,
):
    required = list(required_dimensions or [])
    met = [name for name in required if float(dimension_scores.get(name, 0.0)) >= 0.60]
    missing = [name for name in required if name not in met]
    coverage_ratio = len(met) / len(required) if required else 0.0
    uncertainty_score = max(0.0, 1.0 - coverage_ratio)
    contradiction_score = max(
        float(negative.get("contradiction_score", 0.0)),
        float(negative.get("explicit_exclusion_score", 0.0)),
    )
    strengths = [float(value or 0.0) for value in dimension_scores.values()]
    relevance = (
        (sum(strengths) / len(strengths)) if strengths else 0.0
    ) - (0.45 * contradiction_score)
    strongest_dimension = max(
        dimension_scores,
        key=lambda key: float(dimension_scores.get(key, 0.0)),
        default="",
    )
    return {
        "method_evidence_strength": float(dimension_scores.get("method_tool", dimension_scores.get("method", 0.0))),
        "task_evidence_strength": float(dimension_scores.get("review_workflow_process", dimension_scores.get("task_or_outcome", 0.0))),
        "context_evidence_strength": float(dimension_scores.get("review_context", dimension_scores.get("context", 0.0))),
        "outcome_evidence_strength": float(dimension_scores.get("outcome", dimension_scores.get("task_or_outcome", 0.0))),
        "relation_evidence_strength": float(dimension_scores.get("relation", 0.0)),
        "evidence_type_strength": float(dimension_scores.get("evidence_type", 0.0)),
        "method_evidence_terms": dimension_terms.get("method", ""),
        "task_evidence_terms": dimension_terms.get("task", ""),
        "context_evidence_terms": dimension_terms.get("context", ""),
        "outcome_evidence_terms": dimension_terms.get("outcome", ""),
        "relation_evidence_terms": relation.get("relation_evidence_terms", ""),
        "evidence_type_terms": dimension_terms.get("evidence_type", ""),
        "required_dimensions": join_terms(required),
        "required_dimensions_met": join_terms(met),
        "required_dimensions_missing": join_terms(missing),
        "evidence_coverage_count": len(met),
        "evidence_coverage_ratio": round(coverage_ratio, 4),
        "contradiction_score": round(contradiction_score, 4),
        "contradiction_detected": contradiction_score >= 0.75,
        "contradiction_reason": negative.get("contradiction_reason", ""),
        "external_domain_score": float(negative.get("external_domain_score", 0.0)),
        "subject_only_score": float(negative.get("subject_only_score", 0.0)),
        "paper_type_only_score": float(negative.get("paper_type_only_score", 0.0)),
        "background_only_score": float(negative.get("background_only_score", 0.0)),
        "explicit_exclusion_score": float(negative.get("explicit_exclusion_score", 0.0)),
        "uncertainty_score": round(uncertainty_score, 4),
        "uncertainty_reason": (
            "Missing required dimensions: " + join_terms(missing) if missing else ""
        ),
        "abstract_insufficient": bool(negative.get("abstract_insufficient", False)),
        "relation_unclear": not relation.get("relation_match", False) and not relation.get("relation_conflict", False),
        "weak_context": float(dimension_scores.get("context", dimension_scores.get("review_context", 0.0))) < 0.60,
        "weak_method": float(dimension_scores.get("method", dimension_scores.get("method_tool", 0.0))) < 0.60,
        "weak_task": float(dimension_scores.get("task_or_outcome", dimension_scores.get("review_workflow_process", 0.0))) < 0.60,
        "final_relevance_score": round(max(0.0, min(1.0, relevance)), 4),
        "strongest_inclusion_evidence": (
            f"{strongest_dimension}: {dimension_terms.get(strongest_dimension, '')}"
            if strongest_dimension else ""
        ),
        "strongest_exclusion_evidence": negative.get("contradiction_reason", ""),
    }
