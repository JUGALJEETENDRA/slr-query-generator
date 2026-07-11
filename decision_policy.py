from calibration_profiles import calibration_for


class DecisionPolicy:
    def decide(self, rq_contract, ledger, relation, review_intent=None, stage=1):
        review_intent = review_intent or {}
        rq_type = rq_contract.get("rq_type", "")
        profile = calibration_for(rq_type)
        required_count = len(rq_contract.get("rq_required_dimensions", []))
        coverage = int(ledger.get("evidence_coverage_count", 0))
        conflict = bool(relation.get("relation_conflict"))
        contradiction = float(ledger.get("contradiction_score", 0.0))
        relation_match = bool(relation.get("relation_match"))
        workflow_use = float(relation.get("workflow_use_score", 0.0))
        implied_intent = float(relation.get("implied_workflow_score", 0.0))
        subject_direction = bool(review_intent.get("subject_review_direction_detected"))
        workflow_direction = bool(review_intent.get("workflow_direction_detected"))
        observed_relation = str(relation.get("paper_observed_relation", ""))
        task_sense = str(review_intent.get("workflow_task_sense", ""))
        external_terms = str(review_intent.get("external_domain_terms", "") or "").strip()
        explicit_external_task = bool(review_intent.get("external_domain_task_detected"))
        weak_subject_review_specificity = bool(
            rq_type == "review_workflow_automation"
            and observed_relation in {
                "external_domain_review",
                "technology_reviewed_as_subject",
            }
            and task_sense in {"no_task_detected", "ambiguous_task", ""}
            and not external_terms
            and not explicit_external_task
            and not workflow_direction
            and workflow_use < 0.75
        )

        false_keep_risk = max(
            float(relation.get("subject_only_score", 0.0)),
            float(relation.get("paper_type_only_score", 0.0)),
            float(relation.get("external_domain_topic_score", 0.0)),
            0.65 if rq_type == "review_workflow_automation" and workflow_use < 0.75 else 0.0,
        )
        keep_gate = bool(
            not profile["relation_required_for_keep"]
            or (
                relation_match
                and not conflict
                and (
                    rq_type != "review_workflow_automation"
                    or workflow_use >= 0.75
                )
            )
        )

        if (
            conflict
            and contradiction >= profile["contradiction_threshold"]
            and weak_subject_review_specificity
            and coverage >= 2
        ):
            decision = "MAYBE"
            path = "subject_review_uncertainty_maybe"
            rejection = ""
        elif conflict and contradiction >= profile["contradiction_threshold"]:
            decision = "REJECT"
            path = "relation_conflict_reject"
            rejection = relation.get("relation_mismatch_reason") or "explicit relation conflict"
        elif coverage >= required_count and keep_gate:
            decision = "KEEP"
            path = "relation_keep_confirmed"
            rejection = ""
        elif (
            coverage >= max(2, required_count - 1)
            or (rq_type == "review_workflow_automation" and coverage >= 2)
        ) and not conflict:
            decision = "MAYBE"
            path = "relation_uncertain_maybe"
            rejection = ""
        elif rq_type == "broad_scoping_review" and coverage >= 1 and not conflict:
            decision = "MAYBE"
            path = "broad_scoping_partial_maybe"
            rejection = ""
        else:
            decision = "REJECT"
            path = "insufficient_required_evidence_reject"
            rejection = (
                ledger.get("uncertainty_reason")
                or relation.get("relation_mismatch_reason")
                or "insufficient evidence"
            )

        workflow_intent_rescue = False
        workflow_intent_rescue_reason = ""
        if (
            rq_type == "review_workflow_automation"
            and implied_intent >= 0.60
            and workflow_direction
            and not subject_direction
            and not conflict
        ):
            if decision == "REJECT":
                decision = "MAYBE"
                path = "workflow_intent_rescue_maybe"
                workflow_intent_rescue = True
                workflow_intent_rescue_reason = (
                    "Medium or strong AI-for-review direction supports review-workflow relevance."
                )
            elif decision in {"KEEP", "MAYBE"}:
                workflow_intent_rescue = True
                workflow_intent_rescue_reason = (
                    "Positive AI-for-review phrase direction supports the decision."
                )

        suppression = bool(
            profile["false_keep_suppression_enabled"]
            and rq_type == "review_workflow_automation"
            and not keep_gate
            and false_keep_risk >= 0.60
        )
        suppression_reason = ""
        if suppression:
            if relation.get("external_domain_topic_score", 0.0) >= 0.75:
                suppression_reason = "external-domain AI review without review-workflow tool use"
            elif relation.get("subject_only_score", 0.0) >= 0.75:
                suppression_reason = "AI/LLM is the reviewed subject, not a review-workflow tool"
            elif relation.get("paper_type_only_score", 0.0) >= 0.75:
                suppression_reason = "systematic review is only the paper type"
            else:
                suppression_reason = "required workflow-use relation is missing"
            if decision == "KEEP":
                decision = "MAYBE"
                path = "keep_downgraded_missing_workflow_use_maybe"

        suspicious_keep = decision == "KEEP" and (not relation_match or conflict)
        suspicious_reject = decision == "REJECT" and coverage >= required_count and relation_match
        return {
            "decision": decision,
            "decision_path": path,
            "rejection_reason_category": rejection,
            "suspicious_keep": suspicious_keep,
            "suspicious_reject": suspicious_reject,
            "false_keep_risk_score": round(false_keep_risk, 4),
            "keep_suppression_applied": suppression,
            "keep_suppression_reason": suppression_reason,
            "keep_required_relation_missing": not relation_match,
            "external_domain_subject_only": bool(
                relation.get("external_domain_topic_score", 0.0) >= 0.75
                and workflow_use < 0.75
            ),
            "workflow_use_required_for_keep": rq_type == "review_workflow_automation",
            "workflow_use_evidence_missing": (
                rq_type == "review_workflow_automation" and workflow_use < 0.75
            ),
            "subject_only_overrides_keep": bool(
                relation.get("subject_only_score", 0.0) >= 0.75
                and workflow_use < 0.75
            ),
            "relation_keep_gate_passed": keep_gate,
            "policy_stage": stage,
            "workflow_intent_rescue_applied": workflow_intent_rescue,
            "workflow_intent_rescue_reason": workflow_intent_rescue_reason,
            "weak_subject_review_specificity": weak_subject_review_specificity,
        }


DEFAULT_DECISION_POLICY = DecisionPolicy()
