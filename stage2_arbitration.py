from calibration_profiles import calibration_for


def arbitrate_stage2(stage1_decision, stage2_result, rq_contract, keep_allowed=True):
    stage1 = str(stage1_decision or "").upper()
    stage2 = str(stage2_result.get("decision", stage1)).upper()
    profile = calibration_for(rq_contract.get("rq_type", ""))
    relation_conflict = bool(stage2_result.get("relation_conflict"))
    contradiction = float(stage2_result.get("contradiction_score", 0.0) or 0.0)
    explicit = (
        relation_conflict
        and contradiction >= profile["contradiction_threshold"]
    )
    task_sense = str(stage2_result.get("workflow_task_sense", ""))
    intent_score = float(stage2_result.get("implied_workflow_intent_score", 0.0) or 0.0)
    strong_intent = bool(stage2_result.get("strong_implied_workflow_intent")) or intent_score >= 0.85
    medium_intent = bool(stage2_result.get("medium_implied_workflow_intent")) or intent_score >= 0.60
    workflow_direction = bool(stage2_result.get("workflow_direction_detected"))
    subject_direction = bool(stage2_result.get("subject_review_direction_detected"))
    directional_workflow = bool(stage2_result.get("directional_uses_ai_for_review_workflow"))
    directional_external = bool(stage2_result.get("directional_is_review_about_ai_external_domain"))
    directional_confidence = float(stage2_result.get("directional_confidence", 0.0) or 0.0)
    observed_relation = str(stage2_result.get("paper_observed_relation", ""))
    task_sense_conflict = bool(
        task_sense == "external_domain_task"
        and not strong_intent
        and observed_relation in {
            "external_domain_review",
            "technology_reviewed_as_subject",
            "paper_type_only_review",
        }
    )
    explicit = explicit or task_sense_conflict

    final = stage1
    action = "preserve_stage1"
    reason = "Stage 2 did not provide stronger decisive evidence."
    override = False
    directional_override = False
    directional_reason = ""
    directional_conflict = False
    directional_rescue = False

    if directional_workflow and directional_confidence >= 0.70 and not directional_external:
        if stage1 == "MAYBE" or stage2 == "KEEP":
            final = "KEEP"
            action = "directional_promote_to_keep"
            reason = "Stage 2 directional judge found AI/tool use for the review workflow."
        else:
            final = "MAYBE"
            action = "directional_rescue_to_maybe"
            reason = "Stage 2 directional judge rescued workflow-use evidence to MAYBE."
        override = final != stage1
        directional_override = True
        directional_reason = reason
        directional_rescue = True
    elif directional_external and not directional_workflow and directional_confidence >= 0.70:
        directional_conflict = True
        if stage1 == "KEEP":
            final = "MAYBE"
            action = "directional_demote_external_domain"
            reason = "Stage 2 directional judge found external-domain AI review direction."
            override = True
            directional_override = True
            directional_reason = reason
        else:
            final = stage1
            action = "directional_preserve_external_domain"
            reason = "Stage 2 external-domain direction blocked promotion."
            directional_reason = reason
    elif (
        stage2 == "KEEP"
        and strong_intent
        and workflow_direction
        and not subject_direction
        and not relation_conflict
    ):
        final = "KEEP"
        action = "promote_to_keep"
        reason = "Stage 2 strong implied review-workflow intent supports promotion."
        override = final != stage1
    elif stage2 == "KEEP" and keep_allowed:
        final = "KEEP"
        action = "promote_to_keep"
        reason = "Stage 2 found complete evidence and a matching relation."
        override = final != stage1
    elif (
        stage1 == "MAYBE"
        and strong_intent
        and workflow_direction
        and not subject_direction
        and not relation_conflict
    ):
        final = "KEEP"
        action = "promote_to_keep"
        reason = "Strong implied review-workflow intent supports promotion."
        override = True
    elif stage2 == "REJECT" and explicit:
        final = "REJECT"
        action = "demote_to_reject"
        reason = (
            stage2_result.get("relation_mismatch_reason")
            or stage2_result.get("rejection_reason_category")
            or "Stage 2 found an explicit contradiction."
        )
        override = final != stage1
    elif stage2 == "MAYBE" or (stage2 == "REJECT" and not explicit):
        final = "MAYBE" if stage1 == "MAYBE" else stage1
        action = "preserve_uncertainty"
        reason = "Stage 2 uncertainty was preserved without explicit contradiction."
        override = final != stage1
        if medium_intent and workflow_direction and not subject_direction:
            final = "MAYBE"
            action = "workflow_intent_rescue"
            reason = "Medium implied review-workflow intent rescued the paper to MAYBE."
            override = final != stage1

    return {
        "decision": final,
        "final_arbitration_action": action,
        "stage2_override_applied": override,
        "stage2_override_reason": reason,
        "stage2_explicit_contradiction": explicit,
        "stage2_uncertainty_preserved": action == "preserve_uncertainty",
        "stage2_demoted_to_reject": action == "demote_to_reject",
        "stage2_promoted_to_keep": action == "promote_to_keep",
        "stage2_task_sense_conflict": task_sense_conflict,
        "stage2_task_object_mismatch": bool(stage2_result.get("task_object_mismatch")),
        "stage2_preserved_due_to_ambiguous_task_sense": bool(
            action == "preserve_uncertainty" and task_sense == "ambiguous_task"
        ),
        "stage2_workflow_intent_rescue": action in {
            "workflow_intent_rescue",
            "promote_to_keep",
        } and (medium_intent or strong_intent),
        "stage2_workflow_intent_rescue_reason": (
            reason if action in {"workflow_intent_rescue", "promote_to_keep"} else ""
        ),
        "stage2_subject_review_direction_conflict": bool(
            subject_direction and not workflow_direction
        ),
        "stage2_relation_direction_used": stage2_result.get("relation_direction", "none"),
        "stage2_directional_override_applied": directional_override,
        "stage2_directional_override_reason": directional_reason,
        "stage2_directional_conflict": directional_conflict,
        "stage2_directional_rescue": directional_rescue,
    }
