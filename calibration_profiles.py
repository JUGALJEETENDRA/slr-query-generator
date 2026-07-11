DEFAULT_PROFILE = {
    "keep_threshold": 0.72,
    "maybe_threshold": 0.45,
    "contradiction_threshold": 0.75,
    "relation_required_for_keep": True,
    "stage2_demote_requires_explicit_contradiction": True,
    "false_keep_suppression_enabled": True,
    "semantic_rescue_enabled": True,
    "evidence_first_enabled": True,
}

PROFILES = {
    "review_workflow_automation": {
        **DEFAULT_PROFILE,
        "keep_threshold": 0.78,
        "maybe_threshold": 0.38,
    },
    "technology_for_outcome_in_domain": {
        **DEFAULT_PROFILE,
        "keep_threshold": 0.65,
        "maybe_threshold": 0.35,
        "false_keep_suppression_enabled": False,
    },
    "method_for_task_in_domain": {
        **DEFAULT_PROFILE,
        "keep_threshold": 0.65,
        "maybe_threshold": 0.35,
        "false_keep_suppression_enabled": False,
    },
    "broad_scoping_review": {
        **DEFAULT_PROFILE,
        "keep_threshold": 0.70,
        "maybe_threshold": 0.25,
    },
}


def calibration_for(rq_type):
    return dict(PROFILES.get(str(rq_type or ""), DEFAULT_PROFILE))
