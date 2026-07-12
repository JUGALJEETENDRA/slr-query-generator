from __future__ import annotations

import concurrent.futures
import os
import time
from typing import Any

from llm_structured_judge import (
    EMPTY_LLM_JUDGE,
    directional_trigger_diagnostics,
    judge_with_llm,
    should_run_llm_judge,
)
from model_registry import configured_model_names, get_runtime_status, model_judges_enabled, model_judge_mode, bool_config
from nli_judge import judge_nli
from reranker_judge import judge_reranker
from runtime_config import get_model_judge_config
from zeroshot_relation_judge import judge_zero_shot_relation


MODEL_DIAGNOSTIC_FIELDS = {
    "model_judges_enabled": False,
    "model_judge_mode": "off",
    "model_judge_models_used": "",
    "model_judge_runtime_source": "fallback_not_attempted",
    "model_profile": "",
    "model_real_models_loaded": False,
    "model_fallback_reason": "",
    "model_timing_seconds": 0.0,
    "model_judge_fallback_used": False,
    "model_judge_error": "",
    "model_reranker_relevance_score": 0.0,
    "model_reranker_negative_score": 0.0,
    "model_reranker_positive_raw_score": 0.0,
    "model_reranker_negative_raw_score": 0.0,
    "model_reranker_margin": 0.0,
    "model_reranker_decision_hint": "",
    "reranker_runtime_source": "",
    "reranker_timing_seconds": 0.0,
    "reranker_timeout": False,
    "nli_positive_entailment_score": 0.0,
    "nli_negative_entailment_score": 0.0,
    "nli_contradiction_score": 0.0,
    "nli_margin": 0.0,
    "nli_decision_hint": "",
    "nli_top_positive_hypothesis": "",
    "nli_top_negative_hypothesis": "",
    "nli_positive_hypothesis_scores": "",
    "nli_negative_hypothesis_scores": "",
    "nli_runtime_source": "",
    "nli_timing_seconds": 0.0,
    "nli_timeout": False,
    "nli_ignored_reason": "",
    "zeroshot_relation_label": "",
    "zeroshot_relation_score": 0.0,
    "zeroshot_ai_tool_for_review_score": 0.0,
    "zeroshot_subject_review_score": 0.0,
    "zeroshot_top_labels": "",
    "zeroshot_runtime_source": "",
    "zeroshot_timing_seconds": 0.0,
    "zeroshot_timeout": False,
    "zeroshot_ignored_reason": "",
    **EMPTY_LLM_JUDGE,
    "model_positive_score": 0.0,
    "model_negative_score": 0.0,
    "model_conflict_score": 0.0,
    "model_uncertainty_score": 0.0,
    "model_decision_hint": "",
    "model_fusion_reason": "",
    "model_fusion_action": "disabled",
    "model_fusion_blocked_reason": "",
    "fast_mode_current_equivalence_blocked_reason": "",
    "fast_preserved_current_uncertainty": False,
    "model_primary_signal": "fallback",
    "model_primary_margin": 0.0,
    "model_promoted_from_reject": False,
    "model_promoted_from_maybe": False,
    "model_demoted_from_keep": False,
}


def apply_model_score_fusion(
    *,
    rq_frame: dict[str, Any],
    paper_frame: dict[str, Any],
    deterministic_result: dict[str, Any],
    research_question: str = "",
    model: str = "qwen2.5:3b",
    inference_engine=None,
    mode: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_mode = model_judge_mode(mode)
    cfg = get_model_judge_config(selected_mode)
    started_at = time.perf_counter()
    diagnostics = dict(MODEL_DIAGNOSTIC_FIELDS)
    diagnostics["model_judge_mode"] = selected_mode
    diagnostics["model_profile"] = cfg["model_judge_profile"]
    rq_type = str(rq_frame.get("review_question_type") or rq_frame.get("question_type") or rq_frame.get("rq_type") or "")
    title = str(paper_frame.get("source_title") or paper_frame.get("primary_subject") or "")
    abstract = str(paper_frame.get("source_abstract") or "")
    trigger_diagnostics = directional_trigger_diagnostics(
        rq_frame=rq_frame,
        deterministic_result=deterministic_result,
        title=title,
        abstract=abstract,
        aggressive_gating=(
            cfg.get("screening_pipeline_mode") == "two_pass_fast"
            and cfg.get("enable_aggressive_llm_gating")
        ),
    )
    diagnostics.update(trigger_diagnostics)
    if cfg.get("screening_pipeline_mode") == "two_pass_fast":
        route = str(diagnostics.get("llm_route") or "")
        deterministic_decision = str(deterministic_result.get("decision") or "").upper()
        equivalence_reason = ""
        if route == "skipped_high_confidence_workflow" and deterministic_decision == "KEEP":
            equivalence_reason = "workflow_keep_would_demote"
        elif route == "skipped_high_confidence_external" and deterministic_decision == "REJECT":
            equivalence_reason = "external_reject_mixed_evidence"
        if equivalence_reason:
            diagnostics["fast_mode_current_equivalence_blocked_reason"] = equivalence_reason
            diagnostics["llm_route"] = "llm_required_current_equivalence"
            diagnostics["llm_directional_triggered"] = True
            diagnostics["llm_required_reason"] = equivalence_reason
            diagnostics["llm_directional_skipped_reason"] = ""
            diagnostics["llm_skip_reason"] = ""

    if not model_judges_enabled(selected_mode):
        return deterministic_result, diagnostics

    diagnostics["model_judges_enabled"] = True
    names = configured_model_names()
    diagnostics["model_judge_models_used"] = "; ".join(f"{k}:{v}" for k, v in names.items() if v)

    if cfg["enable_reranker_judge"] and selected_mode in {"fast", "balanced", "full"}:
        _safe_update(diagnostics, "reranker", judge_reranker, cfg["model_judge_timeout_seconds"], research_question, title, abstract, selected_mode)
    if cfg["enable_nli_judge"] and selected_mode in {"balanced", "full"}:
        _safe_update(diagnostics, "nli", judge_nli, cfg["model_judge_timeout_seconds"], title, abstract, selected_mode)
    elif cfg["model_judge_profile"] == "light":
        diagnostics["nli_ignored_reason"] = "disabled_by_light_profile"
    if cfg["enable_zero_shot_judge"] and selected_mode in {"balanced", "full"}:
        _safe_update(diagnostics, "zeroshot", judge_zero_shot_relation, cfg["model_judge_timeout_seconds"], title, abstract, selected_mode)
    elif cfg["model_judge_profile"] == "light":
        diagnostics["zeroshot_ignored_reason"] = "disabled_by_light_profile"

    _mark_uniform_auxiliary_judges(diagnostics)

    runtime_status = get_runtime_status()
    if diagnostics["model_judge_runtime_source"] == "fallback_not_attempted":
        diagnostics["model_judge_runtime_source"] = runtime_status.get("runtime_source", "fallback_not_attempted")
    diagnostics["model_real_models_loaded"] = bool(runtime_status.get("real_models_loaded", False))
    if not diagnostics["model_fallback_reason"]:
        diagnostics["model_fallback_reason"] = str(runtime_status.get("fallback_reason", ""))
    if diagnostics["model_judge_runtime_source"].startswith("fallback"):
        diagnostics["model_judge_fallback_used"] = True

    model_scores = _fuse_scores(diagnostics)
    diagnostics.update(model_scores)
    pre_llm_positive = float(model_scores.get("model_positive_score") or 0.0)
    pre_llm_negative = float(model_scores.get("model_negative_score") or 0.0)
    reranker_positive = float(diagnostics.get("model_reranker_relevance_score") or 0.0)
    reranker_negative = float(diagnostics.get("model_reranker_negative_score") or 0.0)
    if diagnostics.get("model_judge_error"):
        diagnostics["model_fusion_blocked_reason"] = "model_judge_exception"
    elif reranker_positive >= 0.95 and reranker_negative >= 0.95 and abs(reranker_positive - reranker_negative) < 0.10:
        diagnostics["model_fusion_blocked_reason"] = "high_high_small_margin_uncertain"
    elif max(pre_llm_positive, pre_llm_negative) > 0.0 and abs(pre_llm_positive - pre_llm_negative) < 0.10:
        diagnostics["model_fusion_blocked_reason"] = "model_disagreement"
    elif (
        model_scores.get("model_primary_signal") == "reranker"
        and pre_llm_positive >= 0.62
        and pre_llm_negative < 0.68
    ):
        diagnostics["model_fusion_blocked_reason"] = "reranker_without_directional_support"
    primary = diagnostics.get("model_primary_signal", "fallback")
    if primary in {"reranker", "nli", "zeroshot"}:
        source = diagnostics.get(f"{primary}_runtime_source") or diagnostics.get("model_judge_runtime_source")
        if source:
            diagnostics["model_judge_runtime_source"] = source
        diagnostics["model_real_models_loaded"] = source == "hf_model"
    elif primary == "fallback":
        diagnostics["model_real_models_loaded"] = False

    llm_for_smoke = os.getenv("ENABLE_LLM_JUDGE_FOR_SMOKE", "").strip().lower() in {"1", "true", "yes", "on"}
    llm_route = str(diagnostics.get("llm_route") or "")
    llm_route_requires_judge = llm_route.startswith("llm_required")
    equivalence_judge_required = llm_route == "llm_required_current_equivalence"
    fast_pass2_judge_required = bool(
        cfg.get("screening_pipeline_mode") == "two_pass_fast" and llm_route_requires_judge
    )
    if (
        (
            bool_config("ENABLE_LLM_JUDGE", False)
            or llm_for_smoke
            or equivalence_judge_required
            or fast_pass2_judge_required
        )
        and selected_mode in {"balanced", "full"}
        and (
            llm_for_smoke
            or llm_route_requires_judge
            or diagnostics.get("llm_directional_triggered")
            or (
                not llm_route.startswith("skipped_")
                and should_run_llm_judge(str(deterministic_result.get("decision", "")), model_scores)
            )
        )
    ):
        before_error = diagnostics.get("model_judge_error", "")
        _safe_update(diagnostics, "llm", judge_with_llm, cfg["model_judge_timeout_seconds"], title, abstract, research_question, model, inference_engine)
        _preserve_deterministic_trigger_fields(diagnostics, trigger_diagnostics)
        if not diagnostics.get("llm_directional_judge_used"):
            _apply_directional_timeout_fallback(diagnostics)
        if diagnostics.get("model_judge_error", "") == before_error:
            diagnostics.update(_llm_adjusted_scores(diagnostics))
        if diagnostics.get("llm_directional_judge_used"):
            _apply_directional_sanity_override(diagnostics)
            diagnostics["model_judge_runtime_source"] = "llm"
            diagnostics["model_real_models_loaded"] = True
            diagnostics["model_primary_signal"] = "llm"
            diagnostics["model_primary_margin"] = round(
                abs(
                    float(diagnostics.get("model_positive_score") or 0.0)
                    - float(diagnostics.get("model_negative_score") or 0.0)
                ),
                4,
            )

    fused = dict(deterministic_result)
    if (
        llm_for_smoke
        and diagnostics.get("llm_directional_judge_used")
        and not diagnostics.get("model_judge_error")
        and diagnostics.get("model_fusion_blocked_reason") == "reranker_without_directional_support"
    ):
        diagnostics["model_fusion_blocked_reason"] = ""
    if diagnostics.get("model_judge_error"):
        diagnostics["model_fusion_blocked_reason"] = "model_judge_exception"
    elif (
        str(deterministic_result.get("decision", "")).upper() == "MAYBE"
        and not diagnostics.get("model_fusion_blocked_reason")
        and not diagnostics.get("llm_directional_judge_used")
    ):
        if diagnostics.get("model_primary_signal") == "reranker" and pre_llm_positive > pre_llm_negative:
            diagnostics["model_fusion_blocked_reason"] = "reranker_without_directional_support"
        else:
            diagnostics["model_fusion_blocked_reason"] = "missing_directional_support"
    action, reason, decision = _decision_action(
        rq_type,
        str(deterministic_result.get("decision", "")).upper(),
        diagnostics,
        deterministic_result,
    )
    if diagnostics.get("model_judge_error") and not diagnostics.get("llm_directional_judge_used"):
        action = "fallback_error_preserve"
        reason = "Model judge failed; deterministic decision preserved."
        decision = str(deterministic_result.get("decision", "")).upper()
    diagnostics["model_fusion_action"] = action
    diagnostics["model_fusion_reason"] = reason
    diagnostics["model_decision_hint"] = decision
    diagnostics["model_promoted_from_reject"] = deterministic_result.get("decision") == "REJECT" and decision != "REJECT"
    diagnostics["model_promoted_from_maybe"] = deterministic_result.get("decision") == "MAYBE" and decision == "KEEP"
    diagnostics["model_demoted_from_keep"] = deterministic_result.get("decision") == "KEEP" and decision != "KEEP"
    diagnostics["fast_preserved_current_uncertainty"] = bool(
        diagnostics.get("fast_mode_current_equivalence_blocked_reason") == "external_reject_mixed_evidence"
        and str(deterministic_result.get("decision") or "").upper() == "REJECT"
        and decision == "MAYBE"
    )

    if action != "preserve":
        fused["decision"] = decision
        fused["decision_path"] = f"model_fusion_{action}"
        fused["reason"] = reason
    diagnostics["model_timing_seconds"] = round(time.perf_counter() - started_at, 4)
    return fused, diagnostics


def _safe_update(diagnostics: dict[str, Any], judge_name: str, judge, timeout_seconds: float, *args) -> None:
    started_at = time.perf_counter()
    try:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(judge, *args)
        try:
            diagnostics.update(future.result(timeout=max(0.1, float(timeout_seconds or 8.0))))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        diagnostics[f"{judge_name}_timing_seconds"] = round(time.perf_counter() - started_at, 4)
        status = get_runtime_status()
        if not diagnostics.get(f"{judge_name}_runtime_source"):
            diagnostics[f"{judge_name}_runtime_source"] = status.get("runtime_source", "")
    except concurrent.futures.TimeoutError:
        diagnostics["model_judge_fallback_used"] = True
        existing = str(diagnostics.get("model_judge_error") or "")
        message = f"{judge.__name__}: timeout after {timeout_seconds}s"
        diagnostics["model_judge_error"] = "; ".join(
            part for part in (existing, message[:220]) if part
        )
        diagnostics[f"{judge_name}_timing_seconds"] = round(time.perf_counter() - started_at, 4)
        diagnostics[f"{judge_name}_runtime_source"] = "fallback_timeout"
        diagnostics[f"{judge_name}_timeout"] = True
        diagnostics["model_judge_runtime_source"] = "fallback_timeout"
        diagnostics["model_fallback_reason"] = message
        diagnostics["model_fusion_action"] = "fallback_error_preserve"
    except Exception as exc:
        diagnostics["model_judge_fallback_used"] = True
        existing = str(diagnostics.get("model_judge_error") or "")
        message = f"{judge.__name__}: {type(exc).__name__}: {exc}"
        diagnostics["model_judge_error"] = "; ".join(
            part for part in (existing, message[:220]) if part
        )
        diagnostics[f"{judge_name}_timing_seconds"] = round(time.perf_counter() - started_at, 4)
        diagnostics[f"{judge_name}_runtime_source"] = "fallback_error"
        diagnostics["model_judge_runtime_source"] = "fallback_error"
        diagnostics["model_fallback_reason"] = message
        diagnostics["model_fusion_action"] = "fallback_error_preserve"


def _fuse_scores(diagnostics: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    reranker_pos = float(diagnostics.get("model_reranker_relevance_score") or 0.0)
    reranker_neg = float(diagnostics.get("model_reranker_negative_score") or 0.0)
    reranker_margin = reranker_pos - reranker_neg
    if not diagnostics.get("reranker_timeout") and max(reranker_pos, reranker_neg) > 0.0:
        candidates.append(("reranker", reranker_pos, reranker_neg, abs(reranker_margin)))

    if not diagnostics.get("nli_ignored_reason"):
        nli_pos = float(diagnostics.get("nli_positive_entailment_score") or 0.0)
        nli_neg = float(diagnostics.get("nli_negative_entailment_score") or 0.0)
        if max(nli_pos, nli_neg) > 0.0:
            candidates.append(("nli", nli_pos, nli_neg, abs(nli_pos - nli_neg)))

    if not diagnostics.get("zeroshot_ignored_reason"):
        zs_pos = float(diagnostics.get("zeroshot_ai_tool_for_review_score") or 0.0)
        zs_neg = float(diagnostics.get("zeroshot_subject_review_score") or 0.0)
        if max(zs_pos, zs_neg) > 0.0:
            candidates.append(("zeroshot", zs_pos, zs_neg, abs(zs_pos - zs_neg)))

    if candidates:
        primary, raw_positive, raw_negative, primary_gap = max(candidates, key=lambda item: item[3])
    else:
        primary, raw_positive, raw_negative, primary_gap = "fallback", 0.0, 0.0, 0.0
    margin = raw_positive - raw_negative
    if raw_positive >= 0.95 and raw_negative >= 0.95 and abs(margin) < 0.10:
        positive = min(raw_positive, 0.50 + max(0.0, margin) / 2.0)
        negative = min(raw_negative, 0.50 + max(0.0, -margin) / 2.0)
        uncertainty = 0.95
    else:
        positive = raw_positive
        negative = raw_negative
        uncertainty = 1.0 - min(1.0, abs(positive - negative) + max(positive, negative) * 0.35)
    conflict = max(0.0, negative - positive)
    return {
        "model_positive_score": round(positive, 4),
        "model_negative_score": round(negative, 4),
        "model_conflict_score": round(conflict, 4),
        "model_uncertainty_score": round(max(0.0, uncertainty), 4),
        "model_primary_signal": primary,
        "model_primary_margin": round(primary_gap, 4),
    }


def _mark_uniform_auxiliary_judges(diagnostics: dict[str, Any]) -> None:
    nli_pos = float(diagnostics.get("nli_positive_entailment_score") or 0.0)
    nli_neg = float(diagnostics.get("nli_negative_entailment_score") or 0.0)
    if max(nli_pos, nli_neg) > 0.0 and abs(nli_pos - nli_neg) < 0.10:
        diagnostics["nli_ignored_reason"] = "uniform_low_gap"

    zs_pos = float(diagnostics.get("zeroshot_ai_tool_for_review_score") or 0.0)
    zs_neg = float(diagnostics.get("zeroshot_subject_review_score") or 0.0)
    if max(zs_pos, zs_neg) > 0.0 and abs(zs_pos - zs_neg) < 0.10:
        diagnostics["zeroshot_ignored_reason"] = "uniform_low_gap"


def _apply_directional_sanity_override(diagnostics: dict[str, Any]) -> None:
    if (
        diagnostics.get("positive_workflow_candidate_detected")
        and (
            not diagnostics.get("external_domain_candidate_detected")
            or _has_strong_workflow_candidate(diagnostics)
        )
        and diagnostics.get("directional_is_review_about_ai_external_domain")
        and not diagnostics.get("directional_uses_ai_for_review_workflow")
    ):
        diagnostics["directional_judge_source"] = "llm+deterministic"
        diagnostics["directional_relation"] = "ai_tool_for_review_workflow"
        diagnostics["directional_uses_ai_for_review_workflow"] = True
        diagnostics["directional_is_review_about_ai_external_domain"] = False
        diagnostics["directional_reason"] = (
            "Workflow-task terms were explicit and no external-domain task terms were detected; "
            "LLM external-domain label was corrected."
        )
        diagnostics["llm_uses_ai_for_review_workflow"] = True
        diagnostics["llm_is_review_about_ai"] = False
    if (
        diagnostics.get("external_domain_candidate_detected")
        and not diagnostics.get("positive_workflow_candidate_detected")
    ):
        diagnostics["directional_judge_source"] = "llm+deterministic"
        diagnostics["directional_relation"] = "review_about_ai_external_domain"
        diagnostics["directional_uses_ai_for_review_workflow"] = False
        diagnostics["directional_is_review_about_ai_external_domain"] = True
        diagnostics["directional_confidence"] = max(float(diagnostics.get("directional_confidence") or 0.0), 0.75)
        diagnostics["directional_reason"] = (
            "External-domain AI review terms were detected without review-workflow task evidence; "
            "workflow promotion was blocked."
        )
        diagnostics["llm_uses_ai_for_review_workflow"] = False
        diagnostics["llm_is_review_about_ai"] = True


def _has_strong_workflow_candidate(diagnostics: dict[str, Any]) -> bool:
    terms = str(diagnostics.get("positive_workflow_candidate_terms") or "").lower()
    strong_terms = (
        "title/abstract screening",
        "title abstract screening",
        "citation screening",
        "study selection",
        "selection of studies",
        "literature search",
        "search strategy",
        "screening phase",
        "identification of relevant articles",
        "systematic review automation",
        "automated systematic review",
        "research assistant for literature review",
        "review assistant",
        "systematic review workflow",
        "systematic review updates",
        "accelerate systematic reviews",
        "accelerating systematic reviews",
        "accelerate systematic literature reviews",
        "accelerating systematic literature reviews",
        "llm-assisted methodology",
    )
    return any(term in terms for term in strong_terms)


def _preserve_deterministic_trigger_fields(
    diagnostics: dict[str, Any],
    trigger_diagnostics: dict[str, Any],
) -> None:
    for field in (
        "ai_review_candidate_detected",
        "positive_workflow_candidate_detected",
        "external_domain_candidate_detected",
    ):
        if trigger_diagnostics.get(field):
            diagnostics[field] = True
    for field in (
        "positive_workflow_candidate_terms",
        "external_domain_candidate_terms",
        "llm_route",
        "llm_required_reason",
        "llm_gate_confidence",
    ):
        if trigger_diagnostics.get(field) and not diagnostics.get(field):
            diagnostics[field] = trigger_diagnostics[field]


def _apply_directional_timeout_fallback(diagnostics: dict[str, Any]) -> None:
    if not diagnostics.get("model_judge_error"):
        return
    if "timeout" not in str(diagnostics.get("model_judge_error") or "").lower():
        return
    positive_candidate = bool(diagnostics.get("positive_workflow_candidate_detected"))
    external_candidate = bool(diagnostics.get("external_domain_candidate_detected"))
    if positive_candidate and not external_candidate:
        diagnostics["directional_judge_source"] = "deterministic_timeout_fallback"
        diagnostics["directional_relation"] = "ai_tool_for_review_workflow"
        diagnostics["directional_uses_ai_for_review_workflow"] = True
        diagnostics["directional_is_review_about_ai_external_domain"] = False
        diagnostics["directional_confidence"] = max(float(diagnostics.get("directional_confidence") or 0.0), 0.72)
        diagnostics["directional_reason"] = (
            "LLM directional judge timed out; explicit review-workflow task terms were detected without external-domain terms."
        )
        diagnostics["model_positive_score"] = max(float(diagnostics.get("model_positive_score") or 0.0), 0.72)
        diagnostics["model_negative_score"] = min(float(diagnostics.get("model_negative_score") or 0.0), 0.35)
        diagnostics["model_conflict_score"] = 0.0
        return
    if external_candidate and not positive_candidate:
        diagnostics["directional_judge_source"] = "deterministic_timeout_fallback"
        diagnostics["directional_relation"] = "review_about_ai_external_domain"
        diagnostics["directional_uses_ai_for_review_workflow"] = False
        diagnostics["directional_is_review_about_ai_external_domain"] = True
        diagnostics["directional_confidence"] = max(float(diagnostics.get("directional_confidence") or 0.0), 0.72)
        diagnostics["directional_reason"] = (
            "LLM directional judge timed out; external-domain AI review terms were detected without review-workflow task terms."
        )
        diagnostics["model_positive_score"] = min(float(diagnostics.get("model_positive_score") or 0.0), 0.35)
        diagnostics["model_negative_score"] = max(float(diagnostics.get("model_negative_score") or 0.0), 0.72)
        diagnostics["model_conflict_score"] = max(0.0, diagnostics["model_negative_score"] - diagnostics["model_positive_score"])


def _llm_adjusted_scores(diagnostics: dict[str, Any]) -> dict[str, Any]:
    positive = float(diagnostics.get("model_positive_score") or 0.0)
    negative = float(diagnostics.get("model_negative_score") or 0.0)
    if diagnostics.get("llm_uses_ai_for_review_workflow"):
        positive = max(positive, 0.86)
    if diagnostics.get("llm_is_review_about_ai"):
        negative = max(negative, 0.86)
    return {
        "model_positive_score": round(positive, 4),
        "model_negative_score": round(negative, 4),
        "model_conflict_score": round(max(0.0, negative - positive), 4),
    }


def _decision_action(rq_type: str, current: str, d: dict[str, Any], result: dict[str, Any]) -> tuple[str, str, str]:
    positive = float(d.get("model_positive_score") or 0.0)
    negative = float(d.get("model_negative_score") or 0.0)
    directional_workflow = bool(
        d.get("directional_uses_ai_for_review_workflow")
        or d.get("llm_uses_ai_for_review_workflow")
    )
    directional_external = bool(
        d.get("directional_is_review_about_ai_external_domain")
        or d.get("llm_is_review_about_ai")
    )
    directional_relation = str(d.get("directional_relation") or d.get("llm_judge_relation") or "")
    directional_confidence = float(d.get("directional_confidence") or 0.0)
    explicit_llm_workflow = bool(
        d.get("llm_directional_judge_used")
        and (
            directional_relation == "ai_tool_for_review_workflow"
            or d.get("llm_uses_ai_for_review_workflow")
        )
        and not directional_external
        and directional_confidence >= 0.62
    )
    subject_conflict = negative >= 0.72 and positive < 0.55
    ai_method_present = bool(
        str(result.get("ai_tool_terms") or result.get("method_evidence_terms") or "").strip()
        or d.get("llm_uses_ai_for_review_workflow")
        or directional_workflow
    )
    if rq_type != "review_workflow_automation":
        return "preserve", "Model judges are secondary for non-SLR workflow RQs.", current
    if directional_external and not directional_workflow:
        if current == "KEEP":
            return "directional_demote_keep_to_maybe", "Directional judge found an external-domain AI review, not AI used for the review workflow.", "MAYBE"
        return "directional_preserve_external_domain", "Directional judge found external-domain AI review direction; no promotion allowed.", current
    blocked_reason = str(d.get("model_fusion_blocked_reason") or "")
    if current == "MAYBE" and blocked_reason:
        if blocked_reason in {"model_disagreement", "high_high_small_margin_uncertain"}:
            return "directional_preserve_unclear", "Model judges were directionally uncertain; deterministic decision preserved.", current
        if blocked_reason == "model_judge_exception":
            return "fallback_error_preserve", "Model judge failed; deterministic decision preserved.", current
        return "preserve", "Explicit directional LLM workflow support was absent; deterministic MAYBE preserved.", current
    if abs(positive - negative) < 0.10:
        return "directional_preserve_unclear", "Model judges were directionally uncertain; deterministic decision preserved.", current
    if current == "REJECT" and ai_method_present and directional_workflow and positive >= 0.62 and negative < 0.68:
        return "directional_promote_reject_to_maybe", "Directional judge found plausible AI-for-review workflow evidence.", "MAYBE"
    if current == "MAYBE" and ai_method_present and explicit_llm_workflow and positive >= 0.62 and negative < 0.68:
        return "directional_promote_maybe_to_keep", "Directional judge found workflow-use evidence with low subject-review risk.", "KEEP"
    if current == "KEEP" and directional_workflow:
        return "directional_confirm_keep", "Directional judge confirmed AI/tool use for the review workflow.", "KEEP"
    if current == "KEEP" and subject_conflict:
        return "directional_demote_keep_to_maybe", "Model judges found strong subject-review risk and weak workflow-use evidence.", "MAYBE"
    return "preserve", "Model and deterministic evidence did not justify changing the decision.", current
