from __future__ import annotations

import re
from typing import Any


STAGE0_DIAGNOSTICS = {
    "stage0_route": "full_extraction_required",
    "stage0_confidence": 0.0,
    "stage0_reason": "",
    "stage0_timing_seconds": 0.0,
    "stage0_requires_full_extraction": True,
    "heuristic_frame_used": False,
    "full_extraction_forced_reason": "",
    "paper_frame_source": "ollama_semantic_frame",
    "semantic_frame_ollama_call_skipped": False,
    "stage0_ollama_calls_avoided": 0,
    "stage0_false_shortcut_risk": False,
}

AI_TERMS = (
    "artificial intelligence", " ai ", "machine learning", "deep learning",
    "large language model", "large language models", "llm", "llms",
    "generative ai", "chatgpt", "gpt",
)
REVIEW_TERMS = (
    "systematic review", "systematic literature review", "literature review",
    "systematic reviews", "evidence review",
)
REVIEW_WORKFLOW_TERMS = (
    "title/abstract screening", "title abstract screening", "citation screening",
    "abstract screening", "study selection", "selection of studies",
    "literature search", "search strategy", "data extraction", "risk of bias",
    "quality assessment", "evidence synthesis", "review workflow",
    "review process", "review automation", "systematic review automation",
    "automating systematic reviews", "automating systematic literature reviews",
    "automate and facilitate the review process",
)
EXTERNAL_SLR_TERMS = (
    "diagnosis", "diagnostic", "disease", "cancer", "healthcare", "medical",
    "clinical", "education", "student", "software", "cybersecurity",
    "fintech", "finance", "supply chain", "manufacturing", "programming",
    "drug discovery", "classification", "prediction",
)

BLOCKCHAIN_TERMS = ("blockchain", "distributed ledger", "smart contract", "hyperledger")
SUPPLY_CHAIN_TERMS = ("supply chain", "logistics", "traceability", "transparency", "provenance")
SUPPLY_TASK_TERMS = ("traceability", "transparency", "trust", "security", "provenance", "integrity")

ML_TERMS = (
    "machine learning", "deep learning", "artificial intelligence", "neural network",
    "random forest", "support vector machine", "svm", "xgboost", "cnn", "lstm",
)
HEART_TERMS = ("heart disease", "cardiovascular", "cardiac", "cardiology")
PREDICTION_TERMS = ("prediction", "diagnosis", "detection", "classification", "risk")


def route_stage0(
    *,
    rq_frame: dict[str, Any],
    title: str,
    abstract: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    output = dict(STAGE0_DIAGNOSTICS)
    if not _stage0_enabled(cfg):
        output["stage0_reason"] = "Stage-0 disabled or not in two_pass_fast."
        output["full_extraction_forced_reason"] = "stage0_disabled"
        return output

    rq_text = _norm(" ".join(str(value or "") for value in (rq_frame or {}).values()))
    text = _norm(f"{title} {abstract}")

    if _is_review_workflow_rq(rq_frame):
        return _route_slr(text)
    if _is_blockchain_supply_chain_rq(rq_text) and _obvious_blockchain_supply_chain(text):
        return _heuristic_route(
            "Obvious blockchain supply-chain domain/method/task match.",
            confidence=0.94,
        )
    if _is_heart_ml_rq(rq_text) and _obvious_heart_ml(text):
        return _heuristic_route(
            "Obvious heart-disease ML/DL prediction/diagnosis match.",
            confidence=0.94,
        )

    output["stage0_reason"] = "No conservative obvious-domain route matched."
    output["full_extraction_forced_reason"] = "stage0_no_safe_route"
    return output


def build_stage0_heuristic_frame(
    *,
    rq_frame: dict[str, Any],
    title: str,
    abstract: str,
    route: dict[str, Any],
) -> dict[str, str]:
    text = _norm(f"{title} {abstract}")
    rq_text = _norm(" ".join(str(value or "") for value in (rq_frame or {}).values()))
    frame = {
        "primary_subject": title,
        "intervention_or_method": "",
        "target_problem_or_task": "",
        "application_context": "",
        "evidence_type": "",
        "study_role": "",
        "review_role": "",
        "question_type": "",
        "frame_source": "heuristic_stage0",
        "frame_diagnostic": str(route.get("stage0_reason") or ""),
        "source_title": str(title or ""),
        "source_abstract": str(abstract or ""),
    }
    if _is_blockchain_supply_chain_rq(rq_text):
        frame.update({
            "primary_subject": "supply chain management",
            "intervention_or_method": "blockchain technology",
            "target_problem_or_task": "transparency traceability trust security",
            "application_context": "supply chain management",
            "methods_or_technologies": "blockchain technology",
            "target_tasks_or_outcomes": "transparency; traceability; trust; security",
            "application_contexts": "supply chain management",
        })
    elif _is_heart_ml_rq(rq_text):
        frame.update({
            "primary_subject": "heart disease prediction and diagnosis",
            "intervention_or_method": "machine learning and deep learning",
            "target_problem_or_task": "heart disease prediction and diagnosis",
            "application_context": "cardiovascular healthcare",
            "methods_or_technologies": "machine learning; deep learning",
            "target_tasks_or_outcomes": "prediction; diagnosis",
            "application_contexts": "heart disease; cardiovascular healthcare",
        })
    elif route.get("stage0_route") == "external_domain_fast_route_but_still_final_safe":
        frame.update({
            "intervention_or_method": _join_hits(text, AI_TERMS) or "artificial intelligence",
            "target_problem_or_task": "external-domain AI application reviewed as subject",
            "application_context": _join_hits(text, EXTERNAL_SLR_TERMS),
            "review_role": "technology_being_reviewed",
            "methods_or_technologies": _join_hits(text, AI_TERMS),
            "target_tasks_or_outcomes": "external-domain AI application",
            "application_contexts": _join_hits(text, EXTERNAL_SLR_TERMS),
        })
    return frame


def _route_slr(text: str) -> dict[str, Any]:
    ai = _has_any(text, AI_TERMS)
    review = _has_any(text, REVIEW_TERMS)
    workflow = _has_any(text, REVIEW_WORKFLOW_TERMS)
    external = _has_any(text, EXTERNAL_SLR_TERMS)
    if ai and review and workflow:
        return _full_route(
            "SLR automation row has AI/review/workflow evidence; full extraction required.",
            risk=True,
        )
    if ai and review and external and not workflow:
        return _external_route("Obvious external-domain AI review without workflow evidence.")
    return _full_route("SLR row is not safe for Stage-0 shortcut.", risk=bool(ai and review))


def _heuristic_route(reason: str, confidence: float) -> dict[str, Any]:
    output = dict(STAGE0_DIAGNOSTICS)
    output.update({
        "stage0_route": "heuristic_frame_allowed_for_obvious_domain_match",
        "stage0_confidence": confidence,
        "stage0_reason": reason,
        "stage0_requires_full_extraction": False,
        "heuristic_frame_used": True,
        "paper_frame_source": "heuristic_stage0",
        "semantic_frame_ollama_call_skipped": True,
        "stage0_ollama_calls_avoided": 1,
    })
    return output


def _external_route(reason: str) -> dict[str, Any]:
    output = _heuristic_route(reason, confidence=0.88)
    output["stage0_route"] = "external_domain_fast_route_but_still_final_safe"
    output["stage0_false_shortcut_risk"] = True
    return output


def _full_route(reason: str, *, risk: bool = False) -> dict[str, Any]:
    output = dict(STAGE0_DIAGNOSTICS)
    output["stage0_reason"] = reason
    output["full_extraction_forced_reason"] = reason
    output["stage0_false_shortcut_risk"] = risk
    return output


def _stage0_enabled(cfg: dict[str, Any]) -> bool:
    return bool(
        cfg.get("screening_pipeline_mode") == "two_pass_fast"
        and cfg.get("enable_stage0_fast_triage")
        and cfg.get("enable_heuristic_fast_frames")
    )


def _is_review_workflow_rq(rq_frame: dict[str, Any]) -> bool:
    values = _norm(" ".join(str((rq_frame or {}).get(name) or "") for name in (
        "review_question_type", "question_type", "rq_type", "rq_desired_relation",
        "target_problem_or_task", "application_context",
    )))
    return "review_workflow_automation" in values or "tool_used_for_workflow" in values


def _is_blockchain_supply_chain_rq(rq_text: str) -> bool:
    return _has_any(rq_text, BLOCKCHAIN_TERMS) and _has_any(rq_text, SUPPLY_CHAIN_TERMS)


def _is_heart_ml_rq(rq_text: str) -> bool:
    return _has_any(rq_text, HEART_TERMS) and _has_any(rq_text, ML_TERMS)


def _obvious_blockchain_supply_chain(text: str) -> bool:
    return (
        _has_any(text, BLOCKCHAIN_TERMS)
        and _has_any(text, SUPPLY_CHAIN_TERMS)
        and _has_any(text, SUPPLY_TASK_TERMS)
    )


def _obvious_heart_ml(text: str) -> bool:
    return _has_any(text, HEART_TERMS) and _has_any(text, ML_TERMS) and _has_any(text, PREDICTION_TERMS)


def _has_any(text: str, terms) -> bool:
    padded = f" {_norm(text)} "
    return any(term in padded for term in terms)


def _join_hits(text: str, terms) -> str:
    return "; ".join(dict.fromkeys(term.strip() for term in terms if term.strip() in text))


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()
