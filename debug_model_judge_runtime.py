from __future__ import annotations

import argparse
import os
from pprint import pprint

from model_registry import configured_model_names, get_cross_encoder, model_preflight_rows
from model_score_fusion import apply_model_score_fusion
from runtime_config import get_model_judge_config
from llm_structured_judge import judge_batch_with_llm, get_cache_stats
from semantic_frame import initialize_semantic_frame_cache


RQ = "Can large language models and artificial intelligence tools help automate systematic literature reviews?"
POSITIVE_TITLE = "Large Language Models for Title and Abstract Screening in Systematic Reviews"
POSITIVE_ABSTRACT = (
    "This study evaluates large language models for automating title and abstract "
    "screening, study selection, and reviewer workload reduction in systematic reviews."
)
NEGATIVE_TITLE = "Applications of Large Language Models in Breast Cancer Diagnosis: A Systematic Review"
NEGATIVE_ABSTRACT = (
    "This systematic review summarizes applications of large language models for "
    "breast cancer diagnosis, clinical decision support, and medical prediction tasks."
)


def _fusion_for(title: str, abstract: str) -> dict:
    rq_frame = {
        "rq_text": RQ,
        "review_question_type": "review_workflow_automation",
        "question_type": "review_workflow_automation",
    }
    paper_frame = {
        "source_title": title,
        "source_abstract": abstract,
        "primary_subject": title,
    }
    deterministic = {
        "decision": "MAYBE",
        "reason": "Deterministic baseline.",
        "method_evidence_terms": "large language models",
    }
    _, diagnostics = apply_model_score_fusion(
        rq_frame=rq_frame,
        paper_frame=paper_frame,
        deterministic_result=deterministic,
        research_question=RQ,
    )
    return diagnostics


def _print_preflight() -> None:
    cfg = get_model_judge_config()
    print("effective_config:")
    pprint(cfg)
    print("model_names:")
    pprint(configured_model_names())
    print("download_check:")
    for row in model_preflight_rows():
        print(
            f"- {row['role']}: {row['model_name']} | "
            f"exists_locally={row['exists_locally']} | "
            f"estimated_mode={row['estimated_mode']} | "
            f"internet_or_download_needed={row['internet_or_download_needed']} | "
            f"{row['reason']}"
        )
    print("next_command:")
    print(
        "set MODEL_JUDGE_MODE=balanced && set MODEL_JUDGE_PROFILE=light && "
        "set ENABLE_MODEL_JUDGES=true && set ENABLE_HF_MODEL_LOADING=true && "
        "set ENABLE_HF_MODEL_DOWNLOAD=true && python debug_model_judge_runtime.py --profile light --real-model-smoke"
    )


def _print_smoke() -> int:
    os.environ.setdefault("ENABLE_LLM_JUDGE_FOR_SMOKE", "true")
    _warmup_reranker()
    positive = _fusion_for(POSITIVE_TITLE, POSITIVE_ABSTRACT)
    negative = _fusion_for(NEGATIVE_TITLE, NEGATIVE_ABSTRACT)
    print("positive:")
    pprint(_smoke_view(POSITIVE_TITLE, positive))
    print("negative:")
    pprint(_smoke_view(NEGATIVE_TITLE, negative))
    pos_gap = float(positive.get("model_positive_score") or 0.0) - float(positive.get("model_negative_score") or 0.0)
    neg_gap = float(negative.get("model_negative_score") or 0.0) - float(negative.get("model_positive_score") or 0.0)
    reranker_pos_gap = float(positive.get("model_reranker_relevance_score") or 0.0) - float(positive.get("model_reranker_negative_score") or 0.0)
    reranker_neg_gap = float(negative.get("model_reranker_negative_score") or 0.0) - float(negative.get("model_reranker_relevance_score") or 0.0)
    positive_direction_ok = (
        positive.get("directional_uses_ai_for_review_workflow") is True
        and positive.get("directional_is_review_about_ai_external_domain") is False
        and str(positive.get("model_decision_hint") or "") in {"KEEP", "MAYBE"}
    )
    negative_direction_ok = (
        negative.get("directional_uses_ai_for_review_workflow") is False
        and negative.get("directional_is_review_about_ai_external_domain") is True
        and str(negative.get("model_decision_hint") or "") in {"REJECT", "MAYBE"}
    )
    real_models = (
        positive.get("model_real_models_loaded")
        and negative.get("model_real_models_loaded")
        and positive.get("directional_judge_source") in {"llm", "nli"}
        and negative.get("directional_judge_source") in {"llm", "nli"}
    )
    high_high = (
        float(positive.get("model_positive_score") or 0.0) > 0.95
        and float(positive.get("model_negative_score") or 0.0) > 0.95
    ) or (
        float(negative.get("model_positive_score") or 0.0) > 0.95
        and float(negative.get("model_negative_score") or 0.0) > 0.95
    )
    passed = bool(real_models and positive_direction_ok and negative_direction_ok and not high_high)
    print(f"positive_gap={pos_gap:.4f}")
    print(f"negative_gap={neg_gap:.4f}")
    print(f"reranker_positive_gap={reranker_pos_gap:.4f}")
    print(f"reranker_negative_gap={reranker_neg_gap:.4f}")
    print(f"real_models_loaded={real_models}")
    print(f"positive_direction_ok={positive_direction_ok}")
    print(f"negative_direction_ok={negative_direction_ok}")
    print(f"high_high_failure={high_high}")
    print(f"smoke_passed={passed}")
    return 0 if passed else 2


SMOKE_SET = [
    ("positive", "Large Language Models for Title/Abstract Screening in Systematic Reviews", "This paper evaluates LLMs for title and abstract screening, study selection, and workload reduction in systematic reviews."),
    ("positive", "Screening Automation for Systematic Reviews", "We evaluate automated tools for citation screening and study selection in systematic review workflows."),
    ("positive", "Can Machine Learning Support Selection of Studies for Systematic Review Updates?", "This study evaluates machine learning to support selection of studies when updating systematic reviews."),
    ("positive", "Implementation of a research assistant for literature review with LLMs", "We implement an LLM-based research assistant to support literature review search, screening, and extraction tasks."),
    ("positive", "Feature selection analysis of a scalable citation screening algorithm", "This work studies feature selection for citation screening algorithms used in systematic review automation."),
    ("negative", "Artificial Intelligence Methods in Software Refactoring: A Systematic Literature Review", "This systematic literature review summarizes AI methods for software refactoring tasks."),
    ("negative", "Applications of Large Language Models in Breast Cancer Diagnosis: A Systematic Review", "This systematic review summarizes LLM applications for breast cancer diagnosis and medical prediction."),
    ("negative", "XAI in Fintech: A Systematic Literature Review", "This paper systematically reviews explainable AI applications in financial technology and credit risk."),
    ("negative", "AI in Education: A Systematic Literature Review", "This systematic review surveys artificial intelligence applications in education and learning analytics."),
    ("negative", "LLMs in Cybersecurity: A Systematic Literature Review", "This systematic review examines large language model applications in cybersecurity tasks."),
]


def _print_directional_smoke_set() -> int:
    os.environ.setdefault("ENABLE_LLM_JUDGE_FOR_SMOKE", "true")
    positives_ok = 0
    negatives_ok = 0
    negatives_keep = 0
    rows = []
    for expected, title, abstract in SMOKE_SET:
        diagnostics = _fusion_for(title, abstract)
        workflow = diagnostics.get("directional_uses_ai_for_review_workflow") is True
        external = diagnostics.get("directional_is_review_about_ai_external_domain") is True
        hint = str(diagnostics.get("model_decision_hint") or "")
        if expected == "positive" and (workflow or (hint == "MAYBE" and not external)):
            positives_ok += 1
        if expected == "negative" and (external or (hint == "MAYBE" and not workflow)):
            negatives_ok += 1
        if expected == "negative" and hint == "KEEP":
            negatives_keep += 1
        rows.append({
            "expected": expected,
            "title": title,
            "relation": diagnostics.get("directional_relation"),
            "workflow": workflow,
            "external": external,
            "decision_hint": hint,
            "fusion_action": diagnostics.get("model_fusion_action"),
            "triggered": diagnostics.get("llm_directional_triggered"),
            "trigger_reason": diagnostics.get("llm_directional_trigger_reason"),
            "cache_hit": diagnostics.get("llm_directional_cache_hit"),
            "timing": diagnostics.get("llm_directional_timing_seconds"),
        })
    for row in rows:
        pprint(row)
    passed = positives_ok >= 4 and negatives_ok >= 4 and negatives_keep == 0
    print("confusion_matrix:")
    print(f"positive_ok={positives_ok}/5")
    print(f"negative_ok={negatives_ok}/5")
    print(f"negative_keep_errors={negatives_keep}")
    print(f"directional_smoke_passed={passed}")
    return 0 if passed else 2


def _print_batch_directional_smoke_set() -> int:
    records = [
        {"id": f"R{i}", "title": title, "abstract": abstract}
        for i, (_, title, abstract) in enumerate(SMOKE_SET, start=1)
    ]
    results = judge_batch_with_llm(records, RQ, os.getenv("FIRST_STAGE_MODEL", "qwen2.5:3b"))
    positives_ok = 0
    negatives_ok = 0
    negatives_keep = 0
    for (expected, title, _), record in zip(SMOKE_SET, records):
        diagnostics = results.get(record["id"], {})
        workflow = diagnostics.get("directional_uses_ai_for_review_workflow") is True
        external = diagnostics.get("directional_is_review_about_ai_external_domain") is True
        hint = str(diagnostics.get("llm_judge_decision") or diagnostics.get("model_decision_hint") or "")
        if expected == "positive" and workflow and not external:
            positives_ok += 1
        if expected == "negative" and external and not workflow:
            negatives_ok += 1
        if expected == "negative" and hint == "KEEP":
            negatives_keep += 1
        pprint({
            "expected": expected,
            "title": title,
            "relation": diagnostics.get("directional_relation"),
            "workflow": workflow,
            "external": external,
            "cache_hit": diagnostics.get("llm_directional_cache_hit"),
            "source": diagnostics.get("directional_judge_source"),
        })
    passed = positives_ok >= 4 and negatives_ok >= 5 and negatives_keep == 0
    print("batch_confusion_matrix:")
    print(f"positive_ok={positives_ok}/5")
    print(f"negative_ok={negatives_ok}/5")
    print(f"negative_keep_errors={negatives_keep}")
    print(f"batch_directional_smoke_passed={passed}")
    return 0 if passed else 2


def _warmup_reranker() -> None:
    names = configured_model_names()
    model = get_cross_encoder(names["reranker"])
    if model is None:
        print("warmup_reranker=skipped_no_model")
        return
    try:
        model.predict([
            (
                "AI used to automate systematic review screening workflows.",
                "Large language models for title and abstract screening in systematic reviews.",
            )
        ])
        print("warmup_reranker=ok")
    except Exception as exc:
        print(f"warmup_reranker=failed:{type(exc).__name__}:{exc}")


def _smoke_view(title: str, diagnostics: dict) -> dict:
    return {
        "title": title,
        "reranker_positive_raw_score": diagnostics.get("model_reranker_positive_raw_score"),
        "reranker_negative_raw_score": diagnostics.get("model_reranker_negative_raw_score"),
        "reranker_positive_score": diagnostics.get("model_reranker_relevance_score"),
        "reranker_negative_score": diagnostics.get("model_reranker_negative_score"),
        "reranker_margin": diagnostics.get("model_reranker_margin"),
        "nli_positive_entailment_score": diagnostics.get("nli_positive_entailment_score"),
        "nli_negative_entailment_score": diagnostics.get("nli_negative_entailment_score"),
        "nli_positive_hypothesis_scores": diagnostics.get("nli_positive_hypothesis_scores"),
        "nli_negative_hypothesis_scores": diagnostics.get("nli_negative_hypothesis_scores"),
        "zeroshot_relation_label": diagnostics.get("zeroshot_relation_label"),
        "zeroshot_ai_tool_for_review_score": diagnostics.get("zeroshot_ai_tool_for_review_score"),
        "zeroshot_subject_review_score": diagnostics.get("zeroshot_subject_review_score"),
        "zeroshot_top_labels": diagnostics.get("zeroshot_top_labels"),
        "model_positive_score": diagnostics.get("model_positive_score"),
        "model_negative_score": diagnostics.get("model_negative_score"),
        "model_uncertainty_score": diagnostics.get("model_uncertainty_score"),
        "model_decision_hint": diagnostics.get("model_decision_hint"),
        "directional_judge_source": diagnostics.get("directional_judge_source"),
        "directional_relation": diagnostics.get("directional_relation"),
        "directional_confidence": diagnostics.get("directional_confidence"),
        "directional_uses_ai_for_review_workflow": diagnostics.get("directional_uses_ai_for_review_workflow"),
        "directional_is_review_about_ai_external_domain": diagnostics.get("directional_is_review_about_ai_external_domain"),
        "directional_reason": diagnostics.get("directional_reason"),
        "llm_directional_judge_used": diagnostics.get("llm_directional_judge_used"),
        "llm_directional_judge_error": diagnostics.get("llm_directional_judge_error"),
        "model_judge_runtime_source": diagnostics.get("model_judge_runtime_source"),
        "model_real_models_loaded": diagnostics.get("model_real_models_loaded"),
        "model_fallback_reason": diagnostics.get("model_fallback_reason"),
        "model_timing_seconds": diagnostics.get("model_timing_seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-check", action="store_true")
    parser.add_argument("--real-model-smoke", action="store_true")
    parser.add_argument("--directional-smoke-set", action="store_true")
    parser.add_argument("--batch-directional-smoke-set", action="store_true")
    parser.add_argument("--cache-info", action="store_true")
    parser.add_argument("--semantic-frame-cache-info", action="store_true")
    parser.add_argument("--profile", choices=["light", "balanced", "full"])
    args = parser.parse_args()
    if args.profile:
        os.environ["MODEL_JUDGE_PROFILE"] = args.profile

    print("cwd:", os.getcwd())
    print("env:")
    pprint(get_model_judge_config()["raw_env"])
    if args.download_check:
        _print_preflight()
        return 0
    if args.real_model_smoke:
        return _print_smoke()
    if args.directional_smoke_set:
        return _print_directional_smoke_set()
    if args.batch_directional_smoke_set:
        return _print_batch_directional_smoke_set()
    if args.cache_info:
        pprint(get_cache_stats())
        return 0
    if args.semantic_frame_cache_info:
        pprint(initialize_semantic_frame_cache())
        return 0

    cfg = get_model_judge_config()
    print("effective_config:")
    pprint(cfg)
    print("model_names:")
    pprint(configured_model_names())
    diagnostics = _fusion_for(POSITIVE_TITLE, POSITIVE_ABSTRACT)
    print("fusion_called:", diagnostics.get("model_judges_enabled") is True)
    print("result:")
    pprint({
        "model_judges_enabled": diagnostics.get("model_judges_enabled"),
        "model_judge_mode": diagnostics.get("model_judge_mode"),
        "model_profile": diagnostics.get("model_profile"),
        "model_judge_runtime_source": diagnostics.get("model_judge_runtime_source"),
        "model_fusion_action": diagnostics.get("model_fusion_action"),
        "model_positive_score": diagnostics.get("model_positive_score"),
        "model_negative_score": diagnostics.get("model_negative_score"),
        "model_judge_error": diagnostics.get("model_judge_error"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
