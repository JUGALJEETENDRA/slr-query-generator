import os
import asyncio
import json
import sys
import types
from pathlib import Path

import pandas as pd

import model_registry
from model_registry import get_cross_encoder, model_judges_enabled
from model_score_fusion import apply_model_score_fusion
from runtime_config import get_model_judge_config
from nli_judge import judge_nli
from reranker_judge import judge_reranker
from zeroshot_relation_judge import judge_zero_shot_relation
from bulk_screen import _result_semantic_fields
from screening_contracts import build_rq_contract


def _rq(rq_type="review_workflow_automation"):
    return {
        "rq_text": "Can large language models help automate systematic literature reviews?",
        "review_question_type": rq_type,
        "question_type": rq_type,
    }


def _paper(title="LLM screening for systematic reviews", abstract="Uses LLMs for title abstract screening."):
    return {
        "source_title": title,
        "source_abstract": abstract,
        "primary_subject": title,
    }


def test_semicolon_method_terms_normalize_safely():
    contract = build_rq_contract(
        "Can AI and LLMs automate systematic literature reviews?",
        {
            "review_question_type": "review_workflow_automation",
            "method_or_technology": "artificial intelligence; large language models",
            "target_tasks_or_outcomes": "screening; study selection",
            "application_context": "systematic literature reviews",
            "required_dimensions": "method_tool; review_workflow_process; review_context",
        },
    )
    assert contract["rq_method_terms"] == [
        "artificial intelligence",
        "large language models",
    ]


def test_slr_rq_method_string_does_not_keyerror_with_hf_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_HF_MODEL_LOADING", "false")
    from debug_stage1_one_row import FakeEngine, RQ, TITLE, ABSTRACT
    from semantic_frame import extract_semantic_frame
    from semantic_comparator import compare_semantic_frames

    rq_frame = {
        "rq_text": RQ,
        "review_question_type": "review_workflow_automation",
        "question_type": "review_workflow_automation",
        "method_or_technology": "artificial intelligence; large language models",
        "intervention_or_method": "artificial intelligence; large language models",
        "target_tasks_or_outcomes": "automate systematic literature reviews",
        "target_problem_or_task": "automate systematic literature reviews",
        "application_context": "systematic literature reviews",
        "required_dimensions": "method_tool; review_workflow_process; review_context",
        "rq_desired_relation": "tool_used_for_workflow",
    }
    paper_frame = extract_semantic_frame(
        title=TITLE,
        abstract=ABSTRACT,
        inference_engine=FakeEngine(),
    )
    result = compare_semantic_frames(rq_frame, paper_frame)
    assert result["decision"] in {"KEEP", "MAYBE", "REJECT"}


def test_model_registry_can_disable_model_judges(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "off")
    assert model_judges_enabled() is False


def test_cmd_style_env_variables_activate_model_judges(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setenv("ENABLE_HF_MODEL_LOADING", "false")
    cfg = get_model_judge_config()
    assert cfg["enable_model_judges"] is True
    assert cfg["model_judge_mode"] == "balanced"
    assert cfg["enable_hf_model_loading"] is False


def test_uppercase_env_boolean_values(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "BALANCED")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "TRUE")
    monkeypatch.setenv("ENABLE_HF_MODEL_LOADING", "FALSE")
    cfg = get_model_judge_config()
    assert cfg["enable_model_judges"] is True
    assert cfg["model_judge_mode"] == "balanced"
    assert cfg["enable_hf_model_loading"] is False


def test_hf_loading_false_blocks_registry_loads(monkeypatch):
    monkeypatch.setenv("ENABLE_HF_MODEL_LOADING", "false")
    assert get_cross_encoder("any/model") is None
    from model_registry import get_transformers_pipeline

    assert get_transformers_pipeline("zero-shot-classification", "any/model") is None


def test_profile_light_selects_lightweight_models(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_PROFILE", "light")
    monkeypatch.delenv("RERANKER_MODEL_NAME", raising=False)
    monkeypatch.delenv("NLI_MODEL_NAME", raising=False)
    monkeypatch.delenv("ZERO_SHOT_MODEL_NAME", raising=False)
    names = model_registry.configured_model_names()
    assert names["reranker"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert names["nli"] == "typeform/distilbert-base-uncased-mnli"
    assert names["zero_shot"] == "typeform/distilbert-base-uncased-mnli"


def test_light_profile_disables_uniform_auxiliary_judges_by_default(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_PROFILE", "light")
    monkeypatch.delenv("ENABLE_NLI_JUDGE", raising=False)
    monkeypatch.delenv("ENABLE_ZERO_SHOT_JUDGE", raising=False)
    cfg = get_model_judge_config()
    assert cfg["enable_reranker_judge"] is True
    assert cfg["enable_nli_judge"] is False
    assert cfg["enable_zero_shot_judge"] is False


def test_local_only_missing_model_falls_back_before_import(monkeypatch):
    get_cross_encoder.cache_clear()
    model_registry.reset_runtime_status()
    monkeypatch.setenv("ENABLE_HF_MODEL_LOADING", "true")
    monkeypatch.setenv("ENABLE_HF_MODEL_DOWNLOAD", "false")
    monkeypatch.setattr(model_registry, "hf_model_exists_locally", lambda name: False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert get_cross_encoder("missing/model") is None
    status = model_registry.get_runtime_status("missing/model")
    assert status["runtime_source"] == "fallback_no_local_model"


def test_download_enabled_allows_cross_encoder_load_path(monkeypatch):
    get_cross_encoder.cache_clear()
    model_registry.reset_runtime_status()
    monkeypatch.setenv("ENABLE_HF_MODEL_LOADING", "true")
    monkeypatch.setenv("ENABLE_HF_MODEL_DOWNLOAD", "true")

    class FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs

    fake_module = types.SimpleNamespace(CrossEncoder=FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    model = get_cross_encoder("download/model")
    assert isinstance(model, FakeCrossEncoder)
    assert model.model_name == "download/model"
    status = model_registry.get_runtime_status("download/model")
    assert status["runtime_source"] == "hf_model"


def test_reranker_judge_returns_structured_fields(monkeypatch):
    monkeypatch.setattr("reranker_judge.get_cross_encoder", lambda name: None)
    result = judge_reranker(
        "Can LLMs automate systematic reviews?",
        "LLM screening for systematic reviews",
        "We evaluate title abstract screening.",
        "fast",
    )
    assert "model_reranker_relevance_score" in result
    assert result["model_reranker_decision_hint"] in {"positive", "negative", "uncertain"}


def test_reranker_real_scores_use_directional_margin(monkeypatch):
    from reranker_judge import _score_cached

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [12.0, 11.9]

    _score_cached.cache_clear()
    monkeypatch.setattr("reranker_judge.get_cross_encoder", lambda name: FakeCrossEncoder())
    result = judge_reranker("rq", "title", "abstract", "balanced")
    assert result["model_reranker_relevance_score"] < 0.55
    assert result["model_reranker_negative_score"] > 0.45
    assert abs(result["model_reranker_margin"]) < 0.10
    _score_cached.cache_clear()


def test_reranker_negative_margin_produces_negative_score(monkeypatch):
    from reranker_judge import _score_cached

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [2.0, 5.0]

    _score_cached.cache_clear()
    monkeypatch.setattr("reranker_judge.get_cross_encoder", lambda name: FakeCrossEncoder())
    result = judge_reranker("rq", "title", "abstract", "balanced")
    assert result["model_reranker_negative_score"] > result["model_reranker_relevance_score"]
    assert result["model_reranker_decision_hint"] == "negative"
    _score_cached.cache_clear()


def test_nli_judge_returns_positive_and_negative_scores(monkeypatch):
    monkeypatch.setattr("nli_judge.get_transformers_pipeline", lambda task, model: None)
    result = judge_nli(
        "AI applications in education",
        "A systematic review about AI applications in education.",
        "balanced",
    )
    assert "nli_positive_entailment_score" in result
    assert "nli_negative_entailment_score" in result


def test_nli_counts_directional_entailment_scores(monkeypatch):
    from nli_judge import _score_cached, POSITIVE_HYPOTHESES, NEGATIVE_HYPOTHESES

    class FakeClassifier:
        def __call__(self, text, candidate_labels, multi_label=False, hypothesis_template="{}"):
            assert multi_label is False
            labels = [POSITIVE_HYPOTHESES[0], NEGATIVE_HYPOTHESES[0], *candidate_labels[2:]]
            return {"labels": labels, "scores": [0.70, 0.20, 0.04, 0.03, 0.02, 0.01]}

    _score_cached.cache_clear()
    monkeypatch.setattr("nli_judge.get_transformers_pipeline", lambda task, model: FakeClassifier())
    result = judge_nli("title", "abstract", "balanced")
    assert result["nli_positive_entailment_score"] > result["nli_negative_entailment_score"]
    assert result["nli_decision_hint"] == "positive"
    _score_cached.cache_clear()


def test_zero_shot_judge_returns_relation_label(monkeypatch):
    monkeypatch.setattr("zeroshot_relation_judge.get_transformers_pipeline", lambda task, model: None)
    result = judge_zero_shot_relation(
        "LLM title abstract screening",
        "A tool for citation screening in systematic reviews.",
        "balanced",
    )
    assert result["zeroshot_relation_label"]
    assert result["zeroshot_ai_tool_for_review_score"] > 0


def test_zero_shot_uses_directional_top_label(monkeypatch):
    from zeroshot_relation_judge import _score_cached, LABELS

    class FakeClassifier:
        def __call__(self, text, candidate_labels, multi_label=False):
            assert multi_label is False
            return {
                "labels": [
                    "systematic review about AI in external domain",
                    "AI tool for systematic review workflow",
                    "unrelated or external-domain AI task",
                ],
                "scores": [0.75, 0.20, 0.05],
            }

    _score_cached.cache_clear()
    monkeypatch.setattr("zeroshot_relation_judge.get_transformers_pipeline", lambda task, model: FakeClassifier())
    result = judge_zero_shot_relation("title", "abstract", "balanced")
    assert result["zeroshot_subject_review_score"] > result["zeroshot_ai_tool_for_review_score"]
    assert result["zeroshot_relation_label"] == "systematic review about AI in external domain"
    _score_cached.cache_clear()


def test_model_fusion_promotes_reject_to_maybe(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setenv("ENABLE_LLM_JUDGE_FOR_SMOKE", "true")
    monkeypatch.setattr(
        "model_score_fusion.judge_reranker",
        lambda *args, **kwargs: {
            "model_reranker_relevance_score": 0.72,
            "model_reranker_negative_score": 0.20,
            "model_reranker_margin": 0.52,
            "model_reranker_decision_hint": "positive",
        },
    )
    monkeypatch.setattr(
        "model_score_fusion.judge_nli",
        lambda *args, **kwargs: {
            "nli_positive_entailment_score": 0.70,
            "nli_negative_entailment_score": 0.10,
            "nli_contradiction_score": 0.0,
            "nli_margin": 0.60,
            "nli_decision_hint": "positive",
            "nli_top_positive_hypothesis": "positive",
            "nli_top_negative_hypothesis": "",
        },
    )
    monkeypatch.setattr(
        "model_score_fusion.judge_zero_shot_relation",
        lambda *args, **kwargs: {
            "zeroshot_relation_label": "AI tool for systematic review workflow",
            "zeroshot_relation_score": 0.75,
            "zeroshot_ai_tool_for_review_score": 0.75,
            "zeroshot_subject_review_score": 0.20,
            "zeroshot_top_labels": "",
        },
    )
    monkeypatch.setattr("model_score_fusion.judge_with_llm", lambda *a, **k: {
        "llm_judge_decision": "KEEP",
        "llm_judge_relation": "ai_tool_for_review_workflow",
        "llm_uses_ai_for_review_workflow": True,
        "llm_is_review_about_ai": False,
        "directional_judge_source": "llm",
        "directional_relation": "ai_tool_for_review_workflow",
        "directional_confidence": 0.9,
        "directional_uses_ai_for_review_workflow": True,
        "directional_is_review_about_ai_external_domain": False,
        "directional_reason": "workflow",
        "llm_directional_judge_used": True,
        "llm_directional_judge_error": "",
    })
    fused, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper(),
        deterministic_result={
            "decision": "REJECT",
            "reason": "deterministic",
            "method_evidence_terms": "large language models",
        },
    )
    assert fused["decision"] == "MAYBE"
    assert diagnostics["model_promoted_from_reject"] is True


def test_model_fusion_promotes_maybe_to_keep(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setenv("ENABLE_LLM_JUDGE_FOR_SMOKE", "true")
    monkeypatch.setattr("model_score_fusion.judge_reranker", lambda *a, **k: {
        "model_reranker_relevance_score": 0.84,
        "model_reranker_negative_score": 0.10,
        "model_reranker_margin": 0.74,
        "model_reranker_decision_hint": "positive",
    })
    monkeypatch.setattr("model_score_fusion.judge_nli", lambda *a, **k: {
        "nli_positive_entailment_score": 0.82,
        "nli_negative_entailment_score": 0.10,
        "nli_contradiction_score": 0.0,
        "nli_margin": 0.72,
        "nli_decision_hint": "positive",
        "nli_top_positive_hypothesis": "",
        "nli_top_negative_hypothesis": "",
    })
    monkeypatch.setattr("model_score_fusion.judge_zero_shot_relation", lambda *a, **k: {
        "zeroshot_relation_label": "AI tool for systematic review workflow",
        "zeroshot_relation_score": 0.82,
        "zeroshot_ai_tool_for_review_score": 0.82,
        "zeroshot_subject_review_score": 0.10,
        "zeroshot_top_labels": "",
    })
    monkeypatch.setattr("model_score_fusion.judge_with_llm", lambda *a, **k: {
        "llm_judge_decision": "KEEP",
        "llm_judge_relation": "ai_tool_for_review_workflow",
        "llm_uses_ai_for_review_workflow": True,
        "llm_is_review_about_ai": False,
        "directional_judge_source": "llm",
        "directional_relation": "ai_tool_for_review_workflow",
        "directional_confidence": 0.9,
        "directional_uses_ai_for_review_workflow": True,
        "directional_is_review_about_ai_external_domain": False,
        "directional_reason": "workflow",
        "llm_directional_judge_used": True,
        "llm_directional_judge_error": "",
    })
    fused, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper(),
        deterministic_result={
            "decision": "MAYBE",
            "reason": "deterministic",
            "method_evidence_terms": "large language models",
        },
    )
    assert fused["decision"] == "KEEP"
    assert diagnostics["model_promoted_from_maybe"] is True


def test_model_fusion_demotes_keep_on_subject_review_risk(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setattr("model_score_fusion.judge_reranker", lambda *a, **k: {
        "model_reranker_relevance_score": 0.20,
        "model_reranker_negative_score": 0.85,
        "model_reranker_margin": -0.65,
        "model_reranker_decision_hint": "negative",
    })
    monkeypatch.setattr("model_score_fusion.judge_nli", lambda *a, **k: {
        "nli_positive_entailment_score": 0.15,
        "nli_negative_entailment_score": 0.82,
        "nli_contradiction_score": 0.67,
        "nli_margin": -0.67,
        "nli_decision_hint": "negative",
        "nli_top_positive_hypothesis": "",
        "nli_top_negative_hypothesis": "negative",
    })
    monkeypatch.setattr("model_score_fusion.judge_zero_shot_relation", lambda *a, **k: {
        "zeroshot_relation_label": "systematic review about AI applications",
        "zeroshot_relation_score": 0.80,
        "zeroshot_ai_tool_for_review_score": 0.15,
        "zeroshot_subject_review_score": 0.80,
        "zeroshot_top_labels": "",
    })
    fused, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper("AI in education", "A systematic review about AI in education."),
        deterministic_result={"decision": "KEEP", "reason": "deterministic"},
    )
    assert fused["decision"] == "MAYBE"
    assert diagnostics["model_demoted_from_keep"] is True


def test_model_fusion_preserves_maybe_when_models_disagree(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setattr("model_score_fusion.judge_reranker", lambda *a, **k: {
        "model_reranker_relevance_score": 0.60,
        "model_reranker_negative_score": 0.58,
        "model_reranker_margin": 0.02,
        "model_reranker_decision_hint": "uncertain",
    })
    monkeypatch.setattr("model_score_fusion.judge_nli", lambda *a, **k: {
        "nli_positive_entailment_score": 0.55,
        "nli_negative_entailment_score": 0.56,
        "nli_contradiction_score": 0.01,
        "nli_margin": -0.01,
        "nli_decision_hint": "uncertain",
        "nli_top_positive_hypothesis": "",
        "nli_top_negative_hypothesis": "",
    })
    monkeypatch.setattr("model_score_fusion.judge_zero_shot_relation", lambda *a, **k: {
        "zeroshot_relation_label": "systematic review about AI applications",
        "zeroshot_relation_score": 0.56,
        "zeroshot_ai_tool_for_review_score": 0.55,
        "zeroshot_subject_review_score": 0.56,
        "zeroshot_top_labels": "",
    })
    fused, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper(),
        deterministic_result={"decision": "MAYBE", "reason": "deterministic"},
    )
    assert fused["decision"] == "MAYBE"
    assert diagnostics["model_fusion_action"] == "directional_preserve_unclear"


def test_model_fusion_marks_high_high_small_margin_uncertain(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setattr("model_score_fusion.judge_reranker", lambda *a, **k: {
        "model_reranker_relevance_score": 0.9994,
        "model_reranker_negative_score": 0.9995,
        "model_reranker_margin": -0.0001,
        "model_reranker_decision_hint": "uncertain",
    })
    monkeypatch.setattr("model_score_fusion.judge_nli", lambda *a, **k: {})
    monkeypatch.setattr("model_score_fusion.judge_zero_shot_relation", lambda *a, **k: {})
    fused, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper(),
        deterministic_result={"decision": "MAYBE", "reason": "deterministic", "method_evidence_terms": "large language models"},
    )
    assert fused["decision"] == "MAYBE"
    assert diagnostics["model_uncertainty_score"] >= 0.90
    assert diagnostics["model_fusion_action"] == "directional_preserve_unclear"


def test_reranker_strong_positive_dominates_fusion(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("MODEL_JUDGE_PROFILE", "light")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setenv("ENABLE_LLM_JUDGE_FOR_SMOKE", "true")
    monkeypatch.setattr("model_score_fusion.judge_reranker", lambda *a, **k: {
        "model_reranker_relevance_score": 0.82,
        "model_reranker_negative_score": 0.18,
        "model_reranker_margin": 0.64,
        "model_reranker_decision_hint": "positive",
        "reranker_runtime_source": "hf_model",
    })
    monkeypatch.setattr("model_score_fusion.judge_with_llm", lambda *a, **k: {
        "llm_judge_decision": "KEEP",
        "llm_judge_relation": "ai_tool_for_review_workflow",
        "llm_uses_ai_for_review_workflow": True,
        "llm_is_review_about_ai": False,
        "directional_judge_source": "llm",
        "directional_relation": "ai_tool_for_review_workflow",
        "directional_confidence": 0.9,
        "directional_uses_ai_for_review_workflow": True,
        "directional_is_review_about_ai_external_domain": False,
        "directional_reason": "workflow",
        "llm_directional_judge_used": True,
        "llm_directional_judge_error": "",
    })
    fused, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper(),
        deterministic_result={"decision": "MAYBE", "reason": "deterministic", "method_evidence_terms": "large language models"},
    )
    assert diagnostics["model_primary_signal"] == "llm"
    assert diagnostics["model_positive_score"] > diagnostics["model_negative_score"] + 0.10
    assert fused["decision"] == "KEEP"


def test_reranker_strong_negative_dominates_fusion(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("MODEL_JUDGE_PROFILE", "light")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setenv("ENABLE_LLM_JUDGE_FOR_SMOKE", "true")
    monkeypatch.setattr("model_score_fusion.judge_reranker", lambda *a, **k: {
        "model_reranker_relevance_score": 0.15,
        "model_reranker_negative_score": 0.85,
        "model_reranker_margin": -0.70,
        "model_reranker_decision_hint": "negative",
        "reranker_runtime_source": "hf_model",
    })
    monkeypatch.setattr("model_score_fusion.judge_with_llm", lambda *a, **k: {
        "llm_judge_decision": "REJECT",
        "llm_judge_relation": "review_about_ai_external_domain",
        "llm_uses_ai_for_review_workflow": False,
        "llm_is_review_about_ai": True,
        "directional_judge_source": "llm",
        "directional_relation": "review_about_ai_external_domain",
        "directional_confidence": 0.9,
        "directional_uses_ai_for_review_workflow": False,
        "directional_is_review_about_ai_external_domain": True,
        "directional_reason": "external",
        "llm_directional_judge_used": True,
        "llm_directional_judge_error": "",
    })
    fused, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper("LLMs in breast cancer diagnosis", "A systematic review of LLM applications in breast cancer diagnosis."),
        deterministic_result={"decision": "KEEP", "reason": "deterministic"},
    )
    assert diagnostics["model_primary_signal"] == "llm"
    assert diagnostics["model_negative_score"] > diagnostics["model_positive_score"] + 0.10
    assert fused["decision"] == "MAYBE"


def test_reranker_alone_cannot_promote_without_directional_support(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("MODEL_JUDGE_PROFILE", "light")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.delenv("ENABLE_LLM_JUDGE_FOR_SMOKE", raising=False)
    monkeypatch.setattr("model_score_fusion.judge_reranker", lambda *a, **k: {
        "model_reranker_relevance_score": 0.90,
        "model_reranker_negative_score": 0.10,
        "model_reranker_margin": 0.80,
        "model_reranker_decision_hint": "positive",
        "reranker_runtime_source": "hf_model",
    })
    fused, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper(),
        deterministic_result={"decision": "MAYBE", "reason": "deterministic", "method_evidence_terms": "large language models"},
    )
    assert fused["decision"] == "MAYBE"
    assert diagnostics["model_fusion_action"] == "preserve"


def test_llm_directional_external_demotes_false_keep(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("MODEL_JUDGE_PROFILE", "light")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setenv("ENABLE_LLM_JUDGE_FOR_SMOKE", "true")
    monkeypatch.setattr("model_score_fusion.judge_reranker", lambda *a, **k: {
        "model_reranker_relevance_score": 0.60,
        "model_reranker_negative_score": 0.40,
        "model_reranker_margin": 0.20,
        "model_reranker_decision_hint": "positive",
        "reranker_runtime_source": "hf_model",
    })
    monkeypatch.setattr("model_score_fusion.judge_with_llm", lambda *a, **k: {
        "llm_judge_decision": "REJECT",
        "llm_judge_relation": "review_about_ai_external_domain",
        "llm_uses_ai_for_review_workflow": False,
        "llm_is_review_about_ai": True,
        "directional_judge_source": "llm",
        "directional_relation": "review_about_ai_external_domain",
        "directional_confidence": 0.9,
        "directional_uses_ai_for_review_workflow": False,
        "directional_is_review_about_ai_external_domain": True,
        "directional_reason": "external",
        "llm_directional_judge_used": True,
        "llm_directional_judge_error": "",
    })
    fused, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper("LLMs for breast cancer diagnosis", "A systematic review about LLMs in breast cancer diagnosis."),
        deterministic_result={"decision": "KEEP", "reason": "deterministic"},
    )
    assert fused["decision"] == "MAYBE"
    assert diagnostics["model_demoted_from_keep"] is True


def test_llm_directional_parse_failure_preserves(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("MODEL_JUDGE_PROFILE", "light")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setenv("ENABLE_LLM_JUDGE_FOR_SMOKE", "true")
    monkeypatch.setattr("model_score_fusion.judge_reranker", lambda *a, **k: {
        "model_reranker_relevance_score": 0.90,
        "model_reranker_negative_score": 0.10,
        "model_reranker_margin": 0.80,
        "model_reranker_decision_hint": "positive",
        "reranker_runtime_source": "hf_model",
    })
    monkeypatch.setattr("model_score_fusion.judge_with_llm", lambda *a, **k: {
        "llm_directional_judge_error": "bad json",
        "directional_judge_source": "none",
    })
    fused, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper(),
        deterministic_result={"decision": "MAYBE", "reason": "deterministic", "method_evidence_terms": "large language models"},
    )
    assert fused["decision"] == "MAYBE"
    assert diagnostics["model_fusion_action"] == "preserve"


def test_model_fusion_is_secondary_for_medical_and_blockchain(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setattr("model_score_fusion.judge_reranker", lambda *a, **k: {
        "model_reranker_relevance_score": 0.10,
        "model_reranker_negative_score": 0.90,
        "model_reranker_margin": -0.80,
        "model_reranker_decision_hint": "negative",
    })
    monkeypatch.setattr("model_score_fusion.judge_nli", lambda *a, **k: {
        "nli_positive_entailment_score": 0.10,
        "nli_negative_entailment_score": 0.90,
        "nli_contradiction_score": 0.80,
        "nli_margin": -0.80,
        "nli_decision_hint": "negative",
        "nli_top_positive_hypothesis": "",
        "nli_top_negative_hypothesis": "",
    })
    monkeypatch.setattr("model_score_fusion.judge_zero_shot_relation", lambda *a, **k: {
        "zeroshot_relation_label": "external-domain AI task",
        "zeroshot_relation_score": 0.90,
        "zeroshot_ai_tool_for_review_score": 0.10,
        "zeroshot_subject_review_score": 0.90,
        "zeroshot_top_labels": "",
    })
    fused, diagnostics = apply_model_score_fusion(
        rq_frame=_rq("method_for_task_in_domain_review"),
        paper_frame=_paper(),
        deterministic_result={"decision": "KEEP", "reason": "deterministic"},
    )
    assert fused["decision"] == "KEEP"
    assert diagnostics["model_fusion_action"] == "preserve"


def test_model_diagnostics_fields_exported():
    fields = _result_semantic_fields({"model_judges_enabled": True}, "stage1_")
    assert "stage1_model_judges_enabled" in fields
    assert "stage1_model_positive_score" in fields
    assert "stage1_model_judge_runtime_source" in fields
    assert "stage1_model_profile" in fields
    assert "stage1_model_timing_seconds" in fields


def test_model_judge_exception_does_not_create_parse_error(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setattr("model_score_fusion.judge_reranker", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("model_score_fusion.judge_nli", lambda *a, **k: {})
    monkeypatch.setattr("model_score_fusion.judge_zero_shot_relation", lambda *a, **k: {})
    fused, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper(),
        deterministic_result={"decision": "MAYBE", "reason": "baseline"},
    )
    assert fused["decision"] == "MAYBE"
    assert diagnostics["model_fusion_action"] == "fallback_error_preserve"
    assert diagnostics["model_judge_fallback_used"] is True


def test_deterministic_stage1_exception_falls_back_and_attaches_models(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setenv("ENABLE_HF_MODEL_LOADING", "false")
    import screener

    monkeypatch.setattr(screener, "extract_semantic_frame", lambda **k: _paper())
    monkeypatch.setattr(
        screener,
        "compare_semantic_frames",
        lambda *a, **k: (_ for _ in ()).throw(KeyError("artificial intelligence; large language models")),
    )
    result = screener.screen_paper(
        title="LLM screening",
        abstract="Uses LLMs for systematic review screening.",
        research_question=_rq()["rq_text"],
        rq_frame=_rq(),
    )
    assert result["decision"] in {"KEEP", "MAYBE", "REJECT"}
    assert result["decision"] != "PARSE_ERROR"
    assert result["stage1_error_type"] == "KeyError"
    assert result["model_judges_enabled"] is True
    assert result["model_judge_mode"] == "balanced"


def test_local_screening_csv_exports_model_config_and_stage1(monkeypatch):
    import bulk_screen

    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setenv("ENABLE_HF_MODEL_LOADING", "false")
    tmp_dir = Path(".tmp_model_judge_test")
    tmp_dir.mkdir(exist_ok=True)
    csv_path = tmp_dir / "input.csv"
    output_path = tmp_dir / "screened.csv"
    pd.DataFrame([{"Title": "Paper", "Abstract": "Abstract"}]).to_csv(csv_path, index=False)

    monkeypatch.setattr(bulk_screen, "extract_research_question_frame", lambda **k: _rq())
    monkeypatch.setattr(bulk_screen, "profile_corpus", lambda *a, **k: {})
    monkeypatch.setattr(bulk_screen, "enrich_research_question_frame_with_corpus", lambda rq, profile: rq)
    monkeypatch.setattr(
        bulk_screen,
        "screen_candidate",
        lambda **k: {
            "title": k["title"],
            "abstract": k["abstract"],
            "decision": "MAYBE",
            "reason": "stub",
            "model_judges_enabled": True,
            "model_judge_mode": "balanced",
            "model_judge_runtime_source": "hf_model",
            "model_profile": "light",
            "model_real_models_loaded": True,
            "model_fallback_reason": "",
            "model_timing_seconds": 0.12,
            "model_fusion_action": "preserve",
        },
    )
    try:
        summary = bulk_screen.screen_csv(
            csv_path=str(csv_path),
            research_question=_rq()["rq_text"],
            output_path=str(output_path),
            two_stage_enabled=False,
        )
        assert summary["total_papers"] == 1
        row = pd.read_csv(output_path).iloc[0].to_dict()
        assert row["model_config_enable_model_judges"] == True
        assert row["model_config_model_judge_mode"] == "balanced"
        assert row["model_config_enable_hf_model_loading"] == False
        assert row["stage1_model_judges_enabled"] == True
        assert row["stage1_model_judge_mode"] == "balanced"
        assert row["stage1_model_judge_runtime_source"] == "hf_model"
        assert row["stage1_model_profile"] == "light"
        assert row["stage2_model_fusion_action"] == "stage2_not_run"
    finally:
        for path in (csv_path, output_path):
            if path.exists():
                path.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


def test_bulk_screen_without_row_limit_screens_all_rows(monkeypatch):
    import bulk_screen

    monkeypatch.setattr(bulk_screen, "DEV_SCREENING_ROW_LIMIT", None)
    csv_path, output_path, tmp_dir = _write_row_limit_csv(5, "all")
    _patch_bulk_screen_for_row_limit(monkeypatch, bulk_screen)
    try:
        summary = bulk_screen.screen_csv(
            csv_path=str(csv_path),
            research_question=_rq()["rq_text"],
            output_path=str(output_path),
            two_stage_enabled=False,
            max_rows=None,
        )
        rows = pd.read_csv(output_path)
        assert summary["total_papers"] == 5
        assert summary["screened_total_rows"] == 5
        assert summary["row_limit_applied"] is False
        assert len(rows) == 5
        assert rows["screened_total_rows"].iloc[0] == 5
        assert rows["row_limit_applied"].astype(str).str.lower().iloc[0] == "false"
    finally:
        _cleanup_row_limit_files(csv_path, output_path, tmp_dir)


def test_bulk_screen_row_limit_positive_screens_limited_rows(monkeypatch):
    import bulk_screen

    monkeypatch.setattr(bulk_screen, "DEV_SCREENING_ROW_LIMIT", None)
    csv_path, output_path, tmp_dir = _write_row_limit_csv(120, "limited")
    _patch_bulk_screen_for_row_limit(monkeypatch, bulk_screen)
    try:
        summary = bulk_screen.screen_csv(
            csv_path=str(csv_path),
            research_question=_rq()["rq_text"],
            output_path=str(output_path),
            two_stage_enabled=False,
            max_rows=100,
        )
        rows = pd.read_csv(output_path)
        assert summary["total_papers"] == 100
        assert summary["input_total_rows"] == 120
        assert summary["screened_total_rows"] == 100
        assert summary["row_limit_applied"] is True
        assert summary["row_limit_value"] == 100
        assert len(rows) == 100
    finally:
        _cleanup_row_limit_files(csv_path, output_path, tmp_dir)


def test_bulk_screen_zero_row_limit_screens_all_rows(monkeypatch):
    import bulk_screen

    monkeypatch.setattr(bulk_screen, "DEV_SCREENING_ROW_LIMIT", None)
    csv_path, output_path, tmp_dir = _write_row_limit_csv(4, "zero")
    _patch_bulk_screen_for_row_limit(monkeypatch, bulk_screen)
    try:
        summary = bulk_screen.screen_csv(
            csv_path=str(csv_path),
            research_question=_rq()["rq_text"],
            output_path=str(output_path),
            two_stage_enabled=False,
            max_rows=0,
        )
        assert summary["total_papers"] == 4
        assert summary["row_limit_applied"] is False
        assert pd.read_csv(output_path).shape[0] == 4
    finally:
        _cleanup_row_limit_files(csv_path, output_path, tmp_dir)


def test_benchmark_uses_first_n_without_forcing_production_default(monkeypatch):
    import benchmark_local_screening

    scratch = Path(".tmp_model_judge_test")
    scratch.mkdir(exist_ok=True)
    input_path = scratch / "benchmark_input.csv"
    config_path = scratch / "benchmark_config.json"
    output_path = scratch / "benchmark_output.csv"
    pd.DataFrame([{"Title": "Paper", "Abstract": "Abstract"}]).to_csv(input_path, index=False)
    config_path.write_text(
        json.dumps({
            "dataset_path": str(input_path),
            "research_question": "RQ",
            "first_n": 100,
            "output_path": str(output_path),
        }),
        encoding="utf-8",
    )
    seen = {}

    def fake_screen_csv(**kwargs):
        seen["max_rows"] = kwargs.get("max_rows")
        pd.DataFrame([{"Title": "Paper", "Decision": "KEEP"}]).to_csv(output_path, index=False)

    monkeypatch.setattr(benchmark_local_screening, "screen_csv", fake_screen_csv)
    try:
        benchmark_local_screening.run_benchmark(str(config_path))
        assert seen["max_rows"] == 100
    finally:
        for path in (input_path, config_path, output_path):
            path.unlink(missing_ok=True)
        try:
            scratch.rmdir()
        except OSError:
            pass


def test_server_row_limit_default_and_zero_are_full_dataset():
    import server

    assert server._normalize_row_limit(None) is None
    assert server._normalize_row_limit(0) is None
    assert server._normalize_row_limit("") is None
    assert server._normalize_row_limit(100) == 100


def test_high_confidence_workflow_keep_uses_equivalence_fallback(monkeypatch):
    monkeypatch.setenv("SCREENING_PIPELINE_MODE", "two_pass_fast")
    monkeypatch.setenv("ENABLE_LLM_JUDGE", "true")
    monkeypatch.setenv("ENABLE_HF_MODEL_LOADING", "false")
    monkeypatch.setenv("ENABLE_AGGRESSIVE_LLM_GATING", "true")

    class DirectionalEngine:
        def ask(self, *args, **kwargs):
            return json.dumps({
                "decision": "KEEP",
                "relation": "ai_tool_for_review_workflow",
                "uses_ai_for_review_workflow": True,
                "is_review_about_ai_external_domain": False,
                "confidence": 0.9,
                "relation_confidence": 0.9,
                "workflow_evidence_quote": "title abstract screening in systematic review workflows",
                "task_object_type": "review_object",
            })

    fused, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper(
            "LLM title abstract screening for systematic reviews",
            "Large language models automate title abstract screening in systematic review workflows.",
        ),
        deterministic_result={
            "decision": "KEEP",
            "relation_match": True,
            "paper_observed_relation": "tool_used_for_workflow",
            "relation_evidence_strength": 0.9,
            "method_evidence_terms": "large language models",
        },
        research_question=_rq()["rq_text"],
        inference_engine=DirectionalEngine(),
        mode="balanced",
    )
    assert diagnostics["llm_route"] == "llm_required_current_equivalence"
    assert diagnostics["llm_directional_judge_used"] is True
    assert diagnostics["fast_mode_current_equivalence_blocked_reason"] == "workflow_keep_would_demote"
    assert fused["decision"] == "KEEP"


def test_current_mode_does_not_enable_aggressive_gating_by_default(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_JUDGE", "true")
    monkeypatch.setenv("SCREENING_PIPELINE_MODE", "current")
    monkeypatch.delenv("ENABLE_AGGRESSIVE_LLM_GATING", raising=False)
    monkeypatch.setenv("ENABLE_HF_MODEL_LOADING", "false")

    _, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper(
            "LLM title abstract screening for systematic reviews",
            "Large language models automate title abstract screening in systematic review workflows.",
        ),
        deterministic_result={
            "decision": "KEEP",
            "relation_match": True,
            "paper_observed_relation": "tool_used_for_workflow",
            "relation_evidence_strength": 0.9,
            "method_evidence_terms": "large language models",
        },
        research_question=_rq()["rq_text"],
        mode="balanced",
    )
    assert diagnostics["llm_route"] == "legacy_current_mode"


def test_two_pass_fast_enables_aggressive_gating(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_JUDGE", "true")
    monkeypatch.setenv("SCREENING_PIPELINE_MODE", "two_pass_fast")
    monkeypatch.setenv("ENABLE_AGGRESSIVE_LLM_GATING", "true")
    monkeypatch.setenv("ENABLE_HF_MODEL_LOADING", "false")

    class DirectionalEngine:
        def ask(self, *args, **kwargs):
            return json.dumps({
                "decision": "KEEP", "relation": "ai_tool_for_review_workflow",
                "uses_ai_for_review_workflow": True, "is_review_about_ai_external_domain": False,
                "confidence": 0.9, "relation_confidence": 0.9,
                "workflow_evidence_quote": "screening in systematic review workflows",
                "task_object_type": "review_object",
            })

    _, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper(
            "LLM title abstract screening for systematic reviews",
            "Large language models automate title abstract screening in systematic review workflows.",
        ),
        deterministic_result={
            "decision": "KEEP",
            "relation_match": True,
            "paper_observed_relation": "tool_used_for_workflow",
            "relation_evidence_strength": 0.9,
            "method_evidence_terms": "large language models",
        },
        research_question=_rq()["rq_text"],
        inference_engine=DirectionalEngine(),
        mode="balanced",
    )
    assert diagnostics["llm_route"] == "llm_required_current_equivalence"


def test_high_confidence_external_skips_llm_and_blocks_keep(monkeypatch):
    monkeypatch.setenv("SCREENING_PIPELINE_MODE", "two_pass_fast")
    monkeypatch.setenv("ENABLE_LLM_JUDGE", "true")
    monkeypatch.setenv("ENABLE_HF_MODEL_LOADING", "false")
    monkeypatch.setenv("ENABLE_AGGRESSIVE_LLM_GATING", "true")

    class FailEngine:
        def ask(self, *args, **kwargs):
            raise AssertionError("LLM should be skipped")

    fused, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper(
            "Applications of LLMs in Breast Cancer Diagnosis: A Systematic Review",
            "This systematic review summarizes large language model applications for breast cancer diagnosis in patients.",
        ),
        deterministic_result={
            "decision": "KEEP",
            "relation_match": True,
            "paper_observed_relation": "external_domain_review",
            "method_evidence_terms": "large language models",
        },
        research_question=_rq()["rq_text"],
        inference_engine=FailEngine(),
        mode="balanced",
    )
    assert diagnostics["llm_route"] == "skipped_high_confidence_external"
    assert diagnostics["llm_directional_judge_used"] is False
    assert fused["decision"] != "KEEP"


def test_uncertain_row_routes_to_llm(monkeypatch):
    monkeypatch.setenv("SCREENING_PIPELINE_MODE", "two_pass_fast")
    monkeypatch.setenv("ENABLE_LLM_JUDGE", "true")
    monkeypatch.setenv("ENABLE_HF_MODEL_LOADING", "false")
    monkeypatch.setenv("ENABLE_AGGRESSIVE_LLM_GATING", "true")

    class FakeEngine:
        def ask(self, *args, **kwargs):
            return json.dumps({
                "decision": "KEEP",
                "relation": "ai_tool_for_review_workflow",
                "uses_ai_for_review_workflow": True,
                "is_review_about_ai_external_domain": False,
                "confidence": 0.9,
                "workflow_tasks_detected": ["study selection"],
                "workflow_evidence_quote": "study selection in systematic reviews",
                "task_object_type": "review_object",
                "reason": "workflow",
            })

    _, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(),
        paper_frame=_paper("LLM support for study selection", "The system may support study selection in reviews."),
        deterministic_result={
            "decision": "MAYBE",
            "relation_match": False,
            "paper_observed_relation": "unclear_relation",
            "method_evidence_terms": "large language models",
        },
        research_question=_rq()["rq_text"],
        inference_engine=FakeEngine(),
        mode="balanced",
    )
    assert diagnostics["llm_route"].startswith("llm_required")
    assert diagnostics["llm_directional_judge_used"] is True


def test_batch_llm_parser_returns_per_row_outputs(monkeypatch):
    import llm_structured_judge as judge

    judge._DIRECTIONAL_CACHE.clear()

    class FakeEngine:
        def ask(self, *args, **kwargs):
            return json.dumps([
                {
                    "id": "A",
                    "relation": "ai_tool_for_review_workflow",
                    "uses_ai_for_review_workflow": True,
                    "is_review_about_ai_external_domain": False,
                    "confidence": 0.9,
                    "workflow_tasks_detected": ["citation screening"],
                    "task_object_type": "review_object",
                },
                {
                    "id": "B",
                    "relation": "review_about_ai_external_domain",
                    "uses_ai_for_review_workflow": False,
                    "is_review_about_ai_external_domain": True,
                    "confidence": 0.9,
                    "external_domain_tasks_detected": ["diagnosis"],
                    "task_object_type": "external_domain_object",
                },
            ])

    rows = [
        {"id": "A", "title": "A", "abstract": "LLM citation screening."},
        {"id": "B", "title": "B", "abstract": "LLM diagnosis review."},
    ]
    result = judge.judge_batch_with_llm(rows, "RQ", "model", FakeEngine())
    assert result["A"]["directional_relation"] == "ai_tool_for_review_workflow"
    assert result["B"]["directional_relation"] == "review_about_ai_external_domain"


def test_batch_missing_item_falls_back_only_that_item(monkeypatch):
    import llm_structured_judge as judge

    judge._DIRECTIONAL_CACHE.clear()
    calls = {"count": 0}

    class FakeEngine:
        def ask(self, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return json.dumps([
                    {
                        "id": "A",
                        "relation": "ai_tool_for_review_workflow",
                        "uses_ai_for_review_workflow": True,
                        "is_review_about_ai_external_domain": False,
                        "confidence": 0.9,
                    }
                ])
            return json.dumps({
                "decision": "MAYBE",
                "relation": "unclear",
                "uses_ai_for_review_workflow": False,
                "is_review_about_ai_external_domain": False,
                "confidence": 0.4,
            })

    rows = [
        {"id": "A", "title": "A", "abstract": "LLM citation screening."},
        {"id": "B", "title": "B", "abstract": "Unclear."},
    ]
    result = judge.judge_batch_with_llm(rows, "RQ", "model", FakeEngine())
    assert calls["count"] == 2
    assert "A" in result and "B" in result
    assert result["A"]["directional_relation"] == "ai_tool_for_review_workflow"


def test_persistent_cache_prevents_repeated_llm_call(monkeypatch):
    import llm_structured_judge as judge

    scratch = Path(".tmp_model_judge_test")
    scratch.mkdir(exist_ok=True)
    cache_path = scratch / "cache_repeat.jsonl"
    calls = {"count": 0}

    class FakeEngine:
        def ask(self, *args, **kwargs):
            calls["count"] += 1
            return json.dumps({
                "decision": "KEEP",
                "relation": "ai_tool_for_review_workflow",
                "uses_ai_for_review_workflow": True,
                "is_review_about_ai_external_domain": False,
                "confidence": 0.9,
            })

    try:
        monkeypatch.setattr(judge, "CACHE_PATH", str(cache_path))
        judge.clear_cache()
        first = judge.judge_with_llm("T", "A", "RQ", "model", FakeEngine())
        second = judge.judge_with_llm("T", "A", "RQ", "model", FakeEngine())
        assert calls["count"] == 1
        assert first["llm_directional_cache_hit"] is False
        assert second["llm_directional_cache_hit"] is True
    finally:
        cache_path.unlink(missing_ok=True)
        try:
            scratch.rmdir()
        except OSError:
            pass


def _write_row_limit_csv(rows, suffix):
    tmp_dir = Path(".tmp_model_judge_test")
    tmp_dir.mkdir(exist_ok=True)
    csv_path = tmp_dir / f"row_limit_{suffix}.csv"
    output_path = tmp_dir / f"row_limit_{suffix}_screened.csv"
    pd.DataFrame([
        {"Title": f"Paper {i}", "Abstract": f"Abstract {i}"}
        for i in range(rows)
    ]).to_csv(csv_path, index=False)
    return csv_path, output_path, tmp_dir


def _patch_bulk_screen_for_row_limit(monkeypatch, bulk_screen):
    monkeypatch.setattr(bulk_screen, "extract_research_question_frame", lambda **k: _rq())
    monkeypatch.setattr(bulk_screen, "profile_corpus", lambda *a, **k: {})
    monkeypatch.setattr(bulk_screen, "enrich_research_question_frame_with_corpus", lambda rq, profile: rq)
    monkeypatch.setattr(
        bulk_screen,
        "screen_candidate",
        lambda **k: {
            "title": k["title"],
            "abstract": k["abstract"],
            "decision": "MAYBE",
            "reason": "stub",
        },
    )


def _cleanup_row_limit_files(csv_path, output_path, tmp_dir):
    for path in (csv_path, output_path):
        path.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass


def test_debug_endpoint_returns_effective_config(monkeypatch):
    monkeypatch.setenv("MODEL_JUDGE_MODE", "balanced")
    monkeypatch.setenv("ENABLE_MODEL_JUDGES", "true")
    monkeypatch.setenv("ENABLE_HF_MODEL_LOADING", "false")
    import server

    payload = asyncio.run(server.debug_model_judge_config())
    assert payload["enable_model_judges"] is True
    assert payload["model_judge_mode"] == "balanced"
    assert payload["enable_hf_model_loading"] is False


def test_real_model_smoke_command_exists():
    path = Path("debug_model_judge_runtime.py")
    text = path.read_text(encoding="utf-8")
    assert "--real-model-smoke" in text
    assert "--download-check" in text


def test_reranker_cache_avoids_duplicate_model_calls(monkeypatch):
    from reranker_judge import _score_cached

    calls = {"count": 0}

    class FakeCrossEncoder:
        def predict(self, pairs):
            calls["count"] += 1
            return [0.8, 0.2]

    _score_cached.cache_clear()
    monkeypatch.setattr("reranker_judge.get_cross_encoder", lambda name: FakeCrossEncoder())
    judge_reranker("rq", "title", "abstract", "balanced")
    judge_reranker("rq", "title", "abstract", "balanced")
    assert calls["count"] == 1
    _score_cached.cache_clear()


def test_benchmark_supports_model_mode_argument(monkeypatch):
    import benchmark_local_screening
    import pandas as pd
    from pathlib import Path

    scratch = Path(".codex_test_outputs")
    scratch.mkdir(exist_ok=True)
    input_path = scratch / "input.csv"
    output_path = scratch / "screened_model_mode.csv"
    config_path = scratch / "benchmark_model_mode.json"
    try:
        pd.DataFrame([{"Title": "A", "Abstract": "B"}]).to_csv(input_path, index=False)
        config_path.write_text(
            '{"dataset_path":"%s","research_question":"RQ","output_path":"%s"}'
            % (
                str(input_path).replace("\\", "\\\\"),
                str(output_path).replace("\\", "\\\\"),
            ),
            encoding="utf-8",
        )

        def fake_screen_csv(**kwargs):
            assert os.environ["MODEL_JUDGE_MODE"] == "balanced"
            pd.DataFrame([{
                "Title": "A",
                "Decision": "KEEP",
                "stage1_model_judges_enabled": True,
                "stage1_model_judge_mode": "balanced",
                "stage1_model_positive_score": 0.9,
            }]).to_csv(output_path, index=False)

        monkeypatch.setattr(benchmark_local_screening, "screen_csv", fake_screen_csv)
        summary = benchmark_local_screening.run_benchmark(str(config_path), "balanced")
        assert summary["model_assisted_summary"]["enabled"] is True
        assert summary["model_assisted_summary"]["mode"] == "balanced"
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)


def test_final_adjudicator_promotes_valid_workflow_maybe():
    from final_adjudicator import adjudicate_row

    row = {
        "Decision": "MAYBE",
        "Title": "Large Language Models for Title and Abstract Screening in Systematic Reviews",
        "Abstract": "LLMs perform title abstract screening and study selection for systematic reviews.",
        "stage1_rq_review_question_type": "review_workflow_automation",
        "stage1_directional_relation": "ai_tool_for_review_workflow",
        "stage1_directional_confidence": 0.9,
        "stage1_directional_uses_ai_for_review_workflow": True,
        "stage1_workflow_evidence_quote": "title abstract screening and study selection for systematic reviews",
        "stage1_task_object_type": "review_object",
    }
    adjudication = adjudicate_row(row)
    assert adjudication["final_adjudicated_decision"] == "KEEP"
    assert adjudication["final_adjudication_action"] == "promote_maybe_to_keep"
    assert adjudication["relation_evidence_valid"] is True


def test_final_adjudicator_demotes_external_domain_keep():
    from final_adjudicator import adjudicate_row

    row = {
        "Decision": "KEEP",
        "Title": "Applications of Large Language Models in Breast Cancer Diagnosis: A Systematic Review",
        "Abstract": "The review summarizes LLM applications for breast cancer diagnosis in patients.",
        "stage1_rq_review_question_type": "review_workflow_automation",
        "stage1_directional_relation": "review_about_ai_external_domain",
        "stage1_directional_confidence": 0.88,
        "stage1_directional_is_review_about_ai_external_domain": True,
        "stage1_external_domain_evidence_quote": "breast cancer diagnosis in patients",
        "stage1_task_object_type": "external_domain_object",
    }
    adjudication = adjudicate_row(row)
    assert adjudication["final_adjudicated_decision"] == "MAYBE"
    assert adjudication["final_adjudication_action"] == "demote_keep_to_maybe"
    assert adjudication["final_external_domain"] is True


def test_final_adjudicator_softens_unclear_external_reject_with_review_process_evidence():
    from final_adjudicator import adjudicate_row

    row = {
        "Decision": "REJECT",
        "Title": "Application of Explainable Artificial Intelligence in Fintech",
        "Abstract": (
            "A systematic literature review is conducted by screening over 1000 articles "
            "from finance and information systems outlets."
        ),
        "stage1_rq_review_question_type": "review_workflow_automation",
        "stage1_directional_relation": "review_about_ai_external_domain",
        "stage1_directional_confidence": 0.86,
        "stage1_directional_is_review_about_ai_external_domain": True,
        "stage1_external_domain_evidence_quote": "fintech and finance outlets",
        "stage1_workflow_evidence_quote": "screening over 1000 articles",
        "stage1_task_object_type": "unclear",
    }

    adjudication = adjudicate_row(row)

    assert adjudication["final_adjudicated_decision"] == "MAYBE"
    assert adjudication["final_adjudication_action"] == "external_domain_uncertain_reject_to_maybe"


def test_final_adjudicator_keeps_external_reject_when_only_systematic_review_phrase():
    from final_adjudicator import adjudicate_row

    row = {
        "Decision": "REJECT",
        "Title": "Applications of Large Language Models in Breast Cancer Diagnosis: A Systematic Review",
        "Abstract": "This systematic review summarizes LLM applications for breast cancer diagnosis in patients.",
        "stage1_rq_review_question_type": "review_workflow_automation",
        "stage1_directional_relation": "review_about_ai_external_domain",
        "stage1_directional_confidence": 0.86,
        "stage1_directional_is_review_about_ai_external_domain": True,
        "stage1_external_domain_evidence_quote": "breast cancer diagnosis in patients",
        "stage1_task_object_type": "unclear",
    }

    adjudication = adjudicate_row(row)

    assert adjudication["final_adjudicated_decision"] == "REJECT"
    assert adjudication["final_adjudication_action"] == "confirm_external_domain"


def test_llm_directional_disk_cache_skips_corrupt_lines(monkeypatch):
    import llm_structured_judge as judge

    scratch = Path(".tmp_model_judge_test")
    scratch.mkdir(exist_ok=True)
    cache_path = scratch / "llm_directional_cache.jsonl"
    key = judge._cache_key("rq", "title", "abstract", "model")
    try:
        cache_path.write_text(
            "not-json\n"
            + '{"cache_key":"%s","value":{"directional_judge_source":"llm","directional_relation":"ai_tool_for_review_workflow","llm_directional_judge_used":true}}\n'
            % key,
            encoding="utf-8",
        )
        monkeypatch.setattr(judge, "CACHE_PATH", str(cache_path))
        judge._DIRECTIONAL_CACHE.clear()
        monkeypatch.setattr(judge, "_DISK_CACHE_LOADED", False)

        result = judge.judge_with_llm("title", "abstract", "rq", "model")
        assert result["llm_directional_cache_hit"] is True
        assert result["llm_directional_cache_source"] == "disk"
        assert result["directional_relation"] == "ai_tool_for_review_workflow"
    finally:
        cache_path.unlink(missing_ok=True)
        try:
            scratch.rmdir()
        except OSError:
            pass


def test_csv_validator_reports_adjudication_and_suspicious_counts():
    from validate_screened_csv import build_report

    df = pd.DataFrame([
        {
            "Title": "External keep",
            "Decision": "KEEP",
            "final_external_domain": True,
            "final_relation_confidence": 0.9,
            "final_adjudication_action": "demote_keep_to_maybe",
            "stage1_model_fusion_action": "preserve",
            "stage1_directional_confidence": 0.9,
        },
        {
            "Title": "Workflow reject",
            "Decision": "REJECT",
            "final_workflow_use": True,
            "final_relation_confidence": 0.8,
        },
    ])
    report = build_report(df, rq_type="review_workflow_automation")
    assert "keep_with_validated_external_domain=1" in report
    assert "reject_with_validated_workflow=1" in report
    assert "final_adjudication_action_counts" in report


def test_current_mode_masks_fast_only_flags(monkeypatch):
    monkeypatch.setenv("SCREENING_PIPELINE_MODE", "current")
    monkeypatch.setenv("ENABLE_AGGRESSIVE_LLM_GATING", "true")
    monkeypatch.setenv("ENABLE_BATCH_LLM_JUDGE", "true")
    monkeypatch.setenv("ENABLE_SEMANTIC_FRAME_CACHE", "true")
    monkeypatch.delenv("ENABLE_CURRENT_MODE_CACHE", raising=False)
    cfg = get_model_judge_config()
    assert cfg["enable_aggressive_llm_gating"] is False
    assert cfg["enable_batch_llm_judge"] is False
    assert cfg["enable_semantic_frame_cache"] is False


def test_current_mode_cache_requires_separate_opt_in(monkeypatch):
    import semantic_frame

    monkeypatch.setenv("SCREENING_PIPELINE_MODE", "current")
    monkeypatch.setenv("ENABLE_SEMANTIC_FRAME_CACHE", "true")
    monkeypatch.setenv("ENABLE_CURRENT_MODE_CACHE", "false")
    assert semantic_frame._semantic_frame_cache_enabled() is False
    monkeypatch.setenv("ENABLE_CURRENT_MODE_CACHE", "true")
    assert semantic_frame._semantic_frame_cache_enabled() is True


def test_fast_mode_requires_explicit_aggressive_gating(monkeypatch):
    monkeypatch.setenv("SCREENING_PIPELINE_MODE", "two_pass_fast")
    monkeypatch.setenv("ENABLE_AGGRESSIVE_LLM_GATING", "false")
    monkeypatch.setenv("ENABLE_LLM_JUDGE", "false")
    _, diagnostics = apply_model_score_fusion(
        rq_frame=_rq(), paper_frame=_paper(),
        deterministic_result={"decision": "KEEP", "relation_match": True, "paper_observed_relation": "tool_used_for_workflow"},
        research_question=_rq()["rq_text"], mode="balanced",
    )
    assert diagnostics["llm_route"] == "legacy_current_mode"


def test_semantic_frame_cache_versioning_and_duplicate_last_wins(monkeypatch):
    import semantic_frame

    monkeypatch.setenv("SCREENING_PIPELINE_MODE", "two_pass_fast")
    monkeypatch.setenv("ENABLE_SEMANTIC_FRAME_CACHE", "true")
    scratch = Path(".codex_test_outputs")
    scratch.mkdir(exist_ok=True)
    path = scratch / "semantic_frame_version_test.jsonl"
    key = "same-key"
    first = semantic_frame._normalize_frame({
        "primary_subject": "first", "intervention_or_method": "LLM",
        "source_title": "first", "source_abstract": "abstract",
    })
    second = semantic_frame._normalize_frame({
        "primary_subject": "second", "intervention_or_method": "LLM",
        "source_title": "second", "source_abstract": "abstract",
    })
    path.write_text(
        "not-json\n"
        + json.dumps({"schema_version": 1, "cache_key": "old", "value": first}) + "\n"
        + json.dumps({"schema_version": semantic_frame.SEMANTIC_FRAME_CACHE_SCHEMA_VERSION, "cache_key": key, "value": first}) + "\n"
        + json.dumps({"schema_version": semantic_frame.SEMANTIC_FRAME_CACHE_SCHEMA_VERSION, "cache_key": key, "value": second}) + "\n",
        encoding="utf-8",
    )
    try:
        semantic_frame.configure_semantic_frame_cache(str(path))
        info = semantic_frame.initialize_semantic_frame_cache()
        assert info["entries"] == 1
        assert info["semantic_frame_cache_invalid"] == 2
        assert semantic_frame._FRAME_CACHE[("disk", key)] == second
    finally:
        path.unlink(missing_ok=True)
