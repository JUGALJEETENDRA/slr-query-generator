from __future__ import annotations

from functools import lru_cache

from domain_vocabulary import find_terms, normalize
from model_registry import cache_key, configured_model_names, get_cross_encoder


POSITIVE_QUERY = (
    "AI or large language models used to automate systematic literature review workflows, "
    "including title abstract screening, citation screening, study selection, data extraction "
    "from studies, literature search, evidence synthesis, or review assistant tools."
)

NEGATIVE_QUERY = (
    "Systematic review about artificial intelligence applications in an external domain, "
    "where AI is the subject being reviewed, such as diagnosis, medicine, fintech, education, "
    "software refactoring, cybersecurity, drug discovery, or other domain tasks."
)

POSITIVE_TERMS = {
    "review automation", "systematic review automation", "title abstract screening",
    "citation screening", "study selection", "data extraction", "risk of bias",
    "evidence synthesis", "literature search", "review workflow", "review pipeline",
    "research assistant for literature review", "ai-assisted systematic review",
    "llm-assisted methodology", "automated systematic review",
}

NEGATIVE_TERMS = {
    "diagnosis", "classification", "prediction", "education", "healthcare",
    "medicine", "finance", "fintech", "cybersecurity", "software development",
    "drug discovery", "cancer", "telemedicine", "manufacturing", "speech",
}


def judge_reranker(rq_text: str, title: str, abstract: str, mode: str = "balanced") -> dict:
    names = configured_model_names()
    model_name = names["reranker"]
    doc = f"{title}\n{abstract}".strip()
    scores = _score_cached(model_name, rq_text, doc, mode)
    relevance = scores["positive"]
    negative = scores["negative"]
    margin = relevance - negative
    if margin >= 0.25:
        hint = "positive"
    elif margin <= -0.20:
        hint = "negative"
    else:
        hint = "uncertain"
    return {
        "model_reranker_relevance_score": round(relevance, 4),
        "model_reranker_negative_score": round(negative, 4),
        "model_reranker_positive_raw_score": round(float(scores.get("positive_raw", relevance)), 4),
        "model_reranker_negative_raw_score": round(float(scores.get("negative_raw", negative)), 4),
        "model_reranker_margin": round(margin, 4),
        "model_reranker_decision_hint": hint,
    }


@lru_cache(maxsize=4096)
def _score_cached(model_name: str, rq_text: str, doc: str, mode: str) -> dict:
    model = get_cross_encoder(model_name) if mode in {"balanced", "full"} else None
    if model is not None:
        try:
            raw = model.predict([(POSITIVE_QUERY, doc), (NEGATIVE_QUERY, doc)])
            positive_raw = float(raw[0])
            negative_raw = float(raw[1])
            positive, negative = _paired_probabilities(positive_raw, negative_raw)
            return {
                "positive": positive,
                "negative": negative,
                "positive_raw": positive_raw,
                "negative_raw": negative_raw,
            }
        except Exception:
            pass
    return _lexical_scores(doc)


def _lexical_scores(doc: str) -> dict:
    positive_hits = find_terms(doc, POSITIVE_TERMS)
    negative_hits = find_terms(doc, NEGATIVE_TERMS)
    review_context = 1.0 if find_terms(doc, {"systematic review", "systematic literature review", "literature review"}) else 0.0
    ai_context = 1.0 if find_terms(doc, {"artificial intelligence", "large language model", "llm", "machine learning"}) else 0.0
    positive = min(1.0, (0.18 * len(positive_hits)) + (0.20 * review_context) + (0.15 * ai_context))
    negative = min(1.0, (0.16 * len(negative_hits)) + (0.20 * review_context if negative_hits else 0.0))
    return {
        "positive": positive,
        "negative": negative,
        "positive_raw": positive,
        "negative_raw": negative,
    }


def _squash(value: float) -> float:
    if 0.0 <= value <= 1.0:
        return value
    return 1.0 / (1.0 + pow(2.718281828, -value))


def _paired_probabilities(positive_raw: float, negative_raw: float) -> tuple[float, float]:
    high = max(positive_raw, negative_raw)
    positive_exp = pow(2.718281828, positive_raw - high)
    negative_exp = pow(2.718281828, negative_raw - high)
    total = positive_exp + negative_exp
    if total <= 0:
        return 0.5, 0.5
    return positive_exp / total, negative_exp / total
