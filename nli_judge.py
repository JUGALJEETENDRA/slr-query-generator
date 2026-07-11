from __future__ import annotations

from functools import lru_cache

from domain_vocabulary import find_terms
from model_registry import configured_model_names, get_transformers_pipeline


POSITIVE_HYPOTHESES = [
    "This paper uses artificial intelligence or large language models to automate or support the systematic literature review process.",
    "This paper evaluates AI or machine learning tools for screening, study selection, data extraction, literature search, or evidence synthesis in systematic reviews.",
    "This paper proposes or studies an AI-assisted systematic review workflow.",
]

NEGATIVE_HYPOTHESES = [
    "This paper is a systematic review about artificial intelligence applications in an external domain.",
    "This paper reviews AI methods for a domain task, not AI tools for conducting systematic reviews.",
    "This paper is only a literature review where AI is the subject, not the review assistant.",
]


def judge_nli(title: str, abstract: str, mode: str = "balanced") -> dict:
    text = f"{title}\n{abstract}".strip()
    model_name = configured_model_names()["nli"]
    scores = _score_cached(model_name, text, mode)
    pos = scores["positive"]
    neg = scores["negative"]
    margin = pos - neg
    hint = "positive" if margin >= 0.25 else "negative" if margin <= -0.20 else "uncertain"
    return {
        "nli_positive_entailment_score": round(pos, 4),
        "nli_negative_entailment_score": round(neg, 4),
        "nli_contradiction_score": round(max(0.0, neg - pos), 4),
        "nli_margin": round(margin, 4),
        "nli_decision_hint": hint,
        "nli_top_positive_hypothesis": scores["top_positive"],
        "nli_top_negative_hypothesis": scores["top_negative"],
        "nli_positive_hypothesis_scores": scores.get("positive_scores", ""),
        "nli_negative_hypothesis_scores": scores.get("negative_scores", ""),
    }


@lru_cache(maxsize=4096)
def _score_cached(model_name: str, text: str, mode: str) -> dict:
    classifier = get_transformers_pipeline("zero-shot-classification", model_name) if mode in {"balanced", "full"} else None
    if classifier is not None:
        try:
            labels = POSITIVE_HYPOTHESES + NEGATIVE_HYPOTHESES
            result = classifier(
                text,
                candidate_labels=labels,
                multi_label=False,
                hypothesis_template="{}",
            )
            by_label = dict(zip(result["labels"], result["scores"]))
            positive_scores = {label: float(by_label.get(label, 0.0)) for label in POSITIVE_HYPOTHESES}
            negative_scores = {label: float(by_label.get(label, 0.0)) for label in NEGATIVE_HYPOTHESES}
            top_positive = max(positive_scores, key=positive_scores.get)
            top_negative = max(negative_scores, key=negative_scores.get)
            return {
                "positive": positive_scores[top_positive],
                "negative": negative_scores[top_negative],
                "top_positive": top_positive,
                "top_negative": top_negative,
                "positive_scores": _format_scores(positive_scores),
                "negative_scores": _format_scores(negative_scores),
            }
        except Exception:
            pass
    return _lexical_scores(text)


def _lexical_scores(text: str) -> dict:
    positive_terms = {
        "automate systematic", "review automation", "screening", "study selection",
        "data extraction", "evidence synthesis", "literature search", "review workflow",
        "assistant for literature review",
    }
    negative_terms = {
        "applications in", "diagnosis", "classification", "prediction", "education",
        "healthcare", "finance", "cybersecurity", "drug discovery", "software development",
    }
    pos_hits = find_terms(text, positive_terms)
    neg_hits = find_terms(text, negative_terms)
    pos = min(1.0, 0.15 * len(pos_hits))
    neg = min(1.0, 0.14 * len(neg_hits))
    return {
        "positive": pos,
        "negative": neg,
        "top_positive": POSITIVE_HYPOTHESES[0] if pos_hits else "",
        "top_negative": NEGATIVE_HYPOTHESES[0] if neg_hits else "",
        "positive_scores": "",
        "negative_scores": "",
    }


def _format_scores(scores: dict[str, float]) -> str:
    return "; ".join(f"{label}:{score:.4f}" for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True))
