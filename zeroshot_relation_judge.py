from __future__ import annotations

from functools import lru_cache

from domain_vocabulary import find_terms, join_terms
from model_registry import configured_model_names, get_transformers_pipeline


LABELS = [
    "AI tool for systematic review workflow",
    "systematic review about AI in external domain",
    "unrelated or external-domain AI task",
]


def judge_zero_shot_relation(title: str, abstract: str, mode: str = "balanced") -> dict:
    text = f"{title}\n{abstract}".strip()
    model_name = configured_model_names()["zero_shot"]
    scored = _score_cached(model_name, text, mode)
    top_label, top_score = max(scored.items(), key=lambda item: item[1])
    tool_score = scored["AI tool for systematic review workflow"]
    subject_score = max(
        scored["systematic review about AI in external domain"],
        scored["unrelated or external-domain AI task"],
    )
    top_labels = "; ".join(f"{label}:{score:.3f}" for label, score in sorted(scored.items(), key=lambda item: item[1], reverse=True)[:3])
    return {
        "zeroshot_relation_label": top_label,
        "zeroshot_relation_score": round(top_score, 4),
        "zeroshot_ai_tool_for_review_score": round(tool_score, 4),
        "zeroshot_subject_review_score": round(subject_score, 4),
        "zeroshot_top_labels": top_labels,
    }


@lru_cache(maxsize=4096)
def _score_cached(model_name: str, text: str, mode: str) -> dict:
    classifier = get_transformers_pipeline("zero-shot-classification", model_name) if mode in {"balanced", "full"} else None
    if classifier is not None:
        try:
            result = classifier(text, candidate_labels=LABELS, multi_label=False)
            return {label: float(score) for label, score in zip(result["labels"], result["scores"])}
        except Exception:
            pass
    return _lexical_scores(text)


def _lexical_scores(text: str) -> dict:
    scores = {label: 0.02 for label in LABELS}
    if find_terms(text, {"title abstract screening", "abstract screening", "citation screening"}):
        scores["AI tool for systematic review workflow"] = 0.82
    if find_terms(text, {"study selection", "selection of studies"}):
        scores["AI tool for systematic review workflow"] = max(scores["AI tool for systematic review workflow"], 0.78)
    if find_terms(text, {"data extraction", "extract data from studies", "included studies"}):
        scores["AI tool for systematic review workflow"] = max(scores["AI tool for systematic review workflow"], 0.78)
    if find_terms(text, {"evidence synthesis", "synthesize study findings"}):
        scores["AI tool for systematic review workflow"] = max(scores["AI tool for systematic review workflow"], 0.76)
    if find_terms(text, {"literature search", "search strategy", "query generation"}):
        scores["AI tool for systematic review workflow"] = max(scores["AI tool for systematic review workflow"], 0.74)
    workflow = scores["AI tool for systematic review workflow"]
    if find_terms(text, {"review automation", "ai-assisted systematic review", "automated systematic review", "review workflow"}):
        workflow = max(workflow, 0.86)
    scores["AI tool for systematic review workflow"] = workflow
    if find_terms(text, {"applications in", "diagnosis", "classification", "prediction", "education", "healthcare", "finance", "cybersecurity"}):
        scores["systematic review about AI in external domain"] = 0.78
        scores["unrelated or external-domain AI task"] = 0.70
    return scores
