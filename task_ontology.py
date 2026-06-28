import re


RESEARCH_TASK_ONTOLOGY = {
    "prediction": [
        "prediction",
        "predict",
        "predicting",
        "predictive modeling",
        "risk prediction",
        "outcome prediction",
        "prognosis",
        "prognostic modeling",
    ],
    "diagnosis": [
        "diagnosis",
        "diagnose",
        "diagnosing",
        "diagnostic",
        "diagnostic assessment",
        "differential diagnosis",
    ],
    "classification": [
        "classification",
        "classify",
        "classifying",
        "categorization",
        "categorisation",
        "labeling",
        "labelling",
        "class assignment",
    ],
    "detection": [
        "detection",
        "detect",
        "detecting",
        "identification",
        "recognition",
        "anomaly detection",
        "event detection",
        "object detection",
    ],
    "segmentation": [
        "segmentation",
        "segment",
        "segmenting",
        "semantic segmentation",
        "instance segmentation",
        "partitioning",
    ],
    "retrieval": [
        "retrieval",
        "retrieve",
        "retrieving",
        "information retrieval",
        "document retrieval",
        "search",
        "lookup",
    ],
    "generation": [
        "generation",
        "generate",
        "generating",
        "text generation",
        "content generation",
        "response generation",
        "synthesis generation",
    ],
    "summarization": [
        "summarization",
        "summarisation",
        "summarize",
        "summarise",
        "summarizing",
        "summarising",
        "summary generation",
        "abstractive summarization",
        "extractive summarization",
    ],
    "screening": [
        "screening",
        "screen",
        "screening for eligibility",
        "study screening",
        "title screening",
        "abstract screening",
        "eligibility screening",
    ],
    "ranking": [
        "ranking",
        "rank",
        "ranking items",
        "prioritization",
        "prioritisation",
        "ordering",
        "scoring for rank",
    ],
    "recommendation": [
        "recommendation",
        "recommend",
        "recommending",
        "recommender",
        "recommendation system",
        "suggestion",
    ],
    "forecasting": [
        "forecasting",
        "forecast",
        "forecasting outcomes",
        "time series forecasting",
        "trend forecasting",
    ],
    "regression": [
        "regression",
        "regress",
        "regression modeling",
        "continuous outcome estimation",
        "value estimation",
    ],
    "clustering": [
        "clustering",
        "cluster",
        "clustering items",
        "cluster analysis",
        "grouping",
        "unsupervised grouping",
    ],
    "optimization": [
        "optimization",
        "optimisation",
        "optimize",
        "optimise",
        "optimizing",
        "optimising",
        "parameter optimization",
        "resource optimization",
    ],
    "planning": [
        "planning",
        "plan",
        "planning actions",
        "plan generation",
        "scheduling",
        "decision planning",
    ],
}


def _normalize(text):
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _phrase_matches(normalized_text, phrase):
    normalized_phrase = _normalize(phrase)
    if not normalized_phrase:
        return False
    return bool(
        re.search(
            rf"(^|\s){re.escape(normalized_phrase)}(\s|$)",
            normalized_text,
        )
    )


def canonicalize_task(task_string):
    normalized = _normalize(task_string)
    if not normalized:
        return ""

    matches = []
    for canonical_task, phrases in RESEARCH_TASK_ONTOLOGY.items():
        for phrase in phrases:
            if _phrase_matches(normalized, phrase):
                matches.append((len(_normalize(phrase)), canonical_task))

    if not matches:
        return str(task_string or "").strip()

    matches.sort(reverse=True)
    longest_length = matches[0][0]
    strongest = {
        canonical_task
        for length, canonical_task in matches
        if length == longest_length
    }

    if len(strongest) == 1:
        return strongest.pop()

    return str(task_string or "").strip()
