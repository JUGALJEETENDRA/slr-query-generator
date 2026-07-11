import re


RESEARCH_TASK_ONTOLOGY = {
    "prediction": [
        "prediction",
        "predict",
        "predicting",
        "predictive modeling",
        "risk assessment",
        "risk identification",
        "risk prediction",
        "risk estimation",
        "risk stratification",
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

CLINICAL_CONTEXT_MARKERS = {
    "biomedical",
    "cardiology",
    "clinical",
    "diagnostic",
    "disease",
    "health",
    "healthcare",
    "hospital",
    "medical",
    "medicine",
    "patient",
    "prognosis",
    "risk",
}

REVIEW_SCREENING_MARKERS = {
    "abstract screening",
    "citation screening",
    "eligibility screening",
    "paper screening",
    "review screening",
    "screening for eligibility",
    "study screening",
    "study selection",
    "systematic review",
    "title abstract screening",
    "title screening",
}

CLINICAL_SCREENING_MARKERS = {
    "clinical screening",
    "disease identification",
    "disease screening",
    "early detection",
    "early screening",
    "health screening",
    "patient screening",
    "predictive screening",
    "risk assessment",
    "risk identification",
    "risk prediction",
    "risk screening",
}

SECURITY_CONTEXT_MARKERS = {
    "access control",
    "cybersecurity",
    "fraud",
    "malware",
    "security",
    "threat",
    "vulnerability",
}

# Families describe related research operations. Domain and method compatibility
# are evaluated separately by the semantic comparator.
TASK_FAMILIES = {
    "predictive_assessment": {
        "tasks": {
            "prediction",
            "diagnosis",
            "detection",
            "classification",
            "forecasting",
            "regression",
        },
    },
    "information_access": {
        "tasks": {
            "retrieval",
            "ranking",
            "recommendation",
        },
    },
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


def _has_any_phrase(normalized_text, phrases):
    return any(_phrase_matches(normalized_text, phrase) for phrase in phrases)


def _screening_sense(task_text, context_text=""):
    normalized_task = _normalize(task_text)
    normalized_context = _normalize(context_text)
    combined = " ".join(part for part in [normalized_task, normalized_context] if part)

    if _has_any_phrase(combined, REVIEW_SCREENING_MARKERS):
        return "review_workflow"
    if _has_any_phrase(combined, SECURITY_CONTEXT_MARKERS):
        return "security"
    if (
        _has_any_phrase(combined, CLINICAL_SCREENING_MARKERS)
        or _has_any_phrase(normalized_context, CLINICAL_CONTEXT_MARKERS)
    ):
        return "clinical"
    return "generic"


def canonicalize_task(task_string):
    normalized = _normalize(task_string)
    if not normalized:
        return ""

    matches_by_task = {}
    for canonical_task, phrases in RESEARCH_TASK_ONTOLOGY.items():
        for phrase in phrases:
            if _phrase_matches(normalized, phrase):
                matches_by_task[canonical_task] = max(
                    matches_by_task.get(canonical_task, 0),
                    len(_normalize(phrase)),
                )

    if not matches_by_task:
        return str(task_string or "").strip()

    if len(matches_by_task) == 1:
        return next(iter(matches_by_task))

    return "|".join(sorted(matches_by_task))


def compatible_task_families(left_task, right_task, context_text=""):
    left_tasks = {item for item in canonicalize_task(left_task).split("|") if item}
    right_tasks = {item for item in canonicalize_task(right_task).split("|") if item}
    if not left_tasks or not right_tasks or left_tasks & right_tasks:
        return set()

    compatible = set()
    if "screening" in left_tasks or "screening" in right_tasks:
        other_tasks = (left_tasks | right_tasks) - {"screening"}
        sense = _screening_sense(f"{left_task} {right_task}", context_text)
        if (
            sense == "clinical"
            and other_tasks & TASK_FAMILIES["predictive_assessment"]["tasks"]
        ):
            compatible.add("predictive_assessment")
        elif sense == "review_workflow":
            return compatible
        elif sense == "security":
            return compatible
        else:
            return compatible

    for family_name, family in TASK_FAMILIES.items():
        if left_tasks & family["tasks"] and right_tasks & family["tasks"]:
            compatible.add(family_name)
    return compatible
