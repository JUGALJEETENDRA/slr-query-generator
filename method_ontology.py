import re


METHOD_FAMILY_ALIASES = {
    "artificial_intelligence": [
        "artificial intelligence",
        "ai",
    ],
    "machine_learning": [
        "machine learning",
        "ml",
        "random forest",
        "support vector machine",
        "support vector classifier",
        "svm",
        "knn",
        "k nearest neighbor",
        "naive bayes",
        "decision tree",
        "xgboost",
        "lightgbm",
        "catboost",
        "logistic regression",
        "gradient boosting",
    ],
    "deep_learning": [
        "deep learning",
        "neural network",
        "convolutional neural network",
        "cnn",
        "recurrent neural network",
        "rnn",
        "long short term memory",
        "lstm",
        "autoencoder",
        "deep belief network",
    ],
    "transformer_models": [
        "transformer",
        "swin transformer",
        "tabtransformer",
        "vision transformer",
        "bert",
        "roberta",
        "gpt",
        "t5",
    ],
    "large_language_models": [
        "large language model",
        "large language models",
        "llm",
        "llms",
        "llama",
        "gemini",
        "qwen",
        "mistral",
        "chatgpt",
    ],
    "blockchain_distributed_ledger": [
        "blockchain",
        "blockchain technology",
        "distributed ledger",
        "distributed ledger technology",
        "smart contract",
        "smart contracts",
        "ethereum",
        "hyperledger",
        "ipfs",
    ],
    "natural_language_processing": [
        "natural language processing",
        "nlp",
        "text mining",
        "language model",
    ],
    "computer_vision": [
        "computer vision",
        "image recognition",
        "image classification",
        "visual recognition",
    ],
    "explainable_ai": [
        "explainable ai",
        "explainable artificial intelligence",
        "xai",
        "shap",
        "kernel shap",
        "lime",
        "explainability",
        "feature attribution",
    ],
    "optimization_methods": [
        "optimization",
        "optimisation",
        "particle swarm optimization",
        "particle swarm optimisation",
        "pso",
        "genetic algorithm",
        "metaheuristic",
        "whale optimization",
        "grey wolf optimization",
    ],
    "statistical_models": [
        "statistical model",
        "linear regression",
        "survival analysis",
        "cox regression",
    ],
}

METHOD_FAMILY_PARENTS = {
    "machine_learning": {"artificial_intelligence"},
    "deep_learning": {"machine_learning", "artificial_intelligence"},
    "transformer_models": {
        "deep_learning",
        "machine_learning",
        "artificial_intelligence",
    },
    "large_language_models": {
        "transformer_models",
        "deep_learning",
        "machine_learning",
        "natural_language_processing",
        "artificial_intelligence",
    },
    "natural_language_processing": {"artificial_intelligence"},
    "computer_vision": {"artificial_intelligence"},
    "explainable_ai": {"artificial_intelligence"},
}

BROAD_METHOD_FAMILIES = {
    "artificial_intelligence",
    "blockchain_distributed_ledger",
    "machine_learning",
    "deep_learning",
    "large_language_models",
    "natural_language_processing",
    "computer_vision",
}

AUXILIARY_METHOD_FAMILIES = {"explainable_ai", "optimization_methods"}

COMPATIBLE_AUXILIARY_TASK_MARKERS = {
    "classification",
    "diagnosis",
    "detection",
    "forecast",
    "prediction",
    "prognosis",
    "risk",
}


def _normalize(text):
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_phrase(text, phrase):
    normalized_phrase = _normalize(phrase)
    return bool(
        normalized_phrase
        and re.search(rf"(^|\s){re.escape(normalized_phrase)}(\s|$)", text)
    )


def method_families(text):
    normalized = _normalize(text)
    if not normalized:
        return set()
    return {
        family
        for family, aliases in METHOD_FAMILY_ALIASES.items()
        if any(_contains_phrase(normalized, alias) for alias in aliases)
    }


def _expanded_families(families):
    expanded = set(families)
    pending = list(families)
    while pending:
        family = pending.pop()
        for parent in METHOD_FAMILY_PARENTS.get(family, set()):
            if parent not in expanded:
                expanded.add(parent)
                pending.append(parent)
    return expanded


def compare_method_families(rq_method, paper_method, paper_task=""):
    rq_families = method_families(rq_method)
    paper_families = method_families(paper_method)
    paper_expanded = _expanded_families(paper_families)
    direct_matches = rq_families & paper_expanded
    broad_query = bool(rq_families & BROAD_METHOD_FAMILIES)
    auxiliary_matches = paper_families & AUXILIARY_METHOD_FAMILIES
    normalized_task = _normalize(paper_task)
    auxiliary_task_compatible = bool(
        auxiliary_matches
        and broad_query
        and any(
            _contains_phrase(normalized_task, marker)
            for marker in COMPATIBLE_AUXILIARY_TASK_MARKERS
        )
    )
    compatible = bool(direct_matches or auxiliary_task_compatible)

    if direct_matches:
        matched = direct_matches
        reason = (
            "The paper method is a specific instance of the broader method "
            "family requested by the RQ."
        )
        confidence = 0.90
    elif auxiliary_task_compatible:
        matched = auxiliary_matches
        reason = (
            "The paper uses an explainability or optimization method with a "
            "compatible predictive, diagnostic, detection, or classification task."
        )
        confidence = 0.75
    elif rq_families and paper_families:
        matched = set()
        reason = "Recognized method families do not have a directional hierarchy match."
        confidence = 0.0
    else:
        matched = set()
        reason = "One or both method descriptions could not be mapped to a known family."
        confidence = 0.0

    return {
        "method_family_left": "|".join(sorted(rq_families)),
        "method_family_right": "|".join(sorted(paper_families)),
        "method_family_compatible": compatible,
        "method_family_match": "|".join(sorted(matched)),
        "method_family_reason": reason,
        "method_family_confidence": confidence,
        "broad_method_query_detected": broad_query,
    }
