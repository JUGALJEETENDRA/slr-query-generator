import re


METHOD_GROUPS = {
    "blockchain_distributed_ledger": {
        "blockchain",
        "blockchain technology",
        "distributed ledger",
        "distributed ledger technology",
        "smart contract",
        "smart contracts",
        "ethereum",
        "hyperledger",
        "ipfs",
    },
    "machine_learning": {
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "random forest",
        "support vector machine",
        "svm",
        "xgboost",
        "logistic regression",
        "neural network",
        "cnn",
        "lstm",
        "transformer",
    },
    "large_language_models": {
        "large language model",
        "large language models",
        "llm",
        "llms",
        "gpt",
        "chatgpt",
        "llama",
        "qwen",
        "gemini",
        "bert",
    },
    "learning_analytics": {
        "learning analytics",
        "educational data mining",
        "student analytics",
    },
    "natural_language_processing": {
        "natural language processing",
        "nlp",
        "transformer",
        "transformer model",
        "transformer models",
        "bert",
        "information extraction model",
    },
}

CONTEXT_GROUPS = {
    "supply_chain_management": {
        "supply chain",
        "supply chain management",
        "logistics",
        "agri food",
        "agri-food",
        "agriculture supply chain",
        "food supply chain",
        "pharmaceutical supply chain",
        "manufacturing supply chain",
        "green supply chain",
        "cold chain",
        "product provenance",
    },
    "cardiovascular_healthcare": {
        "heart disease",
        "cardiovascular disease",
        "cardiac",
        "cardiology",
        "clinical",
        "medical",
        "healthcare",
        "patient",
    },
    "systematic_literature_review": {
        "systematic literature review",
        "systematic literature reviews",
        "systematic review",
        "systematic reviews",
        "literature review",
        "evidence review",
        "study selection",
        "title abstract screening",
    },
    "cybersecurity": {
        "cybersecurity",
        "network security",
        "intrusion detection",
        "malware",
        "access control",
        "threat detection",
    },
    "internet_of_things": {
        "internet of things",
        "iot",
        "iot network",
        "iot networks",
        "sensor network",
        "smart device",
    },
    "education": {
        "education",
        "educational",
        "student",
        "students",
        "academic",
        "school",
        "university",
        "learning environment",
    },
    "software_engineering": {
        "software engineering",
        "software",
        "source code",
        "code repository",
        "program",
        "bug report",
    },
    "legal_documents": {
        "legal document",
        "legal documents",
        "legal text",
        "legal texts",
        "law",
        "legislation",
        "contract",
        "case law",
    },
}

TASK_GROUPS = {
    "supply_chain_assurance": {
        "transparency",
        "traceability",
        "trust",
        "security",
        "provenance",
        "accountability",
        "data integrity",
        "integrity",
        "anti counterfeit",
        "anti-counterfeit",
        "visibility",
        "tracking",
        "tamper resistance",
        "immutability",
        "authentication",
    },
    "predictive_assessment": {
        "prediction",
        "diagnosis",
        "detection",
        "classification",
        "prognosis",
        "risk estimation",
        "risk assessment",
        "risk stratification",
    },
    "review_workflow": {
        "screening",
        "study selection",
        "search strategy generation",
        "deduplication",
        "data extraction",
        "pico extraction",
        "risk of bias",
        "evidence synthesis",
        "review assistance",
    },
    "intrusion_threat_detection": {
        "intrusion detection",
        "threat classification",
        "threat detection",
        "attack detection",
        "anomaly detection",
        "malware classification",
    },
    "student_outcome_prediction": {
        "student dropout",
        "dropout prediction",
        "academic performance",
        "performance prediction",
        "student success",
        "student retention",
    },
    "software_quality_prediction": {
        "software defect prediction",
        "defect prediction",
        "bug detection",
        "fault prediction",
        "fault detection",
        "software quality",
    },
    "information_extraction": {
        "information extraction",
        "entity extraction",
        "named entity recognition",
        "relation extraction",
        "document extraction",
    },
}

EVIDENCE_TERMS = {
    "application",
    "architecture",
    "benchmark",
    "case study",
    "empirical",
    "evaluation",
    "framework",
    "model",
    "platform",
    "proposal",
    "prototype",
    "review",
    "survey",
    "system",
}

NEGATIVE_CONTEXTS = {
    "blockchain_distributed_ledger": {
        "bitcoin price",
        "cryptocurrency trading",
        "token speculation",
        "mining",
        "proof of work",
        "consensus protocol",
        "consensus algorithm",
        "blockchain scalability",
    },
    "large_language_models": {
        "general llm survey",
        "language model survey",
    },
}

QUESTION_TEMPLATES = {
    "technology_in_domain_review": ("method", "task_or_outcome", "context"),
    "method_for_task_in_domain_review": ("method", "task_or_outcome", "context"),
    "method_comparison_review": ("method", "task_or_outcome", "context"),
    "review_workflow_automation": (
        "method_tool",
        "review_workflow_process",
        "review_context",
        "relation",
    ),
    "security_risk_review": ("method", "task_or_outcome", "context"),
    "application_mapping_review": ("method", "task_or_outcome", "context"),
    "mapping_scoping_review": ("method", "task_or_outcome", "context"),
    "domain_literature_review": ("method", "task_or_outcome", "context"),
}


def normalize(text):
    text = str(text or "").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(text, phrase):
    normalized_text = normalize(text)
    normalized_phrase = normalize(phrase)
    return bool(
        normalized_phrase
        and re.search(rf"(^|\s){re.escape(normalized_phrase)}(\s|$)", normalized_text)
    )


def find_terms(text, terms):
    return sorted({term for term in terms if contains_phrase(text, term)})


def join_terms(terms):
    return "; ".join(dict.fromkeys(str(term).strip() for term in terms if str(term).strip()))


def split_terms(value):
    if isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = re.split(r"[;,|]\s*|\s+and\s+", str(value or ""))
    return [term.strip() for term in raw if str(term).strip()]


def method_families_for_text(text):
    return {
        family
        for family, terms in METHOD_GROUPS.items()
        if find_terms(text, terms)
    }


def context_families_for_text(text):
    return {
        family
        for family, terms in CONTEXT_GROUPS.items()
        if find_terms(text, terms)
    }


def task_families_for_text(text):
    return {
        family
        for family, terms in TASK_GROUPS.items()
        if find_terms(text, terms)
    }


def analyze_research_question(research_question):
    text = str(research_question or "")
    method_families = method_families_for_text(text)
    context_families = context_families_for_text(text)
    task_families = task_families_for_text(text)

    method_terms = []
    for family in method_families:
        method_terms.extend(find_terms(text, METHOD_GROUPS[family]))
    context_terms = []
    for family in context_families:
        context_terms.extend(find_terms(text, CONTEXT_GROUPS[family]))
    task_terms = []
    for family in task_families:
        task_terms.extend(find_terms(text, TASK_GROUPS[family]))

    normalized = normalize(text)
    if "large_language_models" in method_families and "systematic_literature_review" in context_families:
        question_type = "review_workflow_automation"
        core_domain = "systematic literature reviews"
        task_families.add("review_workflow")
    elif any(term in normalized for term in ("risk", "threat", "intrusion", "attack")):
        question_type = "security_risk_review"
        core_domain = join_terms(context_terms)
    elif any(term in normalized for term in ("mapping review", "scoping review", "map the literature")):
        question_type = "mapping_scoping_review"
        core_domain = join_terms(context_terms)
    elif method_families and context_families and any(
        marker in normalized
        for marker in ("technology", "technologies", "application of", "applications of")
    ):
        question_type = "technology_in_domain_review"
        core_domain = join_terms(context_terms)
    elif method_families and task_families and context_families:
        question_type = "method_for_task_in_domain_review"
        core_domain = join_terms(context_terms)
    elif method_families and context_families:
        question_type = "technology_in_domain_review"
        core_domain = join_terms(context_terms)
    elif method_families:
        question_type = "method_comparison_review"
        core_domain = join_terms(context_terms)
    else:
        question_type = "domain_literature_review"
        core_domain = join_terms(context_terms)

    if "machine_learning" in method_families and "cardiovascular_healthcare" in context_families:
        core_domain = "cardiovascular healthcare"

    method_family = join_terms(sorted(method_families))
    method = join_terms(method_terms)
    context = join_terms(context_terms) or core_domain
    outcomes = join_terms(task_terms)

    domain_synonyms = []
    method_synonyms = []
    task_synonyms = []
    for family in context_families:
        domain_synonyms.extend(sorted(CONTEXT_GROUPS[family]))
    for family in method_families:
        method_synonyms.extend(sorted(METHOD_GROUPS[family]))
    for family in task_families:
        task_synonyms.extend(sorted(TASK_GROUPS[family]))

    exclusion = []
    for family in method_families:
        exclusion.extend(sorted(NEGATIVE_CONTEXTS.get(family, set())))

    if question_type == "review_workflow_automation":
        from review_workflow_ontology import review_question_analysis
        review_analysis = review_question_analysis()
        outcomes = review_analysis["target_tasks_or_outcomes"]
        task_synonyms.extend(split_terms(review_analysis["task_outcome_synonyms"]))
        exclusion.extend(split_terms(review_analysis["exclusion_concepts"]))

    required_dimensions = QUESTION_TEMPLATES.get(
        question_type,
        QUESTION_TEMPLATES["domain_literature_review"],
    )
    minimum_inclusion_rule = (
        "all_required_dimensions_and_review_role"
        if question_type == "review_workflow_automation"
        else "three_dimensions_keep_two_dimensions_maybe"
    )
    desired_relations = {
        "review_workflow_automation": "tool_used_for_workflow",
        "method_comparison_review": "method_compared_for_task",
        "method_for_task_in_domain_review": "method_used_for_task",
        "technology_in_domain_review": "technology_used_for_outcome",
        "application_mapping_review": "applications_mapped_in_domain",
    }

    return {
        "review_question_type": question_type,
        "core_domain": core_domain,
        "application_context": context,
        "method_or_technology": method,
        "method_family": method_family,
        "target_tasks_or_outcomes": outcomes,
        "required_inclusion_concepts": join_terms(method_terms + context_terms),
        "optional_related_concepts": "",
        "exclusion_concepts": join_terms(exclusion),
        "expected_evidence_types": join_terms(sorted(EVIDENCE_TERMS)),
        "domain_synonyms": join_terms(domain_synonyms),
        "method_synonyms": join_terms(method_synonyms),
        "task_outcome_synonyms": join_terms(task_synonyms),
        "context_synonyms": join_terms(domain_synonyms),
        "negative_contexts": join_terms(exclusion),
        "required_dimensions": join_terms(required_dimensions),
        "minimum_inclusion_rule": minimum_inclusion_rule,
        "rq_extraction_suspect": str(
            not method_terms
            or not outcomes
            or not context_terms
            or normalize(method) == normalized
            or normalize(outcomes) == normalized
            or normalize(context) == normalized
        ),
        "rq_desired_relation": desired_relations.get(question_type, "unclear_relation"),
    }


def analyze_paper_text(title, abstract):
    text = f"{title or ''} {abstract or ''}"
    method_families = method_families_for_text(text)
    context_families = context_families_for_text(text)
    task_families = task_families_for_text(text)

    methods = []
    contexts = []
    tasks = []
    exclusion = []
    for family in method_families:
        methods.extend(find_terms(text, METHOD_GROUPS[family]))
        exclusion.extend(find_terms(text, NEGATIVE_CONTEXTS.get(family, set())))
    for family in context_families:
        contexts.extend(find_terms(text, CONTEXT_GROUPS[family]))
    for family in task_families:
        tasks.extend(find_terms(text, TASK_GROUPS[family]))

    evidence = find_terms(text, EVIDENCE_TERMS)
    direct_application_present = bool(methods and contexts and tasks)
    return {
        "main_domain": join_terms(contexts),
        "application_contexts": join_terms(contexts),
        "methods_or_technologies": join_terms(methods),
        "specific_models_or_systems": join_terms(methods),
        "target_tasks_or_outcomes": join_terms(tasks),
        "evidence_type": join_terms(evidence),
        "contribution_type": join_terms(evidence),
        "inclusion_cues": join_terms(methods + contexts + tasks),
        "exclusion_cues": join_terms(exclusion),
        "direct_application_present": str(direct_application_present),
    }


def enrich_rq_analysis_with_profile(analysis, profile):
    enriched = dict(analysis or {})
    for target, source in (
        ("method_synonyms", "corpus_method_terms"),
        ("task_outcome_synonyms", "corpus_task_terms"),
        ("context_synonyms", "corpus_context_terms"),
        ("domain_synonyms", "corpus_context_terms"),
    ):
        terms = split_terms(enriched.get(target, "")) + split_terms((profile or {}).get(source, ""))
        enriched[target] = join_terms(terms)
    profile_terms = split_terms((profile or {}).get("corpus_profile_terms", ""))
    enriched["optional_related_concepts"] = join_terms(
        split_terms(enriched.get("optional_related_concepts", "")) + profile_terms
    )
    return enriched
