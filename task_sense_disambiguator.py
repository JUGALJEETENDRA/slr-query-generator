import re

from domain_vocabulary import find_terms, join_terms, normalize


GENERIC_TASK_TERMS = {
    "screening", "classification", "prediction", "diagnosis", "extraction",
    "data extraction", "information extraction", "feature extraction",
    "text extraction", "summarization", "ranking", "analysis",
    "summarize", "summarizing",
    "decision support", "selection", "search",
}

REVIEW_WORKFLOW_OBJECTS = {
    "paper", "papers", "study", "studies", "article", "articles", "citation",
    "citations", "abstract", "abstracts", "title", "titles", "full text",
    "full texts", "publication", "publications", "record", "records",
    "reference", "references", "search result", "search results", "database",
    "databases", "query", "queries", "search string", "search strings",
    "evidence table", "evidence tables", "study characteristics",
    "included studies", "excluded studies", "eligibility criteria",
    "pico", "pico elements", "risk of bias", "quality assessment",
    "evidence synthesis", "review findings", "review stages", "review workflow",
    "review pipeline", "systematic review process", "literature review process",
}

DIRECT_REVIEW_TASKS = {
    "title abstract screening", "title and abstract screening", "citation screening",
    "study selection", "eligibility assessment", "full text screening",
    "literature search", "search strategy generation", "query generation",
    "deduplication", "pico extraction", "risk of bias assessment",
    "evidence synthesis", "evidence table generation", "review assistance",
    "review methodology", "systematic review methodology",
    "systematic literature review methodology",
}

EXTERNAL_TASK_TERMS = {
    "disease diagnosis", "cancer classification", "breast cancer diagnosis",
    "telemedicine diagnosis", "software refactoring", "threat classification",
    "grain classification", "feature extraction from images",
    "text extraction from medical reports", "financial document extraction",
    "classification of patients", "classification of products",
    "patient screening", "clinical screening", "cybersecurity screening",
    "drug discovery", "academic performance prediction", "speech classification",
}

STRONG_IMPLIED_WORKFLOW_TERMS = {
    "automated systematic literature review", "automated systematic review",
    "automating systematic reviews", "ai assisted systematic review",
    "ai-assisted systematic review", "llm assisted literature review",
    "llm-assisted literature review", "research assistant for literature review",
    "systematic review assistant", "systematic-review assistant",
    "review automation pipeline", "review workflow automation",
    "conducting systematic reviews with ai", "accelerating systematic reviews",
    "supporting reviewers", "reducing reviewer workload",
    "tool for systematic reviews", "software for evidence reviews",
    "review process automation",
    "automated slr", "automated slrs",
    "conduction of automated systematic literature reviews",
    "conduct automated systematic reviews",
    "generative ai to accelerate systematic literature reviews",
    "accelerate systematic literature reviews",
    "enhancing systematic literature reviews with ai",
    "enhancing systematic literature reviews with llms",
    "ai assisted methodology for systematic reviews",
    "ai-assisted methodology for systematic reviews",
    "llm assisted methodology for systematic reviews",
    "llm-assisted methodology for systematic reviews",
    "ai assisted methodology for systematic literature reviews",
    "ai-assisted methodology for systematic literature reviews",
    "llm assisted methodology for systematic literature reviews",
    "llm-assisted methodology for systematic literature reviews",
    "slr automation", "slr assistant",
}

EXTERNAL_OBJECT_TERMS = {
    "patient", "patients", "disease", "cancer", "medical report",
    "medical reports", "image", "images", "product", "products", "grain",
    "grains", "software", "source code", "threat", "network", "speech",
    "student", "students", "financial document", "financial documents",
}


def classify_task_sense(paper_frame):
    source_title = str(
        paper_frame.get("source_title")
        or paper_frame.get("primary_subject", "")
    )
    focused_task_text = " ".join(str(paper_frame.get(field, "")) for field in (
        "source_title", "primary_subject", "target_problem_or_task",
        "target_tasks_or_outcomes",
    ))
    task_text = " ".join(str(paper_frame.get(field, "")) for field in (
        "source_title", "source_abstract", "primary_subject",
        "target_problem_or_task", "target_tasks_or_outcomes",
        "application_context", "application_contexts", "inclusion_cues",
    ))
    task_hits = find_terms(focused_task_text, GENERIC_TASK_TERMS | DIRECT_REVIEW_TASKS)
    object_hits = find_terms(focused_task_text, REVIEW_WORKFLOW_OBJECTS)
    direct_hits = find_terms(task_text, DIRECT_REVIEW_TASKS)
    focused_direct_hits = find_terms(focused_task_text, DIRECT_REVIEW_TASKS)
    external_hits = find_terms(
        focused_task_text,
        EXTERNAL_TASK_TERMS | EXTERNAL_OBJECT_TERMS,
    )
    implied_hits = find_terms(task_text, STRONG_IMPLIED_WORKFLOW_TERMS)
    normalized = normalize(task_text)
    focused_normalized = normalize(focused_task_text)
    title_normalized = normalize(source_title)
    ai_agent = (
        r"(?:artificial intelligence|generative ai|ai|large language models?|"
        r"llms?|gpt(?:-\d+)?|chatgpt)"
    )
    review_target = (
        r"(?:systematic literature reviews?|systematic reviews?|slrs?|"
        r"literature reviews?|evidence reviews?)"
    )
    strong_direction_patterns = (
        rf"\b{ai_agent}\s+to\s+(?:accelerate|automate|conduct|assist|support)\s+{review_target}\b",
        rf"\b{ai_agent}[- ]assisted\s+.{0,50}{review_target}\b",
        rf"\b{ai_agent}[- ]assisted\s+(?:methodolog(?:y|ies)|frameworks?|tools?|pipelines?)\s+(?:for|to|in)\s+.{0,50}{review_target}\b",
        rf"\b(?:methodolog(?:y|ies)|frameworks?|tools?|pipelines?)\s+(?:for|to)\s+.{0,35}(?:automate|conduct|support|assist|accelerate)\s+{review_target}\b",
        rf"\b(?:automated|automating)\s+{review_target}\b",
        rf"\b(?:research|review|slr)\s+assistant\s+(?:for|with)\s+.{0,40}{review_target}\b",
    )
    medium_direction_patterns = (
        rf"\b{ai_agent}\s+(?:for|in)\s+{review_target}\b",
        rf"\b(?:role|potential|efficacy)\s+of\s+{ai_agent}\s+in\s+{review_target}\b",
        rf"\b(?:tools?|software)\s+for\s+{review_target}\b",
    )
    subject_direction_patterns = (
        r"\b(?:systematic(?: literature)?|scoping|evidence|literature) "
        r"reviews?\s+(?:of|on)\s+.{0,60}"
        r"(?:artificial intelligence|generative ai|ai|large language models?|llms?|gpt)\b",
        rf"\b{ai_agent}\s+(?:in|for)\s+.{1,60}\b(?:diagnosis|classification|"
        rf"prediction|healthcare|medicine|education|fintech|cybersecurity)\b",
    )
    strong_direction = any(re.search(pattern, normalized) for pattern in strong_direction_patterns)
    medium_direction = any(re.search(pattern, normalized) for pattern in medium_direction_patterns)
    subject_direction = any(re.search(pattern, normalized) for pattern in subject_direction_patterns)
    if implied_hits or strong_direction:
        implied_score = 0.95
    elif medium_direction:
        implied_score = 0.70
    elif find_terms(task_text, {"review support", "review assistance", "review process"}):
        implied_score = 0.45
    else:
        implied_score = 0.0
    workflow_direction = implied_score >= 0.60
    title_direct_hits = find_terms(source_title, DIRECT_REVIEW_TASKS)
    paper_type_title = bool(re.search(
        r"\b(?:systematic(?: literature)?|scoping|evidence|literature)\s+reviews?\b",
        title_normalized,
    ))
    ai_in_title = bool(re.search(ai_agent, title_normalized))
    if paper_type_title and ai_in_title and not workflow_direction and not title_direct_hits:
        subject_direction = True
    relation_direction = (
        "ai_tool_for_review" if workflow_direction and not subject_direction
        else "review_about_ai" if subject_direction and not workflow_direction
        else "ambiguous" if workflow_direction and subject_direction
        else "none"
    )
    review_target_pattern = bool(re.search(
        r"\b(?:screening|selection|search|extraction|summarization|classification|"
        r"ranking|synthesis)\b.{0,50}\b(?:for|in|from|of)\s+"
        r"(?:systematic|scoping|literature|evidence)\s+reviews?\b",
        normalized,
    ))
    linked_task_object_pattern = bool(re.search(
        r"\b(?:screening|classification|ranking|selection|search|extraction|"
        r"summarization|summarizing|analysis)\b.{0,45}\b(?:papers?|studies|"
        r"articles?|citations?|abstracts?|titles?|records?|references?|queries|"
        r"evidence tables?|pico|eligibility criteria|review stages?|review workflow)\b",
        focused_normalized,
    ) or re.search(
        r"\b(?:papers?|studies|articles?|citations?|abstracts?|titles?|records?|"
        r"references?|queries|evidence tables?|pico|eligibility criteria)\b"
        r".{0,45}\b(?:screening|classification|ranking|selection|search|"
        r"extraction|summarization|summarizing|analysis)\b",
        focused_normalized,
    ))
    if review_target_pattern:
        object_hits = sorted(set(object_hits + ["review process target"]))

    strong_implied = implied_score >= 0.85
    medium_implied = implied_score >= 0.60
    review_object = bool(object_hits)
    linked_review_object = bool(
        focused_direct_hits or review_target_pattern or linked_task_object_pattern
    )
    task_detected = bool(task_hits)
    external_detected = bool(external_hits and task_detected)

    if strong_implied or focused_direct_hits or (
        task_detected and linked_review_object and not external_detected
    ):
        sense = "review_workflow_task"
        confidence = 0.95 if strong_implied or focused_direct_hits else 0.80
    elif external_detected and not strong_implied:
        sense = "external_domain_task"
        confidence = 0.90
    elif task_detected:
        sense = "ambiguous_task"
        confidence = 0.50
    else:
        sense = "no_task_detected"
        confidence = 0.80

    mismatch = sense == "external_domain_task"
    return {
        "workflow_task_term_detected": task_detected,
        "workflow_task_object_detected": review_object,
        "workflow_task_object_linked": linked_review_object,
        "workflow_task_object_terms": join_terms(object_hits),
        "workflow_task_sense": sense,
        "workflow_task_sense_confidence": confidence,
        "external_domain_task_detected": sense == "external_domain_task",
        "external_domain_task_terms": join_terms(external_hits),
        "task_object_mismatch": mismatch,
        "task_object_mismatch_reason": (
            "Detected task acts on an external-domain object rather than papers, "
            "studies, citations, or another review-workflow object."
            if mismatch else ""
        ),
        "implied_workflow_intent_score": implied_score,
        "implied_workflow_intent_terms": join_terms(
            implied_hits
            + (["workflow-direction phrase"] if workflow_direction else [])
        ),
        "strong_implied_workflow_intent": strong_implied,
        "medium_implied_workflow_intent": medium_implied,
        "workflow_direction_detected": workflow_direction,
        "subject_review_direction_detected": subject_direction,
        "relation_direction": relation_direction,
        "workflow_task_object_required": True,
        "workflow_task_object_missing": (
            task_detected and not linked_review_object and not strong_implied
        ),
    }
