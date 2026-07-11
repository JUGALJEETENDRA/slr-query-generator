import re

from domain_vocabulary import find_terms, join_terms, normalize
from task_sense_disambiguator import classify_task_sense


REVIEW_CONTEXT_TERMS = {
    "systematic literature review",
    "systematic literature reviews",
    "systematic review",
    "systematic reviews",
    "slr",
    "scoping review",
    "rapid review",
    "evidence review",
    "literature review",
    "review workflow",
}

WORKFLOW_TASK_FAMILIES = {
    "search_strategy_generation": {
        "search strategy generation", "search query generation", "database querying",
        "query formulation",
    },
    "literature_search": {"literature search", "database search", "evidence search"},
    "query_expansion": {"query expansion", "search expansion"},
    "citation_retrieval": {"citation retrieval", "citation discovery", "paper retrieval"},
    "deduplication": {"deduplication", "duplicate removal", "duplicate detection"},
    "title_abstract_screening": {
        "title abstract screening", "title and abstract screening", "abstract screening",
        "title screening", "citation screening", "paper screening", "article screening",
        "systematic review screening", "review screening",
    },
    "full_text_screening": {"full text screening", "full-text screening"},
    "study_selection": {
        "study selection", "paper selection", "article selection", "eligibility assessment",
        "eligibility screening",
    },
    "data_extraction": {
        "data extraction", "structured extraction", "study characteristics extraction",
    },
    "information_extraction": {"information extraction", "entity extraction"},
    "pico_extraction": {"pico extraction", "picos extraction"},
    "risk_of_bias_assessment": {
        "risk of bias", "risk-of-bias", "rob assessment", "quality appraisal",
        "quality assessment",
    },
    "evidence_synthesis": {"evidence synthesis", "research synthesis"},
    "meta_analysis_support": {"meta analysis support", "meta-analysis support"},
    "summarization": {"review summarization", "evidence summarization"},
    "report_generation": {"review report generation", "systematic review writing"},
    "review_assistance": {"review assistance", "review support", "review assistant"},
    "automation_pipeline": {
        "review automation", "systematic review automation", "automation pipeline",
        "automated systematic review", "automating systematic reviews",
    },
    "human_in_the_loop_screening": {
        "human in the loop screening", "human-in-the-loop screening",
    },
    "active_learning_screening": {"active learning screening", "active learning"},
}

AI_TOOL_TERMS = {
    "large language model", "large language models", "llm", "llms", "gpt", "gpt-4",
    "chatgpt", "gemini", "claude", "llama", "qwen", "bert", "t5",
    "artificial intelligence", "ai tool", "ai tools", "machine learning", "deep learning",
    "natural language processing", "nlp", "text mining", "active learning",
    "automation tool", "automation tools", "screening tool", "screening tools",
    "systematic review software", "classifier", "classifiers",
}

EXTERNAL_DOMAIN_TERMS = {
    "cardiology", "medicine", "medical imaging", "psychiatry", "dentistry", "nutrition",
    "pharmacovigilance", "education", "fintech", "cybersecurity", "agriculture",
    "software development", "mobility", "diagnosis", "healthcare", "radiology",
    "breast cancer", "cancer", "telemedicine", "drug discovery", "finance",
    "financial", "security operations center", "soc", "k 12", "k-12", "law",
    "legal", "manufacturing",
}

AUTOMATION_INTENT_TERMS = {
    "automate systematic reviews", "automating systematic reviews",
    "automated systematic review", "ai assisted systematic review",
    "ai-assisted systematic review", "llm assisted review", "llm-assisted review",
    "review automation", "systematic review automation", "evidence review automation",
    "accelerate reviews", "accelerating reviews", "reduce reviewer workload",
    "reducing reviewer burden", "support reviewers", "support systematic reviews",
    "conduct systematic reviews", "review pipeline", "review workflow",
    "review process", "review stages", "systematic review tool",
    "literature review assistant", "research assistant for literature review",
}

IMPLIED_AUTOMATION_TERMS = {
    "llm in systematic reviews", "llms in systematic reviews",
    "ai in systematic reviews", "artificial intelligence in systematic reviews",
    "ai for systematic reviews", "llm for systematic reviews",
}

EVALUATION_TERMS = {
    "evaluate llm", "evaluate llms", "assess llm", "assess llms",
    "evaluating llm", "evaluating llms", "evaluating the performance",
    "performance of llm", "performance of llms",
    "performance in systematic reviews", "probability rating",
    "relevance ranking", "screening prioritization", "reviewer agreement",
    "included study prediction", "excluded study prediction",
    "classification of studies", "classification of papers",
    "systematic review stages", "review stages",
}

METHODOLOGY_TERMS = {
    "review methodology", "review method", "review methods", "methodology tool",
    "methods for conducting reviews", "review process improvement",
    "review workflow design", "review framework", "review protocol automation",
}

SYNTHESIS_SUPPORT_TERMS = {
    "summarize included studies", "structured evidence extraction",
    "evidence table generation", "generate evidence tables",
    "synthesize study findings", "clinical evidence review support",
    "meta analysis support", "meta-analysis support", "evidence synthesis",
}


def workflow_terms(text):
    context_present = bool(find_terms(text, REVIEW_CONTEXT_TERMS))
    detected = []
    families = []
    for family, terms in WORKFLOW_TASK_FAMILIES.items():
        hits = find_terms(text, terms)
        if hits:
            detected.extend(hits)
            families.append(family)
    clinical_screening = bool(find_terms(
        text,
        {
            "disease screening", "clinical screening", "patient screening",
            "cancer screening", "health screening", "diagnostic screening",
        },
    ))
    # Bare screening counts only in review context and never in a clinical sense.
    if context_present and not clinical_screening:
        detected.extend(find_terms(text, {"screening"}))
    return sorted(set(families)), sorted(set(detected))


def classify_review_intent(paper_frame):
    fields = (
        "source_title", "source_abstract", "primary_subject",
        "intervention_or_method", "target_problem_or_task",
        "application_context", "evidence_type", "study_role", "review_role",
        "methods_or_technologies", "target_tasks_or_outcomes", "application_contexts",
        "inclusion_cues",
    )
    text = " ".join(str(paper_frame.get(field, "")) for field in fields)
    normalized = normalize(text)
    primary_text = str(
        paper_frame.get("source_title")
        or paper_frame.get("primary_subject", "")
    )
    primary_normalized = normalize(primary_text)
    context_hits = find_terms(text, REVIEW_CONTEXT_TERMS)
    tool_hits = find_terms(text, AI_TOOL_TERMS)
    task_families, task_hits = workflow_terms(text)
    external_hits = find_terms(text, EXTERNAL_DOMAIN_TERMS)
    automation_hits = find_terms(text, AUTOMATION_INTENT_TERMS)
    implied_automation_hits = find_terms(text, IMPLIED_AUTOMATION_TERMS)
    evaluation_hits = find_terms(text, EVALUATION_TERMS)
    methodology_hits = find_terms(text, METHODOLOGY_TERMS)
    synthesis_hits = find_terms(text, SYNTHESIS_SUPPORT_TERMS)
    task_sense = classify_task_sense(paper_frame)
    valid_methodology = bool(
        methodology_hits
        and (
            task_sense["workflow_task_sense"] == "review_workflow_task"
            or task_sense["workflow_task_object_linked"]
            or task_sense["workflow_direction_detected"]
            or task_sense["strong_implied_workflow_intent"]
        )
    )
    process_evidence = bool(
        task_hits or automation_hits or evaluation_hits or valid_methodology or synthesis_hits
    )
    strong_process_evidence = bool(
        task_sense["workflow_task_sense"] == "review_workflow_task"
        or task_sense["strong_implied_workflow_intent"]
        or automation_hits
        or evaluation_hits
        or valid_methodology
        or synthesis_hits
    )
    primary_task_families, _ = workflow_terms(primary_text)
    primary_automation_hits = find_terms(primary_text, AUTOMATION_INTENT_TERMS)
    primary_evaluation_hits = find_terms(primary_text, EVALUATION_TERMS)
    primary_workflow_evidence = bool(
        primary_task_families or primary_automation_hits or primary_evaluation_hits
    )
    direct_workflow_override = bool(
        primary_workflow_evidence
        or task_sense["workflow_direction_detected"]
        or task_sense["strong_implied_workflow_intent"]
        or (
            task_sense["workflow_task_sense"] == "review_workflow_task"
            and task_sense["workflow_task_object_linked"]
        )
        or evaluation_hits
        or valid_methodology
        or synthesis_hits
    )
    ai_pattern = (
        r"(?:artificial intelligence|explainable artificial intelligence|xai|"
        r"generative ai|ai|large language models?|llms?|machine learning)"
    )
    review_of_subject_pattern = bool(re.search(
        rf"\b(?:systematic\s+(?:literature\s+)?review|scoping\s+review|"
        rf"literature\s+review|survey)\s+(?:of|on)\s+.{0,50}{ai_pattern}\b",
        primary_normalized,
    ))
    application_subject_pattern = bool(
        context_hits
        and re.search(
            rf"\b(?:applications?|impact|role|use|approaches?|integration)\s+(?:of\s+)?"
            rf"{ai_pattern}\s+(?:in|for|on)\s+",
            primary_normalized,
        )
    )
    ai_in_domain_pattern = bool(
        context_hits
        and re.search(rf"\b{ai_pattern}\s+(?:in|for)\s+", primary_normalized)
        and not re.search(
            rf"\b{ai_pattern}\s+(?:in|for)\s+(?:systematic|scoping|literature|evidence)\s+review",
            primary_normalized,
        )
    )
    subject_pattern = bool(
        review_of_subject_pattern
        or application_subject_pattern
        or ai_in_domain_pattern
        or task_sense["subject_review_direction_detected"]
    )
    explicit_workflow_override = bool(
        direct_workflow_override
        or (automation_hits and not subject_pattern)
    )
    subject_review_contrast = bool(subject_pattern and not explicit_workflow_override)
    tool_for_workflow = bool(
        tool_hits
        and strong_process_evidence
        and context_hits
        and not subject_review_contrast
    )
    model_subject_role = str(paper_frame.get("review_role", "")).strip() == "technology_being_reviewed"
    external_subject_pattern = bool(
        context_hits
        and tool_hits
        and (external_hits or application_subject_pattern or ai_in_domain_pattern)
        and not explicit_workflow_override
        and not implied_automation_hits
    )
    if task_sense["workflow_task_sense"] == "external_domain_task":
        external_subject_pattern = bool(context_hits and tool_hits)
    technology_subject = bool(
        not tool_for_workflow
        and not implied_automation_hits
        and not task_sense["workflow_direction_detected"]
        and (
            subject_pattern
            or subject_review_contrast
            or external_subject_pattern
            or (model_subject_role and tool_hits and (external_hits or not context_hits))
        )
    )
    domain_ai_review = bool(
        technology_subject
        and (
            external_hits
            or application_subject_pattern
            or ai_in_domain_pattern
            or task_sense["subject_review_direction_detected"]
        )
    )
    review_type_only = bool(context_hits and not tool_hits and not process_evidence)

    if tool_for_workflow and synthesis_hits:
        relation = "evidence_synthesis_tool"
        confidence = 0.95
    elif tool_for_workflow and evaluation_hits:
        relation = "review_workflow_evaluation"
        confidence = 0.95
    elif tool_for_workflow and valid_methodology:
        relation = "review_workflow_methodology"
        confidence = 0.90
    elif tool_for_workflow:
        relation = "review_workflow_tool"
        confidence = 0.90
    elif domain_ai_review:
        relation = "external_domain_review"
        confidence = 0.90
    elif technology_subject:
        relation = "technology_subject_review"
        confidence = 0.85
    elif review_type_only:
        relation = "review_context_only"
        confidence = 0.85
    elif context_hits and (tool_hits or process_evidence or implied_automation_hits):
        relation = "unclear_review_relation"
        confidence = 0.55
    elif not context_hits and not process_evidence:
        relation = "unrelated_to_review_workflow"
        confidence = 0.85
    else:
        relation = "unclear_review_relation"
        confidence = 0.45

    return {
        "review_intent_relation": relation,
        "review_relation_confidence": confidence,
        "review_workflow_task_detected": strong_process_evidence,
        "review_workflow_task_terms": join_terms(
            task_hits + automation_hits + evaluation_hits + methodology_hits + synthesis_hits
        ),
        "review_workflow_task_families": join_terms(task_families),
        "review_automation_intent_detected": bool(
            automation_hits
            or implied_automation_hits
            or task_sense["implied_workflow_intent_score"] >= 0.35
        ),
        "review_automation_intent_terms": join_terms(
            automation_hits
            + implied_automation_hits
            + [task_sense["implied_workflow_intent_terms"]]
        ),
        "review_context_detected": bool(context_hits),
        "review_context_terms": join_terms(context_hits),
        "ai_tool_terms": join_terms(tool_hits),
        "external_domain_terms": join_terms(external_hits),
        "ai_tool_for_review_workflow": tool_for_workflow,
        "strong_workflow_use_evidence": strong_process_evidence,
        "subject_review_pattern_detected": subject_pattern,
        "explicit_workflow_override_detected": explicit_workflow_override,
        "technology_subject_review_detected": technology_subject,
        "external_domain_review_detected": domain_ai_review,
        "review_context_only": review_type_only,
        "review_workflow_methodology_validated": valid_methodology,
        **task_sense,
    }


def review_question_analysis():
    task_terms = []
    for terms in WORKFLOW_TASK_FAMILIES.values():
        task_terms.extend(sorted(terms))
    task_terms.extend(sorted(AUTOMATION_INTENT_TERMS))
    task_terms.extend(sorted(EVALUATION_TERMS))
    task_terms.extend(sorted(METHODOLOGY_TERMS))
    task_terms.extend(sorted(SYNTHESIS_SUPPORT_TERMS))
    return {
        "target_tasks_or_outcomes": join_terms([
            "automate systematic literature reviews",
            "review workflow automation",
            *task_terms,
        ]),
        "task_outcome_synonyms": join_terms(task_terms),
        "exclusion_concepts": join_terms([
            "AI as reviewed subject only",
            "systematic review of AI in an external domain",
            "generic AI application review",
        ]),
    }
