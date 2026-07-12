import json
import hashlib
import os
import re
from threading import Lock
from ollama_client import ask_ollama
from domain_vocabulary import (
    analyze_paper_text,
    analyze_research_question,
    enrich_rq_analysis_with_profile,
)
from screening_contracts import build_rq_contract
from domain_vocabulary import join_terms


SEMANTIC_FRAME_FIELDS = (
    "primary_subject",
    "intervention_or_method",
    "target_problem_or_task",
    "application_context",
    "evidence_type",
    "study_role",
    "review_role",   # new field
    "question_type",
    "frame_source",
    "frame_diagnostic",
    "review_question_type",
    "core_domain",
    "method_or_technology",
    "method_family",
    "target_tasks_or_outcomes",
    "required_inclusion_concepts",
    "optional_related_concepts",
    "exclusion_concepts",
    "expected_evidence_types",
    "domain_synonyms",
    "method_synonyms",
    "task_outcome_synonyms",
    "context_synonyms",
    "negative_contexts",
    "required_dimensions",
    "minimum_inclusion_rule",
    "rq_extraction_suspect",
    "rq_desired_relation",
    "rq_id",
    "rq_text",
    "rq_type",
    "rq_scope_width",
    "rq_strictness",
    "rq_required_dimensions",
    "rq_optional_dimensions",
    "rq_method_terms",
    "rq_method_families",
    "rq_task_terms",
    "rq_task_families",
    "rq_context_terms",
    "rq_context_families",
    "rq_outcome_terms",
    "rq_evidence_types_expected",
    "rq_inclusion_concepts",
    "rq_exclusion_concepts",
    "rq_positive_relation_patterns",
    "rq_negative_relation_patterns",
    "rq_ambiguity_policy",
    "rq_stage2_policy",
    "corpus_profile_terms",
    "corpus_method_terms",
    "corpus_task_terms",
    "corpus_context_terms",
    "corpus_evidence_terms",
    "corpus_domain_specific_synonyms",
    "corpus_review_context_terms",
    "corpus_workflow_task_terms",
    "corpus_ai_tool_terms",
    "corpus_external_domain_terms",
    "corpus_technology_subject_terms",
    "corpus_automation_intent_terms",
    "corpus_review_workflow_terms",
    "corpus_tool_use_terms",
    "corpus_subject_review_terms",
    "corpus_relation_clusters",
    "main_domain",
    "application_contexts",
    "methods_or_technologies",
    "specific_models_or_systems",
    "contribution_type",
    "inclusion_cues",
    "exclusion_cues",
    "direct_application_present",
    "source_title",
    "source_abstract",
)

_FRAME_CACHE = {}
_FRAME_CACHE_LOCK = Lock()
_INCOMPLETE_DISK_KEYS = {}
_DISK_FRAME_CACHE_LOADED = False
SEMANTIC_FRAME_CACHE_SCHEMA_VERSION = 2
_DISK_FRAME_CACHE_PATH = os.getenv(
    "SEMANTIC_FRAME_CACHE_PATH",
    os.path.join("outputs", "cache", "semantic_frame_cache.jsonl"),
)
_SEMANTIC_FRAME_CACHE_STATS = {
    "semantic_frame_cache_loaded": 0,
    "semantic_frame_cache_hits": 0,
    "semantic_frame_cache_misses": 0,
    "semantic_frame_cache_invalid": 0,
    "semantic_frame_cache_write_count": 0,
    "semantic_frame_cache_lookup_seconds": 0.0,
    "semantic_frame_ollama_seconds": 0.0,
}


def _empty_frame():
    return {field: "" for field in SEMANTIC_FRAME_FIELDS}


def _normalize_frame(raw_frame):
    frame = _empty_frame()

    if not isinstance(raw_frame, dict):
        return frame

    for field in SEMANTIC_FRAME_FIELDS:
        value = raw_frame.get(field, "")
        if value is None:
            value = ""
        elif not isinstance(value, str):
            value = str(value)
        frame[field] = value.strip()

    return frame


def _cache_frame_completeness(raw_frame):
    if not isinstance(raw_frame, dict):
        return False, ["value_not_object"]
    missing = [field for field in SEMANTIC_FRAME_FIELDS if field not in raw_frame]
    for field in ("source_title", "source_abstract"):
        if not str(raw_frame.get(field) or "").strip() and field not in missing:
            missing.append(field)
    evidence_fields = ("intervention_or_method", "target_problem_or_task", "review_role")
    if not any(str(raw_frame.get(field) or "").strip() for field in evidence_fields):
        missing.append("semantic_evidence")
    return not missing, missing


def _json_object_from_response(response):
    text = str(response or "").strip()
    if not text:
        raise ValueError("empty model response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("model response did not contain a JSON object")


def _contains_any(text, terms):
    normalized = str(text or "").lower()
    return any(term in normalized for term in terms)


def _extract_method_hint(text):
    method_terms = [
        "large language models",
        "language models",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "neural networks",
        "transformer models",
        "natural language processing",
        "computer vision",
    ]
    found = [term for term in method_terms if term in str(text or "").lower()]
    return " and ".join(dict.fromkeys(found))


def _extract_task_hint(text):
    normalized = str(text or "").lower()
    task_terms = [
        "prediction",
        "diagnosis",
        "detection",
        "classification",
        "screening",
        "study selection",
        "data extraction",
        "risk of bias",
        "evidence synthesis",
        "search strategy generation",
        "deduplication",
        "summarization",
        "generation",
        "retrieval",
    ]
    found = [term for term in task_terms if term in normalized]

    if _contains_any(normalized, ["systematic literature review", "systematic reviews", "literature reviews"]):
        if found:
            return " and ".join(dict.fromkeys(found)) + " for literature review workflows"
        if _contains_any(normalized, ["automate", "automation", "assist", "help"]):
            return "automating systematic literature review workflows"

    disease = ""
    if "heart disease" in normalized:
        disease = "heart disease"
    elif "cardiovascular disease" in normalized:
        disease = "cardiovascular disease"
    else:
        disease_match = re.search(r"\b(cancer|diabetes|[a-z0-9 -]+ disease)\b", normalized)
        if disease_match:
            disease = re.sub(r"\s+", " ", disease_match.group(1)).strip(" -")

    if disease and found:
        return f"{disease} " + " and ".join(dict.fromkeys(found))

    if found:
        return " and ".join(dict.fromkeys(found))

    return str(text or "").strip()


def _extract_context_hint(text):
    normalized = str(text or "").lower()
    if _contains_any(normalized, ["heart", "cardiovascular", "cardiac"]):
        return "cardiovascular healthcare and medical diagnosis"
    if _contains_any(normalized, ["disease", "diagnosis", "clinical", "medical", "healthcare", "patient"]):
        return "healthcare and medical diagnosis"
    if _contains_any(normalized, ["systematic literature review", "systematic reviews", "literature reviews", "evidence reviews"]):
        return "evidence reviews and systematic literature review workflows"
    if _contains_any(normalized, ["cybersecurity", "security"]):
        return "cybersecurity"
    if _contains_any(normalized, ["education", "learning analytics"]):
        return "education"
    if _contains_any(normalized, ["software engineering", "code", "programming"]):
        return "software engineering"
    return ""


def _infer_question_type(text):
    normalized = str(text or "").lower()
    if _contains_any(normalized, ["systematic literature review", "systematic reviews", "evidence review", "literature review"]):
        if _contains_any(normalized, ["automate", "automation", "assist", "help", "screening", "study selection", "data extraction", "risk of bias", "evidence synthesis"]):
            return "review_workflow_automation"
    return "domain_literature_review"


def _heuristic_research_question_frame(research_question, diagnostic=""):
    analysis = analyze_research_question(research_question)
    contract = build_rq_contract(research_question, analysis)
    analysis.update({
        key: join_terms(value) if isinstance(value, list) else str(value)
        for key, value in contract.items()
    })
    method = _extract_method_hint(research_question)
    task = _extract_task_hint(research_question)
    context = _extract_context_hint(research_question)
    question_type = analysis.get("review_question_type") or _infer_question_type(research_question)
    review_role = "review_assistance" if question_type == "review_workflow_automation" else ""
    method = analysis.get("method_or_technology") or method
    task = analysis.get("target_tasks_or_outcomes") or task
    context = analysis.get("application_context") or context

    frame = _normalize_frame({
        "primary_subject": task or str(research_question or "").strip(),
        "intervention_or_method": method,
        "target_problem_or_task": task or str(research_question or "").strip(),
        "application_context": context,
        "evidence_type": "",
        "study_role": "",
        "review_role": review_role,
        "question_type": question_type,
        "frame_source": "heuristic",
        "frame_diagnostic": diagnostic,
        **analysis,
    })
    return frame


def _merge_rq_with_fallback(model_frame, fallback_frame):
    merged = dict(model_frame)
    for field in SEMANTIC_FRAME_FIELDS:
        if not merged.get(field):
            merged[field] = fallback_frame.get(field, "")
    if (
        fallback_frame.get("application_context")
        and merged.get("application_context", "").lower() == "cybersecurity"
        and "supply chain" in fallback_frame.get("application_context", "").lower()
    ):
        merged["application_context"] = fallback_frame.get("application_context", "")
    if fallback_frame.get("intervention_or_method") and not model_frame.get("intervention_or_method"):
        merged["intervention_or_method"] = fallback_frame.get("intervention_or_method", "")

    if merged.get("review_question_type") and not merged.get("question_type"):
        merged["question_type"] = merged["review_question_type"]

    if merged.get("question_type") not in {"review_workflow_automation"}:
        merged["review_role"] = ""
        merged["study_role"] = ""
        merged["evidence_type"] = ""
    elif not merged.get("review_role"):
        merged["review_role"] = fallback_frame.get("review_role", "")

    merged["frame_source"] = "model"
    if not merged.get("frame_diagnostic"):
        missing = [
            field
            for field in ("primary_subject", "intervention_or_method", "target_problem_or_task", "application_context")
            if not model_frame.get(field)
        ]
        merged["frame_diagnostic"] = (
            "model frame missing " + ", ".join(missing) + "; heuristic fallback filled gaps"
            if missing
            else ""
        )
    return _normalize_frame(merged)


def enrich_research_question_frame_with_corpus(rq_frame, corpus_profile):
    enriched = dict(rq_frame or {})
    enriched.update(corpus_profile or {})
    analysis = enrich_rq_analysis_with_profile(enriched, corpus_profile or {})
    enriched.update(analysis)
    return _normalize_frame(enriched)


def extract_research_question_frame(
    research_question,
    model="qwen2.5:3b",
    inference_engine=None,
):
    cache_key = (
        getattr(inference_engine, "engine_id", "local"),
        str(model or ""),
        "research_question",
        " ".join(str(research_question or "").split()),
    )
    with _FRAME_CACHE_LOCK:
        cached = _FRAME_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    fallback = _heuristic_research_question_frame(research_question)
    prompt = f"""
Research Question:
{research_question}

Extract the semantic role frame for this REVIEW QUESTION.

This is not a paper title. Interpret it as the scope of a literature review.

Fields:

primary_subject:
The overall review topic.

intervention_or_method:
The technology, model, method, intervention, or class of methods the review is asking about.
For broad review questions, keep broad classes such as "machine learning and deep learning" when that is the intended scope.

target_problem_or_task:
The task, problem, workflow, decision, activity, or use case the review wants evidence about.
Include all explicitly named tasks when the question is multi-task, such as "prediction and diagnosis".

application_context:
The domain, field, setting, population, or context where the task occurs.

evidence_type:
Leave blank unless the question explicitly restricts evidence type.

study_role:
Leave blank unless the question explicitly restricts paper type.

review_role:
Use a review workflow role only when the question is about technology helping perform systematic-review work.
Examples: screening, study_selection, search_strategy_generation, deduplication, data_extraction, pico_extraction, risk_of_bias, evidence_synthesis, review_assistance.
Leave blank for normal domain review questions where review/survey papers can be relevant evidence.

question_type:
Use "review_workflow_automation" only for questions about automating or assisting systematic/literature review workflows.
Use "domain_literature_review" for ordinary domain questions such as medical AI, cybersecurity, education, or software engineering.

Return ONLY JSON with exactly these keys:

{{
  "primary_subject": "",
  "intervention_or_method": "",
  "target_problem_or_task": "",
  "application_context": "",
  "evidence_type": "",
  "study_role": "",
  "review_role": "",
  "question_type": "",
  "frame_source": "model",
  "frame_diagnostic": ""
}}

No KEEP/REJECT decision.
No explanation.
No markdown.
"""

    try:
        ask = inference_engine.ask if inference_engine is not None else ask_ollama
        response = ask(prompt, model=model)
        frame = _merge_rq_with_fallback(_normalize_frame(_json_object_from_response(response)), fallback)
    except Exception as exc:
        frame = _heuristic_research_question_frame(
            research_question,
            diagnostic=f"RQ frame model/parse fallback used: {exc}",
        )

    with _FRAME_CACHE_LOCK:
        _FRAME_CACHE[cache_key] = dict(frame)
    return frame


def extract_semantic_frame(
    title,
    abstract,
    model="qwen2.5:3b",
    inference_engine=None,
):
    cache_key = (
        getattr(inference_engine, "engine_id", "local"),
        str(model or ""),
        "paper",
        " ".join(str(title or "").split()),
        " ".join(str(abstract or "").split()),
    )
    with _FRAME_CACHE_LOCK:
        cached = _FRAME_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    disk_key = _semantic_frame_disk_key(title, abstract, model)
    use_disk_cache = _semantic_frame_cache_enabled()
    if use_disk_cache:
        lookup_started = __import__("time").perf_counter()
        _load_disk_frame_cache()
        with _FRAME_CACHE_LOCK:
            cached = _FRAME_CACHE.get(disk_key)
        _SEMANTIC_FRAME_CACHE_STATS["semantic_frame_cache_lookup_seconds"] += (
            __import__("time").perf_counter() - lookup_started
        )
        if cached is not None:
            _SEMANTIC_FRAME_CACHE_STATS["semantic_frame_cache_hits"] += 1
            with _FRAME_CACHE_LOCK:
                _FRAME_CACHE[cache_key] = dict(cached)
            output = dict(cached)
            output["cache_schema_complete"] = True
            output["cache_missing_adjudication_fields"] = ""
            output["fast_recomputed_due_to_incomplete_cache"] = False
            return output
        _SEMANTIC_FRAME_CACHE_STATS["semantic_frame_cache_misses"] += 1
    incomplete_fields = list(_INCOMPLETE_DISK_KEYS.get(disk_key[1], [])) if use_disk_cache else []

    prompt = f"""
Paper Title:
{title}

Paper Abstract:
{abstract}

Extract the semantic role frame for this paper.

Definitions:

primary_subject:
The main thing the paper is about at the highest level.

intervention_or_method:
The technology, model, framework, method, tool, system, intervention, or class of techniques being used, evaluated, reviewed, proposed, or studied.

This field answers:
"What is the underlying method or technology?"

Return the most specific method, model, algorithm, architecture, tool, or system that is explicitly evaluated, proposed, used, or reviewed in the text.

Prefer the specific named method over a broad umbrella category.

Examples:
- If the text evaluates a named classifier, return that classifier rather than "machine learning" or "artificial intelligence".
- If the text evaluates a named neural architecture, return that architecture rather than "deep learning".
- If the text evaluates a named language model or model family, return that model or family rather than "AI".
- If only a broad class is given, use the narrowest class explicitly stated in the text.

Do NOT put any of the following in intervention_or_method:
- applications
- use cases
- downstream tasks
- outcomes
- datasets
- benchmarks
- evaluation metrics
- domains
- settings
- fields of application

If the paper discusses many applications or use cases of a technology, intervention_or_method should still be the technology itself, not the list of applications.

For review, survey, scoping review, and mapping study papers:
intervention_or_method should be the technology, model, method, or class of technologies being reviewed.
It should NOT be the review method itself.
It should NOT be the applications covered by the review.

IMPORTANT:
If the paper is a review, survey, scoping review, or mapping study:

primary_subject =
    what the review is about

intervention_or_method =
    the technology being reviewed

target_problem_or_task =
    what that technology is used for

Example:

Review of transformer models for code generation

primary_subject = transformer models for code generation
intervention_or_method = transformer models
target_problem_or_task = code generation

target_problem_or_task:
The task, problem, workflow, decision, activity, or use case that the intervention_or_method is used for, applied to, evaluated on, or intended to address.

This field answers:
"What is it used for?"

Return the verb-oriented objective of the method.

A good target_problem_or_task usually combines:
- the action or objective, such as prediction, diagnosis, classification, detection, segmentation, retrieval, screening, extraction, generation, ranking, recommendation, forecasting, summarization, or synthesis
- the object or workflow being acted on

Do NOT return only a broad domain, condition, population, dataset, or field as the task.

Examples:
- Use "disease risk prediction" rather than only "disease".
- Use "image classification" rather than only "medical imaging".
- Use "document retrieval" rather than only "documents".
- Use "study screening for evidence reviews" rather than only "systematic reviews".

Keep related tasks distinct. Do not normalize one task into a neighboring task merely because they occur in the same domain.

Examples:
- prediction is not the same task as diagnosis
- screening is not the same task as evidence synthesis
- retrieval is not the same task as generation
- detection is not the same task as classification unless the text explicitly treats them as the same objective

If the text has a pattern like "X for Y":
- X is usually intervention_or_method
- Y is usually target_problem_or_task

application_context:
The broader domain, field, setting, population, environment, or context where the target task occurs.

This field answers:
"Where or in what domain is it used?"

If the text has a pattern like "X in Y":
- X is usually intervention_or_method
- Y is usually application_context

evidence_type:
The kind of evidence or contribution the paper provides, such as empirical evaluation, benchmark study, systematic review, survey, scoping review, mapping study, case study, dataset, framework, tool paper, or method proposal.

study_role:
The role of the paper itself, such as systematic review, survey, scoping review, mapping study, empirical evaluation, benchmark paper, tool/method paper, framework paper, case study, dataset paper, or position paper.

IMPORTANT:

study_role describes the role of THIS PAPER itself.

Do not infer "systematic review", "scoping review", "mapping study", or "survey" merely because the paper supports, automates, evaluates, assists, or improves a review workflow.

Examples:

Paper evaluating a method:
study_role = empirical evaluation

Paper proposing a tool:
study_role = tool/method paper

Paper proposing a framework:
study_role = framework paper

Paper introducing a dataset:
study_role = dataset paper

Only use:
systematic review
scoping review
mapping study
survey

when the paper itself synthesizes prior literature.

review_role:

Determine the relationship between the technology and the review process.

technology_being_reviewed:
The paper is a review, survey, scoping review, or mapping study whose purpose is to summarize, analyze, categorize, evaluate, or discuss the technology itself and its applications.

Examples:
- review of a technology
- survey of a technology
- mapping study of a technology
- review of applications of a technology

screening:
Technology used to screen papers.

study_selection:
Technology used to select studies.

search_strategy_generation:
Technology used to generate search queries or search strategies.

deduplication:
Technology used to identify duplicate records.

data_extraction:
Technology used to extract study information.

pico_extraction:
Technology used to extract PICO elements.

risk_of_bias:
Technology used to assess risk of bias.

evidence_synthesis:
Technology used to synthesize evidence.

review_assistance:
Technology used to assist review workflows.

IMPORTANT:

Ask:

Is the technology helping perform a review task?

OR

Is the technology itself being reviewed?

If the paper reviews the technology, its applications, capabilities, limitations, or use cases:

review_role = technology_being_reviewed

Anti-confusion checks before returning JSON:

1. If intervention_or_method is a list of tasks, use cases, applications, outcomes, datasets, metrics, or domains, move that content to target_problem_or_task or application_context and replace intervention_or_method with the underlying technology, method, model, framework, tool, or system.

2. If the paper is a review or survey, do not confuse:
- the technology being reviewed
with
- the applications covered by the review
or
- the review methodology.

3. Keep intervention_or_method concise. It should usually be a noun phrase naming the method or technology, not a long list.

4. Keep target_problem_or_task as a precise task phrase, not just a topic. It should usually contain an action/objective and the thing being acted on.

5. Canonicalize entity names only when this preserves the same meaning. Do not canonicalize tasks into broader or adjacent tasks.

Return ONLY JSON with exactly these keys:

{{
  "primary_subject": "",
  "intervention_or_method": "",
  "target_problem_or_task": "",
  "application_context": "",
  "evidence_type": "",
  "study_role": "",
  "review_role": ""
}}

No KEEP/REJECT decision.
No relevance judgment.
No explanation.
No markdown.
"""

    ask = inference_engine.ask if inference_engine is not None else ask_ollama
    extraction_started = __import__("time").perf_counter()
    response = ask(prompt, model=model)
    _SEMANTIC_FRAME_CACHE_STATS["semantic_frame_ollama_seconds"] += (
        __import__("time").perf_counter() - extraction_started
    )

    frame = _normalize_frame(_json_object_from_response(response))
    frame["frame_source"] = "model"
    analysis = analyze_paper_text(title, abstract)
    for field, value in analysis.items():
        if value and not frame.get(field):
            frame[field] = value
    if analysis.get("methods_or_technologies") and not frame.get("intervention_or_method"):
        frame["intervention_or_method"] = analysis["methods_or_technologies"]
    if analysis.get("target_tasks_or_outcomes") and not frame.get("target_problem_or_task"):
        frame["target_problem_or_task"] = analysis["target_tasks_or_outcomes"]
    if analysis.get("application_contexts") and not frame.get("application_context"):
        frame["application_context"] = analysis["application_contexts"]
    frame["source_title"] = str(title or "").strip()
    frame["source_abstract"] = str(abstract or "").strip()
    frame = _normalize_frame(frame)
    frame["cache_schema_complete"] = not bool(incomplete_fields)
    frame["cache_missing_adjudication_fields"] = "; ".join(incomplete_fields)
    frame["fast_recomputed_due_to_incomplete_cache"] = bool(incomplete_fields)
    with _FRAME_CACHE_LOCK:
        _FRAME_CACHE[cache_key] = dict(frame)
        if use_disk_cache:
            _FRAME_CACHE[disk_key] = dict(frame)
    if use_disk_cache:
        _append_disk_frame_cache(disk_key, frame)
    return frame


def _semantic_frame_disk_key(title, abstract, model) -> tuple:
    text = "\n".join([
        f"semantic_frame_v{SEMANTIC_FRAME_CACHE_SCHEMA_VERSION}",
        _cache_normalize(model),
        _cache_normalize(title),
        hashlib.sha1(_cache_normalize(abstract).encode("utf-8", errors="ignore")).hexdigest(),
    ])
    return ("disk", hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest())


def _cache_normalize(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _semantic_frame_cache_enabled() -> bool:
    from runtime_config import get_model_judge_config
    cfg = get_model_judge_config()
    if cfg["screening_pipeline_mode"] == "current":
        return bool(cfg.get("enable_current_mode_cache"))
    return bool(cfg.get("enable_semantic_frame_cache"))


def get_semantic_frame_cache_stats() -> dict:
    return dict(_SEMANTIC_FRAME_CACHE_STATS)


def initialize_semantic_frame_cache() -> dict:
    """Load the enabled disk cache once and return process-level cache information."""
    if _semantic_frame_cache_enabled():
        _load_disk_frame_cache()
    with _FRAME_CACHE_LOCK:
        entries = sum(1 for key in _FRAME_CACHE if isinstance(key, tuple) and key[:1] == ("disk",))
    return {
        "path": _DISK_FRAME_CACHE_PATH,
        "schema_version": SEMANTIC_FRAME_CACHE_SCHEMA_VERSION,
        "entries": entries,
        **get_semantic_frame_cache_stats(),
    }


def configure_semantic_frame_cache(path: str, *, reset_memory: bool = True) -> None:
    """Select an isolated cache file before a benchmark run."""
    global _DISK_FRAME_CACHE_PATH, _DISK_FRAME_CACHE_LOADED
    _DISK_FRAME_CACHE_PATH = str(path)
    _DISK_FRAME_CACHE_LOADED = False
    if reset_memory:
        with _FRAME_CACHE_LOCK:
            _FRAME_CACHE.clear()
            _INCOMPLETE_DISK_KEYS.clear()
        for key in _SEMANTIC_FRAME_CACHE_STATS:
            _SEMANTIC_FRAME_CACHE_STATS[key] = 0.0 if key.endswith("_seconds") else 0


def _load_disk_frame_cache() -> None:
    global _DISK_FRAME_CACHE_LOADED
    if _DISK_FRAME_CACHE_LOADED:
        return
    _DISK_FRAME_CACHE_LOADED = True
    _SEMANTIC_FRAME_CACHE_STATS["semantic_frame_cache_loaded"] = 1
    if not os.path.exists(_DISK_FRAME_CACHE_PATH):
        return
    try:
        with open(_DISK_FRAME_CACHE_PATH, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                    if item.get("schema_version") != SEMANTIC_FRAME_CACHE_SCHEMA_VERSION:
                        _SEMANTIC_FRAME_CACHE_STATS["semantic_frame_cache_invalid"] += 1
                        continue
                    key = item.get("cache_key")
                    value = item.get("value")
                    complete, missing = _cache_frame_completeness(value)
                    if key and complete:
                        with _FRAME_CACHE_LOCK:
                            _FRAME_CACHE[("disk", key)] = _normalize_frame(value)
                    else:
                        _SEMANTIC_FRAME_CACHE_STATS["semantic_frame_cache_invalid"] += 1
                        if key:
                            _INCOMPLETE_DISK_KEYS[key] = missing
                except Exception:
                    _SEMANTIC_FRAME_CACHE_STATS["semantic_frame_cache_invalid"] += 1
                    continue
    except Exception:
        return


def _append_disk_frame_cache(cache_key: tuple, frame: dict) -> None:
    if not cache_key or len(cache_key) != 2:
        return
    try:
        os.makedirs(os.path.dirname(_DISK_FRAME_CACHE_PATH), exist_ok=True)
        with open(_DISK_FRAME_CACHE_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "schema_version": SEMANTIC_FRAME_CACHE_SCHEMA_VERSION,
                "cache_key": cache_key[1],
                "value": _normalize_frame(frame),
            }, ensure_ascii=True) + "\n")
        _SEMANTIC_FRAME_CACHE_STATS["semantic_frame_cache_write_count"] += 1
    except Exception:
        return
