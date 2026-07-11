from __future__ import annotations

import hashlib
import json
import os
import re
import time


EMPTY_LLM_JUDGE = {
    "llm_judge_decision": "",
    "llm_judge_relation": "",
    "llm_uses_ai_for_review_workflow": False,
    "llm_is_review_about_ai": False,
    "llm_workflow_tasks_detected": "",
    "llm_external_domain_tasks_detected": "",
    "llm_judge_reason": "",
    "workflow_evidence_quote": "",
    "external_domain_evidence_quote": "",
    "task_object": "",
    "task_object_type": "unclear",
    "relation_confidence": 0.0,
    "uncertainty_reason": "",
    "directional_judge_source": "none",
    "directional_relation": "",
    "directional_confidence": 0.0,
    "directional_uses_ai_for_review_workflow": False,
    "directional_is_review_about_ai_external_domain": False,
    "directional_reason": "",
    "llm_directional_judge_used": False,
    "llm_directional_judge_error": "",
    "llm_directional_cache_hit": False,
    "llm_directional_cache_source": "miss",
    "llm_directional_cache_key": "",
    "llm_directional_timing_seconds": 0.0,
    "llm_route": "",
    "llm_skip_reason": "",
    "llm_required_reason": "",
    "llm_gate_confidence": 0.0,
}

AI_TERMS = (
    "artificial intelligence", "ai", "machine learning", "ml", "large language model",
    "large language models", "llm", "llms", "generative ai", "chatgpt", "gpt",
)
REVIEW_TERMS = (
    "systematic review", "systematic literature review", "literature review",
    "evidence review", "slr", "review update",
)
WORKFLOW_TERMS = (
    "title/abstract screening", "title abstract screening", "citation screening",
    "study selection", "selection of studies", "study screening", "paper screening", "article screening",
    "abstract screening", "literature search", "search strategy", "search query",
    "query generation", "data extraction from studies", "data extraction",
    "pico extraction", "evidence synthesis", "risk of bias", "quality assessment",
    "systematic review automation", "automated systematic review", "review assistant",
    "research assistant for literature review", "slr automation",
    "systematic literature review automation", "literature review automation",
    "automating systematic reviews", "automating systematic literature reviews",
    "automate systematic reviews", "automate systematic literature reviews",
    "automating literature reviews", "automate literature reviews",
    "automate and facilitate the review process", "facilitate the review process",
    "automating the review process", "automate the review process",
    "review process automation", "llms for systematic reviews",
    "ai-assisted systematic review", "generative ai for systematic reviews",
    "accelerating systematic reviews", "accelerate systematic reviews",
    "accelerating systematic literature reviews", "accelerate systematic literature reviews",
    "conducting systematic reviews",
    "systematic review workflow", "review pipeline", "systematic review updates",
    "inclusion/exclusion classification", "reviewer workload reduction",
    "screening phase", "identification of relevant articles",
    "llm-assisted methodology", "llm-assisted methodology for systematic reviews",
)
GENERIC_REVIEW_PROCEDURE_TERMS = (
    "data extraction", "evidence synthesis", "quality assessment", "risk of bias",
)
WORKFLOW_AUTOMATION_CUES = (
    "automate", "automated", "automation", "ai-assisted", "llm-assisted",
    "llm-based", "ai-based", "using llm", "using llms", "using artificial intelligence",
    "with llm", "with llms", "large language models for", "ai tool",
)
EXTERNAL_DOMAIN_TERMS = (
    "diagnosis", "medicine", "medical", "clinical", "patient", "cancer",
    "breast cancer", "fintech", "finance", "education", "student",
    "software refactoring", "software engineering", "cybersecurity",
    "drug discovery", "healthcare", "disease", "prediction", "diagnostic",
    "students", "software artifacts", "security events", "products", "images",
)
PROMPT_VERSION = "directional_v2"
SCHEMA_VERSION = "directional_schema_v2"
_DIRECTIONAL_CACHE: dict[str, dict] = {}
_DISK_CACHE_LOADED = False
_CACHE_LOADED_ENTRIES = 0
_CACHE_HITS = 0
_CACHE_MISSES = 0
_CACHE_WRITES = 0
CACHE_PATH = os.path.join("outputs", "cache", "llm_directional_cache.jsonl")


def should_run_llm_judge(deterministic_decision: str, model_scores: dict) -> bool:
    if deterministic_decision == "MAYBE":
        return True
    if deterministic_decision == "REJECT" and model_scores.get("model_positive_score", 0.0) >= 0.65:
        return True
    if deterministic_decision == "KEEP" and model_scores.get("model_negative_score", 0.0) >= 0.70:
        return True
    return False


def directional_trigger_diagnostics(
    *,
    rq_frame: dict,
    deterministic_result: dict,
    title: str,
    abstract: str,
    aggressive_gating: bool = False,
) -> dict:
    rq_type = str(
        rq_frame.get("review_question_type")
        or rq_frame.get("question_type")
        or rq_frame.get("rq_type")
        or ""
    )
    desired_relation = str(rq_frame.get("rq_desired_relation") or "")
    review_rq = rq_type == "review_workflow_automation" or desired_relation == "tool_used_for_workflow"
    text = f"{title}\n{abstract}".lower()
    ai_hit = _contains_any(text, AI_TERMS)
    review_hit = _contains_any(text, REVIEW_TERMS)
    workflow_hits = _workflow_hits(text)
    external_hits = [term for term in EXTERNAL_DOMAIN_TERMS if term in text]
    ai_review_candidate = bool(review_rq and ai_hit and review_hit)
    positive_workflow = bool(workflow_hits)
    decision = str(deterministic_result.get("decision", "")).upper()
    relation_unclear = bool(
        not deterministic_result.get("relation_match")
        or deterministic_result.get("relation_unclear")
        or deterministic_result.get("paper_observed_relation") in {"unclear_relation", ""}
    )
    score = 0.0
    reasons = []
    if not review_rq:
        return {
            **_trigger_base(False, False, workflow_hits, external_hits, score),
            "llm_directional_skipped_reason": "not_review_workflow_automation_rq",
        }
    if ai_review_candidate:
        score += 0.35
        reasons.append("ai_review_candidate")
    if positive_workflow:
        score += 0.45
        reasons.append("positive_workflow_candidate")
    if decision == "MAYBE":
        score += 0.25
        reasons.append("deterministic_maybe")
    if decision == "REJECT" and ai_review_candidate:
        score += 0.25
        reasons.append("reject_with_ai_review_terms")
    if decision == "KEEP" and relation_unclear:
        score += 0.20
        reasons.append("keep_relation_not_confirmed")
    if relation_unclear:
        score += 0.10
        reasons.append("relation_unclear")
    legacy_triggered = score >= 0.35
    if external_hits and ai_hit and review_hit:
        score += 0.20
        reasons.append("external_domain_candidate")
    output = _trigger_base(ai_review_candidate, positive_workflow, workflow_hits, external_hits, score)
    if not aggressive_gating:
        output["llm_directional_triggered"] = legacy_triggered
        output["llm_directional_trigger_reason"] = "; ".join(reasons) if legacy_triggered else ""
        output["llm_directional_skipped_reason"] = "" if legacy_triggered else "candidate_score_below_threshold"
        output["llm_route"] = "legacy_current_mode"
        output["llm_required_reason"] = output["llm_directional_trigger_reason"]
        output["llm_gate_confidence"] = round(min(1.0, score), 4)
        return output
    route, confidence, skip_reason, required_reason = _llm_route(
        decision=decision,
        ai_hit=ai_hit,
        review_hit=review_hit,
        positive_workflow=positive_workflow,
        external_hits=external_hits,
        relation_unclear=relation_unclear,
        deterministic_result=deterministic_result,
    )
    triggered = route.startswith("llm_required")
    output["llm_directional_triggered"] = triggered
    output["llm_directional_trigger_reason"] = "; ".join(reasons) if triggered else ""
    output["llm_directional_skipped_reason"] = skip_reason
    output["llm_route"] = route
    output["llm_skip_reason"] = skip_reason
    output["llm_required_reason"] = required_reason
    output["llm_gate_confidence"] = confidence
    if route == "skipped_high_confidence_workflow":
        output.update(_deterministic_directional_output(
            "ai_tool_for_review_workflow",
            confidence,
            "Strong review-workflow task object and AI/tool terms were detected deterministically.",
        ))
    elif route == "skipped_high_confidence_external":
        output.update(_deterministic_directional_output(
            "review_about_ai_external_domain",
            confidence,
            "External-domain AI review terms were detected without review-workflow task evidence.",
        ))
    return output


def judge_with_llm(title: str, abstract: str, research_question: str, model: str, inference_engine=None) -> dict:
    global _CACHE_HITS, _CACHE_MISSES
    started_at = time.perf_counter()
    _load_disk_cache()
    cache_key = _cache_key(research_question, title, abstract, model)
    if cache_key in _DIRECTIONAL_CACHE:
        _CACHE_HITS += 1
        cached = dict(_DIRECTIONAL_CACHE[cache_key])
        cached["llm_directional_cache_hit"] = True
        cached["llm_directional_cache_key"] = cache_key
        cached["llm_directional_cache_source"] = cached.get("llm_directional_cache_source") or "memory"
        cached["llm_directional_timing_seconds"] = round(time.perf_counter() - started_at, 4)
        return cached
    _CACHE_MISSES += 1

    prompt = f"""
Return ONLY strict JSON for this screening judgement.

You are deciding relation direction for a systematic literature review automation question.

Positive relation:
AI/LLM is used as a tool to perform systematic review workflow tasks such as
title/abstract screening, citation screening, study selection, literature search,
data extraction from studies, risk of bias assessment, evidence synthesis, or review assistant tooling.
This remains workflow-use even when the systematic review topic/domain is medicine,
education, software engineering, finance, livestock, or another field.

Negative relation:
The paper is a systematic review about AI/LLM applications in an external domain,
where AI is the subject being reviewed, such as diagnosis, medicine, fintech,
education, software refactoring, cybersecurity, drug discovery, or other domain tasks.

Prioritize the task object:
- If the object is studies, papers, abstracts, citations, reviewers, review stages,
  included studies, literature search, or the systematic review process, classify workflow-use.
- If the object is patients, images, diseases, products, students, software artifacts,
  security events, financial transactions, or other domain objects, classify external-domain.

Do not classify a paper as workflow-use just because the paper is a systematic review
and mentions AI. The abstract must say AI/LLM is used to conduct or automate review tasks.

Examples:
- "LLMs for title/abstract screening in systematic reviews" => ai_tool_for_review_workflow
- "screening automation for systematic reviews" => ai_tool_for_review_workflow
- "research assistant for literature review with LLMs" => ai_tool_for_review_workflow
- "ML support selection of studies for systematic review updates" => ai_tool_for_review_workflow
- "LLM to accelerate systematic reviews" => ai_tool_for_review_workflow
- "LLMs in breast cancer diagnosis: a systematic review" => review_about_ai_external_domain
- "AI in software refactoring: systematic review" => review_about_ai_external_domain

Research question:
{research_question}

Title:
{title}

Abstract:
{abstract}

JSON schema:
{{
  "decision": "KEEP|MAYBE|REJECT",
  "relation": "ai_tool_for_review_workflow|review_about_ai_external_domain|unclear",
  "uses_ai_for_review_workflow": true,
  "is_review_about_ai_external_domain": false,
  "confidence": 0.0,
  "workflow_tasks_detected": [],
  "external_domain_tasks_detected": [],
  "workflow_evidence": "",
  "external_domain_evidence": "",
  "workflow_evidence_quote": "",
  "external_domain_evidence_quote": "",
  "task_object": "",
  "task_object_type": "review_object|external_domain_object|unclear",
  "relation_confidence": 0.0,
  "uncertainty_reason": "",
  "reason": ""
}}
""".strip()
    try:
        from ollama_client import ask_ollama

        ask = inference_engine.ask if inference_engine is not None else ask_ollama
        parsed = json.loads(ask(prompt, model=model))
        relation = _normalize_relation(str(parsed.get("relation", "")).strip())
        uses_workflow = bool(parsed.get("uses_ai_for_review_workflow"))
        external_review = bool(
            parsed.get("is_review_about_ai_external_domain")
            or parsed.get("is_systematic_review_about_ai")
        )
        confidence = _as_confidence(parsed.get("confidence"))
        relation_confidence = _as_confidence(parsed.get("relation_confidence", confidence))
        decision = str(parsed.get("decision", "")).upper()
        workflow_quote = str(parsed.get("workflow_evidence_quote") or parsed.get("workflow_evidence") or "")
        external_quote = str(parsed.get("external_domain_evidence_quote") or parsed.get("external_domain_evidence") or "")
        reason = str(
            parsed.get("reason")
            or workflow_quote
            or external_quote
            or parsed.get("evidence_quote")
            or ""
        )
        task_object_type = str(parsed.get("task_object_type") or "unclear").strip()
        output = {
            "llm_judge_decision": decision,
            "llm_judge_relation": relation,
            "llm_uses_ai_for_review_workflow": uses_workflow,
            "llm_is_review_about_ai": external_review,
            "llm_workflow_tasks_detected": "; ".join(parsed.get("workflow_tasks_detected") or []),
            "llm_external_domain_tasks_detected": "; ".join(parsed.get("external_domain_tasks_detected") or []),
            "llm_judge_reason": reason,
            "workflow_evidence_quote": workflow_quote,
            "external_domain_evidence_quote": external_quote,
            "task_object": str(parsed.get("task_object") or ""),
            "task_object_type": task_object_type if task_object_type in {"review_object", "external_domain_object", "unclear"} else "unclear",
            "relation_confidence": relation_confidence,
            "uncertainty_reason": str(parsed.get("uncertainty_reason") or ""),
            "directional_judge_source": "llm",
            "directional_relation": relation,
            "directional_confidence": relation_confidence or confidence,
            "directional_uses_ai_for_review_workflow": uses_workflow,
            "directional_is_review_about_ai_external_domain": external_review,
            "directional_reason": reason,
            "llm_directional_judge_used": True,
            "llm_directional_judge_error": "",
            "llm_directional_cache_hit": False,
            "llm_directional_cache_source": "miss",
            "llm_directional_cache_key": cache_key,
            "llm_directional_timing_seconds": round(time.perf_counter() - started_at, 4),
        }
        _DIRECTIONAL_CACHE[cache_key] = dict(output)
        _append_disk_cache(cache_key, output)
        return output
    except Exception as exc:
        output = dict(EMPTY_LLM_JUDGE)
        output["llm_judge_reason"] = f"LLM judge unavailable: {exc}"
        output["llm_directional_judge_error"] = str(exc)
        output["llm_directional_cache_key"] = cache_key
        output["llm_directional_timing_seconds"] = round(time.perf_counter() - started_at, 4)
        return output


def judge_batch_with_llm(records: list[dict], research_question: str, model: str, inference_engine=None) -> dict[str, dict]:
    """Optional batch directional judge. Returns per-record diagnostics keyed by id."""
    if not records:
        return {}
    _load_disk_cache()
    output: dict[str, dict] = {}
    uncached = []
    for record in records:
        record_id = str(record.get("id") or "")
        cache_key = _cache_key(
            research_question,
            str(record.get("title") or ""),
            str(record.get("abstract") or ""),
            model,
        )
        if cache_key in _DIRECTIONAL_CACHE:
            cached = dict(_DIRECTIONAL_CACHE[cache_key])
            cached["llm_directional_cache_hit"] = True
            cached["llm_directional_cache_key"] = cache_key
            cached["llm_directional_cache_source"] = cached.get("llm_directional_cache_source") or "memory"
            output[record_id] = cached
        else:
            uncached.append((record, cache_key))
    if not uncached:
        return output

    prompt = _batch_prompt(research_question, [record for record, _ in uncached])
    try:
        from ollama_client import ask_ollama

        ask = inference_engine.ask if inference_engine is not None else ask_ollama
        parsed_items = json.loads(ask(prompt, model=model))
        if not isinstance(parsed_items, list):
            raise ValueError("Batch LLM response was not a JSON list.")
        by_id = {str(item.get("id") or ""): item for item in parsed_items if isinstance(item, dict)}
        for record, cache_key in uncached:
            record_id = str(record.get("id") or "")
            item = by_id.get(record_id)
            if item is None:
                output[record_id] = judge_with_llm(
                    str(record.get("title") or ""),
                    str(record.get("abstract") or ""),
                    research_question,
                    model,
                    inference_engine,
                )
                continue
            normalized = _normalize_parsed_judge(item, cache_key, source="llm_batch")
            _DIRECTIONAL_CACHE[cache_key] = dict(normalized)
            _append_disk_cache(cache_key, normalized)
            output[record_id] = normalized
        return output
    except Exception:
        for record, _ in uncached:
            record_id = str(record.get("id") or "")
            output[record_id] = judge_with_llm(
                str(record.get("title") or ""),
                str(record.get("abstract") or ""),
                research_question,
                model,
                inference_engine,
            )
        return output


def _batch_prompt(research_question: str, records: list[dict]) -> str:
    safe_records = [
        {
            "id": str(record.get("id") or ""),
            "title": str(record.get("title") or ""),
            "abstract": str(record.get("abstract") or "")[:1800],
        }
        for record in records
    ]
    return f"""
Return ONLY a strict JSON list for this screening judgement.

Use the same directional task as the single-paper judge:
- workflow-use means AI/LLM performs systematic review tasks such as screening, search, extraction, synthesis, or review assistant work.
- external-domain means the paper is a systematic review about AI/LLM applications in a non-review task/domain.

Research question:
{research_question}

Records:
{json.dumps(safe_records, ensure_ascii=True)}

JSON item schema:
{{
  "id": "...",
  "relation": "ai_tool_for_review_workflow|review_about_ai_external_domain|unclear",
  "uses_ai_for_review_workflow": true,
  "is_review_about_ai_external_domain": false,
  "workflow_tasks_detected": [],
  "external_domain_tasks_detected": [],
  "workflow_evidence_quote": "",
  "external_domain_evidence_quote": "",
  "task_object": "",
  "task_object_type": "review_object|external_domain_object|unclear",
  "confidence": 0.0,
  "decision_hint": "KEEP|MAYBE|REJECT",
  "reason": ""
}}
""".strip()


def _normalize_parsed_judge(parsed: dict, cache_key: str, source: str = "llm") -> dict:
    relation = _normalize_relation(str(parsed.get("relation", "")).strip())
    uses_workflow = bool(parsed.get("uses_ai_for_review_workflow"))
    external_review = bool(
        parsed.get("is_review_about_ai_external_domain")
        or parsed.get("is_systematic_review_about_ai")
    )
    confidence = _as_confidence(parsed.get("relation_confidence", parsed.get("confidence")))
    decision = str(parsed.get("decision") or parsed.get("decision_hint") or "").upper()
    workflow_quote = str(parsed.get("workflow_evidence_quote") or parsed.get("workflow_evidence") or "")
    external_quote = str(parsed.get("external_domain_evidence_quote") or parsed.get("external_domain_evidence") or "")
    reason = str(parsed.get("reason") or workflow_quote or external_quote or "")
    task_object_type = str(parsed.get("task_object_type") or "unclear").strip()
    return {
        "llm_judge_decision": decision,
        "llm_judge_relation": relation,
        "llm_uses_ai_for_review_workflow": uses_workflow,
        "llm_is_review_about_ai": external_review,
        "llm_workflow_tasks_detected": "; ".join(parsed.get("workflow_tasks_detected") or []),
        "llm_external_domain_tasks_detected": "; ".join(parsed.get("external_domain_tasks_detected") or []),
        "llm_judge_reason": reason,
        "workflow_evidence_quote": workflow_quote,
        "external_domain_evidence_quote": external_quote,
        "task_object": str(parsed.get("task_object") or ""),
        "task_object_type": task_object_type if task_object_type in {"review_object", "external_domain_object", "unclear"} else "unclear",
        "relation_confidence": confidence,
        "uncertainty_reason": str(parsed.get("uncertainty_reason") or ""),
        "directional_judge_source": source,
        "directional_relation": relation,
        "directional_confidence": confidence,
        "directional_uses_ai_for_review_workflow": uses_workflow,
        "directional_is_review_about_ai_external_domain": external_review,
        "directional_reason": reason,
        "llm_directional_judge_used": True,
        "llm_directional_judge_error": "",
        "llm_directional_cache_hit": False,
        "llm_directional_cache_source": "miss",
        "llm_directional_cache_key": cache_key,
        "llm_directional_timing_seconds": 0.0,
    }


def _as_confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _contains_any(text: str, terms) -> bool:
    return any(term in text for term in terms)


def _workflow_hits(text: str) -> list[str]:
    hits = []
    for term in WORKFLOW_TERMS:
        if term not in text:
            continue
        if term in GENERIC_REVIEW_PROCEDURE_TERMS and not _generic_workflow_term_is_automated(text, term):
            continue
        hits.append(term)
    return hits


def _generic_workflow_term_is_automated(text: str, term: str) -> bool:
    for match in re.finditer(re.escape(term), text):
        start = max(0, match.start() - 90)
        end = min(len(text), match.end() + 90)
        if _contains_any(text[start:end], WORKFLOW_AUTOMATION_CUES):
            return True
    return False


def _trigger_base(ai_review_candidate, positive_workflow, workflow_hits, external_hits=None, score=0.0):
    external_hits = external_hits or []
    return {
        "llm_directional_triggered": False,
        "llm_directional_trigger_reason": "",
        "llm_directional_skipped_reason": "",
        "llm_directional_candidate_score": round(min(1.0, score), 4),
        "ai_review_candidate_detected": bool(ai_review_candidate),
        "positive_workflow_candidate_detected": bool(positive_workflow),
        "positive_workflow_candidate_terms": "; ".join(dict.fromkeys(workflow_hits)),
        "external_domain_candidate_detected": bool(external_hits),
        "external_domain_candidate_terms": "; ".join(dict.fromkeys(external_hits)),
        "llm_route": "",
        "llm_skip_reason": "",
        "llm_required_reason": "",
        "llm_gate_confidence": 0.0,
    }


def _llm_route(
    *,
    decision: str,
    ai_hit: bool,
    review_hit: bool,
    positive_workflow: bool,
    external_hits: list[str],
    relation_unclear: bool,
    deterministic_result: dict,
) -> tuple[str, float, str, str]:
    if not ai_hit and not review_hit:
        return "skipped_irrelevant", 0.95, "No AI/tool or review terms detected.", ""
    if not ai_hit:
        return "skipped_irrelevant", 0.90, "No AI/LLM/automation/tool terms detected.", ""
    if not review_hit and not positive_workflow:
        return "skipped_irrelevant", 0.90, "No systematic/literature-review context detected.", ""
    if positive_workflow and not external_hits and decision in {"KEEP", "MAYBE"} and not relation_unclear:
        return "skipped_high_confidence_workflow", 0.90, "Deterministic workflow relation is clear.", ""
    if positive_workflow and not external_hits and decision == "KEEP":
        return "skipped_high_confidence_workflow", 0.85, "Strong workflow terms support deterministic KEEP.", ""
    if external_hits and review_hit and not positive_workflow:
        return "skipped_high_confidence_external", 0.88, "External-domain review is clear and no review-workflow object was detected.", ""
    if decision == "MAYBE":
        return "llm_required_uncertain", 0.55, "", "Deterministic decision is MAYBE."
    if relation_unclear:
        return "llm_required_uncertain", 0.55, "", "Deterministic relation is unclear."
    if decision == "KEEP" and deterministic_result.get("relation_evidence_strength") in (None, "", 0):
        return "llm_required_boundary", 0.50, "", "KEEP has weak relation evidence validation."
    if decision == "REJECT" and ai_hit and review_hit and positive_workflow:
        return "llm_required_boundary", 0.60, "", "REJECT has AI+review+workflow candidate evidence."
    if external_hits and positive_workflow:
        return "llm_required_conflict", 0.50, "", "Workflow and external-domain signals conflict."
    return "skipped_irrelevant", 0.75, "No boundary or conflict signal requiring LLM.", ""


def _deterministic_directional_output(relation: str, confidence: float, reason: str) -> dict:
    workflow = relation == "ai_tool_for_review_workflow"
    external = relation == "review_about_ai_external_domain"
    return {
        "directional_judge_source": "deterministic_gate",
        "directional_relation": relation,
        "directional_confidence": confidence,
        "directional_uses_ai_for_review_workflow": workflow,
        "directional_is_review_about_ai_external_domain": external,
        "directional_reason": reason,
    }


def _cache_key(research_question: str, title: str, abstract: str, model: str) -> str:
    normalized_abstract = _normalize_key_text(abstract)
    abstract_hash = hashlib.sha1(normalized_abstract.encode("utf-8", errors="ignore")).hexdigest()
    text = "\n".join([
        PROMPT_VERSION,
        SCHEMA_VERSION,
        _normalize_key_text(model),
        _normalize_key_text(research_question),
        _normalize_key_text(title),
        abstract_hash,
    ])
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_key_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _load_disk_cache() -> None:
    global _DISK_CACHE_LOADED, _CACHE_LOADED_ENTRIES
    if _DISK_CACHE_LOADED:
        return
    _DISK_CACHE_LOADED = True
    if not os.path.exists(CACHE_PATH):
        return
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                    key = item.get("cache_key")
                    value = item.get("value")
                    if key and isinstance(value, dict):
                        value["llm_directional_cache_source"] = "disk"
                        _DIRECTIONAL_CACHE[key] = value
                        _CACHE_LOADED_ENTRIES += 1
                except Exception:
                    continue
    except Exception:
        return


def _append_disk_cache(cache_key: str, value: dict) -> None:
    global _CACHE_WRITES
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"cache_key": cache_key, "value": value}, ensure_ascii=True) + "\n")
        _CACHE_WRITES += 1
    except Exception:
        return


def get_cache_stats() -> dict:
    _load_disk_cache()
    return {
        "cache_loaded_entries": _CACHE_LOADED_ENTRIES,
        "cache_memory_entries": len(_DIRECTIONAL_CACHE),
        "cache_hits": _CACHE_HITS,
        "cache_misses": _CACHE_MISSES,
        "cache_write_count": _CACHE_WRITES,
        "cache_file_path": CACHE_PATH,
    }


def clear_cache() -> None:
    global _DISK_CACHE_LOADED, _CACHE_LOADED_ENTRIES, _CACHE_HITS, _CACHE_MISSES, _CACHE_WRITES
    _DIRECTIONAL_CACHE.clear()
    _DISK_CACHE_LOADED = False
    _CACHE_LOADED_ENTRIES = 0
    _CACHE_HITS = 0
    _CACHE_MISSES = 0
    _CACHE_WRITES = 0
    try:
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
    except Exception:
        return


def _normalize_relation(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    aliases = {
        "is_review_about_ai_external_domain": "review_about_ai_external_domain",
        "review_about_ai": "review_about_ai_external_domain",
        "ai_tool_for_review": "ai_tool_for_review_workflow",
    }
    return aliases.get(normalized, normalized)
