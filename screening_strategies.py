from __future__ import annotations

import json
from typing import Any

from ollama_client import ask_ollama
from screener import screen_paper


LITSYNC_WORKFLOW = "litsync_workflow"
DIRECT_AI = "direct_ai"
DEFAULT_SCREENING_STRATEGY = LITSYNC_WORKFLOW
PUBLIC_SCREENING_STRATEGIES = {LITSYNC_WORKFLOW}
SCREENING_STRATEGIES = PUBLIC_SCREENING_STRATEGIES | {DIRECT_AI}


def normalize_screening_strategy(strategy: str | None) -> str:
    value = (strategy or DEFAULT_SCREENING_STRATEGY).strip().lower()
    aliases = {
        "litsync": LITSYNC_WORKFLOW,
        "litsync workflow": LITSYNC_WORKFLOW,
        "workflow": LITSYNC_WORKFLOW,
    }
    return aliases.get(value, value if value in PUBLIC_SCREENING_STRATEGIES else DEFAULT_SCREENING_STRATEGY)


def strategy_requires_rq_frame(strategy: str | None) -> bool:
    return normalize_screening_strategy(strategy) == LITSYNC_WORKFLOW


def screen_candidate(
    *,
    title: str,
    abstract: str,
    research_question: str,
    strategy: str | None = None,
    rq_frame: dict[str, Any] | None = None,
    model: str = "qwen2.5:3b",
    mode: str = "local",
    inference_engine=None,
) -> dict[str, Any]:
    selected = normalize_screening_strategy(strategy)
    result = screen_paper(
        title=title,
        abstract=abstract,
        research_question=research_question,
        rq_frame=rq_frame,
        mode=mode,
        model=model,
        inference_engine=inference_engine,
    )
    return _with_strategy_metadata(result, LITSYNC_WORKFLOW)


def direct_ai_screen_paper(
    *,
    title: str,
    abstract: str,
    research_question: str,
    model: str = "qwen2.5:3b",
    inference_engine=None,
) -> dict[str, Any]:
    prompt = f"""
You are screening a paper for a Systematic Literature Review.
Act like an experienced systematic review reviewer.

Research Question:
{research_question}

Paper Title:
{title}

Paper Abstract:
{abstract}

Decide whether this paper should be included for the review question.

Decision rules:
- KEEP: directly relevant evidence for the research question. A paper can be KEEP even when it evaluates one specific method, model, dataset, intervention, or application that falls within a broader review question.
- MAYBE: plausibly relevant but the title and abstract do not provide enough detail to confirm fit.
- REJECT: clearly outside the population, domain, intervention, method, task, or outcome required by the research question.

Review guidance:
- Prioritize research relevance over exact keyword overlap.
- Do not reject a paper solely because it uses narrower terminology than the research question.
- Do not broaden beyond the research question's domain or task.
- Ground the rationale in the title and abstract.

Return ONLY JSON:

{{
  "decision": "KEEP | MAYBE | REJECT",
  "reason": "One concise sentence grounded in the title and abstract.",
  "confidence": 0.0
}}
"""

    try:
        ask = inference_engine.ask if inference_engine is not None else ask_ollama
        response = ask(prompt, model=model)
        parsed = json.loads(response)
        decision = _normalize_decision(parsed.get("decision"))
        confidence = _normalize_confidence(parsed.get("confidence"))
        reason = " ".join(str(parsed.get("reason", "")).strip().split())
        if not reason:
            reason = "Direct AI did not provide a rationale."

        return {
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
            "required_evidence": "",
            "paper_contribution": "",
            "metadata": {
                "screening_strategy": DIRECT_AI,
                "model": model,
            },
        }
    except Exception as exc:
        return {
            "decision": "PARSE_ERROR",
            "reason": str(exc),
            "confidence": 0.0,
            "required_evidence": "",
            "paper_contribution": "",
            "metadata": {
                "screening_strategy": DIRECT_AI,
                "model": model,
                "error": str(exc),
            },
        }


def _with_strategy_metadata(result: dict[str, Any], strategy: str) -> dict[str, Any]:
    output = dict(result)
    metadata = dict(output.get("metadata") or {})
    metadata["screening_strategy"] = strategy
    output["metadata"] = metadata
    output.setdefault("confidence", "")
    return output


def _normalize_decision(value: Any) -> str:
    decision = str(value or "MAYBE").strip().upper()
    return decision if decision in {"KEEP", "MAYBE", "REJECT"} else "MAYBE"


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _litsync_confidence(result: dict[str, Any]) -> float:
    technology_match = _as_float(result.get("technology_match"))
    task_match = _as_float(result.get("task_role_match") or result.get("task_match"))
    context_match = _as_float(result.get("context_match"))
    role_match = 1.0 if result.get("review_role_match") is True else 0.0
    confidence = (
        (0.40 * technology_match)
        + (0.35 * task_match)
        + (0.15 * context_match)
        + (0.10 * role_match)
    )
    return round(max(0.0, min(1.0, confidence)), 4)


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
