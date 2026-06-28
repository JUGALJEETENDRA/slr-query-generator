import json

from ollama_client import ask_ollama
from semantic_frame import extract_semantic_frame
from semantic_comparator import compare_semantic_frames


def _clean_reason(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _fallback_reason(decision, paper_frame, comparison_result):
    task = paper_frame.get("target_problem_or_task", "") or "the paper's stated task"
    method = paper_frame.get("intervention_or_method", "") or "the reported method"
    role = paper_frame.get("review_role", "") or paper_frame.get("study_role", "")
    task_score = float(comparison_result.get("task_role_match", 0.0) or 0.0)

    if decision == "KEEP":
        return (
            f"Included because the abstract indicates that {method} is used for "
            f"{task}, which aligns with the review question. "
            f"The semantic task-role match is {task_score:.2f}."
        )
    if decision == "MAYBE":
        return (
            f"Marked maybe because the abstract has partial relevance to {task}, "
            f"but the match to the review question is not strong enough for automatic inclusion. "
            f"The semantic task-role match is {task_score:.2f}."
        )
    if role == "technology_being_reviewed":
        return (
            "Excluded because the abstract appears to review the technology itself "
            "rather than evaluate it as evidence for the review question."
        )
    return (
        f"Excluded because the abstract's main task, {task}, does not sufficiently "
        "match the task required by the review question."
    )


def generate_screening_reason(
    title,
    abstract,
    research_question,
    decision,
    rq_frame,
    paper_frame,
    comparison_result,
    model="qwen2.5:3b",
):
    prompt = f"""
Research Question:
{research_question}

Paper Title:
{title}

Paper Abstract:
{abstract}

Decision:
{decision}

Research Question Semantic Frame:
{json.dumps(rq_frame, ensure_ascii=True)}

Paper Semantic Frame:
{json.dumps(paper_frame, ensure_ascii=True)}

Semantic Match Signals:
{json.dumps(comparison_result, ensure_ascii=True)}

Write the screening rationale for this decision.

Requirements:
- Base the rationale on the title and abstract.
- Explain why the paper is included, excluded, or marked maybe.
- Mention the strongest relevant task/evidence match or mismatch.
- Do not expose internal threshold names, rule labels, JSON keys, or raw scores unless they are essential.
- Keep it to one concise sentence.

Return ONLY JSON:

{{
  "reason": ""
}}
"""

    try:
        response = ask_ollama(prompt, model=model)
        parsed = json.loads(response)
        reason = _clean_reason(parsed.get("reason", ""))
        if reason:
            return reason
    except Exception:
        pass

    return _fallback_reason(decision, paper_frame, comparison_result)


def screen_paper(
    title,
    abstract,
    research_question,
    rq_frame=None,
    model="qwen2.5:3b",
    mode="local",
    generate_reason: bool = True,
):
    try:
        if rq_frame is None:
            rq_frame = extract_semantic_frame(
                title=research_question,
                abstract="",
                model=model,
            )

        paper_frame = extract_semantic_frame(
            title=title,
            abstract=abstract,
            model=model,
        )

        comparison_result = compare_semantic_frames(
            rq_frame,
            paper_frame,
        )
        decision = comparison_result.get("decision", "ERROR")
        if generate_reason:
            reason = generate_screening_reason(
                title=title,
                abstract=abstract,
                research_question=research_question,
                decision=decision,
                rq_frame=rq_frame,
                paper_frame=paper_frame,
                comparison_result=comparison_result,
                model=model,
            )
        else:
            reason = _fallback_reason(decision, paper_frame, comparison_result)

        return {
            "decision": decision,
            "reason": reason,
            "required_evidence": "",
            "paper_contribution": "",
            "paper_primary_subject": paper_frame.get("primary_subject", ""),
            "paper_intervention_or_method": paper_frame.get("intervention_or_method", ""),
            "paper_target_problem_or_task": paper_frame.get("target_problem_or_task", ""),
            "paper_application_context": paper_frame.get("application_context", ""),
            "paper_evidence_type": paper_frame.get("evidence_type", ""),
            "paper_study_role": paper_frame.get("study_role", ""),
            "paper_review_role": paper_frame.get("review_role", ""),
            "technology_match": comparison_result.get("technology_match", 0.0),
            "task_match": comparison_result.get("task_match", 0.0),
            "task_subject_match": comparison_result.get("task_subject_match", 0.0),
            "task_role_match": comparison_result.get("task_role_match", 0.0),
            "subject_match": comparison_result.get("subject_match", 0.0),
            "context_match": comparison_result.get("context_match", 0.0),
            "study_role_match": comparison_result.get("study_role_match", False),
            "review_role_match": comparison_result.get("review_role_match", False),
            "canonical_task_left": comparison_result.get("canonical_task_left", ""),
            "canonical_task_right": comparison_result.get("canonical_task_right", ""),
            "task_identity_match": comparison_result.get("task_identity_match", False),
        }

    except Exception as e:
        return {
            "decision": "PARSE_ERROR",
            "reason": str(e),
            "required_evidence": "",
            "paper_contribution": "",
            "paper_primary_subject": "",
            "paper_intervention_or_method": "",
            "paper_target_problem_or_task": "",
            "paper_application_context": "",
            "paper_evidence_type": "",
            "paper_study_role": "",
            "paper_review_role": "",
            "technology_match": 0.0,
            "task_match": 0.0,
            "task_subject_match": 0.0,
            "task_role_match": 0.0,
            "subject_match": 0.0,
            "context_match": 0.0,
            "study_role_match": False,
            "review_role_match": False,
            "canonical_task_left": "",
            "canonical_task_right": "",
            "task_identity_match": False,
        }
