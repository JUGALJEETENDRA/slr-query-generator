from semantic_frame import extract_research_question_frame


def extract_rq(
    research_question,
    model="qwen2.5:3b",
    inference_engine=None,
):
    frame = extract_research_question_frame(
        research_question=research_question,
        model=model,
        inference_engine=inference_engine,
    )
    return {
        "technology": frame.get("intervention_or_method", ""),
        "task": frame.get("target_problem_or_task", ""),
        "evidence": frame.get("evidence_type", ""),
        "semantic_frame": frame,
    }
