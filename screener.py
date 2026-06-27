from semantic_frame import extract_semantic_frame
from semantic_comparator import compare_semantic_frames


def screen_paper(
    title,
    abstract,
    research_question,
    rq_frame=None,
    model="qwen2.5:7b",
    mode="local"
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

        return {
            "decision": comparison_result.get("decision", "ERROR"),
            "reason": comparison_result.get("reason", "Comparison failed"),
            "required_evidence": "",
            "paper_contribution": "",
        }

    except Exception as e:
        return {
            "decision": "PARSE_ERROR",
            "reason": str(e),
            "required_evidence": "",
            "paper_contribution": "",
        }