from semantic_comparator import compare_semantic_frames

rq = {
    "primary_subject": "automating systematic literature reviews",
    "intervention_or_method": "large language models",
    "target_problem_or_task": "systematic literature reviews",
    "application_context": "academic research",
    "study_role": "tool/method paper",
}

paper = {
    "primary_subject": "role of generative AI in systematic literature reviews",
    "intervention_or_method": "generative AI tools",
    "target_problem_or_task": "conducting systematic literature reviews",
    "application_context": "research domain",
    "study_role": "systematic review",
}
print(compare_semantic_frames(rq, paper))