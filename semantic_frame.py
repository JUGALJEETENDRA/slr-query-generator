import json
from threading import Lock
from ollama_client import ask_ollama


SEMANTIC_FRAME_FIELDS = (
    "primary_subject",
    "intervention_or_method",
    "target_problem_or_task",
    "application_context",
    "evidence_type",
    "study_role",
    "review_role",   # new field
)

_FRAME_CACHE = {}
_FRAME_CACHE_LOCK = Lock()


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


def extract_semantic_frame(
    title,
    abstract,
    model="qwen2.5:3b",
    inference_engine=None,
):
    cache_key = (
        getattr(inference_engine, "engine_id", "local"),
        str(model or ""),
        " ".join(str(title or "").split()),
        " ".join(str(abstract or "").split()),
    )
    with _FRAME_CACHE_LOCK:
        cached = _FRAME_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

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
    response = ask(prompt, model=model)

    frame = _normalize_frame(json.loads(response))
    with _FRAME_CACHE_LOCK:
        _FRAME_CACHE[cache_key] = dict(frame)
    return frame
