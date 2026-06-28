import json
from ollama_client import ask_ollama


def make_decision(
    research_question,
    paper_topic,
    paper_contribution,
    paper_task,
    model="qwen2.5:3b"
):
    prompt = f"""
Research Question:
{research_question}

Paper Topic:
{paper_topic}

Paper Contribution:
{paper_contribution}

Paper Task:
{paper_task}

Determine:

1. What evidence is required to answer the research question.
2. Whether the paper contribution provides that evidence.

Return ONLY JSON:

{{
  "required_evidence":"",
  "decision":"KEEP",
  "reason":""
}}

Decision must be:
KEEP
MAYBE
REJECT

No markdown.
"""

    response = ask_ollama(
        prompt,
        model=model
    )

    return json.loads(response)