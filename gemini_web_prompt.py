from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ScreeningPaper:
    paper_id: str
    title: str
    abstract: str


def build_screening_prompt(
    *,
    research_question: str,
    inclusion_criteria: str = "",
    exclusion_criteria: str = "",
    papers: Iterable[ScreeningPaper],
) -> str:
    payload = [
        {
            "id": paper.paper_id,
            "title": paper.title,
            "abstract": paper.abstract,
        }
        for paper in papers
    ]

    return f"""
You are screening titles and abstracts for a systematic literature review.

Research Question:
{research_question}

Inclusion Criteria:
{inclusion_criteria or "Include papers that directly provide evidence relevant to the research question."}

Exclusion Criteria:
{exclusion_criteria or "Exclude papers that are outside the population, domain, intervention, method, task, or outcome required by the research question."}

Papers:
{json.dumps(payload, ensure_ascii=True, indent=2)}

For each paper, decide whether it should be included for the review.

Decision labels:
- Include: directly relevant evidence for the research question.
- Maybe: plausibly relevant, but title and abstract do not provide enough detail.
- Exclude: clearly outside the review question.

Return ONLY valid JSON with this shape:
{{
  "decisions": [
    {{
      "id": "same paper id",
      "decision": "Include | Exclude | Maybe",
      "reason": "One concise sentence grounded in the title and abstract."
    }}
  ]
}}
""".strip()
