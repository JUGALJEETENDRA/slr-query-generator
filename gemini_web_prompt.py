from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from local_ai.evidence import build_evidence_units


@dataclass(frozen=True)
class ScreeningPaper:
    paper_id: str
    title: str
    abstract: str


def build_structured_batch_prompt(*, protocol: dict, papers: Iterable[ScreeningPaper], schema: dict) -> str:
    payload = [
        {
            "paper_id": paper.paper_id,
            "evidence_units": build_evidence_units(paper.title, paper.abstract),
        }
        for paper in papers
    ]
    return f"""
Continue the same systematic-review screening job. Independently assess every supplied paper against the immutable
protocol. Judge meaning and relationships, never keyword overlap. Inclusion MET means required evidence is present.
Exclusion MET means affirmative disqualifying evidence is present. Cite only evidence IDs belonging to that paper.
For a required inclusion, use NOT_MET only when the evidence affirmatively establishes a mismatch; missing or
unreported information is UNCLEAR, not NOT_MET. Never REJECT merely because the title or abstract omits a detail.
Use certainty HIGH only for a well-supported definitive result; BORDERLINE or LOW must flag genuine risk. Preserve
MAYBE when title and abstract cannot safely establish KEEP or REJECT. Do not let earlier papers affect this batch.

IMMUTABLE PROTOCOL:
{json.dumps(protocol, ensure_ascii=False)}

FIVE-OR-FEWER PAPER BATCH:
{json.dumps(payload, ensure_ascii=False)}

Return exactly one JSON object matching this schema, with one unique item for every submitted paper ID:
{json.dumps(schema, ensure_ascii=False)}
""".strip()


def build_structured_critic_prompt(
    *, protocol: dict, papers: Iterable[ScreeningPaper], prior: dict[str, dict], schema: dict
) -> str:
    payload = [
        {
            "paper_id": paper.paper_id,
            "evidence_units": build_evidence_units(paper.title, paper.abstract),
            # Supply only why a second look was requested. Hiding the primary
            # decision and rationale makes this a genuinely independent check.
            "review_flags": {
                "validation_errors": prior.get(paper.paper_id, {}).get("validation_errors", []),
                "contradictions": prior.get(paper.paper_id, {}).get("contradictions", []),
            },
        }
        for paper in papers
    ]
    return f"""
Act as an adversarial systematic-review critic, distinct from the primary screener. Reassess every paper from
scratch against the immutable protocol. Challenge absence-based REJECT decisions, weakly supported KEEP decisions,
and any prior uncertainty or validation error. The title and abstract evidence units are the only paper evidence.
The primary decision and rationale are deliberately hidden to prevent anchoring. For a required inclusion, use
NOT_MET only when evidence affirmatively establishes a mismatch; missing information is UNCLEAR. Return a complete
replacement assessment. Use MAYBE when neither definitive outcome is evidence-safe.

IMMUTABLE PROTOCOL:
{json.dumps(protocol, ensure_ascii=False)}

RISKY PAPER BATCH:
{json.dumps(payload, ensure_ascii=False)}

Return exactly one JSON object matching this schema, with one unique item for every submitted paper ID:
{json.dumps(schema, ensure_ascii=False)}
""".strip()


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
