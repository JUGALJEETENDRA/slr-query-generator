from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PROMPT_VERSION = "gemini-web-fast-prompt-v1"
ARCHITECTURE_VERSION = "gemini-web-fast-v1"


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criterion_id: str = Field(min_length=1, max_length=24)
    text: str = Field(min_length=1, max_length=500)


class ScreeningRubric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_summary: str = Field(min_length=1, max_length=800)
    inclusion_criteria: list[RubricCriterion] = Field(min_length=1, max_length=40)
    exclusion_criteria: list[RubricCriterion] = Field(default_factory=list, max_length=40)
    interpretation_notes: list[str] = Field(default_factory=list, max_length=20)
    ambiguity_rules: list[str] = Field(default_factory=list, max_length=20)
    evidence_requirements: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def unique_ids(self) -> "ScreeningRubric":
        ids = [item.criterion_id.strip() for item in [*self.inclusion_criteria, *self.exclusion_criteria]]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion IDs must be unique")
        return self


class CriterionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criterion_id: str = Field(min_length=1, max_length=24)
    status: Literal["MET", "NOT_MET", "UNCLEAR"]


class ScreeningItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str = Field(min_length=1, max_length=120)
    decision: Literal["KEEP", "MAYBE", "REJECT"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=500)
    evidence_quote: str = Field(default="", max_length=600)
    inclusion_assessments: list[CriterionAssessment] = Field(max_length=40)
    exclusion_assessments: list[CriterionAssessment] = Field(max_length=40)
    risk_flags: list[str] = Field(default_factory=list, max_length=12)


class ScreeningBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[ScreeningItem] = Field(min_length=1, max_length=15)


def criterion_entries(value: str) -> list[str]:
    entries: list[str] = []
    for line in re.split(r"[;\r\n]+", str(value or "")):
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if cleaned:
            entries.append(cleaned[:500])
    return entries


def fallback_rubric(inclusion: str, exclusion: str) -> ScreeningRubric:
    inclusions = criterion_entries(inclusion) or ["The paper addresses the supplied research question."]
    exclusions = criterion_entries(exclusion)
    return ScreeningRubric(
        review_summary="Apply the researcher's criteria verbatim to each supplied title and abstract.",
        inclusion_criteria=[
            RubricCriterion(criterion_id=f"I{index}", text=text)
            for index, text in enumerate(inclusions, 1)
        ],
        exclusion_criteria=[
            RubricCriterion(criterion_id=f"E{index}", text=text)
            for index, text in enumerate(exclusions, 1)
        ],
        interpretation_notes=[],
        ambiguity_rules=["Use MAYBE when the supplied title and abstract are insufficient or ambiguous."],
        evidence_requirements=[],
    )


def protocol_prompt(question: str, context: str, inclusion: str, exclusion: str) -> str:
    return f"""You are compiling a domain-neutral title-and-abstract screening rubric.
Preserve every explicit researcher criterion. Do not invent new mandatory requirements.
Return strict JSON only. Criterion text should retain the researcher's explicit meaning and criterion IDs must be unique.

Research question:
{question}

Research context:
{context}

Inclusion criteria:
{inclusion}

Exclusion criteria:
{exclusion}

Required JSON schema:
{json.dumps(ScreeningRubric.model_json_schema(), ensure_ascii=False)}"""


def batch_prompt(
    *, question: str, context: str, inclusion: str, exclusion: str,
    rubric: ScreeningRubric, papers: list[dict[str, str]], verification: bool,
) -> str:
    blind = (
        "This is a fresh prediction-blind independent review. You have not been given any prior decision. "
        "For evidence_quote, copy one continuous span verbatim from the supplied title or abstract; "
        "never join separate fragments and never insert ellipses."
        if verification else
        "Deeply screen every supplied title and abstract."
    )
    return f"""You are a systematic-review title-and-abstract screener. {blind}
Judge what the source paper itself studies, not keyword overlap or incidental background mentions.
Do not invent unstated populations, methods, outcomes, settings, results, or study designs.
KEEP only when title/abstract evidence supports the required scope and no exclusion is affirmatively met.
REJECT only when evidence clearly fails a required condition or clearly meets an exclusion criterion.
Use MAYBE whenever evidence is incomplete, ambiguous, incidental, or insufficient. Prefer MAYBE over an unsupported definitive judgment.
Return exactly one result for every supplied paper_id. Evidence quotes must be exact bounded quotes from that paper's supplied title or abstract.
Keep each reason under 400 characters and each evidence quote under 500 characters.
Do not provide chain-of-thought, essays, Markdown, or any text outside one JSON object.

Research question:
{question}

Research context:
{context}

Original inclusion criteria:
{inclusion}

Original exclusion criteria:
{exclusion}

Compiled rubric:
{json.dumps(rubric.model_dump(mode='json'), ensure_ascii=False)}

Papers:
{json.dumps(papers, ensure_ascii=False)}

Required JSON schema:
{json.dumps(ScreeningBatchResult.model_json_schema(), ensure_ascii=False)}"""
