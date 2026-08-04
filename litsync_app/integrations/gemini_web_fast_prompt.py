from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Prompt v2 invalidates old v1 checkpoints because prompt version is part of
# the protocol/checkpoint identity.
PROMPT_VERSION = "gemini-web-fast-prompt-v2"

# Keep the architecture version unchanged for this focused quality correction.
# Scheduling and browser architecture will be handled separately.
ARCHITECTURE_VERSION = "gemini-web-fast-v1"


CriterionRole = Literal[
    "MANDATORY",
    "ALTERNATIVE",
    "OPTIONAL",
    "EXCLUSION",
    "UNRESOLVED",
]

GroupOperator = Literal[
    "ALL",
    "ANY",
    "AT_LEAST",
]


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str = Field(min_length=1, max_length=32)
    text: str = Field(min_length=1, max_length=4000)

    # These fields preserve the researcher's original logical meaning.
    # Defaults retain compatibility with older/fake test responses.
    role: CriterionRole = "UNRESOLVED"
    group_id: str = Field(default="", max_length=32)
    source_text: str = Field(default="", max_length=4000)


class CriterionGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: str = Field(min_length=1, max_length=32)
    operator: GroupOperator
    member_ids: list[str] = Field(min_length=1, max_length=40)
    minimum_required: int = Field(default=1, ge=1, le=40)
    description: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_group(self) -> "CriterionGroup":
        if len(self.member_ids) != len(set(self.member_ids)):
            raise ValueError("criterion group member IDs must be unique")

        if self.operator == "ANY" and self.minimum_required != 1:
            raise ValueError("ANY groups must have minimum_required=1")

        if self.operator == "ALL" and self.minimum_required != len(self.member_ids):
            raise ValueError(
                "ALL groups must require every listed member"
            )

        if self.minimum_required > len(self.member_ids):
            raise ValueError(
                "minimum_required cannot exceed the number of group members"
            )

        return self


class ScreeningRubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_summary: str = Field(min_length=1, max_length=1200)

    inclusion_criteria: list[RubricCriterion] = Field(
        min_length=1,
        max_length=40,
    )

    exclusion_criteria: list[RubricCriterion] = Field(
        default_factory=list,
        max_length=40,
    )

    criterion_groups: list[CriterionGroup] = Field(
        default_factory=list,
        max_length=20,
    )

    interpretation_notes: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    ambiguity_rules: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    evidence_requirements: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    original_logic_preserved: bool = True

    @model_validator(mode="after")
    def validate_rubric(self) -> "ScreeningRubric":
        criteria = [
            *self.inclusion_criteria,
            *self.exclusion_criteria,
        ]

        criterion_ids = [
            item.criterion_id.strip()
            for item in criteria
        ]

        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("criterion IDs must be unique")

        group_ids = [
            group.group_id.strip()
            for group in self.criterion_groups
        ]

        if len(group_ids) != len(set(group_ids)):
            raise ValueError("criterion group IDs must be unique")

        known_criterion_ids = set(criterion_ids)

        for group in self.criterion_groups:
            unknown_ids = [
                member_id
                for member_id in group.member_ids
                if member_id not in known_criterion_ids
            ]

            if unknown_ids:
                raise ValueError(
                    "criterion group references unknown criterion IDs: "
                    + ", ".join(unknown_ids)
                )

        return self


class CriterionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str = Field(min_length=1, max_length=32)
    status: Literal["MET", "NOT_MET", "UNCLEAR"]


class ScreeningItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str = Field(min_length=1, max_length=120)

    decision: Literal["KEEP", "MAYBE", "REJECT"]

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str = Field(
        min_length=1,
        max_length=500,
    )

    evidence_quote: str = Field(
        default="",
        max_length=600,
    )

    inclusion_assessments: list[CriterionAssessment] = Field(
        max_length=40,
    )

    exclusion_assessments: list[CriterionAssessment] = Field(
        max_length=40,
    )

    risk_flags: list[str] = Field(
        default_factory=list,
        max_length=12,
    )


class ScreeningBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ScreeningItem] = Field(
        min_length=1,
        max_length=15,
    )


def criterion_entries(value: str) -> list[str]:
    """
    Return line-level entries for input-preservation checks only.

    These entries are not a Boolean expression and must never be interpreted
    locally as though every line is an independently mandatory condition.
    """
    entries: list[str] = []

    for line in re.split(r"[;\r\n]+", str(value or "")):
        cleaned = re.sub(
            r"^\s*(?:[-*•]|\d+[.)])\s*",
            "",
            line,
        ).strip()

        if cleaned:
            entries.append(cleaned[:4000])

    return entries


def fallback_rubric(
    inclusion: str,
    exclusion: str,
) -> ScreeningRubric:
    """
    Safe deterministic fallback.

    Preserve each complete original criteria block as one authoritative
    natural-language condition. Do not split every bullet into an implicit
    mandatory AND requirement.
    """
    inclusion_text = str(inclusion or "").strip()
    exclusion_text = str(exclusion or "").strip()

    if not inclusion_text:
        inclusion_text = (
            "The paper must substantively address the supplied research "
            "question using the supplied title and abstract."
        )

    inclusion_criterion = RubricCriterion(
        criterion_id="I1",
        text=inclusion_text,
        role="UNRESOLVED",
        source_text=inclusion_text,
    )

    exclusion_criteria: list[RubricCriterion] = []

    if exclusion_text:
        exclusion_criteria.append(
            RubricCriterion(
                criterion_id="E1",
                text=exclusion_text,
                role="EXCLUSION",
                source_text=exclusion_text,
            )
        )

    return ScreeningRubric(
        review_summary=(
            "Apply the researcher's complete original criteria using their "
            "natural-language logical relationships."
        ),
        inclusion_criteria=[inclusion_criterion],
        exclusion_criteria=exclusion_criteria,
        criterion_groups=[],
        interpretation_notes=[
            (
                "Protocol compilation was unavailable. The complete original "
                "criteria blocks remain authoritative."
            ),
            (
                "Do not interpret line breaks or bullets as an implicit "
                "requirement that every listed example must be satisfied."
            ),
        ],
        ambiguity_rules=[
            (
                "Use MAYBE when the title and abstract are insufficient to "
                "resolve the original criteria safely."
            )
        ],
        evidence_requirements=[
            (
                "For KEEP and REJECT, copy exactly one continuous verbatim "
                "span from the supplied title or abstract."
            )
        ],
        original_logic_preserved=False,
    )


def protocol_prompt(
    question: str,
    context: str,
    inclusion: str,
    exclusion: str,
) -> str:
    return f"""You are compiling a domain-neutral title-and-abstract screening rubric.

Your task is to preserve the researcher's original screening logic exactly.

NON-NEGOTIABLE LOGIC RULES:

1. Preserve every explicit researcher condition.
2. Do not invent new mandatory requirements.
3. Do not broaden or narrow the requested review scope.
4. Do not assume every bullet or line is independently mandatory.
5. Preserve explicit logical relationships such as:
   - AND
   - OR
   - either
   - both
   - all
   - each
   - every
   - at least one
   - one or more
   - any of
6. Phrases such as "including", "such as", "for example", and "e.g." normally
   introduce examples or alternatives unless the researcher explicitly states
   that every listed item is required.
7. Optional application domains, methods, outcomes, populations, or evidence
   types must not become mandatory conditions.
8. Exclusion criteria should remain independently decisive when explicitly met.
9. When logical meaning is genuinely ambiguous, preserve the ambiguity in
   interpretation_notes instead of inventing an AND relationship.
10. Criterion text and source_text should preserve the researcher's original
    wording as closely as possible.
11. Return strict JSON only.

LOGIC EXAMPLES:

Example A

Original:
"The paper must evaluate at least one of privacy, robustness, communication
efficiency, or predictive performance."

Correct interpretation:
One ANY group containing four alternative members.

Incorrect interpretation:
Four independently mandatory criteria.

Example B

Original:
"Include simulations, experiments, prototypes, or formal analyses."

Correct interpretation:
These are alternative eligible evidence types unless surrounding text explicitly
requires more than one.

Example C

Original:
"The paper must use federated learning and must report quantitative evaluation."

Correct interpretation:
Two mandatory conditions combined using ALL.

Example D

Original:
"Application domains may include healthcare, finance, IoT, and transportation."

Correct interpretation:
These are examples of eligible domains. A healthcare paper does not also need to
study finance, IoT, and transportation.

ROLE RULES:

- MANDATORY:
  A standalone condition that must be satisfied.

- ALTERNATIVE:
  A member of an ANY or AT_LEAST group.

- OPTIONAL:
  An example, optional context, or non-mandatory scope illustration.

- EXCLUSION:
  A condition whose presence may justify exclusion.

- UNRESOLVED:
  Use only when the natural-language relationship cannot be safely determined.

GROUP RULES:

- ALL:
  Every member is required.
  Set minimum_required equal to the number of members.

- ANY:
  At least one member is required.
  Set minimum_required to 1.

- AT_LEAST:
  A specific minimum number of members is required.

Research question:
{question}

Research context:
{context}

Original inclusion criteria:
{inclusion}

Original exclusion criteria:
{exclusion}

Required JSON schema:
{json.dumps(ScreeningRubric.model_json_schema(), ensure_ascii=False)}"""


def batch_prompt(
    *,
    question: str,
    context: str,
    inclusion: str,
    exclusion: str,
    rubric: ScreeningRubric,
    papers: list[dict[str, str]],
    verification: bool,
) -> str:
    review_mode = (
        (
            "This is a fresh prediction-blind independent review. "
            "You have not been given any previous decision, confidence, "
            "reason, evidence, or expected result."
        )
        if verification
        else
        (
            "This is the primary independent title-and-abstract screening "
            "assessment."
        )
    )

    return f"""You are a systematic-review title-and-abstract screener.

{review_mode}

Apply the complete original research question, context, inclusion criteria, and
exclusion criteria.

LOGICAL INTERPRETATION RULES:

1. Preserve the researcher's original logical relationships.
2. Do not assume every bullet or line is independently mandatory.
3. "At least one", "one or more", "any of", "either", and "one of" describe
   alternatives or minimum-count groups.
4. "Including", "such as", "for example", and "e.g." normally introduce examples
   or alternatives unless the researcher explicitly says every item is required.
5. A paper satisfying one valid member of an ANY group may satisfy that group.
6. A paper does not need to satisfy every example in an alternative list.
7. Explicit mandatory conditions must be supported by the supplied title or
   abstract.
8. Explicitly met exclusion criteria may justify REJECT.
9. Criterion assessments are audit information. The final decision must respect
   the complete original natural-language logic, not a simplistic rule that all
   listed inclusion entries must be MET.
10. When the compiled rubric is unresolved or conflicts with the complete
    original criteria, the complete original criteria remain authoritative.

SCREENING RULES:

1. Judge what the source paper itself studies.
2. Do not rely on incidental keyword overlap.
3. Do not invent unstated populations, methods, outcomes, settings, results, or
   study designs.
4. KEEP only when title-and-abstract evidence supports the original inclusion
   logic and no explicit exclusion is affirmatively met.
5. REJECT only when evidence clearly fails a genuinely mandatory condition or
   clearly meets an exclusion criterion.
6. Use MAYBE when the supplied title and abstract are incomplete, ambiguous,
   incidental, contradictory, or insufficient.
7. Prefer MAYBE over an unsupported definitive decision.
8. Return exactly one result for every supplied paper_id.

EXACT EVIDENCE RULE:

For KEEP and REJECT, evidence_quote is mandatory.

Copy exactly one continuous, nonempty span verbatim from that paper's supplied
title or abstract. The evidence must be one continuous span verbatim and must
never contain joined fragments. For clarity, never insert ellipses.

Never:

- paraphrase;
- summarize;
- combine separate fragments;
- insert three dots;
- insert "...";
- insert the Unicode ellipsis character "…";
- replace words;
- normalize abbreviations;
- quote the research question;
- quote the inclusion or exclusion criteria;
- quote another paper.

For MAYBE, evidence_quote may be blank when no exact useful ambiguity span exists.

Keep every reason under 400 characters.
Keep every evidence quote under 500 characters.

Do not provide chain-of-thought, essays, Markdown, code fences, headings,
commentary, or any text outside one JSON object.

Research question:
{question}

Research context:
{context}

Original inclusion criteria:
{inclusion}

Original exclusion criteria:
{exclusion}

Compiled rubric:
{json.dumps(rubric.model_dump(mode="json"), ensure_ascii=False)}

Papers:
{json.dumps(papers, ensure_ascii=False)}

Required JSON schema:
{json.dumps(ScreeningBatchResult.model_json_schema(), ensure_ascii=False)}"""