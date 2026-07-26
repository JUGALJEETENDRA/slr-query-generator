from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from litsync_app.screening.local.evidence import build_evidence_units


@dataclass(frozen=True)
class V24Paper:
    paper_id: str
    title: str
    abstract: str


def build_protocol_prompt(
    *,
    research_question: str,
    research_context: str,
    inclusion_criteria: str,
    exclusion_criteria: str,
    schema: dict,
) -> str:
    return f"""
Compile an immutable systematic-review screening protocol from only the researcher-supplied material below.
Interpret meaning and relationships rather than matching keywords. Preserve the original question and every explicit
researcher criterion. Derive semantic equivalents, ambiguities, and near-neighbour boundaries only from this supplied
material. They are interpretation guidance, not automatic exclusion criteria. Never invent a domain ontology,
population, method, task, outcome, or exclusion that is not logically required by the question.

For required inclusion criteria, describe what affirmative title/abstract evidence would establish eligibility.
For exclusions, create a criterion only when the researcher supplied it or it logically follows as an explicit
incompatibility with the requested scope. A distinction such as one task versus another may guide interpretation for
this protocol, but must not become a universal rule. Never create an inferred exclusion whose only condition is that
a required concept is absent or unmentioned. Represent the required concept as an inclusion criterion instead.
An inferred exclusion must describe an affirmatively observable conflicting focus, not missing evidence.

RESEARCH QUESTION:
{research_question}

OPTIONAL CONTEXT:
{research_context or "None supplied."}

RESEARCHER INCLUSION CRITERIA:
{inclusion_criteria or "None supplied."}

RESEARCHER EXCLUSION CRITERIA:
{exclusion_criteria or "None supplied."}

Return exactly one JSON object matching this schema:
{json.dumps(schema, ensure_ascii=False)}
""".strip()


def _paper_payload(papers: Iterable[V24Paper]) -> list[dict]:
    return [
        {
            "paper_id": paper.paper_id,
            "evidence_units": build_evidence_units(paper.title, paper.abstract),
        }
        for paper in papers
    ]


def build_primary_prompt(*, protocol: dict, papers: Iterable[V24Paper], schema: dict) -> str:
    return _assessment_prompt(
        role="primary evidence-first screener",
        protocol=protocol,
        payload=_paper_payload(papers),
        schema=schema,
    )


def build_verification_prompt(
    *, protocol: dict, papers: Iterable[V24Paper], flags: dict[str, dict], schema: dict,
) -> str:
    payload = []
    for paper in papers:
        item = {
            "paper_id": paper.paper_id,
            "evidence_units": build_evidence_units(paper.title, paper.abstract),
            "review_flags": flags.get(paper.paper_id, {}),
        }
        payload.append(item)
    return _assessment_prompt(
        role="independent prediction-blind verifier",
        protocol=protocol,
        payload=payload,
        schema=schema,
    )


def _assessment_prompt(*, role: str, protocol: dict, payload: list[dict], schema: dict) -> str:
    return f"""
Act as a {role}. Assess each paper independently against the immutable protocol using only its supplied title and
abstract evidence units. Do not use external knowledge, expected label distributions, keyword-only matching, or
domain assumptions. The final policy is enforced by software, so report criterion evidence honestly.

For every protocol criterion:
- MET requires cited evidence that substantively supports the criterion.
- Criterion polarity is literal: inclusion MET means the eligibility condition is present; exclusion MET means the
  disqualifying condition itself is present. Never mark an exclusion MET because the paper avoids, differs from, or
  is unrelated to that exclusion. Evidence that affirmatively disproves an exclusion may support NOT_MET; otherwise
  an unobserved exclusion is UNCLEAR.
- NOT_MET is allowed only when cited evidence explicitly establishes a conflicting subject, task, method, population,
  context, or other protocol relationship. A merely unmentioned requirement is UNCLEAR, never NOT_MET.
- Distinguish absence from affirmative alternative focus. If the title or abstract substantively establishes that the
  paper studies a different subject, task, method, population, or context that conflicts with a required criterion,
  cite that positive focus and use NOT_MET with CONFLICTS. Do not call it UNCLEAR merely because the requested concept
  is also absent. If no affirmative alternative focus is established, use UNCLEAR.
- UNCLEAR is required when evidence is absent, vague, incidental, definitional, background-only, or insufficient.
- A missing abstract cannot safely establish a definitive paper decision; preserve MAYBE rather than extrapolating
  from external knowledge or an underspecified title.
- scope_support is SUBSTANTIVE only when the relationship is part of the paper's objective, method, system, analysis,
  experiment, evaluation, findings, or contribution. Otherwise use INCIDENTAL or INSUFFICIENT.
- evidence_relationship is SUPPORTS for affirmative matching evidence, CONFLICTS for affirmative incompatibility,
  INCIDENTAL for background/example/list-level relevance, and INSUFFICIENT when the text cannot resolve the criterion.
- Cite only evidence IDs supplied for that paper. Never manufacture quotations or expand unexplained abbreviations.

Use decision_risk LOW only when every required relationship is clearly resolved. The verifier receives no primary
decision or rationale and must make a fresh assessment.

IMMUTABLE PROTOCOL:
{json.dumps(protocol, ensure_ascii=False)}

PAPER BATCH:
{json.dumps(payload, ensure_ascii=False)}

Return exactly one JSON object matching this schema with one unique item for every paper ID:
{json.dumps(schema, ensure_ascii=False)}
""".strip()
