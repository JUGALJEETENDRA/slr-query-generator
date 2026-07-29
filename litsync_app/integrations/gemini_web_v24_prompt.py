from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from litsync_app.screening.local.evidence import build_evidence_units


@dataclass(frozen=True)
class V24Paper:
    paper_id: str
    title: str
    abstract: str


def authoritative_criterion_entries(value: str) -> list[str]:
    entries: list[str] = []
    for line in str(value or "").splitlines():
        for item in line.split(";"):
            canonical = re.sub(
                r"^\s*(?:(?:[-*•]+)|(?:\d+[.)]))\s*",
                "",
                item,
            )
            canonical = re.sub(r"\s+", " ", canonical).strip()
            if canonical:
                entries.append(canonical)
    return entries


def build_protocol_prompt(
    *,
    research_question: str,
    research_context: str,
    inclusion_criteria: str,
    exclusion_criteria: str,
    schema: dict,
) -> str:
    authoritative_inclusions = [
        {
            "kind": "inclusion",
            "source": "user",
            "authoritative_text": entry,
        }
        for entry in authoritative_criterion_entries(inclusion_criteria)
    ]
    authoritative_exclusions = [
        {
            "kind": "exclusion",
            "source": "user",
            "authoritative_text": entry,
        }
        for entry in authoritative_criterion_entries(exclusion_criteria)
    ]
    return f"""
Compile an immutable systematic-review screening protocol from only the researcher-supplied material below.
Interpret meaning and relationships rather than matching keywords. Preserve the original question and every explicit
researcher criterion. Derive semantic equivalents, ambiguities, and near-neighbour boundaries only from this supplied
material. They are interpretation guidance, not automatic exclusion criteria. Never invent a domain ontology,
population, method, task, outcome, or exclusion that is not logically required by the question.

Protocol integrity is mandatory:
- Create one distinct source='user' criterion for every authoritative criterion object below. Copy authoritative_text
  exactly into that criterion, preserve its kind, and do not merge, omit, paraphrase, weaken, or replace it.
- authoritative_text controls the user criterion. The generated description may explain how to assess it but cannot
  narrow or override the authoritative text.
- In addition to all user criteria, create at least one required source='research_question' inclusion criterion with
  is_composite_relationship=true. It must preserve the complete relationship expressed by the question, coherently
  connecting its requested method or intervention, application context/population/task, outcome or purpose, and
  study/evaluation relationship when those elements are present. Do not replace explicit user criteria with it.
- All other criteria use is_composite_relationship=false and research-question criteria have authoritative_text="".

For required inclusion criteria, describe what affirmative title/abstract evidence would establish eligibility.
For exclusions, create a criterion only when the researcher supplied it or it logically follows as an explicit
incompatibility with the requested scope. A distinction such as one task versus another may guide interpretation for
this protocol, but must not become a universal rule. Never create an inferred exclusion whose only condition is that
a required concept is absent or unmentioned. Represent the required concept as an inclusion criterion instead.
An inferred exclusion must describe an affirmatively observable conflicting focus, not missing evidence.

Do not narrow an allowed study or evaluation relationship to physical deployment, laboratory experiments, or
real-world case studies. Original analytical, game-theoretic, simulation, econometric, observational, optimization,
methodological, and quantitative-comparison work can be primary evaluation when it develops, applies, estimates,
compares, or analyzes an original method and reports results. A proposal or desired property without implementation,
estimation, simulation, comparison, analysis, or measured results remains insufficient.

RESEARCH QUESTION:
{research_question}

OPTIONAL CONTEXT:
{research_context or "None supplied."}

AUTHORITATIVE USER INCLUSION CRITERIA:
{json.dumps(authoritative_inclusions, ensure_ascii=False)}

AUTHORITATIVE USER EXCLUSION CRITERIA:
{json.dumps(authoritative_exclusions, ensure_ascii=False)}

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


def assessment_protocol_projection(protocol: dict) -> dict:
    criteria = [
        *protocol.get("required_inclusion_criteria", []),
        *protocol.get("exclusion_boundaries", []),
    ]
    return {
        "protocol_id": protocol.get("protocol_id", ""),
        "research_question": protocol.get("research_question", ""),
        "criteria": [
            {
                "id": criterion.get("id", ""),
                "kind": criterion.get("kind", ""),
                "description": criterion.get("description", ""),
                "required": bool(criterion.get("required", True)),
                "expected_evidence": criterion.get("expected_evidence", ""),
                "source": criterion.get("source", "research_question"),
                "authoritative_text": criterion.get("authoritative_text", ""),
                "is_composite_relationship": bool(
                    criterion.get("is_composite_relationship", False)
                ),
            }
            for criterion in criteria
        ],
    }


def build_primary_prompt(*, protocol: dict, papers: Iterable[V24Paper], schema: dict) -> str:
    return _assessment_prompt(
        role="primary evidence-first screener",
        protocol=assessment_protocol_projection(protocol),
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
        protocol=assessment_protocol_projection(protocol),
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
- For a source='user' criterion, authoritative_text is controlling. Never narrow, weaken, or override it using the
  generated description.
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

Relationship integrity:
- For is_composite_relationship=true, MET requires coherent evidence that the source paper's own objective, method,
  analysis, evaluation, findings, or contribution directly connects the requested method/intervention,
  context/population/task, outcome/purpose, and study/evaluation relationship expressed by the criterion.
- Separate mentions across unrelated evidence units do not establish the composite relationship. Keep it UNCLEAR.
- A requested method appearing only in a broad list is INCIDENTAL unless its individual role is developed, applied,
  estimated, compared, analyzed, or evaluated.
- A case-study setting cannot transfer an outcome evaluated for an unrelated method or development process into the
  requested application outcome.

Study role and method neutrality:
- Assess what the source paper itself does, not what studies it cites or summarizes did. A review, editorial, survey,
  bibliometric study, or conceptual synthesis does not become a primary study by describing eligible work elsewhere.
  Findings attributed to included or cited studies are not the source paper's own implementation or evaluation.
- Original analytical, game-theoretic, simulation, econometric, observational, optimization, methodological, and
  quantitative-comparison studies can satisfy a primary-study or evaluation criterion when the source paper develops,
  applies, estimates, compares, or analyzes an original method and reports results.
- Merely proposing a model, framework, or desired performance property without implementation, estimation,
  simulation, comparison, analysis, or measured results is insufficient.
- A requested outcome may be substantively supported when directly measured or analyzed as an endpoint, mediator,
  causal pathway, mechanism, or comparative performance measure. Background discussion alone is insufficient.

Use decision_risk LOW only when every required relationship is clearly resolved. The verifier receives no primary
decision or rationale and must make a fresh assessment.

IMMUTABLE PROTOCOL:
{json.dumps(protocol, ensure_ascii=False)}

PAPER BATCH:
{json.dumps(payload, ensure_ascii=False)}

Use this compact wire mapping exactly:
- paper: p=paper_id, d=decision, f=confidence, k=decision_risk, r=reason, c=criterion assessments
- criterion: c=criterion_id, v=verdict, u=scope_support, l=evidence_relationship, r=rationale,
  e=evidence references
- evidence reference: s=source, e=evidence_id

Return exactly one JSON object matching this strict compact schema with one unique item for every paper ID:
{json.dumps(schema, ensure_ascii=False)}
""".strip()
