from __future__ import annotations

import json

from .contracts import ReviewProtocol


SYSTEM_RULES = """
You are LitSync's local systematic-review screening intelligence.
Judge research meaning and relationships, not keyword overlap.
Use only the supplied research question, optional explanatory context, criteria, title, and abstract.
Never invent evidence. Cite only supplied evidence-unit IDs; the application resolves them to exact source text.
Return only the requested JSON object. Give concise audit rationales, never hidden chain-of-thought.
""".strip()


def protocol_prompt(
    research_question: str,
    inclusion: str,
    exclusion: str,
    research_context: str = "",
) -> str:
    payload = {
        "research_question": research_question,
        "research_context_for_interpretation_only": research_context,
        "authoritative_inclusion_criteria": inclusion,
        "authoritative_exclusion_criteria": exclusion,
    }
    return f"""{SYSTEM_RULES}

Compile an immutable review protocol. Understand the intended population/domain, intervention or phenomenon,
task/outcome, evidence relationship, and study scope without using a fixed ontology. Explicit user criteria are
authoritative and every non-empty user criterion must appear with source='user'. Create stable short criterion ids.
All criteria inferred from the research question must use source='research_question'. Create at least one required
positive inclusion criterion. Do not create an exclusion that merely negates an inclusion. An exclusion criterion
must describe an affirmative disqualifying condition and is MET only when evidence of that condition appears.
Inclusion MET means the required condition is present. Exclusion MET always means the paper should be excluded.
UNCLEAR means the supplied paper text cannot establish the criterion. Never phrase an exclusion as a requirement
that the paper "must not" do something.
Represent all scope inferred from the research question as positive inclusion criteria. Create exclusion criteria
only for explicit authoritative_exclusion_criteria supplied by the user; never invent an exclusion from RQ scope.
The research context is explanatory background only. It may clarify the meaning and boundaries of the research
question, but it must not create a new required criterion or exclusion that the question and explicit criteria do
not support. Preserve it in research_context for provenance.
Do not add dataset-specific rules.

INPUT:
{json.dumps(payload, ensure_ascii=False)}

Return a ReviewProtocol matching this JSON schema:
{json.dumps(ReviewProtocol.model_json_schema(), ensure_ascii=False)}
"""


def protocol_critic_prompt(
    protocol: ReviewProtocol,
    inclusion: str,
    exclusion: str,
    research_context: str = "",
) -> str:
    payload = {
        "candidate_protocol": protocol.model_dump(mode="json"),
        "authoritative_inclusion_criteria": inclusion,
        "authoritative_exclusion_criteria": exclusion,
        "research_context_for_interpretation_only": research_context,
    }
    return f"""{SYSTEM_RULES}

Audit the candidate protocol for semantic mistakes, missing relationships, accidental scope broadening, and missing
user criteria. Every inclusion must be MET when inclusion evidence is present. Every exclusion must describe an
affirmative disqualifying condition and be MET only when that out-of-scope evidence is present. Rewrite exclusions
that are negated inclusions or "must not" requirements. Ensure criteria are independent and not logical inverses.
Remove every research-question-derived exclusion: express RQ scope as positive inclusions instead. Only explicit
authoritative user exclusions may remain as exclusion criteria.
Ensure the research context only explains the question. Remove any required criterion or exclusion created solely
from contextual background, and preserve the supplied context verbatim in research_context.
Return a corrected complete ReviewProtocol. Preserve useful criterion ids where possible.

INPUT:
{json.dumps(payload, ensure_ascii=False)}

SCHEMA:
{json.dumps(ReviewProtocol.model_json_schema(), ensure_ascii=False)}
"""
