from __future__ import annotations

import json

from .contracts import ReviewProtocol, ScreeningRQFrame


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
    rq_frame: ScreeningRQFrame | None = None,
) -> str:
    payload = {
        "research_question": research_question,
        "research_context_for_interpretation_only": research_context,
        "authoritative_inclusion_criteria": inclusion,
        "authoritative_exclusion_criteria": exclusion,
        "validated_rq_frame": rq_frame.compact_prompt_payload() if rq_frame else None,
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
Preserve the RQ's logical structure exactly. Do not turn OR alternatives into an AND, and do not split one required
relationship into several mandatory facets unless every responsive paper logically must establish every facet.
Set required=true only for conditions that every valid answer to the RQ must satisfy. Examples, common mechanisms,
possible outcomes, and details that help answer "how" or "in what way" belong in expected_evidence,
expected_relationships, or ambiguities; they are not additional gates. A question asking how one thing is used in a
setting requires evidence of that use relationship, not every typical implementation feature the model can imagine.
HARD STRUCTURE: create exactly one criterion inferred from the RQ. Its id must be "rq_core_relationship", kind must
be "inclusion", source must be "research_question", and required must be true. Its description must be one compact
semantic test of the complete minimally necessary relationship in the RQ, preserving all AND/OR logic. Do not make
any other research_question criteria. Put populations, mechanisms, outcomes, examples, and implementation facets
inside that composite description only when logically necessary; otherwise place them in expected evidence,
relationships, ambiguities, or boundaries. Explicit user criteria remain separate source="user" criteria.
When validated_rq_frame is present, use its required groups and source spans to understand the complete relationship.
Its allowed variants and advisory concepts are interpretation aids only; they must never add or widen eligibility.
The core description may paraphrase only entities, scope, and relationships actually required by the RQ. Never add
an inferred mechanism, implementation feature, outcome, or example to that mandatory description, even as one item
in an OR list. Put such useful possibilities in expected_evidence instead, and make expected_evidence non-empty.
Represent all scope inferred from the research question as positive inclusion criteria. Create exclusion criteria
only for explicit authoritative_exclusion_criteria supplied by the user; never invent an exclusion from RQ scope.
The research context is explanatory background only. It may clarify the meaning and boundaries of the research
question, but it must not create a new required criterion or exclusion that the question and explicit criteria do
not support. Preserve it in research_context for provenance.
Create two to four concise semantic_boundaries that distinguish genuinely responsive evidence from the closest
topical near misses. Derive them only from the supplied question and authoritative criteria. Boundaries must test
meaning and relationships, never literal words, synonyms, or domain vocabulary. Include distinctions such as a
paper's actual contribution versus background discussion, a required relationship versus separately mentioned
concepts, or the requested setting/population versus a neighboring one when those distinctions follow from the RQ.
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
    rq_frame: ScreeningRQFrame | None = None,
) -> str:
    payload = {
        "candidate_protocol": protocol.model_dump(mode="json"),
        "authoritative_inclusion_criteria": inclusion,
        "authoritative_exclusion_criteria": exclusion,
        "research_context_for_interpretation_only": research_context,
        "validated_rq_frame": rq_frame.compact_prompt_payload() if rq_frame else None,
    }
    return f"""{SYSTEM_RULES}

Audit the candidate protocol for semantic mistakes, missing relationships, accidental scope broadening, and missing
user criteria. Every inclusion must be MET when inclusion evidence is present. Every exclusion must describe an
affirmative disqualifying condition and be MET only when that out-of-scope evidence is present. Rewrite exclusions
that are negated inclusions or "must not" requirements. Ensure criteria are independent and not logical inverses.
When validated_rq_frame is present, preserve every required source-backed relationship and treat its variants,
ambiguities, and broadening warnings as advisory safeguards rather than new eligibility requirements.
Remove every research-question-derived exclusion: express RQ scope as positive inclusions instead. Only explicit
authoritative user exclusions may remain as exclusion criteria.
Apply a minimal-necessity test to every required research-question criterion: would every paper that genuinely
answers the RQ have to satisfy it? If not, make it non-required, merge it into the actual required relationship, or
move it to expected evidence/relationships. Preserve AND/OR alternatives and do not promote examples, likely
mechanisms, implementation details, or desirable outcomes into mandatory gates.
The corrected protocol must contain exactly one source="research_question" criterion. It must have
id="rq_core_relationship", kind="inclusion", required=true, and express the complete minimally necessary RQ
relationship. Merge all other inferred RQ criteria into it or move their nonessential detail to expected evidence,
relationships, ambiguities, or boundaries. Keep explicit source="user" criteria separate.
Remove every mechanism, implementation feature, outcome, and example from the mandatory core description unless
the RQ itself requires it. This applies even to OR lists, because every listed alternative can still narrow scope.
Place useful inferred examples in the core criterion's non-empty expected_evidence field instead.
Ensure the research context only explains the question. Remove any required criterion or exclusion created solely
from contextual background, and preserve the supplied context verbatim in research_context.
Audit semantic_boundaries as adversarial near-miss tests. They must be supported by the RQ or explicit criteria,
must not introduce new scope, and must describe semantic distinctions rather than keywords or synonym lists.
Return a corrected complete ReviewProtocol. Preserve useful criterion ids where possible.

INPUT:
{json.dumps(payload, ensure_ascii=False)}

SCHEMA:
{json.dumps(ReviewProtocol.model_json_schema(), ensure_ascii=False)}
"""
