from __future__ import annotations

import json

from local_ai.contracts import PaperAssessment, PaperEvidence, ReviewProtocol
from local_ai.evidence import build_evidence_units


SYSTEM_RULES = """
You are LitSync's local systematic-review screening intelligence.
Judge research meaning and relationships, not keyword overlap.
Use only the supplied research question, criteria, title, and abstract.
Never invent evidence. Cite only supplied evidence-unit IDs; the application resolves them to exact source text.
Return only the requested JSON object. Give concise audit rationales, never hidden chain-of-thought.
""".strip()


def protocol_prompt(research_question: str, inclusion: str, exclusion: str) -> str:
    payload = {
        "research_question": research_question,
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
Do not add dataset-specific rules.

INPUT:
{json.dumps(payload, ensure_ascii=False)}

Return a ReviewProtocol matching this JSON schema:
{json.dumps(ReviewProtocol.model_json_schema(), ensure_ascii=False)}
"""


def protocol_critic_prompt(protocol: ReviewProtocol, inclusion: str, exclusion: str) -> str:
    payload = {
        "candidate_protocol": protocol.model_dump(mode="json"),
        "authoritative_inclusion_criteria": inclusion,
        "authoritative_exclusion_criteria": exclusion,
    }
    return f"""{SYSTEM_RULES}

Audit the candidate protocol for semantic mistakes, missing relationships, accidental scope broadening, and missing
user criteria. Every inclusion must be MET when inclusion evidence is present. Every exclusion must describe an
affirmative disqualifying condition and be MET only when that out-of-scope evidence is present. Rewrite exclusions
that are negated inclusions or "must not" requirements. Ensure criteria are independent and not logical inverses.
Remove every research-question-derived exclusion: express RQ scope as positive inclusions instead. Only explicit
authoritative user exclusions may remain as exclusion criteria.
Return a corrected complete ReviewProtocol. Preserve useful criterion ids where possible.

INPUT:
{json.dumps(payload, ensure_ascii=False)}

SCHEMA:
{json.dumps(ReviewProtocol.model_json_schema(), ensure_ascii=False)}
"""


def protocol_repair_prompt(
    candidate: dict,
    error: str,
    research_question: str,
    inclusion: str,
    exclusion: str,
) -> str:
    payload = {
        "research_question": research_question,
        "authoritative_inclusion_criteria": inclusion,
        "authoritative_exclusion_criteria": exclusion,
        "invalid_candidate": candidate,
        "validation_error": error,
    }
    return f"""{SYSTEM_RULES}

Repair this invalid review protocol. It must contain at least one REQUIRED positive inclusion criterion.
Research-question-derived criteria use source='research_question'. Only explicitly supplied user criteria use
source='user'. Inclusion MET means inclusion evidence is present. Exclusion MET means affirmative disqualifying
evidence is present. Do not represent a missing inclusion as an exclusion. Return a complete corrected protocol.
Research-question-derived scope must use positive inclusion criteria. Create exclusions only from explicit user
exclusion criteria, and phrase each as the affirmative disqualifying condition whose presence makes it MET.

INPUT:
{json.dumps(payload, ensure_ascii=False)}

SCHEMA:
{json.dumps(ReviewProtocol.model_json_schema(), ensure_ascii=False)}
"""


def evidence_prompt(protocol: ReviewProtocol, title: str, abstract: str) -> str:
    payload = {
        "protocol": protocol.model_dump(mode="json"),
        "paper_evidence_units": build_evidence_units(title, abstract),
    }
    return f"""{SYSTEM_RULES}

Understand this paper independently, then assess every protocol criterion. Cite only supplied evidence_id values
supporting MET or NOT_MET; never create an id. Use UNCLEAR when the evidence units do not establish the criterion.
For inclusion criteria, MET means inclusion evidence is present. For exclusion criteria, MET means affirmative
disqualifying evidence is present. Use at most the two strongest evidence IDs per criterion. Do not make a final
decision yet.

INPUT:
{json.dumps(payload, ensure_ascii=False)}

SCHEMA:
{json.dumps(PaperEvidence.model_json_schema(), ensure_ascii=False)}
"""


def assessment_prompt(
    protocol: ReviewProtocol,
    title: str,
    abstract: str,
    evidence: PaperEvidence | None = None,
) -> str:
    payload = {
        "protocol": protocol.model_dump(mode="json"),
        "paper_evidence_units": build_evidence_units(title, abstract),
        "prior_evidence_pass": evidence.model_dump(mode="json") if evidence else None,
    }
    return f"""{SYSTEM_RULES}

Understand the paper and compare it directly with every protocol criterion. Return one criterion assessment for every
criterion id. KEEP requires all required inclusions and no met exclusion. REJECT requires affirmative referenced evidence
for a met exclusion or an evidence-backed contradicted required inclusion. Cite only supplied evidence_id values.
For exclusion criteria, MET means affirmative disqualifying evidence appears; absence of an exclusion is NOT_MET.
For a required inclusion, NOT_MET requires affirmative contradictory evidence; mere silence or missing detail is
UNCLEAR, not NOT_MET. A cited unit that directly states a criterion cannot support NOT_MET for that criterion.
Use at most the two strongest evidence IDs per criterion. Keep the summary, rationales, and reason concise. Use MAYBE
when evidence is incomplete or ambiguous.
If a prior evidence pass is supplied, audit it against the original text rather than trusting it blindly.

INPUT:
{json.dumps(payload, ensure_ascii=False)}

SCHEMA:
{json.dumps(PaperAssessment.model_json_schema(), ensure_ascii=False)}
"""


def critic_prompt(
    protocol: ReviewProtocol,
    title: str,
    abstract: str,
    candidate: PaperAssessment,
    validator_errors: list[str],
    validator_warnings: list[str],
) -> str:
    payload = {
        "protocol": protocol.model_dump(mode="json"),
        "paper_evidence_units": build_evidence_units(title, abstract),
        "candidate_assessment": candidate.model_dump(mode="json"),
        "validator_errors": validator_errors,
        "validator_warnings": validator_warnings,
    }
    return f"""{SYSTEM_RULES}

Act as an independent senior screening critic. Re-evaluate the original paper against the protocol, correct unsupported
evidence ids and inconsistent verdicts, and resolve uncertainty only when supplied evidence units support it. For an
exclusion criterion, MET means affirmative disqualifying evidence is present. Cite only supplied evidence_id values.
Return a complete replacement assessment. Do not defend the candidate merely because it exists.

INPUT:
{json.dumps(payload, ensure_ascii=False)}

SCHEMA:
{json.dumps(PaperAssessment.model_json_schema(), ensure_ascii=False)}
"""
