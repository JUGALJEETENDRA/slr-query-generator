from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Literal, cast

from pydantic import Field, create_model

from .contracts import CriterionAssessment, ScreeningProtocolV2, StrictModel
from .evidence import build_evidence_units


ASSESSOR_VERSION = "local-v2-assessor-v2"

AssessmentIssueCode = Literal[
    "EMPTY_RESPONSE",
    "NO_JSON_OBJECT",
    "INVALID_ENVELOPE",
    "PROTOCOL_ID_MISMATCH",
    "PAPER_ID_MISMATCH",
    "ASSESSMENTS_NOT_LIST",
    "INVALID_ASSESSMENT",
    "MISSING_CRITERION",
    "DUPLICATE_CRITERION",
    "UNKNOWN_CRITERION",
]

_ALLOWED_TOP_LEVEL_FIELDS = {"protocol_id", "paper_id", "assessments"}


class ModelAssessmentEnvelope(StrictModel):
    protocol_id: str = Field(min_length=1, max_length=80)
    paper_id: str = Field(min_length=1, max_length=200)
    assessments: list[CriterionAssessment] = Field(min_length=1)


@lru_cache(maxsize=64)
def _model_assessment_envelope_schema_for_count(
    criterion_count: int,
) -> type[ModelAssessmentEnvelope]:
    if criterion_count < 1:
        raise ValueError("criterion_count must be at least one")

    schema = create_model(
        f"ModelAssessmentEnvelopeExact{criterion_count}",
        __base__=ModelAssessmentEnvelope,
        assessments=(
            list[CriterionAssessment],
            Field(min_length=criterion_count, max_length=criterion_count),
        ),
    )
    return cast(type[ModelAssessmentEnvelope], schema)


def model_assessment_envelope_schema(
    protocol: ScreeningProtocolV2,
) -> type[ModelAssessmentEnvelope]:
    """Require one structured assessment slot per compiled criterion.

    This constrains structural completeness only. Criterion identities, semantic
    relations, evidence quotations, and policy consequences remain independently
    validated by the parser, evidence validator, and deterministic policy.
    """

    return _model_assessment_envelope_schema_for_count(len(protocol.criteria))


class AssessmentIssue(StrictModel):
    code: AssessmentIssueCode
    message: str = Field(min_length=1, max_length=1200)
    criterion_id: str | None = Field(default=None, max_length=80)
    assessment_index: int | None = Field(default=None, ge=0)


class ModelAssessmentParseResult(StrictModel):
    assessor_version: Literal[ASSESSOR_VERSION] = ASSESSOR_VERSION
    success: bool
    protocol_id: str
    paper_id: str
    assessments: list[CriterionAssessment] = Field(default_factory=list)
    parsed_assessments: list[CriterionAssessment] = Field(default_factory=list)
    issues: list[AssessmentIssue] = Field(default_factory=list)
    safe_fallback: bool = False


def _issue(
    code: AssessmentIssueCode,
    message: str,
    *,
    criterion_id: str | None = None,
    assessment_index: int | None = None,
) -> AssessmentIssue:
    bounded_message = str(message or code)[:1200]
    bounded_criterion_id = (str(criterion_id)[:80] if criterion_id is not None else None)
    return AssessmentIssue(
        code=code,
        message=bounded_message,
        criterion_id=bounded_criterion_id,
        assessment_index=assessment_index,
    )


def _extract_json_object(raw: str | Mapping[str, Any]) -> dict[str, Any] | None:
    if isinstance(raw, Mapping):
        return dict(raw)

    text = str(raw or "").strip()
    if not text:
        return None

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def build_assessment_prompt(
    protocol: ScreeningProtocolV2,
    *,
    paper_id: str,
    title: str | None,
    abstract: str | None,
) -> str:
    """Build the criterion-level semantic task for the local model.

    Python supplies structure and evidence-unit IDs only. The model performs the
    semantic interpretation of every criterion against the paper.
    """

    normalized_paper_id = str(paper_id or "").strip()
    if not normalized_paper_id:
        raise ValueError("paper_id is required")

    protocol_payload = {
        "protocol_id": protocol.protocol_id,
        "research_question": protocol.research_question,
        "research_context": protocol.research_context,
        "criteria": [criterion.model_dump(mode="json") for criterion in protocol.criteria],
    }
    evidence_payload = [
        {
            "evidence_id": unit.evidence_id,
            "source": unit.source,
            "text": unit.text,
        }
        for unit in build_evidence_units(title, abstract)
    ]
    required_output = {
        "protocol_id": protocol.protocol_id,
        "paper_id": normalized_paper_id,
        "assessments": [
            {
                "criterion_id": criterion.id,
                "relation": (
                    "DIRECT_SUPPORT | DIRECT_CONTRADICTION | "
                    "MISSING_OR_UNCLEAR | NOT_APPLICABLE"
                ),
                "rationale": "brief criterion-specific explanation",
                "evidence": [
                    {
                        "evidence_id": "exact evidence unit id",
                        "source": "title | abstract",
                        "quote": "exact contiguous quotation from that unit",
                    }
                ],
            }
            for criterion in protocol.criteria
        ],
    }

    instructions = (
        "You are the primary local semantic screener. Assess every protocol criterion "
        "against only the supplied title and abstract evidence units. Perform the "
        "semantic reasoning yourself; do not rely on keyword overlap alone. Return "
        f"exactly {len(protocol.criteria)} assessments in protocol order, one for "
        "each supplied criterion id. Preserve every criterion id. "
        "DIRECT_SUPPORT means the paper explicitly supports the criterion. "
        "DIRECT_CONTRADICTION means the paper explicitly rules the criterion out. "
        "MISSING_OR_UNCLEAR means the available text does not resolve it. "
        "NOT_APPLICABLE means the criterion genuinely does not apply. For a required "
        "inclusion criterion, absence of information is MISSING_OR_UNCLEAR, never "
        "DIRECT_CONTRADICTION. For an exclusion trigger, use DIRECT_SUPPORT only when "
        "the trigger is explicitly established. Every decisive relation must cite one "
        "or two evidence units and quote exact contiguous wording. Do not use outside "
        "knowledge. Do not produce an overall KEEP, MAYBE, or REJECT decision. Return "
        "one JSON object only, with no markdown or commentary."
    )

    return "\n\n".join(
        [
            instructions,
            "PROTOCOL_JSON:\n"
            + json.dumps(
                protocol_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "PAPER_JSON:\n"
            + json.dumps(
                {
                    "paper_id": normalized_paper_id,
                    "evidence_units": evidence_payload,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "REQUIRED_OUTPUT_SHAPE:\n"
            + json.dumps(
                required_output,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )


def parse_model_assessment_response(
    raw: str | Mapping[str, Any],
    *,
    protocol: ScreeningProtocolV2,
    paper_id: str,
) -> ModelAssessmentParseResult:
    """Parse a local-model response without inventing or repairing semantics.

    Any structural defect makes ``assessments`` empty so direct policy use safely
    produces MAYBE. Successfully parsed entries remain in ``parsed_assessments`` for
    diagnostics or an explicit retry, but are not policy-ready unless the whole
    envelope is complete and valid.
    """

    expected_paper_id = str(paper_id or "").strip()
    if not expected_paper_id:
        raise ValueError("paper_id is required")

    text_is_empty = not isinstance(raw, Mapping) and not str(raw or "").strip()
    if text_is_empty:
        issue = _issue("EMPTY_RESPONSE", "The local model returned an empty response.")
        return ModelAssessmentParseResult(
            success=False,
            protocol_id=protocol.protocol_id,
            paper_id=expected_paper_id,
            issues=[issue],
            safe_fallback=True,
        )

    payload = _extract_json_object(raw)
    if payload is None:
        issue = _issue(
            "NO_JSON_OBJECT",
            "The local model response did not contain a decodable JSON object.",
        )
        return ModelAssessmentParseResult(
            success=False,
            protocol_id=protocol.protocol_id,
            paper_id=expected_paper_id,
            issues=[issue],
            safe_fallback=True,
        )

    issues: list[AssessmentIssue] = []
    unexpected_fields = sorted(set(payload) - _ALLOWED_TOP_LEVEL_FIELDS)
    if unexpected_fields:
        issues.append(
            _issue(
                "INVALID_ENVELOPE",
                "Unexpected top-level fields: " + ", ".join(unexpected_fields),
            )
        )

    returned_protocol_id = str(payload.get("protocol_id") or "").strip()
    returned_paper_id = str(payload.get("paper_id") or "").strip()
    if returned_protocol_id != protocol.protocol_id:
        issues.append(
            _issue(
                "PROTOCOL_ID_MISMATCH",
                "The response protocol_id does not match the compiled protocol.",
            )
        )
    if returned_paper_id != expected_paper_id:
        issues.append(
            _issue(
                "PAPER_ID_MISMATCH",
                "The response paper_id does not match the requested paper.",
            )
        )

    raw_assessments = payload.get("assessments")
    if not isinstance(raw_assessments, list):
        issues.append(
            _issue(
                "ASSESSMENTS_NOT_LIST",
                "The response assessments field must be a list.",
            )
        )
        return ModelAssessmentParseResult(
            success=False,
            protocol_id=protocol.protocol_id,
            paper_id=expected_paper_id,
            issues=issues,
            safe_fallback=True,
        )

    parsed: list[CriterionAssessment] = []
    for index, item in enumerate(raw_assessments):
        try:
            assessment = CriterionAssessment.model_validate(item)
        except Exception as exc:
            criterion_id = None
            if isinstance(item, Mapping):
                criterion_id = str(item.get("criterion_id") or "").strip() or None
            issues.append(
                _issue(
                    "INVALID_ASSESSMENT",
                    f"Assessment {index} is invalid: {str(exc)}",
                    criterion_id=criterion_id,
                    assessment_index=index,
                )
            )
            continue
        parsed.append(assessment)

    protocol_ids = [criterion.id for criterion in protocol.criteria]
    protocol_id_set = set(protocol_ids)
    counts = Counter(item.criterion_id for item in parsed)

    for criterion_id in protocol_ids:
        if counts.get(criterion_id, 0) == 0:
            issues.append(
                _issue(
                    "MISSING_CRITERION",
                    f"Missing assessment for criterion {criterion_id!r}.",
                    criterion_id=criterion_id,
                )
            )
        elif counts[criterion_id] > 1:
            issues.append(
                _issue(
                    "DUPLICATE_CRITERION",
                    f"Duplicate assessments for criterion {criterion_id!r}.",
                    criterion_id=criterion_id,
                )
            )

    seen_unknown: set[str] = set()
    for item in parsed:
        if item.criterion_id not in protocol_id_set and item.criterion_id not in seen_unknown:
            issues.append(
                _issue(
                    "UNKNOWN_CRITERION",
                    f"Unknown assessment criterion {item.criterion_id!r}.",
                    criterion_id=item.criterion_id,
                )
            )
            seen_unknown.add(item.criterion_id)

    if issues:
        return ModelAssessmentParseResult(
            success=False,
            protocol_id=protocol.protocol_id,
            paper_id=expected_paper_id,
            parsed_assessments=parsed,
            issues=issues,
            safe_fallback=True,
        )

    by_id = {item.criterion_id: item for item in parsed}
    ordered = [by_id[criterion_id] for criterion_id in protocol_ids]
    return ModelAssessmentParseResult(
        success=True,
        protocol_id=protocol.protocol_id,
        paper_id=expected_paper_id,
        assessments=ordered,
        parsed_assessments=ordered,
    )
