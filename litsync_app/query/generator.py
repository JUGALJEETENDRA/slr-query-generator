"""Latency-bounded Boolean query generation with guarded AI term expansion."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from litsync_app.screening.engines import GEMINI_WEB_V24_ENGINE, LOCAL_ENGINE
from litsync_app.screening.local.engine import LocalAIError, OllamaStructuredEngine
from litsync_app.screening.local.hardware import RuntimeProfile, resolve_runtime_profile


QUERY_DEADLINE_SECONDS = 15.0
GEMINI_WEB_QUERY_DEADLINE_SECONDS = 120.0
MAX_AI_ADDITIONS_PER_GROUP = 4
MAX_AI_ADDITIONS_TOTAL = 12
DEFAULT_QUERY_MODELS = (
    "qwen3.5:4b",
    "qwen3:4b-instruct-2507-q4_K_M",
    "phi4-mini:3.8b-q4_K_M",
)
ALLOWED_TERM_SOURCES = frozenset({
    "literal", "morphology", "source_acronym", "typo_correction",
    "validated_model", "ai_assisted_query_expansion",
})

PARSER_CORE_ROLES = frozenset({
    "technology", "intervention_or_method", "method", "task", "outcome",
    "domain", "population", "comparison",
})
PARSER_CONTEXT_ROLES = frozenset({"context", "limitation"})


class ConceptGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=60)
    role: Literal[
        "technology", "intervention_or_method", "method", "task", "outcome",
        "domain", "population", "comparison", "context", "limitation", "other",
    ]
    terms: list[str] = Field(min_length=1, max_length=8)
    source_spans: list[str] = Field(default_factory=list, max_length=5)


class StructuredQueryDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: list[ConceptGroup] = Field(min_length=2, max_length=4)
    needs_grounding: bool = False
    uncertain_terms: list[str] = Field(default_factory=list, max_length=5)


class AIQueryTermGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_label: str = Field(min_length=1, max_length=60)
    terms: list[str] = Field(min_length=1, max_length=8)


class AIQueryExpansionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_groups: list[AIQueryTermGroup] = Field(max_length=4)
    optional_groups: list[AIQueryTermGroup] = Field(max_length=4)
    uncertain_terms: list[str] = Field(max_length=8)


@dataclass(frozen=True)
class GroundingPaper:
    """Legacy agentic-workflow value type; query generation performs no grounding."""

    paper_id: str
    title: str
    abstract: str
    url: str = ""
    year: int | None = None

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract}".strip()


class SemanticScholarGrounder:
    """Legacy agentic-workflow contract with external retrieval disabled."""

    def search(self, question: str, timeout: float = 2.5) -> list[GroundingPaper]:
        del question, timeout
        return []


@dataclass
class GeneratedQueryBundle:
    google_scholar: str
    scopus: str
    web_of_science: str
    ieee_xplore: str
    pubmed: str
    concepts: dict[str, Any]
    query_versions: dict[str, dict[str, str]] | None = None

    def to_api_response(self) -> dict[str, Any]:
        payload = {
            "status": "success",
            "google_scholar": self.google_scholar,
            "scopus": self.scopus,
            "web_of_science": self.web_of_science,
            "ieee_xplore": self.ieee_xplore,
            "pubmed": self.pubmed,
            "concepts": self.concepts,
        }
        payload["query_versions"] = self.query_versions or {
            "balanced": {
                "google_scholar": self.google_scholar,
                "scopus": self.scopus,
                "web_of_science": self.web_of_science,
                "ieee_xplore": self.ieee_xplore,
                "pubmed": self.pubmed,
            },
            "high_recall": {
                "google_scholar": self.google_scholar,
                "scopus": self.scopus,
                "web_of_science": self.web_of_science,
                "ieee_xplore": self.ieee_xplore,
                "pubmed": self.pubmed,
            },
        }
        return payload


SYSTEM_PROMPT = """
You are a systematic-review information specialist working across all academic domains.
Decompose the complete research question into 2 to 4 essential Boolean concept groups. Each group
must include source_spans copied verbatim from the question. Preserve every important literal
method, task or outcome, domain object, population, comparator, and operating or environmental
condition. Never drop a complete phrase after a relational preposition. Keep coordinated tasks in
one outcome group. Terms may contain literal source phrases, morphology variants, and directly
verifiable acronyms or safe spelling corrections only; do not add related concepts or broad
synonyms. Preserve the original spelling in source_spans even when correcting a term. Return only
schema-valid JSON and do not explain your reasoning. Mark needs_grounding true when later corpus
expansion could help.
""".strip()


GEMINI_AI_EXPANSION_PROMPT = """
You are an expert systematic-review search strategist.

Given the research question and parser-owned concept groups below, suggest concise search-term expansions.

For each existing group:
- preserve every original literal term;
- add only direct synonyms, standard acronyms, spelling variants, and established technical terminology;
- do not add neighbouring topics, applications, causes, interventions, populations, or outcomes that change the scope;
- do not create new concept groups.

Core technology, method, outcome, population, comparison, and domain groups must remain required.
Contextual conditions or limitations must remain optional.

Return JSON only in this structure:

{
  "required_groups": [
    {
      "group_label": "existing group label",
      "terms": ["original term", "additional term"]
    }
  ],
  "optional_groups": [
    {
      "group_label": "existing group label",
      "terms": ["original term", "additional term"]
    }
  ],
  "uncertain_terms": []
}
""".strip()

QUESTION_SCAFFOLD_RE = re.compile(
    r"^(?:how|what|which|where|when|why|do|does|did|is|are|was|were|can|could|should|would)\b",
    re.IGNORECASE,
)
EMBEDDED_SCAFFOLD_RE = re.compile(
    r"\b(?:how|what|which|where|when|why|who)\s+"
    r"(?:do|does|did|is|are|was|were|can|could|should|would)\b",
    re.IGNORECASE,
)
GRAMMAR_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "do", "does", "did",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "their", "these",
    "this", "those", "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
    "within", "under", "among", "between", "can", "could", "should", "would", "used", "using", "use",
}

RELATION_PATTERN = re.compile(
    r"\s+(versus|vs\.?|compare(?:d)?\s+(?:with|to)|for|in|to|across|under|within|among|between|with|on)\s+",
    re.IGNORECASE,
)

ROLE_BY_RELATION = {
    "for": "outcome",
    "in": "domain",
    "to": "outcome",
    "across": "domain",
    "under": "context",
    "within": "context",
    "among": "context",
    "between": "context",
    "with": "context",
    "on": "context",
    "versus": "comparison",
    "vs": "comparison",
    "compared with": "comparison",
    "compared to": "comparison",
    "compare with": "comparison",
    "compare to": "comparison",
}


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _clean_term(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value).replace("_", " ")).strip(" \t\r\n'\"")
    return re.sub(r"\s+", " ", value).strip(" ()")


def _select_model(profile: RuntimeProfile) -> str:
    explicit = os.getenv("LOCAL_QUERY_MODEL", "").strip()
    if explicit:
        return explicit
    installed = profile.hardware.installed_models
    for candidate in DEFAULT_QUERY_MODELS:
        if candidate in installed:
            return candidate
    return DEFAULT_QUERY_MODELS[1]


def _review_scope(prefix: str) -> str:
    match = re.search(
        r"(?:^|\b)(?:in\s+)?(?:(?:systematic|scoping|literature)\s+reviews?|reviews?)\s+of\s+(.+)$",
        prefix.strip(" ,;:"), flags=re.IGNORECASE,
    )
    return _clean_term(match.group(1)) if match else ""


def _question_parts(question: str) -> tuple[str, str, str]:
    cleaned = question.strip(" ?.!")
    embedded = re.search(
        r"(?P<question>\b(?:how|what|which|where|when|why|who)\s+"
        r"(?:(?:do|does|did|is|are|was|were|can|could|should|would)\s+)?)",
        cleaned, flags=re.IGNORECASE,
    )
    prefix = cleaned[:embedded.start("question")].strip(" ,;:") if embedded else ""
    clause = cleaned[embedded.start("question"):] if embedded else cleaned
    body = re.sub(
        r"^(?:how|what|which|where|when|why|who)\s+"
        r"(?:(?:do|does|did|is|are|was|were|can|could|should|would)\s+)?",
        "", clause, flags=re.IGNORECASE,
    ).strip()
    return body, _review_scope(prefix), prefix


def _strip_question_scaffold(question: str) -> str:
    return _question_parts(question)[0]


def _singular_form(term: str) -> str:
    words = term.split()
    if not words:
        return term
    last = words[-1]
    lowered = last.lower()
    if lowered.endswith("ies") and len(last) > 4:
        words[-1] = last[:-3] + "y"
    elif lowered.endswith("s") and not lowered.endswith(("ss", "us", "is")) and len(last) > 3:
        words[-1] = last[:-1]
    return " ".join(words)


def _coordination_spans(span: str) -> list[str]:
    return [
        _clean_term(item) for item in re.split(
            r"\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+", span, flags=re.IGNORECASE,
        )
        if _clean_term(item)
    ]


def _infinitive_object_spans(span: str) -> list[str]:
    objects = []
    for phrase in _coordination_spans(span):
        match = re.match(r"^\S+\s+(.+)$", phrase)
        objects.append(_clean_term(match.group(1) if match else phrase))
    return [item for item in objects if item]


def _segment_source_spans(opened_by: str | None, span: str) -> list[str]:
    return _infinitive_object_spans(span) if opened_by == "to" else _coordination_spans(span)


def _literal_terms(span: str) -> list[str]:
    coordinated = _coordination_spans(span)
    terms: list[str] = []
    for item in coordinated:
        literal = _clean_term(item)
        if not literal or len(literal) > 160:
            continue
        singular = _singular_form(literal)
        terms.extend([singular, literal] if _normalize(singular) != _normalize(literal) else [literal])
    return list(dict.fromkeys(terms))[:8]


def _fallback_label(role: str, index: int) -> str:
    return {
        "technology": "Technology",
        "outcome": "Tasks and outcomes",
        "domain": "Domain object",
        "context": "Conditions and context",
        "comparison": "Comparison",
    }.get(role, f"Concept {index + 1}")


def _lossless_structural_draft(question: str) -> StructuredQueryDraft:
    cleaned, topical_scope, _ = _question_parts(question)
    matches = list(RELATION_PATTERN.finditer(cleaned))
    segments: list[tuple[str | None, str]] = []
    cursor = 0
    relation: str | None = None
    for match in matches:
        span = _clean_term(cleaned[cursor:match.start()])
        if span:
            segments.append((relation, span))
        relation = re.sub(r"\s+", " ", match.group(1).lower()).rstrip(".")
        cursor = match.end()
    tail = _clean_term(cleaned[cursor:])
    if tail:
        segments.append((relation, tail))
    if segments:
        segments[0] = (
            segments[0][0],
            re.sub(
                r"\s+(?:(?:is|are|was|were|be|been|being)\s+)?(?:used|using|use)$",
                "", segments[0][1], flags=re.I,
            ),
        )
    groups: list[ConceptGroup] = []
    for index, (opened_by, span) in enumerate(segments):
        span = _clean_term(span)
        if not span:
            continue
        role = "technology" if index == 0 else ROLE_BY_RELATION.get(opened_by or "", "context")
        source_spans = _segment_source_spans(opened_by, span)
        terms = list(dict.fromkeys(
            term for source_span in source_spans for term in _literal_terms(source_span)
        ))[:8]
        group = ConceptGroup(
            label=_fallback_label(role, index), role=role,
            terms=terms, source_spans=source_spans[:5],
        )
        same_role = next((existing for existing in groups if existing.role == role), None)
        if same_role:
            same_role.source_spans = list(dict.fromkeys(
                [*same_role.source_spans, *group.source_spans]
            ))[:5]
            same_role.terms = list(dict.fromkeys([*same_role.terms, *group.terms]))[:8]
        elif len(groups) < 4:
            groups.append(group)
        else:
            groups[-1].source_spans = list(dict.fromkeys(
                [*groups[-1].source_spans, *group.source_spans]
            ))[:5]
            groups[-1].terms = list(dict.fromkeys([*groups[-1].terms, *group.terms]))[:8]
    if topical_scope:
        domain_group = next((group for group in groups if group.role == "domain"), None)
        scope_terms = _literal_terms(topical_scope)
        if domain_group:
            domain_group.source_spans = list(dict.fromkeys(
                [topical_scope, *domain_group.source_spans]
            ))[:5]
            domain_group.terms = list(dict.fromkeys(
                [*scope_terms, *domain_group.terms]
            ))[:8]
        elif len(groups) < 4:
            groups.append(ConceptGroup(
                label="Domain object", role="domain",
                terms=scope_terms, source_spans=[topical_scope],
            ))
    if len(groups) < 2:
        words = cleaned.split()
        midpoint = max(1, len(words) // 2)
        spans = [_clean_term(" ".join(words[:midpoint])), _clean_term(" ".join(words[midpoint:]))]
        groups = [
            ConceptGroup(
                label=f"Concept {index + 1}", role="other",
                terms=_literal_terms(span), source_spans=[span],
            )
            for index, span in enumerate(spans) if span
        ]
    return StructuredQueryDraft(groups=groups[:4], needs_grounding=True, uncertain_terms=[])


def _deterministic_seed(question: str) -> StructuredQueryDraft:
    """Compatibility alias for the general, vocabulary-free fallback."""
    return _lossless_structural_draft(question)


def decompose_literal_question(question: str) -> StructuredQueryDraft:
    """Public vocabulary-free source-span decomposition API."""
    return _lossless_structural_draft(question)


def _literal_fallback(question: str) -> StructuredQueryDraft:
    """Compatibility alias for the parser-first deterministic fallback."""
    return _deterministic_seed(question)


def _content_tokens(value: str) -> set[str]:
    return {token for token in _normalize(value).split() if token not in GRAMMAR_STOPWORDS and len(token) > 1}


def _term_is_valid(term: str, role: str) -> bool:
    normalized = _normalize(term)
    if (
        not normalized or len(term) > 160
        or QUESTION_SCAFFOLD_RE.search(normalized)
        or EMBEDDED_SCAFFOLD_RE.search(term)
    ):
        return False
    tokens = normalized.split()
    if not _content_tokens(term) or sum(token in GRAMMAR_STOPWORDS for token in tokens) >= len(tokens):
        return False
    return True


def _span_is_verbatim(span: str, question: str) -> bool:
    return bool(_normalize(span)) and _normalize(span) in _normalize(question)


def _span_has_question_scaffolding(span: str, question: str) -> bool:
    _, _, prefix = _question_parts(question)
    return bool(
        EMBEDDED_SCAFFOLD_RE.search(span)
        or (prefix and _normalize(prefix) in _normalize(span))
    )


def _is_verified_abbreviation(term: str, source_spans: list[str]) -> bool:
    candidate_tokens = re.findall(r"[A-Za-z0-9]+", term)
    for source in source_spans:
        source_tokens = re.findall(r"[A-Za-z0-9]+", source)
        for start in range(len(source_tokens)):
            for end in range(start + 2, len(source_tokens) + 1):
                acronym = "".join(token[0] for token in source_tokens[start:end]).upper()
                if not candidate_tokens:
                    continue
                first = candidate_tokens[0].rstrip("s").upper()
                suffix_length = len(candidate_tokens) - 1
                suffix = [token.lower() for token in source_tokens[end:end + suffix_length]]
                if first == acronym and [token.lower() for token in candidate_tokens[1:]] == suffix:
                    return True
    return False


def _edit_distance(left: str, right: str) -> int:
    left, right = left.lower(), right.lower()
    matrix = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for index in range(len(left) + 1):
        matrix[index][0] = index
    for index in range(len(right) + 1):
        matrix[0][index] = index
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and left[i - 1] == right[j - 2] and left[i - 2] == right[j - 1]:
                matrix[i][j] = min(matrix[i][j], matrix[i - 2][j - 2] + 1)
    return matrix[-1][-1]


def _safe_spelling_correction(term: str, source_spans: list[str]) -> dict[str, Any] | None:
    candidate_tokens = re.findall(r"[A-Za-z0-9]+", term)
    for source in source_spans:
        source_tokens = re.findall(r"[A-Za-z0-9]+", source)
        if len(candidate_tokens) != len(source_tokens) or not candidate_tokens:
            continue
        distances = [_edit_distance(original, corrected) for original, corrected in zip(source_tokens, candidate_tokens)]
        changed = [
            original for original, corrected, distance in zip(source_tokens, candidate_tokens, distances)
            if distance and len(original) >= 5 and len(corrected) >= 5
        ]
        if changed and len(changed) == sum(distance > 0 for distance in distances) and sum(distances) <= 2:
            return {
                "original": source,
                "corrected": term,
                "distance": sum(distances),
                "changed_tokens": changed,
            }
    return None


def _term_is_source_linked(term: str, source_spans: list[str]) -> bool:
    normalized = _normalize(term)
    for source in source_spans:
        source_normalized = _normalize(source)
        if normalized in source_normalized or source_normalized in normalized:
            return True
        if _normalize(_singular_form(term)) == _normalize(_singular_form(source)):
            return True
    return bool(
        _is_verified_abbreviation(term, source_spans)
        or _safe_spelling_correction(term, source_spans)
    )


def _clean_model_groups(
    draft: StructuredQueryDraft,
    question: str,
) -> tuple[list[ConceptGroup], list[dict[str, Any]]]:
    groups: list[ConceptGroup] = []
    corrections: list[dict[str, Any]] = []
    seen_labels = set()
    for raw_group in draft.groups[:4]:
        label = _clean_term(raw_group.label) or f"concept_{len(groups) + 1}"
        if _normalize(label) in seen_labels:
            label = f"{label}_{len(groups) + 1}"
        seen_labels.add(_normalize(label))
        source_spans = [
            _clean_term(span) for span in raw_group.source_spans
            if (
                _clean_term(span) and _span_is_verbatim(span, question)
                and not _span_has_question_scaffolding(span, question)
            )
        ][:5]
        if not source_spans:
            continue
        terms = _literal_terms(" and ".join(source_spans))
        seen_terms = set()
        for raw_term in raw_group.terms:
            term = _clean_term(raw_term)
            key = _normalize(term)
            if (
                term and key and key not in seen_terms
                and _term_is_valid(term, raw_group.role)
                and _term_is_source_linked(term, source_spans)
            ):
                terms.append(term)
                seen_terms.add(key)
                correction = _safe_spelling_correction(term, source_spans)
                if correction:
                    corrections.append({
                        key: value for key, value in {
                            **correction, "group": label,
                        }.items() if key != "changed_tokens"
                    })
        if terms:
            groups.append(ConceptGroup(
                label=label, role=raw_group.role,
                terms=list(dict.fromkeys(terms))[:8], source_spans=source_spans,
            ))
    return groups, corrections


def _apply_corrections(
    groups: list[ConceptGroup],
    corrections: list[dict[str, Any]],
) -> None:
    for correction in corrections:
        original_tokens = re.findall(r"[A-Za-z0-9]+", correction["original"])
        corrected_tokens = re.findall(r"[A-Za-z0-9]+", correction["corrected"])
        changed = {
            original.lower() for original, corrected in zip(original_tokens, corrected_tokens)
            if original.lower() != corrected.lower()
        }
        changed.update(_singular_form(token).lower() for token in list(changed))
        target = max(
            groups,
            key=lambda group: len(
                set().union(*(_content_tokens(span) for span in group.source_spans))
                & _content_tokens(correction["original"])
            ),
        )
        target.terms = [
            term for term in target.terms
            if not changed & set(re.findall(r"[a-z0-9]+", term.lower()))
        ]
        corrected = correction["corrected"]
        singular = _singular_form(corrected)
        additions = [singular, corrected] if _normalize(singular) != _normalize(corrected) else [corrected]
        target.terms = list(dict.fromkeys([*target.terms, *additions]))[:8]


def _sanitize_draft(
    draft: StructuredQueryDraft,
    question: str,
    seed: StructuredQueryDraft | None = None,
) -> tuple[StructuredQueryDraft, list[str], list[str], list[dict[str, Any]]]:
    seed = seed or _deterministic_seed(question)
    model_groups, corrections = _clean_model_groups(draft, question)
    uncovered = []
    for seed_group in seed.groups:
        role_groups = [
            group for group in model_groups
            if seed_group.role == "other" or group.role == seed_group.role
        ]
        role_tokens = set().union(*(
            _content_tokens(span) for group in role_groups for span in group.source_spans
        )) if role_groups else set()
        uncovered.extend(
            span for span in seed_group.source_spans
            if not _content_tokens(span).issubset(role_tokens)
        )
    if not uncovered and len(model_groups) >= 2:
        groups = model_groups
        repaired: list[str] = []
    else:
        groups = [group.model_copy(deep=True) for group in seed.groups]
        repaired = list(uncovered)
        for model_group in model_groups:
            target = max(
                groups,
                key=lambda group: len(
                    set().union(*(_content_tokens(span) for span in group.source_spans))
                    & set().union(*(_content_tokens(span) for span in model_group.source_spans))
                ),
            )
            for term in model_group.terms:
                if len(target.terms) < 8 and (
                    _is_verified_abbreviation(term, target.source_spans)
                    or _safe_spelling_correction(term, target.source_spans)
                ):
                    target.terms.append(term)
            target.terms = list(dict.fromkeys(target.terms))[:8]
    _apply_corrections(groups, corrections)
    result = StructuredQueryDraft(
        groups=groups[:4],
        needs_grounding=draft.needs_grounding,
        uncertain_terms=[_clean_term(term) for term in draft.uncertain_terms[:5] if _clean_term(term)],
    )
    return result, uncovered, repaired, corrections


def _is_bounded_acronym(term: str) -> bool:
    return bool(
        re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{1,11}", term)
        and any(character.isupper() for character in term)
    )


def _is_safe_variant(term: str, group: ConceptGroup) -> bool:
    normalized = _normalize(term)
    if not normalized:
        return False
    for span in group.source_spans:
        if normalized == _normalize(span):
            return True
        if normalized == _normalize(_singular_form(span)):
            return True
        if _normalize(_singular_form(term)) == _normalize(_singular_form(span)):
            return True
    return bool(
        _is_verified_abbreviation(term, group.source_spans)
        or _safe_spelling_correction(term, group.source_spans)
    )


def _default_search_roles(groups: list[ConceptGroup]) -> dict[str, dict[str, Any]]:
    return {
        _normalize(group.label): {
            "search_role": (
                "required" if group.role in PARSER_CORE_ROLES else
                "optional" if group.role in PARSER_CONTEXT_ROLES else
                "required"  # Preserve the parser's existing decision for ambiguous/other groups.
            ),
            "balanced_use": False,
        }
        for group in groups
    }


def _local_search_roles(groups: list[ConceptGroup]) -> dict[str, dict[str, Any]]:
    """Local has no evidence basis for demoting literal RQ concepts from the search."""
    return {
        _normalize(group.label): {"search_role": "required", "balanced_use": False}
        for group in groups
    }


def _label_key(value: str) -> str:
    return " ".join(str(value or "").split()).casefold()


def _resolve_group_label(
    proposed_label: str,
    groups: list[ConceptGroup],
) -> ConceptGroup | None:
    exact = [group for group in groups if group.label == proposed_label]
    if len(exact) == 1:
        return exact[0]
    key = _label_key(proposed_label)
    normalized = [group for group in groups if _label_key(group.label) == key]
    return normalized[0] if len(normalized) == 1 else None


def _validate_ai_expansion(
    seed: StructuredQueryDraft,
    proposal: AIQueryExpansionProposal,
) -> tuple[
    StructuredQueryDraft, dict[str, dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[ConceptGroup], list[ConceptGroup],
]:
    display_groups = [group.model_copy(deep=True) for group in seed.groups]
    balanced_by_label = {
        group.label: group.model_copy(deep=True)
        for group in seed.groups if group.role not in PARSER_CONTEXT_ROLES
    }
    high_recall_by_label = {
        label: group.model_copy(deep=True) for label, group in balanced_by_label.items()
    }
    roles = _default_search_roles(display_groups)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    term_owners = {
        _normalize(term): group.label
        for group in display_groups for term in group.terms
    }
    additions_by_group: dict[str, int] = {}
    total_additions = 0

    for proposed_group in [*proposal.required_groups, *proposal.optional_groups]:
        group = _resolve_group_label(proposed_group.group_label, display_groups)
        if group is None:
            for raw_term in proposed_group.terms:
                rejected.append({
                    "term": _clean_term(raw_term) or raw_term,
                    "group": proposed_group.group_label,
                    "term_type": "ai_assisted_query_expansion",
                    "status": "rejected",
                    "reason": "unknown_or_ambiguous_group",
                })
            continue
        display_group = next(item for item in display_groups if item.label == group.label)
        for raw_term in proposed_group.terms:
            term = _clean_term(raw_term)
            normalized = _normalize(term)
            reason = ""
            if not term or not _term_is_valid(term, group.role):
                reason = "invalid_term"
            elif normalized in term_owners:
                reason = (
                    "duplicate_term" if term_owners[normalized] == group.label
                    else "scope_or_group_mismatch"
                )
            elif group.role in PARSER_CONTEXT_ROLES:
                reason = "context_not_compiled"
            elif any(
                _term_is_source_linked(term, other.source_spans)
                for other in display_groups if other.label != group.label
            ) and not _term_is_source_linked(term, group.source_spans):
                reason = "scope_or_group_mismatch"
            elif additions_by_group.get(group.label, 0) >= MAX_AI_ADDITIONS_PER_GROUP:
                reason = "group_addition_limit"
            elif total_additions >= MAX_AI_ADDITIONS_TOTAL:
                reason = "global_addition_limit"
            elif len(display_group.terms) >= 8:
                reason = "group_term_limit"
            if reason:
                rejected.append({
                    "term": term or raw_term,
                    "group": group.label,
                    "term_type": "ai_assisted_query_expansion",
                    "status": "rejected",
                    "reason": reason,
                })
                continue

            mechanically_safe_acronym = _is_verified_abbreviation(
                term, group.source_spans,
            )
            safe_variant = _is_safe_variant(term, group)
            if mechanically_safe_acronym:
                term_type = "safe_acronym"
            elif safe_variant:
                term_type = "safe_variant"
            elif _is_bounded_acronym(term):
                term_type = "ai_acronym"
            else:
                term_type = "ai_semantic_term"

            display_group.terms.append(term)
            high_recall_by_label[group.label].terms.append(term)
            compiled_versions = ["high_recall"]
            if term_type in {"safe_acronym", "safe_variant"}:
                balanced_by_label[group.label].terms.append(term)
                compiled_versions.insert(0, "balanced")
            term_owners[normalized] = group.label
            additions_by_group[group.label] = additions_by_group.get(group.label, 0) + 1
            total_additions += 1
            accepted.append({
                "term": term,
                "group": group.label,
                "term_type": term_type,
                "status": "accepted",
                "proposal_source": "ai_assisted_query_expansion",
                "compiled_versions": compiled_versions,
            })

    return (
        StructuredQueryDraft(
            groups=display_groups,
            needs_grounding=False,
            uncertain_terms=[
                cleaned for value in proposal.uncertain_terms
                if (cleaned := _clean_term(value))
            ][:8],
        ),
        roles,
        accepted,
        rejected,
        list(balanced_by_label.values()),
        list(high_recall_by_label.values()),
    )


def _select_query_groups(
    draft: StructuredQueryDraft,
    roles: dict[str, dict[str, Any]],
    accepted: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> tuple[list[ConceptGroup], list[ConceptGroup], str]:
    del accepted, sources
    required = [
        group for group in draft.groups
        if roles[_normalize(group.label)]["search_role"] == "required"
    ]
    return required, required, ""


def _prefix_base(left: str, right: str) -> str | None:
    left_words, right_words = left.split(), right.split()
    if len(left_words) != len(right_words) or left_words[:-1] != right_words[:-1]:
        return None
    a, b = left_words[-1].lower(), right_words[-1].lower()
    if b == f"{a}s" or a == f"{b}s":
        singular_words = left_words if len(a) < len(b) else right_words
        return " ".join(singular_words)
    if a.endswith("y") and b == f"{a[:-1]}ies":
        return " ".join([*left_words[:-1], a[:-1]])
    if b.endswith("y") and a == f"{b[:-1]}ies":
        return " ".join([*right_words[:-1], b[:-1]])
    return None


def _render_term(term: str, platform: str, *, prefix: bool = False) -> str:
    escaped = term.replace(chr(34), "") + ("*" if prefix else "")
    rendered = f'"{escaped}"' if " " in escaped or not prefix else escaped
    if platform == "pubmed":
        rendered += "[tiab]"
    return rendered


def compile_boolean_query(
    groups: list[ConceptGroup],
    platform: str = "google_scholar",
) -> str:
    blocks = []
    for group in groups[:4]:
        cleaned_terms = []
        seen = set()
        for value in group.terms:
            term = _clean_term(value)
            key = _normalize(term)
            if not term or not key or key in seen:
                continue
            seen.add(key)
            cleaned_terms.append(term)
        terms: list[str] = []
        consumed: set[int] = set()
        if platform != "google_scholar":
            for left_index, left in enumerate(cleaned_terms):
                if left_index in consumed:
                    continue
                for right_index in range(left_index + 1, len(cleaned_terms)):
                    base = _prefix_base(left, cleaned_terms[right_index])
                    if base:
                        terms.append(_render_term(base, platform, prefix=True))
                        consumed.update({left_index, right_index})
                        break
        for index, term in enumerate(cleaned_terms):
            if index not in consumed:
                terms.append(_render_term(term, platform))
        terms = terms[:8 if platform == "google_scholar" else 5]
        if terms:
            blocks.append(f"({' OR '.join(terms)})")
    return " AND ".join(blocks)


def _compile_query_set(groups: list[ConceptGroup]) -> dict[str, str]:
    return {
        "google_scholar": compile_boolean_query(groups, "google_scholar"),
        "scopus": f'TITLE-ABS-KEY({compile_boolean_query(groups, "scopus")})',
        "web_of_science": f'TS=({compile_boolean_query(groups, "web_of_science")})',
        "ieee_xplore": compile_boolean_query(groups, "ieee_xplore"),
        "pubmed": compile_boolean_query(groups, "pubmed"),
    }


def _make_bundle(
    balanced_groups: list[ConceptGroup],
    high_recall_groups: list[ConceptGroup],
    concepts: dict[str, Any],
) -> GeneratedQueryBundle:
    balanced = _compile_query_set(balanced_groups)
    high_recall = _compile_query_set(high_recall_groups)
    versions = {"balanced": balanced, "high_recall": high_recall}
    return GeneratedQueryBundle(
        google_scholar=balanced["google_scholar"],
        scopus=balanced["scopus"],
        web_of_science=balanced["web_of_science"],
        ieee_xplore=balanced["ieee_xplore"],
        pubmed=balanced["pubmed"],
        concepts=concepts,
        query_versions=versions,
    )


def _term_provenance_source(
    term: str,
    group: ConceptGroup,
    seed_terms: set[str],
    corrections: list[dict[str, Any]],
    proposal_detail: dict[str, Any] | None,
) -> str:
    if proposal_detail:
        return "ai_assisted_query_expansion"
    if any(
        _normalize(_singular_form(term))
        == _normalize(_singular_form(item.get("corrected", "")))
        for item in corrections
    ):
        return "typo_correction"
    if any(_normalize(term) == _normalize(span) for span in group.source_spans):
        return "literal"
    if _is_verified_abbreviation(term, group.source_spans):
        return "source_acronym"
    if _normalize(term) in seed_terms:
        return "morphology"
    if _term_is_source_linked(term, group.source_spans):
        return "validated_model"
    raise ValueError(f"term lacks admissible provenance: {term}")


def _validate_term_details(term_details: list[dict[str, Any]]) -> None:
    for detail in term_details:
        source = detail.get("source")
        if source not in ALLOWED_TERM_SOURCES:
            raise ValueError(f"unsupported query-term provenance: {source}")


def generate_query_bundle(
    question: str,
    model: str | None = None,
    *,
    processing_engine: str = LOCAL_ENGINE,
    deadline_seconds: float | None = None,
    profile: RuntimeProfile | None = None,
    engine: Any | None = None,
    grounder: Any | None = None,
) -> GeneratedQueryBundle:
    """Generate Balanced and High Recall queries without external academic APIs."""
    del grounder  # Backward-compatible parameter; external grounding is intentionally disabled.
    started = time.monotonic()
    question = str(question or "").strip()
    if not question:
        raise ValueError("research question is required")
    selected_processing_engine = str(processing_engine or LOCAL_ENGINE).strip().lower()
    if selected_processing_engine not in {LOCAL_ENGINE, GEMINI_WEB_V24_ENGINE}:
        raise ValueError(
            f"Unsupported query-generation engine: {processing_engine}. "
            f"Choose '{LOCAL_ENGINE}' or '{GEMINI_WEB_V24_ENGINE}'."
        )
    maximum_deadline = (
        GEMINI_WEB_QUERY_DEADLINE_SECONDS
        if selected_processing_engine == GEMINI_WEB_V24_ENGINE
        else QUERY_DEADLINE_SECONDS
    )
    if (
        selected_processing_engine == GEMINI_WEB_V24_ENGINE
        and deadline_seconds is not None
        and float(deadline_seconds) <= 0
    ):
        requested_deadline = maximum_deadline
    else:
        requested_deadline = (
            maximum_deadline if deadline_seconds is None else float(deadline_seconds)
        )
    deadline_seconds = max(0.5, min(requested_deadline, maximum_deadline))
    deadline = started + deadline_seconds
    seed = _deterministic_seed(question)
    _, topical_scope, removed_prefix = _question_parts(question)
    draft = seed.model_copy(deep=True)
    roles = _default_search_roles(draft.groups)
    selected_model = ""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    repaired_spans: list[str] = []
    model_failed = False
    fallback_reason = ""
    warning = ""
    timings: dict[str, float] = {}
    query_debug: dict[str, Any] = {
        "raw_source_records_returned_count": 0,
        "schema_valid_source_count": 0,
        "deduplicated_source_count": 0,
        "usable_source_count": 0,
        "source_rejections_by_reason": {},
        "proposal_count": 0,
        "accepted_proposal_count": 0,
        "rejected_proposal_count": 0,
        "proposal_rejections_by_reason": {},
    }
    balanced_groups: list[ConceptGroup] | None = None
    high_recall_groups: list[ConceptGroup] | None = None

    generation_started = time.monotonic()
    if selected_processing_engine == GEMINI_WEB_V24_ENGINE:
        selected_model = model or GEMINI_WEB_V24_ENGINE
        if engine is None:
            from litsync_app.query.engines import GeminiWebQueryEngine
            engine = GeminiWebQueryEngine()
        seed_payload = [
            {
                "group_label": group.label,
                "semantic_role": group.role,
                "literal_source_spans": group.source_spans,
                "literal_terms": group.terms,
            }
            for group in seed.groups
        ]
        prompt = (
            f"{GEMINI_AI_EXPANSION_PROMPT}\n\n"
            f"Research question:\n{question}\n\n"
            f"Parser-owned groups:\n"
            f"{json.dumps(seed_payload, ensure_ascii=False)}"
        )
        try:
            result = engine.generate(
                selected_model,
                prompt,
                AIQueryExpansionProposal,
                timeout_seconds=max(0.1, _remaining(deadline)),
            )
            proposal = AIQueryExpansionProposal.model_validate(result.value)
            (
                draft, roles, accepted, rejected, balanced_groups,
                high_recall_groups,
            ) = _validate_ai_expansion(seed, proposal)
            if not accepted:
                fallback_reason = "no_usable_ai_additions"
                warning = (
                    "Gemini returned no usable query additions; LitSync kept the "
                    "safe parser-first query."
                )
        except (LocalAIError, ValueError) as exc:
            model_failed = True
            fallback_reason = f"gemini_web_draft_failed:{type(exc).__name__}"
            warning = (
                "Gemini Web returned unavailable or malformed expansion data; "
                "LitSync kept the safe parser-first query."
            )
    else:
        profile = profile or resolve_runtime_profile()
        selected_model = model or _select_model(profile)
        engine = engine or OllamaStructuredEngine(profile)
        prompt = (
            f"{SYSTEM_PROMPT}\n\nThese parser-owned literal spans must all be represented. "
            "Local output may add only terms explicitly represented by those spans, safe lexical "
            "variants, spelling corrections, or directly demonstrated acronyms."
            f"\n\nRequired literal spans:\n"
            f"{[span for group in seed.groups for span in group.source_spans]}"
            f"\n\nResearch question:\n{question}"
        )
        try:
            result = engine.generate(
                selected_model,
                prompt,
                StructuredQueryDraft,
                timeout_seconds=max(0.1, _remaining(deadline)),
            )
            draft, _, repaired_spans, corrections = _sanitize_draft(
                StructuredQueryDraft.model_validate(result.value), question, seed,
            )
            roles = _local_search_roles(draft.groups)
            seed_by_role = {
                (group.role, tuple(group.source_spans)): {_normalize(term) for term in group.terms}
                for group in seed.groups
            }
            for group in draft.groups:
                original_terms = seed_by_role.get(
                    (group.role, tuple(group.source_spans)), set(),
                )
                for term in group.terms:
                    if _normalize(term) in original_terms:
                        continue
                    accepted.append({
                        "term": term,
                        "group": group.label,
                        "term_type": (
                            "acronym" if _is_verified_abbreviation(term, group.source_spans)
                            else "lexical_variant"
                        ),
                        "status": "accepted",
                        "proposal_source": "local_model_proposed",
                    })
        except (LocalAIError, ValueError) as exc:
            model_failed = True
            fallback_reason = f"local_draft_failed:{type(exc).__name__}"
            warning = (
                "The local model was unavailable or returned invalid structured output. "
                "LitSync returned parser-first concepts and safe lexical variants only."
            )

    query_debug["proposal_count"] = len(accepted) + len(rejected)
    query_debug["accepted_proposal_count"] = len(accepted)
    query_debug["rejected_proposal_count"] = len(rejected)
    for item in rejected:
        reason = str(item.get("reason") or "unspecified")
        counts = query_debug["proposal_rejections_by_reason"]
        counts[reason] = counts.get(reason, 0) + 1

    timings["structured_generation_ms"] = round(
        (time.monotonic() - generation_started) * 1000, 1,
    )
    if balanced_groups is None or high_recall_groups is None:
        high_recall_groups, balanced_groups, _ = _select_query_groups(
            draft, roles, accepted, [],
        )
    if not compile_boolean_query(high_recall_groups):
        draft = seed.model_copy(deep=True)
        roles = _default_search_roles(draft.groups)
        high_recall_groups, balanced_groups, _ = _select_query_groups(
            draft, roles, [], [],
        )
        fallback_reason = fallback_reason or "empty_query_repaired"
        warning = warning or "LitSync repaired an empty result with parser-first concepts."
        model_failed = True

    if selected_processing_engine == GEMINI_WEB_V24_ENGINE:
        generation_status = (
            "ai_assisted_expansion" if accepted else "literal_fallback"
        )
        evidence_level = "not_literature_checked"
    else:
        evidence_level = "local_not_academically_grounded"
        generation_status = (
            "local_fallback" if model_failed else
            "repaired" if repaired_spans or corrections or topical_scope else
            "full"
        )
        warning = warning or (
            "Local additions are model-proposed and not academically grounded. "
            "LitSync compiled only literal or deterministically source-linked terms."
        )

    final_source_tokens = set().union(*(
        _content_tokens(span) for group in draft.groups for span in group.source_spans
    )) if draft.groups else set()
    required_tokens = set().union(*(
        _content_tokens(span) for group in seed.groups for span in group.source_spans
    )) if seed.groups else set()
    literal_coverage = round(
        len(final_source_tokens & required_tokens) / max(1, len(required_tokens)), 4,
    )
    final_uncovered = [
        span for group in seed.groups for span in group.source_spans
        if not _content_tokens(span).issubset(final_source_tokens)
    ]

    accepted_by_term = {
        (_normalize(item["term"]), item["group"]): item for item in accepted
    }
    seed_terms = {_normalize(term) for group in seed.groups for term in group.terms}
    term_details = []
    for group in draft.groups:
        for term in group.terms:
            proposal_detail = accepted_by_term.get((_normalize(term), group.label), {})
            source = _term_provenance_source(
                term, group, seed_terms, corrections, proposal_detail,
            )
            term_details.append({
                "term": term,
                "group": group.label,
                "source": source,
                "proposal_source": proposal_detail.get("proposal_source", source),
                "term_type": proposal_detail.get("term_type", source),
                "validation_status": "accepted",
                "compiled_versions": proposal_detail.get("compiled_versions", []),
                "supporting_paper_ids": [],
            })
    _validate_term_details(term_details)
    timings["total_ms"] = round((time.monotonic() - started) * 1000, 1)
    concept_groups = [group.model_dump() for group in draft.groups]
    concepts = {
        "groups": concept_groups,
        "concept_classifications": [
            {
                "group_label": group.label,
                "search_role": roles[_normalize(group.label)]["search_role"],
                "balanced_use": roles[_normalize(group.label)]["balanced_use"],
            }
            for group in draft.groups
        ],
        "term_details": term_details,
        "expansion_proposals": [*accepted, *rejected],
        "grounded_terms": [],
        "grounding_papers": [],
        "gemini_reported_sources": [],
        "rejected_sources": [],
        "evidence_label": (
            "AI-assisted query expansion"
            if selected_processing_engine == GEMINI_WEB_V24_ENGINE
            else "local_model_proposed"
        ),
        "evidence_limitation": (
            "Gemini-proposed terminology was not independently checked against "
            "academic literature."
            if selected_processing_engine == GEMINI_WEB_V24_ENGINE
            else "Local model proposals are not academically grounded."
        ),
        "evidence_level": evidence_level,
        "reported_outcome": "not_available",
        "no_evidence_reason": "",
        "usable_source_count": 0,
        "selected_optional_block": "",
        "model": selected_model,
        "processing_engine": selected_processing_engine,
        "engine_display_name": (
            "Gemini Web Automation"
            if selected_processing_engine == GEMINI_WEB_V24_ENGINE
            else "Local Ollama"
        ),
        "mode": (
            "ai_assisted" if selected_processing_engine == GEMINI_WEB_V24_ENGINE
            else "local"
        ),
        "needs_grounding": draft.needs_grounding,
        "uncertain_terms": draft.uncertain_terms,
        "fallback_reason": fallback_reason,
        "generation_status": generation_status,
        "warning": warning,
        "literal_coverage": literal_coverage,
        "uncovered_spans": final_uncovered,
        "repaired_spans": repaired_spans,
        "corrections": corrections,
        "removed_scaffolding": removed_prefix,
        "timings": timings,
        "deadline_seconds": deadline_seconds,
        "query_debug": query_debug,
    }
    return _make_bundle(balanced_groups, high_recall_groups, concepts)


def generate_query(question: str, model: str | None = None) -> str:
    """Backward-compatible query-only entry point."""
    return generate_query_bundle(question, model=model).google_scholar
