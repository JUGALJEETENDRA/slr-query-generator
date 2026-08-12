"""Latency-bounded Boolean query generation with guarded AI term expansion."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from litsync_app.screening.engines import GEMINI_WEB_ENGINE, LOCAL_ENGINE
from litsync_app.screening.local.engine import LocalAIError, OllamaStructuredEngine
from litsync_app.screening.local.hardware import RuntimeProfile, resolve_runtime_profile


QUERY_DEADLINE_SECONDS = 15.0
GEMINI_WEB_QUERY_DEADLINE_SECONDS = 120.0
MAX_QUESTION_CODEPOINTS = 4000
MAX_PARSER_CLAUSES = 16
MAX_COORDINATION_CANDIDATES = 32
MAX_PARSER_CONCEPTS = 16
MAX_COMPILED_REQUIRED_GROUPS = 8
MAX_SOURCE_SPANS_PER_GROUP = 8
MAX_TERMS_PER_GROUP = 8
MAX_UNCERTAINTIES = 16
MAX_BOUNDED_MESSAGE_LENGTH = 240
DEFAULT_QUERY_MODELS = (
    "qwen3.5:4b",
    "qwen3:4b-instruct-2507-q4_K_M",
    "phi4-mini:3.8b-q4_K_M",
)
ALLOWED_TERM_SOURCES = frozenset({
    "literal", "morphology", "explicit_original_acronym", "parser_normalization",
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
    terms: list[str] = Field(min_length=1, max_length=MAX_TERMS_PER_GROUP)
    source_spans: list[str] = Field(
        default_factory=list, max_length=MAX_SOURCE_SPANS_PER_GROUP,
    )
    canonical_text: str = Field(default="", max_length=160)
    search_role: Literal["required", "optional", "screening_only"] = "required"
    status_reason: str = Field(default="explicit_core_concept", max_length=240)
    coordination: Literal[
        "single", "alternatives", "co_required", "shared_head", "ambiguous",
    ] = "single"
    confidence: Literal["high", "medium", "low"] = "low"
    source_offsets: list[dict[str, int | str]] = Field(
        default_factory=list, max_length=MAX_SOURCE_SPANS_PER_GROUP,
    )
    source_order: int = Field(default=0, ge=0)
    deterministic_variants: list[str] = Field(
        default_factory=list, max_length=MAX_TERMS_PER_GROUP,
    )
    compiled: bool = False
    core_attachment: bool = False


class StructuredQueryDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: list[ConceptGroup] = Field(min_length=1, max_length=MAX_PARSER_CONCEPTS)
    needs_grounding: bool = False
    uncertain_terms: list[str] = Field(default_factory=list, max_length=MAX_UNCERTAINTIES)


class GeminiDirectConcept(BaseModel):
    """One Gemini-owned search concept; no parser structure is involved."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=60)
    balanced_terms: list[str] = Field(min_length=1, max_length=MAX_TERMS_PER_GROUP)
    high_recall_terms: list[str] = Field(min_length=1, max_length=MAX_TERMS_PER_GROUP)

    @model_validator(mode="after")
    def validate_terms(self) -> "GeminiDirectConcept":
        label = _clean_term(self.label)
        if not label or label != self.label.strip():
            raise ValueError("concept labels must be clean, non-empty text")
        for field_name in ("balanced_terms", "high_recall_terms"):
            values = getattr(self, field_name)
            normalized: set[str] = set()
            for value in values:
                cleaned = _clean_term(value)
                if not cleaned or len(cleaned) > 160 or cleaned != value.strip():
                    raise ValueError(f"{field_name} contains a malformed term")
                key = _normalize(cleaned)
                if not key or key in normalized:
                    raise ValueError(f"{field_name} contains duplicate terms")
                normalized.add(key)
        balanced = {_normalize(value) for value in self.balanced_terms}
        high_recall = {_normalize(value) for value in self.high_recall_terms}
        if not balanced.issubset(high_recall):
            raise ValueError("high_recall_terms must contain every balanced term")
        return self


class GeminiDirectConceptProposal(BaseModel):
    """Compact Gemini Web response used by direct query generation."""

    model_config = ConfigDict(extra="forbid")

    concepts: list[GeminiDirectConcept] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def validate_concepts(self) -> "GeminiDirectConceptProposal":
        labels: set[str] = set()
        has_high_recall_addition = False
        for concept in self.concepts:
            label = _normalize(concept.label)
            if not label or label in labels:
                raise ValueError("concept labels must be unique")
            labels.add(label)
            balanced = {_normalize(value) for value in concept.balanced_terms}
            high_recall = {_normalize(value) for value in concept.high_recall_terms}
            has_high_recall_addition = has_high_recall_addition or bool(
                high_recall - balanced
            )
        if not has_high_recall_addition:
            raise ValueError("High Recall must add at least one term")
        return self


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
Represent the supplied parser-owned structure without creating, deleting, merging, splitting, or
reordering concept groups. Each group must include source_spans copied verbatim from the question.
Preserve every important literal
method, task or outcome, domain object, population, comparator, and operating or environmental
condition. Never drop a complete parser-owned phrase. Terms may contain literal source phrases,
deterministically safe dehyphenated variants, and explicitly stated linked
acronyms only; do not add related concepts, broad synonyms, or spelling guesses. Return only
schema-valid JSON and do not explain your reasoning. Mark needs_grounding true only when
terminology remains uncertain.
""".strip()


GEMINI_DIRECT_CONCEPT_PROMPT = """
You are an expert systematic-review search strategist.

Read the complete research question or short research paragraph below and identify 2 to 5 distinct required search concepts.

For each concept:
- remove question scaffolding such as Can, Does, How, help, and effect of when it is not itself searchable;
- keep distinct required concepts in separate groups;
- put direct synonyms, equivalent terms, standard acronyms, and spelling variants inside the same group;
- make balanced_terms conservative and precise;
- make high_recall_terms contain every balanced term plus broader but still directly relevant terminology;
- never quote or preserve the complete research question as one concept;
- never create database-specific syntax or Boolean query strings;
- remain domain-neutral and do not explain your reasoning.

Return strict JSON only. The JSON must contain only the concepts array required by the schema.
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
    r"\s+(compared?\s+(?:with|to)|versus|vs\.?|rather\s+than|relative\s+to|"
    r"among|under|within|across|"
    r"between|with|for|from|during|before|after|in|on|to|by)\s+",
    re.IGNORECASE,
)
COMPARISON_MARKER_PATTERN = re.compile(
    r"\s+(?P<marker>compared\s+(?:with|to)|versus|vs\.?|rather\s+than|relative\s+to)\s+",
    re.IGNORECASE,
)
LEADING_ATTACHMENT_PATTERN = re.compile(
    r"^\s*(?P<marker>among|within|in|across|under|with)\s+"
    r"(?P<complement>[^,]{1,240}),\s*"
    r"(?P<question>(?:how|what|which|where|when|why|who|to\s+what\s+extent)\b)",
    re.IGNORECASE,
)
TEMPORAL_SHAPE_PATTERN = re.compile(
    r"^(?:(?:19|20)\d{2}(?:\s*(?:-|\u2013|\u2014|/|to|and)\s*(?:19|20)?\d{2})?|"
    r"\d{1,4}\s+(?:days?|weeks?|months?|years?|hours?|minutes?)|"
    r"(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|"
    r"(?:before|after|during)\s+(?:19|20)\d{2})$",
    re.IGNORECASE,
)
PREDICATE_PATTERN = re.compile(
    r"\b(?:affect|affects|affected|impact|impacts|influence|influences|improve|improves|"
    r"improved|improving|reduce|reduces|reduced|reducing|increase|increases|increased|"
    r"enhance|enhances|enhanced|predict|predicts|predicted|detect|detects|detected|"
    r"estimate|estimates|estimated|support|supports|supported|evaluate|evaluates|evaluated|"
    r"analyze|analyzes|analyzed|analyse|analyses|analysed|"
    r"change|changes|changed|determine|determines|determined|prevent|prevents|prevented|"
    r"promote|promotes|promoted|enable|enables|enabled)\b",
    re.IGNORECASE,
)

AUXILIARY_FIRST_QUESTION_PATTERN = re.compile(
    r"^\s*(?P<auxiliary>"
    r"do|does|did|is|are|was|were|can|could|should|would|will"
    r")\s+",
    re.IGNORECASE,
)

DECLARATIVE_MODAL_PATTERN = re.compile(
    r"\b(?P<modal>can|could|should|would|will)$",
    re.IGNORECASE,
)

QUESTION_STARTERS = frozenset({
    "how", "what", "which", "where", "when", "why", "who",
    "do", "does", "did", "is", "are", "was", "were",
    "can", "could", "should", "would", "will",
})

GRAMMATICAL_PREPOSITIONS = frozenset({
    "in", "on", "to", "for", "by", "from", "with", "within",
    "under", "across", "over", "through", "into", "against",
    "around", "between", "among", "during", "before", "after",
    "without", "of",
})

FRAME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("effectiveness", re.compile(
        r"^\s*how\s+effective\s+(?:is|are|was|were)\s+", re.I,
    )),
    ("impact", re.compile(
        r"^\s*what\s+(?:is|are|was|were)\s+(?:the\s+)?"
        r"(?:effect|impact|effectiveness)\s+of\s+", re.I,
    )),
    ("extent", re.compile(
        r"^\s*to\s+what\s+extent\s+(?:do|does|did|can|could|will|would)\s+", re.I,
    )),
    ("question", re.compile(
        r"^\s*(?:how|what|which|where|when|why|who)\s+"
        r"(?:(?:do|does|did|is|are|was|were|can|could|should|would|will)\s+)?", re.I,
    )),
)


@dataclass(frozen=True)
class _TextRange:
    start: int
    end: int

    def text(self, original: str) -> str:
        return original[self.start:self.end]


@dataclass(frozen=True)
class _LeadingAttachment:
    marker: str
    source: _TextRange
    complement: _TextRange
    question_start: int


@dataclass(frozen=True)
class _ComparisonFrame:
    subject: _TextRange
    comparator: _TextRange
    predicate: _TextRange
    governed: _TextRange


@dataclass(frozen=True)
class _PredicateCandidate:
    start: int
    end: int
    score: int
    finite: bool
    lexical_position: int


@dataclass(frozen=True)
class _GovernedClause:
    span: _TextRange
    predicate: _PredicateCandidate


@dataclass(frozen=True)
class _OutcomeStructure:
    component_ranges: tuple[_TextRange, ...]
    canonical_components: tuple[str, ...]
    coordination: Literal["single", "alternatives"]
    confidence: Literal["high", "low"]
    status_reason: str
    uncertainty: bool


def _validate_question_length(question: str) -> None:
    if len(question) > MAX_QUESTION_CODEPOINTS:
        raise ValueError(
            "Research question exceeds the 4,000 Unicode code-point limit."
        )


def _leading_attachment(
    question: str, whole: _TextRange,
) -> _LeadingAttachment | None:
    match = LEADING_ATTACHMENT_PATTERN.match(whole.text(question))
    if not match:
        return None
    base = whole.start
    source = _trim_range(question, base + match.start(), base + match.end("complement"))
    complement = _trim_range(
        question, base + match.start("complement"), base + match.end("complement"),
    )
    if not source or not complement:
        return None
    if _review_scope(source.text(question)):
        return None
    return _LeadingAttachment(
        marker=match.group("marker").casefold(),
        source=source,
        complement=complement,
        question_start=base + match.start("question"),
    )


def _is_structural_set_phrase(value: str) -> bool:
    cleaned = _clean_term(value)
    if not cleaned:
        return False
    tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", cleaned)
    if not tokens:
        return False
    return _plural_shaped(tokens[-1])


def _plural_shaped(token: str) -> bool:
    lowered = token.casefold()
    return (
        len(lowered) > 3 and lowered.endswith("s")
        and not lowered.endswith(("ss", "us", "is", "ics", "as"))
    )


def _is_temporal_shape(value: str) -> bool:
    return bool(TEMPORAL_SHAPE_PATTERN.fullmatch(_clean_term(value)))


def _grammar_tokens(value: str) -> list[str]:
    return [
        token for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", value)
        if token.casefold() not in GRAMMAR_STOPWORDS
    ]


def _predicate_candidates(
    text: str, *, comparison_compatible: bool = False,
    prefer_earlier_viable: bool = False,
) -> list[_PredicateCandidate]:
    matches = list(PREDICATE_PATTERN.finditer(text))[:MAX_COORDINATION_CANDIDATES]
    candidates: list[_PredicateCandidate] = []
    for match in matches:
        subject = text[:match.start()]
        governed = text[match.end():]
        subject_tokens = _grammar_tokens(subject)
        governed_tokens = _grammar_tokens(governed)
        lexical_position = len(_grammar_tokens(text[:match.start()]))
        if not subject_tokens or not governed_tokens or lexical_position == 0:
            continue
        word = match.group(0).casefold()
        finite = word.endswith(("s", "ed"))
        preceding = re.findall(r"[A-Za-z]+", subject.casefold())
        preceded_by_to = bool(preceding and preceding[-1] == "to")
        preceded_by_auxiliary = bool(
            preceding and preceding[-1] in {
                "do", "does", "did", "is", "are", "was", "were",
                "can", "could", "should", "would", "will",
            }
        )
        score = 6
        score += 2 if finite else 0
        score += 2 if preceded_by_auxiliary else 0
        score += 1 if len(subject_tokens) > 1 else 0
        score += 1 if len(governed_tokens) > 1 else 0
        score += 1 if comparison_compatible else 0
        score -= 3 if preceded_by_to else 0
        candidates.append(_PredicateCandidate(
            match.start(), match.end(), score, finite, lexical_position,
        ))

    adjusted: list[_PredicateCandidate] = []
    for candidate in candidates:
        earlier = [item for item in candidates if item.start < candidate.start]
        later = [item for item in candidates if item.start > candidate.start]
        embedded_competition = any(
            1 <= len(_grammar_tokens(text[candidate.end:item.start])) <= 3
            and (
                item.score >= candidate.score
                or item.finite
                or _plural_shaped(_grammar_tokens(text[candidate.end:item.start])[-1])
            )
            and not _is_coordinated_predicate_successor(text, candidate, item)
            for item in later
        )
        immediate_object_candidate = any(
            not _grammar_tokens(text[item.end:candidate.start]) for item in earlier
        )
        penalty = (3 if embedded_competition else 0) + (
            2 if immediate_object_candidate else 0
        )
        adjusted.append(_PredicateCandidate(
            candidate.start, candidate.end, candidate.score - penalty,
            candidate.finite, candidate.lexical_position,
        ))
    if prefer_earlier_viable and adjusted:
        earliest = min(item.start for item in adjusted)
        adjusted = [
            _PredicateCandidate(
                item.start, item.end, item.score + (1 if item.start == earliest else 0),
                item.finite, item.lexical_position,
            )
            for item in adjusted
        ]
    return adjusted


def _is_coordinated_predicate_successor(
    text: str,
    earlier: _PredicateCandidate,
    later: _PredicateCandidate,
) -> bool:
    between = text[earlier.end:later.start]

    match = re.search(
        r"(?P<governed>\S(?:.*\S)?)"
        r"(?:\s+(?:and|or)\s*|,\s*(?:(?:and|or)\s*)?)$",
        between,
        re.I,
    )
    return bool(
        match
        and _grammar_tokens(match.group("governed"))
    )


def _select_governing_predicate(
    text: str, *, comparison_compatible: bool = False,
    prefer_earlier_viable: bool = False,
) -> _PredicateCandidate | None:
    candidates = _predicate_candidates(
        text, comparison_compatible=comparison_compatible,
        prefer_earlier_viable=prefer_earlier_viable,
    )
    if not candidates:
        return None
    candidates = [
        candidate for candidate in candidates
        if not any(
            earlier.start < candidate.start
            and _is_coordinated_predicate_successor(text, earlier, candidate)
            for earlier in candidates
        )
    ] or candidates
    ranked = sorted(
        candidates,
        key=lambda item: (item.score, item.finite, item.lexical_position),
        reverse=True,
    )
    if len(ranked) > 1:
        first, second = ranked[:2]
        if first.score == second.score and first.finite == second.finite:
            return None
    return ranked[0]


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _clean_term(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value).replace("_", " ")).strip(" \t\r\n'\"")
    return re.sub(r"\s+", " ", value).strip(" ()[]{}.,;:!?—–")


def _trim_range(original: str, start: int, end: int) -> _TextRange | None:
    while start < end and (original[start].isspace() or original[start] in ",;:"):
        start += 1
    while end > start and (original[end - 1].isspace() or original[end - 1] in "?.!,;:"):
        end -= 1
    return _TextRange(start, end) if start < end else None


def _one_edit_or_transposition(left: str, right: str) -> bool:
    """Return true only for one insertion, deletion, substitution, or adjacent swap."""
    left = left.casefold()
    right = right.casefold()

    if left == right:
        return False

    if len(left) == len(right):
        differences = [
            index for index, (a, b) in enumerate(zip(left, right))
            if a != b
        ]
        if len(differences) == 1:
            return True
        if len(differences) == 2:
            first, second = differences
            return (
                second == first + 1
                and left[first] == right[second]
                and left[second] == right[first]
            )
        return False

    if abs(len(left) - len(right)) != 1:
        return False

    shorter, longer = (
        (left, right) if len(left) < len(right) else (right, left)
    )
    short_index = 0
    long_index = 0
    skipped = False

    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        if skipped:
            return False
        skipped = True
        long_index += 1

    return True


def _likely_misspelled_question_starter(question: str) -> bool:
    if LEADING_ATTACHMENT_PATTERN.match(question):
        return False
    first = re.match(r"\s*([A-Za-z]+)\b", question)
    if not first:
        return False

    token = first.group(1).casefold()
    if token in QUESTION_STARTERS:
        return False

    return any(
        _one_edit_or_transposition(token, starter)
        or (
            3 <= len(token) <= 5
            and len(token) == len(starter)
            and token[0] == starter[0]
            and sorted(token) == sorted(starter)
        )
        for starter in QUESTION_STARTERS
        if abs(len(token) - len(starter)) <= 1
    )


def _is_multi_sentence_or_paragraph(question: str) -> bool:
    stripped = question.strip()
    if not stripped:
        return False

    nonempty_lines = [
        line.strip() for line in stripped.splitlines() if line.strip()
    ]
    if len(nonempty_lines) > 1:
        return True

    sentence_boundaries = list(re.finditer(
        r"[.!?]\s+(?=[A-Za-z0-9])",
        stripped,
    ))
    return bool(sentence_boundaries)


def _requires_whole_phrase_fallback(question: str) -> bool:
    return (
        _is_multi_sentence_or_paragraph(question)
        or _likely_misspelled_question_starter(question)
    )


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


def _question_frame(
    question: str,
) -> tuple[_TextRange, str, str, _TextRange | None, str, bool]:
    _validate_question_length(question)
    whole = _trim_range(question, 0, len(question))
    if whole is None:
        raise ValueError("research question is required")

    clause_start = whole.start
    prefix = ""
    attachment = _leading_attachment(question, whole)

    review_match = re.search(
        r"\b(?:how|what|which|where|when|why|who|to\s+what\s+extent)\b",
        question[whole.start:whole.end],
        re.I,
    )

    if attachment:
        clause_start = attachment.question_start
        prefix = attachment.source.text(question)
    elif review_match and _review_scope(
        question[whole.start:whole.start + review_match.start()]
    ):
        clause_start = whole.start + review_match.start()
        prefix = question[whole.start:clause_start].strip(" ,;:")

    clause = question[clause_start:whole.end]
    auxiliary_first = AUXILIARY_FIRST_QUESTION_PATTERN.match(clause)

    wh_auxiliary = re.match(
        r"^\s*(?:how|what|which|where|when|why|who|to\s+what\s+extent)\s+"
        r"(?:do|does|did|is|are|was|were|can|could|should|would|will)\b",
        clause,
        re.I,
    )

    auxiliary_inversion = bool(auxiliary_first or wh_auxiliary)

    frame_kind = "none"
    body_start = clause_start

    for name, pattern in FRAME_PATTERNS:
        match = pattern.match(clause)
        if match:
            frame_kind = name
            body_start = clause_start + match.end()
            break

    if frame_kind == "none" and auxiliary_first:
        frame_kind = "question"
        body_start = clause_start + auxiliary_first.end()

    body = _trim_range(question, body_start, whole.end)
    if body is None:
        raise ValueError(
            "research question contains no searchable concept after framing"
        )

    scope_range = None
    if prefix:
        scope = _review_scope(prefix)
        if scope:
            local = prefix.lower().rfind(scope.lower())
            if local >= 0:
                scope_range = _trim_range(
                    question,
                    whole.start + local,
                    whole.start + local + len(scope),
                )

    clause_mode = (
        "interrogative"
        if frame_kind != "none" or auxiliary_inversion
        else "declarative"
    )

    return (
        body,
        frame_kind,
        prefix,
        scope_range,
        clause_mode,
        auxiliary_inversion,
    )


def _question_parts(question: str) -> tuple[str, str, str]:
    body, _, prefix, scope, _, _ = _question_frame(question)
    return body.text(question), scope.text(question) if scope else "", prefix


def _strip_question_scaffold(question: str) -> str:
    return _question_parts(question)[0]


def _coordination_spans(span: str) -> list[str]:
    return [item for item in _coordination_terms(span)[0] if item]


def _safe_lexical_variants(term: str) -> list[str]:
    variants: list[str] = []
    if "-" in term:
        variants.append(re.sub(r"(?<=\w)-(?=\w)", " ", term))
    return list(dict.fromkeys(item for item in variants if _clean_term(item)))


def _literal_terms(span: str) -> list[str]:
    coordinated = _coordination_spans(span)
    terms: list[str] = []
    for item in coordinated:
        literal = _clean_term(item)
        if not literal or len(literal) > 160:
            continue
        terms.append(literal)
        terms.extend(_safe_lexical_variants(literal))
    return list(dict.fromkeys(terms))[:MAX_TERMS_PER_GROUP]


def _fallback_label(role: str, index: int) -> str:
    return {
        "technology": "Technology",
        "outcome": "Tasks and outcomes",
        "domain": "Domain object",
        "context": "Conditions and context",
        "comparison": "Comparison",
        "population": "Population",
        "intervention_or_method": "Intervention or method",
        "limitation": "Conditions and context",
    }.get(role, f"Concept {index + 1}")


def _coordination_terms(span: str) -> tuple[list[str], str]:
    cleaned = _clean_term(span)
    if not cleaned:
        return [], "single"
    if re.search(r"\b(?:combined|joint|jointly)\b", cleaned, re.I):
        return [cleaned], "shared_head"
    if re.search(r"\s+or\s+|\s*,\s*or\s+", cleaned, re.I):
        parts = [
            _clean_term(part) for part in re.split(r"\s*,?\s+or\s+", cleaned, flags=re.I)
            if _clean_term(part)
        ]
        return (parts, "alternatives") if len(parts) > 1 else ([cleaned], "single")
    if re.search(r"\s+and\s+|,", cleaned, re.I):
        parts = [
            _clean_term(part) for part in re.split(
                r"\s*,\s*(?:and\s+)?|\s+and\s+", cleaned, flags=re.I,
            ) if _clean_term(part)
        ][:MAX_COORDINATION_CANDIDATES]
        if len(parts) > 1:
            word_counts = [len(part.split()) for part in parts]
            if len(parts) == 2 and min(word_counts) == 1 and max(word_counts) > 1:
                return [cleaned], "shared_head"
            if all(count == 1 for count in word_counts):
                return [cleaned], "ambiguous"
            return parts, "alternatives"
    return [cleaned], "single"


def _population_coordination_terms(span: str) -> tuple[list[str], str]:
    cleaned = _clean_term(span)
    parts = [
        _clean_term(part) for part in re.split(
            r"\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+", cleaned, flags=re.I,
        ) if _clean_term(part)
    ][:MAX_COORDINATION_CANDIDATES]
    if len(parts) < 2:
        return [cleaned], "single"
    tokens = [part.split() for part in parts]
    if all(len(item) == 1 for item in tokens):
        return parts, "alternatives"
    heads = [item[-1].casefold() for item in tokens if item]
    if len(set(heads)) == 1:
        return parts, "alternatives"
    if _plural_shaped(tokens[0][-1]) and len(tokens[-1]) >= 1:
        return parts, "alternatives"
    if (
        len(tokens[0]) == 1 and not _plural_shaped(tokens[0][-1])
        and len(tokens[-1]) >= 2 and _plural_shaped(tokens[-1][-1])
    ):
        return [cleaned], "shared_head"
    return [cleaned], "ambiguous"


def _bounded_acronym(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9-]{1,9}", value))


def _initials_match(full_form: str, acronym: str) -> bool:
    tokens = [
        token for token in re.findall(r"[A-Za-z0-9]+", full_form)
        if token.lower() not in GRAMMAR_STOPWORDS
    ]
    return 2 <= len(tokens) <= 8 and "".join(token[0] for token in tokens).upper() == acronym.upper()


def _explicit_acronym(
    original: str, span: _TextRange,
) -> tuple[str, _TextRange, str, _TextRange] | None:
    text = span.text(original)
    patterns = (
        re.compile(r"^(?P<full>.+?)\s*\(\s*(?P<acr>[A-Z][A-Z0-9-]{1,9})\s*\)$"),
        re.compile(r"^(?P<acr>[A-Z][A-Z0-9-]{1,9})\s*\(\s*(?P<full>.+?)\s*\)$"),
        re.compile(r"^(?P<full>.+?)\s*(?:,\s*or|/)\s*(?P<acr>[A-Z][A-Z0-9-]{1,9})$", re.I),
    )
    for pattern in patterns:
        match = pattern.match(text.strip())
        if not match:
            continue
        full = _clean_term(match.group("full"))
        acronym = match.group("acr").strip()
        if not _bounded_acronym(acronym) or not _initials_match(full, acronym):
            continue
        full_start = span.start + match.start("full")
        acr_start = span.start + match.start("acr")
        full_range = _trim_range(original, full_start, span.start + match.end("full"))
        acr_range = _trim_range(original, acr_start, span.start + match.end("acr"))
        if full_range and acr_range:
            return full, full_range, acronym, acr_range
    return None


def _canonical_components(
    text: str, role: str,
) -> tuple[list[str], str]:
    components, coordination = (
        _population_coordination_terms(text)
        if role == "population" else _coordination_terms(text)
    )
    if (
        role == "outcome" and re.search(r"\s+and\s+|,", text, re.I)
        and not re.search(r"\b(?:combined|joint|jointly)\b", text, re.I)
    ):
        components = [
            _clean_term(part) for part in re.split(
                r"\s*,\s*(?:and\s+)?|\s+and\s+", text, flags=re.I,
            ) if _clean_term(part)
        ][:MAX_COORDINATION_CANDIDATES]
        if len(components) > 1:
            coordination = "alternatives"
    canonical: list[str] = []
    for component in components:
        value = component
        if not re.search(r"\bof\s+use$", value, re.I):
            value = re.sub(
                r"\s+(?:(?:is|are|was|were|be|been|being)\s+)?(?:used|using|use)$",
                "", value, flags=re.I,
            )
        value = _clean_term(value)
        if value:
            canonical.append(value)
    return list(dict.fromkeys(canonical)), coordination


def _coordinated_component_ranges(
    original: str, span: _TextRange,
) -> list[_TextRange]:
    text = span.text(original)
    if re.search(r"\b(?:combined|joint|jointly)\b", text, re.I):
        return [span]
    separators = list(re.finditer(r"\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+", text, re.I))[
        :MAX_COORDINATION_CANDIDATES - 1
    ]
    if not separators:
        return [span]
    ranges: list[_TextRange] = []
    cursor = 0
    for separator in [*separators, None]:
        end = separator.start() if separator else len(text)
        item = _trim_range(original, span.start + cursor, span.start + end)
        if item:
            ranges.append(item)
        cursor = separator.end() if separator else len(text)
    return ranges if len(ranges) > 1 else [span]


def _leading_exact_predicate(text: str) -> tuple[re.Match[str], int] | None:
    auxiliary = re.match(
        r"\s*(?:do|does|did|is|are|was|were|can|could|should|would|will)\s+",
        text, re.I,
    )
    start = auxiliary.end() if auxiliary else 0
    match = PREDICATE_PATTERN.match(text, start)
    return (match, start) if match else None


def _finite_morphology_class(value: str) -> str | None:
    lowered = value.casefold()

    if lowered.endswith("ed"):
        return "ed"

    if lowered.endswith("s") and not lowered.endswith(("ss", "us", "is")):
        return "s"

    return None


def _has_prepositional_remainder(value: str) -> bool:
    tokens = re.findall(
        r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*",
        value,
    )
    return bool(
        tokens
        and tokens[0].casefold() in GRAMMATICAL_PREPOSITIONS
    )


def _canonicalize_outcome(
    original: str,
    span: _TextRange,
    *,
    frame_kind: str,
    owning_relation: str | None,
    clause_mode: str,
    auxiliary_inversion: bool,
    first_predicate: _TextRange | None,
) -> _OutcomeStructure | None:
    components = _coordinated_component_ranges(original, span)
    canonical = [
        _clean_term(item.text(original))
        for item in components
    ]

    if not canonical or any(not value for value in canonical):
        return None

    coordination: Literal["single", "alternatives"] = (
        "alternatives" if len(components) > 1 else "single"
    )
    reason = "bounded_outcome_phrase"
    confidence: Literal["high", "low"] = "high"
    uncertainty = False

    effectiveness = (
        frame_kind == "effectiveness"
        and owning_relation == "in"
    )

    if effectiveness and len(components) > 1:
        stripped: list[str] = []
        valid = True

        for item in components:
            local = item.text(original)
            leading = _leading_exact_predicate(local)

            if not leading:
                valid = False
                break

            match, start = leading
            lexeme = match.group(0).casefold()
            remainder = _trim_range(
                original,
                item.start + match.end(),
                item.end,
            )

            if (
                start != 0
                or not lexeme.endswith("ing")
                or not remainder
                or not _grammar_tokens(remainder.text(original))
            ):
                valid = False
                break

            stripped.append(
                _clean_term(remainder.text(original))
            )

        if valid:
            return _OutcomeStructure(
                tuple(components),
                tuple(stripped),
                coordination,
                "high",
                "anchored_effectiveness_parallel_gerunds",
                False,
            )

        reason = "ambiguous_effectiveness_coordination"
        confidence = "low"
        uncertainty = True

    elif effectiveness:
        leading = _leading_exact_predicate(
            components[0].text(original)
        )
        if (
            leading
            and leading[0].group(0).casefold().endswith("ing")
        ):
            reason = "ambiguous_effectiveness_coordination"
            confidence = "low"
            uncertainty = True

    first_morphology: str | None = None

    if first_predicate:
        first_text = first_predicate.text(original)
        if (
            clause_mode == "declarative"
            and not auxiliary_inversion
            and PREDICATE_PATTERN.fullmatch(first_text)
        ):
            first_morphology = _finite_morphology_class(
                first_text
            )

    if len(components) > 1 and not effectiveness:
        for index, item in enumerate(components[1:], start=1):
            local = item.text(original)
            leading = _leading_exact_predicate(local)

            if not leading:
                continue

            match, start = leading
            lexeme = match.group(0)
            remainder = _trim_range(
                original,
                item.start + match.end(),
                item.end,
            )

            later_morphology = (
                _finite_morphology_class(lexeme)
                if start == 0
                and PREDICATE_PATTERN.fullmatch(lexeme)
                else None
            )

            can_strip = bool(
                first_morphology
                and later_morphology == first_morphology
                and remainder
                and _grammar_tokens(remainder.text(original))
                and not _has_prepositional_remainder(
                    remainder.text(original)
                )
            )

            if can_strip:
                canonical[index] = _clean_term(
                    remainder.text(original)
                )
            else:
                confidence = "low"
                reason = "ambiguous_coordinated_predicate"
                uncertainty = True

    return _OutcomeStructure(
        tuple(components),
        tuple(canonical),
        coordination,
        confidence,
        reason,
        uncertainty,
    )


def _make_group(
    original: str,
    span: _TextRange,
    *,
    role: str,
    search_role: str,
    reason: str,
    confidence: str,
    order: int,
    label_counts: dict[str, int],
    core_attachment: bool,
    canonical_value: str | None = None,
) -> ConceptGroup | None:
    acronym = None if canonical_value is not None else _explicit_acronym(original, span)
    source_ranges = [span]
    if acronym:
        full, full_range, short, short_range = acronym
        canonical = [full]
        coordination = "single"
        source_ranges = [full_range, short_range]
        terms = [full, *_safe_lexical_variants(full), short]
        variants = [*_safe_lexical_variants(full), short]
    else:
        canonical, coordination = _canonical_components(
            canonical_value if canonical_value is not None else span.text(original), role,
        )
        if not canonical:
            return None
        terms = []
        variants = []
        for value in canonical:
            terms.append(value)
            safe = _safe_lexical_variants(value)
            variants.extend(safe)
            terms.extend(safe)
    base_label = _fallback_label(role, order)
    label_counts[base_label] = label_counts.get(base_label, 0) + 1
    label = (
        base_label if label_counts[base_label] == 1
        else f"{base_label} {label_counts[base_label]}"
    )
    source_offsets = [
        {"start": item.start, "end": item.end, "text": item.text(original)}
        for item in source_ranges
    ]
    return ConceptGroup(
        label=label,
        role=role,
        terms=list(dict.fromkeys(terms))[:MAX_TERMS_PER_GROUP],
        source_spans=[item.text(original) for item in source_ranges],
        canonical_text=" OR ".join(canonical),
        search_role=search_role,
        status_reason=reason[:MAX_BOUNDED_MESSAGE_LENGTH],
        coordination=coordination,
        confidence=confidence,
        source_offsets=source_offsets,
        source_order=span.start,
        deterministic_variants=list(dict.fromkeys(variants))[:MAX_TERMS_PER_GROUP],
        core_attachment=core_attachment,
    )


def _make_structural_outcome_group(
    original: str,
    span: _TextRange,
    structure: _OutcomeStructure,
    *,
    order: int,
    label_counts: dict[str, int],
) -> ConceptGroup:
    base_label = _fallback_label("outcome", order)
    label_counts[base_label] = label_counts.get(base_label, 0) + 1
    label = base_label if label_counts[base_label] == 1 else f"{base_label} {label_counts[base_label]}"
    terms: list[str] = []
    variants: list[str] = []
    for value in structure.canonical_components:
        terms.append(value)
        safe = _safe_lexical_variants(value)
        terms.extend(safe)
        variants.extend(safe)
    ranges = list(structure.component_ranges)
    return ConceptGroup(
        label=label,
        role="outcome",
        terms=list(dict.fromkeys(terms))[:MAX_TERMS_PER_GROUP],
        source_spans=[item.text(original) for item in ranges],
        canonical_text=" OR ".join(structure.canonical_components),
        search_role="required",
        status_reason=structure.status_reason[:MAX_BOUNDED_MESSAGE_LENGTH],
        coordination=structure.coordination,
        confidence=structure.confidence,
        source_offsets=[
            {"start": item.start, "end": item.end, "text": item.text(original)}
            for item in ranges
        ],
        source_order=span.start,
        deterministic_variants=list(dict.fromkeys(variants))[:MAX_TERMS_PER_GROUP],
        core_attachment=True,
    )


def _make_ambiguous_group(
    original: str,
    span: _TextRange,
    *,
    role: str,
    search_role: str,
    reason: str,
    order: int,
    label_counts: dict[str, int],
    core_attachment: bool,
) -> ConceptGroup | None:
    canonical = _clean_term(span.text(original))
    if not canonical:
        return None
    variants = _safe_lexical_variants(canonical)
    base_label = _fallback_label(role, order)
    label_counts[base_label] = label_counts.get(base_label, 0) + 1
    label = base_label if label_counts[base_label] == 1 else f"{base_label} {label_counts[base_label]}"
    return ConceptGroup(
        label=label, role=role,
        terms=[canonical, *variants][:MAX_TERMS_PER_GROUP],
        source_spans=[span.text(original)], canonical_text=canonical,
        search_role=search_role, status_reason=reason[:MAX_BOUNDED_MESSAGE_LENGTH],
        coordination="ambiguous", confidence="low",
        source_offsets=[{"start": span.start, "end": span.end, "text": span.text(original)}],
        source_order=span.start, deterministic_variants=variants[:MAX_TERMS_PER_GROUP],
        core_attachment=core_attachment,
    )


def _segment_ranges(original: str, body: _TextRange) -> list[tuple[str | None, _TextRange]]:
    local = body.text(original)
    matches = list(RELATION_PATTERN.finditer(local))[:MAX_PARSER_CLAUSES - 1]
    segments: list[tuple[str | None, _TextRange]] = []
    cursor = 0
    relation: str | None = None
    for match in matches:
        item = _trim_range(original, body.start + cursor, body.start + match.start())
        if item:
            segments.append((relation, item))
        relation = re.sub(r"\s+", " ", match.group(1).lower()).rstrip(".")
        cursor = match.end()
    tail = _trim_range(original, body.start + cursor, body.end)
    if tail:
        segments.append((relation, tail))
    return segments[:MAX_PARSER_CLAUSES]


GOVERNED_ATTACHMENT_PATTERN = re.compile(
    r"\s+(among|under|within|across|during|before|after|in|with)(?=\s|,)",
    re.IGNORECASE,
)


def _segment_governed_ranges(
    original: str,
    governed: _TextRange,
) -> list[tuple[str | None, _TextRange]]:
    """Split only structurally external attachments from a fixed governed range."""
    local = governed.text(original)
    matches = list(
        GOVERNED_ATTACHMENT_PATTERN.finditer(local)
    )[:MAX_PARSER_CLAUSES - 1]

    strong_markers = {
        "among", "under", "within", "across",
        "during", "before", "after",
    }

    strong_matches = [
        match
        for match in matches
        if match.group(1).casefold() in strong_markers
    ]

    if len(strong_matches) >= 2:
        coordinated = True

        for left, right in zip(
            strong_matches,
            strong_matches[1:],
        ):
            between = local[left.end():right.start()]

            if not re.search(
                r"(?:,\s*(?:(?:and|or)\s*)?$|\s+(?:and|or)\s*$)",
                between,
                re.I,
            ):
                coordinated = False
                break

        if coordinated:
            first = strong_matches[0]
            before = _trim_range(
                original,
                governed.start,
                governed.start + first.start(),
            )
            chain = _trim_range(
                original,
                governed.start + first.start(1),
                governed.end,
            )

            if (
                before
                and chain
                and _grammar_tokens(before.text(original))
                and _grammar_tokens(chain.text(original))
            ):
                prefix_segments = _segment_governed_ranges(
                    original,
                    before,
                )
                return [
                    *prefix_segments,
                    ("coordinated_attachment", chain),
                ][:MAX_PARSER_CLAUSES]

    marker_counts: dict[str, int] = {}
    for match in matches:
        marker = match.group(1).casefold()
        marker_counts[marker] = (
            marker_counts.get(marker, 0) + 1
        )

    accepted: list[re.Match[str]] = []

    for match in matches:
        marker = match.group(1).casefold()

        if marker in strong_markers:
            accepted.append(match)
            continue

        before_text = local[:match.start()]
        after_text = local[match.end():]

        comma_delimited = before_text.rstrip().endswith(",")
        repeated = marker_counts.get(marker, 0) > 1
        temporal_tail = _is_temporal_shape(after_text)
        later_strong = any(
            item.start() > match.start()
            and item.group(1).casefold() in strong_markers
            for item in matches
        )

        if (
            comma_delimited
            or repeated
            or temporal_tail
            or later_strong
        ):
            accepted.append(match)

    segments: list[tuple[str | None, _TextRange]] = []
    cursor = 0
    relation: str | None = None

    for match in accepted:
        item = _trim_range(
            original,
            governed.start + cursor,
            governed.start + match.start(),
        )

        if item and _grammar_tokens(item.text(original)):
            segments.append((relation, item))

        relation = match.group(1).casefold()
        cursor = match.end()

    tail = _trim_range(
        original,
        governed.start + cursor,
        governed.end,
    )

    if tail and _grammar_tokens(tail.text(original)):
        segments.append((relation, tail))

    return segments[:MAX_PARSER_CLAUSES]


def _infer_parallel_comparison_predicate(
    after_marker: str,
) -> tuple[int, int, int] | None:
    coordination = list(re.finditer(r"\s+(?:and|or)\s+", after_marker, re.I))
    if not coordination:
        return None
    first_coordination = coordination[0]
    trailing_pattern = re.compile(
        r"\s+(?:among|within|during|under|with|in|across|before|after)\s+",
        re.I,
    )
    trailing = next(
        (
            match for match in trailing_pattern.finditer(after_marker)
            if match.start() > first_coordination.end()
        ),
        None,
    )
    core_end = trailing.start() if trailing else len(after_marker)
    core = after_marker[:core_end]
    separators = list(re.finditer(r"\s+(?:and|or)\s+", core, re.I))
    if not separators:
        return None
    phrase_ranges: list[tuple[int, int]] = []
    cursor = separators[0].end()
    for separator in separators[1:]:
        phrase_ranges.append((cursor, separator.start()))
        cursor = separator.end()
    phrase_ranges.append((cursor, len(core)))
    trailing_phrases = [core[start:end].strip() for start, end in phrase_ranges]
    if not trailing_phrases or any(not _grammar_tokens(value) for value in trailing_phrases):
        return None
    attachment_markers = {
        "among", "within", "during", "under", "with", "in", "across", "before", "after",
    }
    if any(
        _grammar_tokens(value)[0].casefold() in attachment_markers
        for value in trailing_phrases
    ):
        return None
    parallel_width = len(_grammar_tokens(trailing_phrases[-1]))
    if not 1 <= parallel_width <= 4 or any(
        len(_grammar_tokens(value)) != parallel_width for value in trailing_phrases
    ):
        return None
    left = core[:separators[0].start()]
    left_tokens = list(re.finditer(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", left))
    if len(left_tokens) <= parallel_width + 1:
        return None
    first_outcome_start = left_tokens[-parallel_width].start()
    before_outcome = left[:first_outcome_start].rstrip()
    before_tokens = list(re.finditer(r"[A-Za-z]+(?:-[A-Za-z]+)*", before_outcome))
    if len(before_tokens) < 2:
        return None
    predicate_width = 2 if before_tokens[-1].group(0).casefold() == "to" else 1
    if len(before_tokens) <= predicate_width:
        return None
    predicate_tokens = before_tokens[-predicate_width:]
    predicate_head = predicate_tokens[0].group(0).casefold()
    if predicate_head in GRAMMAR_STOPWORDS:
        return None
    predicate_start = predicate_tokens[0].start()
    predicate_end = predicate_tokens[-1].end()
    comparator = before_outcome[:predicate_start].strip()
    if not _grammar_tokens(comparator):
        return None
    return predicate_start, predicate_end, first_outcome_start


def _has_explicit_embedded_clause(text: str) -> bool:
    """Recognize only explicit clause structure inside a bounded noun-phrase range."""
    candidates = _predicate_candidates(text)
    for candidate in candidates:
        prefix = text[:candidate.start]
        auxiliary = re.search(
            r"\b(?:do|does|did|is|are|was|were|can|could|should|would|will)\s*$",
            prefix, re.I,
        )
        if auxiliary and _grammar_tokens(prefix[:auxiliary.start()]):
            return True
    boundaries = list(re.finditer(r";\s*|,\s+and\s+", text, re.I))
    if not boundaries:
        return False
    cursor = 0
    complete = 0
    for boundary in [*boundaries, None]:
        end = boundary.start() if boundary else len(text)
        clause = text[cursor:end]
        if _select_governing_predicate(clause, prefer_earlier_viable=True):
            complete += 1
        cursor = boundary.end() if boundary else len(text)
    return complete >= 2


def _comparison_frame(
    original: str, body: _TextRange,
) -> tuple[Literal["none", "complete", "ambiguous"], _ComparisonFrame | None]:
    local = body.text(original)
    marker = COMPARISON_MARKER_PATTERN.search(local)
    if not marker:
        return "none", None
    after_marker = local[marker.end():]
    following_candidates = _predicate_candidates(
        after_marker, comparison_compatible=True, prefer_earlier_viable=True,
    )
    following_predicate = _select_governing_predicate(
        after_marker, comparison_compatible=True, prefer_earlier_viable=True,
    )
    inferred: tuple[int, int, int] | None = None
    if following_predicate is None and not following_candidates:
        inferred = _infer_parallel_comparison_predicate(after_marker)
    if following_predicate is None and inferred is None:
        preceding = _select_governing_predicate(
            local[:marker.start()], prefer_earlier_viable=True,
        )
        return ("none", None) if preceding else ("ambiguous", None)
    predicate_start = (
        marker.end() + following_predicate.start
        if following_predicate else marker.end() + inferred[0]
    )
    predicate_end = (
        marker.end() + following_predicate.end
        if following_predicate else marker.end() + inferred[1]
    )
    governed_start = (
        predicate_end if following_predicate else marker.end() + inferred[2]
    )
    subject = _trim_range(original, body.start, body.start + marker.start())
    comparator = _trim_range(
        original, body.start + marker.end(), body.start + predicate_start,
    )
    predicate = _trim_range(
        original, body.start + predicate_start, body.start + predicate_end,
    )
    governed = _trim_range(
        original, body.start + governed_start, body.end,
    )
    if not subject or not comparator or not predicate or not governed:
        return "ambiguous", None
    if any(_has_explicit_embedded_clause(item.text(original)) for item in (subject, comparator)):
        return "ambiguous", None
    return "complete", _ComparisonFrame(subject, comparator, predicate, governed)


def _ambiguous_finite_comparison(text: str) -> bool:
    marker = re.search(r"\s+compare(?:s)?\s+(?:with|to)\s+", text, re.I)
    if not marker:
        return False
    remainder = text[marker.end():]
    return (
        bool(re.search(r"\s+(?:and|or)\s+", remainder, re.I))
        and _select_governing_predicate(remainder) is None
    )


def _independently_governed_clauses(
    original: str,
    span: _TextRange,
) -> list[_GovernedClause]:
    """Return clauses only when each side has its own subject and predicate."""
    local = span.text(original)

    explicit_boundaries = list(re.finditer(
        r";\s*|,\s+and\s+",
        local,
        re.I,
    ))

    if explicit_boundaries:
        clauses: list[_GovernedClause] = []
        cursor = 0

        for boundary in [*explicit_boundaries, None]:
            end = boundary.start() if boundary else len(local)
            item = _trim_range(
                original,
                span.start + cursor,
                span.start + end,
            )
            if not item:
                clauses = []
                break

            predicate = _select_governing_predicate(
                item.text(original),
                prefer_earlier_viable=True,
            )
            if predicate is None:
                clauses = []
                break

            clauses.append(
                _GovernedClause(item, predicate)
            )
            cursor = (
                boundary.end()
                if boundary
                else len(local)
            )

        if 1 < len(clauses) <= MAX_PARSER_CLAUSES:
            return clauses

    plain_boundaries = list(re.finditer(
        r"\s+(?:and|or)\s+",
        local,
        re.I,
    ))[:MAX_COORDINATION_CANDIDATES]

    for boundary in plain_boundaries:
        left = _trim_range(
            original,
            span.start,
            span.start + boundary.start(),
        )
        right = _trim_range(
            original,
            span.start + boundary.end(),
            span.end,
        )

        if not left or not right:
            continue

        left_predicate = _select_governing_predicate(
            left.text(original),
            prefer_earlier_viable=True,
        )
        right_predicate = _select_governing_predicate(
            right.text(original),
            prefer_earlier_viable=True,
        )

        if not left_predicate or not right_predicate:
            continue

        # A predicate immediately after the conjunction is a coordinated
        # shared-subject predicate, not the start of a second governed clause.
        # This remains true when another coordinated predicate follows it.
        leading_right_predicate = _leading_exact_predicate(right.text(original))
        if leading_right_predicate and leading_right_predicate[1] == 0:
            continue

        right_subject = _trim_range(
            original,
            right.start,
            right.start + right_predicate.start,
        )

        if (
            not right_subject
            or not _grammar_tokens(right_subject.text(original))
        ):
            continue

        return [
            _GovernedClause(left, left_predicate),
            _GovernedClause(right, right_predicate),
        ]

    return []


def _assign_segment_role(
    relation: str | None,
    index: int,
    total: int,
    frame_kind: str,
    prior_roles: list[str],
    repeated_relation: bool,
    next_relation: str | None,
    segment_text: str = "",
) -> tuple[str, str, str, str, bool]:
    relation = relation or ""
    if relation == "coordinated_attachment":
        return "limitation", "optional", "coordinated_attachment_chain", "low", False
    if relation.startswith(("compare", "versus", "vs", "rather", "relative")):
        return "comparison", "required", "explicit_comparison_attachment", "high", True
    if relation in {"for", "to", "on"} and (
        frame_kind in {"impact", "effectiveness"} or index == 1
        or "comparison" in prior_roles
    ):
        return "outcome", "required", "principal_relation_object", "high", True
    if relation == "in" and frame_kind == "effectiveness" and index == 1:
        return "outcome", "required", "effectiveness_relation_object", "high", True
    if relation == "among" and "outcome" in prior_roles and index < total:
        return "population", "required", "participant_adjunct_to_core_relation", "medium", True
    if relation in {"under", "with"} and index == total - 1:
        return "limitation", "optional", "trailing_constraint_adjunct", "medium", False
    if relation in {"during", "before", "after"}:
        if _is_temporal_shape(segment_text):
            return "other", "screening_only", "explicit_temporal_restriction", "medium", False
        return "limitation", "optional", "event_or_condition_adjunct", "medium", False
    if relation in {"in", "within", "across"}:
        if (
            relation == "in" and next_relation == "in"
            and "outcome" in prior_roles and "population" not in prior_roles
        ):
            return "population", "required", "participant_between_core_and_scope", "medium", True
        same_markers = sum(1 for value in prior_roles if value in {"population", "domain"})
        if "population" in prior_roles or same_markers:
            return "domain", "required", "setting_or_domain_after_participant", "medium", True
        if relation == "across" and prior_roles:
            return "domain", "required", "distributed_scope_attachment", "medium", True
        return "other", "required", "ambiguous_prepositional_core_attachment", "low", True
    if relation == "by" and index == total - 1:
        return "other", "optional", "ambiguous_trailing_modifier", "low", False
    if repeated_relation:
        return "outcome", "required", "repeated_independent_relation", "high", True
    return "other", "required", "explicit_core_concept_with_ambiguous_role", "low", True


def _select_compiled_groups(groups: list[ConceptGroup]) -> tuple[list[ConceptGroup], list[ConceptGroup]]:
    tier = {"high": 0, "medium": 1, "low": 2}
    required = [group for group in groups if group.search_role == "required"]
    selected_ids = {
        id(group) for group in sorted(
            required,
            key=lambda group: (
                tier[group.confidence], 0 if group.core_attachment else 1,
                group.source_order,
            ),
        )[:MAX_COMPILED_REQUIRED_GROUPS]
    }
    selected: list[ConceptGroup] = []
    overflow: list[ConceptGroup] = []
    for group in groups:
        group.compiled = id(group) in selected_ids
        if group.compiled:
            selected.append(group)
        elif group.search_role == "required":
            overflow.append(group)
    return selected, overflow


def _leading_attachment_group(
    original: str,
    attachment: _LeadingAttachment,
    *,
    order: int,
    label_counts: dict[str, int],
) -> ConceptGroup | None:
    complement = attachment.complement.text(original)
    population_shape = (
        attachment.marker == "among" and _is_structural_set_phrase(complement)
    )
    constraint_shape = attachment.marker in {"under", "with"}
    role = "population" if population_shape else "limitation" if constraint_shape else "other"
    search_role = "required" if not constraint_shape else "optional"
    if population_shape:
        reason = "preposed_set_attachment_to_central_relation"
        confidence = "medium"
        core_attachment = True
    elif constraint_shape:
        reason = "preposed_constraint_adjunct"
        confidence = "medium"
        core_attachment = False
    else:
        reason = "ambiguous_preposed_attachment"
        confidence = "low"
        core_attachment = False
    group = _make_group(
        original,
        attachment.source,
        role=role,
        search_role=search_role,
        reason=reason,
        confidence=confidence,
        order=order,
        label_counts=label_counts,
        core_attachment=core_attachment,
        canonical_value=complement,
    )
    if group and not population_shape and not constraint_shape:
        group.coordination = "ambiguous"
    return group


def _comparison_groups(
    original: str,
    frame: _ComparisonFrame,
    leading: _LeadingAttachment | None,
    label_counts: dict[str, int],
    *,
    frame_kind: str,
    clause_mode: str,
    auxiliary_inversion: bool,
) -> list[ConceptGroup]:
    groups: list[ConceptGroup] = []
    for span, role, reason in (
        (frame.subject, "intervention_or_method", "comparison_frame_subject"),
        (frame.comparator, "comparison", "comparison_frame_comparator"),
    ):
        group = _make_group(
            original, span, role=role, search_role="required", reason=reason,
            confidence="high", order=len(groups), label_counts=label_counts,
            core_attachment=True,
        )
        if group:
            groups.append(group)

    governed_segments = _segment_governed_ranges(original, frame.governed)
    if not governed_segments or governed_segments[0][0] is not None:
        return []
    _, outcome_span = governed_segments[0]
    outcome_structure = _canonicalize_outcome(
        original, outcome_span, frame_kind=frame_kind, owning_relation=None,
        clause_mode=clause_mode, auxiliary_inversion=auxiliary_inversion,
        first_predicate=frame.predicate,
    )
    if not outcome_structure:
        return []
    outcome = _make_structural_outcome_group(
        original, outcome_span, outcome_structure, order=len(groups),
        label_counts=label_counts,
    )
    groups.append(outcome)

    if leading and len(groups) < MAX_PARSER_CONCEPTS:
        attachment_group = _leading_attachment_group(
            original, leading, order=len(groups), label_counts=label_counts,
        )
        if attachment_group:
            groups.append(attachment_group)

    prior_roles = [group.role for group in groups]
    for index, (relation, span) in enumerate(governed_segments[1:], start=1):
        if len(groups) >= MAX_PARSER_CONCEPTS:
            break
        role, search_role, reason, confidence, core = _assign_segment_role(
            relation, index, len(governed_segments), "question", prior_roles,
            False,
            governed_segments[index + 1][0]
            if index + 1 < len(governed_segments) else None,
            span.text(original),
        )
        group = (
            _make_ambiguous_group(
                original, span, role=role, search_role=search_role, reason=reason,
                order=len(groups), label_counts=label_counts, core_attachment=core,
            ) if relation == "coordinated_attachment" else _make_group(
                original, span, role=role, search_role=search_role, reason=reason,
                confidence=confidence, order=len(groups), label_counts=label_counts,
                core_attachment=core,
            )
        )
        if group:
            groups.append(group)
            prior_roles.append(group.role)
    return groups


def _core_clause_groups(
    original: str,
    body: _TextRange,
    predicate: _PredicateCandidate,
    leading: _LeadingAttachment | None,
    label_counts: dict[str, int],
    subject_role: str = "intervention_or_method",
    *,
    frame_kind: str,
    clause_mode: str,
    auxiliary_inversion: bool,
) -> list[ConceptGroup]:
    subject_span = _trim_range(original, body.start, body.start + predicate.start)
    governed = _trim_range(original, body.start + predicate.end, body.end)
    if not subject_span or not governed:
        return []

    subject_text = subject_span.text(original)
    declarative_modal = DECLARATIVE_MODAL_PATTERN.search(
        subject_text
    )

    if (
        declarative_modal
        and clause_mode == "declarative"
        and not auxiliary_inversion
    ):
        trimmed_subject = _trim_range(
            original,
            subject_span.start,
            subject_span.start + declarative_modal.start(),
        )
        if trimmed_subject:
            subject_span = trimmed_subject

    trailing_use = re.search(
        r"\s+(?:(?:is|are|was|were|be|been|being)\s+)?(?:used|using|use)\s+to\s*$",
        subject_span.text(original), re.I,
    )
    if trailing_use:
        trimmed = _trim_range(
            original, subject_span.start, subject_span.start + trailing_use.start(),
        )
        if trimmed:
            subject_span = trimmed
    groups: list[ConceptGroup] = []
    subject = _make_group(
        original, subject_span, role=subject_role, search_role="required",
        reason="direct_core_subject", confidence="high", order=0,
        label_counts=label_counts, core_attachment=True,
    )
    governed_segments = _segment_governed_ranges(original, governed)
    if not subject or not governed_segments or governed_segments[0][0] is not None:
        return []
    groups.append(subject)
    _, outcome_span = governed_segments[0]
    predicate_range = _trim_range(
        original, body.start + predicate.start, body.start + predicate.end,
    )
    outcome_structure = _canonicalize_outcome(
        original, outcome_span, frame_kind=frame_kind, owning_relation=None,
        clause_mode=clause_mode, auxiliary_inversion=auxiliary_inversion,
        first_predicate=predicate_range,
    )
    if not outcome_structure:
        return []
    outcome = _make_structural_outcome_group(
        original, outcome_span, outcome_structure, order=1,
        label_counts=label_counts,
    )
    groups.append(outcome)
    if leading and len(groups) < MAX_PARSER_CONCEPTS:
        attachment = _leading_attachment_group(
            original, leading, order=len(groups), label_counts=label_counts,
        )
        if attachment:
            groups.append(attachment)
    prior_roles = [group.role for group in groups]
    for index, (relation, span) in enumerate(governed_segments[1:], start=1):
        if len(groups) >= MAX_PARSER_CONCEPTS:
            break
        role, search_role, reason, confidence, core = _assign_segment_role(
            relation, index, len(governed_segments), "question", prior_roles,
            False,
            governed_segments[index + 1][0]
            if index + 1 < len(governed_segments) else None,
            span.text(original),
        )
        group = (
            _make_ambiguous_group(
                original, span, role=role, search_role=search_role, reason=reason,
                order=len(groups), label_counts=label_counts, core_attachment=core,
            ) if relation == "coordinated_attachment" else _make_group(
                original, span, role=role, search_role=search_role, reason=reason,
                confidence=confidence, order=len(groups), label_counts=label_counts,
                core_attachment=core,
            )
        )
        if group:
            groups.append(group)
            prior_roles.append(group.role)
    return groups


def _finalize_structural_draft(groups: list[ConceptGroup]) -> StructuredQueryDraft:
    _select_compiled_groups(groups)
    uncertainties = _merge_uncertainty_terms([
        group.canonical_text
        for group in groups
        if (
            group.coordination == "ambiguous"
            or group.confidence == "low"
            or any(
                marker in group.status_reason.casefold()
                for marker in ("incomplete", "ambiguous", "residual")
            )
        )
    ])
    return StructuredQueryDraft(
        groups=groups[:MAX_PARSER_CONCEPTS], needs_grounding=True,
        uncertain_terms=uncertainties,
    )


def _merge_uncertainty_terms(*collections: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for collection in collections:
        for value in collection:
            cleaned = _clean_term(value)
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
            if len(merged) >= MAX_UNCERTAINTIES:
                return merged
    return merged


def _lossless_structural_draft(question: str) -> StructuredQueryDraft:
    _validate_question_length(question)

    if _requires_whole_phrase_fallback(question):
        whole = _trim_range(question, 0, len(question))
        if whole is None:
            raise ValueError("research question is required")

        label_counts: dict[str, int] = {}
        group = _make_ambiguous_group(
            question,
            whole,
            role="other",
            search_role="required",
            reason="non_question_or_uncertain_input_shape",
            order=0,
            label_counts=label_counts,
            core_attachment=True,
        )

        if group is None:
            raise ValueError(
                "research question contains no searchable concept"
            )

        return _finalize_structural_draft([group])

    body, frame_kind, _, scope_range, clause_mode, auxiliary_inversion = _question_frame(question)
    whole = _trim_range(question, 0, len(question))
    leading = _leading_attachment(question, whole) if whole else None
    comparison_state, comparison = _comparison_frame(question, body)
    if comparison_state == "complete" and comparison:
        label_counts: dict[str, int] = {}
        groups = _comparison_groups(
            question, comparison, leading, label_counts,
            frame_kind=frame_kind, clause_mode=clause_mode,
            auxiliary_inversion=auxiliary_inversion,
        )
        if groups:
            if scope_range and len(groups) < MAX_PARSER_CONCEPTS:
                scope_group = _make_group(
                    question, scope_range, role="domain", search_role="required",
                    reason="explicit_review_scope", confidence="high", order=len(groups),
                    label_counts=label_counts, core_attachment=True,
                )
                if scope_group:
                    groups.append(scope_group)
            return _finalize_structural_draft(groups)
        comparison_state = "ambiguous"
    if comparison_state == "ambiguous":
        label_counts = {}
        ambiguous = _make_ambiguous_group(
            question, body, role="other", search_role="required",
            reason="incomplete_comparison_frame", order=0,
            label_counts=label_counts, core_attachment=True,
        )
        groups = []
        if ambiguous:
            groups.append(ambiguous)
        if leading and len(groups) < MAX_PARSER_CONCEPTS:
            attachment_group = _leading_attachment_group(
                question, leading, order=len(groups), label_counts=label_counts,
            )
            if attachment_group:
                groups.append(attachment_group)
        if groups:
            return _finalize_structural_draft(groups)
    core_text = body.text(question)
    if _ambiguous_finite_comparison(core_text):
        label_counts = {}
        group = _make_ambiguous_group(
            question, body, role="other", search_role="required",
            reason="incomplete_comparison_frame", order=0,
            label_counts=label_counts, core_attachment=True,
        )
        if group:
            return _finalize_structural_draft([group])
    independent_core = _independently_governed_clauses(question, body)
    used_construction = bool(re.search(r"\b(?:used|using|use)\b", core_text, re.I))
    comparison_relation = bool(
        comparison_state == "none" and COMPARISON_MARKER_PATTERN.search(core_text)
    )
    finite_comparison_relation = bool(re.search(
        r"\s+compare(?:s)?\s+(?:with|to)\s+", core_text, re.I,
    ))
    nominal_relation = bool(re.match(
        r"^(?:the\s+)?(?:combined|joint)\s+(?:effect|impact)\s+of\s+",
        core_text, re.I,
    ))
    allow_core_predicate = (
        frame_kind not in {"effectiveness", "impact"}
        and len(independent_core) <= 1
        and not used_construction
        and not comparison_relation
    )
    core_candidates = (
        _predicate_candidates(core_text, prefer_earlier_viable=True)
        if allow_core_predicate else []
    )
    core_predicate = (
        _select_governing_predicate(core_text, prefer_earlier_viable=True)
        if allow_core_predicate else None
    )
    if core_predicate:
        label_counts = {}
        groups = _core_clause_groups(
            question, body, core_predicate, leading, label_counts,
            subject_role="technology" if scope_range else "intervention_or_method",
            frame_kind=frame_kind, clause_mode=clause_mode,
            auxiliary_inversion=auxiliary_inversion,
        )
        if groups:
            if scope_range and len(groups) < MAX_PARSER_CONCEPTS:
                scope_group = _make_group(
                    question, scope_range, role="domain", search_role="required",
                    reason="explicit_review_scope", confidence="high", order=len(groups),
                    label_counts=label_counts, core_attachment=True,
                )
                if scope_group:
                    groups.insert(0, scope_group)
            return _finalize_structural_draft(groups)
    elif core_candidates:
        label_counts = {}
        ambiguous = _make_ambiguous_group(
            question, body, role="other", search_role="required",
            reason="ambiguous_predicate_candidates", order=0,
            label_counts=label_counts, core_attachment=True,
        )
        groups = []
        if ambiguous:
            groups.append(ambiguous)
        if leading:
            attachment = _leading_attachment_group(
                question, leading, order=len(groups), label_counts=label_counts,
            )
            if attachment:
                groups.append(attachment)
        if groups:
            return _finalize_structural_draft(groups)
    if (
        frame_kind == "question" and not used_construction
        and len(independent_core) <= 1
        and not comparison_relation and not finite_comparison_relation
        and not nominal_relation
    ):
        label_counts = {}
        ambiguous = _make_ambiguous_group(
            question, body, role="other", search_role="required",
            reason="ambiguous_unestablished_question_predicate",
            order=0, label_counts=label_counts, core_attachment=True,
        )
        groups = []
        if ambiguous:
            groups.append(ambiguous)
        if leading and len(groups) < MAX_PARSER_CONCEPTS:
            attachment = _leading_attachment_group(
                question, leading, order=len(groups), label_counts=label_counts,
            )
            if attachment:
                groups.append(attachment)
        if groups:
            return _finalize_structural_draft(groups)
    segments = _segment_ranges(question, body)
    groups: list[ConceptGroup] = []
    label_counts: dict[str, int] = {}
    prior_roles: list[str] = []
    if scope_range:
        scope_group = _make_group(
            question, scope_range, role="domain", search_role="required",
            reason="explicit_review_scope", confidence="high", order=0,
            label_counts=label_counts, core_attachment=True,
        )
        if scope_group:
            groups.append(scope_group)
            prior_roles.append(scope_group.role)
    relation_counts: dict[str, int] = {}
    for index, (relation, span) in enumerate(segments):
        if len(groups) >= MAX_PARSER_CONCEPTS - 1 and index < len(segments) - 1:
            residual = _trim_range(question, span.start, body.end)
            if residual:
                residual_group = _make_ambiguous_group(
                    question, residual, role="other", search_role="required",
                    reason="bounded_residual_fallback_span",
                    order=len(groups), label_counts=label_counts, core_attachment=False,
                )
                if residual_group:
                    groups.append(residual_group)
            break
        if len(groups) >= MAX_PARSER_CONCEPTS:
            break
        relation_key = relation or ""
        relation_counts[relation_key] = relation_counts.get(relation_key, 0) + 1
        if index == 0:
            core_text = span.text(question)
            independent_clauses = _independently_governed_clauses(question, span)
            if len(independent_clauses) > 1:
                for governed_clause in independent_clauses:
                    clause = governed_clause.span
                    predicate = governed_clause.predicate
                    if len(groups) >= MAX_PARSER_CONCEPTS:
                        break
                    subject = _trim_range(
                        question, clause.start, clause.start + predicate.start,
                    )
                    outcome = _trim_range(
                        question, clause.start + predicate.end, clause.end,
                    )
                    for item, role, reason in (
                        (subject, "intervention_or_method", "independently_governed_subject"),
                        (outcome, "outcome", "independently_governed_predicate_object"),
                    ):
                        if not item or len(groups) >= MAX_PARSER_CONCEPTS:
                            continue
                        clause_group = _make_group(
                            question, item, role=role, search_role="required",
                            reason=reason, confidence="high", order=len(groups),
                            label_counts=label_counts, core_attachment=True,
                        )
                        if clause_group:
                            clause_group.coordination = "co_required"
                            groups.append(clause_group)
                            prior_roles.append(clause_group.role)
                continue
            predicate = (
                None if frame_kind in {"effectiveness", "impact"}
                else _select_governing_predicate(
                    core_text, prefer_earlier_viable=True,
                )
            )
            if predicate:
                subject = _trim_range(question, span.start, span.start + predicate.start)
                outcome = _trim_range(question, span.start + predicate.end, span.end)
                if subject:
                    subject_group = _make_group(
                        question, subject, role="intervention_or_method",
                        search_role="required", reason="direct_core_subject",
                        confidence="high", order=len(groups), label_counts=label_counts,
                        core_attachment=True,
                    )
                    if subject_group:
                        groups.append(subject_group)
                        prior_roles.append(subject_group.role)
                if outcome and len(groups) < MAX_PARSER_CONCEPTS:
                    outcome_group = _make_group(
                        question, outcome, role="outcome", search_role="required",
                        reason="direct_core_predicate_object", confidence="high",
                        order=len(groups), label_counts=label_counts,
                        core_attachment=True,
                    )
                    if outcome_group:
                        groups.append(outcome_group)
                        prior_roles.append(outcome_group.role)
                continue
            trailing_use = re.search(
                r"\s+(?:(?:is|are|was|were|be|been|being)\s+)?(?:used|using|use)\s*$",
                core_text, re.I,
            )
            if trailing_use:
                trimmed_subject = _trim_range(
                    question, span.start, span.start + trailing_use.start(),
                )
                if trimmed_subject:
                    span = trimmed_subject
            role = "intervention_or_method" if frame_kind in {"impact", "effectiveness", "extent"} else "technology"
            canonical_subject = span.text(question)
            if frame_kind in {"impact", "effectiveness"}:
                canonical_subject = re.sub(
                    r"^(?:a|an|the)\s+", "", canonical_subject, flags=re.I,
                )
            group = _make_group(
                question, span, role=role, search_role="required",
                reason="direct_core_subject", confidence="high", order=len(groups),
                label_counts=label_counts, core_attachment=True,
                canonical_value=(
                    canonical_subject
                    if canonical_subject != span.text(question) else None
                ),
            )
        else:
            role, search_role, reason, confidence, core = _assign_segment_role(
                relation, index, len(segments), frame_kind, prior_roles,
                relation_counts[relation_key] > 1,
                segments[index + 1][0] if index + 1 < len(segments) else None,
                span.text(question),
            )
            if relation == "coordinated_attachment":
                group = _make_ambiguous_group(
                    question, span, role=role, search_role=search_role, reason=reason,
                    order=len(groups), label_counts=label_counts, core_attachment=core,
                )
            elif role == "outcome" and search_role == "required":
                outcome_structure = _canonicalize_outcome(
                    question, span, frame_kind=frame_kind, owning_relation=relation,
                    clause_mode=clause_mode, auxiliary_inversion=auxiliary_inversion,
                    first_predicate=None,
                )
                group = (
                    _make_structural_outcome_group(
                        question, span, outcome_structure, order=len(groups),
                        label_counts=label_counts,
                    ) if outcome_structure else None
                )
            else:
                group = _make_group(
                    question, span, role=role, search_role=search_role, reason=reason,
                    confidence=confidence, order=len(groups), label_counts=label_counts,
                    core_attachment=core,
                )
        if group:
            groups.append(group)
            prior_roles.append(group.role)
    if leading and len(groups) < MAX_PARSER_CONCEPTS:
        attachment_group = _leading_attachment_group(
            question, leading, order=len(groups), label_counts=label_counts,
        )
        if attachment_group:
            groups.append(attachment_group)
    if not groups:
        groups = []
        label_counts = {}
        group = _make_ambiguous_group(
            question, body, role="other", search_role="required",
            reason="residual_fallback_span", order=0,
            label_counts=label_counts, core_attachment=True,
        )
        if group:
            groups.append(group)
    return _finalize_structural_draft(groups)


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


def _sanitize_draft(
    draft: StructuredQueryDraft,
    question: str,
    seed: StructuredQueryDraft | None = None,
) -> tuple[StructuredQueryDraft, list[str], list[str], list[dict[str, Any]]]:
    seed = seed or _deterministic_seed(question)
    groups = [group.model_copy(deep=True) for group in seed.groups]
    represented: set[tuple[str, str]] = set()
    for model_group in draft.groups[:MAX_PARSER_CONCEPTS]:
        valid_spans = [
            _clean_term(span) for span in model_group.source_spans
            if _clean_term(span) and _span_is_verbatim(span, question)
        ]
        if not valid_spans:
            continue
        target = max(
            groups,
            key=lambda group: len(
                set().union(*(_content_tokens(span) for span in group.source_spans))
                & set().union(*(_content_tokens(span) for span in valid_spans))
            ),
        )
        overlap = set().union(*(_content_tokens(span) for span in target.source_spans)) & set().union(*(
            _content_tokens(span) for span in valid_spans
        ))
        if not overlap:
            continue
        represented.update((target.label, _normalize(span)) for span in target.source_spans)
        for term in model_group.terms:
            cleaned = _clean_term(term)
            if (
                cleaned and len(target.terms) < MAX_TERMS_PER_GROUP
                and _is_safe_variant(cleaned, target)
                and _normalize(cleaned) not in {_normalize(value) for value in target.terms}
            ):
                target.terms.append(cleaned)
    uncovered = [
        span for group in groups for span in group.source_spans
        if (group.label, _normalize(span)) not in represented
    ]
    repaired = list(uncovered)
    corrections: list[dict[str, Any]] = []
    result = StructuredQueryDraft(
        groups=groups[:MAX_PARSER_CONCEPTS],
        needs_grounding=draft.needs_grounding,
        uncertain_terms=_merge_uncertainty_terms(
            seed.uncertain_terms, draft.uncertain_terms,
        ),
    )
    return result, uncovered, repaired, corrections


def _is_safe_variant(term: str, group: ConceptGroup) -> bool:
    normalized = _normalize(term)
    if not normalized:
        return False
    for span in group.source_spans:
        if normalized == _normalize(span):
            return True
    return normalized in {_normalize(value) for value in group.deterministic_variants}


def _default_search_roles(groups: list[ConceptGroup]) -> dict[str, dict[str, Any]]:
    return {
        _normalize(group.label): {
            "search_role": group.search_role,
            "balanced_use": False,
        }
        for group in groups
    }


def _apply_balanced_use(
    roles: dict[str, dict[str, Any]],
    groups: list[ConceptGroup],
    balanced_groups: list[ConceptGroup],
) -> None:
    balanced_identities = {
        (group.label, group.source_order) for group in balanced_groups
    }
    for group in groups:
        identity = (group.label, group.source_order)
        roles[_normalize(group.label)]["balanced_use"] = (
            group.search_role == "required"
            and group.compiled
            and identity in balanced_identities
        )


def _local_search_roles(groups: list[ConceptGroup]) -> dict[str, dict[str, Any]]:
    """Local preserves the parser-owned structural classification."""
    return _default_search_roles(groups)


def _select_query_groups(
    draft: StructuredQueryDraft,
    roles: dict[str, dict[str, Any]],
    accepted: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> tuple[list[ConceptGroup], list[ConceptGroup], str]:
    del accepted, sources
    required = [
        group for group in draft.groups
        if (
            roles[_normalize(group.label)]["search_role"] == "required"
            and group.compiled
        )
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
    for group in groups[:MAX_COMPILED_REQUIRED_GROUPS]:
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


def _gemini_direct_schema(question: str) -> type[GeminiDirectConceptProposal]:
    normalized_question = _normalize(question)

    class _QuestionBoundGeminiProposal(GeminiDirectConceptProposal):
        @model_validator(mode="after")
        def reject_whole_input_term(self) -> "_QuestionBoundGeminiProposal":
            if normalized_question and any(
                _normalize(term) == normalized_question
                for concept in self.concepts
                for term in (*concept.balanced_terms, *concept.high_recall_terms)
            ):
                raise ValueError(
                    "Gemini must not return the complete input as one search term"
                )
            return self

    _QuestionBoundGeminiProposal.__name__ = "GeminiDirectConceptProposal"
    _QuestionBoundGeminiProposal.__qualname__ = "GeminiDirectConceptProposal"
    return _QuestionBoundGeminiProposal


def _direct_concept_group(
    concept: GeminiDirectConcept,
    terms: list[str],
    index: int,
) -> ConceptGroup:
    return ConceptGroup(
        label=concept.label.strip(),
        role="other",
        terms=[_clean_term(value) for value in terms],
        source_spans=[],
        canonical_text=" OR ".join(_clean_term(value) for value in terms),
        search_role="required",
        status_reason="gemini_direct_required_concept",
        coordination="alternatives" if len(terms) > 1 else "single",
        confidence="medium",
        source_offsets=[],
        source_order=index,
        deterministic_variants=[],
        compiled=True,
        core_attachment=True,
    )


def _generate_direct_gemini_bundle(
    question: str,
    *,
    model: str | None,
    deadline_seconds: float,
    engine: Any | None,
) -> GeneratedQueryBundle:
    if engine is None:
        from litsync_app.query.engines import GeminiWebQueryEngine
        engine = GeminiWebQueryEngine()

    selected_model = model or GEMINI_WEB_ENGINE
    schema = _gemini_direct_schema(question)
    prompt = (
        f"{GEMINI_DIRECT_CONCEPT_PROMPT}\n\n"
        f"Research question or paragraph:\n{question}"
    )
    started = time.monotonic()
    result = engine.generate(
        selected_model,
        prompt,
        schema,
        timeout_seconds=deadline_seconds,
    )
    proposal = schema.model_validate(result.value)

    balanced_groups = [
        _direct_concept_group(concept, concept.balanced_terms, index)
        for index, concept in enumerate(proposal.concepts)
    ]
    high_recall_groups = [
        _direct_concept_group(concept, concept.high_recall_terms, index)
        for index, concept in enumerate(proposal.concepts)
    ]

    term_details: list[dict[str, Any]] = []
    expansion_proposals: list[dict[str, Any]] = []
    for concept in proposal.concepts:
        balanced_keys = {_normalize(value) for value in concept.balanced_terms}
        for term in concept.high_recall_terms:
            compiled_versions = (
                ["balanced", "high_recall"]
                if _normalize(term) in balanced_keys else ["high_recall"]
            )
            term_details.append({
                "term": _clean_term(term),
                "group": concept.label.strip(),
                "source": "ai_assisted_query_expansion",
                "proposal_source": "ai_assisted_query_expansion",
                "term_type": "ai_assisted_query_expansion",
                "validation_status": "accepted",
                "compiled_versions": compiled_versions,
                "supporting_paper_ids": [],
                "source_offsets": [],
            })
            expansion_proposals.append({
                "term": _clean_term(term),
                "group": concept.label.strip(),
                "term_type": "ai_assisted_query_expansion",
                "status": "accepted",
                "proposal_source": "ai_assisted_query_expansion",
                "compiled_versions": compiled_versions,
            })

    query_debug = {
        "raw_source_records_returned_count": 0,
        "schema_valid_source_count": 0,
        "deduplicated_source_count": 0,
        "usable_source_count": 0,
        "source_rejections_by_reason": {},
        "proposal_count": len(expansion_proposals),
        "accepted_proposal_count": len(expansion_proposals),
        "rejected_proposal_count": 0,
        "proposal_rejections_by_reason": {},
    }
    groups_payload = [group.model_dump() for group in high_recall_groups]
    concepts = {
        "groups": groups_payload,
        "concept_classifications": [
            {
                "group_label": group.label,
                "search_role": "required",
                "balanced_use": True,
                "status_reason": group.status_reason,
                "confidence": group.confidence,
                "coordination": group.coordination,
                "compiled": True,
            }
            for group in high_recall_groups
        ],
        "concept_counts": {
            "required": len(high_recall_groups),
            "optional": 0,
            "screening_only": 0,
            "compiled_required": len(high_recall_groups),
        },
        "uncompiled_required_groups": [],
        "parser_warnings": [],
        "term_details": term_details,
        "expansion_proposals": expansion_proposals,
        "grounded_terms": [],
        "grounding_papers": [],
        "gemini_reported_sources": [],
        "rejected_sources": [],
        "evidence_label": "AI-generated query",
        "evidence_limitation": (
            "Gemini-generated terminology was not independently checked against "
            "academic literature."
        ),
        "evidence_level": "not_literature_checked",
        "reported_outcome": "not_available",
        "no_evidence_reason": "",
        "usable_source_count": 0,
        "selected_optional_block": "",
        "model": selected_model,
        "processing_engine": GEMINI_WEB_ENGINE,
        "engine_display_name": "Gemini Web Automation",
        "mode": "ai_assisted",
        "needs_grounding": False,
        "uncertain_terms": [],
        "fallback_reason": "",
        "generation_status": "ai_assisted_expansion",
        "warning": "",
        "literal_coverage": 0.0,
        "uncovered_spans": [],
        "repaired_spans": [],
        "corrections": [],
        "removed_scaffolding": "",
        "timings": {
            "structured_generation_ms": round(
                (time.monotonic() - started) * 1000, 1,
            ),
        },
        "deadline_seconds": deadline_seconds,
        "query_debug": query_debug,
    }
    return _make_bundle(balanced_groups, high_recall_groups, concepts)


def _is_explicit_original_acronym(term: str, group: ConceptGroup) -> bool:
    if not _bounded_acronym(term):
        return False
    acronym_present = any(
        _normalize(term) == _normalize(span) for span in group.source_spans
    )
    return acronym_present and any(
        span != term and _initials_match(span, term) for span in group.source_spans
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
    del corrections
    if _is_explicit_original_acronym(term, group):
        return "explicit_original_acronym"
    if any(
        _clean_term(term).casefold() == _clean_term(span).casefold()
        for span in group.source_spans
    ):
        return "literal"
    if _normalize(term) in {
        _normalize(value) for value in group.canonical_text.split(" OR ")
    }:
        return "parser_normalization"
    if _normalize(term) in seed_terms:
        return "morphology"
    if _normalize(term) in {_normalize(value) for value in group.deterministic_variants}:
        return "morphology"
    raise ValueError(f"term lacks admissible provenance: {term}")


def _aligned_term_source_offsets(
    term: str,
    group: ConceptGroup,
) -> list[dict[str, int | str]]:
    normalized_term = _normalize(term)
    if not normalized_term:
        return []

    exact = [
        offset
        for offset in group.source_offsets
        if _normalize(str(offset.get("text") or ""))
        == normalized_term
    ]
    if exact:
        return exact

    containing: list[dict[str, int | str]] = []
    term_pattern = re.compile(
        rf"(?:^|\s){re.escape(normalized_term)}(?:$|\s)"
    )

    for offset in group.source_offsets:
        normalized_source = _normalize(
            str(offset.get("text") or "")
        )
        if normalized_source and term_pattern.search(normalized_source):
            containing.append(offset)

    if containing:
        return [
            min(
                containing,
                key=lambda item: (
                    int(item.get("end", 0))
                    - int(item.get("start", 0))
                ),
            )
        ]

    return group.source_offsets


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
    original_question = str(question or "")
    _validate_question_length(original_question)
    question = original_question
    if not question.strip():
        raise ValueError("research question is required")
    selected_processing_engine = str(processing_engine or LOCAL_ENGINE).strip().lower()
    if selected_processing_engine not in {LOCAL_ENGINE, GEMINI_WEB_ENGINE}:
        raise ValueError(
            f"Unsupported query-generation engine: {processing_engine}. "
            f"Choose '{LOCAL_ENGINE}' or '{GEMINI_WEB_ENGINE}'."
        )
    maximum_deadline = (
        GEMINI_WEB_QUERY_DEADLINE_SECONDS
        if selected_processing_engine == GEMINI_WEB_ENGINE
        else QUERY_DEADLINE_SECONDS
    )
    if (
        selected_processing_engine == GEMINI_WEB_ENGINE
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
    if selected_processing_engine == GEMINI_WEB_ENGINE:
        return _generate_direct_gemini_bundle(
            question,
            model=model,
            deadline_seconds=deadline_seconds,
            engine=engine,
        )

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
    profile = profile or resolve_runtime_profile()
    selected_model = model or _select_model(profile)
    engine = engine or OllamaStructuredEngine(profile)
    prompt = (
        f"{SYSTEM_PROMPT}\n\nThese parser-owned literal spans must all be represented. "
        "Local output may add only terms explicitly represented by those spans, source-present "
        "dehyphenated variants, or directly demonstrated acronyms."
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

    _apply_balanced_use(roles, draft.groups, balanced_groups)

    required_groups = [
        group for group in draft.groups if group.search_role == "required"
    ]
    optional_groups = [
        group for group in draft.groups if group.search_role == "optional"
    ]
    screening_groups = [
        group for group in draft.groups if group.search_role == "screening_only"
    ]
    compiled_required = [group for group in required_groups if group.compiled]
    overflow_required = [group for group in required_groups if not group.compiled]
    parser_warnings: list[dict[str, Any]] = []
    if len(compiled_required) > 5:
        parser_warnings.append({
            "code": "more_than_five_required_groups",
            "message": (
                "More than five mandatory AND blocks are compiled; retrieval recall may decrease."
            )[:MAX_BOUNDED_MESSAGE_LENGTH],
        })
    if overflow_required:
        parser_warnings.append({
            "code": "required_group_compile_limit_exceeded",
            "message": (
                "More than eight required concepts were detected; overflow concepts remain "
                "required metadata but are not compiled."
            )[:MAX_BOUNDED_MESSAGE_LENGTH],
        })
    if parser_warnings:
        parser_warning_text = " ".join(item["message"] for item in parser_warnings)
        warning = f"{warning} {parser_warning_text}".strip()[:MAX_BOUNDED_MESSAGE_LENGTH]

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
            source_offsets = (
                []
                if source == "ai_assisted_query_expansion"
                else _aligned_term_source_offsets(term, group)
            )
            term_details.append({
                "term": term,
                "group": group.label,
                "source": source,
                "proposal_source": proposal_detail.get("proposal_source", source),
                "term_type": proposal_detail.get("term_type", source),
                "validation_status": "accepted",
                "compiled_versions": proposal_detail.get(
                    "compiled_versions",
                    ["balanced", "high_recall"]
                    if group.compiled and group.search_role == "required" else [],
                ),
                "supporting_paper_ids": [],
                "source_offsets": source_offsets,
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
                "status_reason": group.status_reason,
                "confidence": group.confidence,
                "coordination": group.coordination,
                "compiled": group.compiled,
            }
            for group in draft.groups
        ],
        "concept_counts": {
            "required": len(required_groups),
            "optional": len(optional_groups),
            "screening_only": len(screening_groups),
            "compiled_required": len(compiled_required),
        },
        "uncompiled_required_groups": [
            {
                "group_label": group.label,
                "canonical_text": group.canonical_text,
                "source_spans": group.source_spans,
                "search_role": "required",
            }
            for group in overflow_required
        ],
        "parser_warnings": parser_warnings,
        "parser_bounds": {
            "max_question_codepoints": MAX_QUESTION_CODEPOINTS,
            "max_clauses": MAX_PARSER_CLAUSES,
            "max_coordination_candidates": MAX_COORDINATION_CANDIDATES,
            "max_parser_concepts": MAX_PARSER_CONCEPTS,
            "max_compiled_required_groups": MAX_COMPILED_REQUIRED_GROUPS,
            "max_source_spans_per_group": MAX_SOURCE_SPANS_PER_GROUP,
            "max_terms_per_group": MAX_TERMS_PER_GROUP,
            "max_uncertainty_records": MAX_UNCERTAINTIES,
            "max_bounded_message_length": MAX_BOUNDED_MESSAGE_LENGTH,
        },
        "term_details": term_details,
        "expansion_proposals": [*accepted, *rejected],
        "grounded_terms": [],
        "grounding_papers": [],
        "gemini_reported_sources": [],
        "rejected_sources": [],
        "evidence_label": "local_model_proposed",
        "evidence_limitation": "Local model proposals are not academically grounded.",
        "evidence_level": evidence_level,
        "reported_outcome": "not_available",
        "no_evidence_reason": "",
        "usable_source_count": 0,
        "selected_optional_block": "",
        "model": selected_model,
        "processing_engine": selected_processing_engine,
        "engine_display_name": "Local Ollama",
        "mode": "local",
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
