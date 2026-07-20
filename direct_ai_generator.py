"""Latency-bounded, corpus-grounded Boolean query generation."""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field

from local_ai.engine import LocalAIError, OllamaStructuredEngine
from local_ai.hardware import RuntimeProfile, resolve_runtime_profile


QUERY_DEADLINE_SECONDS = 15.0
GROUNDING_TIMEOUT_SECONDS = 2.5
DEFAULT_QUERY_MODELS = (
    "qwen3.5:4b",
    "qwen3:4b-instruct-2507-q4_K_M",
    "phi4-mini:3.8b-q4_K_M",
)
ALLOWED_TERM_SOURCES = frozenset({
    "literal", "morphology", "source_acronym", "typo_correction",
    "validated_model", "corpus",
})


class ConceptGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=60)
    role: Literal["technology", "population", "domain", "comparison", "outcome", "context", "other"]
    terms: list[str] = Field(min_length=1, max_length=8)
    source_spans: list[str] = Field(default_factory=list, max_length=5)


class StructuredQueryDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: list[ConceptGroup] = Field(min_length=2, max_length=4)
    needs_grounding: bool = False
    uncertain_terms: list[str] = Field(default_factory=list, max_length=5)


class GroundedAddition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_label: str = Field(min_length=1, max_length=60)
    term: str = Field(min_length=1, max_length=80)
    support_ids: list[str] = Field(default_factory=list, max_length=8)


class GroundedRefinement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    additions: list[GroundedAddition] = Field(default_factory=list, max_length=12)


@dataclass(frozen=True)
class GroundingPaper:
    paper_id: str
    title: str
    abstract: str
    url: str = ""
    year: int | None = None

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract}".strip()


@dataclass
class GeneratedQueryBundle:
    google_scholar: str
    scopus: str
    web_of_science: str
    ieee_xplore: str
    pubmed: str
    concepts: dict[str, Any]

    def to_api_response(self) -> dict[str, Any]:
        return {
            "status": "success",
            "google_scholar": self.google_scholar,
            "scopus": self.scopus,
            "web_of_science": self.web_of_science,
            "ieee_xplore": self.ieee_xplore,
            "pubmed": self.pubmed,
            "concepts": self.concepts,
        }


class SemanticScholarGrounder:
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
    _cache: dict[str, tuple[float, list[GroundingPaper]]] = {}
    _cache_lock = threading.Lock()

    def search(self, question: str, timeout: float = GROUNDING_TIMEOUT_SECONDS) -> list[GroundingPaper]:
        cache_key = _normalize(question)
        cache_ttl = float(os.getenv("QUERY_GROUNDING_CACHE_SECONDS", "900"))
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached and time.monotonic() - cached[0] <= cache_ttl:
                return list(cached[1])
        headers = {"User-Agent": "LitSync/2.0 corpus-grounded-query-generation"}
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        if api_key:
            headers["x-api-key"] = api_key
        response = requests.get(
            self.endpoint,
            params={
                "query": re.sub(r"[-–—]", " ", question),
                "limit": 8,
                "fields": "paperId,title,abstract,url,year",
            },
            headers=headers,
            timeout=max(0.1, min(timeout, GROUNDING_TIMEOUT_SECONDS)),
        )
        response.raise_for_status()
        papers = []
        for item in response.json().get("data", []):
            title = str(item.get("title") or "").strip()
            abstract = str(item.get("abstract") or "").strip()
            if not title:
                continue
            papers.append(GroundingPaper(
                paper_id=str(item.get("paperId") or title),
                title=title,
                abstract=abstract,
                url=str(item.get("url") or ""),
                year=item.get("year") if isinstance(item.get("year"), int) else None,
            ))
        with self._cache_lock:
            self._cache[cache_key] = (time.monotonic(), list(papers))
        return papers


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


REFINEMENT_PROMPT = """
Review the draft concept groups against the supplied academic records. Propose only missing direct
synonyms, acronyms, spelling variants, or controlled terms that occur verbatim in at least two
records. Map each proposal to an existing group label and cite the supporting record IDs. Never
remove, replace, broaden, or reinterpret a draft term. Return only schema-valid JSON.
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


def _recurring_candidates(papers: list[GroundingPaper], draft: StructuredQueryDraft) -> list[str]:
    existing = {_normalize(term) for group in draft.groups for term in group.terms}
    document_counts: dict[str, int] = {}
    for paper in papers:
        tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", paper.text.lower())
        seen_in_document = set()
        for size in (2, 3):
            for index in range(0, max(0, len(tokens) - size + 1)):
                phrase_tokens = tokens[index:index + size]
                if sum(token not in GRAMMAR_STOPWORDS and len(token) > 2 for token in phrase_tokens) < 2:
                    continue
                phrase = " ".join(phrase_tokens)
                if phrase not in existing:
                    seen_in_document.add(phrase)
        for phrase in seen_in_document:
            document_counts[phrase] = document_counts.get(phrase, 0) + 1
    ranked = sorted(
        (phrase for phrase, count in document_counts.items() if count >= 2),
        key=lambda phrase: (-document_counts[phrase], -len(phrase)),
    )
    return ranked[:12]


def _supporting_papers(term: str, papers: list[GroundingPaper]) -> list[GroundingPaper]:
    normalized = _normalize(term)
    pattern = re.compile(rf"(?<!\w){re.escape(normalized)}(?!\w)", re.IGNORECASE)
    return [paper for paper in papers if pattern.search(_normalize(paper.text))]


def _group_candidate_score(group: ConceptGroup, candidate: str) -> int:
    candidate_tokens = _content_tokens(candidate)
    anchor_tokens = set().union(*(_content_tokens(term) for term in group.terms))
    score = len(candidate_tokens & anchor_tokens) * 3
    if group.role == "comparison" and not candidate_tokens & anchor_tokens:
        return 0
    return score


def _merge_deterministic_corpus_terms(
    draft: StructuredQueryDraft,
    papers: list[GroundingPaper],
) -> tuple[StructuredQueryDraft, list[dict[str, Any]]]:
    groups = [group.model_copy(deep=True) for group in draft.groups]
    provenance: list[dict[str, Any]] = []
    for candidate in _recurring_candidates(papers, draft):
        supports = _supporting_papers(candidate, papers)
        if len(supports) < 2:
            continue
        target = max(groups, key=lambda group: _group_candidate_score(group, candidate))
        if _group_candidate_score(target, candidate) <= 0:
            continue
        if not _term_is_valid(candidate, target.role):
            continue
        if len(target.terms) >= 8:
            continue
        if _normalize(candidate) in {_normalize(term) for group in groups for term in group.terms}:
            continue
        target.terms.append(candidate)
        provenance.append({
            "term": candidate,
            "group": target.label,
            "source": "corpus",
            "support": [
                {"paper_id": paper.paper_id, "title": paper.title, "url": paper.url}
                for paper in supports[:8]
            ],
        })
    return StructuredQueryDraft(
        groups=groups,
        needs_grounding=draft.needs_grounding,
        uncertain_terms=draft.uncertain_terms,
    ), provenance


def _merge_verified_additions(
    draft: StructuredQueryDraft,
    refinement: GroundedRefinement,
    papers: list[GroundingPaper],
) -> tuple[StructuredQueryDraft, list[dict[str, Any]]]:
    groups = [group.model_copy(deep=True) for group in draft.groups]
    labels = {_normalize(group.label): group for group in groups}
    provenance = []
    for addition in refinement.additions:
        group = labels.get(_normalize(addition.group_label))
        term = _clean_term(addition.term)
        if group is None or not term or len(group.terms) >= 8:
            continue
        normalized_term = _normalize(term)
        content_tokens = [token for token in normalized_term.split() if token not in GRAMMAR_STOPWORDS]
        if len(content_tokens) < 2 and not re.fullmatch(r"[A-Z0-9-]{2,12}", term):
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        supports = [paper for paper in papers if pattern.search(paper.text)]
        if len(supports) < 2:
            continue
        if normalized_term in {_normalize(value) for value in group.terms}:
            continue
        group.terms.append(term)
        provenance.append({
            "term": term,
            "group": group.label,
            "support": [
                {"paper_id": paper.paper_id, "title": paper.title, "url": paper.url}
                for paper in supports[:8]
            ],
        })
    return StructuredQueryDraft(
        groups=groups,
        needs_grounding=draft.needs_grounding,
        uncertain_terms=draft.uncertain_terms,
    ), provenance


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


def _make_bundle(groups: list[ConceptGroup], concepts: dict[str, Any]) -> GeneratedQueryBundle:
    return GeneratedQueryBundle(
        google_scholar=compile_boolean_query(groups, "google_scholar"),
        scopus=f'TITLE-ABS-KEY({compile_boolean_query(groups, "scopus")})',
        web_of_science=f'TS=({compile_boolean_query(groups, "web_of_science")})',
        ieee_xplore=compile_boolean_query(groups, "ieee_xplore"),
        pubmed=compile_boolean_query(groups, "pubmed"),
        concepts=concepts,
    )


def _term_provenance_source(
    term: str,
    group: ConceptGroup,
    seed_terms: set[str],
    corrections: list[dict[str, Any]],
    grounded: dict[str, Any] | None,
) -> str:
    if grounded:
        return "corpus"
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
        if source == "corpus" and not detail.get("supporting_paper_ids"):
            raise ValueError(f"corpus term lacks supporting papers: {detail.get('term', '')}")


def generate_query_bundle(
    question: str,
    model: str | None = None,
    *,
    deadline_seconds: float = QUERY_DEADLINE_SECONDS,
    profile: RuntimeProfile | None = None,
    engine: OllamaStructuredEngine | None = None,
    grounder: SemanticScholarGrounder | None = None,
) -> GeneratedQueryBundle:
    started = time.monotonic()
    deadline_seconds = max(0.5, min(float(deadline_seconds), QUERY_DEADLINE_SECONDS))
    deadline = started + deadline_seconds
    profile = profile or resolve_runtime_profile()
    selected_model = model or _select_model(profile)
    engine = engine or OllamaStructuredEngine(profile)
    grounder = grounder or SemanticScholarGrounder()
    timings: dict[str, float] = {}
    fallback_reason = ""
    provenance: list[dict[str, Any]] = []
    papers: list[GroundingPaper] = []
    seed = _deterministic_seed(question)
    _, topical_scope, removed_prefix = _question_parts(question)
    model_failed = False
    uncovered_spans: list[str] = []
    repaired_spans: list[str] = []
    coverage_repaired = False
    corrections: list[dict[str, Any]] = []

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="query-grounding")
    grounding_future = pool.submit(
        grounder.search, question, min(GROUNDING_TIMEOUT_SECONDS, max(0.1, _remaining(deadline)))
    )
    draft_started = time.monotonic()
    try:
        prompt = (
            f"{SYSTEM_PROMPT}\n\nThese lossless surface spans must all be represented, but you decide "
            "their semantic roles and grouping."
            f"\n\nRequired literal spans:\n{[span for group in seed.groups for span in group.source_spans]}"
            f"\n\nResearch question:\n{question}"
        )
        result = engine.generate(
            selected_model,
            prompt,
            StructuredQueryDraft,
            timeout_seconds=max(0.1, _remaining(deadline) - 3.0),
        )
        draft, uncovered_spans, repaired_spans, corrections = _sanitize_draft(
            StructuredQueryDraft.model_validate(result.value), question, seed
        )
        coverage_repaired = bool(repaired_spans or corrections or topical_scope)
        if coverage_repaired:
            fallback_reason = "paragraph_structure_repaired"
        timings["draft_ms"] = round((time.monotonic() - draft_started) * 1000, 1)
    except (LocalAIError, ValueError, requests.RequestException) as exc:
        draft = seed.model_copy(deep=True)
        model_failed = True
        repaired_spans = []
        uncovered_spans = []
        corrections = []
        fallback_reason = f"local_draft_failed:{type(exc).__name__}"
        timings["draft_ms"] = round((time.monotonic() - draft_started) * 1000, 1)

    try:
        wait_for = min(GROUNDING_TIMEOUT_SECONDS, max(0.0, _remaining(deadline) - 0.25))
        papers = grounding_future.result(timeout=wait_for) if wait_for > 0 else []
    except (FutureTimeout, requests.RequestException, ValueError):
        grounding_future.cancel()
        if not fallback_reason:
            fallback_reason = "grounding_unavailable"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    timings["grounding_ready_ms"] = round((time.monotonic() - started) * 1000, 1)

    grounding_merge_started = time.monotonic()
    if papers:
        draft, provenance = _merge_deterministic_corpus_terms(draft, papers)
    timings["grounding_merge_ms"] = round((time.monotonic() - grounding_merge_started) * 1000, 1)

    query = compile_boolean_query(draft.groups)
    if not query:
        draft = seed.model_copy(deep=True)
        query = compile_boolean_query(draft.groups)
        fallback_reason = fallback_reason or "empty_query_repaired"
        model_failed = True
    final_source_tokens = set().union(*(
        _content_tokens(span) for group in draft.groups for span in group.source_spans
    )) if draft.groups else set()
    required_tokens = set().union(*(
        _content_tokens(span) for group in seed.groups for span in group.source_spans
    )) if seed.groups else set()
    literal_coverage = round(len(final_source_tokens & required_tokens) / max(1, len(required_tokens)), 4)
    final_uncovered = [
        span for group in seed.groups for span in group.source_spans
        if not _content_tokens(span).issubset(final_source_tokens)
    ]

    generation_status = "full"
    warning = ""
    if model_failed:
        generation_status = "grounded_fallback" if papers else "local_fallback"
        warning = (
            "The local model was unavailable or exceeded its deadline. "
            "LitSync returned validated parser-first concept groups"
            + (" with corpus-supported expansions." if papers else ".")
        )
    elif fallback_reason:
        if coverage_repaired:
            generation_status = "repaired"
            warning = (
                "LitSync removed question framing and repaired source-linked spans or spelling "
                "before compiling."
            )
        else:
            generation_status = "local_fallback"
            warning = "Academic grounding was unavailable; these results use validated local concepts only."

    seed_terms = {_normalize(term) for group in seed.groups for term in group.terms}
    grounded_by_term = {_normalize(item["term"]): item for item in provenance}
    term_details = []
    for group in draft.groups:
        for term in group.terms:
            grounded = grounded_by_term.get(_normalize(term))
            source = _term_provenance_source(
                term, group, seed_terms, corrections, grounded
            )
            term_details.append({
                "term": term,
                "group": group.label,
                "source": source,
                "supporting_paper_ids": [
                    support["paper_id"] for support in (grounded or {}).get("support", [])
                ],
            })
    _validate_term_details(term_details)
    timings["total_ms"] = round((time.monotonic() - started) * 1000, 1)
    mode = "grounded" if papers else "local"
    return _make_bundle(draft.groups, {
        "groups": [group.model_dump() for group in draft.groups],
        "term_details": term_details,
        "grounded_terms": provenance,
        "grounding_papers": [
            {"paper_id": paper.paper_id, "title": paper.title, "url": paper.url, "year": paper.year}
            for paper in papers
        ],
        "model": selected_model,
        "mode": mode,
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
    })


def generate_query(question: str, model: str | None = None) -> str:
    """Backward-compatible query-only entry point."""
    return generate_query_bundle(question, model=model).google_scholar
