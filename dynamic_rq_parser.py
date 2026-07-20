"""Backward-compatible, vocabulary-free research-question parsing helpers."""

from __future__ import annotations

import re
from collections import Counter

from direct_ai_generator import decompose_literal_question


def _join(values) -> str:
    return "; ".join(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def parse_dynamic_research_question(research_question):
    """Expose the neutral source-span parser through the legacy frame-shaped API."""
    question = str(research_question or "").strip()
    if not question:
        return {
            "method_or_technology": "", "application_context": "",
            "target_tasks_or_outcomes": "", "method_synonyms": "",
            "context_synonyms": "", "domain_synonyms": "",
            "task_outcome_synonyms": "", "required_inclusion_concepts": "",
            "rq_dynamic_extraction_used": "False", "rq_dynamic_extraction_reason": "",
            "rq_dynamic_source_text": "",
        }
    draft = decompose_literal_question(question)
    by_role: dict[str, list[str]] = {}
    for group in draft.groups:
        by_role.setdefault(group.role, []).extend(group.source_spans)
    methods = by_role.get("technology", [])
    outcomes = by_role.get("outcome", [])
    contexts = [
        *by_role.get("domain", []), *by_role.get("population", []),
        *by_role.get("context", []), *by_role.get("comparison", []),
        *by_role.get("other", []),
    ]
    return {
        "method_or_technology": _join(methods),
        "application_context": _join(contexts),
        "target_tasks_or_outcomes": _join(outcomes),
        "method_synonyms": _join(methods),
        "context_synonyms": _join(contexts),
        "domain_synonyms": _join(contexts),
        "task_outcome_synonyms": _join(outcomes),
        "required_inclusion_concepts": _join([*methods, *contexts]),
        "rq_dynamic_extraction_used": str(bool(draft.groups)),
        "rq_dynamic_extraction_reason": "source_span_structural_decomposition",
        "rq_dynamic_source_text": question,
    }


def mine_dynamic_corpus_terms(rows, title_col, abstract_col, sample_size=30, rq_frame=None):
    """Mine recurring corpus phrases without subject-matter cues or forced synonyms."""
    anchors = {
        "method": _tokens((rq_frame or {}).get("method_or_technology", "")),
        "task": _tokens((rq_frame or {}).get("target_tasks_or_outcomes", "")),
        "context": _tokens((rq_frame or {}).get("application_context", "")),
    }
    counters = {name: Counter() for name in (*anchors, "profile")}
    for _, row in rows.head(sample_size).iterrows():
        text = f"{row.get(title_col, '')} {row.get(abstract_col, '')}".lower()
        words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
        seen = set()
        for size in (2, 3):
            for index in range(len(words) - size + 1):
                phrase = " ".join(words[index:index + size])
                if phrase in seen:
                    continue
                seen.add(phrase)
                phrase_tokens = _tokens(phrase)
                counters["profile"][phrase] += 1
                for role, role_tokens in anchors.items():
                    if phrase_tokens & role_tokens:
                        counters[role][phrase] += 1

    def supported(counter: Counter) -> str:
        return _join(term for term, count in counter.most_common(30) if count >= 2)

    return {
        "corpus_dynamic_method_terms": supported(counters["method"]),
        "corpus_dynamic_task_terms": supported(counters["task"]),
        "corpus_dynamic_context_terms": supported(counters["context"]),
        "corpus_dynamic_profile_terms": supported(counters["profile"]),
    }
