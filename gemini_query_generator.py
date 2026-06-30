"""Generate database search queries through the Gemini website."""

from __future__ import annotations

import re
from typing import Callable, Dict

from gemini_browser import GeminiBrowser
from response_parser import _extract_json


def _canonical_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _find_string(payload: object, *names: str) -> str:
    """Find a named string in mildly different Gemini JSON layouts."""
    wanted = {_canonical_key(name) for name in names}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if _canonical_key(key) in wanted and isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            found = _find_string(value, *names)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_string(value, *names)
            if found:
                return found
    return ""


def build_query_prompt(research_question: str) -> str:
    return f"""You are an expert systematic-literature-review search strategist.

Create a precise Boolean search query for this research question:
{research_question}

Rules:
- Group synonyms for the same concept with OR.
- Join distinct concepts with AND.
- Quote every phrase and do not add database field prefixes.
- Include the technology/intervention, domain/context, explicit comparator when present,
  and measurable outcomes. Do not invent concepts unrelated to the question.
- Return ONLY valid JSON, without Markdown or commentary, in this exact shape:
{{
  "concepts": {{
    "PRIMARY": "short primary concept",
    "DOMAIN": "short domain/context",
    "COMPARATOR": "short comparator or empty string"
  }},
  "base_query": "(\"term\" OR \"synonym\") AND (\"domain\")"
}}
"""


def parse_query_response(text: str) -> Dict[str, object]:
    payload = _extract_json(text)
    if not isinstance(payload, dict):
        raise ValueError("Gemini query response must be a JSON object.")

    base_query = _find_string(
        payload, "base_query", "boolean_query", "search_query", "query",
        "google_scholar", "google_scholar_query",
    )
    if not base_query:
        raise ValueError("Gemini query response omitted base_query.")
    concepts = payload.get("concepts")
    if not isinstance(concepts, dict):
        concepts = {}

    normalized_concepts = {
        key: _find_string(concepts, key)
        for key in ("PRIMARY", "DOMAIN", "COMPARATOR")
    }
    google_query = _find_string(payload, "google_scholar", "google_scholar_query") or base_query
    scopus_query = _find_string(payload, "scopus", "scopus_query") or f"TITLE-ABS-KEY({base_query})"
    wos_query = _find_string(
        payload, "web_of_science", "web_of_science_query", "wos", "wos_query"
    ) or f"TS=({base_query})"
    ieee_query = _find_string(payload, "ieee_xplore", "ieee_xplore_query", "ieee") or base_query
    pubmed_query = _find_string(payload, "pubmed", "pubmed_query")
    if not pubmed_query:
        pubmed_query = re.sub(r'"([^"]+)"', r'"\1"[tiab]', base_query)
    return {
        "status": "success",
        "provider": "gemini",
        "concepts": normalized_concepts,
        "google_scholar": google_query,
        "scopus": scopus_query,
        "web_of_science": wos_query,
        "ieee_xplore": ieee_query,
        "pubmed": pubmed_query,
    }


def generate_queries_with_gemini(
    research_question: str,
    browser_factory: Callable[[], GeminiBrowser] = GeminiBrowser,
) -> Dict[str, object]:
    with browser_factory() as browser:
        response = browser.submit(build_query_prompt(research_question))
        try:
            return parse_query_response(response)
        except (TypeError, ValueError):
            correction = browser.submit(f"""Your previous response did not match the required schema.
Return ONLY one valid JSON object for this research question:
{research_question}

Use exactly these top-level keys and ensure base_query is a non-empty string:
{{"concepts":{{"PRIMARY":"","DOMAIN":"","COMPARATOR":""}},"base_query":""}}
Do not use Markdown, explanations, arrays, or alternative key names.""")
            return parse_query_response(correction)
