import re
from collections import Counter


STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "by", "for", "from", "how", "in",
    "into", "is", "of", "on", "or", "the", "to", "used", "using", "with",
    "within", "what", "which", "their", "these", "those", "this", "that",
    "study", "studies", "review", "reviews", "systematic", "literature",
    "support", "supports", "improve", "improves", "improved", "improving",
}

BROAD_TASK_TERMS = {
    "application", "applications", "use", "uses", "use cases", "approach",
    "approaches", "implementation", "implementations", "simulation",
    "simulations", "optimization", "optimisation", "monitoring", "control",
    "prediction", "decision", "assessment", "evaluation",
}

METHOD_CUES = (
    "using", "use of", "used", "based on", "enabled by", "powered by",
    "technology", "technologies", "system", "systems", "model", "models",
)
CONTEXT_CUES = (
    "in", "for", "within", "domain", "industry", "manufacturing",
    "environment", "applications",
)
TASK_CUES = (
    "application", "applications", "use", "uses", "simulation",
    "monitoring", "optimization", "optimisation", "prediction", "control",
    "decision", "assessment", "evaluation",
)


def parse_dynamic_research_question(research_question):
    text = str(research_question or "").strip()
    normalized = _normalize_preserving_dots(text)
    method = _extract_method(text)
    context = _extract_context(text)
    tasks = _extract_tasks(text, context)
    used = bool(method or context or tasks)
    reason = ""
    if used:
        reason = "dynamic_rq_terms_extracted_from_question_text"
    return {
        "method_or_technology": _join(method),
        "application_context": _join(context),
        "target_tasks_or_outcomes": _join(tasks),
        "method_synonyms": _join(method),
        "context_synonyms": _join(context),
        "domain_synonyms": _join(context),
        "task_outcome_synonyms": _join(tasks),
        "required_inclusion_concepts": _join(method + context),
        "rq_dynamic_extraction_used": str(used),
        "rq_dynamic_extraction_reason": reason,
        "rq_dynamic_source_text": normalized,
    }


def mine_dynamic_corpus_terms(rows, title_col, abstract_col, sample_size=30, rq_frame=None):
    rq_terms = set()
    rq_method_terms = set()
    rq_task_terms = set()
    rq_context_terms = set()
    if rq_frame:
        rq_terms.update(_tokens(str(rq_frame.get("rq_text", ""))))
        for field in ("method_or_technology", "method_synonyms"):
            rq_method_terms.update(_tokens(str(rq_frame.get(field, ""))))
        for field in ("target_tasks_or_outcomes", "task_outcome_synonyms"):
            rq_task_terms.update(_tokens(str(rq_frame.get(field, ""))))
        for field in ("application_context", "context_synonyms", "domain_synonyms"):
            rq_context_terms.update(_tokens(str(rq_frame.get(field, ""))))
        for field in ("method_or_technology", "target_tasks_or_outcomes", "application_context"):
            rq_terms.update(_tokens(str(rq_frame.get(field, ""))))

    phrase_counter = Counter()
    method_counter = Counter()
    task_counter = Counter()
    context_counter = Counter()
    for _, row in rows.head(sample_size).iterrows():
        text = f"{row.get(title_col, '')} {row.get(abstract_col, '')}"
        normalized = _normalize_preserving_dots(text)
        phrases = _candidate_phrases(normalized)
        for phrase in phrases:
            if _phrase_allowed(phrase, rq_terms):
                phrase_counter[phrase] += 1
                window = _context_window(normalized, phrase)
                method_overlap = _overlaps(phrase, rq_method_terms)
                context_overlap = _overlaps(phrase, rq_context_terms) or _looks_like_context(phrase)
                task_overlap = _overlaps(phrase, rq_task_terms) or phrase in BROAD_TASK_TERMS
                if method_overlap or (_near_any(window, METHOD_CUES) and not context_overlap and not task_overlap):
                    method_counter[phrase] += 1
                if context_overlap:
                    context_counter[phrase] += 1
                if task_overlap or (_near_any(window, TASK_CUES) and not method_overlap and not context_overlap):
                    task_counter[phrase] += 1

    # Frequent phrases can still be useful even without a local cue.
    for phrase, count in phrase_counter.items():
        if count >= 2 and _overlaps(phrase, rq_method_terms):
            if not method_counter[phrase] and not context_counter[phrase] and not task_counter[phrase]:
                method_counter[phrase] += count

    return {
        "corpus_dynamic_method_terms": _top_terms(method_counter),
        "corpus_dynamic_task_terms": _top_terms(task_counter),
        "corpus_dynamic_context_terms": _top_terms(context_counter),
        "corpus_dynamic_profile_terms": _top_terms(phrase_counter),
    }


def _extract_method(text):
    patterns = (
        r"\bhow\s+(?:is|are)\s+(.+?)\s+used\b",
        r"\buse\s+of\s+(.+?)\s+(?:in|for|within)\b",
        r"\busing\s+(.+?)\s+(?:in|for|within)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return _split_phrases(match.group(1))
    return []


def _extract_context(text):
    match = re.search(r"\b(?:in|within|for)\s+(.+?)(?:\?|$)", text, flags=re.I)
    if not match:
        return []
    value = re.sub(r"\b(" + "|".join(re.escape(term) for term in BROAD_TASK_TERMS) + r")\b.*$", "", match.group(1), flags=re.I)
    return _split_phrases(value)


def _extract_tasks(text, context_terms):
    lowered = text.lower()
    found = [term for term in BROAD_TASK_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered)]
    if found:
        return _dedupe([_singularize(term) for term in found])
    if "used" in lowered and context_terms:
        return ["applications"]
    return []


def _split_phrases(value):
    value = str(value or "")
    value = value.replace("/", " / ")
    expanded = []
    industry_match = re.search(r"\bindustry\s+([0-9]+(?:\.[0-9]+)?)(?:\s*/\s*([0-9]+(?:\.[0-9]+)?))?", value, flags=re.I)
    if industry_match:
        expanded.append(f"industry {industry_match.group(1)}")
        if industry_match.group(2):
            expanded.append(f"industry {industry_match.group(2)}")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\bindustry\s+[0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+(?:\.[0-9]+)?)?", " ", value, flags=re.I)
    for part in re.split(r"\s+and\s+|,|;", value, flags=re.I):
        phrase = _clean_phrase(part)
        if phrase:
            expanded.append(phrase)
    return _dedupe(expanded)


def _candidate_phrases(text):
    toks = _tokens(text, keep_numbers=True)
    phrases = []
    for size in range(1, 4):
        for i in range(0, len(toks) - size + 1):
            chunk = toks[i:i + size]
            if all(token in STOPWORDS for token in chunk):
                continue
            if chunk[0] in STOPWORDS or chunk[-1] in STOPWORDS:
                continue
            if size == 1 and chunk[0] not in BROAD_TASK_TERMS:
                continue
            phrase = " ".join(chunk)
            if len(phrase) >= 3:
                phrases.append(phrase)
    return phrases


def _phrase_allowed(phrase, rq_terms):
    tokens = set(_tokens(phrase))
    if not tokens:
        return False
    if tokens & rq_terms:
        return True
    return len(tokens) >= 2


def _context_window(text, phrase, chars=55):
    index = text.find(phrase)
    if index < 0:
        return ""
    return text[max(0, index - chars): index + len(phrase) + chars]


def _near_any(text, cues):
    lowered = text.lower()
    return any(cue in lowered for cue in cues)


def _looks_like_context(phrase):
    return any(token in phrase for token in ("industry", "manufacturing", "environment", "domain"))


def _overlaps(phrase, rq_terms):
    return bool(set(_tokens(phrase)) & set(rq_terms))


def _tokens(text, keep_numbers=False):
    if keep_numbers:
        raw = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text or "").lower())
    else:
        raw = re.findall(r"[a-z]+", str(text or "").lower())
    return [token for token in raw if token not in STOPWORDS and len(token) > 1]


def _clean_phrase(value):
    value = _normalize_preserving_dots(value)
    value = re.sub(r"^(?:the|a|an)\s+", "", value)
    value = re.sub(r"\b(?:used|using|use of)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" -/")
    return value


def _normalize_preserving_dots(value):
    value = str(value or "").lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9./]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _singularize(term):
    if term == "applications":
        return "applications"
    return term


def _dedupe(values):
    return list(dict.fromkeys(value for value in values if value))


def _join(values):
    return "; ".join(_dedupe(str(value).strip() for value in values if str(value).strip()))


def _top_terms(counter, limit=30):
    return _join(term for term, _ in counter.most_common(limit))
