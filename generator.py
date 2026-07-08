# generator.py
import re
from pydantic import BaseModel, Field
from typing import List, Literal
from acronym_expander import ACRONYM_MAP
from schema import SLRQueryContext

class ExpandedTermItem(BaseModel):
    term: str = Field(..., description="The direct lowercase index keyword or acronym stem.")
    relationship_type: Literal["EXACT_SYNONYM", "NEAR_SYNONYM", "CANONICAL_REALIZATION", "RELATED_CONCEPT"] = Field(
        ..., 
        description="Categorize the exact semantic relationship to the input query term."
    )

class FacetExpansionContainer(BaseModel):
    expansions: List[ExpandedTermItem] = Field(default_factory=list)

GENERIC_TOKENS = {
    "a", "an", "and", "are", "as", "at", "based", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "over", "the", "through", "to", "using", "via", "with",
    "approach", "approaches", "architecture", "architectures", "area", "areas",
    "concept", "concepts", "environment", "environments", "field", "fields",
    "framework", "frameworks", "infrastructure", "infrastructures", "method",
    "methods", "model", "models", "paradigm", "paradigms", "platform", "platforms",
    "practice", "practices", "process", "processes", "research", "solution",
    "solutions", "system", "systems", "technique", "techniques", "technologies",
    "technology", "tool", "tools",
}

BROAD_PARENT_PHRASES = {
    "artificial intelligence", "machine learning", "deep learning",
    "computer science", "software engineering", "information technology",
    "cloud computing", "cybersecurity", "information security",
    "network security", "medical care", "healthcare", "clinical medicine",
    "cyber physical systems", "automated driving systems",
}

GENERIC_OUTCOME_PHRASES = {
    "performance", "performance metric", "performance metrics",
    "performance evaluation", "performance improvement", "performance optimization",
    "performance limitation", "performance limitations", "system performance",
    "system throughput", "delay", "efficiency", "effectiveness",
    "model accuracy", "academic performance",
}

COMPARATOR_CUES = {
    "baseline", "baselines", "traditional", "legacy", "alternative", "alternatives",
    "versus", "compared", "comparison", "control", "controls",
}

SYNONYM_BRIDGES = [
    {"latency", "delay", "lag", "response", "responsiveness", "time"},
    {"throughput", "bandwidth", "capacity"},
    {"accuracy", "precision", "recall", "sensitivity", "specificity", "auc", "roc"},
    {"energy", "power", "electricity", "consumption", "efficiency"},
    {"privacy", "confidentiality", "anonymity", "deidentification", "reidentification"},
    {"security", "secure", "vulnerability", "vulnerabilities", "threat", "attack", "risk"},
    {"fairness", "bias", "equity", "ethics", "ethical"},
    {"scalability", "scalable", "scale", "scaling"},
    {"reliability", "availability", "resilience", "robustness", "fault", "failure"},
    {"cost", "expense", "economic", "financial"},
]

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_MODEL = None
_EMBEDDINGS_AVAILABLE = None
_SEMANTIC_SCORE_CACHE = {}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().replace("*", " ").replace("_", " ")).strip()


def _singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(value: str) -> set[str]:
    return {
        _singularize(token)
        for token in re.findall(r"[a-z0-9]+", _normalize_text(value))
        if len(token) > 1 and token not in GENERIC_TOKENS
    }


def _initialism(seed: str) -> str:
    return "".join(token[0] for token in re.findall(r"[a-z0-9]+", seed.lower()) if token not in GENERIC_TOKENS)


def _cosine_similarity(left_embedding, right_embedding) -> float:
    left = list(left_embedding)
    right = list(right_embedding)
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _semantic_similarity(left: str, right: str) -> float | None:
    global _EMBEDDING_MODEL, _EMBEDDINGS_AVAILABLE

    if _EMBEDDINGS_AVAILABLE is False:
        return None

    cache_key = tuple(sorted((_normalize_text(left), _normalize_text(right))))
    if cache_key in _SEMANTIC_SCORE_CACHE:
        return _SEMANTIC_SCORE_CACHE[cache_key]

    try:
        if _EMBEDDING_MODEL is None:
            from sentence_transformers import SentenceTransformer

            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
            _EMBEDDINGS_AVAILABLE = True

        embeddings = _EMBEDDING_MODEL.encode([left, right])
        score = _cosine_similarity(embeddings[0], embeddings[1])
        _SEMANTIC_SCORE_CACHE[cache_key] = score
        return score
    except Exception:
        _EMBEDDINGS_AVAILABLE = False
        return None


def _has_synonym_bridge(term_tokens: set[str], seed_tokens: set[str]) -> bool:
    return any(term_tokens & bridge and seed_tokens & bridge for bridge in SYNONYM_BRIDGES)


def _is_known_acronym_equivalent(term: str, seed: str) -> bool:
    compact_term = re.sub(r"[^a-z0-9]", "", _normalize_text(term))
    compact_seed = re.sub(r"[^a-z0-9]", "", _normalize_text(seed))
    if not compact_term or not compact_seed:
        return False

    if compact_term == _initialism(seed) or compact_seed == _initialism(term):
        return True

    for acronym, expansions in ACRONYM_MAP.items():
        compact_acronym = re.sub(r"[^a-z0-9]", "", acronym)
        compact_expansions = {
            re.sub(r"[^a-z0-9]", "", _normalize_text(expansion))
            for expansion in expansions
        }
        if compact_term == compact_acronym and compact_seed in compact_expansions:
            return True
        if compact_seed == compact_acronym and compact_term in compact_expansions:
            return True

    return False


def _has_lexical_anchor(term: str, seed_terms: list[str]) -> bool:
    normalized_term = _normalize_text(term)
    term_tokens = _tokens(term)
    if not term_tokens:
        return False

    for seed in seed_terms:
        normalized_seed = _normalize_text(seed)
        seed_tokens = _tokens(seed)
        if not seed_tokens:
            continue
        if normalized_term == normalized_seed:
            return True
        if _is_known_acronym_equivalent(term, seed):
            return True
        if term_tokens & seed_tokens:
            return True
        if _has_synonym_bridge(term_tokens, seed_tokens):
            return True
    return False


def _semantic_anchor_score(term: str, seed_terms: list[str]) -> float | None:
    scores = [
        score
        for seed in seed_terms
        for score in [_semantic_similarity(term, seed)]
        if score is not None
    ]
    if not scores:
        return None
    return max(scores)


def _semantic_threshold(relationship_type: str) -> float:
    if relationship_type == "EXACT_SYNONYM":
        return 0.72
    if relationship_type == "CANONICAL_REALIZATION":
        return 0.66
    return 0.68


def _is_broad_parent_without_anchor(term: str, seed_terms: list[str]) -> bool:
    normalized_term = _normalize_text(term)
    term_tokens = _tokens(term)
    if not term_tokens:
        return True

    if normalized_term in BROAD_PARENT_PHRASES:
        return True

    has_anchor = _has_lexical_anchor(term, seed_terms)
    if not has_anchor and any(phrase in normalized_term for phrase in BROAD_PARENT_PHRASES):
        return True

    return False


def _is_generic_outcome(term: str, seed_terms: list[str]) -> bool:
    normalized_term = _normalize_text(term)
    if normalized_term not in GENERIC_OUTCOME_PHRASES:
        return False
    return not _has_lexical_anchor(term, seed_terms)


def _is_comparator_leak(term: str, field_name: str) -> bool:
    if field_name == "comparison":
        return False
    return bool(_tokens(term) & COMPARATOR_CUES)


def _accept_generated_term(field_name: str, seed_terms: list[str], item: ExpandedTermItem) -> bool:
    if item.relationship_type == "RELATED_CONCEPT":
        return False
    term = item.term.strip().strip("'\"")
    if not term:
        return False
    if _is_comparator_leak(term, field_name):
        return False
    if _is_broad_parent_without_anchor(term, seed_terms):
        return False
    if field_name == "outcomes" and _is_generic_outcome(term, seed_terms):
        return False
    if any(_is_known_acronym_equivalent(term, seed) for seed in seed_terms):
        return True
    if _has_lexical_anchor(term, seed_terms):
        return True

    semantic_score = _semantic_anchor_score(term, seed_terms)
    if semantic_score is not None:
        return semantic_score >= _semantic_threshold(item.relationship_type)

    return False


def expand_base_synonyms(client, model: str, extracted_context: SLRQueryContext) -> SLRQueryContext:
    """
    Surgically expands academic facets. 
    Cleaned for production: No print statements, silent execution.
    """
    accumulated_tech = list(extracted_context.technology)
    accumulated_domain = list(extracted_context.domain)
    accumulated_context = list(extracted_context.context)
    accumulated_outcomes = list(extracted_context.outcomes)

    execution_queue = [
        {"field_name": "technology", "input_terms": extracted_context.technology, "target_registry": accumulated_tech},
        {"field_name": "domain", "input_terms": extracted_context.domain, "target_registry": accumulated_domain},
        {"field_name": "context", "input_terms": extracted_context.context, "target_registry": accumulated_context},
        {"field_name": "outcomes", "input_terms": extracted_context.outcomes, "target_registry": accumulated_outcomes},
    ]

    # (System Prompt remains unchanged as it is the core brain of the engine)
    system_prompt = (
        "[ROLE]: You are an expert Systematic Literature Review (SLR) academic search string engineer for IEEE Xplore and Scopus.\n"
        "[CRITICAL COMPLIANCE RULES]: Operate at the EXACT same semantic granularity as the seed term. "
        "NEVER replace specialized paradigms with broad parent disciplines. "
        "Every generated variant MUST preserve the same narrow concept, including accepted canonical acronyms.\n\n"
        "[TAXONOMY REGULATION]: EXACT_SYNONYM, NEAR_SYNONYM, CANONICAL_REALIZATION (e.g. 'Kubernetes', 'MEC').\n"
        "Filter out RELATED_CONCEPT (broad or adjacent noise).\n"
        "[TYPOGRAPHY]: Natural spaces only, no snake_case or code-style identifiers."
    )

    for stage in execution_queue:
        if not stage["input_terms"]:
            continue

        user_content = f"Classify and generate formal bibliographic search variations for this isolated array: {stage['input_terms']}"

        try:
            llm_expansion = client.chat.completions.create(
                model=model,
                response_model=FacetExpansionContainer,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1
            )
            
            if llm_expansion.expansions:
                for item in llm_expansion.expansions:
                    if not _accept_generated_term(stage["field_name"], stage["input_terms"], item):
                        continue
                    
                    term_string = item.term.strip().strip("'\"")
                    # Sanitize snake_case
                    if "_" in term_string:
                        term_string = term_string.replace("_", " ")
                        
                    stage["target_registry"].append(term_string)

        except Exception:
            # Silent failure: we prefer to return the original terms rather than crash the API
            continue

    return SLRQueryContext(
        technology=list(dict.fromkeys(accumulated_tech)),
        domain=list(dict.fromkeys(accumulated_domain)),
        comparison=extracted_context.comparison,
        context=list(dict.fromkeys(accumulated_context)),
        outcomes=list(dict.fromkeys(accumulated_outcomes))
    )
