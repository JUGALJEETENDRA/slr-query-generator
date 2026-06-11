# generator.py
from pydantic import BaseModel, Field
from typing import List, Literal
from schema import SLRQueryContext

class ExpandedTermItem(BaseModel):
    term: str = Field(..., description="The direct lowercase index keyword or acronym stem.")
    relationship_type: Literal["EXACT_SYNONYM", "NEAR_SYNONYM", "CANONICAL_REALIZATION", "RELATED_CONCEPT"] = Field(
        ..., 
        description="Categorize the exact semantic relationship to the input query term."
    )

class FacetExpansionContainer(BaseModel):
    expansions: List[ExpandedTermItem] = Field(default_factory=list)

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
        "Every generated variant MUST contain the core specialized structural anchor.\n\n"
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
                    term_string = item.term.strip().strip("'\"")
                    if not term_string:
                        continue
                        
                    # Drop RELATED_CONCEPT silently (No print statements here)
                    if item.relationship_type == "RELATED_CONCEPT":
                        continue
                    
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