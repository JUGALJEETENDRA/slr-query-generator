# compiler.py
from schema import SLRQueryContext

def compile_boolean_query(context: SLRQueryContext) -> str:
    """
    Ingests a pristine, verified 5-facet data context and compiles it into 
    an academic database compliant Boolean query string.
    """
    facet_blocks = []
    
    # Process facets sequentially to ensure predictable structural assembly
    facets_to_compile = [
        context.technology,
        context.domain,
        context.comparison,
        context.context,
        context.outcomes
    ]
    
    for facet_array in facets_to_compile:
        # If a facet block was left empty or filtered out completely, skip it cleanly
        if not facet_array:
            continue
            
        formatted_phrases = []
        for phrase in facet_array:
            # Strip out any erratic internal quotes to prevent string execution breaks
            clean_phrase = phrase.replace('"', '').strip()
            # Lock the exact-phrase boundary using explicit string literal notation
            formatted_phrases.append(f'"{clean_phrase}"')
            
        # Synthesize the internal facet block separated by logical OR operators
        inner_or_sequence = " OR ".join(formatted_phrases)
        facet_blocks.append(f"({inner_or_sequence})")
        
    # Bind all parenthetical blocks together with absolute outer AND requirements
    final_boolean_query_string = " AND ".join(facet_blocks)
    return final_boolean_query_string