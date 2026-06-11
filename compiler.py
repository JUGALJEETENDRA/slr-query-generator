from schema import SLRQueryContext

def compile_boolean_query(context: SLRQueryContext) -> str:
    facet_blocks = []
    facets = [context.technology, context.domain, context.comparison, context.context, context.outcomes]
    
    for facet_array in facets:
        if not facet_array:
            continue
        formatted = [f'"{p.replace("\"", "").strip()}"' for p in facet_array]
        facet_blocks.append(f'({" OR ".join(formatted)})')
        
    return " AND ".join(facet_blocks)