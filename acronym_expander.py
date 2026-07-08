# acronym_expander.py
import re
from schema import SLRQueryContext

# Comprehensive academic engineering acronym map
ACRONYM_MAP = {
    "ci/cd": ["continuous integration", "continuous deployment", "continuous delivery"],
    "llm": ["large language model"],
    "rag": ["retrieval-augmented generation"],
    "nlp": ["natural language processing"],
    "iot": ["internet of things"],
    "vm": ["virtual machine"],
    "vms": ["virtual machines"],
    "gnn": ["graph neural network"],
    "gcn": ["graph convolutional network"],
    "gat": ["graph attention network"],
    "cv": ["computer vision"],
    "vit": ["vision transformer"],
    "siem": ["security information and event management"],
    "aiops": ["artificial intelligence for it operations"],
    "cnn": ["convolutional neural network"],
    "svm": ["support vector machine"],
    "rf": ["random forest"],
    "ann": ["artificial neural network"]
}

def process_array_acronyms(target_list: list[str]) -> list[str]:
    """
    Scans list phrases using strict regex word boundaries to instantly catch
    acronym anchors wrapped in hyphens or slashes without text fragmentation.
    """
    expanded_pool = list(target_list)
    
    for term in target_list:
        normalized_term = term.lower().strip()
        
        for acronym, expansions in ACRONYM_MAP.items():
            # \b locks match evaluations onto true word boundaries, cleanly isolating 
            # keys out of compound sequences like 'LLM-based' or 'CI/CD-driven'
            boundary_pattern = r'\b' + re.escape(acronym) + r'\b'
            
            if re.search(boundary_pattern, normalized_term):
                for expansion in expansions:
                    # Deduplicate against root forms to maintain array cleanliness
                    if expansion.lower() not in [t.lower().replace("*", "") for t in expanded_pool]:
                        expanded_pool.append(f"{expansion}*")
                        
    return list(dict.fromkeys(expanded_pool))

def expand_acronym_layer(current_context: SLRQueryContext) -> SLRQueryContext:
    """
    Ingests the intermediate data payload and runs a deterministic pass across
    all 5 fields to resolve shorthand variants with zero VRAM overhead.
    """
    return SLRQueryContext(
        technology=process_array_acronyms(current_context.technology),
        domain=process_array_acronyms(current_context.domain),
        comparison=process_array_acronyms(current_context.comparison),
        context=process_array_acronyms(current_context.context),
        outcomes=process_array_acronyms(current_context.outcomes)
    )
