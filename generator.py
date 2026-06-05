# generator.py
import json
from schema import SLRQueryContext

def expand_base_synonyms(client, model: str, extracted_context: SLRQueryContext) -> SLRQueryContext:
    """
    Guides the local 7B model using concrete, high-density hardware/software system 
    exemplars instead of generic language rewrites.
    """
    raw_input_json = extracted_context.model_dump_json(indent=2)

    system_prompt = (
        "[ROLE]: You are an advanced academic engineering thesaurus for IEEE and ACM indexing.\n"
        "[TASK]: Expand each populated array field with 2-3 precise technical synonyms or core sub-components. Never provide conversational rephrasings.\n\n"
        
        "[CRITICAL KNOWLEDGE GUIDELINES]:\n"
        "1. HARDWARE/SOFTWARE COMPONENTS: When expanding technical systems, provide actual underlying components or modalities (e.g., expand 'sensor fusion' to include terms like 'LiDAR', 'radar', or 'camera').\n"
        "2. INDUSTRY ABBREVIATIONS: Always pair core systems with their standard acronym variants (e.g., 'VMs', 'K8s', 'SOC', 'SIEM').\n"
        "3. TRUNCATION RULES: Append wildcards (*) cleanly to noun stems to capture morphological variations.\n\n"
        
        "[OUT-OF-SAMPLE EXEMPLAR 1: COMPILER DEVOPS]\n"
        "Input:\n"
        "{\n"
        "  \"technology\": [\"CI/CD\"],\n"
        "  \"domain\": [\"agile startups\"],\n"
        "  \"comparison\": [],\n"
        "  \"context\": [],\n"
        "  \"outcomes\": [\"deployment frequency\"]\n"
        "}\n"
        "Output:\n"
        "{\n"
        "  \"technology\": [\"CI/CD\", \"continuous integration*\", \"continuous delivery*\"],\n"
        "  \"domain\": [\"agile startups\", \"lean software development*\"],\n"
        "  \"comparison\": [],\n"
        "  \"context\": [],\n"
        "  \"outcomes\": [\"deployment frequency\", \"deployment automation*\", \"release automation*\"]\n"
        "}\n\n"
        
        "[OUT-OF-SAMPLE EXEMPLAR 2: PERCEPTION SYSTEMS]\n"
        "Input:\n"
        "{\n"
        "  \"technology\": [\"sensor fusion\"],\n"
        "  \"domain\": [\"autonomous vehicles\"],\n"
        "  \"comparison\": [],\n"
        "  \"context\": [],\n"
        "  \"outcomes\": [\"object detection accuracy\"]\n"
        "}\n"
        "Output:\n"
        "{\n"
        "  \"technology\": [\"sensor fusion\", \"multi-sensor fusion*\", \"LiDAR*\", \"radar*\", \"camera*\"],\n"
        "  \"domain\": [\"autonomous vehicles\", \"self-driving cars*\", \"automated driving systems*\"],\n"
        "  \"comparison\": [],\n"
        "  \"context\": [],\n"
        "  \"outcomes\": [\"object detection accuracy\", \"perception precision*\", \"recognition accuracy*\"]\n"
        "}"
    )

    llm_expansion = client.chat.completions.create(
        model=model,
        response_model=SLRQueryContext,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate academic expansions for this structure:\n{raw_input_json}"}
        ],
        temperature=0.1  # Locked down variance to enforce structural alignment
    )
    
    # Safe Python Merger Layer to neutralize token drop behavior
    final_payload = SLRQueryContext(
        technology=list(dict.fromkeys(extracted_context.technology + llm_expansion.technology)),
        domain=list(dict.fromkeys(extracted_context.domain + llm_expansion.domain)),
        comparison=list(dict.fromkeys(extracted_context.comparison + llm_expansion.comparison)),
        context=list(dict.fromkeys(extracted_context.context + llm_expansion.context)),
        outcomes=list(dict.fromkeys(extracted_context.outcomes + llm_expansion.outcomes))
    )
    
    return final_payload