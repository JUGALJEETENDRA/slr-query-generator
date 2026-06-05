# extractor.py
import json
from schema import SLRQueryContext

def extract_5_facets(client, model: str, question: str) -> SLRQueryContext:
    """
    Pure token isolation engine. Extracts text segments into the data contract
    without triggering iterative validation crashes.
    """
    
    system_prompt = (
        "Context: You are a structural token parsing layer for academic queries.\n"
        "Task: Slice the user's research question into 5 distinct JSON arrays.\n\n"
        
        "FACET FIELD DEFINITIONS:\n"
        "- technology: The core tool, algorithm, framework, or architecture being evaluated.\n"
        "- domain: The broad problem space, scientific field, or application area.\n"
        "- comparison: Legacy baselines, standards, or alternative architectures being compared against.\n"
        "- context: The specific platform setting, scale, operational environment, or target group.\n"
        "- outcomes: The performance metrics, targets, variables, or phenomena being observed.\n\n"
        
        "CRITICAL EXTRACTION PROTOCOLS:\n"
        "1. Extract word-for-word substrings directly from the text. Never invent generic words.\n"
        "2. If a field is not explicitly present in the sentence, return it as an empty array [].\n"
        "3. THE VERSUS RULE: When phrases like 'vs' or 'compare to' appear, isolate the primary tool before the operator into 'technology', and the baseline tool after the operator into 'comparison'. Never extract the operator word itself.\n\n"
        
        "STRUCTURAL SAMPLES:\n"
        "User: 'What is the computational overhead of homomorphic encryption inside cloud databases?'\n"
        "Output: {\"technology\":[\"homomorphic encryption\"],\"domain\":[\"databases\"],\"comparison\":[],\"context\":[\"cloud databases\"],\"outcomes\":[\"computational overhead\"]}\n\n"
        "User: 'How does ToolX compare to ToolY for fault isolation latency?'\n"
        "Output: {\"technology\":[\"ToolX\"],\"domain\":[],\"comparison\":[\"ToolY\"],\"context\":[],\"outcomes\":[\"fault isolation latency\"]}"
    )

    response = client.chat.completions.create(
        model=model,
        response_model=SLRQueryContext,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Slice this question: '{question}'"}
        ],
        temperature=0.0
    )
    
    return response