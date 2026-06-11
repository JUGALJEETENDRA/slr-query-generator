from pydantic import BaseModel
from schema import SLRExtractionContract

class ComparatorAnchor(BaseModel):
    baseline_term: str

def extract_5_facets(client, model: str, question: str) -> SLRExtractionContract:
    """
    Hardened Two-Pass Isolation Gateway.
    Cleaned for production: No console logs, silent error handling, optimized latency.
    """
    
    # ─── PASS 1: SEMANTIC ANCHOR ISOLATION ───
    anchor_system_prompt = (
        "Identify the literal word-for-word substring representing the baseline or "
        "traditional method compared against. If none exists, return 'NONE'."
    )

    try:
        anchor_response = client.chat.completions.create(
            model=model,
            response_model=ComparatorAnchor,
            messages=[
                {"role": "system", "content": anchor_system_prompt},
                {"role": "user", "content": f"Isolate comparison baseline from: '{question}'"}
            ],
            temperature=0.0
        )
        isolated_comparator = anchor_response.baseline_term.strip()
    except Exception:
        # Silent failure: Proceed without a comparator rather than crashing the request
        isolated_comparator = "NONE"

    # ─── PASS 2: HARDENED SCHEMA FACET ALLOCATION ───
    if isolated_comparator and isolated_comparator.upper() != "NONE":
        exclusion_rule = (
            f"🚨 [CONSTRAINT]: Baseline is verified as: '{isolated_comparator}'.\n"
            f"- MUST place '{isolated_comparator}' into 'comparator_baseline'.\n"
            f"- FORBIDDEN from putting '{isolated_comparator}' into 'primary_paradigm' or 'domain_context'."
        )
    else:
        exclusion_rule = "🚨 [CONSTRAINT]: No baseline detected. 'comparator_baseline' must be []."

    system_prompt = (
        "Role: Structural token parsing layer for academic SLR queries.\n"
        "Task: Slice research question into 4 arrays: primary_paradigm, comparator_baseline, domain_context, outcome_variables.\n\n"
        f"{exclusion_rule}\n"
        "Protocols:\n"
        "1. Extract word-for-word substrings. Do not invent words.\n"
        "2. If a field is missing, return [].\n"
        "3. Strictly separate innovation from baseline controls."
    )

    response = client.chat.completions.create(
        model=model,
        response_model=SLRExtractionContract,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Slice this question: '{question}'"}
        ],
        temperature=0.0
    )
    
    return response