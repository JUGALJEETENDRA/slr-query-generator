# extractor.py
# This file contains the extraction logic for SLR research questions.
# Production ready: Only uses the two-pass extractor with anchor isolation.

from pydantic import BaseModel
from schema import SLRExtractionContract

# ─── NEW CLASS FOR ANCHOR ISOLATION ──────────────────────────────────────────
class ComparatorAnchor(BaseModel):
    baseline_term: str

# ─── HARDENED TWO‑PASS EXTRACTION FUNCTION ─────────────────────────────
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

    # ─── UPDATED SYSTEM PROMPT ──────────────────────────────────────────────
    system_prompt = (
        "You are an academic Systematic Literature Review (SLR) research-question parser.\n\n"
        "Your job is to extract ONLY literal phrases from the question.\n"
        "Do NOT infer.\n"
        "Do NOT generalize.\n"
        "Do NOT replace words with broader concepts.\n"
        "Do NOT invent terms that do not appear.\n\n"
        f"{exclusion_rule}\n\n"
        "Return exactly four arrays.\n\n"
        "primary_paradigm:\n"
        "- Technology, algorithm, model, framework or methodology.\n"
        "- Examples: machine learning, deep learning, federated learning.\n\n"
        "comparator_baseline:\n"
        "- Explicit comparison or baseline only.\n"
        "- If none exists return [].\n\n"
        "domain_context:\n"
        "- Preserve the MOST SPECIFIC application or problem.\n"
        "- Never generalize.\n"
        "- Example:\n"
        "software defect -> software defect\n"
        "heart disease -> heart disease\n"
        "crop yield -> crop yield\n"
        "- NOT software development\n"
        "- NOT healthcare\n"
        "- NOT agriculture\n\n"
        "outcome_variables:\n"
        "- Preserve the RESEARCH TASK exactly.\n"
        "- prediction -> prediction\n"
        "- classification -> classification\n"
        "- detection -> detection\n"
        "- segmentation -> segmentation\n"
        "- retrieval -> retrieval\n"
        "- recommendation -> recommendation\n"
        "- forecasting -> forecasting\n\n"
        "NEVER replace research tasks with evaluation metrics.\n"
        "prediction is NOT accuracy.\n"
        "prediction is NOT precision.\n"
        "prediction is NOT recall.\n"
        "classification is NOT F1-score.\n"
        "segmentation is NOT Dice coefficient.\n\n"
        "Extract only literal substrings from the research question."
    )
    # ─── END UPDATED BLOCK ────────────────────────────────────────────────────

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