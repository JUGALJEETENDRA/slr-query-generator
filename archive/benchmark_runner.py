import json
import csv
import re
import os
import requests

# Execution Configuration Targets
API_URL = "http://localhost:8000/generate"
BENCHMARK_INPUT = "benchmark_questions.json"
CSV_SUMMARY_OUTPUT = "benchmark_summary.csv"
TELEMETRY_JSON_OUTPUT = "term_telemetry.json"


def classify_evaluated_term(term: str, forbidden_comparators: list) -> str:
    """
    Granular behavioral evaluation spectrometer.
    Deconstructs the old 'topic_brainstorming' catch-all into explicit failure profiles.
    """
    val = term.lower().strip().replace("*", "")

    # 1. HARD SECURITY BLOCK: Target control-plane leaks
    if any(comp in val for comp in forbidden_comparators):
        return "comparator_leak"

    # 2. CANONICAL_REALIZATION: Valid production tools, explicit implementations, or intertwined tracks
    canonical_tokens = ["kubernetes", "k8s", "mec", "multi-access edge", "fabric",
                        "hyperledger", "da vinci", "fog computing", "cloudlet"]
    if any(token in val for token in canonical_tokens):
        return "canonical_realization"

    # 3. FABRICATED_TERM: Direct linguistic inventions or pure LLM hallucinations
    fabricated_tokens = ["consented ledger", "privately managed blockchain"]
    if any(token in val for token in fabricated_tokens):
        return "fabricated_term"

    # 4. RELATED_CONCEPT: Legitimate ecosystem layers, parent structures, or adjacent environments
    related_tokens = ["infrastructure", "computing environment", "cloud native architecture", "management platform"]
    if any(token in val for token in related_tokens):
        return "related_concept"

    # 5. METRIC_INFLATION: Unused slot padding using KPI variables, performance parameters, or metrics
    metric_tokens = ["latency", "throughput", "performance", "delivery",
                     "speed", "cadence", "lead time", "delay", "variance"]
    if any(token in val for token in metric_tokens):
        return "metric_inflation"

    # 6. SEMANTIC_GENERALIZATION: Abstraction drift (dropping specific qualifiers to climb upward)
    general_tokens = ["surgical systems", "autonomous systems", "automated systems",
                      "systems", "technologies", "architectures"]
    if val in general_tokens or (val.endswith(("systems", "technologies", "architectures")) and
                                 not any(anchor in val for anchor in ["zero trust", "robotic", "container",
                                                                       "blockchain", "cybersecurity"])):
        return "semantic_generalization"

    # 7. TRUE TOPIC_BRAINSTORMING: Actual out-of-bounds contextual drift or loose dictionary synonyms
    brainstorm_tokens = ["fringe", "periphery", "doubly-encrypted"]
    if any(token in val for token in brainstorm_tokens):
        return "topic_brainstorming"

    # 8. DEFAULT RECOVERY VALUE: Valid targeted multi-word search phrase
    return "pending_manual_audit"


def tokenize_query(query_string: str) -> list:
    """Extracts raw individual word tokens from string components, discarding operators."""
    cleaned = re.sub(r'[()"\',*]', '', query_string)
    words = [w.strip().lower() for w in cleaned.split() if w.strip().upper() not in ["AND", "OR", "NOT"]]
    return words


def parse_phrases(query_string: str) -> list:
    """Extracts quoted exact phrases from the generated string blocks."""
    return re.findall(r'"([^"]+)"', query_string)


def run_automated_harness():
    if not os.path.exists(BENCHMARK_INPUT):
        print(f"❌ Error: Missing input file target -> {BENCHMARK_INPUT}")
        return

    with open(BENCHMARK_INPUT, "r") as f:
        questions = json.load(f)

    # Accumulation Registers for Global Summary
    total_runs = 0
    compilation_failures = 0
    comparator_leaks = 0
    broadness_warnings = 0
    facet_bloat_warnings = 0

    csv_rows = []
    term_telemetry_log = []

    print("\n=========================================================")
    print("🚀 INITIALIZING SLR COMPILER VALIDATION RUN")
    print(f"Target Baseline: V4.0 Stable Engine")
    print(f"Total Seeding Elements: {len(questions)}")
    print("=========================================================\n")

    for case in questions:
        q_id = case["question_id"]
        text = case["question_text"]
        forbidden = [t.lower() for t in case.get("forbidden_comparator_tokens", [])]

        # Local State Switches Per Iteration Run
        compilation_pass = True
        comparator_integrity_pass = True
        broadness_warning_triggered = False
        bloat_warning_triggered = False

        print(f"⚙️ Compiling Case [{q_id}] -> Requesting local model pipeline...")

        try:
            # Execute active live API port endpoint context
            res = requests.post(API_URL, json={"question": text}, timeout=45)
            if res.status_code != 200:
                compilation_pass = False
                data = {}
            else:
                data = res.json()
                if data.get("status") == "error":
                    compilation_pass = False
        except Exception:
            compilation_pass = False
            data = {}

        if not compilation_pass:
            compilation_failures += 1
            csv_rows.append([q_id, "FAIL", "FAIL", 0, 0, 0, "API_OR_SYNTAX_ERROR", "FALSE", "FALSE"])
            continue

        total_runs += 1

        # Gather Target Query Output Strings
        gs_query = data.get("google_scholar", "")
        ieee_query = data.get("ieee_xplore", "")

        # Track Tokens and Extracted String Layers
        all_tokens = tokenize_query(gs_query)
        phrases = parse_phrases(gs_query)

        # 1. Evaluate Hard Gate: Comparator Integrity Leaks
        leaked_tokens = [tok for tok in all_tokens if tok in forbidden]
        if leaked_tokens:
            comparator_integrity_pass = False
            comparator_leaks += 1

        # 2. Evaluate Soft Warning Layer 1: Broad-Term Density
        unquoted_words = [w for w in gs_query.split() if w.upper() not in ["AND", "OR"] and '"' not in w]
        total_term_blocks = len(phrases) + len(unquoted_words)
        broad_ratio = len(unquoted_words) / total_term_blocks if total_term_blocks > 0 else 0
        if broad_ratio > 0.40:
            broadness_warning_triggered = True
            broadness_warnings += 1

        # 3. Evaluate Soft Warning Layer 2: Facet Bloat
        if len(phrases) > 12:
            bloat_warning_triggered = True
            facet_bloat_warnings += 1

        # Calculate Local Analytical Score Profiles (Placeholder Mock Scores for manual human elements)
        extraction_acc = 100 if compilation_pass else 0
        synonym_purity = int((1.0 - (len(leaked_tokens) / len(all_tokens))) * 100) if all_tokens else 100
        ontology_isolation = 100 if not broadness_warning_triggered else 60

        final_score = int(0.40 * extraction_acc + 0.35 * synonym_purity + 0.25 * ontology_isolation)
        if not comparator_integrity_pass:
            final_score = 0  # Enforce structural indicator drop

        # 🔬 REFINED SPECTROMETER EVALUATOR (using the new classify_evaluated_term function)
        for p in phrases:
            classification = classify_evaluated_term(p, forbidden)
            term_telemetry_log.append({
                "question_id": q_id,
                "term": p,
                "classification": classification
            })

        # Append Summary Records
        csv_rows.append([
            q_id,
            "PASS" if compilation_pass else "FAIL",
            "PASS" if comparator_integrity_pass else "FAIL",
            extraction_acc,
            synonym_purity,
            ontology_isolation,
            final_score,
            "TRUE" if broadness_warning_triggered else "FALSE",
            "TRUE" if bloat_warning_triggered else "FALSE"
        ])

    # Save Records to Flat Storage Units
    with open(CSV_SUMMARY_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question_ID", "Compilation_Pass", "Comparator_Integrity_Pass",
                         "Extraction_Accuracy", "Synonym_Purity", "Ontology_Isolation",
                         "Final_Benchmark_Score", "Broadness_Warning", "Facet_Bloat_Warning"])
        writer.writerows(csv_rows)

    with open(TELEMETRY_JSON_OUTPUT, "w") as f:
        json.dump(term_telemetry_log, f, indent=2)

    # Render Terminal Metrics Dashboard
    print("\n=========================================================")
    print("SLR QUERY GENERATOR BENCHMARK SUMMARY")
    print(f"Version Baseline : V4.0 Production Freeze + Refined Spectrometer")
    print(f"Total Cases Run  : {len(questions)}")
    print("=========================================================")
    print("\nHard Gate Violations")
    print("------------------")
    print(f"Compilation Failures : {compilation_failures}")
    print(f"Comparator Leaks     : {comparator_leaks}")
    print("\nSoft Review Warnings")
    print("-------------")
    print(f"Broadness Warnings   : {broadness_warnings}")
    print(f"Facet Bloat Warnings : {facet_bloat_warnings}")
    print("\nTelemetry Repository File Outputs Generated:")
    print(f" -> Question-Level Grid Table Metrics : {CSV_SUMMARY_OUTPUT}")
    print(f" -> Term-Level Granular Audit Records: {TELEMETRY_JSON_OUTPUT}")
    print("=========================================================\n")


if __name__ == "__main__":
    run_automated_harness()