# benchmark_runner.py
from openai import OpenAI
import instructor
from extractor import extract_5_facets
from generator import expand_base_synonyms
from acronym_expander import expand_acronym_layer
from ontology_expander import expand_ontology_layer
from validator import run_validation_sieve
from compiler import compile_boolean_query

def run_macro_benchmark():
    # Patch and connect our instructor execution client
    client = instructor.from_openai(
        OpenAI(base_url="http://localhost:11434/v1", api_key="ollama-local"),
        mode=instructor.Mode.JSON
    )
    MODEL = "qwen2.5:7b"

    # Populate this array with your full catalog of 24 benchmark questions
    FULL_BENCHMARK_SUITE = [
        "What is the deployment frequency of CI/CD within agile startups?",
        "What is the microservice latency overhead when using Docker vs Virtual Machines?",
        "How does LLM-based threat hunting compare to rule-based signature systems?",
        "Can sensor fusion improve object detection accuracy in autonomous vehicles?",
        # -- DROP THE REMAINING 20 QUESTIONS HERE --
    ]

    print("=" * 100)
    print(f" 🚀 EXECUTING MACRO-BENCHMARK PASS ({len(FULL_BENCHMARK_SUITE)} QUESTIONS LOADED)")
    print("=" * 100)

    print(f"{'ID':<4} | {'RESEARCH QUESTION':<65} | {'STATUS'}")
    print("-" * 100)

    compiled_queries = {}
    
    for idx, rq in enumerate(FULL_BENCHMARK_SUITE, 1):
        try:
            # Full linear pipeline loop execution
            s1 = extract_5_facets(client, MODEL, rq)
            s2 = expand_base_synonyms(client, MODEL, s1)
            s3 = expand_acronym_layer(s2)
            s4 = expand_ontology_layer(s3)
            s5 = run_validation_sieve(s4)
            final_string = compile_boolean_query(s5)
            
            compiled_queries[f"Q{idx}"] = (rq, final_string)
            print(f"Q{idx:<3} | {rq[:63]:<65} | ✅ SUCCESS")
            
        except Exception as e:
            print(f"Q{idx:<3} | {rq[:63]:<65} | ❌ FAILED ({str(e)[:15]})")

    print("\n" + "=" * 100)
    print(" 📋 COMPILATION REPORT DATA PRESET")
    print("=" * 100)
    
    for q_id, (original_rq, boolean_query) in compiled_queries.items():
        print(f"\n[{q_id}] RQ: '{original_rq}'")
        print(f" └─ Compiled String: {boolean_query}")

if __name__ == "__main__":
    run_macro_benchmark()