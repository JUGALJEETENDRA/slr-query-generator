from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import time
from pathlib import Path
from queue import Empty
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "xdq_checkpoint.json"
GROUNDED_MODELS = [
    "qwen3.5:4b",
    "qwen3:4b-instruct-2507-q4_K_M",
    "phi4-mini:3.8b-q4_K_M",
]
DIRECT_MODELS = [
    "qwen3:8b",
    "qwen3.5:4b",
    "hf.co/mradermacher/Autobool-Qwen4b-No-reasoning-GGUF:Q4_K_M",
    "qwen3:4b-instruct-2507-q4_K_M",
    "phi4-mini:3.8b-q4_K_M",
    "llama3.1:8b",
]
DIRECT_PROMPT = """You are an expert systematic-review search strategist. Convert the research
question into a concise, high-recall Boolean query. Preserve specialist concepts and relationships,
add only direct academic synonyms, quote phrases, join synonyms with OR, join concept groups with
AND, and return only the query.\n\nResearch question:\n"""
AUTOBOOL_PROMPT = """You are an expert systematic review information specialist. Formulate a
high-recall Boolean query in MEDLINE format for PubMed. Use free-text terms and controlled terms,
combine direct synonyms within concepts using OR, combine concepts using AND, do not add date
limits, and output only the query inside <answer></answer> tags.\n\nResearch topic:\n"""


def load_cases() -> list[dict[str, Any]]:
    basic = json.loads((Path(__file__).parent / "capability_benchmark_questions.json").read_text(encoding="utf-8"))
    hard = json.loads((Path(__file__).parent / "xdq_hard_cases.json").read_text(encoding="utf-8"))
    normalized = []
    for index, case in enumerate(basic, start=1):
        expected = [case.get("technology", ""), case.get("domain", ""), case.get("task", "")]
        normalized.append({
            **case,
            "id": f"legacy-{case.get('id', index)}",
            "expected_concepts": [value for value in expected if value],
            "acceptable_variants": [],
            "forbidden_concepts": [],
            "known_relevant_papers": [],
        })
    return normalized + hard


def _worker(result_queue: mp.Queue, strategy: str, model: str, question: str, deadline: float) -> None:
    try:
        if strategy == "grounded":
            from direct_ai_generator import generate_query_bundle
            bundle = generate_query_bundle(question, model=model, deadline_seconds=deadline)
            result_queue.put({
                "query": bundle.google_scholar,
                "concepts": bundle.concepts,
            })
        else:
            from ollama_client import ask_ollama
            is_autobool = "autobool" in model.lower()
            response = ask_ollama(
                (AUTOBOOL_PROMPT if is_autobool else DIRECT_PROMPT) + question,
                model=model,
                timeout=max(0.5, deadline - 0.25),
                num_predict=320,
            ).strip()
            match = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL | re.IGNORECASE)
            query = match.group(1).strip() if match else response
            result_queue.put({"query": query, "concepts": {}})
    except Exception as exc:
        result_queue.put({"error": f"{type(exc).__name__}: {exc}"})


def run_with_timeout(strategy: str, model: str, question: str, timeout: float = 15.0) -> dict[str, Any]:
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_worker, args=(result_queue, strategy, model, question, timeout))
    started = time.perf_counter()
    process.start()
    process.join(timeout)
    runtime_ms = round((time.perf_counter() - started) * 1000, 1)
    if process.is_alive():
        process.terminate()
        process.join(0.5)
        if process.is_alive():
            process.kill()
            process.join(0.5)
        return {"status": "TIMEOUT", "runtime_ms": runtime_ms}
    try:
        payload = result_queue.get_nowait()
    except Empty:
        return {"status": "ERROR", "runtime_ms": runtime_ms, "error": "worker returned no result"}
    if payload.get("error"):
        return {"status": "ERROR", "runtime_ms": runtime_ms, **payload}
    return {"status": "OK", "runtime_ms": runtime_ms, **payload}


def evaluate(case: dict[str, Any], query: str) -> dict[str, Any]:
    lowered = query.lower()
    expected = [str(value) for value in case.get("expected_concepts", [])]
    acceptable = [str(value) for value in case.get("acceptable_variants", [])]
    missing = [term for term in expected if term.lower() not in lowered]
    forbidden = [
        term for term in case.get("forbidden_concepts", []) if str(term).lower() in lowered
    ]
    syntax_valid = (
        query.count("(") == query.count(")")
        and query.count('"') % 2 == 0
        and bool(query.strip())
    )
    groups = len(re.findall(r"\bAND\b", query, flags=re.IGNORECASE)) + 1 if query.strip() else 0
    return {
        "boolean_valid": syntax_valid,
        "literal_concept_coverage": 1.0 if not expected else round((len(expected) - len(missing)) / len(expected), 4),
        "missing_expected": missing,
        "acceptable_variants_present": [term for term in acceptable if term.lower() in lowered],
        "forbidden_present": forbidden,
        "and_group_count": groups,
    }


def evaluate_retrieval(
    known_relevant: list[str],
    retrieved: list[str],
    *,
    total_candidates: int | None = None,
) -> dict[str, Any]:
    """Compute retrieval measures only from externally supplied gold judgments."""
    gold = {str(value).strip().lower() for value in known_relevant if str(value).strip()}
    ranked = [str(value).strip().lower() for value in retrieved if str(value).strip()]
    if not gold:
        return {"available": False, "reason": "missing_gold_relevance_labels"}
    retrieved_set = set(ranked)
    relevant_retrieved = len(gold & retrieved_set)
    recall = relevant_retrieved / len(gold)
    precision = relevant_retrieved / len(retrieved_set) if retrieved_set else 0.0
    beta_squared = 9.0
    denominator = beta_squared * precision + recall
    f3 = ((1 + beta_squared) * precision * recall / denominator) if denominator else 0.0

    def recall_at(limit: int) -> float:
        return len(gold & set(ranked[:limit])) / len(gold)

    wss95 = None
    if total_candidates and total_candidates > 0 and recall >= 0.95:
        screened_fraction = min(len(ranked), total_candidates) / total_candidates
        wss95 = max(0.0, 1.0 - screened_fraction - 0.05)
    return {
        "available": True,
        "recall": round(recall, 6),
        "recall_at_50": round(recall_at(50), 6),
        "recall_at_100": round(recall_at(100), 6),
        "precision": round(precision, 6),
        "f3": round(f3, 6),
        "wss_at_95": round(wss95, 6) if wss95 is not None else None,
        "retrieved_count": len(ranked),
        "gold_count": len(gold),
    }


def checkpoint(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 120-topic LitSync-XDQ benchmark safely.")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--strategy", choices=("direct", "grounded"), default="grounded")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    timeout = min(15.0, max(0.5, args.timeout))
    cases = load_cases()[:args.limit] if args.limit else load_cases()
    models = args.models or (GROUNDED_MODELS if args.strategy == "grounded" else DIRECT_MODELS)
    results: list[dict[str, Any]] = []
    for case in cases:
        for model in models:
            execution = run_with_timeout(args.strategy, model, case["question"], timeout)
            if execution.get("query"):
                execution["metrics"] = evaluate(case, execution["query"])
            results.append({
                "case_id": case["id"],
                "suite": case.get("suite", "XDQ"),
                "domain": case.get("domain", ""),
                "question": case["question"],
                "strategy": args.strategy,
                "model": model,
                **execution,
            })
            checkpoint(results, args.output)
            print(f"{case['id']} | {model} | {execution['status']} | {execution['runtime_ms']} ms", flush=True)
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
