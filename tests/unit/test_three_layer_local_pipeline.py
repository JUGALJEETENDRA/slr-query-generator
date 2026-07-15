from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

import bulk_screen
from benchmark_local_screening import compare_screening_runs
from local_ai.contracts import ReviewProtocol
from local_ai.engine import GenerationResult
from local_ai.hardware import HardwareSnapshot, resolve_runtime_profile
from local_ai.three_layer import (
    DEEP_MODEL,
    EDGE_MODEL,
    THREE_LAYER_PROMPT_VERSION,
    TRIAGE_MODEL,
    LayerResult,
    ThreeLayerLocalOrchestrator,
    assessment_batch_prompt,
    critic_batch_prompt,
    triage_batch_prompt,
)
from local_ai.prompts import protocol_critic_prompt, protocol_prompt


def _profile():
    hardware = HardwareSnapshot(
        total_ram_gb=24.0, available_ram_gb=18.0, cpu_cores=8, platform="test",
        gpu_name="RTX 3050", gpu_vram_gb=6.0,
        installed_models={TRIAGE_MODEL: 1, DEEP_MODEL: 1, EDGE_MODEL: 1},
    )
    return resolve_runtime_profile("performance", "balanced", hardware)


def _protocol():
    return ReviewProtocol.model_validate({
        "research_question": "Does a glimmer operate on a norvax process?",
        "objective": "Find direct glimmer operation.",
        "scope_interpretation": "Include direct applications.",
        "criteria": [{
            "id": "direct_application", "kind": "inclusion", "required": True,
            "description": "The glimmer operates on a norvax process.",
            "expected_evidence": "Direct application evidence.", "source": "research_question",
        }],
        "expected_relationships": ["glimmer operates on norvax"],
        "semantic_boundaries": [
            "A glimmer must operate on the process; merely describing both is insufficient."
        ],
        "ambiguities": [], "model": DEEP_MODEL,
        "prompt_version": THREE_LAYER_PROMPT_VERSION,
    }).with_identity()


class QueueEngine:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []
        self.unloaded = []

    def generate(self, model, prompt, schema):
        self.calls.append((model, schema.__name__, prompt))
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return GenerationResult(
            value, model, 0.08, prompt_tokens=80, output_tokens=16,
            tokens_per_second=200.0, model_duration_seconds=0.07,
            total_duration_seconds=0.08,
        )

    def unload(self, model):
        self.unloaded.append(model)


def _paper(number, abstract="The glimmer operates on a norvax process."):
    return {"id": f"p{number}", "title": f"Paper {number}", "abstract": abstract}


def _triage_item(paper_id, decision="KEEP", risk="LOW", evidence="abstract_001"):
    basis = {
        "KEEP": "S",
        "REJECT": "C",
        "MAYBE": "U",
    }[decision]
    return {
        "p": paper_id, "d": decision, "k": risk, "b": basis,
        "e": [evidence] if evidence else [],
    }


def _assessment_item(paper_id, decision="KEEP", risk="LOW", verdict="MET", evidence="abstract_001"):
    return {
        "p": paper_id, "d": decision, "k": risk, "r": "The relationship is directly supported.",
        "c": [{"c": "direct_application", "v": verdict, "e": evidence, "r": "Direct evidence is present."}],
    }


def test_layer_one_sends_four_papers_in_one_3b_call(tmp_path):
    papers = [_paper(i) for i in range(4)]
    engine = QueueEngine([{"items": [_triage_item(p["id"]) for p in papers]}])
    progress_events = []
    pipeline = ThreeLayerLocalOrchestrator(_profile(), engine, QueueEngine([]))
    pipeline.cache.directory = tmp_path
    results, events = pipeline.triage_batch(
        "Does a glimmer operate on norvax?", papers, on_batch=progress_events.append
    )
    assert len(engine.calls) == 1
    assert engine.calls[0][0] == TRIAGE_MODEL
    assert set(results) == {p["id"] for p in papers}
    assert events[0]["batch_size"] == 4
    assert progress_events[0]["completed_papers"] == 4
    assert progress_events[0]["invalid_papers"] == 0
    assert progress_events[0]["decision_counts"]["KEEP"] == 4
    assert all(result.result["decision"] == "KEEP" for result in results.values())
    assert "OUTPUT SHAPE:" in engine.calls[0][2]
    assert "SCHEMA:" not in engine.calls[0][2]


def test_valid_items_are_not_repeated_when_bad_batch_ids_are_split(tmp_path):
    papers = [_paper(i) for i in range(3)]
    engine = QueueEngine([
        {"items": [_triage_item("p0")]},
        {"items": [_triage_item("p1")]},
        {"items": [_triage_item("p2")]},
    ])
    pipeline = ThreeLayerLocalOrchestrator(_profile(), engine, QueueEngine([]))
    pipeline.cache.directory = tmp_path
    results, _ = pipeline.triage_batch("Does a glimmer operate on norvax?", papers)
    assert set(results) == {"p0", "p1", "p2"}
    assert len(engine.calls) == 3
    assert sum('"p":"p0"' in call[2] for call in engine.calls) == 1


def test_invalid_evidence_retries_then_fails_safe(tmp_path):
    paper = _paper(1)
    invalid = {"items": [_triage_item("p1", evidence="abstract_999")]}
    engine = QueueEngine([invalid, invalid])
    progress_events = []
    pipeline = ThreeLayerLocalOrchestrator(_profile(), engine, QueueEngine([]))
    pipeline.cache.directory = tmp_path
    results, _ = pipeline.triage_batch(
        "Does a glimmer operate on norvax?", [paper], on_batch=progress_events.append
    )
    result = results["p1"].result
    assert result["decision"] == "MAYBE"
    assert result["validation_status"] == "unresolved"
    assert result["evidence"] == []
    assert sum(event["completed_papers"] for event in progress_events) == 1
    assert sum(event["invalid_papers"] > 0 for event in progress_events) == 2


def test_routing_is_categorical_not_numeric():
    def layer(decision, risk, valid=True):
        return LayerResult(
            {"decision": decision, "decision_risk": risk,
             "validation_status": "validated" if valid else "unresolved"},
            None, None, 0, 0,
        )
    assert not ThreeLayerLocalOrchestrator.needs_deep_review(layer("KEEP", "LOW"))
    assert ThreeLayerLocalOrchestrator.needs_deep_review(layer("KEEP", "BORDERLINE"))
    assert ThreeLayerLocalOrchestrator.needs_deep_review(layer("REJECT", "LOW"))
    assert not ThreeLayerLocalOrchestrator.needs_edge_critic(layer("REJECT", "LOW"))
    assert ThreeLayerLocalOrchestrator.needs_edge_critic(layer("REJECT", "BORDERLINE"))
    assert ThreeLayerLocalOrchestrator.needs_edge_critic(layer("MAYBE", "HIGH"))


def test_protocol_provenance_wording_is_normalized_without_becoming_user_scope():
    raw = _protocol().model_dump(mode="json")
    raw["criteria"][0]["source"] = "research_context_for_interpretation_only"
    normalized = ThreeLayerLocalOrchestrator._normalize_protocol_provenance(raw)
    assert normalized["criteria"][0]["source"] == "research_question"

    raw["criteria"][0]["source"] = "user"
    normalized = ThreeLayerLocalOrchestrator._normalize_protocol_provenance(raw)
    assert normalized["criteria"][0]["source"] == "user"

    normalized = ThreeLayerLocalOrchestrator._normalize_protocol_provenance(
        raw, allow_user=False
    )
    assert normalized["criteria"][0]["source"] == "research_question"

    raw["criteria"].append({
        **raw["criteria"][0], "id": "invented_exclusion",
        "kind": "exclusion", "source": "research_context",
    })
    normalized = ThreeLayerLocalOrchestrator._normalize_protocol_provenance(raw)
    assert [item["id"] for item in normalized["criteria"]] == ["direct_application"]


def test_research_context_changes_protocol_identity_but_is_not_repeated_in_paper_batches():
    pipeline = ThreeLayerLocalOrchestrator(_profile(), QueueEngine([]), QueueEngine([]))
    without_context = pipeline.run_protocol_id("Does a glimmer operate on norvax?")
    with_context = pipeline.run_protocol_id(
        "Does a glimmer operate on norvax?", research_context="Norvax is a process, not a location."
    )
    assert without_context != with_context
    assert "Norvax is a process" in protocol_prompt(
        "Does a glimmer operate on norvax?", "", "", "Norvax is a process, not a location."
    )
    protocol = _protocol().model_copy(update={
        "research_context": "This long background must not be repeated for every paper."
    })
    batch_prompt = triage_batch_prompt(
        protocol.research_question, "", "", [_paper(1)],
        protocol.research_context, protocol,
    )
    assert "This long background must not be repeated" not in batch_prompt
    assert protocol.objective not in batch_prompt
    assert protocol.research_question in batch_prompt


def test_definitive_triage_with_wrong_generic_basis_fails_safe(tmp_path):
    paper = _paper(1)
    wrong = _triage_item("p1", "REJECT", "LOW")
    wrong["b"] = "U"
    engine = QueueEngine([{"items": [wrong]}, {"items": [wrong]}])
    pipeline = ThreeLayerLocalOrchestrator(_profile(), engine, QueueEngine([]))
    pipeline.cache.directory = tmp_path
    result = pipeline.triage_batch("Does a glimmer operate on norvax?", [paper])[0]["p1"].result
    assert result["decision"] == "MAYBE"
    assert result["validation_status"] == "unresolved"


def test_4b_deep_and_critic_use_different_prompts_and_no_8b(tmp_path):
    paper = _paper(1)
    triage = LayerResult(
        {"layer_trace": [{"name": "quick_triage"}], "layer_metrics": [],
         "processing_seconds": 0.01}, None, None, 0.01, 0.01,
    )
    deep_engine = QueueEngine([
        {"items": [_assessment_item("p1", "REJECT", "LOW", "NOT_MET")]},
        {"items": [_assessment_item("p1", "KEEP", "LOW", "MET")]},
    ])
    pipeline = ThreeLayerLocalOrchestrator(_profile(), QueueEngine([]), deep_engine)
    pipeline.cache.directory = tmp_path
    deep, _ = pipeline.deep_review_batch(_protocol(), "run", [paper], {"p1": triage})
    pipeline.prepare_edge_critic()
    edge, _ = pipeline.edge_critic_batch(_protocol(), "run", [paper], deep)
    assert edge["p1"].result["decision"] == "KEEP"
    assert [call[0] for call in deep_engine.calls] == [DEEP_MODEL, EDGE_MODEL]
    assert deep_engine.unloaded == ([] if EDGE_MODEL == DEEP_MODEL else [DEEP_MODEL])
    assert all(":8b" not in call[0].lower() and ":14b" not in call[0].lower() for call in deep_engine.calls)
    assert "deep-review layer" in deep_engine.calls[0][2]
    assert "prediction-blind adjudicator" in deep_engine.calls[1][2]
    assert deep_engine.calls[0][2] != deep_engine.calls[1][2]


def test_protocol_boundaries_are_reserved_for_deep_layers_without_domain_rules():
    protocol = _protocol()
    paper = _paper(1)
    boundary = protocol.semantic_boundaries[0]
    triage = triage_batch_prompt(
        protocol.research_question, "", "", [paper], "", protocol,
    )
    deep = assessment_batch_prompt(protocol, [paper])
    prior = {
        "p1": LayerResult(
            {
                "decision": "REJECT", "decision_risk": "BORDERLINE",
                "reason": "DISTINCTIVE_PRIOR_REASON_MUST_BE_HIDDEN",
                "criteria": [{
                    "criterion_id": "direct_application", "verdict": "NOT_MET",
                    "rationale": "DISTINCTIVE_PRIOR_CRITERION_MUST_BE_HIDDEN",
                }],
                "validation_errors": ["Evidence requires independent review."],
            }, None, None, 0, 0,
        )
    }
    critic = critic_batch_prompt(protocol, [paper], prior)
    assert boundary not in triage
    assert all(boundary in prompt for prompt in (deep, critic))
    assert "keyword" in triage.lower()
    assert "semantically entails" in deep
    assert "DISTINCTIVE_PRIOR_REASON_MUST_BE_HIDDEN" not in critic
    assert "DISTINCTIVE_PRIOR_CRITERION_MUST_BE_HIDDEN" not in critic
    assert '"d": "REJECT"' not in critic
    assert "Evidence requires independent review." not in critic
    assert '"needs_independent_validation":true' in critic


def test_semantic_boundaries_default_empty_for_backward_input_shape():
    payload = _protocol().model_dump(mode="json")
    payload.pop("semantic_boundaries")
    assert ReviewProtocol.model_validate(payload).semantic_boundaries == []


def test_protocol_prompts_preserve_logic_and_forbid_invented_mandatory_facets():
    compiled = protocol_prompt(
        "How is a glim used in either a tor or a vex setting?", "", "",
    )
    criticised = protocol_critic_prompt(_protocol(), "", "")
    assert "Do not turn OR alternatives into an AND" in compiled
    assert "every valid answer to the RQ must satisfy" in compiled
    assert "minimal-necessity test" in criticised
    assert "do not promote examples" in criticised


def test_rq_anchor_removes_model_invented_gates_but_preserves_user_criteria():
    raw = _protocol().model_dump(mode="json")
    raw["criteria"].append({
        "id": "invented_model_gate", "kind": "inclusion", "required": True,
        "description": "An inferred implementation detail becomes mandatory.",
        "expected_evidence": "Inferred detail.", "source": "research_question",
    })
    raw["criteria"].append({
        "id": "explicit_user_rule", "kind": "inclusion", "required": True,
        "description": "A rule explicitly supplied by the user.",
        "expected_evidence": "User-requested evidence.", "source": "user",
    })
    anchored = ThreeLayerLocalOrchestrator._anchor_rq_contract(raw)
    assert [item["id"] for item in anchored["criteria"]] == [
        "rq_core_relationship", "explicit_user_rule",
    ]
    assert "invented implementation detail" not in json.dumps(anchored)
    assert anchored["expected_relationships"] == [
        "Use the complete relationship in the original research question without inferred requirements."
    ]


def test_evidence_invalid_deep_item_routes_to_edge_without_model_retry(tmp_path):
    paper = _paper(1)
    triage = LayerResult(
        {"layer_trace": [{"name": "quick_triage"}], "layer_metrics": [],
         "processing_seconds": 0.01}, None, None, 0.01, 0.01,
    )
    invalid = _assessment_item("p1", "KEEP", "LOW", "MET", evidence="")
    engine = QueueEngine([{"items": [invalid]}])
    pipeline = ThreeLayerLocalOrchestrator(_profile(), QueueEngine([]), engine)
    pipeline.cache.directory = tmp_path
    results, _ = pipeline.deep_review_batch(
        _protocol(), "run", [paper], {"p1": triage}
    )
    assert len(engine.calls) == 1
    assert results["p1"].result["decision"] == "MAYBE"
    assert results["p1"].result["validation_status"] == "unresolved"
    assert pipeline.needs_edge_critic(results["p1"])


def test_invalid_edge_critic_cannot_replace_valid_deep_assessment(tmp_path):
    paper = _paper(1)
    triage = LayerResult(
        {"layer_trace": [{"name": "quick_triage"}], "layer_metrics": [],
         "processing_seconds": 0.01}, None, None, 0.01, 0.01,
    )
    invalid_edge = _assessment_item("p1", "KEEP", "LOW", "MET", evidence="")
    engine = QueueEngine([
        {"items": [_assessment_item("p1", "REJECT", "HIGH", "NOT_MET")]},
        {"items": [invalid_edge]},
    ])
    pipeline = ThreeLayerLocalOrchestrator(_profile(), QueueEngine([]), engine)
    pipeline.cache.directory = tmp_path
    deep, _ = pipeline.deep_review_batch(_protocol(), "run", [paper], {"p1": triage})
    edge, _ = pipeline.edge_critic_batch(_protocol(), "run", [paper], deep)
    result = edge["p1"].result
    assert result["decision"] == "REJECT"
    assert result["validation_status"] == "validated"
    assert result["layer_trace"][-1]["validation_status"] == "invalid_fallback_to_deep"
    assert "Invalid edge critic ignored" in result["validation_warnings"][-1]


def test_bulk_runs_complete_batched_phases_in_order(monkeypatch, tmp_path):
    events = []

    def public(decision, risk, layer, model):
        return {
            "schema_version": "2.0", "decision": decision, "decision_risk": risk,
            "reason": f"{layer} result", "confidence": 0.9, "protocol_id": "run-1",
            "criteria": [], "evidence": [], "uncertainty": [],
            "validation_status": "validated", "validation_errors": [],
            "model_tier": "resident_three_layer_local", "model": model,
            "prompt_version": THREE_LAYER_PROMPT_VERSION, "processing_seconds": 0.01,
            "original_processing_seconds": 0.01,
            "layer_trace": [{"name": layer, "risk": risk}], "layer_metrics": [],
        }

    class FakePipeline:
        def __init__(self, profile):
            self.triage_profile = SimpleNamespace(fast_model=TRIAGE_MODEL)
            self.deep_profile = SimpleNamespace(fast_model=DEEP_MODEL)

        def run_protocol_id(self, *_): return "run-1"
        def compile_protocol(self, *_): events.append(("compile", DEEP_MODEL)); return _protocol()
        def unload_deep(self): events.append(("unload", DEEP_MODEL))
        def triage_batch(self, question, papers, inclusion, exclusion, context, protocol, on_batch=None):
            events.append(("triage", [p["title"] for p in papers]))
            if on_batch: on_batch({"batch_size": len(papers)})
            values = {}
            for paper in papers:
                decision = "KEEP" if paper["title"] == "A" else ("MAYBE" if paper["title"] == "B" else "REJECT")
                values[paper["id"]] = LayerResult(public(decision, "LOW", "quick_triage", TRIAGE_MODEL), None, None, 0, 0)
            return values, []
        def needs_deep_review(self, layer): return layer.result["decision"] != "KEEP"
        def unload_triage(self): events.append(("unload", TRIAGE_MODEL))
        def deep_review_batch(self, protocol, run_id, papers, prior, on_batch=None):
            events.append(("deep", [p["title"] for p in papers]))
            if on_batch: on_batch({"batch_size": len(papers)})
            return {p["id"]: LayerResult(public("REJECT", "LOW", "deep_review", DEEP_MODEL), None, None, 0, 0) for p in papers}, []
        def needs_edge_critic(self, layer): return layer.result["decision"] == "REJECT"
        def prepare_edge_critic(self): events.append(("unload", DEEP_MODEL))
        def edge_critic_batch(self, protocol, run_id, papers, prior, on_batch=None):
            events.append(("edge", [p["title"] for p in papers]))
            if on_batch: on_batch({"batch_size": len(papers)})
            return {p["id"]: LayerResult(public("KEEP", "LOW", "edge_critic", DEEP_MODEL), None, None, 0, 0) for p in papers}, []

    monkeypatch.setattr(bulk_screen, "ThreeLayerLocalOrchestrator", FakePipeline)
    monkeypatch.setattr(bulk_screen, "resolve_runtime_profile", lambda *args: _profile())
    source = tmp_path / "papers.csv"
    output = tmp_path / "screened.csv"
    pd.DataFrame({"Title": ["A", "B", "C"], "Abstract": ["a", "b", "c"]}).to_csv(source, index=False)
    summary = bulk_screen.screen_csv(str(source), "Invented question?", output_path=str(output), resume=False)
    assert events == [
        ("compile", DEEP_MODEL), ("unload", DEEP_MODEL),
        ("triage", ["A", "B", "C"]), ("unload", TRIAGE_MODEL),
        ("deep", ["B", "C"]), ("unload", DEEP_MODEL), ("edge", ["B", "C"]),
    ]
    assert summary["keep"] == 3
    saved = pd.read_csv(output)
    assert set(saved["Prompt_Version"]) == {THREE_LAYER_PROMPT_VERSION}
    assert "Layer_Metrics_JSON" in saved


def test_checkpoint_identity_changes_with_file_contract_or_batch_size(monkeypatch):
    first = bulk_screen._local_checkpoint_key("file-a", "run-a")
    assert first == bulk_screen._local_checkpoint_key("file-a", "run-a")
    assert first != bulk_screen._local_checkpoint_key("file-b", "run-a")
    monkeypatch.setattr(bulk_screen, "TRIAGE_BATCH_SIZE", 7)
    assert first != bulk_screen._local_checkpoint_key("file-a", "run-a")


def test_legacy_rows_never_resume_and_only_complete_new_rows_do(tmp_path):
    checkpoint = tmp_path / "checkpoint.csv"
    base = {
        "Source_Row_Index": 1, "Protocol_ID": "run-1", "Validation_Status": "validated",
        "Decision": "KEEP", "Confidence": 0.99, "Decision_Risk": "LOW",
    }
    pd.DataFrame([{**base, "Prompt_Version": "local-three-layer-v1",
                   "Layer_Trace_JSON": '[{"name":"quick_triage"}]'}]).to_csv(checkpoint, index=False)
    assert bulk_screen._local_resume_rows(str(checkpoint), "run-1") == {}
    pd.DataFrame([{**base, "Prompt_Version": THREE_LAYER_PROMPT_VERSION,
                   "Layer_Trace_JSON": '[{"name":"quick_triage"}]'}]).to_csv(checkpoint, index=False)
    assert set(bulk_screen._local_resume_rows(str(checkpoint), "run-1")) == {"1"}


def test_external_inference_engine_does_not_construct_local_pipeline(monkeypatch):
    import screening_strategies
    calls = []
    screen_calls = []
    class External:
        def __init__(self, profile, inference_engine): calls.append(inference_engine)
        def screen_paper(self, **kwargs):
            screen_calls.append(kwargs)
            return {"schema_version": "2.0", "decision": "MAYBE", "reason": "external",
                    "confidence": 0.5, "model_tier": "external", "model": "gemini"}
    class Forbidden:
        def __init__(self, *args, **kwargs): raise AssertionError("local pipeline constructed")
    monkeypatch.setattr("external_ai.orchestrator.ExternalAIScreeningOrchestrator", External)
    monkeypatch.setattr(screening_strategies, "ThreeLayerLocalOrchestrator", Forbidden)
    result = screening_strategies.screen_candidate(
        title="Paper", abstract="Abstract", research_question="Question?",
        research_context="Local-only context", inference_engine=object()
    )
    assert result["reason"] == "external"
    assert len(calls) == 1
    assert "research_context" not in screen_calls[0]


def test_baseline_comparison_is_diagnostic_and_checks_exact_evidence():
    candidate = [{
        "Source_Row_Index": 1, "Title": "Direct glimmer", "Abstract": "Evidence sentence.",
        "Decision": "KEEP", "Validation_Status": "validated",
        "Evidence_JSON": json.dumps([{"source": "abstract", "quote": "Evidence sentence."}]),
        "Layer_Trace_JSON": json.dumps([
            {"name": "deep_review", "decision": "REJECT"},
            {"name": "edge_critic", "decision": "KEEP"},
        ]),
    }]
    report = compare_screening_runs(candidate, [{"Source_Row_Index": 1, "Decision": "REJECT"}])
    assert report["decision_disagreements"] == 1
    assert report["exact_evidence_rate"] == 1.0
    assert len(report["critic_reversals"]) == 1
    assert report["note"] == "Comparison baseline is not gold truth."
