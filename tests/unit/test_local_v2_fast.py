from __future__ import annotations

import pytest

from litsync_app.screening.local.engine import GenerationResult, LocalAIOutputError
from litsync_app.screening.local_v2 import compile_protocol_draft
from litsync_app.screening.local_v2.production import build_local_v2_protocol_draft


def _protocol():
    compiled = compile_protocol_draft(build_local_v2_protocol_draft(
        research_question="Which studies evaluate the supplied intervention?",
        inclusion_criteria="The paper evaluates the supplied intervention.",
        exclusion_criteria="The paper is a review without an original evaluation.",
    ))
    assert compiled.protocol
    return compiled.protocol


def _paper(paper_id="p-1"):
    return {"paper_id": paper_id, "title": "Applied intervention evaluation", "abstract": "We evaluated the intervention in practice."}


class ScriptedEngine:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        self.calls.append({"model": model, "prompt": prompt})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return GenerationResult(value=response, model=model, elapsed_seconds=1.0, prompt_tokens=1, output_tokens=1, tokens_per_second=1.0, model_duration_seconds=1.0, total_duration_seconds=1.0)


def _decision(paper_id, decision, criterion_id, evidence_id="title_001"):
    return {"items": [{"p": paper_id, "d": decision, "c": [criterion_id], "e": [evidence_id]}]}


def test_fast_prompt_is_binary_and_blind():
    from litsync_app.screening.local_v2.fast import _build_prompt
    from litsync_app.screening.local_v2.runner import LocalV2Paper
    protocol = _protocol()
    paper = LocalV2Paper.model_validate(_paper())
    prompt = _build_prompt(protocol, [paper], stage="primary")
    reviewer = _build_prompt(protocol, [paper], stage="reviewer")
    assert '"d":"K or R"' in prompt
    assert "high-precision KEEP gate" in prompt
    assert "MAYBE" not in prompt and "U means" not in prompt and "abstain" not in prompt.lower()
    assert "primary output" not in reviewer.lower()


def test_primary_keep_is_final_with_one_model_call():
    from litsync_app.screening.local_v2.fast import run_compiled_local_v2_fast_batch
    protocol = _protocol()
    engine = ScriptedEngine(_decision("p-1", "K", protocol.criteria[0].id))
    result = run_compiled_local_v2_fast_batch(engine, protocol, papers=[_paper()], resume=False)
    assert result.results[0].final_policy.decision == "KEEP"
    assert result.results[0].route == "FAST_BINARY_PRIMARY_KEEP"
    assert len(engine.calls) == 1


@pytest.mark.parametrize(("review", "final", "route"), [("R", "REJECT", "FAST_BINARY_REVIEWER_REJECT"), ("K", "KEEP", "FAST_BINARY_REVIEWER_KEEP")])
def test_primary_reject_gets_blind_binary_review(review, final, route):
    from litsync_app.screening.local_v2.fast import run_compiled_local_v2_fast_batch
    protocol = _protocol()
    exclusion = protocol.criteria[1].id
    engine = ScriptedEngine(_decision("p-1", "R", exclusion, "abstract_001"), _decision("p-1", review, protocol.criteria[0].id))
    result = run_compiled_local_v2_fast_batch(engine, protocol, papers=[_paper()], resume=False)
    assert result.results[0].final_policy.decision == final
    assert result.results[0].route == route
    assert len(engine.calls) == 2
    assert "primary" not in engine.calls[1]["prompt"].lower()


def test_primary_gate_metrics_count_direct_keeps_and_reviewer_candidates():
    from litsync_app.screening.local_v2.fast import run_compiled_local_v2_fast_batch
    protocol = _protocol()
    engine = ScriptedEngine(
        {"items": [
            {"p": "p-keep", "d": "K", "c": [protocol.criteria[0].id], "e": ["title_001"]},
            {"p": "p-review", "d": "R", "c": [protocol.criteria[1].id], "e": ["abstract_001"]},
        ]},
        _decision("p-review", "R", protocol.criteria[1].id, "abstract_001"),
    )
    result = run_compiled_local_v2_fast_batch(engine, protocol, papers=[_paper("p-keep"), _paper("p-review")], resume=False)
    assert result.metrics.primary_papers_assessed == 2
    assert result.metrics.primary_direct_keep_count == 1
    assert result.metrics.reviewer_candidate_count == 1
    assert result.metrics.reviewer_papers_assessed == 1
    assert result.metrics.reviewer_reject_count == 1


def test_invalid_reject_evidence_retries_then_fails():
    from litsync_app.screening.local_v2.fast import run_compiled_local_v2_fast_batch
    protocol = _protocol()
    invalid = _decision("p-1", "R", protocol.criteria[1].id, "abstract_999")
    engine = ScriptedEngine(invalid, invalid)
    with pytest.raises(LocalAIOutputError, match="p-1"):
        run_compiled_local_v2_fast_batch(engine, protocol, papers=[_paper()], resume=False)
    assert len(engine.calls) == 2


def test_missing_text_fails_before_model_call():
    from litsync_app.screening.local_v2.fast import run_compiled_local_v2_fast_batch
    engine = ScriptedEngine()
    with pytest.raises(ValueError, match="p-empty"):
        run_compiled_local_v2_fast_batch(engine, _protocol(), papers=[{"paper_id": "p-empty", "title": "", "abstract": ""}], resume=False)
    assert engine.calls == []


def test_fast_public_output_is_never_maybe():
    from litsync_app.screening.local_v2.fast import local_v2_fast_result_to_public_result, run_compiled_local_v2_fast_batch
    protocol = _protocol()
    result = run_compiled_local_v2_fast_batch(ScriptedEngine(_decision("p-1", "K", protocol.criteria[0].id)), protocol, papers=[_paper()], resume=False)
    public = local_v2_fast_result_to_public_result(result.results[0], resource_profile="balanced", resumed=False)
    assert public["decision"] == "KEEP"
    assert public["validation_status"] == "validated"
