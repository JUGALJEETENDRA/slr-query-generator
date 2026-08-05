from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import litsync_app.screening.local_v2.batch as batch_module
from litsync_app.screening.local_v2 import (
    BATCH_CHECKPOINT_VERSION,
    BATCH_RUNNER_VERSION,
    LocalModelPlan,
    LocalV2BatchCheckpoint,
    LocalV2BatchCheckpointEntry,
    LocalV2BatchPipelineVersions,
    LocalV2BatchRunResult,
    LocalV2Paper,
    ProtocolCriterion,
    ScreeningProtocolV2,
    build_local_v2_batch_id,
    current_local_v2_pipeline_versions,
    is_local_v2_result_resumable,
    local_v2_paper_fingerprint,
    run_compiled_local_v2_batch,
    run_compiled_local_v2_paper,
)


TITLE = "Large language models automate abstract screening."
ABSTRACT = (
    "We evaluate a large language model for title and abstract screening in systematic "
    "reviews."
)
REJECT_ABSTRACT = ABSTRACT + " The work is unrelated to literature-review screening."


@dataclass
class FakeGeneration:
    value: Any
    elapsed_seconds: float = 0.25


class FakeEngine:
    def __init__(self, responses: list[Any] | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []
        self.active = 0
        self.peak_active = 0

    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        assert self.active == 0, "batch runner attempted overlapping local model calls"
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            self.calls.append(
                {
                    "model": model,
                    "prompt": prompt,
                    "schema": schema,
                    "timeout_seconds": timeout_seconds,
                }
            )
            if not self.responses:
                raise AssertionError("unexpected model call")
            response = self.responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            if isinstance(response, FakeGeneration):
                return response
            return FakeGeneration(response)
        finally:
            self.active -= 1


def protocol(
    *, question: str = "Can LLMs automate systematic-review screening?"
) -> ScreeningProtocolV2:
    return ScreeningProtocolV2(
        research_question=question,
        criteria=[
            ProtocolCriterion(
                id="uses_llm",
                role="REQUIRED_INCLUSION",
                description="The paper studies a large language model.",
                expected_evidence="Explicit mention of a large language model.",
                resolution_required=True,
            ),
            ProtocolCriterion(
                id="unrelated_task",
                role="EXCLUSION_TRIGGER",
                description="The work is explicitly unrelated to review screening.",
                expected_evidence="An explicit incompatible task or setting.",
                resolution_required=False,
            ),
        ],
        model="qwen3.5:4b",
    ).with_identity()


def paper(paper_id: str, **updates: Any) -> dict[str, Any]:
    value = {"paper_id": paper_id, "title": TITLE, "abstract": ABSTRACT}
    value.update(updates)
    return value


def citation(evidence_id: str, source: str, quote: str) -> list[dict[str, str]]:
    return [{"evidence_id": evidence_id, "source": source, "quote": quote}]


def envelope(
    proto: ScreeningProtocolV2,
    paper_id: str,
    *,
    uses_llm: str = "DIRECT_SUPPORT",
    unrelated_task: str = "MISSING_OR_UNCLEAR",
    rejection: bool = False,
) -> dict[str, Any]:
    uses_evidence = (
        citation(
            "abstract_001",
            "abstract",
            "We evaluate a large language model for title and abstract screening",
        )
        if uses_llm in {"DIRECT_SUPPORT", "DIRECT_CONTRADICTION"}
        else []
    )
    exclusion_evidence = (
        citation(
            "abstract_002",
            "abstract",
            "The work is unrelated to literature-review screening",
        )
        if rejection and unrelated_task == "DIRECT_SUPPORT"
        else []
    )
    return {
        "protocol_id": proto.protocol_id,
        "paper_id": paper_id,
        "assessments": [
            {
                "criterion_id": "uses_llm",
                "relation": uses_llm,
                "rationale": "The paper explicitly describes its model.",
                "evidence": uses_evidence,
            },
            {
                "criterion_id": "unrelated_task",
                "relation": unrelated_task,
                "rationale": "The exclusion trigger was assessed independently.",
                "evidence": exclusion_evidence,
            },
        ],
    }


def keep_responses(proto: ScreeningProtocolV2, paper_ids: list[str]) -> list[dict[str, Any]]:
    return [envelope(proto, paper_id) for paper_id in paper_ids]


def read_checkpoint(path: Path) -> LocalV2BatchCheckpoint:
    return LocalV2BatchCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))


def checkpoint_identity_fields() -> dict[str, Any]:
    return {
        "checkpoint_version": BATCH_CHECKPOINT_VERSION,
        "batch_runner_version": BATCH_RUNNER_VERSION,
        "pipeline_versions": current_local_v2_pipeline_versions(),
    }


def test_batch_versions_are_frozen():
    assert BATCH_RUNNER_VERSION == "local-v2-batch-v1"
    assert BATCH_CHECKPOINT_VERSION == "local-v2-batch-checkpoint-v1"


def test_pipeline_versions_are_frozen_into_results_and_checkpoints(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    result = run_compiled_local_v2_batch(
        FakeEngine([envelope(proto, "p-1")]),
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
    )
    checkpoint = read_checkpoint(path)

    assert result.pipeline_versions == current_local_v2_pipeline_versions()
    assert checkpoint.pipeline_versions == result.pipeline_versions
    assert result.pipeline_versions.runner_version == "local-v2-runner-v1"
    assert result.pipeline_versions.orchestrator_version == "local-v2-orchestrator-v2"
    assert result.pipeline_versions.assessor_version == "local-v2-assessor-v2"


def test_checkpoint_with_stale_pipeline_version_is_ignored_as_invalid(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    run_compiled_local_v2_batch(
        FakeEngine([envelope(proto, "p-1")]),
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pipeline_versions"]["runner_version"] = "old-runner"
    path.write_text(json.dumps(payload), encoding="utf-8")

    engine = FakeEngine([envelope(proto, "p-1")])
    result = run_compiled_local_v2_batch(
        engine,
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
    )

    assert result.checkpoint_disposition == "IGNORED_INVALID"
    assert result.resumed_positions == []
    assert len(engine.calls) == 1


def test_checkpoint_missing_pipeline_identity_is_ignored_as_invalid(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    run_compiled_local_v2_batch(
        FakeEngine([envelope(proto, "p-1")]),
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("pipeline_versions")
    path.write_text(json.dumps(payload), encoding="utf-8")

    engine = FakeEngine([envelope(proto, "p-1")])
    result = run_compiled_local_v2_batch(
        engine,
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
    )

    assert result.checkpoint_disposition == "IGNORED_INVALID"
    assert result.resumed_positions == []
    assert len(engine.calls) == 1


def test_paper_fingerprint_is_deterministic_after_normalization():
    left = LocalV2Paper(paper_id=" p-1 ", title="  Title  ", abstract=" Abstract ")
    right = LocalV2Paper(paper_id="p-1", title="Title", abstract="Abstract")

    assert local_v2_paper_fingerprint(left) == local_v2_paper_fingerprint(right)
    assert len(local_v2_paper_fingerprint(left)) == 64


def test_batch_id_is_deterministic_for_equivalent_inputs():
    proto = protocol()

    first = build_local_v2_batch_id(
        proto,
        [paper(" p-1 ", title=f"  {TITLE}  ")],
    )
    second = build_local_v2_batch_id(proto, [paper("p-1")])

    assert first == second
    assert len(first) == 20


def test_batch_id_changes_when_paper_order_changes():
    proto = protocol()

    first = build_local_v2_batch_id(proto, [paper("p-1"), paper("p-2")])
    second = build_local_v2_batch_id(proto, [paper("p-2"), paper("p-1")])

    assert first != second


def test_batch_id_changes_when_paper_text_changes():
    proto = protocol()

    first = build_local_v2_batch_id(proto, [paper("p-1")])
    second = build_local_v2_batch_id(proto, [paper("p-1", abstract="Changed")])

    assert first != second


def test_batch_id_changes_when_model_plan_changes():
    proto = protocol()

    first = build_local_v2_batch_id(proto, [paper("p-1")])
    second = build_local_v2_batch_id(
        proto,
        [paper("p-1")],
        model_plan=LocalModelPlan(primary_model="other-primary"),
    )

    assert first != second


def test_batch_rejects_empty_collection_before_model_work():
    with pytest.raises(ValueError, match="at least one paper"):
        run_compiled_local_v2_batch(FakeEngine(), protocol(), papers=[])


def test_batch_rejects_duplicate_paper_ids_before_model_work(tmp_path: Path):
    engine = FakeEngine()
    checkpoint = tmp_path / "checkpoint.json"

    with pytest.raises(ValueError, match="paper_id values must be unique"):
        run_compiled_local_v2_batch(
            engine,
            protocol(),
            papers=[paper("p-1"), paper("p-1")],
            checkpoint_path=checkpoint,
        )

    assert not engine.calls
    assert not checkpoint.exists()


def test_batch_rejects_invalid_mapping_before_checkpoint_or_model(tmp_path: Path):
    engine = FakeEngine()
    checkpoint = tmp_path / "checkpoint.json"

    with pytest.raises(ValidationError):
        run_compiled_local_v2_batch(
            engine,
            protocol(),
            papers=[{"paper_id": "p-1", "title": TITLE, "extra": True}],
            checkpoint_path=checkpoint,
        )

    assert not engine.calls
    assert not checkpoint.exists()


def test_batch_rejects_stale_protocol_identity_before_checkpoint_or_model(tmp_path: Path):
    stale = protocol().model_copy(update={"research_question": "Changed question"})
    engine = FakeEngine()
    checkpoint = tmp_path / "checkpoint.json"

    with pytest.raises(ValueError, match="protocol_id does not match"):
        run_compiled_local_v2_batch(
            engine,
            stale,
            papers=[paper("p-1")],
            checkpoint_path=checkpoint,
        )

    assert not engine.calls
    assert not checkpoint.exists()


def test_batch_without_checkpoint_preserves_order_and_uses_one_call_at_a_time():
    proto = protocol()
    engine = FakeEngine(keep_responses(proto, ["p-1", "p-2", "p-3"]))

    result = run_compiled_local_v2_batch(
        engine,
        proto,
        papers=[paper("p-1"), paper("p-2"), paper("p-3")],
        checkpoint_path=None,
    )

    assert [item.paper.paper_id for item in result.results] == ["p-1", "p-2", "p-3"]
    assert result.checkpoint_disposition == "NOT_REQUESTED"
    assert result.checkpoint_path is None
    assert result.metrics.checkpoint_write_count == 0
    assert result.metrics.keep_count == 3
    assert result.metrics.fresh_count == 3
    assert result.metrics.resumed_count == 0
    assert result.metrics.model_call_count == 3
    assert result.metrics.fresh_model_call_count == 3
    assert engine.peak_active == 1


def test_new_checkpoint_is_initialized_then_replaced_after_every_fresh_paper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    proto = protocol()
    path = tmp_path / "nested" / "checkpoint.json"
    engine = FakeEngine(keep_responses(proto, ["p-1", "p-2", "p-3"]))
    entry_counts: list[int] = []
    original = batch_module._atomic_write_checkpoint

    def recording_write(target, checkpoint):
        entry_counts.append(len(checkpoint.entries))
        original(target, checkpoint)

    monkeypatch.setattr(batch_module, "_atomic_write_checkpoint", recording_write)

    result = run_compiled_local_v2_batch(
        engine,
        proto,
        papers=[paper("p-1"), paper("p-2"), paper("p-3")],
        checkpoint_path=path,
    )

    assert entry_counts == [0, 1, 2, 3]
    assert result.metrics.checkpoint_write_count == 4
    assert path.exists()
    saved = read_checkpoint(path)
    assert saved.complete
    assert [entry.position for entry in saved.entries] == [0, 1, 2]


def test_complete_stable_checkpoint_resumes_without_model_calls(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    first_engine = FakeEngine(keep_responses(proto, ["p-1", "p-2"]))
    first = run_compiled_local_v2_batch(
        first_engine,
        proto,
        papers=[paper("p-1"), paper("p-2")],
        checkpoint_path=path,
    )

    second_engine = FakeEngine()
    second = run_compiled_local_v2_batch(
        second_engine,
        proto,
        papers=[paper("p-1"), paper("p-2")],
        checkpoint_path=path,
        resume=True,
    )

    assert first.batch_id == second.batch_id
    assert second.checkpoint_disposition == "LOADED"
    assert second.resumed_positions == [0, 1]
    assert second.metrics.resumed_count == 2
    assert second.metrics.fresh_count == 0
    assert second.metrics.fresh_model_call_count == 0
    assert second.metrics.checkpoint_write_count == 0
    assert not second_engine.calls


def test_interruption_leaves_previous_atomic_checkpoint_and_resume_continues(
    tmp_path: Path,
):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    interrupted_engine = FakeEngine(
        [
            envelope(proto, "p-1"),
            KeyboardInterrupt(),
        ]
    )

    with pytest.raises(KeyboardInterrupt):
        run_compiled_local_v2_batch(
            interrupted_engine,
            proto,
            papers=[paper("p-1"), paper("p-2"), paper("p-3")],
            checkpoint_path=path,
        )

    partial = read_checkpoint(path)
    assert not partial.complete
    assert [entry.position for entry in partial.entries] == [0]

    resumed_engine = FakeEngine(keep_responses(proto, ["p-2", "p-3"]))
    resumed = run_compiled_local_v2_batch(
        resumed_engine,
        proto,
        papers=[paper("p-1"), paper("p-2"), paper("p-3")],
        checkpoint_path=path,
    )

    assert resumed.resumed_positions == [0]
    assert resumed.metrics.resumed_count == 1
    assert resumed.metrics.fresh_count == 2
    assert [item.paper.paper_id for item in resumed.results] == ["p-1", "p-2", "p-3"]
    assert read_checkpoint(path).complete


def test_missing_text_result_is_deterministic_and_resumable(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"

    first = run_compiled_local_v2_batch(
        FakeEngine(),
        proto,
        papers=[paper("p-1", title=None, abstract=" ")],
        checkpoint_path=path,
    )
    second_engine = FakeEngine()
    second = run_compiled_local_v2_batch(
        second_engine,
        proto,
        papers=[paper("p-1", title=None, abstract=None)],
        checkpoint_path=path,
    )

    assert first.results[0].status == "NO_SCREENABLE_TEXT"
    assert is_local_v2_result_resumable(first.results[0])
    assert second.resumed_positions == [0]
    assert second.metrics.no_screenable_text_count == 1
    assert not second_engine.calls


def test_genuine_semantic_maybe_is_resumable(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    first_engine = FakeEngine(
        [
            envelope(proto, "p-1", uses_llm="MISSING_OR_UNCLEAR"),
            envelope(proto, "p-1", uses_llm="MISSING_OR_UNCLEAR"),
        ]
    )
    first = run_compiled_local_v2_batch(
        first_engine,
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
    )

    assert first.results[0].final_policy.decision == "MAYBE"
    assert not first.results[0].safe_fallback
    assert is_local_v2_result_resumable(first.results[0])

    second_engine = FakeEngine()
    second = run_compiled_local_v2_batch(
        second_engine,
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
    )
    assert second.resumed_positions == [0]
    assert not second_engine.calls


def test_technical_fallback_is_never_resumed(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    first_engine = FakeEngine([RuntimeError("primary down"), TimeoutError("review down")])
    first = run_compiled_local_v2_batch(
        first_engine,
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
    )

    assert first.results[0].safe_fallback
    assert not is_local_v2_result_resumable(first.results[0])

    second_engine = FakeEngine([envelope(proto, "p-1")])
    second = run_compiled_local_v2_batch(
        second_engine,
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
    )

    assert second.resumed_positions == []
    assert second.metrics.fresh_count == 1
    assert second.results[0].final_policy.decision == "KEEP"
    assert any("scheduled for fresh" in warning for warning in second.checkpoint_warnings)
    assert len(second_engine.calls) == 1


def test_mixed_checkpoint_resumes_only_stable_positions(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    first_engine = FakeEngine(
        [
            envelope(proto, "p-1"),
            RuntimeError("primary down"),
            TimeoutError("review down"),
        ]
    )
    first = run_compiled_local_v2_batch(
        first_engine,
        proto,
        papers=[
            paper("p-1"),
            paper("p-2"),
            paper("p-3", title=None, abstract=None),
        ],
        checkpoint_path=path,
    )
    assert first.results[0].final_policy.decision == "KEEP"
    assert first.results[1].safe_fallback
    assert first.results[2].status == "NO_SCREENABLE_TEXT"

    second_engine = FakeEngine([envelope(proto, "p-2")])
    second = run_compiled_local_v2_batch(
        second_engine,
        proto,
        papers=[
            paper("p-1"),
            paper("p-2"),
            paper("p-3", title=None, abstract=None),
        ],
        checkpoint_path=path,
    )

    assert second.resumed_positions == [0, 2]
    assert second.metrics.resumed_count == 2
    assert second.metrics.fresh_count == 1
    assert [item.paper.paper_id for item in second.results] == ["p-1", "p-2", "p-3"]
    assert second.results[1].final_policy.decision == "KEEP"


def test_changed_paper_collection_ignores_mismatched_checkpoint(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    run_compiled_local_v2_batch(
        FakeEngine([envelope(proto, "p-1")]),
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
    )

    engine = FakeEngine([envelope(proto, "p-1")])
    result = run_compiled_local_v2_batch(
        engine,
        proto,
        papers=[paper("p-1", abstract=ABSTRACT + " Changed.")],
        checkpoint_path=path,
    )

    assert result.checkpoint_disposition == "IGNORED_MISMATCH"
    assert result.resumed_positions == []
    assert result.metrics.fresh_count == 1
    assert result.metrics.checkpoint_write_count == 2
    assert any("identity did not match" in warning for warning in result.checkpoint_warnings)


def test_changed_model_plan_ignores_mismatched_checkpoint(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    run_compiled_local_v2_batch(
        FakeEngine([envelope(proto, "p-1")]),
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
    )
    plan = LocalModelPlan(primary_model="replacement-primary")
    engine = FakeEngine([envelope(proto, "p-1")])

    result = run_compiled_local_v2_batch(
        engine,
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
        model_plan=plan,
    )

    assert result.checkpoint_disposition == "IGNORED_MISMATCH"
    assert result.model_plan == plan
    assert engine.calls[0]["model"] == "replacement-primary"


def test_corrupt_checkpoint_is_ignored_and_replaced(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    path.write_text("{not-json", encoding="utf-8")
    engine = FakeEngine([envelope(proto, "p-1")])

    result = run_compiled_local_v2_batch(
        engine,
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
    )

    assert result.checkpoint_disposition == "IGNORED_INVALID"
    assert result.resumed_positions == []
    assert any("invalid and was ignored" in warning for warning in result.checkpoint_warnings)
    assert read_checkpoint(path).complete


def test_resume_false_discards_matching_checkpoint(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    run_compiled_local_v2_batch(
        FakeEngine([envelope(proto, "p-1")]),
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
    )
    engine = FakeEngine([envelope(proto, "p-1")])

    result = run_compiled_local_v2_batch(
        engine,
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
        resume=False,
    )

    assert result.checkpoint_disposition == "DISABLED"
    assert result.resumed_positions == []
    assert result.metrics.fresh_count == 1
    assert result.metrics.checkpoint_write_count == 2
    assert len(engine.calls) == 1


def test_aggregate_metrics_cover_keep_maybe_reject_and_model_routes(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    engine = FakeEngine(
        [
            FakeGeneration(envelope(proto, "p-keep"), 0.1),
            FakeGeneration(envelope(proto, "p-reject", uses_llm="MISSING_OR_UNCLEAR"), 0.2),
            FakeGeneration(
                envelope(
                    proto,
                    "p-reject",
                    unrelated_task="DIRECT_SUPPORT",
                    rejection=True,
                ),
                0.3,
            ),
            FakeGeneration(
                envelope(
                    proto,
                    "p-reject",
                    unrelated_task="DIRECT_SUPPORT",
                    rejection=True,
                ),
                0.4,
            ),
        ]
    )

    result = run_compiled_local_v2_batch(
        engine,
        proto,
        papers=[
            paper("p-keep"),
            paper("p-missing", title=None, abstract=None),
            paper("p-reject", abstract=REJECT_ABSTRACT),
        ],
        checkpoint_path=path,
    )
    metrics = result.metrics

    assert [item.final_policy.decision for item in result.results] == [
        "KEEP",
        "MAYBE",
        "REJECT",
    ]
    assert metrics.total_papers == 3
    assert metrics.keep_count == 1
    assert metrics.maybe_count == 1
    assert metrics.reject_count == 1
    assert metrics.no_screenable_text_count == 1
    assert metrics.safe_fallback_count == 1
    assert metrics.resumable_result_count == 3
    assert metrics.model_call_count == 4
    assert metrics.fresh_model_call_count == 4
    assert metrics.review_used_count == 1
    assert metrics.validator_used_count == 1
    assert metrics.model_elapsed_seconds == 1.0
    assert metrics.fresh_model_elapsed_seconds == 1.0
    assert metrics.route_counts == {
        "PRIMARY_KEEP_FAST_PATH": 1,
        "REJECTION_CONFIRMED": 1,
    }


def test_resumed_metrics_separate_fresh_model_work(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    interrupted_engine = FakeEngine(
        [FakeGeneration(envelope(proto, "p-1"), 0.3), KeyboardInterrupt()]
    )
    with pytest.raises(KeyboardInterrupt):
        run_compiled_local_v2_batch(
            interrupted_engine,
            proto,
            papers=[paper("p-1"), paper("p-2")],
            checkpoint_path=path,
        )

    resumed = run_compiled_local_v2_batch(
        FakeEngine([FakeGeneration(envelope(proto, "p-2"), 0.7)]),
        proto,
        papers=[paper("p-1"), paper("p-2")],
        checkpoint_path=path,
    )

    assert resumed.metrics.model_call_count == 2
    assert resumed.metrics.fresh_model_call_count == 1
    assert resumed.metrics.model_elapsed_seconds == 1.0
    assert resumed.metrics.fresh_model_elapsed_seconds == 0.7


def test_checkpoint_round_trips_through_strict_model(tmp_path: Path):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    run_compiled_local_v2_batch(
        FakeEngine([envelope(proto, "p-1")]),
        proto,
        papers=[paper("p-1")],
        checkpoint_path=path,
    )

    checkpoint = read_checkpoint(path)
    restored = LocalV2BatchCheckpoint.model_validate_json(checkpoint.model_dump_json())

    assert restored == checkpoint


def test_batch_result_round_trips_through_strict_model():
    proto = protocol()
    result = run_compiled_local_v2_batch(
        FakeEngine([envelope(proto, "p-1")]),
        proto,
        papers=[paper("p-1")],
    )

    restored = LocalV2BatchRunResult.model_validate_json(result.model_dump_json())

    assert restored == result


def test_checkpoint_entry_rejects_fingerprint_mismatch():
    proto = protocol()
    run = run_compiled_local_v2_paper(
        FakeEngine([envelope(proto, "p-1")]),
        proto,
        paper=paper("p-1"),
    )

    with pytest.raises(ValidationError, match="fingerprint must match"):
        LocalV2BatchCheckpointEntry(
            position=0,
            paper_fingerprint="0" * 64,
            result=run,
        )


def test_checkpoint_rejects_duplicate_positions():
    proto = protocol()
    run = run_compiled_local_v2_paper(
        FakeEngine([envelope(proto, "p-1")]),
        proto,
        paper=paper("p-1"),
    )
    entry = LocalV2BatchCheckpointEntry(
        position=0,
        paper_fingerprint=local_v2_paper_fingerprint(run.paper),
        result=run,
    )

    with pytest.raises(ValidationError, match="positions must be unique"):
        LocalV2BatchCheckpoint(
            **checkpoint_identity_fields(),
            batch_id="a" * 20,
            protocol_id=proto.protocol_id,
            model_plan=LocalModelPlan(),
            paper_count=2,
            entries=[entry, entry],
            complete=True,
        )


def test_checkpoint_rejects_incorrect_complete_flag():
    proto = protocol()

    with pytest.raises(ValidationError, match="complete flag"):
        LocalV2BatchCheckpoint(
            **checkpoint_identity_fields(),
            batch_id="a" * 20,
            protocol_id=proto.protocol_id,
            model_plan=LocalModelPlan(),
            paper_count=1,
            entries=[],
            complete=True,
        )


def test_checkpoint_rejects_result_from_other_protocol():
    first_protocol = protocol()
    other_protocol = protocol(question="A different valid research question?")
    run = run_compiled_local_v2_paper(
        FakeEngine([envelope(other_protocol, "p-1")]),
        other_protocol,
        paper=paper("p-1"),
    )

    with pytest.raises(ValidationError, match="protocol_id must match"):
        LocalV2BatchCheckpoint(
            **checkpoint_identity_fields(),
            batch_id="a" * 20,
            protocol_id=first_protocol.protocol_id,
            model_plan=LocalModelPlan(),
            paper_count=1,
            entries=[
                LocalV2BatchCheckpointEntry(
                    position=0,
                    paper_fingerprint=local_v2_paper_fingerprint(run.paper),
                    result=run,
                )
            ],
            complete=True,
        )


def test_atomic_replace_failure_preserves_previous_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    proto = protocol()
    path = tmp_path / "checkpoint.json"
    path.write_text("previous-checkpoint\n", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(batch_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        run_compiled_local_v2_batch(
            FakeEngine([envelope(proto, "p-1")]),
            proto,
            papers=[paper("p-1")],
            checkpoint_path=path,
            resume=False,
        )

    assert path.read_text(encoding="utf-8") == "previous-checkpoint\n"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_batch_result_rejects_tampered_metrics():
    proto = protocol()
    result = run_compiled_local_v2_batch(
        FakeEngine([envelope(proto, "p-1")]),
        proto,
        papers=[paper("p-1")],
    )
    payload = result.model_dump(mode="json")
    payload["metrics"]["keep_count"] = 0
    payload["metrics"]["maybe_count"] = 1

    with pytest.raises(ValidationError, match="metrics must match"):
        LocalV2BatchRunResult.model_validate(payload)


def test_batch_result_model_rejects_hidden_resume_position():
    proto = protocol()
    result = run_compiled_local_v2_batch(
        FakeEngine([envelope(proto, "p-1")]),
        proto,
        papers=[paper("p-1")],
    )
    payload = result.model_dump(mode="json")
    payload["resumed_positions"] = [0]

    with pytest.raises(ValidationError, match="resumed positions must match"):
        LocalV2BatchRunResult.model_validate(payload)


def test_batch_result_model_rejects_checkpoint_disposition_without_path():
    proto = protocol()
    result = run_compiled_local_v2_batch(
        FakeEngine([envelope(proto, "p-1")]),
        proto,
        papers=[paper("p-1")],
    )
    payload = result.model_dump(mode="json")
    payload["checkpoint_disposition"] = "LOADED"

    with pytest.raises(ValidationError, match="requires a checkpoint path"):
        LocalV2BatchRunResult.model_validate(payload)


def test_public_exports_are_available_from_package():
    import litsync_app.screening.local_v2 as local_v2

    assert local_v2.run_compiled_local_v2_batch is run_compiled_local_v2_batch
    assert local_v2.BATCH_RUNNER_VERSION == BATCH_RUNNER_VERSION
    assert local_v2.BATCH_CHECKPOINT_VERSION == BATCH_CHECKPOINT_VERSION
