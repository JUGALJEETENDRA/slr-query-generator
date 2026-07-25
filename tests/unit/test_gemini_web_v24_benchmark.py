from pathlib import Path

from evaluation.gemini_web_v24_benchmark import (
    _quality_metrics,
    _run_once,
    _seed_immutable_protocol,
)


def test_v24_benchmark_quality_metrics_do_not_assume_decision_ratios():
    score = {
        "matched_rows": 8,
        "confusion": {
            "KEEP": {"KEEP": 3, "MAYBE": 1, "REJECT": 0},
            "MAYBE": {"KEEP": 0, "MAYBE": 1, "REJECT": 0},
            "REJECT": {"KEEP": 0, "MAYBE": 1, "REJECT": 2},
        },
    }
    metrics = _quality_metrics(score)
    assert metrics == {
        "relevant_recall_keep_or_maybe": 1.0,
        "false_reject_rate": 0.0,
        "definitive_keep_precision": 1.0,
        "manual_review_rate": 0.375,
    }


def test_benchmark_repetitions_use_independent_cache_roots(monkeypatch, tmp_path):
    calls = []

    def fake_screen_csv(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("Source_Row_Index,Decision\n0,KEEP\n", encoding="utf-8")
        diagnostics = output.parent.parent / "cache" / "gemini_web_v24" / "diagnostics" / "run.jsonl"
        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.write_text("", encoding="utf-8")
        diagnostics.with_suffix(".summary.json").write_text(
            '{"runtime_seconds": 1, "retry_count": 0}', encoding="utf-8"
        )
        return {
            "architecture_version": "gemini-web-batched-v2.4",
            "diagnostics_path": str(diagnostics),
            "runtime_seconds": 1,
        }

    monkeypatch.setattr("bulk_screen.screen_csv", fake_screen_csv)
    monkeypatch.setattr(
        "evaluation.gemini_web_v24_benchmark.score_mixed_control",
        lambda *args, **kwargs: {
            "matched_rows": 1,
            "confusion": {
                "KEEP": {"KEEP": 1, "MAYBE": 0, "REJECT": 0},
                "MAYBE": {"KEEP": 0, "MAYBE": 0, "REJECT": 0},
                "REJECT": {"KEEP": 0, "MAYBE": 0, "REJECT": 0},
            },
        },
    )
    common = {
        "version": "v2.4",
        "papers": tmp_path / "papers.csv",
        "gold": tmp_path / "gold.csv",
        "question": "Question?",
        "context": "",
        "inclusion": "",
        "exclusion": "",
        "output_root": tmp_path,
    }
    first = _run_once(**common, repetition=1)
    second = _run_once(**common, repetition=2)
    assert Path(first["screened"]).parent.parent.name == "fresh-1"
    assert Path(second["screened"]).parent.parent.name == "fresh-2"
    assert Path(first["screened"]).parent.parent != Path(second["screened"]).parent.parent


def test_repeatability_seeds_protocol_without_assessment_cache(tmp_path):
    source_cache = tmp_path / "fresh-1" / "cache" / "gemini_web_v24"
    (source_cache / "protocols").mkdir(parents=True)
    (source_cache / "protocols" / "protocol.json").write_text("{}", encoding="utf-8")
    (source_cache / "assessments").mkdir()
    (source_cache / "assessments" / "paper.json").write_text("{}", encoding="utf-8")
    _seed_immutable_protocol(
        tmp_path, "v2.4", source_repetition=1, target_repetition=2,
    )
    target_cache = tmp_path / "fresh-2" / "cache" / "gemini_web_v24"
    assert (target_cache / "protocols" / "protocol.json").read_text(encoding="utf-8") == "{}"
    assert not (target_cache / "assessments").exists()
