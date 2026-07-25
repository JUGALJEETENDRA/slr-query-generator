from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import evaluation.research_validation as research_validation
from evaluation.research_validation import (
    FROZEN_GEMINI_VERSION,
    _cohen_kappa,
    _repeatability,
    compare_reports,
    export_adjudication,
    export_review_packs,
    generate_report,
    import_adjudication,
    import_review,
    import_root_cause_confirmation,
    initialize_study,
    run_study,
    study_status,
)


FROZEN_HASHES = {
    "gemini_web_automation.py": "8d53705d9617016a2c9be275f7de2fddb3a84d9441734c4dead79b3827e8783d",
    "gemini_web_prompt.py": "d18fb51560c97366d4160f80e917663f2fc138882a9afcdb1f9047b5cb063e8f",
    "gemini_web_screening.py": "f03bdd424fe253187f49e762d3328d6046759fbb7ccc57c235f7b28f3ebfe4af",
}


def _corpus(path: Path, rows: int = 100) -> Path:
    pd.DataFrame({
        "Paper Title": [f"Paper {number}" for number in range(rows)],
        "Paper Abstract": [f"Independent abstract {number}." for number in range(rows)],
        "Year": [2020 + number % 5 for number in range(rows)],
    }).to_csv(path, index=False)
    return path


def _init(tmp_path: Path, rows: int = 100) -> tuple[str, Path, Path]:
    private, output = tmp_path / "private", tmp_path / "outputs"
    result = initialize_study(
        corpus_path=_corpus(tmp_path / "papers.csv", rows),
        research_question="Which interventions improve an outcome?",
        title_column="Paper Title", abstract_column="Paper Abstract", year_column="Year",
        reviewer_ids=["reviewer-a", "reviewer-b"], private_root=private,
        output_root=output, pilot_size=min(100, rows), core_sample_size=min(60, rows),
        risk_sample_size=min(30, max(0, rows - min(60, rows))),
    )
    return result["study_id"], private, output


def _screen_factory(tmp_path: Path, decisions: list[list[str]] | None = None):
    calls = []

    def fake_screen(**kwargs):
        call = len(calls)
        source = pd.read_csv(kwargs["csv_path"])
        selected = decisions[call] if decisions else ["KEEP" if index % 3 == 0 else "MAYBE" for index in range(len(source))]
        rows = []
        for index, (_, paper) in enumerate(source.iterrows()):
            rows.append({
                "Source_Row_Index": str(index), "Title": paper["Title"], "Abstract": paper["Abstract"],
                "Decision": selected[index], "Reason": "Structured reason", "Confidence": "HIGH",
                "Validation_Status": "validated", "Failure_Class": "", "Critic_Route": "",
                "Verification_Status": "not_required", "Criteria_JSON": json.dumps([
                    {"criterion_id": "inc_1", "kind": "inclusion", "verdict": "MET", "scope_support": "SUBSTANTIVE"}
                ]),
            })
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
        protocols = output.parent.parent / "cache" / "gemini_web" / "protocols"
        protocols.mkdir(parents=True, exist_ok=True)
        (protocols / "fixed.json").write_text(json.dumps({
            "protocol_id": "fixed-protocol", "objective": "Test objective",
            "scope_interpretation": "Test scope", "criteria": [
                {"id": "inc_1", "kind": "inclusion", "required": True, "description": "Required relation"}
            ], "ambiguities": [], "semantic_boundaries": [],
        }), encoding="utf-8")
        diagnostics = tmp_path / f"diagnostics-{call}.jsonl"
        diagnostics.write_text(json.dumps({
            "event": "attempt", "submission_number": call + 1, "stage": "primary",
            "retry_number": 0, "outcome": "success", "recovery_action": "none",
            "attempt_duration_ms": 10, "response_selector": "model-response",
            "response_container_count": 1, "response_state": "complete",
            "generation_detected": True, "timeout_stage": "", "fallback_reason": "",
        }) + "\n", encoding="utf-8")
        calls.append(kwargs)
        return {
            "architecture_version": FROZEN_GEMINI_VERSION, "resumed_count": 0,
            "protocol_id": "fixed-protocol", "diagnostics_path": str(diagnostics),
            "runtime_seconds": 1.0, "retry_count": 0, "timeout_fallback_count": 0,
        }

    return fake_screen, calls


def _complete_pack(path: Path, decisions: dict[str, str] | None = None) -> Path:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["Human_Decision"] = [
        (decisions or {}).get(row_id, "KEEP") for row_id in frame["Review_Row_ID"]
    ]
    frame["Reviewer_Rationale"] = "Independent title-and-abstract adjudication."
    frame["Reviewer_Confidence"] = "HIGH"
    completed = path.with_name(path.stem + "-completed.csv")
    frame.to_csv(completed, index=False)
    return completed


def _adjudication_ready_study(tmp_path: Path) -> tuple[str, Path, Path]:
    study, private, _ = _init(tmp_path, rows=12)
    fake_screen, _ = _screen_factory(tmp_path)
    run_study(study, private_root=private, screen=fake_screen)
    packs = export_review_packs(study, private_root=private)["review_packs"]
    first_pack = pd.read_csv(packs["reviewer-a"], dtype=str, keep_default_na=False)
    first = _complete_pack(Path(packs["reviewer-a"]), {
        first_pack["Review_Row_ID"].iloc[0]: "ABSTAIN",
    })
    second = _complete_pack(Path(packs["reviewer-b"]))
    import_review(study, "reviewer-a", first, private_root=private)
    import_review(study, "reviewer-b", second, private_root=private)
    adjudication = export_adjudication(study, private_root=private)
    frame = pd.read_csv(adjudication["adjudication_path"], dtype=str, keep_default_na=False)
    frame["Final_Gold_Decision"] = "MAYBE"
    frame["Adjudication_Rationale"] = "The title and abstract remain genuinely insufficient."
    completed = tmp_path / "adjudication-completed.csv"
    frame.to_csv(completed, index=False)
    return study, private, completed


def _artifact_snapshot(paths: list[Path]) -> dict[str, tuple[bytes, int]]:
    return {
        str(path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in paths
    }


def test_sampling_is_deterministic_and_selected_before_screening(tmp_path):
    study, private, _ = _init(tmp_path)
    manifest = json.loads((private / "research_validation" / study / "study.json").read_text())
    first_core = manifest["sampling"]["core_source_ids"]
    assert manifest["sampling"]["method"] == "uniform_probability_before_screening"
    assert manifest["sampling"]["decision_ratio_targets"] is False
    assert len(first_core) == 60
    assert initialize_study(
        corpus_path=tmp_path / "papers.csv", research_question="Which interventions improve an outcome?",
        title_column="Paper Title", abstract_column="Paper Abstract", year_column="Year",
        reviewer_ids=["reviewer-a", "reviewer-b"], private_root=private,
        output_root=tmp_path / "outputs",
    )["existing"] is True


def test_leakage_fields_are_rejected_before_sampling(tmp_path):
    corpus = pd.DataFrame({"Title": ["A"], "Abstract": ["B"], "Gold_Decision": ["KEEP"]})
    path = tmp_path / "leaked.csv"; corpus.to_csv(path, index=False)
    with pytest.raises(ValueError, match="forbidden evaluation fields"):
        initialize_study(
            corpus_path=path, research_question="Question?", title_column="Title",
            abstract_column="Abstract", reviewer_ids=["a", "b"],
            private_root=tmp_path / "private", output_root=tmp_path / "outputs",
        )


def test_two_fresh_runs_and_blinded_independent_review_packs(tmp_path):
    study, private, _ = _init(tmp_path)
    fake_screen, calls = _screen_factory(tmp_path)
    result = run_study(study, private_root=private, screen=fake_screen)
    assert result["status"] == "SCREENED"
    assert len(calls) == 2 and all(call["resume"] is False for call in calls)
    packs = export_review_packs(study, private_root=private)["review_packs"]
    first = pd.read_csv(packs["reviewer-a"], dtype=str, keep_default_na=False)
    second = pd.read_csv(packs["reviewer-b"], dtype=str, keep_default_na=False)
    forbidden = {"Decision", "Confidence", "Critic_Route", "Protocol_ID", "Gold_Decision", "Source_Row_Index"}
    assert not forbidden.intersection(first.columns)
    assert first["Review_Row_ID"].tolist() != second["Review_Row_ID"].tolist()
    assert set(first["Human_Decision"]) == {""}


def test_review_tampering_and_abstention_require_adjudication(tmp_path):
    study, private, _ = _init(tmp_path, rows=12)
    fake_screen, _ = _screen_factory(tmp_path)
    run_study(study, private_root=private, screen=fake_screen)
    packs = export_review_packs(study, private_root=private)["review_packs"]
    tampered = _complete_pack(Path(packs["reviewer-a"]))
    frame = pd.read_csv(tampered); frame.loc[0, "Title"] = "Changed"; frame.to_csv(tampered, index=False)
    with pytest.raises(ValueError, match="altered"):
        import_review(study, "reviewer-a", tampered, private_root=private)
    first = _complete_pack(Path(packs["reviewer-a"]), {
        pd.read_csv(packs["reviewer-a"])["Review_Row_ID"].iloc[0]: "ABSTAIN"
    })
    second = _complete_pack(Path(packs["reviewer-b"]), {
        pd.read_csv(packs["reviewer-b"])["Review_Row_ID"].iloc[0]: "MAYBE"
    })
    import_review(study, "reviewer-a", first, private_root=private)
    import_review(study, "reviewer-b", second, private_root=private)
    adjudication = export_adjudication(study, private_root=private)
    assert adjudication["disagreement_count"] >= 1
    frame = pd.read_csv(adjudication["adjudication_path"], dtype=str, keep_default_na=False)
    frame["Final_Gold_Decision"] = "MAYBE"
    frame["Adjudication_Rationale"] = "The title and abstract remain insufficient."
    completed = tmp_path / "adjudicated.csv"; frame.to_csv(completed, index=False)
    locked = import_adjudication(study, completed, private_root=private)
    assert locked["status"] == "GOLD_LOCKED"


def test_report_is_insufficient_before_human_gold(tmp_path):
    study, private, _ = _init(tmp_path)
    report = generate_report(study, private_root=private)
    assert report["trust"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "human gold" in report["most_important_limitation"]


def test_complete_dual_review_report_uses_both_repeats(tmp_path):
    study, private, _ = _init(tmp_path, rows=60)
    fake_screen, calls = _screen_factory(tmp_path, decisions=[["KEEP"] * 60, ["KEEP"] * 60])
    run_study(study, private_root=private, screen=fake_screen)
    packs = export_review_packs(study, private_root=private)["review_packs"]
    for reviewer, path in packs.items():
        import_review(study, reviewer, _complete_pack(Path(path)), private_root=private)
    adjudication = export_adjudication(study, private_root=private)
    import_adjudication(study, adjudication["adjudication_path"], private_root=private)
    report = generate_report(study, private_root=private)
    assert report["trust"]["verdict"] in {"TRUST", "CONDITIONAL"}
    assert report["metrics"]["human_review"]["cohen_kappa"] == 1.0
    assert set(report["metrics"]["screening_quality_by_repeat"]) == {"repeat_a", "repeat_b"}
    assert len(report["paired_records"]) == 60
    assert all(call["resume"] is False for call in calls)
    assert all("gold" not in " ".join(call).casefold() for call in calls)


def test_identical_adjudication_reimport_is_byte_preserving(tmp_path):
    study, private, completed = _adjudication_ready_study(tmp_path)
    first = import_adjudication(study, completed, private_root=private)
    manifest_path = private / "research_validation" / study / "study.json"
    gold_path = private / "research_validation" / study / "gold.json"
    before = _artifact_snapshot([manifest_path, gold_path])
    second = import_adjudication(study, completed, private_root=private)
    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert second["gold_fingerprint"] == first["gold_fingerprint"]
    assert _artifact_snapshot([manifest_path, gold_path]) == before


def test_conflicting_decision_and_rationale_cannot_change_reported_study(tmp_path):
    study, private, completed = _adjudication_ready_study(tmp_path)
    import_adjudication(study, completed, private_root=private)
    report = generate_report(study, private_root=private)
    manifest_path = private / "research_validation" / study / "study.json"
    gold_path = private / "research_validation" / study / "gold.json"
    report_json = Path(report["report_path"])
    report_html = report_json.with_suffix(".html")
    protected = [manifest_path, gold_path, report_json, report_html]
    before = _artifact_snapshot(protected)

    decision_conflict = pd.read_csv(completed, dtype=str, keep_default_na=False)
    decision_conflict["Final_Gold_Decision"] = "REJECT"
    decision_path = tmp_path / "decision-conflict.csv"
    decision_conflict.to_csv(decision_path, index=False)
    with pytest.raises(ValueError, match="Gold is immutable"):
        import_adjudication(study, decision_path, private_root=private)
    assert _artifact_snapshot(protected) == before

    rationale_conflict = pd.read_csv(completed, dtype=str, keep_default_na=False)
    rationale_conflict["Adjudication_Rationale"] = "A changed rationale must conflict."
    rationale_path = tmp_path / "rationale-conflict.csv"
    rationale_conflict.to_csv(rationale_path, index=False)
    with pytest.raises(ValueError, match="Gold is immutable"):
        import_adjudication(study, rationale_path, private_root=private)
    assert _artifact_snapshot(protected) == before

    identical = import_adjudication(study, completed, private_root=private)
    assert identical["idempotent"] is True
    assert identical["status"] == "REPORTED"
    assert _artifact_snapshot(protected) == before


def test_inconsistent_gold_manifest_or_report_fingerprint_fails_closed(tmp_path):
    study, private, completed = _adjudication_ready_study(tmp_path)
    import_adjudication(study, completed, private_root=private)
    report = generate_report(study, private_root=private)
    manifest_path = private / "research_validation" / study / "study.json"
    gold_path = private / "research_validation" / study / "gold.json"
    report_path = Path(report["report_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["adjudication"]["gold_fingerprint"] = "conflicting-manifest-fingerprint"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    before = _artifact_snapshot([manifest_path, gold_path, report_path])
    with pytest.raises(ValueError, match="fingerprints disagree"):
        import_adjudication(study, completed, private_root=private)
    assert _artifact_snapshot([manifest_path, gold_path, report_path]) == before


def test_partial_gold_write_before_manifest_update_fails_closed(tmp_path, monkeypatch):
    study, private, completed = _adjudication_ready_study(tmp_path)
    manifest_path = private / "research_validation" / study / "study.json"
    gold_path = private / "research_validation" / study / "gold.json"
    original_save = research_validation._save_manifest

    def interrupted_save(*_args, **_kwargs):
        raise RuntimeError("simulated interruption before manifest update")

    monkeypatch.setattr(research_validation, "_save_manifest", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        import_adjudication(study, completed, private_root=private)
    assert gold_path.exists()
    monkeypatch.setattr(research_validation, "_save_manifest", original_save)
    before = _artifact_snapshot([manifest_path, gold_path])
    with pytest.raises(ValueError, match="fingerprint metadata is incomplete"):
        import_adjudication(study, completed, private_root=private)
    assert _artifact_snapshot([manifest_path, gold_path]) == before


def test_confirmed_gold_adjudication_error_preserves_gold_and_metrics(tmp_path):
    study, private, _ = _init(tmp_path, rows=12)
    fake_screen, _ = _screen_factory(tmp_path, decisions=[["KEEP"] * 12, ["KEEP"] * 12])
    run_study(study, private_root=private, screen=fake_screen)
    packs = export_review_packs(study, private_root=private)["review_packs"]
    for reviewer, path in packs.items():
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame["Human_Decision"] = "KEEP"
        frame.loc[frame["Title"] == "Paper 0", "Human_Decision"] = "REJECT"
        frame["Reviewer_Rationale"] = "Independent eligibility judgment."
        frame["Reviewer_Confidence"] = "HIGH"
        completed = tmp_path / f"{reviewer}-gold-error.csv"
        frame.to_csv(completed, index=False)
        import_review(study, reviewer, completed, private_root=private)
    adjudication = export_adjudication(study, private_root=private)
    import_adjudication(study, adjudication["adjudication_path"], private_root=private)
    initial = generate_report(study, private_root=private)
    assert len(initial["paper_level_root_causes"]) == 1
    confirmation = pd.read_csv(
        Path(initial["report_path"]).with_name("root-cause-confirmation.csv"),
        dtype=str, keep_default_na=False,
    )
    duplicate = pd.concat([confirmation, confirmation.iloc[[0]]], ignore_index=True)
    duplicate["confirmed_root_cause"] = "gold_adjudication_error"
    duplicate["researcher_notes"] = "Duplicate-row rejection test."
    duplicate_path = tmp_path / "duplicate-root-cause.csv"
    duplicate.to_csv(duplicate_path, index=False)
    gold_path = private / "research_validation" / study / "gold.json"
    adjudication_path = Path(adjudication["adjudication_path"])
    report_path = Path(initial["report_path"])
    report_html = report_path.with_suffix(".html")
    protected = _artifact_snapshot([gold_path, adjudication_path, report_path, report_html])
    confirmation_store = private / "research_validation" / study / "root_cause_confirmations.json"
    assert not confirmation_store.exists()
    with pytest.raises(ValueError, match="Duplicate root-cause confirmation rows"):
        import_root_cause_confirmation(study, duplicate_path, private_root=private)
    assert not confirmation_store.exists()
    assert _artifact_snapshot([gold_path, adjudication_path, report_path, report_html]) == protected

    confirmation["confirmed_root_cause"] = "not_a_valid_category"
    confirmation["researcher_notes"] = "Invalid category test."
    invalid = tmp_path / "invalid-root-cause.csv"
    confirmation.to_csv(invalid, index=False)
    with pytest.raises(ValueError, match="approved category"):
        import_root_cause_confirmation(study, invalid, private_root=private)

    confirmation["confirmed_root_cause"] = "gold_adjudication_error"
    confirmation["researcher_notes"] = (
        "The model decision is defensible from the supplied abstract; locked gold remains historical."
    )
    completed = tmp_path / "confirmed-root-cause.csv"
    confirmation.to_csv(completed, index=False)
    protected = _artifact_snapshot([gold_path, adjudication_path])
    quality_before = initial["metrics"]["screening_quality"]
    result = import_root_cause_confirmation(study, completed, private_root=private)
    updated = result["report"]
    assert updated["root_cause_confirmation"]["confirmed_rows"] == 1
    assert updated["interpretation"]["confirmed_gold_adjudication_error_count"] == 1
    assert updated["metrics"]["screening_quality"] == quality_before
    assert _artifact_snapshot([gold_path, adjudication_path]) == protected


def test_operational_cross_domain_evidence_requires_distinct_domains(tmp_path):
    registry = tmp_path / "registry.jsonl"

    def manifest(study_id, question):
        return {
            "study_id": study_id, "review": {"research_question": question},
            "screening": {"protocol_id": "fixed"},
        }

    report = {
        "trust": {"verdict": "CONDITIONAL"},
        "paper_level_root_causes": [],
        "metrics": {"operations": {
            "runs": [{
                "late_session_degradation_observed": True,
                "recovery_actions": {"browser_recycle_after_no_container_timeout": 2},
            }],
            "total_transport_fallbacks": 0,
        }},
    }
    first = research_validation._append_registry(
        registry, manifest("study-a1", "Domain A question"), report,
    )
    same_domain = research_validation._append_registry(
        registry, manifest("study-a2", "Domain A question"), report,
    )
    new_domain = research_validation._append_registry(
        registry, manifest("study-b1", "Domain B question"), report,
    )
    assert first["cross_domain_operational_weakness_confirmed"] is False
    assert same_domain["cross_domain_operational_weakness_confirmed"] is False
    assert new_domain["cross_domain_operational_weakness_confirmed"] is True
    assert new_domain["same_signature_in_independently_reviewed_domains"] == []
    assert new_domain["same_operational_signature_in_independently_reviewed_domains"] == [
        "late_session_degradation", "no_container_timeout_recovery",
    ]


def test_trust_feasibility_reports_sixty_and_seventy_three_boundary():
    assert research_validation._minimum_perfect_sample_for_lower_bound(0.95) == 73
    assert research_validation._wilson(60, 60)["lower"] == 0.9398
    assert research_validation._wilson(73, 73)["lower"] == 0.95


def test_repeatability_detects_direct_contradictions():
    first = {"0": {"Decision": "KEEP"}, "1": {"Decision": "MAYBE"}}
    second = {"0": {"Decision": "REJECT"}, "1": {"Decision": "MAYBE"}}
    metrics = _repeatability(first, second)
    assert metrics["exact_agreement_rate"] == 0.5
    assert metrics["keep_reject_contradictions"] == ["0"]
    assert metrics["repeated_maybe_rows"] == ["1"]


def test_kappa_distinguishes_abstain_from_gold_maybe():
    kappa, agreement = _cohen_kappa(["MAYBE", "ABSTAIN"], ["MAYBE", "KEEP"])
    assert agreement == 0.5
    assert kappa == pytest.approx(0.3333)


def test_version_comparison_rejects_changed_corpus_or_study(tmp_path):
    baseline = {"study_id": "one", "inputs": {"corpus_fingerprint": "a"}, "trust": {"verdict": "TRUST"}, "metrics": {}}
    candidate = {**baseline, "study_id": "two"}
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(json.dumps(baseline)); second.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="same preregistered study"):
        compare_reports(first, second)


def test_frozen_gemini_web_v23_files_are_byte_identical():
    root = Path(__file__).resolve().parents[2]
    for filename, expected in FROZEN_HASHES.items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == expected
