from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from litsync_app.integrations import gemini_web_fast_screening as fast
from litsync_app.integrations.gemini_web_fast_prompt import ARCHITECTURE_VERSION, PROMPT_VERSION
from litsync_app.screening.bulk import ScreeningProgress, ScreeningSession
from litsync_app.screening.engines import GEMINI_WEB_FAST_ENGINE, normalize_processing_engine
from litsync_app.validation.gold import create_blinded_sample


class FakeFastBrowser:
    def __init__(self, responder=None):
        self.responder = responder or self._default_response
        self.prompts = []
        self.started = 0
        self.closed = 0
        self.active_pages = 0
        self.peak_active_pages = 0
        self.pages_opened = 0
        self.pages_closed = 0
        self.activity_callback = None

    async def start(self):
        self.started += 1

    async def close(self):
        self.closed += 1

    async def submit_fresh(self, prompt, *, timeout_seconds=None):
        self.prompts.append(prompt)
        self.pages_opened += 1
        self.active_pages += 1
        self.peak_active_pages = max(self.peak_active_pages, self.active_pages)
        if self.activity_callback:
            self.activity_callback(self.active_pages)
        try:
            await asyncio.sleep(.003)
            return self.responder(prompt, len(self.prompts))
        finally:
            self.active_pages -= 1
            self.pages_closed += 1
            if self.activity_callback:
                self.activity_callback(self.active_pages)

    @staticmethod
    def _default_response(prompt, _number):
        if "screening rubric" in prompt and "Required JSON schema" in prompt and "Papers:" not in prompt:
            return json.dumps({
                "review_summary": "Relevant studies only",
                "inclusion_criteria": [{"criterion_id": "I1", "text": "Relevant"}],
                "exclusion_criteria": [{"criterion_id": "E1", "text": "Exclude reviews"}],
                "interpretation_notes": [], "ambiguity_rules": [], "evidence_requirements": [],
            })
        papers = json.loads(prompt.split("Papers:\n", 1)[1].split("\n\nRequired JSON schema:", 1)[0])
        return json.dumps({"items": [assessment(paper) for paper in papers]})


def assessment(paper, decision="KEEP", confidence=.95, risk_flags=None):
    statuses = {
        "KEEP": ("MET", "NOT_MET"),
        "REJECT": ("NOT_MET", "NOT_MET"),
        "MAYBE": ("UNCLEAR", "UNCLEAR"),
    }
    inclusion, exclusion = statuses[decision]
    return {
        "paper_id": paper["paper_id"], "decision": decision,
        "confidence": confidence, "reason": f"{decision} assessment",
        "evidence_quote": paper["title"] if decision != "MAYBE" else "",
        "inclusion_assessments": [{"criterion_id": "I1", "status": inclusion}],
        "exclusion_assessments": [{"criterion_id": "E1", "status": exclusion}],
        "risk_flags": risk_flags or [],
    }


def internal_assessment(
    decision="KEEP", confidence=.95, *, valid=True, failure_class="", risk_flags=None,
):
    return {
        "paper_id": "0", "decision": decision if valid else "MAYBE",
        "model_decision": decision, "confidence": confidence,
        "reason": f"{decision} assessment", "evidence_quote": "Paper 0",
        "inclusion_assessments": [], "exclusion_assessments": [],
        "risk_flags": risk_flags or [], "validation_errors": [],
        "validation_status": "valid" if valid else "contradiction",
        "failure_class": failure_class, "valid": valid, "evidence_valid": valid,
    }


def output_row(primary, verifier=None):
    return fast._row(
        {
            "paper_id": "0", "order": 0, "title": "Paper 0",
            "abstract": "Abstract evidence", "original": {
                "Title": "Paper 0", "Abstract": "Abstract evidence",
            },
        },
        primary,
        verifier,
        origin="fresh_verification" if verifier else "fresh_primary",
        protocol_id="protocol-test",
    )


@pytest.mark.parametrize("model_decision", ["KEEP", "REJECT"])
def test_technical_primary_fallback_can_never_become_definitive(model_decision):
    primary = internal_assessment(
        model_decision, valid=False, failure_class="browser_or_transport_failure",
    )
    verifier = internal_assessment(model_decision, valid=True)
    row = output_row(primary, verifier)
    assert row["Decision"] == "MAYBE"
    assert row["Confidence"] == 0
    assert row["Validation_Status"] == "safe_fallback"
    assert row["Execution_Origin"] == "technical_fallback"
    assert row["Primary_Decision"] == model_decision
    assert row["Verifier_Decision"] == model_decision


def test_unresolved_verifier_contradiction_becomes_maybe():
    primary = internal_assessment("REJECT", valid=True)
    verifier = internal_assessment(
        "REJECT", valid=False, failure_class="validation_contradiction",
    )
    row = output_row(primary, verifier)
    assert row["Decision"] == "MAYBE"
    assert row["Confidence"] == 0
    assert row["Failure_Class"] == "validation_contradiction"
    assert row["Agreement_Status"] == "verification_failed"


def test_validated_primary_only_and_blind_resolution_remain_definitive():
    primary_only = output_row(internal_assessment("KEEP", .95))
    assert primary_only["Decision"] == "KEEP"
    assert primary_only["Validation_Status"] == "validated"

    primary_maybe = internal_assessment("MAYBE", .5)
    verifier_keep = internal_assessment("KEEP", .9)
    resolved = output_row(primary_maybe, verifier_keep)
    assert resolved["Decision"] == "KEEP"
    assert resolved["Validation_Status"] == "validated"
    assert resolved["Agreement_Status"] == "resolved_by_verifier"


def make_frame(count=100, missing=0):
    return pd.DataFrame({
        "Title": [f"Paper {index}" for index in range(count)],
        "Abstract": ["" if index < missing else f"Abstract evidence {index}" for index in range(count)],
        "Original": [f"value-{index}" for index in range(count)],
    })


def run_fast(tmp_path, frame, browser, *, resume=False):
    progress = ScreeningProgress()
    session = ScreeningSession()
    job_id = "job-fast"
    progress.start_job(job_id)
    progress.begin_screening(job_id, len(frame), ARCHITECTURE_VERSION)
    session.begin(job_id, str(tmp_path / "screened.csv"), ARCHITECTURE_VERSION)
    result = fast.screen_csv_with_gemini_web_fast(
        frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
        research_question="Question", research_context="Context",
        inclusion_criteria="Relevant", exclusion_criteria="Exclude reviews",
        output_path=str(tmp_path / "screened.csv"), job_id=job_id,
        input_fingerprint="input", source_dataset_fingerprint="dataset",
        resume=resume, limit=0, progress=progress, screening_session=session,
        browser_factory=lambda: browser,
    )
    return result, pd.read_csv(tmp_path / "screened.csv", keep_default_na=False), progress


def test_default_100_papers_use_ten_primary_batches_and_three_fresh_tabs(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_WEB_FAST_BATCH_SIZE", raising=False)
    monkeypatch.delenv("GEMINI_WEB_FAST_CONCURRENCY", raising=False)
    browser = FakeFastBrowser()
    result, output, _ = run_fast(tmp_path, make_frame(), browser)
    assert result["primary_batch_size"] == 10
    assert result["primary_batches_submitted"] == 10
    assert result["primary_papers_requested"] == 100
    assert 1 < browser.peak_active_pages <= 3
    assert browser.pages_opened == 11  # protocol plus ten primary batches
    assert browser.pages_closed == browser.pages_opened
    assert browser.started == browser.closed == 1
    assert result["browser_context_started"] == 1
    assert result["fresh_primary_count"] == 100
    assert result["transport_failure_count"] == 0
    assert result["pages_opened"] == result["pages_closed"] == 11
    assert len(output) == 100
    assert output["Source_Row_Index"].nunique() == 100
    assert "Original" in output


def test_missing_abstract_is_direct_maybe_and_never_submitted(tmp_path):
    browser = FakeFastBrowser()
    result, output, _ = run_fast(tmp_path, make_frame(10, missing=2), browser)
    missing = output[output["Route_Used"] == "missing_abstract"]
    assert len(missing) == result["missing_abstract_count"] == 2
    assert set(missing["Decision"]) == {"MAYBE"}
    assert set(missing["Confidence"]) == {0.0}
    submitted = "\n".join(browser.prompts)
    assert '"paper_id": "0"' not in submitted
    assert '"paper_id": "1"' not in submitted


@pytest.mark.parametrize("domain", ["medical", "software", "education", "energy"])
def test_domain_content_does_not_change_local_validation(domain, tmp_path):
    frame = pd.DataFrame({"Title": [f"{domain} alpha", f"{domain} beta"], "Abstract": ["one", "two"], "Original": [1, 2]})
    result, output, _ = run_fast(tmp_path, frame, FakeFastBrowser())
    assert result["total_papers"] == 2
    assert set(output["Decision"]) == {"KEEP"}


def test_schema_invalid_batch_gets_one_fresh_page_retry(monkeypatch, tmp_path):
    calls = {"batch": 0}
    def responder(prompt, number):
        if "Papers:" not in prompt:
            return FakeFastBrowser._default_response(prompt, number)
        calls["batch"] += 1
        if calls["batch"] == 1:
            return "not-json"
        return FakeFastBrowser._default_response(prompt, number)
    browser = FakeFastBrowser(responder)
    result, output, _ = run_fast(tmp_path, make_frame(5), browser)
    assert result["retry_count"] == 1
    assert result["transport_failure_count"] == 0
    assert result["primary_batches_submitted"] == 1
    assert len(output) == 5
    assert browser.pages_opened == browser.pages_closed == 3


def test_retry_exhaustion_becomes_safe_maybe(tmp_path):
    def responder(prompt, number):
        return FakeFastBrowser._default_response(prompt, number) if "Papers:" not in prompt else "{}"
    result, output, _ = run_fast(tmp_path, make_frame(5), FakeFastBrowser(responder))
    assert set(output["Decision"]) == {"MAYBE"}
    assert result["safe_fallback_count"] == 5
    assert result["transport_failure_count"] == 0
    assert not set(output["Decision"]) & {"KEEP", "REJECT"}


def test_protocol_failure_uses_transparent_original_criteria(tmp_path):
    def responder(prompt, number):
        if "Papers:" not in prompt:
            return "{}"
        papers = json.loads(prompt.split("Papers:\n", 1)[1].split("\n\nRequired JSON schema:", 1)[0])
        return json.dumps({"items": [assessment(p) for p in papers]})
    result, _, _ = run_fast(tmp_path, make_frame(5), FakeFastBrowser(responder))
    assert result["rubric"]["inclusion_criteria"][0]["text"] == "Relevant"
    assert result["rubric"]["exclusion_criteria"][0]["text"] == "Exclude reviews"


def test_reject_maybe_low_keep_and_risk_receive_blind_verification(tmp_path):
    primary_number = {"value": 0}
    verification_prompts = []
    def responder(prompt, number):
        if "Papers:" not in prompt:
            return FakeFastBrowser._default_response(prompt, number)
        papers = json.loads(prompt.split("Papers:\n", 1)[1].split("\n\nRequired JSON schema:", 1)[0])
        if "prediction-blind" in prompt:
            verification_prompts.append(prompt)
            return json.dumps({"items": [assessment(p, "KEEP", .9) for p in papers]})
        decisions = [("REJECT", .9, []), ("MAYBE", .4, []), ("KEEP", .7, []), ("KEEP", .95, ["substantive"]), ("KEEP", .95, [])]
        return json.dumps({"items": [assessment(p, *decisions[i]) for i, p in enumerate(papers)]})
    result, output, _ = run_fast(tmp_path, make_frame(5), FakeFastBrowser(responder))
    assert result["verification_papers_requested"] == 4
    assert len(verification_prompts) == 1
    blind = verification_prompts[0]
    assert "Primary_Decision" not in blind and "primary confidence" not in blind.casefold()
    assert "one continuous span verbatim" in blind
    assert "never insert ellipses" in blind
    assert output.loc[4, "Route_Used"] == "primary_only"
    assert output.loc[0, "Decision"] == "MAYBE"  # REJECT versus blind KEEP disagreement
    assert output.loc[1, "Decision"] == "KEEP"   # primary MAYBE resolved by blind KEEP


def test_invalid_evidence_is_not_definitive_and_is_verified(tmp_path):
    def responder(prompt, number):
        if "Papers:" not in prompt:
            return FakeFastBrowser._default_response(prompt, number)
        papers = json.loads(prompt.split("Papers:\n", 1)[1].split("\n\nRequired JSON schema:", 1)[0])
        values = [assessment(p) for p in papers]
        if "prediction-blind" not in prompt:
            values[0]["evidence_quote"] = "invented quote"
        return json.dumps({"items": values})
    result, output, _ = run_fast(tmp_path, make_frame(5), FakeFastBrowser(responder))
    assert result["verification_papers_requested"] == 1
    assert output.loc[0, "Route_Used"] == "blind_verification"


def test_duplicate_and_unknown_ids_cannot_remove_output_rows(tmp_path):
    def responder(prompt, number):
        if "Papers:" not in prompt:
            return FakeFastBrowser._default_response(prompt, number)
        papers = json.loads(prompt.split("Papers:\n", 1)[1].split("\n\nRequired JSON schema:", 1)[0])
        item = assessment(papers[0])
        item2 = dict(item)
        if number % 2:
            item2["paper_id"] = "unknown"
        return json.dumps({"items": [item, item2]})
    _, output, _ = run_fast(tmp_path, make_frame(5), FakeFastBrowser(responder))
    assert len(output) == 5
    assert output["Source_Row_Index"].nunique() == 5


def test_checkpoint_identity_and_resume_are_fast_v1_only(tmp_path):
    first_browser = FakeFastBrowser()
    first, first_output, _ = run_fast(tmp_path, make_frame(5), first_browser)
    second_browser = FakeFastBrowser()
    second, second_output, _ = run_fast(tmp_path, make_frame(5), second_browser, resume=True)
    assert first["architecture_version"] == second["architecture_version"] == ARCHITECTURE_VERSION
    assert second["resumed_count"] == 5
    assert second["fresh_primary_count"] == 0
    assert second["primary_batches_submitted"] == 0
    assert second_browser.pages_opened == 0  # complete resume needs no browser or network
    assert set(second_output["Execution_Origin"]) == {"resume"}
    checkpoints = list((tmp_path / ".gemini_web_fast" / "checkpoints").glob("*.json"))
    assert checkpoints and "gemini-web-fast-v1" in checkpoints[0].read_text(encoding="utf-8")
    assert "gemini_web_v24" not in checkpoints[0].read_text(encoding="utf-8")


def test_resume_count_includes_only_selected_checkpoint_rows(tmp_path):
    run_fast(tmp_path, make_frame(100), FakeFastBrowser())
    result, output, _ = run_fast(
        tmp_path, make_frame(5), FakeFastBrowser(), resume=True,
    )
    assert result["resumed_count"] == 5
    assert result["fresh_primary_count"] == 0
    assert len(output) == 5


def test_safe_fallback_checkpoint_rows_never_poison_resume(tmp_path):
    frame = make_frame(100)
    run_fast(tmp_path, frame, FakeFastBrowser())
    checkpoint = next((tmp_path / ".gemini_web_fast" / "checkpoints").glob("*.csv"))
    poisoned = pd.read_csv(checkpoint, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    poisoned["Decision"] = "MAYBE"
    poisoned["Confidence"] = "0"
    poisoned["Validation_Status"] = "safe_fallback"
    poisoned["Failure_Class"] = "browser_or_transport_failure"
    poisoned["Route_Used"] = "safe_fallback"
    poisoned["Execution_Origin"] = "technical_fallback"
    poisoned["Primary_Assessment_JSON"] = json.dumps(
        fast._technical_maybe(
            "placeholder",
            "Gemini screening did not return a complete valid assessment.",
            "browser_or_transport_failure",
        )
    )
    poisoned.to_csv(checkpoint, index=False, encoding="utf-8-sig")

    browser = FakeFastBrowser()
    result, output, _ = run_fast(tmp_path, frame, browser, resume=True)

    assert result["resumed_count"] == 0
    assert result["fresh_primary_count"] == 100
    assert result["primary_papers_requested"] == 100
    assert result["primary_batches_submitted"] == 10
    assert result["browser_context_started"] == 1
    assert result["pages_opened"] == result["pages_closed"] == 11
    assert browser.pages_opened == 11
    assert set(output["Execution_Origin"]) == {"fresh_primary"}
    assert set(output["Validation_Status"]) == {"validated"}


def test_checkpoint_metadata_persists_actual_fast_runtime_counters(tmp_path):
    result, _, _ = run_fast(tmp_path, make_frame(5), FakeFastBrowser())
    metadata_path = next((tmp_path / ".gemini_web_fast" / "checkpoints").glob("*.json"))
    counters = json.loads(metadata_path.read_text(encoding="utf-8"))["counters"]
    for key in (
        "resumed_count", "fresh_primary_count", "primary_batches_submitted",
        "browser_context_started", "pages_opened", "pages_closed",
        "peak_simultaneous_tabs", "transport_failure_count",
    ):
        assert counters[key] == result[key]


def test_prompt_and_output_versions_are_fixed(tmp_path):
    result, output, _ = run_fast(tmp_path, make_frame(5), FakeFastBrowser())
    assert result["screening_engine"] == "gemini_web_fast"
    assert set(output["Prompt_Version"]) == {PROMPT_VERSION}
    assert set(output["Architecture_Version"]) == {ARCHITECTURE_VERSION}


def test_every_output_row_has_one_protocol_id_and_gold_validation_accepts_it(tmp_path):
    result, output, _ = run_fast(tmp_path, make_frame(5), FakeFastBrowser())
    assert result["protocol_id"]
    assert set(output["Protocol_ID"]) == {result["protocol_id"]}
    assert set(output["Review_Protocol_ID"]) == {result["protocol_id"]}
    created = create_blinded_sample(
        output.to_dict(orient="records"), "Question", tmp_path / "gold",
        "job-fast", sample_size=5, manifest_root=tmp_path / "private",
    )
    assert created["sample_size"] == 5
    manifest = json.loads(Path(created["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["protocol_id"] == result["protocol_id"]


def test_completed_progress_retains_fast_batch_and_runtime_metadata(tmp_path):
    result, _, progress = run_fast(tmp_path, make_frame(5), FakeFastBrowser())
    snapshot = progress.snapshot("job-fast")
    assert snapshot["status"] == "finished"
    assert snapshot["primary_batch_size"] == result["primary_batch_size"]
    assert snapshot["primary_batches_submitted"] == result["primary_batches_submitted"]
    assert snapshot["primary_batches_completed"] == result["primary_batches_completed"]
    assert snapshot["verification_batches_submitted"] == result["verification_batches_submitted"]
    assert snapshot["verification_batches_completed"] == result["verification_batches_completed"]
    assert snapshot["peak_simultaneous_tabs"] == result["peak_simultaneous_tabs"]
    assert snapshot["runtime_seconds"] == result["runtime_seconds"]


def test_fast_engine_is_active_and_bypasses_external_orchestrator_source():
    import inspect
    import litsync_app.screening.bulk as bulk
    assert normalize_processing_engine("gemini-web-fast") == GEMINI_WEB_FAST_ENGINE
    source = inspect.getsource(bulk.screen_csv)
    fast_branch = source.split("if selected_engine == GEMINI_WEB_FAST_ENGINE:", 1)[1].split(
        "if selected_engine == GEMINI_WEB_V24_ENGINE:", 1,
    )[0]
    assert "screen_csv_with_gemini_web_fast" in fast_branch
    assert "ExternalAIScreeningOrchestrator" not in fast_branch
    assert "screen_csv_with_gemini_web_v24" not in fast_branch
