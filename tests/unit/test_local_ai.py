from __future__ import annotations

import json

import pandas as pd

from litsync_app.screening.bulk import ScreeningSession
from litsync_app.screening.local_ai import (
    ARCHITECTURE_VERSION,
    assess_paper,
    screen_csv_with_local_ai,
    screening_prompt,
)


class FakeProgress:
    def __init__(self) -> None:
        self.resumed = 0
        self.retries = 0
        self.counts = None

    def set_resumed_count(self, _job, count): self.resumed = count
    def begin_batches(self, *_args): pass
    def update_batch(self, *_args): pass
    def record_retry(self, _job): self.retries += 1
    def update_counts(self, _job, total, keep, maybe, reject):
        self.counts = (total, keep, maybe, reject)
    def finish(self, _job): pass


def _response(decision: str, evidence: str) -> dict:
    return {
        "content": json.dumps({
            "decision": decision,
            "confidence": 0.91,
            "reason": "Short semantic judgment.",
            "evidence_quote": evidence,
        }),
        "eval_count": 25,
    }


def test_prompt_is_short_domain_neutral_and_contains_research_inputs():
    prompt = screening_prompt(
        question="Does intervention X improve outcome Y?",
        context="Adults in community settings",
        inclusion="Primary comparative studies",
        exclusion="Reviews",
        title="Trial of intervention X",
        abstract="We compare X and usual care.",
    )
    assert len(prompt) < 1500
    assert "intervention X" in prompt
    assert "Primary comparative studies" in prompt
    assert "blockchain" not in prompt.casefold()
    assert "federated" not in prompt.casefold()


def test_invalid_evidence_is_retried_and_never_accepted():
    calls = []
    responses = iter([
        _response("KEEP", "invented quote"),
        _response("KEEP", "Trial of intervention X"),
    ])

    def generate(prompt, **_kwargs):
        calls.append(prompt)
        return next(responses)

    result = assess_paper(
        question="Question", context="", inclusion="Include trials", exclusion="",
        title="Trial of intervention X", abstract="We report measured outcomes.",
        generate=generate,
    )
    assert result["decision"] == "KEEP"
    assert result["validation_status"] == "validated"
    assert result["attempts"] == 2
    assert len(calls) == 2


def test_balanced_quote_wrappers_are_removed_without_changing_evidence():
    result = assess_paper(
        question="Question", context="", inclusion="", exclusion="Reviews",
        title="A review", abstract="This review summarizes prior work.",
        generate=lambda *_args, **_kwargs: _response(
            "REJECT", '\\"This review summarizes prior work.\\"'
        ),
    )
    assert result["validation_status"] == "validated"
    assert result["evidence_quote"] == "This review summarizes prior work."


def test_repeated_technical_failure_becomes_explicit_maybe():
    def generate(_prompt, **_kwargs):
        raise TimeoutError("offline")

    result = assess_paper(
        question="Question", context="", inclusion="", exclusion="",
        title="Paper", abstract="Abstract", generate=generate,
    )
    assert result["decision"] == "MAYBE"
    assert result["validation_status"] == "technical_failure"
    assert result["failure_class"] == "local_inference_failure"
    assert len(result["validation_errors"]) == 3


def test_csv_path_preserves_source_data_and_resumes_validated_rows(tmp_path):
    frame = pd.DataFrame([
        {"Title": "Included trial", "Abstract": "We evaluated treatment and report outcomes.", "DOI": "10.1/a"},
        {"Title": "Review article", "Abstract": "This review summarizes prior studies.", "DOI": "10.1/b"},
    ])
    output = tmp_path / "runs" / "screened-job.csv"

    def generate(prompt, **_kwargs):
        if "Included trial" in prompt:
            return _response("KEEP", "We evaluated treatment and report outcomes.")
        return _response("REJECT", "This review summarizes prior studies.")

    progress = FakeProgress()
    session = ScreeningSession()
    first = screen_csv_with_local_ai(
        frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
        research_question="Which treatments improve outcomes?", research_context="",
        inclusion_criteria="Evaluated treatments", exclusion_criteria="Reviews",
        output_path=str(output), job_id="job", input_fingerprint="input",
        resume=False, progress=progress, screening_session=session,
        model="test-qwen", generate=generate,
    )
    assert first["architecture_version"] == ARCHITECTURE_VERSION
    assert (first["keep"], first["reject"], first["maybe"]) == (1, 1, 0)
    written = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert written["DOI"].tolist() == ["10.1/a", "10.1/b"]
    assert written["Validation_Status"].tolist() == ["validated", "validated"]

    def should_not_generate(*_args, **_kwargs):
        raise AssertionError("validated rows should resume")

    resumed_progress = FakeProgress()
    second = screen_csv_with_local_ai(
        frame=frame, valid=frame, title_col="Title", abstract_col="Abstract",
        research_question="Which treatments improve outcomes?", research_context="",
        inclusion_criteria="Evaluated treatments", exclusion_criteria="Reviews",
        output_path=str(tmp_path / "runs" / "screened-resume.csv"),
        job_id="resume", input_fingerprint="input", resume=True,
        progress=resumed_progress, screening_session=ScreeningSession(),
        model="test-qwen", generate=should_not_generate,
    )
    assert second["resumed_count"] == 2
    assert resumed_progress.resumed == 2
