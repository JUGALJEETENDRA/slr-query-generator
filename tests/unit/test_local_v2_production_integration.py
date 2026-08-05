from __future__ import annotations

import json

from fastapi.testclient import TestClient

from litsync_app import app as server
from litsync_app.prisma import Prisma2020Manifest
from litsync_app.screening.engines import LOCAL_V2_ENGINE, normalize_processing_engine
from litsync_app.screening.local_v2 import (
    BATCH_RUNNER_VERSION,
    compile_protocol_draft,
    run_compiled_local_v2_paper,
)
from litsync_app.screening.local_v2.production import (
    build_local_v2_protocol_draft,
    local_v2_result_to_public_result,
)


client = TestClient(server.app)


class SupportingEngine:
    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        protocol = json.loads(
            prompt.split("PROTOCOL_JSON:\n", 1)[1].split("\n\nPAPER_JSON:\n", 1)[0]
        )
        paper = json.loads(
            prompt.split("PAPER_JSON:\n", 1)[1].split(
                "\n\nREQUIRED_OUTPUT_SHAPE:\n", 1
            )[0]
        )
        unit = paper["evidence_units"][0]
        return {
            "protocol_id": protocol["protocol_id"],
            "paper_id": paper["paper_id"],
            "assessments": [
                {
                    "criterion_id": criterion["id"],
                    "relation": "DIRECT_SUPPORT",
                    "rationale": (
                        "The supplied paper text explicitly supports this criterion."
                    ),
                    "evidence": [{
                        "evidence_id": unit["evidence_id"],
                        "source": unit["source"],
                        "quote": unit["text"][:1200],
                    }],
                }
                for criterion in protocol["criteria"]
            ],
        }


def test_local_v2_is_explicit_and_separate_from_legacy_local():
    assert normalize_processing_engine("local-v2") == LOCAL_V2_ENGINE
    assert normalize_processing_engine("local_ai_v2") == LOCAL_V2_ENGINE
    html = client.get("/").text
    assert '<option value="local" selected>' in html
    assert '<option value="local_v2">' in html
    assert "Local AI v2" in html
    assert "evidence-grounded" in html
    assert "engine === 'local_v2'" in html


def test_local_v2_protocol_draft_preserves_authoritative_order():
    draft = build_local_v2_protocol_draft(
        research_question=(
            "Do language models automate evidence screening?"
        ),
        research_context="Title and abstract screening only.",
        inclusion_criteria=(
            "Human-reviewed benchmark; Empirical evaluation"
        ),
        exclusion_criteria="- Review article\n2. Editorial",
    )
    assert [item["role"] for item in draft["criteria"]] == [
        "REQUIRED_INCLUSION",
        "REQUIRED_INCLUSION",
        "EXCLUSION_TRIGGER",
        "EXCLUSION_TRIGGER",
    ]
    assert [item["description"] for item in draft["criteria"]] == [
        "Human-reviewed benchmark",
        "Empirical evaluation",
        "Review article",
        "Editorial",
    ]
    compiled = compile_protocol_draft(draft)
    assert compiled.success and compiled.protocol is not None


def test_local_v2_public_adapter_exports_exact_evidence():
    compiled = compile_protocol_draft(build_local_v2_protocol_draft(
        research_question=(
            "Can language models automate review screening?"
        ),
        research_context="",
        inclusion_criteria="",
        exclusion_criteria="",
    ))
    assert compiled.success and compiled.protocol is not None
    result = run_compiled_local_v2_paper(
        SupportingEngine(),
        compiled.protocol,
        paper={
            "paper_id": "adapter-001",
            "title": "Language models automate review screening",
            "abstract": (
                "We evaluated automated title and abstract screening."
            ),
        },
    )
    public = local_v2_result_to_public_result(
        result,
        resource_profile="balanced",
        resumed=False,
    )
    assert public["decision"] == "KEEP"
    assert public["validation_status"] == "validated"
    assert public["triage_basis"] == "PRIMARY_KEEP_FAST_PATH"
    assert public["evidence"][0]["quote"] == (
        "Language models automate review screening"
    )
    assert public["cache_hit"] is False


def test_local_v2_endpoint_reports_its_own_architecture(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(server, "PRISMA_STORE", Prisma2020Manifest())
    monkeypatch.setattr(
        server.PROGRESS,
        "start_job",
        lambda job_id: True,
    )
    started = []

    class NoopThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            started.append(self.kwargs)

    monkeypatch.setattr(server, "Thread", NoopThread)
    response = client.post(
        "/screen_csv",
        data={
            "question": (
                "Can language models automate review screening?"
            ),
            "screening_engine": "local_v2",
        },
        files={
            "file": (
                "papers.csv",
                b"Title,Abstract\nPaper,Text",
                "text/csv",
            )
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["screening_engine"] == "local_v2"
    assert payload["architecture_version"] == BATCH_RUNNER_VERSION
    assert payload["local_profile"] == "local-v2"
    assert payload["prisma"]["screening_engine"] == "local_v2"
    assert started[0]["kwargs"]["screening_engine"] == "local_v2"



def test_local_v2_screen_csv_executes_and_resumes_every_row(
    monkeypatch,
    tmp_path,
):
    import pandas as pd

    from litsync_app.screening import bulk
    from litsync_app.screening.bulk import ScreeningProgress, ScreeningSession

    class FakeProfile:
        resolved_tier = "balanced"
        resource_profile = "balanced"
        num_ctx = 4096
        keep_alive = "5m"

        @staticmethod
        def as_dict():
            return {"resource_profile": "balanced"}

    class CountingSupportingEngine(SupportingEngine):
        def __init__(self):
            self.calls = 0

        def generate(self, model, prompt, schema, *, timeout_seconds=None):
            self.calls += 1
            return super().generate(
                model,
                prompt,
                schema,
                timeout_seconds=timeout_seconds,
            )

    engines = []

    def make_engine(_profile):
        engine = CountingSupportingEngine()
        engines.append(engine)
        return engine

    monkeypatch.setattr(bulk, "PROGRESS", ScreeningProgress())
    monkeypatch.setattr(bulk, "SCREENING_SESSION", ScreeningSession())
    monkeypatch.setattr(bulk, "PRISMA_STORE", Prisma2020Manifest())
    monkeypatch.setattr(
        bulk,
        "resolve_runtime_profile",
        lambda *_args, **_kwargs: FakeProfile(),
    )
    monkeypatch.setattr(bulk, "OllamaStructuredEngine", make_engine)

    csv_path = tmp_path / "mixed.csv"
    output_path = tmp_path / "screened.csv"
    pd.DataFrame([
        {
            "Title": "Language models automate review screening",
            "Abstract": "",
        },
        {
            "Title": "",
            "Abstract": "We evaluated automated title and abstract screening.",
        },
        {
            "Title": "",
            "Abstract": "",
        },
    ]).to_csv(csv_path, index=False)

    first = bulk.screen_csv(
        str(csv_path),
        "Can language models automate review screening?",
        output_path=str(output_path),
        progress_job_id="m10-local-v2-first",
        screening_engine="local_v2",
        resume=True,
    )
    first_rows = pd.read_csv(output_path, keep_default_na=False)
    assert first["screened_total_rows"] == 3
    assert first["keep"] == 2
    assert first["maybe"] == 1
    assert first["reject"] == 0
    assert first["no_screenable_text_count"] == 1
    assert first["resumed_count"] == 0
    assert engines[0].calls == 2
    assert first_rows["Decision"].tolist() == ["KEEP", "KEEP", "MAYBE"]
    assert first_rows["Source_Row_Index"].tolist() == [0, 1, 2]

    second = bulk.screen_csv(
        str(csv_path),
        "Can language models automate review screening?",
        output_path=str(output_path),
        progress_job_id="m10-local-v2-second",
        screening_engine="local_v2",
        resume=True,
    )
    assert second["screened_total_rows"] == 3
    assert second["resumed_count"] == 3
    assert second["fresh_count"] == 0
    assert engines[1].calls == 0
