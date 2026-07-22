from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalScreeningProfile:
    name: str
    structured_rq: bool
    triage_model: str
    protocol_model: str
    deep_model: str
    edge_model: str
    prompt_version: str
    production: bool = False
    rq_frame_version: str = "local-rq-frame-v1"
    evidence_grounded: bool = False
    require_deep_review: bool = False


BASE_TRIAGE = os.getenv("LOCAL_TRIAGE_MODEL", "qwen2.5:3b")
BASE_DEEP = os.getenv("LOCAL_DEEP_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
BASE_EDGE = os.getenv("LOCAL_EDGE_MODEL", BASE_DEEP)

LOCAL_SCREENING_PROFILES = {
    "baseline-v3.12": LocalScreeningProfile(
        "baseline-v3.12", False, BASE_TRIAGE, BASE_DEEP, BASE_DEEP, BASE_EDGE,
        "local-semantic-boundary-v3.12", production=True,
    ),
    "structured-current": LocalScreeningProfile(
        "structured-current", True, BASE_TRIAGE, BASE_DEEP, BASE_DEEP, BASE_EDGE,
        "local-structured-rq-v4.0",
    ),
    "structured-grounded-v4.1": LocalScreeningProfile(
        "structured-grounded-v4.1", True, BASE_TRIAGE, BASE_DEEP, BASE_DEEP, BASE_EDGE,
        "local-evidence-grounded-rq-v4.1", rq_frame_version="local-rq-frame-v2",
        evidence_grounded=True, require_deep_review=True,
    ),
    "structured-grounded-qwen3-8b-v4.1": LocalScreeningProfile(
        "structured-grounded-qwen3-8b-v4.1", True, BASE_TRIAGE,
        "qwen3:8b", "qwen3:8b", "qwen3:8b", "local-evidence-grounded-rq-v4.1",
        rq_frame_version="local-rq-frame-v2", evidence_grounded=True,
        require_deep_review=True,
    ),
    "structured-grounded-qwen35-4b-v4.1": LocalScreeningProfile(
        "structured-grounded-qwen35-4b-v4.1", True, BASE_TRIAGE,
        "qwen3.5:4b", "qwen3.5:4b", "qwen3.5:4b", "local-evidence-grounded-rq-v4.1",
        rq_frame_version="local-rq-frame-v2", evidence_grounded=True,
        require_deep_review=True,
    ),
    "structured-qwen35-4b": LocalScreeningProfile(
        "structured-qwen35-4b", True, BASE_TRIAGE, "qwen3.5:4b", "qwen3.5:4b", "qwen3.5:4b",
        "local-structured-rq-v4.0",
    ),
    "structured-qwen3-8b": LocalScreeningProfile(
        "structured-qwen3-8b", True, BASE_TRIAGE, "qwen3:8b", "qwen3:8b", "qwen3:8b",
        "local-structured-rq-v4.0",
    ),
    "structured-gpt-oss-protocol": LocalScreeningProfile(
        "structured-gpt-oss-protocol", True, BASE_TRIAGE, "gpt-oss:20b", BASE_DEEP, "gpt-oss:20b",
        "local-structured-rq-v4.0",
    ),
}


def resolve_local_screening_profile(name: str | None = None) -> LocalScreeningProfile:
    selected = str(name or "baseline-v3.12").strip()
    try:
        return LOCAL_SCREENING_PROFILES[selected]
    except KeyError as exc:
        raise ValueError(
            f"unknown local screening profile {selected!r}; choose one of "
            + ", ".join(LOCAL_SCREENING_PROFILES)
        ) from exc
