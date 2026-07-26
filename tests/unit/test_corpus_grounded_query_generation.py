from __future__ import annotations

from dataclasses import replace
import inspect

from litsync_app.query import generator as query_module

from litsync_app.query.generator import (
    GroundingPaper,
    GeneratedQueryBundle,
    SemanticScholarGrounder,
    StructuredQueryDraft,
    GroundedRefinement,
    compile_boolean_query,
    generate_query_bundle,
)
from litsync_app.screening.local.engine import GenerationResult, LocalAIError
from litsync_app.screening.local.hardware import HardwareSnapshot, RuntimeProfile


def _profile(models=None):
    hardware = HardwareSnapshot(
        total_ram_gb=24.0,
        available_ram_gb=12.0,
        cpu_cores=12,
        platform="Test",
        gpu_name="RTX",
        gpu_vram_gb=6.0,
        installed_models=models or {"qwen3:4b-instruct-2507-q4_K_M": 1},
    )
    return RuntimeProfile(
        requested_tier="auto",
        resolved_tier="performance",
        resource_profile="balanced",
        fast_model="qwen3:8b",
        strong_model="qwen3:8b",
        num_ctx=4096,
        keep_alive="30m",
        concurrency=1,
        memory_reserve_ratio=0.2,
        downgrade_reasons=(),
        hardware=hardware,
        calibration={},
    )


class FakeGrounder(SemanticScholarGrounder):
    def search(self, question, timeout=2.5):
        return [
            GroundingPaper("p1", "Post-quantum migration", "Quantum resistant cryptography for legacy ICS."),
            GroundingPaper("p2", "Quantum resistant cryptography", "Migration of quantum resistant cryptography in industrial control systems."),
            GroundingPaper("p3", "Cybersecurity overview", "Cybersecurity and artificial intelligence."),
        ]


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        self.calls.append((model, schema.__name__, timeout_seconds))
        if schema is StructuredQueryDraft:
            value = {
                "groups": [
                    {"label": "Cryptography", "role": "technology", "terms": ["post-quantum cryptography", "PQC"]},
                    {"label": "Environment", "role": "domain", "terms": ["legacy industrial control systems"]},
                    {"label": "Outcomes", "role": "outcome", "terms": ["interoperability", "security"]},
                    {"label": "Migration", "role": "context", "terms": ["migration strategies"]},
                ],
                "needs_grounding": True,
                "uncertain_terms": ["post-quantum"],
            }
        elif schema is GroundedRefinement:
            value = {"additions": [
                {"group_label": "Cryptography", "term": "quantum resistant cryptography", "support_ids": ["p1", "p2"]},
                {"group_label": "Cryptography", "term": "artificial intelligence", "support_ids": ["p3"]},
                {"group_label": "Environment", "term": "unsupported adjacent concept", "support_ids": ["p1", "p2"]},
            ]}
        else:
            raise AssertionError(schema)
        return GenerationResult(value=value, model=model, elapsed_seconds=0.01)


def test_grounded_generation_adds_only_corpus_supported_non_parent_terms():
    bundle = generate_query_bundle(
        "How do post-quantum migration strategies affect interoperability and security in legacy industrial control systems?",
        profile=_profile(), engine=FakeEngine(), grounder=FakeGrounder(), deadline_seconds=5,
    )
    assert "quantum resistant cryptography" in bundle.google_scholar
    assert "artificial intelligence" not in bundle.google_scholar
    assert "unsupported adjacent concept" not in bundle.google_scholar
    assert bundle.concepts["mode"] == "grounded"
    assert bundle.concepts["grounded_terms"][0]["support"][0]["paper_id"] == "p1"


def test_compiler_repairs_duplicates_and_enforces_group_and_term_caps():
    draft = StructuredQueryDraft.model_validate({
        "groups": [
            {"label": "A", "role": "technology", "terms": ["PQC", "PQC", "post_quantum cryptography", "x", "y"]},
            {"label": "B", "role": "domain", "terms": ["industrial control systems"]},
        ],
        "needs_grounding": False,
    })
    query = compile_boolean_query(draft.groups)
    assert query.count('"PQC"') == 1
    assert '"post quantum cryptography"' in query
    assert query.count("(") == query.count(")")
    assert query.count(" AND ") == 1


def test_preferred_model_is_qwen35_when_installed():
    engine = FakeEngine()
    bundle = generate_query_bundle(
        "Machine learning for software defect prediction",
        profile=_profile({"qwen3.5:4b": 1, "qwen3:4b-instruct-2507-q4_K_M": 1}),
        engine=engine, grounder=FakeGrounder(), deadline_seconds=5,
    )
    assert bundle.concepts["model"] == "qwen3.5:4b"
    assert engine.calls[0][0] == "qwen3.5:4b"


class FailingEngine:
    def generate(self, *args, **kwargs):
        raise LocalAIError("offline")


class EmptyGrounder(SemanticScholarGrounder):
    def search(self, question, timeout=2.5):
        return []


def test_local_failure_returns_valid_literal_fallback():
    bundle = generate_query_bundle(
        "Federated learning for rare cancer prognosis",
        profile=_profile(), engine=FailingEngine(), grounder=EmptyGrounder(), deadline_seconds=2,
    )
    assert bundle.google_scholar
    assert bundle.google_scholar.count("(") == bundle.google_scholar.count(")")
    assert bundle.google_scholar.count('"') % 2 == 0
    assert bundle.concepts["fallback_reason"].startswith("local_draft_failed")


def test_digital_twin_question_uses_semantic_parser_fallback():
    bundle = generate_query_bundle(
        "How are digital twins used in smart manufacturing and Industry 4.0 applications?",
        profile=_profile(), engine=FailingEngine(), grounder=EmptyGrounder(), deadline_seconds=2,
    )
    groups = bundle.concepts["groups"]
    assert [group["role"] for group in groups] == ["technology", "domain"]
    assert groups[0]["terms"][:2] == ["digital twin", "digital twins"]
    assert groups[1]["source_spans"] == ["smart manufacturing", "Industry 4.0 applications"]
    assert "are digital twins used" not in bundle.google_scholar.lower()
    assert bundle.google_scholar.count(" AND ") == 1
    assert bundle.concepts["literal_coverage"] == 1.0
    assert bundle.concepts["generation_status"] == "local_fallback"
    assert bundle.concepts["warning"]
    assert '"digital twin*"' in bundle.scopus
    assert "application*" in bundle.scopus
    assert '"digital twin" OR "digital twins"' in bundle.google_scholar


class DigitalTwinGrounder(SemanticScholarGrounder):
    def search(self, question, timeout=2.5):
        return [
            GroundingPaper(
                "dt1", "Industry 5.0 and intelligent manufacturing",
                "Digital twins enable simulation in intelligent manufacturing and Industry 5.0.",
            ),
            GroundingPaper(
                "dt2", "Digital twins for intelligent manufacturing",
                "Simulation of smart manufacturing systems in Industry 5.0.",
            ),
        ]


def test_model_failure_still_merges_supported_adjacent_corpus_terms():
    bundle = generate_query_bundle(
        "How are digital twins used in smart manufacturing and Industry 4.0 applications?",
        profile=_profile(), engine=FailingEngine(), grounder=DigitalTwinGrounder(), deadline_seconds=2,
    )
    assert "Industry 5.0" in bundle.google_scholar or "industry 5.0" in bundle.google_scholar
    assert bundle.concepts["generation_status"] == "grounded_fallback"
    supported = {item["term"]: item for item in bundle.concepts["grounded_terms"]}
    industry_term = next(term for term in supported if term.lower() == "industry 5.0")
    assert {paper["paper_id"] for paper in supported[industry_term]["support"]} == {"dt1", "dt2"}


def test_unsupported_adjacent_term_is_not_added():
    bundle = generate_query_bundle(
        "How are digital twins used in smart manufacturing and Industry 4.0 applications?",
        profile=_profile(), engine=FailingEngine(), grounder=EmptyGrounder(), deadline_seconds=2,
    )
    assert "Industry 5.0" not in bundle.google_scholar
    assert "simulation" not in bundle.google_scholar


class UnsupportedExpansionEngine:
    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        return GenerationResult(value={
            "groups": [
                {"label": "Technology", "role": "technology", "terms": ["digital twins"]},
                {"label": "Context", "role": "domain", "terms": ["smart manufacturing", "Industry 5.0"]},
                {"label": "Applications", "role": "outcome", "terms": ["applications", "simulation"]},
            ],
            "needs_grounding": True,
            "uncertain_terms": [],
        }, model=model, elapsed_seconds=0.01)


def test_ungrounded_model_cannot_add_adjacent_version_or_use_type():
    bundle = generate_query_bundle(
        "How are digital twins used in smart manufacturing and Industry 4.0 applications?",
        profile=_profile(), engine=UnsupportedExpansionEngine(), grounder=EmptyGrounder(), deadline_seconds=2,
    )
    assert "Industry 5.0" not in bundle.google_scholar
    assert "simulation" not in bundle.google_scholar


def test_explicit_comparator_stays_in_its_own_group():
    bundle = generate_query_bundle(
        "How does federated learning compare with centralized learning in hospitals?",
        profile=_profile(), engine=FailingEngine(), grounder=EmptyGrounder(), deadline_seconds=2,
    )
    comparison_groups = [group for group in bundle.concepts["groups"] if group["role"] == "comparison"]
    assert comparison_groups == [{
        "label": "Comparison", "role": "comparison", "terms": ["centralized learning"],
        "source_spans": ["centralized learning"],
    }]
    assert all(
        "centralized learning" not in group["terms"]
        for group in bundle.concepts["groups"] if group["role"] != "comparison"
    )


TECHNICAL_QUESTION = (
    "How are physics-informed neural networks used for fault diagnosis and remaining useful life "
    "prediction in lithium-ion batteries under varying operating conditions?"
)


class CompleteTechnicalEngine:
    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        return GenerationResult(value={
            "groups": [
                {
                    "label": "Method", "role": "technology",
                    "terms": ["physics-informed neural networks", "PINN", "PINNs"],
                    "source_spans": ["physics-informed neural networks"],
                },
                {
                    "label": "Tasks", "role": "outcome",
                    "terms": ["fault diagnosis", "remaining useful life prediction", "RUL prediction"],
                    "source_spans": ["fault diagnosis", "remaining useful life prediction"],
                },
                {
                    "label": "Domain", "role": "domain",
                    "terms": ["lithium-ion batteries"],
                    "source_spans": ["lithium-ion batteries"],
                },
                {
                    "label": "Conditions", "role": "context",
                    "terms": ["varying operating conditions"],
                    "source_spans": ["varying operating conditions"],
                },
            ],
            "needs_grounding": True,
            "uncertain_terms": [],
        }, model=model, elapsed_seconds=0.01)


class IncompleteTechnicalEngine(CompleteTechnicalEngine):
    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        result = super().generate(model, prompt, schema, timeout_seconds=timeout_seconds)
        result.value["groups"] = result.value["groups"][:2] + [{
            "label": "Prediction", "role": "outcome", "terms": ["prediction"],
            "source_spans": ["prediction"],
        }, {
            "label": "Battery outcome", "role": "outcome", "terms": ["lithium-ion batteries"],
            "source_spans": ["lithium-ion batteries"],
        }]
        return result


def test_ai_first_technical_question_preserves_four_roles_and_verified_acronyms():
    bundle = generate_query_bundle(
        TECHNICAL_QUESTION, profile=_profile(), engine=CompleteTechnicalEngine(),
        grounder=EmptyGrounder(), deadline_seconds=2,
    )
    groups = bundle.concepts["groups"]
    assert [group["role"] for group in groups] == ["technology", "outcome", "domain", "context"]
    assert bundle.concepts["literal_coverage"] == 1.0
    assert bundle.concepts["uncovered_spans"] == []
    assert bundle.concepts["generation_status"] == "full"
    assert "PINN" in bundle.google_scholar
    assert "RUL prediction" in bundle.google_scholar
    assert "lithium-ion batteries" in bundle.google_scholar
    assert "varying operating conditions" in bundle.google_scholar
    assert '"physics-informed neural network*"' in bundle.scopus
    assert '"lithium-ion batter*"' in bundle.scopus
    assert '"varying operating condition*"' in bundle.scopus


def test_incomplete_ai_draft_is_rebuilt_from_lossless_literal_spans():
    bundle = generate_query_bundle(
        TECHNICAL_QUESTION, profile=_profile(), engine=IncompleteTechnicalEngine(),
        grounder=EmptyGrounder(), deadline_seconds=2,
    )
    assert bundle.concepts["generation_status"] == "repaired"
    assert bundle.concepts["literal_coverage"] == 1.0
    assert bundle.concepts["uncovered_spans"] == []
    assert bundle.concepts["repaired_spans"] == [
        "lithium-ion batteries", "varying operating conditions",
    ]
    assert not any(group["terms"] == ["prediction"] for group in bundle.concepts["groups"])
    assert "lithium-ion batteries" in bundle.google_scholar
    assert "varying operating conditions" in bundle.google_scholar


def test_model_failure_keeps_all_nested_technical_spans_without_keyword_registry():
    bundle = generate_query_bundle(
        TECHNICAL_QUESTION, profile=_profile(), engine=FailingEngine(),
        grounder=EmptyGrounder(), deadline_seconds=2,
    )
    assert [group["role"] for group in bundle.concepts["groups"]] == [
        "technology", "outcome", "domain", "context",
    ]
    source_spans = [span for group in bundle.concepts["groups"] for span in group["source_spans"]]
    assert source_spans == [
        "physics-informed neural networks",
        "fault diagnosis",
        "remaining useful life prediction",
        "lithium-ion batteries",
        "varying operating conditions",
    ]
    assert '"fault diagnosi"' not in bundle.google_scholar
    assert bundle.concepts["literal_coverage"] == 1.0


def test_general_nested_relations_preserve_unrelated_domain_spans():
    cases = [
        (
            "How is telemedicine used for medication adherence in older adults under limited connectivity?",
            ["telemedicine", "medication adherence", "older adults", "limited connectivity"],
        ),
        (
            "How are community solar programs used for energy access among rural households within low-income regions?",
            ["community solar programs", "energy access", "rural households", "low-income regions"],
        ),
    ]
    for question, expected_spans in cases:
        bundle = generate_query_bundle(
            question, profile=_profile(), engine=FailingEngine(),
            grounder=EmptyGrounder(), deadline_seconds=2,
        )
        actual = [span for group in bundle.concepts["groups"] for span in group["source_spans"]]
        assert actual == expected_spans
        assert bundle.concepts["literal_coverage"] == 1.0


def test_first_stage_has_no_task_or_domain_keyword_registry_dependency():
    source = inspect.getsource(query_module)
    assert "BROAD_TASK_TERMS" not in source
    assert "DIRECT_TASK_VARIANTS" not in source
    assert "ONTOLOGY_REGISTRY_PACKS" not in source


PARAGRAPH_QUESTION = (
    "In systematic reviews of cyber-physical energy systems, how can federated graph neural "
    "netwroks and physics-informed transformers be used to detect cascading failures, estimate "
    "remaining useful life, and support real-time resilience optimization across smart grids, "
    "microgrids, and electric-vehicle charging infrastructure under adversarial communication "
    "attacks, missing sensor data, and privacy-preserving data-sharing constraints?"
)


class ParagraphEngine:
    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        return GenerationResult(value={
            "groups": [
                {
                    "label": "Methods", "role": "technology",
                    "terms": ["federated graph neural networks", "physics-informed transformers"],
                    "source_spans": [
                        "federated graph neural netwroks", "physics-informed transformers",
                    ],
                },
                {
                    "label": "Outcomes", "role": "outcome",
                    "terms": [
                        "cascading failures", "remaining useful life",
                        "real-time resilience optimization",
                    ],
                    "source_spans": [
                        "cascading failures", "remaining useful life",
                        "real-time resilience optimization",
                    ],
                },
                {
                    "label": "Infrastructure", "role": "domain",
                    "terms": [
                        "cyber-physical energy systems", "smart grids", "microgrids",
                        "electric-vehicle charging infrastructure",
                    ],
                    "source_spans": [
                        "cyber-physical energy systems", "smart grids", "microgrids",
                        "electric-vehicle charging infrastructure",
                    ],
                },
                {
                    "label": "Constraints", "role": "context",
                    "terms": [
                        "adversarial communication attacks", "missing sensor data",
                        "privacy-preserving data-sharing constraints",
                    ],
                    "source_spans": [
                        "adversarial communication attacks", "missing sensor data",
                        "privacy-preserving data-sharing constraints",
                    ],
                },
            ],
            "needs_grounding": True,
            "uncertain_terms": [],
        }, model=model, elapsed_seconds=0.01)


class IncompleteParagraphEngine(ParagraphEngine):
    def generate(self, model, prompt, schema, *, timeout_seconds=None):
        result = super().generate(model, prompt, schema, timeout_seconds=timeout_seconds)
        result.value["groups"] = [result.value["groups"][0], result.value["groups"][3]]
        return result


def test_paragraph_question_removes_scaffolding_and_preserves_all_four_groups():
    bundle = generate_query_bundle(
        PARAGRAPH_QUESTION, profile=_profile(), engine=ParagraphEngine(),
        grounder=EmptyGrounder(), deadline_seconds=2,
    )
    query = bundle.scopus
    assert [group["role"] for group in bundle.concepts["groups"]] == [
        "technology", "outcome", "domain", "context",
    ]
    for forbidden in ("in systematic reviews of", "how can", "be used to", "physics-informed tra\""):
        assert forbidden not in query.lower()
    for required in (
        "federated graph neural network*", "physics-informed transformer*", "cascading failure*",
        "remaining useful life", "real-time resilience optimization", "cyber-physical energy system*",
        "smart grid*", "microgrid*", "electric-vehicle charging infrastructure",
        "adversarial communication attack*", "missing sensor data",
        "privacy-preserving data-sharing constraint*",
    ):
        assert required in query
    assert bundle.concepts["literal_coverage"] == 1.0
    assert bundle.concepts["generation_status"] == "repaired"
    assert bundle.concepts["removed_scaffolding"].startswith("In systematic reviews of")
    assert bundle.concepts["corrections"] == [{
        "original": "federated graph neural netwroks",
        "corrected": "federated graph neural networks",
        "distance": 1,
        "group": "Methods",
    }]


def test_incomplete_paragraph_ai_is_rebuilt_without_partial_terms():
    bundle = generate_query_bundle(
        PARAGRAPH_QUESTION, profile=_profile(), engine=IncompleteParagraphEngine(),
        grounder=EmptyGrounder(), deadline_seconds=2,
    )
    assert bundle.concepts["literal_coverage"] == 1.0
    assert bundle.concepts["uncovered_spans"] == []
    assert set(bundle.concepts["repaired_spans"]) >= {
        "cascading failures", "remaining useful life", "real-time resilience optimization",
        "cyber-physical energy systems", "smart grids", "microgrids",
        "electric-vehicle charging infrastructure",
    }
    assert "physics-informed tra\"" not in bundle.scopus.lower()
    assert "electric-vehicle charging infrastructure" in bundle.scopus


def test_paragraph_model_failure_retains_scope_without_sentence_scaffolding():
    bundle = generate_query_bundle(
        PARAGRAPH_QUESTION, profile=_profile(), engine=FailingEngine(),
        grounder=EmptyGrounder(), deadline_seconds=2,
    )
    assert bundle.concepts["literal_coverage"] == 1.0
    assert "how can" not in bundle.google_scholar.lower()
    assert "systematic reviews" not in bundle.google_scholar.lower()
    assert "physics-informed transformers" in bundle.google_scholar
    assert "electric-vehicle charging infrastructure" in bundle.google_scholar


def test_review_preamble_variants_keep_only_topical_scope():
    cases = [
        ("In scoping reviews of urban mobility, how can digital twins be used to estimate demand across rail networks?", "urban mobility"),
        ("In literature reviews of clinical decision support, how can language models be used to detect medication errors across hospitals?", "clinical decision support"),
    ]
    for question, scope in cases:
        bundle = generate_query_bundle(
            question, profile=_profile(), engine=FailingEngine(),
            grounder=EmptyGrounder(), deadline_seconds=2,
        )
        assert scope in bundle.google_scholar
        assert "reviews of" not in bundle.google_scholar.lower()
        assert "how can" not in bundle.google_scholar.lower()


def test_api_bundle_shape_is_backward_compatible():
    bundle = GeneratedQueryBundle("q", "s", "w", "i", "p", {"groups": []})
    payload = bundle.to_api_response()
    assert set(payload) == {"status", "google_scholar", "scopus", "web_of_science", "ieee_xplore", "pubmed", "concepts"}

