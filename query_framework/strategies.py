from __future__ import annotations

import re
from typing import Any

from direct_ai_generator import generate_query as generate_direct_query

from acronym_expander import expand_acronym_layer
from classifier import classify_extracted_context
from compiler import compile_boolean_query
from comparator_registry import expand_comparator_registry
from extractor import extract_5_facets
from generator import expand_base_synonyms
from ontology_expander import expand_ontology_layer
from registries import inject_implicit_academic_layers
from schema import SLRQueryContext
from validator import run_validation_sieve

from .models import QueryGenerationResult, StrategyMetadata
from .telemetry import TelemetryCollector


def format_platform_queries(
    question: str,
    strategy: StrategyMetadata,
    base_query: str,
    ieee_query: str | None = None,
    concepts: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None,
) -> QueryGenerationResult:
    ieee_value = ieee_query if ieee_query is not None else base_query
    return QueryGenerationResult(
        question=question,
        strategy_id=strategy.id,
        strategy_label=strategy.label,
        google_scholar=base_query,
        scopus=f"TITLE-ABS-KEY({base_query})",
        web_of_science=f"TS=({base_query})",
        ieee_xplore=ieee_value,
        pubmed=re.sub(r'"([^"]+)"', r'"\1"[tiab]', base_query),
        concepts=concepts or {},
        telemetry=telemetry or {},
    )


def compress_schema_for_ieee(context: SLRQueryContext) -> SLRQueryContext:
    import copy

    compressed = copy.deepcopy(context)
    merged_tech = list(context.technology[:2]) + list(context.comparison[:1])
    compressed.technology = [t.replace("*", "") for t in merged_tech if t]
    compressed.domain = [t.replace("*", "") for t in context.domain[:2]]
    compressed.outcomes = [t.replace("*", "") for t in context.outcomes[:2]]
    compressed.comparison = []
    compressed.context = []
    return compressed


class LitSyncWorkflowStrategy:
    metadata = StrategyMetadata(
        id="litsync_workflow",
        label="LitSync Workflow",
        description="Current modular LitSync extraction, expansion, validation, and compiler pipeline.",
        aliases=("litsync", "LitSync Workflow"),
    )

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def generate(self, question: str) -> QueryGenerationResult:
        telemetry = TelemetryCollector()

        raw = extract_5_facets(self.client, self.model, question)
        telemetry.record_stage("extract", raw)

        s1 = SLRQueryContext(
            technology=raw.primary_paradigm,
            comparison=raw.comparator_baseline,
            domain=raw.domain_context,
            context=[],
            outcomes=raw.outcome_variables,
        )
        telemetry.record_stage("context_from_extraction", s1)

        s2 = expand_base_synonyms(self.client, self.model, s1)
        telemetry.record_stage("base_synonym_expansion", s2)

        s3 = expand_acronym_layer(s2)
        telemetry.record_stage("acronym_expansion", s3)

        primary_domain = classify_extracted_context(s3)
        telemetry.record_stage("classification", primary_domain)

        s3_hydrated = inject_implicit_academic_layers(s3, primary_domain)
        telemetry.record_stage("registry_injection", s3_hydrated)

        s4 = expand_ontology_layer(s3_hydrated, primary_domain)
        telemetry.record_stage("ontology_expansion", s4)

        s4_compared = expand_comparator_registry(s4)
        telemetry.record_stage("comparator_expansion", s4_compared)

        s5 = run_validation_sieve(s4_compared)
        telemetry.record_stage("validation", s5)

        base_query = compile_boolean_query(s5).replace("\n", " ")
        ieee_safe = compress_schema_for_ieee(s5)
        ieee_query = compile_boolean_query(ieee_safe).replace("\n", " ")
        telemetry.record_stage("compile", {"google_scholar": base_query, "ieee_xplore": ieee_query})

        return format_platform_queries(
            question=question,
            strategy=self.metadata,
            base_query=base_query,
            ieee_query=ieee_query,
            telemetry=telemetry.to_dict(),
        )


class DirectAIStrategy:
    metadata = StrategyMetadata(
        id="direct_ai",
        label="Direct AI",
        description="Production one-shot Boolean query generation pipeline.",
        aliases=("direct", "Direct AI"),
        experimental=False,
    )

    def generate(self, question: str) -> QueryGenerationResult:
        base_query = generate_direct_query(question).strip()
        telemetry = TelemetryCollector()
        telemetry.record_stage("direct_llm_baseline", {"query": base_query})
        return format_platform_queries(
            question=question,
            strategy=self.metadata,
            base_query=base_query,
            telemetry=telemetry.to_dict(),
        )
