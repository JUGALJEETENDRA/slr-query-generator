from __future__ import annotations

import copy
import warnings
from typing import Any

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
from query_framework.models import QueryGenerationResult, StrategyMetadata
from query_framework.strategies import format_platform_queries
from query_framework.telemetry import TelemetryCollector


def _compress_for_ieee(context: SLRQueryContext) -> SLRQueryContext:
    compressed = copy.deepcopy(context)
    merged = list(context.technology[:2]) + list(context.comparison[:1])
    compressed.technology = [term.replace("*", "") for term in merged if term]
    compressed.domain = [term.replace("*", "") for term in context.domain[:2]]
    compressed.outcomes = [term.replace("*", "") for term in context.outcomes[:2]]
    compressed.comparison = []
    compressed.context = []
    return compressed


class HistoricalOntologyStrategy:
    """Non-production baseline containing LitSync's former hand-written registries."""

    metadata = StrategyMetadata(
        id="historical_ontology",
        label="Historical Ontology Baseline",
        description="Experimental reproduction of the former hand-written ontology pipeline.",
        aliases=("litsync_workflow", "litsync", "LitSync Workflow"),
        experimental=True,
    )

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def generate(self, question: str) -> QueryGenerationResult:
        warnings.warn(
            "litsync_workflow is a historical ontology baseline, not a production strategy",
            DeprecationWarning,
            stacklevel=2,
        )
        telemetry = TelemetryCollector()
        raw = extract_5_facets(self.client, self.model, question)
        context = SLRQueryContext(
            technology=raw.primary_paradigm,
            comparison=raw.comparator_baseline,
            domain=raw.domain_context,
            context=[],
            outcomes=raw.outcome_variables,
        )
        context = expand_base_synonyms(self.client, self.model, context)
        context = expand_acronym_layer(context)
        primary_domain = classify_extracted_context(context)
        context = inject_implicit_academic_layers(context, primary_domain)
        context = expand_ontology_layer(context, primary_domain)
        context = expand_comparator_registry(context)
        context = run_validation_sieve(context)
        base_query = compile_boolean_query(context).replace("\n", " ")
        ieee_query = compile_boolean_query(_compress_for_ieee(context)).replace("\n", " ")
        telemetry.record_stage("compile", {"google_scholar": base_query, "ieee_xplore": ieee_query})
        return format_platform_queries(
            question=question,
            strategy=self.metadata,
            base_query=base_query,
            ieee_query=ieee_query,
            telemetry=telemetry.to_dict(),
        )
