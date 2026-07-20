from __future__ import annotations

import re
from typing import Any

from direct_ai_generator import generate_query_bundle

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


class DirectAIStrategy:
    metadata = StrategyMetadata(
        id="direct_ai",
        label="Direct AI",
        description="Production one-shot Boolean query generation pipeline.",
        aliases=("direct", "Direct AI"),
        experimental=False,
    )

    def generate(self, question: str) -> QueryGenerationResult:
        bundle = generate_query_bundle(question)
        telemetry = TelemetryCollector()
        telemetry.record_stage("domain_neutral_generation", bundle.concepts)
        return QueryGenerationResult(
            question=question,
            strategy_id=self.metadata.id,
            strategy_label=self.metadata.label,
            google_scholar=bundle.google_scholar,
            scopus=bundle.scopus,
            web_of_science=bundle.web_of_science,
            ieee_xplore=bundle.ieee_xplore,
            pubmed=bundle.pubmed,
            concepts=bundle.concepts,
            telemetry=telemetry.to_dict(),
        )
