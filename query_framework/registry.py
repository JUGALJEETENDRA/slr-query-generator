from __future__ import annotations

from typing import Any

from .models import QueryGenerationStrategy, StrategyMetadata
from .strategies import DirectAIStrategy, LitSyncWorkflowStrategy


class StrategyRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, QueryGenerationStrategy] = {}
        self._aliases: dict[str, str] = {}

    def register(self, strategy: QueryGenerationStrategy) -> None:
        metadata = strategy.metadata
        self._strategies[metadata.id] = strategy
        self._aliases[metadata.id] = metadata.id
        self._aliases[metadata.label] = metadata.id
        for alias in metadata.aliases:
            self._aliases[alias] = metadata.id

    def resolve_id(self, strategy_id: str | None) -> str:
        if not strategy_id:
            return "direct_ai"
        return self._aliases.get(strategy_id, "direct_ai")

    def get(self, strategy_id: str | None) -> QueryGenerationStrategy:
        resolved = self.resolve_id(strategy_id)
        return self._strategies[resolved]

    def list_metadata(self) -> list[StrategyMetadata]:
        return [strategy.metadata for strategy in self._strategies.values()]

    def generate(self, question: str, strategy_id: str | None = None):
        return self.get(strategy_id).generate(question)


def create_default_strategy_registry(client: Any, model: str) -> StrategyRegistry:
    registry = StrategyRegistry()
    registry.register(LitSyncWorkflowStrategy(client=client, model=model))
    registry.register(DirectAIStrategy())
    return registry
