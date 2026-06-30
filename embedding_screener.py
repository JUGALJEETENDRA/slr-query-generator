"""Fast local title screening with embeddings served by Ollama."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import requests


class EmbeddingError(RuntimeError):
    """Raised when Ollama cannot produce usable embeddings."""


class EmbeddingScreener:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed several strings, preferring Ollama's batch endpoint."""
        clean_texts = [str(text or "").strip() for text in texts]
        if not clean_texts:
            return []

        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": clean_texts},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise EmbeddingError(
                f"Could not connect to Ollama at {self.base_url}. Ensure Ollama is running."
            ) from exc

        if response.ok:
            try:
                embeddings = response.json().get("embeddings")
            except ValueError as exc:
                raise EmbeddingError("Ollama returned an invalid embedding response.") from exc
            if embeddings is None:
                raise EmbeddingError("Ollama returned an incomplete embedding response.")
            return self._validate(embeddings, len(clean_texts))

        # A missing model and a missing endpoint are both HTTP 404 responses.
        # Only fall back when the response does not specifically identify the model.
        try:
            current_error = str(response.json().get("error", ""))
        except ValueError:
            current_error = ""
        if response.status_code != 404 or "model" in current_error.lower():
            detail = current_error or f"HTTP {response.status_code}"
            raise EmbeddingError(
                f"Ollama could not load embedding model '{self.model}': {detail}. "
                f"Install it with: ollama pull {self.model}"
            )

        # Older Ollama versions expose only /api/embeddings.
        embeddings = []
        for text in clean_texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
            except (requests.RequestException, KeyError, ValueError) as exc:
                raise EmbeddingError(
                    "Could not generate title embeddings. Ensure Ollama is running "
                    f"and install the model with: ollama pull {self.model}"
                ) from exc
        return self._validate(embeddings, len(clean_texts))

    @staticmethod
    def _validate(embeddings: Iterable[Sequence[float]], expected: int) -> List[List[float]]:
        values = [list(vector) for vector in embeddings]
        if len(values) != expected or any(not vector for vector in values):
            raise EmbeddingError("Ollama returned an incomplete embedding response.")
        dimensions = {len(vector) for vector in values}
        if len(dimensions) != 1:
            raise EmbeddingError("Ollama returned embeddings with inconsistent dimensions.")
        return values

    @staticmethod
    def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Embedding dimensions do not match.")
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def score_titles(self, research_topic: str, titles: Sequence[str]) -> List[float]:
        """Embed the topic once and return one cosine score per title."""
        vectors = self.embed([research_topic, *titles])
        topic_vector = vectors[0]
        return [self.cosine_similarity(topic_vector, vector) for vector in vectors[1:]]
