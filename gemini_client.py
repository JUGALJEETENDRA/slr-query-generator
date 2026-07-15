from __future__ import annotations

import os
import time
from typing import Any

import requests
from local_ai.engine import LocalAIError


DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_API_MODEL", "gemini-3.5-flash")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_SUPPORTED_SCHEMA_FIELDS = {
    "$id", "$defs", "$ref", "$anchor", "type", "format", "title", "description",
    "enum", "items", "prefixItems", "minItems", "maxItems", "minimum", "maximum",
    "anyOf", "oneOf", "properties", "additionalProperties", "required", "propertyOrdering",
}


def _gemini_json_schema(value: Any) -> Any:
    """Keep the documented Gemini JSON-schema subset without changing field names."""
    if isinstance(value, list):
        return [_gemini_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if key not in _SUPPORTED_SCHEMA_FIELDS:
            continue
        if key in {"properties", "$defs"} and isinstance(item, dict):
            cleaned[key] = {name: _gemini_json_schema(schema) for name, schema in item.items()}
        else:
            cleaned[key] = _gemini_json_schema(item)
    return cleaned


class GeminiAPIError(LocalAIError):
    """A safe Gemini error which never contains the submitted API key."""


class GeminiAPIClient:
    """Small job-scoped Gemini REST client with structured output and retries."""

    def __init__(self, api_key: str | None = None, timeout_seconds: float = 90.0):
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        if not self.api_key:
            raise GeminiAPIError("Enter a Gemini API key before starting Gemini API screening.")
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, model: str = DEFAULT_GEMINI_MODEL, schema=None) -> str:
        selected_model = model if str(model).startswith("gemini-") else DEFAULT_GEMINI_MODEL
        generation_config: dict[str, Any] = {
            "temperature": 0,
            "candidateCount": 1,
            "maxOutputTokens": 4096,
        }
        if schema is not None:
            json_schema = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema
            generation_config.update({
                "responseMimeType": "application/json",
                "responseJsonSchema": _gemini_json_schema(json_schema),
            })
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

        for attempt in range(3):
            try:
                response = requests.post(
                    GEMINI_API_URL.format(model=selected_model),
                    headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                if attempt < 2:
                    time.sleep(1.5 * (2 ** attempt))
                    continue
                raise GeminiAPIError("Could not connect to Gemini. Check your internet connection and try again.") from exc

            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = min(float(retry_after), 8.0) if retry_after else 1.5 * (2 ** attempt)
                except ValueError:
                    delay = 1.5 * (2 ** attempt)
                time.sleep(delay)
                continue
            if response.status_code == 400:
                raise GeminiAPIError("Gemini rejected the request. Check that the API key is valid and supports the selected model.")
            if response.status_code in {401, 403}:
                raise GeminiAPIError("Gemini rejected the API key. Create or verify the key in Google AI Studio.")
            if response.status_code == 429:
                raise GeminiAPIError("Gemini API quota or rate limit was reached. Wait briefly or check the key's quota in AI Studio.")
            if not response.ok:
                raise GeminiAPIError(f"Gemini API failed with HTTP {response.status_code}. Try again shortly.")

            data = response.json()
            try:
                parts = data["candidates"][0]["content"]["parts"]
                text = "".join(str(part.get("text") or "") for part in parts).strip()
            except (KeyError, IndexError, TypeError) as exc:
                block = (data.get("promptFeedback") or {}).get("blockReason")
                detail = f" ({block})" if block else ""
                raise GeminiAPIError(f"Gemini returned no usable screening response{detail}.") from exc
            if not text:
                raise GeminiAPIError("Gemini returned an empty screening response.")
            return text

        raise GeminiAPIError("Gemini API did not respond after automatic retries.")

    def close(self) -> None:
        return None


def ask_gemini(prompt, model=DEFAULT_GEMINI_MODEL, api_key=None):
    client = GeminiAPIClient(api_key=api_key)
    try:
        return client.generate(prompt, model=model)
    finally:
        client.close()
