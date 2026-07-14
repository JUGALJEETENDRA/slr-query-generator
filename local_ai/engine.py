from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import requests
from pydantic import BaseModel

from .hardware import RuntimeProfile, inspect_hardware, save_calibration


class LocalAIError(RuntimeError):
    pass


class LocalAIMemoryError(LocalAIError):
    pass


class LocalAIOutputError(LocalAIError):
    """The model responded, but its structured output could not be parsed."""

    pass


@dataclass
class GenerationResult:
    value: dict[str, Any]
    model: str
    elapsed_seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    model_duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0


class OllamaStructuredEngine:
    _schema_grammar_support: dict[str, bool] = {}

    def __init__(self, profile: RuntimeProfile, base_url: str | None = None):
        self.profile = profile
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def generate(self, model: str, prompt: str, schema: type[BaseModel]) -> GenerationResult:
        started = time.perf_counter()
        default_output_tokens = {
            "TriageResult": 220,
            "TriageBatch": 512,
            "AssessmentBatch": 720,
            "CriticBatch": 720,
            "ReviewProtocol": 900,
            "PaperAssessment": 650,
        }.get(schema.__name__, 700)
        output_tokens_setting = int(
            os.getenv(f"LOCAL_AI_MAX_OUTPUT_TOKENS_{schema.__name__.upper()}", default_output_tokens)
        )
        formats: list[Any] = (
            [schema.model_json_schema(), "json"]
            if self._schema_grammar_support.get(self.base_url, True)
            else ["json"]
        )
        response = None
        payload = None
        try:
            for output_format in formats:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "format": output_format,
                        "think": False,
                        "keep_alive": self.profile.keep_alive,
                        "options": {
                            "temperature": 0.0,
                            "num_ctx": self.profile.num_ctx,
                            "num_predict": int(
                                os.getenv("LOCAL_AI_MAX_OUTPUT_TOKENS", output_tokens_setting)
                            ),
                            "seed": 17,
                        },
                    },
                    timeout=float(os.getenv("LOCAL_AI_TIMEOUT_SECONDS", "120")),
                )
                if response.status_code < 400:
                    if output_format != "json":
                        self._schema_grammar_support[self.base_url] = True
                    break
                body = response.text.lower()
                grammar_failed = (
                    output_format != "json"
                    and response.status_code == 400
                    and "failed to parse grammar" in body
                )
                if grammar_failed:
                    self._schema_grammar_support[self.base_url] = False
                    continue
                response.raise_for_status()
            if response is None:
                raise LocalAIError("Ollama did not return a response.")
            response.raise_for_status()
            payload = response.json()
            value = json.loads(payload.get("response", ""))
        except requests.HTTPError as exc:
            raw_body = exc.response.text if exc.response is not None else ""
            body = raw_body.lower()
            if any(term in body for term in ("out of memory", "memory", "cuda")):
                raise LocalAIMemoryError(body[:500]) from exc
            detail = raw_body[:500].strip()
            raise LocalAIError(f"{exc}: {detail}" if detail else str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise LocalAIOutputError(str(exc)) from exc
        except (requests.RequestException, ValueError) as exc:
            raise LocalAIError(str(exc)) from exc
        elapsed = max(0.0001, time.perf_counter() - started)
        output_tokens = int(payload.get("eval_count") or 0)
        model_duration = float(payload.get("eval_duration") or 0) / 1_000_000_000
        total_duration = float(payload.get("total_duration") or 0) / 1_000_000_000
        return GenerationResult(
            value=value,
            model=model,
            elapsed_seconds=round(elapsed, 4),
            prompt_tokens=int(payload.get("prompt_eval_count") or 0),
            output_tokens=output_tokens,
            tokens_per_second=round(
                output_tokens / (model_duration or elapsed), 3
            ) if output_tokens else 0.0,
            model_duration_seconds=round(model_duration, 4),
            total_duration_seconds=round(total_duration, 4),
        )

    def unload(self, model: str) -> None:
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
                timeout=15,
            )
        except requests.RequestException:
            return

    def calibrate(self) -> dict[str, Any]:
        if self.profile.calibration:
            return dict(self.profile.calibration)
        class Calibration(BaseModel):
            status: str
        before = inspect_hardware()
        model_bytes = before.installed_models.get(self.profile.fast_model, 0)
        reserve_gb = before.total_ram_gb * self.profile.memory_reserve_ratio
        usable_after_model = before.available_ram_gb - (model_bytes / (1024 ** 3)) - reserve_gb
        max_candidate = 1
        if self.profile.resource_profile != "eco" and usable_after_model >= 1.0:
            max_candidate = 2
        if self.profile.resource_profile == "maximum" and usable_after_model >= 2.0:
            max_candidate = min(4, self.profile.concurrency)
        candidates = sorted(set([1, max_candidate]))
        measurements = []
        try:
            for concurrency in candidates:
                started = time.perf_counter()
                results = []
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [
                        pool.submit(
                            self.generate,
                            self.profile.fast_model,
                            'Return JSON with exactly {"status":"ready"}.',
                            Calibration,
                        )
                        for _ in range(concurrency)
                    ]
                    for future in as_completed(futures):
                        results.append(future.result())
                elapsed = max(0.0001, time.perf_counter() - started)
                valid = sum(result.value.get("status") == "ready" for result in results)
                measurements.append({
                    "concurrency": concurrency,
                    "valid": valid,
                    "elapsed_seconds": round(elapsed, 4),
                    "aggregate_tokens_per_second": round(
                        sum(result.output_tokens for result in results) / elapsed, 3
                    ),
                })
            viable = [item for item in measurements if item["valid"] == item["concurrency"]]
            best = max(viable, key=lambda item: item["aggregate_tokens_per_second"])
            after = inspect_hardware()
            calibration = {
                "load_success": True,
                "tokens_per_second": best["aggregate_tokens_per_second"],
                "calibration_seconds": round(sum(item["elapsed_seconds"] for item in measurements), 4),
                "recommended_concurrency": best["concurrency"],
                "candidate_measurements": measurements,
                "estimated_peak_ram_delta_gb": round(
                    max(0.0, before.available_ram_gb - after.available_ram_gb), 3
                ),
            }
        except (LocalAIError, ValueError) as exc:
            calibration = {
                "load_success": False,
                "error": str(exc),
                "recommended_concurrency": 1,
            }
        save_calibration(self.profile, calibration)
        return calibration
