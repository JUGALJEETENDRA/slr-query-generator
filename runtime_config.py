from __future__ import annotations

import os
from typing import Any

import config


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
MODEL_JUDGE_MODES = {"off", "fast", "balanced", "full"}
MODEL_JUDGE_PROFILES = {"light", "balanced", "full"}
PIPELINE_MODES = {"current", "two_pass_fast"}


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return bool(default)


def _env_or_config(name: str, default: Any) -> Any:
    value = os.getenv(name)
    if value is not None:
        return value
    return getattr(config, name, default)


def get_model_judge_config(mode: str | None = None) -> dict[str, Any]:
    raw_env = {
        "MODEL_JUDGE_MODE": os.getenv("MODEL_JUDGE_MODE"),
        "ENABLE_MODEL_JUDGES": os.getenv("ENABLE_MODEL_JUDGES"),
        "ENABLE_HF_MODEL_LOADING": os.getenv("ENABLE_HF_MODEL_LOADING"),
        "ENABLE_HF_MODEL_DOWNLOAD": os.getenv("ENABLE_HF_MODEL_DOWNLOAD"),
        "MODEL_JUDGE_PROFILE": os.getenv("MODEL_JUDGE_PROFILE"),
        "ENABLE_RERANKER_JUDGE": os.getenv("ENABLE_RERANKER_JUDGE"),
        "ENABLE_NLI_JUDGE": os.getenv("ENABLE_NLI_JUDGE"),
        "ENABLE_ZERO_SHOT_JUDGE": os.getenv("ENABLE_ZERO_SHOT_JUDGE"),
        "ENABLE_LLM_JUDGE": os.getenv("ENABLE_LLM_JUDGE"),
        "MODEL_JUDGE_TIMEOUT_SECONDS": os.getenv("MODEL_JUDGE_TIMEOUT_SECONDS"),
        "MAX_LLM_DIRECTIONAL_ROWS": os.getenv("MAX_LLM_DIRECTIONAL_ROWS"),
        "SCREENING_PIPELINE_MODE": os.getenv("SCREENING_PIPELINE_MODE"),
        "ENABLE_BATCH_LLM_JUDGE": os.getenv("ENABLE_BATCH_LLM_JUDGE"),
        "ENABLE_AGGRESSIVE_LLM_GATING": os.getenv("ENABLE_AGGRESSIVE_LLM_GATING"),
        "ENABLE_SEMANTIC_FRAME_CACHE": os.getenv("ENABLE_SEMANTIC_FRAME_CACHE"),
        "ENABLE_CURRENT_MODE_CACHE": os.getenv("ENABLE_CURRENT_MODE_CACHE"),
        "LLM_BATCH_SIZE": os.getenv("LLM_BATCH_SIZE"),
        "ENABLE_PARALLEL_SCREENING": os.getenv("ENABLE_PARALLEL_SCREENING"),
        "SCREENING_WORKERS": os.getenv("SCREENING_WORKERS"),
        "OLLAMA_MAX_CONCURRENT": os.getenv("OLLAMA_MAX_CONCURRENT"),
    }
    selected_mode = str(
        mode if mode is not None else _env_or_config("MODEL_JUDGE_MODE", "balanced")
    ).strip().lower()
    if selected_mode not in MODEL_JUDGE_MODES:
        selected_mode = "off"
    selected_profile = str(_env_or_config("MODEL_JUDGE_PROFILE", "light")).strip().lower()
    if selected_profile not in MODEL_JUDGE_PROFILES:
        selected_profile = "light"

    enabled = parse_bool(_env_or_config("ENABLE_MODEL_JUDGES", True), True)
    if selected_mode == "off":
        enabled = False

    nli_default = selected_profile != "light"
    zero_shot_default = selected_profile != "light"

    pipeline_mode = str(_env_or_config("SCREENING_PIPELINE_MODE", "current")).strip().lower()
    if pipeline_mode not in PIPELINE_MODES:
        pipeline_mode = "current"
    current_mode_cache = parse_bool(_env_or_config("ENABLE_CURRENT_MODE_CACHE", False), False)
    semantic_cache = parse_bool(_env_or_config("ENABLE_SEMANTIC_FRAME_CACHE", False), False)

    return {
        "enable_model_judges": enabled,
        "model_judge_mode": selected_mode,
        "model_judge_profile": selected_profile,
        "enable_hf_model_loading": parse_bool(
            _env_or_config("ENABLE_HF_MODEL_LOADING", False),
            False,
        ),
        "enable_hf_model_download": parse_bool(
            _env_or_config("ENABLE_HF_MODEL_DOWNLOAD", False),
            False,
        ),
        "enable_reranker_judge": parse_bool(
            _env_or_config("ENABLE_RERANKER_JUDGE", True),
            True,
        ),
        "enable_nli_judge": parse_bool(
            os.getenv("ENABLE_NLI_JUDGE") if os.getenv("ENABLE_NLI_JUDGE") is not None else nli_default,
            nli_default,
        ),
        "enable_zero_shot_judge": parse_bool(
            os.getenv("ENABLE_ZERO_SHOT_JUDGE") if os.getenv("ENABLE_ZERO_SHOT_JUDGE") is not None else zero_shot_default,
            zero_shot_default,
        ),
        "enable_llm_judge": parse_bool(
            _env_or_config("ENABLE_LLM_JUDGE", False),
            False,
        ),
        "model_judge_timeout_seconds": float(_env_or_config("MODEL_JUDGE_TIMEOUT_SECONDS", 8.0) or 8.0),
        "max_llm_directional_rows": int(float(_env_or_config("MAX_LLM_DIRECTIONAL_ROWS", 100) or 100)),
        "screening_pipeline_mode": pipeline_mode,
        "enable_batch_llm_judge": pipeline_mode == "two_pass_fast" and parse_bool(_env_or_config("ENABLE_BATCH_LLM_JUDGE", False), False),
        "enable_aggressive_llm_gating": pipeline_mode == "two_pass_fast" and parse_bool(_env_or_config("ENABLE_AGGRESSIVE_LLM_GATING", False), False),
        "enable_semantic_frame_cache": semantic_cache if pipeline_mode == "two_pass_fast" else current_mode_cache,
        "enable_current_mode_cache": current_mode_cache,
        "llm_batch_size": max(1, int(float(_env_or_config("LLM_BATCH_SIZE", 5) or 5))),
        "llm_batch_max_chars": max(1000, int(float(_env_or_config("LLM_BATCH_MAX_CHARS", 12000) or 12000))),
        "llm_batch_timeout_seconds": float(_env_or_config("LLM_BATCH_TIMEOUT_SECONDS", 30.0) or 30.0),
        "enable_parallel_screening": parse_bool(_env_or_config("ENABLE_PARALLEL_SCREENING", False), False),
        "screening_workers": max(1, int(float(_env_or_config("SCREENING_WORKERS", 1) or 1))),
        "ollama_max_concurrent": max(1, int(float(_env_or_config("OLLAMA_MAX_CONCURRENT", 1) or 1))),
        "source": "env",
        "raw_env": raw_env,
    }


def model_config_csv_fields() -> dict[str, Any]:
    cfg = get_model_judge_config()
    return {
        "model_config_enable_model_judges": cfg["enable_model_judges"],
        "model_config_model_judge_mode": cfg["model_judge_mode"],
        "model_config_model_judge_profile": cfg["model_judge_profile"],
        "model_config_enable_hf_model_loading": cfg["enable_hf_model_loading"],
        "model_config_enable_hf_model_download": cfg["enable_hf_model_download"],
        "model_config_enable_reranker_judge": cfg["enable_reranker_judge"],
        "model_config_enable_nli_judge": cfg["enable_nli_judge"],
        "model_config_enable_zero_shot_judge": cfg["enable_zero_shot_judge"],
        "model_config_enable_llm_judge": cfg["enable_llm_judge"],
        "model_config_screening_pipeline_mode": cfg["screening_pipeline_mode"],
        "model_config_enable_batch_llm_judge": cfg["enable_batch_llm_judge"],
        "model_config_enable_aggressive_llm_gating": cfg["enable_aggressive_llm_gating"],
        "model_config_enable_semantic_frame_cache": cfg["enable_semantic_frame_cache"],
        "model_config_enable_current_mode_cache": cfg["enable_current_mode_cache"],
        "model_config_llm_batch_size": cfg["llm_batch_size"],
        "model_config_enable_parallel_screening": cfg["enable_parallel_screening"],
        "model_config_screening_workers": cfg["screening_workers"],
        "model_config_source": cfg["source"],
    }


def print_model_judge_config() -> None:
    cfg = get_model_judge_config()
    print("[MODEL JUDGE CONFIG]")
    print(f"ENABLE_MODEL_JUDGES={str(cfg['enable_model_judges']).lower()}")
    print(f"MODEL_JUDGE_MODE={cfg['model_judge_mode']}")
    print(f"MODEL_JUDGE_PROFILE={cfg['model_judge_profile']}")
    print(f"ENABLE_HF_MODEL_LOADING={str(cfg['enable_hf_model_loading']).lower()}")
    print(f"ENABLE_HF_MODEL_DOWNLOAD={str(cfg['enable_hf_model_download']).lower()}")
    print(f"ENABLE_RERANKER_JUDGE={str(cfg['enable_reranker_judge']).lower()}")
    print(f"ENABLE_NLI_JUDGE={str(cfg['enable_nli_judge']).lower()}")
    print(f"ENABLE_ZERO_SHOT_JUDGE={str(cfg['enable_zero_shot_judge']).lower()}")
    print(f"ENABLE_LLM_JUDGE={str(cfg['enable_llm_judge']).lower()}")
    print(f"SCREENING_PIPELINE_MODE={cfg['screening_pipeline_mode']}")
    print(f"ENABLE_BATCH_LLM_JUDGE={str(cfg['enable_batch_llm_judge']).lower()}")
    print(f"ENABLE_AGGRESSIVE_LLM_GATING={str(cfg['enable_aggressive_llm_gating']).lower()}")
    print(f"ENABLE_SEMANTIC_FRAME_CACHE={str(cfg['enable_semantic_frame_cache']).lower()}")
    print(f"ENABLE_CURRENT_MODE_CACHE={str(cfg['enable_current_mode_cache']).lower()}")
    print(f"LLM_BATCH_SIZE={cfg['llm_batch_size']}")
    print(f"ENABLE_PARALLEL_SCREENING={str(cfg['enable_parallel_screening']).lower()}")
    print(f"SCREENING_WORKERS={cfg['screening_workers']}")
