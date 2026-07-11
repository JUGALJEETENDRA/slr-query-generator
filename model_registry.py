from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import config
from runtime_config import get_model_judge_config, parse_bool

MODEL_PROFILES = {
    "light": {
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "nli": "typeform/distilbert-base-uncased-mnli",
        "zero_shot": "typeform/distilbert-base-uncased-mnli",
    },
    "balanced": {
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "nli": "typeform/distilbert-base-uncased-mnli",
        "zero_shot": "typeform/distilbert-base-uncased-mnli",
    },
    "full": {
        "reranker": "BAAI/bge-reranker-v2-m3",
        "nli": "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
        "zero_shot": "facebook/bart-large-mnli",
    },
}

MODEL_RUNTIME_STATUS: dict[str, dict[str, str | bool]] = {}


def bool_config(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        value = getattr(config, name, default)
    return parse_bool(value, default)


def model_judge_mode(default: str | None = None) -> str:
    return get_model_judge_config(default)["model_judge_mode"]


def model_judges_enabled(mode: str | None = None) -> bool:
    return get_model_judge_config(mode)["enable_model_judges"]


def cache_key(*parts) -> str:
    text = "\n".join(str(part or "") for part in parts)
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


@lru_cache(maxsize=8)
def get_cross_encoder(model_name: str):
    cfg = get_model_judge_config()
    if not cfg["enable_hf_model_loading"]:
        _set_runtime_status(model_name, "fallback_hf_loading_disabled", False, "HF model loading is disabled.")
        return None
    try:
        local_only = not cfg["enable_hf_model_download"]
        if local_only and not hf_model_exists_locally(model_name):
            _set_runtime_status(model_name, "fallback_no_local_model", False, "Model is not present in local HuggingFace cache.")
            return None
        from sentence_transformers import CrossEncoder

        if local_only:
            model = CrossEncoder(_local_snapshot_path(model_name))
        else:
            model = CrossEncoder(model_name, local_files_only=False)
        _set_runtime_status(model_name, "hf_model", True, "")
        return model
    except Exception as exc:
        _set_runtime_status(model_name, "fallback_model_load_error", False, f"{type(exc).__name__}: {exc}")
        return None


@lru_cache(maxsize=8)
def get_transformers_pipeline(task: str, model_name: str):
    cfg = get_model_judge_config()
    if not cfg["enable_hf_model_loading"]:
        _set_runtime_status(model_name, "fallback_hf_loading_disabled", False, "HF model loading is disabled.")
        return None
    try:
        local_only = not cfg["enable_hf_model_download"]
        if local_only and not hf_model_exists_locally(model_name):
            _set_runtime_status(model_name, "fallback_no_local_model", False, "Model is not present in local HuggingFace cache.")
            return None
        from transformers import pipeline
        model = pipeline(
            task,
            model=model_name,
            tokenizer=model_name,
            local_files_only=local_only,
        )
        _set_runtime_status(model_name, "hf_model", True, "")
        return model
    except Exception as exc:
        _set_runtime_status(model_name, "fallback_model_load_error", False, f"{type(exc).__name__}: {exc}")
        return None


def configured_model_names() -> dict[str, str]:
    cfg = get_model_judge_config()
    profile_names = MODEL_PROFILES.get(cfg["model_judge_profile"], MODEL_PROFILES["light"])
    return {
        "reranker": os.getenv("RERANKER_MODEL_NAME", profile_names["reranker"]),
        "nli": os.getenv("NLI_MODEL_NAME", profile_names["nli"]),
        "zero_shot": os.getenv("ZERO_SHOT_MODEL_NAME", profile_names["zero_shot"]),
    }


def hf_model_exists_locally(model_name: str) -> bool:
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        return any(repo.repo_id == model_name for repo in cache_info.repos)
    except Exception:
        return False


def _local_snapshot_path(model_name: str) -> str:
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        return model_name


def model_preflight_rows() -> list[dict[str, str | bool]]:
    cfg = get_model_judge_config()
    rows = []
    for role, name in configured_model_names().items():
        exists = hf_model_exists_locally(name)
        if not cfg["enable_hf_model_loading"]:
            mode = "fallback"
            needs_download = False
            reason = "HF model loading disabled"
        elif exists:
            mode = "real_model"
            needs_download = False
            reason = "local cache found"
        elif cfg["enable_hf_model_download"]:
            mode = "real_model"
            needs_download = True
            reason = "download allowed"
        else:
            mode = "fallback"
            needs_download = True
            reason = "missing locally and download disabled"
        rows.append({
            "role": role,
            "model_name": name,
            "exists_locally": exists,
            "estimated_mode": mode,
            "internet_or_download_needed": needs_download,
            "reason": reason,
        })
    return rows


def get_runtime_status(model_name: str | None = None) -> dict[str, str | bool]:
    if model_name:
        return dict(MODEL_RUNTIME_STATUS.get(model_name, {}))
    if any(status.get("runtime_source") == "hf_model" for status in MODEL_RUNTIME_STATUS.values()):
        return {"runtime_source": "hf_model", "real_models_loaded": True, "fallback_reason": ""}
    for status in MODEL_RUNTIME_STATUS.values():
        if status.get("runtime_source"):
            return dict(status)
    return {"runtime_source": "fallback_not_attempted", "real_models_loaded": False, "fallback_reason": "No model load attempted."}


def reset_runtime_status() -> None:
    MODEL_RUNTIME_STATUS.clear()


def _set_runtime_status(model_name: str, source: str, loaded: bool, reason: str) -> None:
    MODEL_RUNTIME_STATUS[model_name] = {
        "runtime_source": source,
        "real_models_loaded": loaded,
        "fallback_reason": reason,
    }
