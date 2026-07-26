from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import requests


TIERS = ("compact", "balanced", "performance")
RESOURCE_PROFILES = ("eco", "balanced", "maximum")
CALIBRATION_VERSION = "hardware-profile-v1"
DEFAULT_MODELS = {
    "compact": ("qwen2.5:3b", "qwen2.5:3b"),
    "balanced": ("qwen3:8b", "qwen3:8b"),
    "performance": ("qwen3:8b", "qwen3:8b"),
}
RESOURCE_SETTINGS = {
    "eco": {"num_ctx": 4096, "keep_alive": "30s", "concurrency": 1, "memory_reserve_ratio": 0.35},
    "balanced": {"num_ctx": 4096, "keep_alive": "5m", "concurrency": 1, "memory_reserve_ratio": 0.20},
    "maximum": {"num_ctx": 4096, "keep_alive": "30m", "concurrency": 2, "memory_reserve_ratio": 0.10},
}


@dataclass(frozen=True)
class HardwareSnapshot:
    total_ram_gb: float
    available_ram_gb: float
    cpu_cores: int
    platform: str
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    installed_models: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeProfile:
    requested_tier: str
    resolved_tier: str
    resource_profile: str
    fast_model: str
    strong_model: str
    num_ctx: int
    keep_alive: str
    concurrency: int
    memory_reserve_ratio: float
    downgrade_reasons: tuple[str, ...]
    hardware: HardwareSnapshot
    calibration: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _memory_bytes() -> tuple[int, int]:
    if platform.system() == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
    if hasattr(os, "sysconf"):
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        return int(total), int(available)
    return 0, 0


def _gpu_info() -> tuple[str, float]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        first = result.stdout.strip().splitlines()[0]
        name, memory_mb = [part.strip() for part in first.rsplit(",", 1)]
        return name, round(float(memory_mb) / 1024, 2)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return "", 0.0


def installed_ollama_models(base_url: str | None = None) -> dict[str, int]:
    base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        response.raise_for_status()
        return {
            str(item.get("name", "")): int(item.get("size") or 0)
            for item in response.json().get("models", []) if item.get("name")
        }
    except (requests.RequestException, ValueError, TypeError):
        return {}


def inspect_hardware() -> HardwareSnapshot:
    total, available = _memory_bytes()
    gpu_name, gpu_vram = _gpu_info()
    gib = 1024 ** 3
    return HardwareSnapshot(
        total_ram_gb=round(total / gib, 2) if total else 0.0,
        available_ram_gb=round(available / gib, 2) if available else 0.0,
        cpu_cores=os.cpu_count() or 1,
        platform=f"{platform.system()} {platform.release()} {platform.machine()}",
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram,
        installed_models=installed_ollama_models(),
    )


def classify_tier(total_ram_gb: float) -> str:
    if total_ram_gb >= 23.0:
        return "performance"
    if total_ram_gb >= 11.5:
        return "balanced"
    return "compact"


def _model_available(models: dict[str, int], requested: str) -> bool:
    if requested in models:
        return True
    return any(name.split(":")[0] == requested.split(":")[0] and requested.endswith(":latest") for name in models)


def _fingerprint(snapshot: HardwareSnapshot, tier: str, resource: str, fast: str, strong: str) -> str:
    payload = json.dumps(
        [CALIBRATION_VERSION, asdict(snapshot), tier, resource, fast, strong],
        sort_keys=True,
        default=str,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _load_calibration(path: Path, fingerprint: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if value.get("fingerprint") == fingerprint else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_calibration(profile: RuntimeProfile, result: dict[str, Any], path: str | None = None) -> None:
    target = Path(path or os.getenv("HARDWARE_PROFILE_PATH", "outputs/cache/hardware_profile.json"))
    target.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(
        profile.hardware, profile.resolved_tier, profile.resource_profile,
        profile.fast_model, profile.strong_model,
    )
    payload = {"fingerprint": fingerprint, "saved_at": time.time(), **result}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def resolve_runtime_profile(
    requested_tier: str | None = None,
    resource_profile: str | None = None,
    hardware: HardwareSnapshot | None = None,
) -> RuntimeProfile:
    hardware = hardware or inspect_hardware()
    requested = str(requested_tier or os.getenv("MODEL_TIER", "auto")).strip().lower()
    if requested not in {*TIERS, "auto"}:
        requested = "auto"
    resource = str(resource_profile or os.getenv("RESOURCE_PROFILE", "balanced")).strip().lower()
    if resource not in RESOURCE_PROFILES:
        resource = "balanced"
    resolved = classify_tier(hardware.total_ram_gb) if requested == "auto" else requested
    reasons: list[str] = []
    order = list(TIERS)

    while True:
        default_fast, default_strong = DEFAULT_MODELS[resolved]
        fast = os.getenv(f"{resolved.upper()}_FAST_MODEL", os.getenv("FAST_MODEL", default_fast))
        strong = os.getenv(f"{resolved.upper()}_STRONG_MODEL", os.getenv("STRONG_MODEL", default_strong))
        required = {fast, strong}
        missing = sorted(model for model in required if not _model_available(hardware.installed_models, model))
        if not missing or not hardware.installed_models or resolved == "compact" or requested != "auto":
            if missing:
                reasons.append("missing local models: " + ", ".join(missing))
            break
        previous = resolved
        resolved = order[max(0, order.index(resolved) - 1)]
        reasons.append(f"downgraded from {previous}: required model unavailable")

    settings = dict(RESOURCE_SETTINGS[resource])
    if resource == "maximum":
        settings["concurrency"] = max(1, min(4, hardware.cpu_cores // 4 or 1))
    path = Path(os.getenv("HARDWARE_PROFILE_PATH", "outputs/cache/hardware_profile.json"))
    fingerprint = _fingerprint(hardware, resolved, resource, fast, strong)
    calibration = _load_calibration(path, fingerprint)
    if resource == "maximum" and calibration.get("recommended_concurrency"):
        settings["concurrency"] = int(calibration["recommended_concurrency"])
    return RuntimeProfile(
        requested_tier=requested,
        resolved_tier=resolved,
        resource_profile=resource,
        fast_model=fast,
        strong_model=strong,
        downgrade_reasons=tuple(reasons),
        hardware=hardware,
        calibration=calibration,
        **settings,
    )
