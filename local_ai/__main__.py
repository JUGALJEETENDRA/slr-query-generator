from __future__ import annotations

import argparse
import json

from .hardware import resolve_runtime_profile
from .three_layer import DEEP_MODEL, EDGE_MODEL, THREE_LAYER_PROMPT_VERSION, TRIAGE_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LitSync's adaptive local-AI runtime")
    parser.add_argument("--tier", choices=["auto", "compact", "balanced", "performance"], default="auto")
    parser.add_argument("--resource", choices=["eco", "balanced", "maximum"], default="balanced")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    profile = resolve_runtime_profile(args.tier, args.resource)
    print(json.dumps({
        **profile.as_dict(),
        "fast_model": TRIAGE_MODEL,
        "strong_model": DEEP_MODEL,
        "architecture_version": THREE_LAYER_PROMPT_VERSION,
        "triage_model": TRIAGE_MODEL,
        "deep_model": DEEP_MODEL,
        "edge_model": EDGE_MODEL,
        "calibration_disabled": True,
        "legacy_hardware_tier_models_ignored": True,
    }, indent=2))


if __name__ == "__main__":
    main()
