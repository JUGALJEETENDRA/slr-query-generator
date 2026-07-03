from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DECISIONS = {"KEEP", "MAYBE", "REJECT"}


@dataclass(frozen=True)
class FusionPolicy:
    id: str = "conservative_inclusion_v1"
    conflict_decision: str = "MAYBE"


DEFAULT_FUSION_POLICY = FusionPolicy()


def fuse_screening_decisions(
    litsync_result: dict[str, Any],
    direct_ai_result: dict[str, Any],
    policy: FusionPolicy = DEFAULT_FUSION_POLICY,
) -> dict[str, Any]:
    litsync_decision = _normalize_decision(litsync_result.get("decision"))
    direct_decision = _normalize_decision(direct_ai_result.get("decision"))
    agreement = _agreement_status(litsync_decision, direct_decision)

    final_decision = _final_decision(litsync_decision, direct_decision, policy)
    confidence = _combined_confidence(
        _as_confidence(litsync_result.get("confidence")),
        _as_confidence(direct_ai_result.get("confidence")),
        agreement,
    )

    return {
        "decision": final_decision,
        "reason": _fused_reason(
            final_decision,
            litsync_decision,
            direct_decision,
            agreement,
            litsync_result.get("reason", ""),
            direct_ai_result.get("reason", ""),
        ),
        "confidence": confidence,
        "agreement": agreement,
        "policy_id": policy.id,
    }


def _final_decision(litsync_decision: str, direct_decision: str, policy: FusionPolicy) -> str:
    if litsync_decision == direct_decision:
        return litsync_decision
    if "KEEP" in {litsync_decision, direct_decision} and "REJECT" in {litsync_decision, direct_decision}:
        return policy.conflict_decision
    if "KEEP" in {litsync_decision, direct_decision}:
        return "KEEP"
    return "MAYBE"


def _agreement_status(litsync_decision: str, direct_decision: str) -> str:
    if litsync_decision == direct_decision:
        return "agree"
    if "KEEP" in {litsync_decision, direct_decision} and "REJECT" in {litsync_decision, direct_decision}:
        return "conflict"
    return "partial"


def _combined_confidence(litsync_confidence: float, direct_confidence: float, agreement: str) -> float:
    base = (litsync_confidence + direct_confidence) / 2
    if agreement == "agree":
        base += 0.08
    elif agreement == "conflict":
        base -= 0.18
    else:
        base -= 0.05
    return round(max(0.0, min(1.0, base)), 4)


def _fused_reason(
    final_decision: str,
    litsync_decision: str,
    direct_decision: str,
    agreement: str,
    litsync_reason: str,
    direct_reason: str,
) -> str:
    if agreement == "agree":
        return (
            f"Final decision is {final_decision} because both LitSync Workflow and Direct AI "
            f"independently reached {final_decision}."
        )
    if agreement == "conflict":
        return (
            "Final decision is MAYBE because LitSync Workflow and Direct AI produced a direct "
            f"KEEP/REJECT conflict. LitSync: {_clean(litsync_reason)} Direct AI: {_clean(direct_reason)}"
        )
    return (
        f"Final decision is {final_decision} after partial agreement: LitSync Workflow returned "
        f"{litsync_decision}, while Direct AI returned {direct_decision}."
    )


def _normalize_decision(value: Any) -> str:
    decision = str(value or "MAYBE").strip().upper()
    return decision if decision in DECISIONS else "MAYBE"


def _as_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, confidence))


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())
