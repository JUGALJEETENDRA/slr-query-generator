# comparator_registry.py
from schema import SLRQueryContext

COMPARATOR_DUALITIES = {
    "VERSION_CONTROL_DUAL": {
        "anchors": ["gitflow", "trunk-based", "branching", "workflows"],
        "forks": ["feature branching*", "long-lived branches*", "branch-per-feature*", "merge conflicts*", "release branching*"]
    },
    "VIRTUALIZATION_DUAL": {
        "anchors": ["docker", "virtual machines", "vms", "containers", "hypervisor"],
        "forks": ["container runtime performance*", "resource isolation impact*", "hypervisor overhead*", "hardware virtualization*"]
    }
}

def expand_comparator_registry(context: SLRQueryContext) -> SLRQueryContext:
    """Expands dualities independently using strict token intersection checks."""
    combined_text = " ".join([
        " ".join(context.technology),
        " ".join(context.domain),
        " ".join(context.comparison)
    ]).lower()

    updated_comparison = list(context.comparison)

    # --- Q55 COMPARTMENTALIZATION FIX ---
    # Require strict intersection matching before triggering the perception pack
    if any(l in combined_text for l in ["lidar", "radar", "sensor fusion"]) and any(c in combined_text for c in ["camera", "vision", "rgb"]):
        perception_forks = ["monocular vision*", "stereo vision*", "rgb perception*", "depth sensing*", "range sensing*", "3d perception*", "point cloud detection*"]
        for token in perception_forks:
            if token.lower().replace("*", "") not in [c.lower().replace("*", "") for c in updated_comparison]:
                updated_comparison.append(token)

    # Standard Anchors Processing
    for duality_name, configuration in COMPARATOR_DUALITIES.items():
        if any(anchor in combined_text for anchor in configuration["anchors"]):
            for fork_token in configuration["forks"]:
                if fork_token.lower().replace("*", "") not in [c.lower().replace("*", "") for c in updated_comparison]:
                    updated_comparison.append(fork_token)

    return SLRQueryContext(
        technology=context.technology,
        domain=context.domain,
        comparison=updated_comparison,
        context=context.context,
        outcomes=context.outcomes
    )