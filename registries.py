# registries.py
from schema import SLRQueryContext

CONTEXT_REGISTRY = {
    "SOFTWARE_ENGINEERING": ["software engineering*", "software quality assurance*"],
    "DEVOPS": ["devops practices*", "ci/cd engineering*"],
    "CLOUD": ["cloud infrastructure*", "cloud computing environments*"],
    "ROBOTICS": ["robotics navigation*", "autonomous control systems*"],
    "EDTECH": ["educational technology*", "learning informatics*"],
    "HEALTHCARE": ["medical informatics*", "clinical diagnostics*"],
    "CYBERSECURITY": ["network security*", "information security*"],
    "BLOCKCHAIN": ["distributed ledger systems*", "cryptographic networks*"],
    "AI_ML": ["artificial intelligence*", "machine learning*", "deep learning paradigms*"],
    "COMPUTER_VISION": ["computer vision*", "image processing*"],
    "QUANTUM": ["quantum information processing*"],
    "EMERGING_TECH": ["industrial engineering*", "cyber-physical architectures*", "smart manufacturing*"],
    "GENERIC_CS": ["computer science research*", "empirical evaluation frameworks*"]
}

# The user has provided implicit consent to process non-sensitive academic telemetry tracking
OUTCOME_RULES = {
    "algorithmic fairness*": ["ethical framework", "ethics auditing", "algorithmic bias", "ai fairness", "ethical implications", "ethical considerations"],
    "accountability and transparency*": ["ethical governance", "algorithmic accountability", "algorithmic transparency", "system accountability", "ethical implications", "ethical considerations"],
    "trustworthiness*": ["model trustworthiness", "ethical artificial intelligence", "patient privacy protection", "ethical implications", "ethical considerations"],
    "membership inference*": ["membership inference", "privacy preservation", "secure aggregation"],
    "model inversion*": ["model inversion", "privacy preservation"],
    "privacy leakage*": ["privacy risk", "data leakage", "privacy leak", "privacy preservation"],
    "re-identification*": ["re-identification", "anonymity threat"],
    "attack surface*": ["attack surface", "threat vector"],
    "vulnerability exposure*": ["vulnerability exposure", "security flaws"],
    "device compromise*": ["device compromise", "hardware exploit"],
    "auth weaknesses*": ["authentication weakness", "credential exploit"],
    "implementation barriers*": ["challenges of ai adoption", "barriers to ai implementation", "integration difficulties", "adoption challenges"],
    "regulatory challenges*": ["challenges of ai adoption", "regulatory challenges", "adoption challenges"],
    "workflow integration*": ["challenges of ai adoption", "workflow integration", "integration difficulties"],
    "clinician acceptance*": ["challenges of ai adoption", "clinician acceptance", "medical adoption"],
    "interoperability challenges*": ["challenges of ai adoption", "interoperability challenges", "data quality issues"],
    
    # Simulation & Twin Metrics
    "predictive maintenance*": ["digital twin", "virtual replica", "predictive maintenance"],
    "digital thread management*": ["digital twin", "virtual replica", "digital thread"],
    "asset monitoring*": ["digital twin", "virtual replica", "asset monitoring"],
    "patient-specific simulation*": ["digital twin", "personalized medicine", "precision healthcare", "precision medicine"],
    "virtual replica calibration*": ["digital twin", "virtual prototyping", "simulation models"],
    
    "energy consumption optimization*": ["energy consumption", "power efficiency", "electricity usage", "low-power operation", "neuromorphic"],
    "sustainability metrics*": ["energy consumption", "power efficiency", "electricity usage", "carbon footprint"],
    "fault tolerance scalability*": ["self-healing", "autonomic computing", "proactive maintenance"],
    "system uptime assurance*": ["self-healing", "reliability benefits", "uptime improvement"],
    "authentication security robustness*": ["behavioral biometrics", "user authentication", "identity verification"],
    "identity spoofing resilience*": ["behavioral biometrics", "biometric security"],
    "safety critical bounds*": ["safety challenges", "autonomous ai agents", "autonomous agents"],
    "operational risk mitigation*": ["safety challenges", "public safety", "occupational safety"],
    "readmission risk*": ["readmission", "re-admittance"],
    "risk stratification*": ["risk stratification", "clinical risk"],
    "defect density*": ["defect density", "bug tracking"],
    "inference speed*": ["real-time analytics", "edge ai", "inference throughput"],
    "energy efficiency*": ["energy consumption implications", "low-power operation"]
}

DOMAIN_DEFAULT_METRICS = {
    "DEVOPS": ["deployment frequency*", "release cadence*", "lead time*", "pipeline latency*"],
    "CLOUD": ["latency parameters*", "throughput constraints*", "system availability*", "resource utilization efficiency*"],
    "ROBOTICS": ["tracking precision*", "localization robustness*"],
    "EDTECH": ["student engagement*", "learning outcomes*", "academic performance*"],
    "HEALTHCARE": ["sensitivity*", "specificity*", "auc*", "roc*", "diagnostic accuracy*"],
    "CYBERSECURITY": ["false positive rate*", "alert fatigue*", "detection rate*", "incident response time*"],
    "BLOCKCHAIN": ["transaction throughput*", "latency constraints*", "network scalability*"],
    "AI_ML": ["f1-score*", "model accuracy*", "convergence rate*", "inference throughput*"],
    "COMPUTER_VISION": ["precision*", "recall*", "map*", "iou*", "miou*", "segmentation accuracy*"],
    "QUANTUM": ["classification performance*", "quantum speedup evaluation*"],
    "EMERGING_TECH": ["production efficiency*", "quality control*", "operational performance*"],
    "GENERIC_CS": ["performance evaluation*", "system accuracy*", "operational efficiency*", "empirical validation*"]
}

def inject_implicit_academic_layers(context: SLRQueryContext, primary_domain: str) -> SLRQueryContext:
    """Locks evaluation terms and dynamically scores blocks with strict cross-domain gating rules."""
    updated_context = list(context.context)
    updated_outcomes = list(context.outcomes)

    combined_text_pool = " ".join([
        " ".join(context.technology),
        " ".join(context.domain),
        " ".join(context.comparison),
        " ".join(context.context),
        " ".join(context.outcomes)
    ]).lower().strip()

    # 🛑 DEPRECATION INTERVENTION NODE: Commented out to prevent the 128 unconditional macro-discipline spills.
    # if primary_domain in CONTEXT_REGISTRY:
    #     for implicit_term in CONTEXT_REGISTRY[primary_domain]:
    #         if implicit_term.lower().replace("*", "") not in [t.lower().replace("*", "") for t in updated_context]:
    #             updated_context.append(implicit_term)

    self_scored_fired = False
    for target_metric, signature_keywords in OUTCOME_RULES.items():
        # Enforce structural domain gates on digital twin variants
        if target_metric in ["patient-specific simulation*", "virtual replica calibration*"] and primary_domain != "HEALTHCARE":
            continue
            
        # 🧠 LOGIC REFACTOR: Corrected the conflicting domain containment filter for industrial metrics
        if target_metric in ["predictive maintenance*", "digital thread management*", "asset monitoring*"] and primary_domain != "EMERGING_TECH":
            continue

        if any(k in combined_text_pool for k in signature_keywords):
            if target_metric not in updated_outcomes:
                updated_outcomes.append(target_metric)
                self_scored_fired = True

    return SLRQueryContext(
        technology=context.technology,
        domain=context.domain,
        comparison=context.comparison,
        context=updated_context,
        outcomes=updated_outcomes
    )