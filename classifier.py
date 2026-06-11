import re
from schema import SLRQueryContext

DOMAIN_RULES = {
    "SOFTWARE_ENGINEERING": {"pair programming", "test-driven development", "tdd", "mutation testing", "code coverage", "software quality", "maintainability", "technical debt", "refactoring", "code review", "defect density", "flaky tests", "low-code", "static application security testing", "sast"},
    "DEVOPS": {"continuous integration", "continuous delivery", "continuous deployment", "ci/cd", "gitflow", "gitops", "trunk-based", "infrastructure as code", "iac", "devsecops", "chaos engineering", "fault injection", "chaos mesh", "argocd", "flux cd", "deployment pipeline"},
    "CLOUD": {"autoscaling", "dynamic autoscaling", "serverless", "faas", "aws lambda", "cloud functions", "service mesh", "istio", "linkerd", "kubernetes scheduling", "container orchestration", "cloud bursting", "distributed tracing", "platform engineering", "stateful applications", "cost optimization", "multi-cloud", "hybrid cloud", "cloud deployment", "edge computing", "mobile edge computing", "mec", "task offloading", "iot devices", "low-power iot"},
    "ROBOTICS": {"autonomous drones", "drones", "uav", "uavs", "unmanned aerial vehicles", "autonomous vehicles", "self-driving", "automated driving", "sensor fusion", "multimodal perception", "lidar", "radar", "slam"},
    "EDTECH": {"intelligent tutoring", "tutoring systems", "higher education", "student engagement", "learning outcomes", "e-learning", "adaptive learning", "educational technology"},
    "HEALTHCARE": {"diabetic retinopathy", "ehr", "electronic health records", "clinical decision support", "healthcare diagnostics", "patient monitoring", "radiology", "medical imaging", "personalized medicine", "hospital readmission", "readmission prediction", "robotic surgery", "surgical", "surgery", "laparoscopic", "laparotomy"},
    "CYBERSECURITY": {"threat hunting", "intrusion detection", "siem", "soc", "ransomware", "zero-trust", "malware analysis", "honeypots", "phishing detection", "edge security", "adversarial training", "insider threats", "evasion attacks", "lateral movement", "vulnerability remediation", "zero trust"},
    "BLOCKCHAIN": {"blockchain", "zero-knowledge proofs", "zk-proofs", "decentralized identity", "smart contracts", "cross-chain", "interoperability", "distributed ledger", "traceability", "voting systems"},
    "AI_ML": {"transformer models", "retrieval-augmented generation", "rag", "model quantization", "explainable ai", "xai", "transfer learning", "multimodal foundation models", "synthetic data", "federated learning", "reinforcement learning", "large language models", "llm", "active learning", "knowledge distillation", "prompt engineering"},
    "COMPUTER_VISION": {"vision transformers", "vit", "deepfake detection", "semantic segmentation", "road scene understanding", "tumor segmentation", "object detection", "object tracking", "synthetic image generation"},
    "QUANTUM": {"quantum support vector", "quantum svm", "quantum machine learning", "qml"},
    "EMERGING_TECH": {"digital twin", "digital twins", "cyber-physical systems", "industrial manufacturing", "smart manufacturing"}
}

def classify_extracted_context(context: SLRQueryContext) -> str:
    combined_text_pool = " ".join([
        " ".join(context.technology),
        " ".join(context.domain),
        " ".join(context.comparison),
        " ".join(context.context),
        " ".join(context.outcomes)
    ]).lower().strip()

    scores = {domain: 0 for domain in DOMAIN_RULES.keys()}

    for domain, keywords in DOMAIN_RULES.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', combined_text_pool):
                scores[domain] += 50 

    primary_domain = max(scores, key=scores.get)
    return primary_domain if scores[primary_domain] > 0 else "GENERIC_CS"