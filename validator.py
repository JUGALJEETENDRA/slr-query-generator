import re
from schema import SLRQueryContext

OUTCOMES_ONLY_BLACKSET = {
    "istio", "service mesh", "kubernetes", "k8s", "docker", "hypervisor", 
    "virtual machine", "vms", "devops", "gitops", "devsecops", "ci/cd", 
    "large language model", "llm", "rag", "nlp", "containerized app",
    "cloud-native", "cloud computing", "monolithic", "monolith"
}

UNIVERSAL_NOISE_BLACKSET = {
    "effective", "effectiveness", "efficient", "efficiency", "computing", "storage",
    "automation efficiency", "enhance perception precision", "increase recognition accuracy",
    "efficacious", "improved", "improvement", "enhance", "increase", "impact", "causes",
    "improved patient outcomes", "efficient care delivery", "therapeutically successful",
    "simulink"
}

NEGATIVE_ONTOLOGY_RULES = {
    "rag": ["security auditing", "patch management", "intrusion detection", "patch analysis", "automated vulnerability"],
    "tumor": ["deepfake detection", "media forensics", "manipulation detection"],
    "edge security": ["membership inference", "model inversion", "federated learning", "secure aggregation"],
    "iot healthcare security": ["sensitivity", "specificity", "auc", "roc", "diagnostic accuracy"]
}

def run_validation_sieve(current_context: SLRQueryContext) -> SLRQueryContext:
    flattened_pool = " ".join([
        " ".join(current_context.technology),
        " ".join(current_context.domain),
        " ".join(current_context.comparison)
    ]).lower()

    active_deny_set = {
        term for anchor, forbidden in NEGATIVE_ONTOLOGY_RULES.items() 
        if anchor in flattened_pool for term in forbidden
    }

    def sanitize_facet(field_array: list[str], is_outcomes=False) -> list[str]:
        sanitized = []
        for term in field_array:
            # Handle parens
            match = re.match(r"(.+?)\s*\((.+?)\)", term)
            terms_to_check = [match.group(1).replace("*", ""), match.group(2).replace("*", "")] if match else [term]
            
            for t in terms_to_check:
                cleaned = t.lower().strip().replace("*", "")
                if len(cleaned) <= 2 or cleaned in UNIVERSAL_NOISE_BLACKSET or re.search(r'[\u4e00-\u9fff]', cleaned):
                    continue
                if any(denied in cleaned for denied in active_deny_set):
                    continue
                if is_outcomes and (cleaned in OUTCOMES_ONLY_BLACKSET or "serverless" in cleaned):
                    continue
                sanitized.append(t)
        return sanitized

    # Final pass
    tech = sanitize_facet(current_context.technology)
    domain = sanitize_facet(current_context.domain)
    comp = sanitize_facet(current_context.comparison)
    ctx = sanitize_facet(current_context.context)
    out = sanitize_facet(current_context.outcomes, is_outcomes=True)

    return SLRQueryContext(technology=list(set(tech)), domain=list(set(domain)), 
                           comparison=list(set(comp)), context=list(set(ctx)), outcomes=list(set(out)))