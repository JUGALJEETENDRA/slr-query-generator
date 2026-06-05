# validator.py
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

# Phase 4B: Explicit Negative Ontology Fencing Contracts
NEGATIVE_ONTOLOGY_RULES = {
    "rag": ["security auditing", "patch management", "intrusion detection", "patch analysis", "automated vulnerability"],
    "tumor": ["deepfake detection", "media forensics", "manipulation detection"],
    "edge security": ["membership inference", "model inversion", "federated learning", "secure aggregation"],
    "iot healthcare security": ["sensitivity", "specificity", "auc", "roc", "diagnostic accuracy"]
}

def clean_term_for_check(text: str) -> str:
    return text.lower().strip().replace("*", "")

def scrub_leading_conversational_filler(text: str) -> str:
    cleaned = re.sub(r'^(main|primary|precision of|impact of|detecting|causes of)\s+', '', text, flags=re.IGNORECASE)
    return cleaned.strip()

def run_validation_sieve(current_context: SLRQueryContext) -> SLRQueryContext:
    """Unpacks configurations, blocks tool names, and drops non-ASCII Unicode leaks completely."""
    
    flattened_pool = " ".join([
        " ".join(current_context.technology),
        " ".join(current_context.domain),
        " ".join(current_context.comparison)
    ]).lower()

    active_deny_set = set()
    for anchor, forbidden_terms in NEGATIVE_ONTOLOGY_RULES.items():
        if anchor in flattened_pool:
            for term in forbidden_terms:
                active_deny_set.add(term)

    def unpack_parentheticals(field_array: list[str]) -> list[str]:
        unpacked = []
        for term in field_array:
            match = re.match(r"(.+?)\s*\((.+?)\)", term)
            if match:
                base_phrase = match.group(1).strip()
                acronym_phrase = match.group(2).strip()
                unpacked.append(base_phrase if "*" in base_phrase else f"{base_phrase}*")
                unpacked.append(acronym_phrase if "*" in acronym_phrase else f"{acronym_phrase}*")
            else:
                unpacked.append(term)
        return unpacked

    def sanitize_facet(field_array: list[str], is_outcomes=False) -> list[str]:
        unpacked_list = unpack_parentheticals(field_array)
        sanitized = []
        for term in unpacked_list:
            cleaned_token = clean_term_for_check(term)
            
            if cleaned_token in UNIVERSAL_NOISE_BLACKSET or len(cleaned_token) <= 2:
                continue
            if "serverless" in cleaned_token:
                continue
                
            # Block tokens containing non-ASCII / CJK character sets
            if re.search(r'[\u4e00-\u9fff]', cleaned_token):
                continue

            # Phase 4B Rule Enforcement: Block tokens matching active deny listings
            if any(denied_term in cleaned_token for denied_term in active_deny_set):
                continue

            processed_term = term
            if is_outcomes:
                processed_term = scrub_leading_conversational_filler(term)
                cleaned_outcome = clean_term_for_check(processed_term)
                if cleaned_outcome in OUTCOMES_ONLY_BLACKSET or cleaned_outcome in UNIVERSAL_NOISE_BLACKSET or len(processed_term) <= 2:
                    continue
            sanitized.append(processed_term)
        return sanitized

    tech_processed = sanitize_facet(current_context.technology, is_outcomes=False)
    domain_processed = sanitize_facet(current_context.domain, is_outcomes=False)
    comp_processed = sanitize_facet(current_context.comparison, is_outcomes=False)
    context_processed = sanitize_facet(current_context.context, is_outcomes=False)
    outcomes_processed = sanitize_facet(current_context.outcomes, is_outcomes=True)

    global_seen_register = set()
    def apply_deduplication_filter(field_array: list[str]) -> list[str]:
        filtered_list = []
        for term in field_array:
            normalized_core = clean_term_for_check(term)
            if normalized_core in global_seen_register:
                continue
            global_seen_register.add(normalized_core)
            filtered_list.append(term)
        return filtered_list

    return SLRQueryContext(
        technology=apply_deduplication_filter(tech_processed),
        domain=apply_deduplication_filter(domain_processed),
        comparison=apply_deduplication_filter(comp_processed),
        context=apply_deduplication_filter(context_processed),
        outcomes=apply_deduplication_filter(outcomes_processed)
    )