# pipeline_test.py
import re
import json
from openai import OpenAI
import instructor
from extractor import extract_5_facets
from generator import expand_base_synonyms
from acronym_expander import expand_acronym_layer
from ontology_expander import expand_ontology_layer
from validator import run_validation_sieve
from compiler import compile_boolean_query

# PHASE 4 DETERMINISTIC COMPILER FIREWALLS
from schema import SLRQueryContext  
from classifier import classify_extracted_context
from registries import inject_implicit_academic_layers
from comparator_registry import expand_comparator_registry

GLOBAL_FORBIDDEN_COMPARATORS = [
    "legacy", "perimeter", "traditional", "firewall", "nac", "bare-metal", 
    "monolithic", "non-containerized", "centralized", "cloud-only", 
    "legacy server", "datacenter", "pooled data", "relational", "sql", 
    "legacy database", "manual", "legacy inspection", "waterfall", 
    "reactive", "schedule-based", "legacy maintenance", "local execution", 
    "on-device", "standalone processing", "black-box", "opaque", 
    "uninterpretable", "traditional ml", "conventional", "traditional surgery", "laparoscopic"
]

def classify_evaluated_term(term: str, forbidden_comparators: list) -> str:
    val = term.lower().strip().replace("*", "")
    if any(comp in val for comp in forbidden_comparators):
        return "comparator_leak"
    canonical_tokens = [
        "kubernetes", "k8s", "mec", "multi-access edge", "fabric", "hyperledger", 
        "da vinci", "fog computing", "cloudlet", "flux cd", "bert", "deep q-network", 
        "dqn", "secure aggregation"
    ]
    if any(token in val for token in canonical_tokens):
        return "canonical_realization"
    fabricated_tokens = ["consented ledger", "privately managed blockchain"]
    if any(token in val for token in fabricated_tokens):
        return "fabricated_term"
    metric_tokens = ["latency", "throughput", "performance", "delivery", "speed", "cadence", "lead time", "delay", "variance", "accuracy"]
    if any(token in val for token in metric_tokens):
        return "metric_inflation"
    generic_structural_tokens = ["surgical systems", "autonomous systems", "automated systems", "systems", "technologies", "architectures"]
    macro_discipline_tokens = [
        "computer vision", "artificial intelligence", "machine learning", "deep learning paradigm",
        "information security", "software engineering", "computer science research", 
        "medical care", "healthcare", "software development", "image processing", 
        "cryptographic networks", "empirical evaluation framework", "devops practice"
    ]
    is_macro_discipline = any(discipline in val for discipline in macro_discipline_tokens)
    is_generic_structural_suffix = val.endswith(("systems", "technologies", "architectures")) and not any(anchor in val for anchor in ["zero trust", "robotic", "container", "blockchain", "cybersecurity"])
    
    if val in generic_structural_tokens or is_macro_discipline or is_generic_structural_suffix:
        return "semantic_generalization"
    related_tokens = [
        "infrastructure", "computing environment", "cloud native architecture", "management platform",
        "privacy protection", "information protection", "security risks", "vulnerabilities", "threat analysis"
    ]
    if any(token in val for token in related_tokens):
        return "related_concept"
    brainstorm_tokens = ["fringe", "periphery", "doubly-encrypted"]
    if any(token in val for token in brainstorm_tokens):
        return "topic_brainstorming"
    return "pending_manual_audit"

def get_context_snapshot(ctx: SLRQueryContext) -> set:
    """Flattens all strings currently inside a context snapshot into a unified unique lookup set."""
    snapshot = set()
    for layer in [ctx.technology, ctx.domain, ctx.comparison, ctx.context, ctx.outcomes]:
        for phrase in layer:
            snapshot.add(phrase.strip().lower())
    return snapshot

def run_stress_test_benchmark():
    local_client = instructor.from_openai(
        OpenAI(base_url="http://localhost:11434/v1", api_key="ollama-local"), mode=instructor.Mode.MD_JSON  
    )
    LOCAL_MODEL = "qwen2.5:3b"

    stress_test_questions = [
        # SOFTWARE ENGINEERING (1–20)
        "How does continuous integration impact software defect density in distributed development teams?",
        "Does trunk-based development improve release frequency compared to GitFlow workflows?",
        "What is the effect of infrastructure as code on deployment reproducibility in cloud environments?",
        "How does GitOps influence configuration drift in Kubernetes clusters?",
        "Does mutation testing improve fault detection compared to code coverage testing?",
        "What are the primary causes of flaky tests in CI/CD pipelines?",
        "How effective is automated regression testing for reducing post-release defects?",
        "Does pair programming improve code quality in agile software projects?",
        "How does feature flag management affect deployment risk in large-scale applications?",
        "What is the impact of DevSecOps adoption on vulnerability remediation time?",
        "How does service virtualization improve integration testing efficiency?",
        "What are the scalability challenges of microservice architectures in enterprise systems?",
        "Does chaos engineering improve resilience in cloud-native applications?",
        "How does container orchestration affect resource utilization in distributed systems?",
        "What are the software maintainability implications of low-code development platforms?",
        "How effective is static application security testing compared to dynamic testing?",
        "Does automated code review improve software quality compared to manual review?",
        "How does technical debt affect software delivery performance?",
        "What is the impact of test-driven development on software reliability?",
        "How do observability platforms improve incident response effectiveness?",

        # DEVOPS / CLOUD COMPUTING (21–35)
        "How does dynamic autoscaling affect cloud resource efficiency?",
        "What are the performance trade-offs of serverless computing versus containerized workloads?",
        "Does edge computing reduce latency compared to centralized cloud architectures?",
        "How does service mesh adoption affect application observability?",
        "What is the impact of multi-cloud deployment strategies on system availability?",
        "How does Kubernetes scheduling affect workload performance?",
        "What are the security risks of containerized cloud environments?",
        "Does infrastructure automation improve operational efficiency?",
        "How does cloud bursting affect workload scalability?",
        "What are the energy consumption implications of cloud-native architectures?",
        "How does distributed tracing improve microservice debugging?",
        "Does platform engineering improve developer productivity?",
        "What are the challenges of managing stateful applications in Kubernetes?",
        "How does cloud cost optimization impact application performance?",
        "What are the reliability benefits of self-healing infrastructure?",

        # CYBERSECURITY (36–50)
        "How effective are graph neural networks for intrusion detection?",
        "Does LLM-assisted threat hunting improve analyst productivity?",
        "What are the major privacy risks of federated learning systems?",
        "How effective is anomaly detection for ransomware identification?",
        "What are the cybersecurity vulnerabilities of IoT healthcare devices?",
        "Does zero-trust architecture reduce insider threats?",
        "How does behavioral biometrics improve user authentication?",
        "What are the limitations of SIEM systems for cyber threat detection?",
        "Does automated malware analysis improve incident response time?",
        "How effective are honeypots in detecting advanced persistent threats?",
        "What are the privacy implications of facial recognition systems?",
        "Does blockchain improve data integrity in distributed systems?",
        "How effective is phishing detection using machine learning?",
        "What are the challenges of securing edge computing environments?",
        "Does adversarial training improve robustness against evasion attacks?",

        # ARTIFICIAL INTELLIGENCE / MACHINE LEARNING (51–65)
        "How effective are transformer models for recommendation systems?",
        "Does retrieval-augmented generation improve factual accuracy in LLMs?",
        "How does model quantization affect inference performance?",
        "What are the limitations of explainable AI techniques in healthcare?",
        "Does transfer learning improve image classification accuracy?",
        "How effective are multimodal foundation models for disease diagnosis?",
        "What is the impact of synthetic data on machine learning model performance?",
        "How does federated learning affect model accuracy?",
        "Does reinforcement learning improve traffic signal optimization?",
        "What are the biases present in large language models?",
        "How effective are graph neural networks for fraud detection?",
        "Does active learning reduce annotation costs?",
        "How does knowledge distillation affect model efficiency?",
        "What are the safety challenges of autonomous AI agents?",
        "Does prompt engineering improve LLM task performance?",

        # COMPUTER VISION (66–75)
        "Do vision transformers outperform CNNs for tumor segmentation?",
        "How effective is sensor fusion for autonomous vehicle perception?",
        "Does synthetic image generation improve object detection accuracy?",
        "How does LiDAR-based perception compare to camera-only perception?",
        "What are the challenges of deepfake detection?",
        "Does multimodal perception improve autonomous driving reliability?",
        "How effective is SLAM in GPS-denied environments?",
        "What is the impact of adverse weather on object tracking systems?",
        "Do self-supervised learning approaches improve image representation quality?",
        "How effective is semantic segmentation for road scene understanding?",

        # HEALTHCARE INFORMATICS (76–85)
        "How effective are machine learning algorithms in diabetic retinopathy detection?",
        "Does federated learning improve privacy preservation in healthcare AI?",
        "How effective are multimodal AI systems for clinical decision support?",
        "What are the challenges of AI adoption in healthcare diagnostics?",
        "Does remote patient monitoring improve healthcare outcomes?",
        "How effective is predictive analytics for hospital readmission prediction?",
        "What are the privacy risks of electronic health records?",
        "Does AI-assisted radiology improve diagnostic accuracy?",
        "How effective are digital twins for personalized medicine?",
        "What are the ethical implications of generative AI in healthcare?",

        # BLOCKCHAIN / DISTRIBUTED SYSTEMS (86–92)
        "Does blockchain-enabled traceability improve supply chain transparency?",
        "How effective are zero-knowledge proofs for privacy preservation?",
        "What are the scalability limitations of blockchain networks?",
        "Does decentralized identity improve authentication security?",
        "How effective are smart contracts for healthcare data sharing?",
        "What are the security risks of cross-chain interoperability?",
        "Does blockchain improve trust in electronic voting systems?",

        # EMERGING TECHNOLOGIES (93–100)
        "How effective are quantum support vector machines for classification tasks?",
        "Does quantum machine learning outperform classical machine learning?",
        "What are the applications of digital twins in industrial manufacturing?",
        "How effective is edge AI for real-time analytics?",
        "Does neuromorphic computing improve energy efficiency?",
        "What are the safety challenges of autonomous drones?",
        "How effective are intelligent tutoring systems in higher education?",
        "Does generative AI-assisted tutoring improve student engagement and learning outcomes?"
    ]

    print("=" * 115)
    print(f" 🚀 EXECUTING 100-QUESTION MULTI-DOMAIN PROVENANCE AUDIT SWEEP")
    print("=" * 115)
    print(f"{'ID':<5} | {'STRESS-TEST TARGET QUESTION CONTEXT':<80} | {'STATUS'}")
    print("-" * 115)

    success_count = 0
    failure_count = 0
    term_telemetry_log = []
    
    # Cross-tabulation Provenance Matrix
    provenance_registry = {
        "extractor.py": {"comparator_leak": 0, "canonical_realization": 0, "fabricated_term": 0, "related_concept": 0, "metric_inflation": 0, "semantic_generalization": 0, "topic_brainstorming": 0, "pending_manual_audit": 0, "total": 0},
        "generator.py": {"comparator_leak": 0, "canonical_realization": 0, "fabricated_term": 0, "related_concept": 0, "metric_inflation": 0, "semantic_generalization": 0, "topic_brainstorming": 0, "pending_manual_audit": 0, "total": 0},
        "registries.py": {"comparator_leak": 0, "canonical_realization": 0, "fabricated_term": 0, "related_concept": 0, "metric_inflation": 0, "semantic_generalization": 0, "topic_brainstorming": 0, "pending_manual_audit": 0, "total": 0},
        "ontology_expander.py": {"comparator_leak": 0, "canonical_realization": 0, "fabricated_term": 0, "related_concept": 0, "metric_inflation": 0, "semantic_generalization": 0, "topic_brainstorming": 0, "pending_manual_audit": 0, "total": 0}
    }

    for idx, rq in enumerate(stress_test_questions, 1):
        q_id = f"Q{idx}"
        try:
            # --- STAGE 1: EXTRACTOR ---
            s1 = extract_5_facets(local_client, LOCAL_MODEL, rq)
            s1_adapted = SLRQueryContext(
                technology=getattr(s1, "primary_paradigm", []),
                domain=getattr(s1, "domain_context", []),
                comparison=getattr(s1, "comparator_baseline", []),
                context=[],
                outcomes=getattr(s1, "outcome_variables", [])
            )
            snapshot_extractor = get_context_snapshot(s1_adapted)
            
            # --- STAGE 2: GENERATOR ---
            s2 = expand_base_synonyms(local_client, LOCAL_MODEL, s1_adapted)
            snapshot_generator_all = get_context_snapshot(s2)
            snapshot_generator_new = snapshot_generator_all - snapshot_extractor
            
            # --- STAGE 3: REGISTRIES / HYDRATOR ---
            s3 = expand_acronym_layer(s2)
            primary_domain = classify_extracted_context(s3)
            s3_hydrated = inject_implicit_academic_layers(s3, primary_domain)
            snapshot_registries_all = get_context_snapshot(s3_hydrated)
            snapshot_registries_new = snapshot_registries_all - snapshot_generator_all
            
            # --- STAGE 4: ONTOLOGY & COMPARATOR EXPANDERS ---
            s4 = expand_ontology_layer(s3_hydrated, primary_domain)
            s4_compared = expand_comparator_registry(s4)
            snapshot_expanders_all = get_context_snapshot(s4_compared)
            snapshot_expanders_new = snapshot_expanders_all - snapshot_registries_all
            
            # --- FINAL SELECTION: SIEVE ---
            s5 = run_validation_sieve(s4_compared)
            final_query = compile_boolean_query(s5)
            
            # Evaluate Lineage Mapping on terms surviving the validation layer
            for layer_name, layer_array in [("technology", s5.technology), ("domain", s5.domain), ("comparison", s5.comparison), ("context", s5.context), ("outcomes", s5.outcomes)]:
                for phrase in layer_array:
                    clean_phrase = phrase.strip().lower()
                    label = classify_evaluated_term(phrase, GLOBAL_FORBIDDEN_COMPARATORS)
                    
                    # Compute Earliest Point of Origin (Birthplace)
                    if clean_phrase in snapshot_extractor:
                        source = "extractor.py"
                    elif clean_phrase in snapshot_generator_new:
                        source = "generator.py"
                    elif clean_phrase in snapshot_registries_new:
                        source = "registries.py"
                    elif clean_phrase in snapshot_expanders_new:
                        source = "ontology_expander.py"
                    else:
                        source = "generator.py" # Fallback mapping safety ring
                        
                    # Attribute to tracking structures
                    provenance_registry[source][label] += 1
                    provenance_registry[source]["total"] += 1
                    
                    term_telemetry_log.append({
                        "question_id": q_id,
                        "term": phrase,
                        "classification": label,
                        "origin_source": source
                    })

            print(f"{q_id:<5} | {rq[:78]:<80} | ✅ COMPILED")
            success_count += 1
        except Exception as e:
            print(f"{q_id:<5} | {rq[:78]:<80} | ❌ FAILED -> {str(e)}")
            failure_count += 1

    with open("term_telemetry.json", "w") as f:
        json.dump(term_telemetry_log, f, indent=2)

    # Render Multi-Dimensional Provenance Matrix Dashboard
    print("\n" + "=" * 115)
    print(" 📊 STAGE PROVENANCE CONTRIBUTION REPORT")
    print("=" * 115)
    print(f" {'CODE FILE STAGE':<22} | {'METRIC_INF':<10} | {'SEM_GEN':<10} | {'REL_CON':<10} | {'COMP_LEAK':<10} | {'PEND_AUD':<10} || {'TOTAL':<6}")
    print("-" * 115)
    for stage, counts in provenance_registry.items():
        print(f" ↳ {stage:<19} | {counts['metric_inflation']:>10} | {counts['semantic_generalization']:>10} | {counts['related_concept']:>10} | {counts['comparator_leak']:>10} | {counts['pending_manual_audit']:>10} || {counts['total']:>6}")
    print("=" * 115)
    print("💾 Full provenance matrix dumped to term_telemetry.json\n")

if __name__ == "__main__":
    run_stress_test_benchmark()