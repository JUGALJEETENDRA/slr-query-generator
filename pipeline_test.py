# pipeline_test.py
from openai import OpenAI
import instructor
from extractor import extract_5_facets
from generator import expand_base_synonyms
from acronym_expander import expand_acronym_layer
from ontology_expander import expand_ontology_layer
from validator import run_validation_sieve
from compiler import compile_boolean_query

# PHASE 4 DETERMINISTIC COMPILER FIREWALLS
from classifier import classify_extracted_context
from registries import inject_implicit_academic_layers
from comparator_registry import expand_comparator_registry

def run_stress_test_benchmark():
    # Connect instructor execution client to local inference node
    local_client = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1", 
            api_key="ollama-local"
        ),
        mode=instructor.Mode.JSON
    )
    LOCAL_MODEL = "qwen2.5:7b"

    # Fully Integrated 100-Question Stress-Test Dataset
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
    print(f" 🚀 EXECUTING 100-QUESTION MULTI-DOMAIN STRESS TEST (PHASE 4 HARDENED)")
    print("=" * 115)
    print(f"{'ID':<5} | {'STRESS-TEST TARGET QUESTION CONTEXT':<80} | {'STATUS'}")
    print("-" * 115)

    compiled_queries = {}
    success_count = 0
    failure_count = 0
    total_facets_populated = 0

    for idx, rq in enumerate(stress_test_questions, 1):
        q_id = f"Q{idx}"
        try:
            # 1. Run structured LLM facet extraction
            s1 = extract_5_facets(local_client, LOCAL_MODEL, rq)
            
            # 2. Generate synonym variations via LLM thesaural pass
            s2 = expand_base_synonyms(local_client, LOCAL_MODEL, s1)
            
            # 3. Clean and resolve basic acronym tokens
            s3 = expand_acronym_layer(s2)
            
            # [PHASE 4 COMPLIANCE]: Calculate a single, strict primary domain track
            primary_domain = classify_extracted_context(s3)
            
            # [PHASE 4 COMPLIANCE]: Hydrate context and self-scoring metrics locked to that track
            s3_hydrated = inject_implicit_academic_layers(s3, primary_domain)
            
            # 4. Map expansion layers against isolated snapshots filtered by the primary track
            s4 = expand_ontology_layer(s3_hydrated, primary_domain)
            
            # [PHASE 4 COMPLIANCE]: Run separate, independent comparator duality expansions
            s4_compared = expand_comparator_registry(s4)
            
            # 5. Filter out out-of-scope phrases and apply negative ontology masks
            s5 = run_validation_sieve(s4_compared)
            
            # 6. Build the final clean search query string
            final_query = compile_boolean_query(s5)
            
            # Record facet data density for reporting metrics
            active_facets = sum(1 for array in [s5.technology, s5.domain, s5.comparison, s5.context, s5.outcomes] if array)
            total_facets_populated += active_facets
            
            compiled_queries[q_id] = (rq, final_query, active_facets)
            print(f"{q_id:<5} | {rq[:78]:<80} | ✅ COMPILED")
            success_count += 1
            
        except Exception as e:
            print(f"{q_id:<5} | {rq[:78]:<80} | ❌ FAILED ({str(e)[:12]})")
            failure_count += 1

    print("\n" + "=" * 115)
    print(" 🏁 MASTER FIREWALLED COMPILATION LOGS REPORT")
    print("=" * 115)
    
    for q_id, (original_rq, boolean_query, facets) in compiled_queries.items():
        print(f"\n[{q_id}] RQ: '{original_rq}' [Active Facets: {facets}/5]")
        print(f" └─ Compiled String:\n    {boolean_query}")
        print("-" * 115)

    # Automated Macro Data Analytics Summary
    print("\n" + "=" * 115)
    print(" 📊 METRIC PIPELINE COVERAGE PERFORMANCE SUMMARY")
    print("=" * 115)
    print(f"  ● Total Input Stress Tests Processed : {len(stress_test_questions)}")
    print(f"  ● Successful Compilations           : {success_count} / {len(stress_test_questions)} ({(success_count/len(stress_test_questions))*100:.1f}%)")
    print(f"  ● Execution Failure Drops            : {failure_count}")
    if success_count > 0:
        print(f"  ● Average Structural Facet Density  : {total_facets_populated / success_count:.2f} / 5.00")
    print("=" * 115)

if __name__ == "__main__":
    run_stress_test_benchmark()