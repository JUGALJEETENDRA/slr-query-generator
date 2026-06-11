# ontology_expander.py
import re
from schema import SLRQueryContext

ONTOLOGY_REGISTRY_PACKS = {
    "SOFTWARE_ENGINEERING": {
        "mutation testing": { "technology": ["mutation analysis*", "program mutation*", "fault injection testing*"], "outcomes": ["fault detection efficiency*", "mutation score*", "test suite strength*"] },
        "service virtualization": { "technology": ["api simulation*", "mocking frameworks*", "environment virtualization*"], "outcomes": ["integration testing efficiency*", "test environment availability*", "dependency mocking*"] }
    },
    "DEVOPS": {
        "ci/cd": { "technology": ["devops*", "gitops*", "devsecops*"], "outcomes": ["deployment frequency*", "release cadence*", "lead time*", "throughput*", "software delivery*", "deployment performance*", "software delivery performance*"] },
        "infrastructure as code": { "technology": ["iac*", "terraform*", "ansible*", "cloudformation*"], "outcomes": ["configuration drift*", "resource tracking*", "deployment consistency*"] },
        "automated regression testing": { "technology": ["test automation*", "continuous testing*", "ci pipeline testing*"], "outcomes": ["release velocity*", "defect detection rate*", "build stability*"] },
        "gitops": { "technology": ["declarative deployment*", "argocd*", "flux cd*"], "comparison": ["push-based deployment*", "traditional cd pipelines*"] },
        "devsecops": { "technology": ["sast*", "dast*", "dependency scanning*", "secret detection*"], "outcomes": ["security bottlenecks*", "vulnerability remediation time*", "pipeline latency*"] },
        "chaos engineering": { "technology": ["fault injection*", "chaos mesh*", "gremlin*"], "outcomes": ["system availability*", "fault tolerance*", "resilience verification*"] }
    },
    "CLOUD": {
        "docker": { "technology": ["containers", "containerization*", "kubernetes*", "k8s*"] },
        "microservices": { "technology": ["microservice architecture*", "cloud-native applications*"] },
        "virtual machines": { "comparison": ["vms*", "hypervisor*", "hardware virtualization*", "virtualization*", "virtual machine monitor*", "vmm*"] },
        "service mesh": { "technology": ["istio*", "linkerd*", "envoy proxy*"] },
        "dynamic auto-scaling": { "technology": ["hpa*", "horizontal pod autoscaler*", "predictive scaling*"] },
        "event sourcing": { "technology": ["cqrs*", "kafka*", "rabbitmq*", "ordered log*"] },
        "edge computing": { "technology": ["fog computing*", "multi-access edge*", "mec*"], "comparison": ["centralized cloud storage*", "remote datacenters*"] }
    },
    "ROBOTICS": {
        "sensor fusion": { "technology": ["multi-sensor fusion*", "lidar*", "radar*", "camera*", "sensor integration*"] },
        "multi-object tracking": { "technology": ["mot*", "kalman filtering*", "deep sort*"], "context": ["adverse weather conditions*", "low visibility*", "rain*", "fog*"] },
        "real-time inference latency": { "technology": ["edge tpu*", "jetson nano*", "hardware accelerators*"], "context": ["unmanned aerial vehicles*", "drones*"] },
        "simultaneous localization and mapping": { "technology": ["slam*", "lidar slam*", "visual odometry*"], "context": ["gps-denied underground environments*", "subterranean tracking*"] }
    },
    "CYBERSECURITY": {
        "rule-based signature systems": { "comparison": ["signature-based detection*", "rule-based detection*", "legacy intrusion detection*", "signature-based intrusion detection*", "ids*"] },
        "siem log anomaly detection": { "technology": ["unsupervised machine learning*", "log parsing*", "isolation forest*"] }
    },
    "AI_ML": {
        "retrieval-augmented generation": { "technology": ["rag*", "vector search*", "dense retrieval*"] },
        "graph neural networks": { "technology": ["gnn*", "gcn*", "gat*", "graph embeddings*"] },
        "federated learning": { "technology": ["decentralized machine learning*", "privacy-preserving ml*", "secure aggregation*"] },
        "reinforcement learning": { "technology": ["deep q-networks*", "dqn*", "ppo*", "policy gradient*"] },
        "synthetic data": { "technology": ["generative adversarial networks*", "gan*", "diffusion models*", "synthetic datasets*", "data generation*"] }
    },
    "COMPUTER_VISION": {
        "vision transformers": { "technology": ["vit*", "self-attention networks*"] },
        "semantic segmentation": { "technology": ["synthetic data training*", "domain adaptation*", "sim-to-real transfer*"], "context": ["automated driving simulators*", "carla simulator*"] }
    },
    "EMERGING_TECH": {
        "twin": { "technology": ["digital twin*", "virtual replica*", "digital thread*", "predictive maintenance*", "asset lifecycle*", "condition monitoring*"] },
        "manufacturing": { "technology": ["digital twin*", "virtual replica*", "digital thread*", "predictive maintenance*", "asset lifecycle*", "condition monitoring*"] },
        "iot": { "technology": ["digital twin*", "virtual replica*", "predictive maintenance*", "asset lifecycle*", "condition monitoring*"] },
        "scada": { "technology": ["digital twin*", "virtual replica*", "predictive maintenance*", "asset lifecycle*", "condition monitoring*"] },
        "mes": { "technology": ["digital twin*", "virtual replica*", "predictive maintenance*", "asset lifecycle*", "condition monitoring*"] }
    }
}

def expand_ontology_layer(current_context: SLRQueryContext, primary_domain: str) -> SLRQueryContext:
    """Executes single-pass isolated vocabulary lookups against a frozen data matrix."""
    input_snapshot = {
        "technology": list(current_context.technology),
        "domain": list(current_context.domain),
        "comparison": list(current_context.comparison),
        "context": list(current_context.context),
        "outcomes": list(current_context.outcomes)
    }
    
    output_pools = {k: list(v) for k, v in input_snapshot.items()}
    allowed_pack = ONTOLOGY_REGISTRY_PACKS.get(primary_domain, {})
    
    for current_facet, term_list in input_snapshot.items():
        for term in term_list:
            normalized_term = term.lower().strip().replace("*", "")
            for anchor_key, target_routing in allowed_pack.items():
                if re.search(r'\b' + re.escape(anchor_key) + r'\b', normalized_term):
                    for destination_facet, expansion_tokens in target_routing.items():
                        
                        # 🔬 THE STRUCTURAL FACET BOUNDARY GUARD
                        # Restricts relational graph expansions to stay within the original source facet.
                        # This blocks technology keywords from cross-pollinating and inflating uninvited KPI arrays.
                        if destination_facet == current_facet:
                            output_pools[destination_facet].extend(expansion_tokens)
                                    
    return SLRQueryContext(
        technology=list(dict.fromkeys(output_pools["technology"])),
        domain=list(dict.fromkeys(output_pools["domain"])),
        comparison=list(dict.fromkeys(output_pools["comparison"])),
        context=list(dict.fromkeys(output_pools["context"])),
        outcomes=list(dict.fromkeys(output_pools["outcomes"]))
    )