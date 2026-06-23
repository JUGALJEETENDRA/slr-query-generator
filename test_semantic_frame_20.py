import json
from semantic_frame import extract_semantic_frame


PAPERS = [
    {
        "title": "Using Large Language Models to Automate Abstract Screening in Systematic Reviews",
        "abstract": (
            "Abstract screening is a labor-intensive stage of systematic reviews. "
            "This study evaluates large language models for classifying candidate "
            "records as potentially relevant or irrelevant. Model classifications "
            "are compared with independent human reviewer decisions across multiple "
            "review topics."
        ),
        "expected": {
            "intervention_or_method": "large language models",
            "target_problem_or_task": "abstract screening in systematic reviews",
            "study_role": "empirical evaluation",
        },
    },
    {
        "title": "ChatGPT for Title Screening in Evidence Synthesis",
        "abstract": (
            "Title screening is commonly performed by reviewers during evidence "
            "synthesis. We assess ChatGPT as a tool for ranking and classifying "
            "titles before manual review. Performance is evaluated against reviewer "
            "inclusion labels and workload reduction is estimated."
        ),
        "expected": {
            "intervention_or_method": "ChatGPT",
            "target_problem_or_task": "title screening in evidence synthesis",
            "study_role": "empirical evaluation",
        },
    },
    {
        "title": "Automated Data Extraction from Clinical Trial Reports Using Large Language Models",
        "abstract": (
            "Systematic reviews require extraction of population, intervention, "
            "comparator, outcome, and study design details from included studies. "
            "This paper evaluates large language models for extracting structured "
            "data from clinical trial reports and compares outputs with human "
            "reviewer extraction forms."
        ),
        "expected": {
            "intervention_or_method": "large language models",
            "target_problem_or_task": "data extraction for systematic reviews",
            "study_role": "empirical evaluation",
        },
    },
    {
        "title": "A Local LLM Assistant for Systematic Review Study Selection",
        "abstract": (
            "We present a local large language model assistant designed to support "
            "study selection in systematic reviews. The assistant summarizes titles "
            "and abstracts, proposes inclusion rationales, and flags uncertain "
            "records for reviewer attention. A pilot evaluation compares assistant "
            "outputs with decisions made by domain experts."
        ),
        "expected": {
            "intervention_or_method": "local large language model assistant",
            "target_problem_or_task": "study selection in systematic reviews",
            "study_role": "tool/method paper",
        },
    },
    {
        "title": "Large Language Models for Deduplicating Records in Systematic Review Searches",
        "abstract": (
            "Bibliographic database searches often produce duplicate records that "
            "must be removed before screening. This study investigates large "
            "language models for identifying duplicate citations using titles, "
            "authors, abstracts, and metadata. Results are compared with curator "
            "verified duplicate sets."
        ),
        "expected": {
            "intervention_or_method": "large language models",
            "target_problem_or_task": "deduplicating records in systematic review searches",
            "study_role": "empirical evaluation",
        },
    },
    {
        "title": "LLM-Based Risk of Bias Assessment for Evidence Reviews",
        "abstract": (
            "Risk of bias assessment is a structured but time-consuming part of "
            "evidence reviews. We evaluate large language models for completing "
            "risk of bias judgments from study reports and compare generated "
            "judgments with trained reviewer assessments."
        ),
        "expected": {
            "intervention_or_method": "large language models",
            "target_problem_or_task": "risk of bias assessment for evidence reviews",
            "study_role": "empirical evaluation",
        },
    },
    {
        "title": "Generating Search Strategies for Systematic Reviews with Large Language Models",
        "abstract": (
            "Developing database search strategies requires translating research "
            "questions into Boolean queries and controlled vocabulary terms. This "
            "paper studies large language models for drafting systematic review "
            "search strategies and compares generated queries with librarian-authored "
            "strategies."
        ),
        "expected": {
            "intervention_or_method": "large language models",
            "target_problem_or_task": "generating search strategies for systematic reviews",
            "study_role": "empirical evaluation",
        },
    },
    {
        "title": "Screening Biomedical Literature Reviews with Instruction-Tuned Language Models",
        "abstract": (
            "This paper evaluates instruction-tuned language models for title and "
            "abstract screening in biomedical systematic reviews. The models receive "
            "review criteria and citation metadata, then produce inclusion labels "
            "that are compared against dual human screening decisions."
        ),
        "expected": {
            "intervention_or_method": "instruction-tuned language models",
            "target_problem_or_task": "title and abstract screening in biomedical systematic reviews",
            "study_role": "empirical evaluation",
        },
    },
    {
        "title": "Extracting PICO Elements for Systematic Reviews Using Generative Language Models",
        "abstract": (
            "PICO extraction supports evidence synthesis by identifying populations, "
            "interventions, comparators, and outcomes from clinical studies. We test "
            "generative language models on PICO extraction for systematic review "
            "workflows and compare extracted elements with expert annotations."
        ),
        "expected": {
            "intervention_or_method": "generative language models",
            "target_problem_or_task": "PICO extraction for systematic reviews",
            "study_role": "empirical evaluation",
        },
    },
    {
        "title": "A Framework for LLM-Assisted Full-Text Screening in Systematic Reviews",
        "abstract": (
            "Full-text screening requires reviewers to apply eligibility criteria "
            "to complete articles. We propose a framework that uses large language "
            "models to summarize eligibility-relevant passages and suggest screening "
            "decisions. The framework is demonstrated on several completed systematic "
            "reviews."
        ),
        "expected": {
            "intervention_or_method": "large language models",
            "target_problem_or_task": "full-text screening in systematic reviews",
            "study_role": "framework/method proposal",
        },
    },
    {
        "title": "Large Language Models in Radiology: A Systematic Review",
        "abstract": (
            "Large language models are increasingly studied for radiology report "
            "generation, clinical question answering, workflow support, and education. "
            "This systematic review summarizes published radiology applications, "
            "evaluation methods, reported benefits, limitations, and safety concerns."
        ),
        "expected": {
            "intervention_or_method": "large language models",
            "target_problem_or_task": "radiology applications",
            "study_role": "systematic review",
        },
    },
    {
        "title": "Large Language Models for Medical Education: A Scoping Review",
        "abstract": (
            "This scoping review maps the use of large language models in medical "
            "education, including tutoring, assessment, feedback generation, and "
            "curriculum design. It characterizes study designs, educational settings, "
            "outcomes, and gaps in the literature."
        ),
        "expected": {
            "intervention_or_method": "large language models",
            "target_problem_or_task": "medical education applications",
            "study_role": "scoping review",
        },
    },
    {
        "title": "Applications of ChatGPT in Software Engineering: A Systematic Literature Review",
        "abstract": (
            "ChatGPT has been applied to code generation, debugging, testing, "
            "requirements analysis, and documentation. This systematic literature "
            "review synthesizes empirical studies on ChatGPT in software engineering "
            "and reports observed capabilities, limitations, and research gaps."
        ),
        "expected": {
            "intervention_or_method": "ChatGPT",
            "target_problem_or_task": "software engineering applications",
            "study_role": "systematic literature review",
        },
    },
    {
        "title": "A Systematic Review of Large Language Models for Legal Question Answering",
        "abstract": (
            "Legal question answering systems increasingly use large language models "
            "to retrieve, reason over, and generate responses from legal texts. This "
            "systematic review analyzes model architectures, datasets, evaluation "
            "metrics, and unresolved challenges in legal question answering."
        ),
        "expected": {
            "intervention_or_method": "large language models",
            "target_problem_or_task": "legal question answering",
            "study_role": "systematic review",
        },
    },
    {
        "title": "Large Language Models for Code Generation: A Survey",
        "abstract": (
            "This survey reviews large language models for code generation, including "
            "training data, model architectures, benchmarks, prompting strategies, "
            "and evaluation metrics. It discusses strengths and weaknesses of current "
            "models and identifies open research directions."
        ),
        "expected": {
            "intervention_or_method": "large language models",
            "target_problem_or_task": "code generation",
            "study_role": "survey",
        },
    },
    {
        "title": "Multimodal Large Language Models for Image Captioning: A Systematic Review",
        "abstract": (
            "Image captioning systems generate textual descriptions for visual inputs. "
            "This systematic review examines multimodal large language models for "
            "image captioning, comparing datasets, architectures, metrics, and "
            "remaining limitations."
        ),
        "expected": {
            "intervention_or_method": "multimodal large language models",
            "target_problem_or_task": "image captioning",
            "study_role": "systematic review",
        },
    },
    {
        "title": "Chatbots Powered by Large Language Models in Mental Health: A Review",
        "abstract": (
            "Large language model chatbots are being explored for mental health "
            "support, psychoeducation, triage, and conversational interventions. "
            "This review summarizes existing systems, evaluation practices, ethical "
            "concerns, and clinical safety issues."
        ),
        "expected": {
            "intervention_or_method": "large language model chatbots",
            "target_problem_or_task": "mental health support",
            "study_role": "review",
        },
    },
    {
        "title": "Large Language Models in Finance: A Systematic Mapping Study",
        "abstract": (
            "This mapping study categorizes applications of large language models "
            "in finance, including sentiment analysis, report generation, fraud "
            "detection, risk assessment, and financial advising. It summarizes "
            "datasets, tasks, evaluation metrics, and publication trends."
        ),
        "expected": {
            "intervention_or_method": "large language models",
            "target_problem_or_task": "finance applications",
            "study_role": "systematic mapping study",
        },
    },
    {
        "title": "The Use of Large Language Models in Education: A Systematic Review",
        "abstract": (
            "This systematic review examines how large language models are used in "
            "educational settings, including writing support, tutoring, feedback, "
            "assessment, and teacher assistance. It summarizes empirical evidence, "
            "pedagogical opportunities, and risks."
        ),
        "expected": {
            "intervention_or_method": "large language models",
            "target_problem_or_task": "education applications",
            "study_role": "systematic review",
        },
    },
    {
        "title": "Large Language Models for Drug Discovery: A Review of Methods and Applications",
        "abstract": (
            "Large language models have been adapted for molecule generation, target "
            "identification, protein sequence modeling, literature mining, and clinical "
            "trial design. This review surveys methods and applications in drug "
            "discovery and discusses evaluation practices and open challenges."
        ),
        "expected": {
            "intervention_or_method": "large language models",
            "target_problem_or_task": "drug discovery applications",
            "study_role": "review",
        },
    },
]


def _norm(value):
    return " ".join(value.lower().replace("-", " ").replace("/", " ").split())


def _contains_expected(actual, expected):
    actual = _norm(actual)
    expected = _norm(expected)
    expected_terms = [term for term in expected.split() if len(term) > 2]

    if not actual or not expected_terms:
        return False

    return sum(1 for term in expected_terms if term in actual) / len(expected_terms) >= 0.5


def main():
    totals = {
        "intervention_or_method": 0,
        "target_problem_or_task": 0,
        "study_role": 0,
    }

    for paper in PAPERS:
        frame = extract_semantic_frame(
            paper["title"],
            paper["abstract"],
        )

        print(f"TITLE: {paper['title']}")
        print()
        print("FRAME:")
        print(json.dumps(frame, indent=2))
        print()

        for field in totals:
            if _contains_expected(frame[field], paper["expected"][field]):
                totals[field] += 1

    print("TOTALS:")
    print(f"intervention_or_method clearly correct: {totals['intervention_or_method']}/20")
    print(f"target_problem_or_task clearly correct: {totals['target_problem_or_task']}/20")
    print(f"study_role clearly correct: {totals['study_role']}/20")


if __name__ == "__main__":
    main()
