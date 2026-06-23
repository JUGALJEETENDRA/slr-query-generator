import json
from semantic_frame import extract_semantic_frame


PAPERS = [
    {
        "title": "Leveraging Large Language Models for Systematic Reviews",
        "abstract": (
            "Systematic reviews require substantial manual effort to screen titles, "
            "abstracts, and full texts. This paper investigates the use of large "
            "language models to assist systematic review workflows, including study "
            "selection and evidence extraction. The study evaluates model outputs "
            "against human reviewer decisions and discusses implications for "
            "review efficiency and reliability."
        ),
    },
    {
        "title": "From Pixels to Paragraphs: A Survey of Large Language Models in Image Captioning",
        "abstract": (
            "Image captioning has advanced through the integration of vision encoders "
            "and large language models. This systematic review surveys recent methods "
            "that generate natural-language descriptions from images, compares model "
            "architectures, datasets, and evaluation metrics, and identifies open "
            "challenges in multimodal caption generation."
        ),
    },
    {
        "title": "Using ChatGPT to Support Title and Abstract Screening in Evidence Synthesis",
        "abstract": (
            "Title and abstract screening is a time-consuming step in evidence "
            "synthesis. We assess whether ChatGPT can prioritize or classify records "
            "during systematic review screening. Model predictions are compared with "
            "human inclusion decisions across multiple review topics, with attention "
            "to sensitivity, specificity, and workload reduction."
        ),
    },
    {
        "title": "Large Language Models in Healthcare: A Systematic Review",
        "abstract": (
            "Large language models have been applied to clinical documentation, "
            "question answering, patient communication, and decision support. This "
            "systematic review summarizes healthcare applications, reported benefits, "
            "limitations, safety concerns, and evaluation practices across published "
            "studies."
        ),
    },
    {
        "title": "Automating Data Extraction for Systematic Reviews with Large Language Models",
        "abstract": (
            "Data extraction from included studies is a major bottleneck in systematic "
            "reviews. This paper evaluates large language models as tools for extracting "
            "intervention, population, outcome, and study design information from "
            "research abstracts and full texts. Results are compared against reviewer "
            "curated extraction forms."
        ),
    },
]


def main():
    for index, paper in enumerate(PAPERS, start=1):
        frame = extract_semantic_frame(
            paper["title"],
            paper["abstract"],
        )

        print(f"\nPaper {index}: {paper['title']}")
        print(json.dumps(frame, indent=2))


if __name__ == "__main__":
    main()
