from __future__ import annotations

import os
import traceback

import pandas as pd

from bulk_screen import process_paper


RQ = "Can large language models and artificial intelligence tools help automate systematic literature reviews?"
TITLE = "Assessing Probability Rating Performance for Large Language Models in Systematic Literature Review Automation"
ABSTRACT = (
    "This paper evaluates large language models for automating title and abstract "
    "screening, study selection, and reviewer workload reduction in systematic reviews."
)


class FakeEngine:
    engine_id = "local"

    def ask(self, prompt, model=""):
        return """
        {
          "primary_subject": "LLM screening for SLR automation",
          "intervention_or_method": "large language models",
          "target_problem_or_task": "title abstract screening for systematic reviews",
          "application_context": "systematic literature reviews",
          "evidence_type": "evaluation",
          "review_role": "screening",
          "source_title": "Assessing Probability Rating Performance for Large Language Models in Systematic Literature Review Automation",
          "source_abstract": "This paper evaluates LLMs for SLR automation."
        }
        """


def main() -> int:
    os.environ.setdefault("MODEL_JUDGE_MODE", "balanced")
    os.environ.setdefault("ENABLE_MODEL_JUDGES", "true")
    os.environ.setdefault("ENABLE_HF_MODEL_LOADING", "false")
    rq_frame = {
        "rq_text": RQ,
        "review_question_type": "review_workflow_automation",
        "question_type": "review_workflow_automation",
        "method_or_technology": "artificial intelligence; large language models",
        "intervention_or_method": "artificial intelligence; large language models",
        "target_tasks_or_outcomes": "automate systematic literature reviews",
        "target_problem_or_task": "automate systematic literature reviews",
        "application_context": "systematic literature reviews",
        "required_dimensions": "method_tool; review_workflow_process; review_context",
        "rq_desired_relation": "tool_used_for_workflow",
    }
    row = pd.Series({"Title": TITLE, "Abstract": ABSTRACT})
    try:
        result = process_paper(
            row,
            "Title",
            "Abstract",
            RQ,
            rq_frame,
            "local",
            "qwen2.5:3b",
            inference_engine=FakeEngine(),
        )
    except Exception:
        traceback.print_exc()
        return 1

    for key in (
        "decision",
        "reason",
        "stage1_error_type",
        "model_judges_enabled",
        "model_judge_mode",
        "model_fusion_action",
    ):
        print(f"{key}={result.get(key, '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
