import argparse
import os
import time
import pandas as pd
from corpus_profiler import profile_corpus
from semantic_frame import (
    enrich_research_question_frame_with_corpus,
    extract_research_question_frame,
    extract_semantic_frame,
    initialize_semantic_frame_cache,
)
from screening_strategies import (
    DEFAULT_SCREENING_STRATEGY,
    normalize_screening_strategy,
    screen_candidate,
    strategy_requires_rq_frame,
)
from config import (
    DEFAULT_MODEL,
    DEV_SCREENING_ROW_LIMIT,
    TWO_STAGE_SCREENING_ENABLED,
    FIRST_STAGE_MODEL,
    SECOND_STAGE_MODEL,
    LOCAL_CHECKPOINT_INTERVAL,
)
from gemini_web_automation import GeminiWebConfig
from processing_engines import (
    GEMINI_WEB_ENGINE,
    normalize_processing_engine,
    resolve_processing_engine,
)
from screening_contracts import build_rq_contract
from stage2_arbitration import arbitrate_stage2
from final_adjudicator import FINAL_ADJUDICATION_FIELDS, adjudicate_row
from runtime_config import (
    get_model_judge_config,
    model_config_csv_fields,
    print_model_judge_config,
)
from llm_structured_judge import clear_cache as clear_llm_cache, get_cache_stats
from performance_profiler import PerformanceProfiler
from semantic_frame import get_semantic_frame_cache_stats

from threading import Lock
import uuid


# ---------- Thread-safe progress state for UI feedback ----------
class ScreeningProgress:
    def __init__(self):
        self._lock = Lock()
        self._state = self._idle_state()
        self._started_at = None

    @staticmethod
    def _idle_state():
        return {
            "status": "idle",
            "phase": "idle",
            "job_id": None,
            "current": 0,
            "total": 0,
            "stage2_current": 0,
            "stage2_total": 0,
            "keep": 0,
            "maybe": 0,
            "reject": 0,
            "error": None,
            "runtime_seconds": None,
            "remaining": 0,
            "estimated_remaining_seconds": None,
        }

    def start_job(self, job_id):
        with self._lock:
            if self._state["status"] == "starting":
                if self._state["job_id"] == job_id:
                    return True
                return False
            if self._state["status"] == "running":
                return False

            self._state = self._idle_state()
            self._state.update({
                "status": "starting",
                "job_id": job_id,
            })
            return True

    def begin_screening(self, job_id, total):
        with self._lock:
            self._assert_active_job(job_id)
            self._state.update({
                "status": "running",
                "phase": "stage1",
                "current": 0,
                "total": int(total),
                "stage2_current": 0,
                "stage2_total": 0,
                "keep": 0,
                "maybe": 0,
                "reject": 0,
                "error": None,
                "runtime_seconds": 0.0,
            })
            self._started_at = time.perf_counter()

    def update_counts(self, job_id, current, keep, maybe, reject):
        with self._lock:
            self._assert_active_job(job_id)
            if current < self._state["current"]:
                raise RuntimeError(
                    f"Progress regression for job {job_id}: "
                    f"{current} < {self._state['current']}"
                )

            self._state.update({
                "status": "running",
                "current": int(current),
                "keep": int(keep),
                "maybe": int(maybe),
                "reject": int(reject),
            })
            self._update_timing_locked()

    def begin_stage2(self, job_id, total):
        with self._lock:
            self._assert_active_job(job_id)
            self._state.update({
                "status": "running",
                "phase": "stage2",
                "stage2_current": 0,
                "stage2_total": int(total),
            })

    def update_stage2(self, job_id, current):
        with self._lock:
            self._assert_active_job(job_id)
            if current < self._state["stage2_current"]:
                raise RuntimeError(
                    f"Stage 2 progress regression for job {job_id}: "
                    f"{current} < {self._state['stage2_current']}"
                )
            self._state.update({
                "status": "running",
                "phase": "stage2",
                "stage2_current": int(current),
            })

    def finish(self, job_id):
        with self._lock:
            self._assert_active_job(job_id)
            self._state["status"] = "finished"
            self._state["phase"] = "finished"
            self._state["current"] = self._state["total"]
            if self._started_at is not None:
                self._state["runtime_seconds"] = round(
                    time.perf_counter() - self._started_at,
                    2,
                )
            self._state["remaining"] = 0
            self._state["estimated_remaining_seconds"] = 0.0

    def fail(self, job_id, message):
        with self._lock:
            self._assert_active_job(job_id)
            self._state["status"] = "error"
            self._state["phase"] = "error"
            self._state["error"] = str(message)
            if self._started_at is not None:
                self._state["runtime_seconds"] = round(
                    time.perf_counter() - self._started_at,
                    2,
                )
            self._state["remaining"] = max(
                0,
                int(self._state.get("total") or 0) - int(self._state.get("current") or 0),
            )

    def snapshot(self):
        with self._lock:
            state = dict(self._state)
            if state["status"] == "running" and self._started_at is not None:
                state["runtime_seconds"] = round(
                    time.perf_counter() - self._started_at,
                    2,
                )
                current = int(state.get("current") or 0)
                total = int(state.get("total") or 0)
                remaining = max(0, total - current)
                state["remaining"] = remaining
                if current:
                    state["estimated_remaining_seconds"] = round(
                        (state["runtime_seconds"] / current) * remaining,
                        2,
                    )
            return state

    def is_running(self):
        with self._lock:
            return self._state["status"] in {"starting", "running"}

    def _assert_active_job(self, job_id):
        if self._state["job_id"] != job_id:
            raise RuntimeError(
                f"Progress update rejected for inactive job {job_id}; "
                f"active job is {self._state['job_id']}"
            )

    def _update_timing_locked(self):
        current = int(self._state.get("current") or 0)
        total = int(self._state.get("total") or 0)
        remaining = max(0, total - current)
        self._state["remaining"] = remaining
        if self._started_at is None:
            return
        runtime = time.perf_counter() - self._started_at
        self._state["runtime_seconds"] = round(runtime, 2)
        self._state["estimated_remaining_seconds"] = (
            round((runtime / current) * remaining, 2)
            if current
            else None
        )


PROGRESS = ScreeningProgress()
# ----------------------------------------------------------------


class ScreeningSession:
    def __init__(self):
        self._lock = Lock()
        self._results = []
        self._finalized_files = {}

    def set_results(self, results):
        with self._lock:
            self._results = [dict(row) for row in results]
            self._finalized_files = {}

    def snapshot(self):
        with self._lock:
            return [dict(row) for row in self._results]

    def counts(self, results=None):
        rows = self.snapshot() if results is None else results
        return {
            "total": len(rows),
            "keep": sum(1 for row in rows if row.get("Decision") == "KEEP"),
            "maybe": sum(1 for row in rows if row.get("Decision") == "MAYBE"),
            "reject": sum(1 for row in rows if row.get("Decision") == "REJECT"),
        }

    def finalize(self, edited_results, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)

        with self._lock:
            if not self._results:
                raise RuntimeError("No screening results are available to finalize.")

            original_by_title = {
                str(row.get("Title", "")): dict(row)
                for row in self._results
            }

            finalized = []
            for edited in edited_results:
                title = str(edited.get("Title", ""))
                if not title:
                    continue

                row = original_by_title.get(title, dict(edited))
                row = dict(row)
                row["Decision"] = str(edited.get("Decision", row.get("Decision", ""))).upper()
                row["Reason"] = edited.get("Reason", row.get("Reason", ""))
                finalized.append(row)

            if not finalized:
                raise RuntimeError("No edited screening results were provided.")

            self._results = finalized

            screened_path = os.path.join(output_dir, "screened.csv")
            included_path = os.path.join(output_dir, "included_studies.csv")
            maybe_path = os.path.join(output_dir, "maybe_studies.csv")
            excluded_path = os.path.join(output_dir, "excluded_studies.csv")

            result_df = pd.DataFrame(finalized)
            result_df.to_csv(screened_path, index=False)

            self._write_category(result_df, "KEEP", included_path)
            self._write_category(result_df, "MAYBE", maybe_path)
            self._write_category(result_df, "REJECT", excluded_path)

            self._finalized_files = {
                "screened": screened_path,
                "included": included_path,
                "maybe": maybe_path,
                "excluded": excluded_path,
            }

            return {
                "counts": self.counts(finalized),
                "files": dict(self._finalized_files),
            }

    @staticmethod
    def _write_category(result_df, decision, path):
        category_df = result_df[result_df["Decision"] == decision]
        category_df.to_csv(path, index=False)


SCREENING_SESSION = ScreeningSession()
# ----------------------------------------------------------------


def _find_col(df, candidates):
    """Return the first column name from candidates that exists in df (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


# ---------- NEW: process_paper function (replaces per-paper logic) ----------
def process_paper(
    row,
    title_col,
    abstract_col,
    research_question,
    rq_frame,
    mode,
    model,
    semantic_strategy=DEFAULT_SCREENING_STRATEGY,
    inference_engine=None,
):
    title = row[title_col]
    abstract = str(row[abstract_col])

    result = screen_candidate(
        title=title,
        abstract=abstract,
        research_question=research_question,
        strategy=semantic_strategy,
        rq_frame=rq_frame,
        mode=mode,
        model=model,
        inference_engine=inference_engine,
    )

    result["title"] = title
    result["abstract"] = abstract

    return result
# ---------------------------------------------------------------------------


def _result_semantic_fields(result, prefix=""):
    return {
        f"{prefix}rq_primary_subject": result.get("rq_primary_subject", ""),
        f"{prefix}rq_intervention_or_method": result.get("rq_intervention_or_method", ""),
        f"{prefix}rq_target_problem_or_task": result.get("rq_target_problem_or_task", ""),
        f"{prefix}rq_application_context": result.get("rq_application_context", ""),
        f"{prefix}rq_evidence_type": result.get("rq_evidence_type", ""),
        f"{prefix}rq_study_role": result.get("rq_study_role", ""),
        f"{prefix}rq_review_role": result.get("rq_review_role", ""),
        f"{prefix}rq_question_type": result.get("rq_question_type", ""),
        f"{prefix}rq_frame_source": result.get("rq_frame_source", ""),
        f"{prefix}rq_frame_diagnostic": result.get("rq_frame_diagnostic", ""),
        f"{prefix}rq_review_question_type": result.get("rq_review_question_type", ""),
        f"{prefix}rq_core_domain": result.get("rq_core_domain", ""),
        f"{prefix}rq_method_or_technology": result.get("rq_method_or_technology", ""),
        f"{prefix}rq_method_family": result.get("rq_method_family", ""),
        f"{prefix}rq_target_outcomes": result.get("rq_target_tasks_or_outcomes", ""),
        f"{prefix}rq_inclusion_concepts": result.get("rq_required_inclusion_concepts", ""),
        f"{prefix}rq_exclusion_concepts": result.get("rq_exclusion_concepts", ""),
        f"{prefix}rq_domain_synonyms": result.get("rq_domain_synonyms", ""),
        f"{prefix}rq_method_synonyms": result.get("rq_method_synonyms", ""),
        f"{prefix}rq_task_synonyms": result.get("rq_task_outcome_synonyms", ""),
        f"{prefix}rq_required_dimensions": result.get("rq_required_dimensions", ""),
        f"{prefix}rq_minimum_inclusion_rule": result.get("rq_minimum_inclusion_rule", ""),
        f"{prefix}rq_extraction_suspect": result.get("rq_rq_extraction_suspect", ""),
        f"{prefix}rq_desired_relation": result.get("rq_rq_desired_relation", ""),
        f"{prefix}rq_type": result.get("rq_rq_type", ""),
        f"{prefix}rq_required_dimensions_contract": result.get("rq_rq_required_dimensions", ""),
        f"{prefix}rq_method_terms": result.get("rq_rq_method_terms", ""),
        f"{prefix}rq_task_terms": result.get("rq_rq_task_terms", ""),
        f"{prefix}rq_context_terms": result.get("rq_rq_context_terms", ""),
        f"{prefix}rq_outcome_terms": result.get("rq_rq_outcome_terms", ""),
        f"{prefix}corpus_profile_terms": result.get("rq_corpus_profile_terms", ""),
        f"{prefix}corpus_method_terms": result.get("rq_corpus_method_terms", ""),
        f"{prefix}corpus_task_terms": result.get("rq_corpus_task_terms", ""),
        f"{prefix}corpus_context_terms": result.get("rq_corpus_context_terms", ""),
        f"{prefix}corpus_evidence_terms": result.get("rq_corpus_evidence_terms", ""),
        f"{prefix}corpus_domain_specific_synonyms": result.get(
            "rq_corpus_domain_specific_synonyms",
            "",
        ),
        f"{prefix}corpus_review_context_terms": result.get("rq_corpus_review_context_terms", ""),
        f"{prefix}corpus_workflow_task_terms": result.get("rq_corpus_workflow_task_terms", ""),
        f"{prefix}corpus_ai_tool_terms": result.get("rq_corpus_ai_tool_terms", ""),
        f"{prefix}corpus_external_domain_terms": result.get("rq_corpus_external_domain_terms", ""),
        f"{prefix}corpus_technology_subject_terms": result.get(
            "rq_corpus_technology_subject_terms",
            "",
        ),
        f"{prefix}corpus_automation_intent_terms": result.get(
            "rq_corpus_automation_intent_terms",
            "",
        ),
        f"{prefix}corpus_review_workflow_terms": result.get("rq_corpus_review_workflow_terms", ""),
        f"{prefix}corpus_tool_use_terms": result.get("rq_corpus_tool_use_terms", ""),
        f"{prefix}corpus_subject_review_terms": result.get("rq_corpus_subject_review_terms", ""),
        f"{prefix}corpus_relation_clusters": result.get("rq_corpus_relation_clusters", ""),
        f"{prefix}paper_primary_subject": result.get("paper_primary_subject", ""),
        f"{prefix}paper_intervention_or_method": result.get("paper_intervention_or_method", ""),
        f"{prefix}paper_target_problem_or_task": result.get("paper_target_problem_or_task", ""),
        f"{prefix}paper_application_context": result.get("paper_application_context", ""),
        f"{prefix}paper_evidence_type": result.get("paper_evidence_type", ""),
        f"{prefix}paper_study_role": result.get("paper_study_role", ""),
        f"{prefix}paper_review_role": result.get("paper_review_role", ""),
        f"{prefix}error_type": result.get(f"{prefix}error_type", result.get("stage1_error_type", "")),
        f"{prefix}error_message": result.get(f"{prefix}error_message", result.get("stage1_error_message", "")),
        f"{prefix}error_trace_short": result.get(f"{prefix}error_trace_short", result.get("stage1_error_trace_short", "")),
        f"{prefix}model_judges_enabled": result.get("model_judges_enabled", False),
        f"{prefix}model_judge_mode": result.get("model_judge_mode", "off"),
        f"{prefix}model_judge_models_used": result.get("model_judge_models_used", ""),
        f"{prefix}model_judge_runtime_source": result.get("model_judge_runtime_source", ""),
        f"{prefix}model_profile": result.get("model_profile", ""),
        f"{prefix}model_real_models_loaded": result.get("model_real_models_loaded", False),
        f"{prefix}model_fallback_reason": result.get("model_fallback_reason", ""),
        f"{prefix}model_timing_seconds": result.get("model_timing_seconds", 0.0),
        f"{prefix}model_judge_fallback_used": result.get("model_judge_fallback_used", False),
        f"{prefix}model_judge_error": result.get("model_judge_error", ""),
        f"{prefix}model_reranker_relevance_score": result.get("model_reranker_relevance_score", 0.0),
        f"{prefix}model_reranker_negative_score": result.get("model_reranker_negative_score", 0.0),
        f"{prefix}model_reranker_positive_raw_score": result.get("model_reranker_positive_raw_score", 0.0),
        f"{prefix}model_reranker_negative_raw_score": result.get("model_reranker_negative_raw_score", 0.0),
        f"{prefix}model_reranker_margin": result.get("model_reranker_margin", 0.0),
        f"{prefix}model_reranker_decision_hint": result.get("model_reranker_decision_hint", ""),
        f"{prefix}reranker_runtime_source": result.get("reranker_runtime_source", ""),
        f"{prefix}reranker_timing_seconds": result.get("reranker_timing_seconds", 0.0),
        f"{prefix}reranker_timeout": result.get("reranker_timeout", False),
        f"{prefix}nli_positive_entailment_score": result.get("nli_positive_entailment_score", 0.0),
        f"{prefix}nli_negative_entailment_score": result.get("nli_negative_entailment_score", 0.0),
        f"{prefix}nli_contradiction_score": result.get("nli_contradiction_score", 0.0),
        f"{prefix}nli_margin": result.get("nli_margin", 0.0),
        f"{prefix}nli_decision_hint": result.get("nli_decision_hint", ""),
        f"{prefix}nli_top_positive_hypothesis": result.get("nli_top_positive_hypothesis", ""),
        f"{prefix}nli_top_negative_hypothesis": result.get("nli_top_negative_hypothesis", ""),
        f"{prefix}nli_positive_hypothesis_scores": result.get("nli_positive_hypothesis_scores", ""),
        f"{prefix}nli_negative_hypothesis_scores": result.get("nli_negative_hypothesis_scores", ""),
        f"{prefix}nli_runtime_source": result.get("nli_runtime_source", ""),
        f"{prefix}nli_timing_seconds": result.get("nli_timing_seconds", 0.0),
        f"{prefix}nli_timeout": result.get("nli_timeout", False),
        f"{prefix}nli_ignored_reason": result.get("nli_ignored_reason", ""),
        f"{prefix}zeroshot_relation_label": result.get("zeroshot_relation_label", ""),
        f"{prefix}zeroshot_relation_score": result.get("zeroshot_relation_score", 0.0),
        f"{prefix}zeroshot_ai_tool_for_review_score": result.get("zeroshot_ai_tool_for_review_score", 0.0),
        f"{prefix}zeroshot_subject_review_score": result.get("zeroshot_subject_review_score", 0.0),
        f"{prefix}zeroshot_top_labels": result.get("zeroshot_top_labels", ""),
        f"{prefix}zeroshot_runtime_source": result.get("zeroshot_runtime_source", ""),
        f"{prefix}zeroshot_timing_seconds": result.get("zeroshot_timing_seconds", 0.0),
        f"{prefix}zeroshot_timeout": result.get("zeroshot_timeout", False),
        f"{prefix}zeroshot_ignored_reason": result.get("zeroshot_ignored_reason", ""),
        f"{prefix}llm_judge_decision": result.get("llm_judge_decision", ""),
        f"{prefix}llm_judge_relation": result.get("llm_judge_relation", ""),
        f"{prefix}llm_uses_ai_for_review_workflow": result.get("llm_uses_ai_for_review_workflow", False),
        f"{prefix}llm_is_review_about_ai": result.get("llm_is_review_about_ai", False),
        f"{prefix}llm_workflow_tasks_detected": result.get("llm_workflow_tasks_detected", ""),
        f"{prefix}llm_external_domain_tasks_detected": result.get("llm_external_domain_tasks_detected", ""),
        f"{prefix}llm_judge_reason": result.get("llm_judge_reason", ""),
        f"{prefix}workflow_evidence_quote": result.get("workflow_evidence_quote", ""),
        f"{prefix}external_domain_evidence_quote": result.get("external_domain_evidence_quote", ""),
        f"{prefix}task_object": result.get("task_object", ""),
        f"{prefix}task_object_type": result.get("task_object_type", "unclear"),
        f"{prefix}llm_relation_confidence": result.get("relation_confidence", 0.0),
        f"{prefix}uncertainty_reason": result.get("uncertainty_reason", ""),
        f"{prefix}directional_judge_source": result.get("directional_judge_source", ""),
        f"{prefix}directional_relation": result.get("directional_relation", ""),
        f"{prefix}directional_confidence": result.get("directional_confidence", 0.0),
        f"{prefix}directional_uses_ai_for_review_workflow": result.get("directional_uses_ai_for_review_workflow", False),
        f"{prefix}directional_is_review_about_ai_external_domain": result.get("directional_is_review_about_ai_external_domain", False),
        f"{prefix}directional_reason": result.get("directional_reason", ""),
        f"{prefix}llm_directional_judge_used": result.get("llm_directional_judge_used", False),
        f"{prefix}llm_directional_judge_error": result.get("llm_directional_judge_error", ""),
        f"{prefix}llm_directional_triggered": result.get("llm_directional_triggered", False),
        f"{prefix}llm_directional_trigger_reason": result.get("llm_directional_trigger_reason", ""),
        f"{prefix}llm_directional_skipped_reason": result.get("llm_directional_skipped_reason", ""),
        f"{prefix}llm_directional_candidate_score": result.get("llm_directional_candidate_score", 0.0),
        f"{prefix}ai_review_candidate_detected": result.get("ai_review_candidate_detected", False),
        f"{prefix}positive_workflow_candidate_detected": result.get("positive_workflow_candidate_detected", False),
        f"{prefix}positive_workflow_candidate_terms": result.get("positive_workflow_candidate_terms", ""),
        f"{prefix}external_domain_candidate_detected": result.get("external_domain_candidate_detected", False),
        f"{prefix}external_domain_candidate_terms": result.get("external_domain_candidate_terms", ""),
        f"{prefix}llm_directional_cache_hit": result.get("llm_directional_cache_hit", False),
        f"{prefix}llm_directional_cache_source": result.get("llm_directional_cache_source", ""),
        f"{prefix}llm_directional_cache_key": result.get("llm_directional_cache_key", ""),
        f"{prefix}llm_directional_timing_seconds": result.get("llm_directional_timing_seconds", 0.0),
        f"{prefix}llm_route": result.get("llm_route", ""),
        f"{prefix}llm_skip_reason": result.get("llm_skip_reason", ""),
        f"{prefix}llm_required_reason": result.get("llm_required_reason", ""),
        f"{prefix}llm_gate_confidence": result.get("llm_gate_confidence", 0.0),
        f"{prefix}model_positive_score": result.get("model_positive_score", 0.0),
        f"{prefix}model_negative_score": result.get("model_negative_score", 0.0),
        f"{prefix}model_conflict_score": result.get("model_conflict_score", 0.0),
        f"{prefix}model_uncertainty_score": result.get("model_uncertainty_score", 0.0),
        f"{prefix}model_decision_hint": result.get("model_decision_hint", ""),
        f"{prefix}model_fusion_reason": result.get("model_fusion_reason", ""),
        f"{prefix}model_fusion_action": result.get("model_fusion_action", "disabled"),
        f"{prefix}model_fusion_blocked_reason": result.get("model_fusion_blocked_reason", ""),
        f"{prefix}fast_mode_current_equivalence_blocked_reason": result.get("fast_mode_current_equivalence_blocked_reason", ""),
        f"{prefix}fast_preserved_current_uncertainty": result.get("fast_preserved_current_uncertainty", False),
        f"{prefix}cache_schema_complete": result.get("cache_schema_complete", True),
        f"{prefix}cache_missing_adjudication_fields": result.get("cache_missing_adjudication_fields", ""),
        f"{prefix}fast_recomputed_due_to_incomplete_cache": result.get("fast_recomputed_due_to_incomplete_cache", False),
        f"{prefix}model_primary_signal": result.get("model_primary_signal", ""),
        f"{prefix}model_primary_margin": result.get("model_primary_margin", 0.0),
        f"{prefix}model_promoted_from_reject": result.get("model_promoted_from_reject", False),
        f"{prefix}model_promoted_from_maybe": result.get("model_promoted_from_maybe", False),
        f"{prefix}model_demoted_from_keep": result.get("model_demoted_from_keep", False),
        f"{prefix}technology_match": result.get("technology_match", 0.0),
        f"{prefix}effective_technology_match": result.get("effective_technology_match", 0.0),
        f"{prefix}method_family_left": result.get("method_family_left", ""),
        f"{prefix}method_family_right": result.get("method_family_right", ""),
        f"{prefix}method_family_compatible": result.get("method_family_compatible", False),
        f"{prefix}method_family_match": result.get("method_family_match", ""),
        f"{prefix}method_family_reason": result.get("method_family_reason", ""),
        f"{prefix}method_family_confidence": result.get("method_family_confidence", 0.0),
        f"{prefix}broad_method_query_detected": result.get("broad_method_query_detected", False),
        f"{prefix}task_match": result.get("task_match", 0.0),
        f"{prefix}task_subject_match": result.get("task_subject_match", 0.0),
        f"{prefix}task_role_match": result.get("task_role_match", 0.0),
        f"{prefix}subject_match": result.get("subject_match", 0.0),
        f"{prefix}context_match": result.get("context_match", 0.0),
        f"{prefix}study_role_match": result.get("study_role_match", False),
        f"{prefix}review_role_match": result.get("review_role_match", False),
        f"{prefix}canonical_task_left": result.get("canonical_task_left", ""),
        f"{prefix}canonical_task_right": result.get("canonical_task_right", ""),
        f"{prefix}task_identity_match": result.get("task_identity_match", False),
        f"{prefix}task_identity_conflict": result.get("task_identity_conflict", False),
        f"{prefix}task_family_compatible": result.get("task_family_compatible", False),
        f"{prefix}task_family_match": result.get("task_family_match", ""),
        f"{prefix}task_family_score": result.get("task_family_score", 0.0),
        f"{prefix}decision_path": result.get("decision_path", ""),
        f"{prefix}semantic_rescue_applied": result.get("semantic_rescue_applied", False),
        f"{prefix}semantic_rescue_reason": result.get("semantic_rescue_reason", ""),
        f"{prefix}reject_blocked_by_family_compatibility": result.get(
            "reject_blocked_by_family_compatibility",
            False,
        ),
        f"{prefix}rejected_despite_task_family_compatibility": result.get(
            "rejected_despite_task_family_compatibility",
            False,
        ),
        f"{prefix}review_workflow_gate_applied": result.get("review_workflow_gate_applied", False),
        f"{prefix}comparison_diagnostic": result.get("comparison_diagnostic", ""),
        f"{prefix}paper_methods": result.get("paper_methods", ""),
        f"{prefix}paper_tasks_or_outcomes": result.get("paper_tasks_or_outcomes", ""),
        f"{prefix}paper_contexts": result.get("paper_contexts", ""),
        f"{prefix}paper_evidence_type": result.get("paper_evidence_type", ""),
        f"{prefix}paper_inclusion_cues": result.get("paper_inclusion_cues", ""),
        f"{prefix}paper_exclusion_cues": result.get("paper_exclusion_cues", ""),
        f"{prefix}paper_observed_relation": result.get("paper_observed_relation", ""),
        f"{prefix}relation_match": result.get("relation_match", False),
        f"{prefix}relation_conflict": result.get("relation_conflict", False),
        f"{prefix}relation_alignment_score": result.get("relation_alignment_score", 0.0),
        f"{prefix}relation_confidence": result.get("relation_confidence", 0.0),
        f"{prefix}relation_evidence_terms": result.get("relation_evidence_terms", ""),
        f"{prefix}relation_negative_terms": result.get("relation_negative_terms", ""),
        f"{prefix}relation_mismatch_reason": result.get("relation_mismatch_reason", ""),
        f"{prefix}workflow_use_score": result.get("workflow_use_score", 0.0),
        f"{prefix}subject_only_score": result.get("subject_only_score", 0.0),
        f"{prefix}paper_type_only_score": result.get("paper_type_only_score", 0.0),
        f"{prefix}external_domain_topic_score": result.get("external_domain_topic_score", 0.0),
        f"{prefix}implied_workflow_score": result.get("implied_workflow_score", 0.0),
        f"{prefix}uncertainty_preservation_score": result.get("uncertainty_preservation_score", 0.0),
        f"{prefix}contradiction_score": result.get("contradiction_score", 0.0),
        f"{prefix}relation_decision_path": result.get("relation_decision_path", ""),
        f"{prefix}review_automation_relevance_score": result.get("review_automation_relevance_score", 0.0),
        f"{prefix}relation_evidence_strength": result.get("relation_evidence_strength", 0.0),
        f"{prefix}relation_dimension_met": result.get("relation_dimension_met", False),
        f"{prefix}relation_dimension_missing": result.get("relation_dimension_missing", False),
        f"{prefix}false_keep_risk_score": result.get("false_keep_risk_score", 0.0),
        f"{prefix}keep_suppression_applied": result.get("keep_suppression_applied", False),
        f"{prefix}keep_suppression_reason": result.get("keep_suppression_reason", ""),
        f"{prefix}keep_required_relation_missing": result.get("keep_required_relation_missing", False),
        f"{prefix}external_domain_subject_only": result.get("external_domain_subject_only", False),
        f"{prefix}workflow_use_required_for_keep": result.get("workflow_use_required_for_keep", False),
        f"{prefix}workflow_use_evidence_missing": result.get("workflow_use_evidence_missing", False),
        f"{prefix}subject_only_overrides_keep": result.get("subject_only_overrides_keep", False),
        f"{prefix}relation_keep_gate_passed": result.get("relation_keep_gate_passed", False),
        f"{prefix}outcome_evidence_strength": result.get("outcome_evidence_strength", 0.0),
        f"{prefix}outcome_evidence_terms": result.get("outcome_evidence_terms", ""),
        f"{prefix}evidence_coverage_ratio": result.get("evidence_coverage_ratio", 0.0),
        f"{prefix}uncertainty_score": result.get("uncertainty_score", 0.0),
        f"{prefix}abstract_insufficient": result.get("abstract_insufficient", False),
        f"{prefix}relation_unclear": result.get("relation_unclear", False),
        f"{prefix}suspicious_keep": result.get("suspicious_keep", False),
        f"{prefix}paper_outcomes": result.get("paper_outcomes", ""),
        f"{prefix}paper_text_quality_score": result.get("paper_text_quality_score", 0.0),
        f"{prefix}review_intent_relation": result.get("review_intent_relation", ""),
        f"{prefix}review_relation_confidence": result.get("review_relation_confidence", 0.0),
        f"{prefix}review_workflow_task_detected": result.get("review_workflow_task_detected", False),
        f"{prefix}review_workflow_task_terms": result.get("review_workflow_task_terms", ""),
        f"{prefix}review_automation_intent_detected": result.get(
            "review_automation_intent_detected",
            False,
        ),
        f"{prefix}review_automation_intent_terms": result.get(
            "review_automation_intent_terms",
            "",
        ),
        f"{prefix}review_context_detected": result.get("review_context_detected", False),
        f"{prefix}ai_tool_for_review_workflow": result.get("ai_tool_for_review_workflow", False),
        f"{prefix}technology_subject_review_detected": result.get(
            "technology_subject_review_detected",
            False,
        ),
        f"{prefix}external_domain_review_detected": result.get(
            "external_domain_review_detected",
            False,
        ),
        f"{prefix}external_domain_terms": result.get("external_domain_terms", ""),
        f"{prefix}review_context_only": result.get("review_context_only", False),
        f"{prefix}strong_workflow_use_evidence": result.get("strong_workflow_use_evidence", False),
        f"{prefix}workflow_task_term_detected": result.get("workflow_task_term_detected", False),
        f"{prefix}workflow_task_object_detected": result.get("workflow_task_object_detected", False),
        f"{prefix}workflow_task_object_linked": result.get("workflow_task_object_linked", False),
        f"{prefix}workflow_task_object_terms": result.get("workflow_task_object_terms", ""),
        f"{prefix}workflow_task_sense": result.get("workflow_task_sense", ""),
        f"{prefix}workflow_task_sense_confidence": result.get("workflow_task_sense_confidence", 0.0),
        f"{prefix}external_domain_task_detected": result.get("external_domain_task_detected", False),
        f"{prefix}external_domain_task_terms": result.get("external_domain_task_terms", ""),
        f"{prefix}task_object_mismatch": result.get("task_object_mismatch", False),
        f"{prefix}task_object_mismatch_reason": result.get("task_object_mismatch_reason", ""),
        f"{prefix}implied_workflow_intent_score": result.get("implied_workflow_intent_score", 0.0),
        f"{prefix}implied_workflow_intent_terms": result.get("implied_workflow_intent_terms", ""),
        f"{prefix}strong_implied_workflow_intent": result.get("strong_implied_workflow_intent", False),
        f"{prefix}medium_implied_workflow_intent": result.get("medium_implied_workflow_intent", False),
        f"{prefix}workflow_direction_detected": result.get("workflow_direction_detected", False),
        f"{prefix}subject_review_direction_detected": result.get("subject_review_direction_detected", False),
        f"{prefix}relation_direction": result.get("relation_direction", "none"),
        f"{prefix}workflow_intent_rescue_applied": result.get("workflow_intent_rescue_applied", False),
        f"{prefix}workflow_intent_rescue_reason": result.get("workflow_intent_rescue_reason", ""),
        f"{prefix}weak_subject_review_specificity": result.get("weak_subject_review_specificity", False),
        f"{prefix}workflow_task_object_required": result.get("workflow_task_object_required", False),
        f"{prefix}workflow_task_object_missing": result.get("workflow_task_object_missing", False),
        f"{prefix}review_role_gate_reason": result.get("review_role_gate_reason", ""),
        f"{prefix}method_evidence_terms": result.get("method_evidence_terms", ""),
        f"{prefix}task_evidence_terms": result.get("task_evidence_terms", ""),
        f"{prefix}context_evidence_terms": result.get("context_evidence_terms", ""),
        f"{prefix}evidence_type_terms": result.get("evidence_type_terms", ""),
        f"{prefix}exclusion_evidence_terms": result.get("exclusion_evidence_terms", ""),
        f"{prefix}method_evidence_strength": result.get("method_evidence_strength", 0.0),
        f"{prefix}task_evidence_strength": result.get("task_evidence_strength", 0.0),
        f"{prefix}context_evidence_strength": result.get("context_evidence_strength", 0.0),
        f"{prefix}evidence_coverage_count": result.get("evidence_coverage_count", 0),
        f"{prefix}required_dimensions_met": result.get("required_dimensions_met", ""),
        f"{prefix}required_dimensions_missing": result.get("required_dimensions_missing", ""),
        f"{prefix}strongest_inclusion_evidence": result.get("strongest_inclusion_evidence", ""),
        f"{prefix}strongest_exclusion_evidence": result.get("strongest_exclusion_evidence", ""),
        f"{prefix}contradiction_detected": result.get("contradiction_detected", False),
        f"{prefix}contradiction_reason": result.get("contradiction_reason", ""),
        f"{prefix}suspicious_reject": result.get("suspicious_reject", False),
        f"{prefix}method_alignment_score": result.get("method_alignment_score", 0.0),
        f"{prefix}task_alignment_score": result.get("task_alignment_score", 0.0),
        f"{prefix}context_alignment_score": result.get("context_alignment_score", 0.0),
        f"{prefix}evidence_alignment_score": result.get("evidence_alignment_score", 0.0),
        f"{prefix}negative_signal_score": result.get("negative_signal_score", 0.0),
        f"{prefix}final_relevance_score": result.get("final_relevance_score", 0.0),
        f"{prefix}inclusion_evidence": result.get("inclusion_evidence", ""),
        f"{prefix}exclusion_evidence": result.get("exclusion_evidence", ""),
        f"{prefix}uncertainty_reason": result.get("uncertainty_reason", ""),
        f"{prefix}rejection_reason_category": result.get("rejection_reason_category", ""),
    }


def _fusion_decision_fields(result):
    return {
        "LitSync_Decision": result.get("litsync_decision", ""),
        "LitSync_Reason": result.get("litsync_reason", ""),
        "LitSync_Confidence": result.get("litsync_confidence", ""),
        "Direct_AI_Decision": result.get("direct_ai_decision", ""),
        "Direct_AI_Reason": result.get("direct_ai_reason", ""),
        "Direct_AI_Confidence": result.get("direct_ai_confidence", ""),
        "Final_Fused_Decision": result.get("decision", ""),
        "Final_Fused_Reason": result.get("reason", ""),
        "Agreement_Status": result.get("agreement", ""),
        "Fusion_Policy": result.get("fusion_policy", ""),
    }


def _stage2_not_run_fields():
    cfg = get_model_judge_config()
    result = {
        "model_judges_enabled": cfg["enable_model_judges"],
        "model_judge_mode": cfg["model_judge_mode"],
        "model_judge_models_used": "",
        "model_judge_runtime_source": (
            "stage2_not_run" if cfg["enable_model_judges"] else "disabled"
        ),
        "model_profile": cfg["model_judge_profile"],
        "model_real_models_loaded": False,
        "model_fallback_reason": "Stage 2 was not run for this row.",
        "model_timing_seconds": 0.0,
        "model_judge_fallback_used": False,
        "model_judge_error": "",
        "model_fusion_action": (
            "stage2_not_run" if cfg["enable_model_judges"] else "disabled"
        ),
        "model_fusion_reason": (
            "Stage 2 was not run for this row."
            if cfg["enable_model_judges"]
            else "Model judges are disabled."
        ),
    }
    return _result_semantic_fields(result, "stage2_")


def _apply_final_adjudication(results):
    changed = 0
    for row in results:
        adjudication = adjudicate_row(row)
        for key, default in FINAL_ADJUDICATION_FIELDS.items():
            row[key] = adjudication.get(key, default)

        final_decision = adjudication.get("final_adjudicated_decision")
        row["stage1_fast_preserved_current_uncertainty"] = bool(
            row.get("stage1_fast_mode_current_equivalence_blocked_reason") == "external_reject_mixed_evidence"
            and final_decision == "MAYBE"
        )
        if final_decision in {"KEEP", "MAYBE", "REJECT"} and final_decision != row.get("Decision"):
            changed += 1
            row["Decision"] = final_decision
            reason = str(row.get("Reason") or "")
            adjudication_reason = str(adjudication.get("final_adjudication_reason") or "")
            row["Reason"] = (
                f"{reason} Final adjudication: {adjudication_reason}".strip()
                if reason
                else f"Final adjudication: {adjudication_reason}"
            )
            row["Final_Fused_Decision"] = final_decision
            row["Final_Fused_Reason"] = row["Reason"]

    return changed


def _apply_fast_safety_wrapper(results):
    """Final invariant guard for opt-in fast mode, after normal adjudication."""
    changed = 0
    for row in results:
        workflow = _as_bool(row.get("stage1_directional_uses_ai_for_review_workflow"))
        external = _as_bool(row.get("stage1_directional_is_review_about_ai_external_domain"))
        if workflow and row.get("Decision") == "REJECT":
            row["Decision"] = row["Final_Fused_Decision"] = "MAYBE"
            row["final_adjudicated_decision"] = "MAYBE"
            row["final_adjudication_action"] = "fast_safety_workflow_reject_to_maybe"
            changed += 1
        elif external and row.get("Decision") == "KEEP":
            row["Decision"] = row["Final_Fused_Decision"] = "MAYBE"
            row["final_adjudicated_decision"] = "MAYBE"
            row["final_adjudication_action"] = "fast_safety_external_keep_to_maybe"
            changed += 1
    return changed


def _finalize_performance_profile(profiler, results, *, input_total_rows, screened_total_rows, output_path):
    cache_stats = get_cache_stats()
    frame_cache_stats = get_semantic_frame_cache_stats()
    llm_calls = sum(1 for row in results if _as_bool(row.get("stage1_llm_directional_judge_used")))
    cache_hits = sum(1 for row in results if _as_bool(row.get("stage1_llm_directional_cache_hit")))
    cache_misses = max(0, llm_calls - cache_hits)
    llm_seconds = sum(_as_float(row.get("stage1_llm_directional_timing_seconds")) for row in results)
    skipped_llm = sum(1 for row in results if str(row.get("stage1_llm_route", "")).startswith("skipped_"))
    deterministic_only = sum(
        1
        for row in results
        if not _as_bool(row.get("stage1_llm_directional_judge_used"))
    )
    for row in results:
        row["perf_total_seconds"] = row.get("stage1_processing_seconds", 0.0)
        row["perf_llm_seconds"] = row.get("stage1_llm_directional_timing_seconds", 0.0)
        row["perf_llm_cache_hit"] = row.get("stage1_llm_directional_cache_hit", False)
        row["perf_route"] = row.get("stage1_llm_route", "")

    profiler.increment("llm_calls", llm_calls)
    profiler.increment("llm_cache_hits", cache_hits)
    profiler.increment("llm_cache_misses", cache_misses)
    profiler.increment("llm_skipped_rows", skipped_llm)
    profiler.increment("deterministic_only_rows", deterministic_only)
    profiler.update_extra(
        input_total_rows=input_total_rows,
        screened_total_rows=screened_total_rows,
        output_path=output_path,
        llm_calls=llm_calls,
        run_cache_hits=cache_hits,
        run_cache_misses=cache_misses,
        average_llm_call_seconds=round(llm_seconds / llm_calls, 4) if llm_calls else 0.0,
        llm_seconds=round(llm_seconds, 4),
        rows_skipped_from_llm=skipped_llm,
        deterministic_only_rows=deterministic_only,
        **cache_stats,
        **frame_cache_stats,
    )
def _as_float(value):
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_score(value):
    return 1.0 if value is True or str(value).strip().lower() == "true" else 0.0


def _as_bool(value):
    return value is True or str(value).strip().lower() == "true"


def _promotion_confidence(stage2_result):
    if stage2_result.get("confidence") not in (None, ""):
        return _as_float(stage2_result.get("confidence"))

    technology_match = _as_float(stage2_result.get("technology_match"))
    task_match = _as_float(stage2_result.get("task_role_match"))
    context_match = _as_float(stage2_result.get("context_match"))
    review_role_match = _as_score(stage2_result.get("review_role_match"))

    return round(
        (0.40 * technology_match)
        + (0.35 * task_match)
        + (0.15 * context_match)
        + (0.10 * review_role_match),
        4,
    )


def _stage2_keep_promotion_allowed(rq_frame, stage2_result):
    paper_review_role = str(stage2_result.get("paper_review_role", "")).strip()
    rq_question_type = str((rq_frame or {}).get("question_type", "")).strip()
    decision_path = str(stage2_result.get("decision_path", "")).strip()
    if (
        rq_question_type == "review_workflow_automation"
        and paper_review_role == "technology_being_reviewed"
        and not _as_bool(stage2_result.get("ai_tool_for_review_workflow"))
    ) or decision_path in {"review_workflow_gate_reject", "evidence_first_review_gate_reject"}:
        return (
            False,
            "Stage 2 KEEP was blocked because the paper appears to review the technology itself rather than provide evidence for the review-workflow task.",
        )

    task_supported = (
        _as_bool(stage2_result.get("task_family_compatible"))
        or _as_bool(stage2_result.get("task_identity_match"))
    )
    if not task_supported:
        return (
            False,
            "Stage 2 KEEP was blocked because the paper task is not compatible with the research-question task.",
        )

    effective_technology_match = max(
        _as_float(stage2_result.get("effective_technology_match")),
        _as_float(stage2_result.get("technology_match")),
    )
    method_supported = (
        _as_bool(stage2_result.get("method_family_compatible"))
        or effective_technology_match >= 0.65
    )
    if not method_supported:
        return (
            False,
            "Stage 2 KEEP was blocked because the method family is unrelated to the research question despite a related task label.",
        )

    subject_match = _as_float(stage2_result.get("subject_match"))
    context_match = _as_float(stage2_result.get("context_match"))
    domain_high = max(subject_match, context_match)
    if decision_path == "family_compatibility_domain_mismatch_reject" or domain_high < 0.10:
        return (
            False,
            "Stage 2 KEEP was blocked because subject and application-context evidence indicate a domain mismatch.",
        )

    if decision_path == "family_compatibility_role_mismatch_reject":
        return (
            False,
            "Stage 2 KEEP was blocked because the study or evidence role is incompatible with the research question.",
        )

    task_family_score = _as_float(stage2_result.get("task_family_score"))
    strong_semantic_evidence = (
        task_family_score >= 0.40
        or _as_bool(stage2_result.get("semantic_rescue_applied"))
        or decision_path in {
            "exact_task_strong_keep",
            "task_family_method_rescue_keep",
            "task_family_strong_keep",
            "semantic_rescue_strong_keep",
            "semantic_alignment_keep",
            "task_similarity_keep",
        }
    )
    if strong_semantic_evidence:
        return (
            True,
            "Stage 2 KEEP was allowed because the paper has compatible method and task evidence with no review-role violation.",
        )

    confidence = _promotion_confidence(stage2_result)
    if confidence < 0.72:
        return (
            False,
            "Stage 2 KEEP was blocked because the newer semantic evidence was incomplete and the fallback promotion confidence was below threshold.",
        )

    return (
        True,
        "Stage 2 KEEP was allowed by fallback promotion confidence after semantic guard checks found no contradiction.",
    )


def _stage2_reject_demotion_allowed(rq_frame, stage2_result):
    question_type = str(
        (rq_frame or {}).get("review_question_type")
        or (rq_frame or {}).get("question_type")
        or ""
    ).strip()
    if question_type != "review_workflow_automation":
        return True, "Stage 2 found a stronger exclusion decision."

    relation_conflict = _as_bool(stage2_result.get("relation_conflict"))
    contradiction_score = _as_float(stage2_result.get("contradiction_score"))
    relation = str(stage2_result.get("paper_observed_relation", "")).strip()
    explicit_contradiction = relation_conflict and contradiction_score >= 0.70
    if relation in {
        "technology_reviewed_as_subject",
        "external_domain_review",
        "paper_type_only_review",
    } and contradiction_score >= 0.70:
        explicit_contradiction = True

    coverage = int(_as_float(stage2_result.get("evidence_coverage_count")))
    if relation == "background_mention_only" and coverage <= 1:
        explicit_contradiction = True

    if explicit_contradiction:
        return True, (
            stage2_result.get("relation_mismatch_reason")
            or stage2_result.get("rejection_reason_category")
            or "Stage 2 found an explicit review-workflow relation contradiction."
        )
    return False, (
        "Stage 1 uncertainty was preserved because Stage 2 did not find an explicit "
        "review-workflow relation contradiction."
    )


def _stage2_escalation_candidate(rq_frame, stage1_result):
    rq_type = str(
        (rq_frame or {}).get("rq_type")
        or (rq_frame or {}).get("review_question_type")
        or (rq_frame or {}).get("question_type")
        or ""
    ).strip()
    if rq_type != "review_workflow_automation":
        return True, "Stage 2 escalation is enabled for this RQ type."

    reasons = []
    if _as_bool(stage1_result.get("medium_implied_workflow_intent")):
        reasons.append("medium_or_strong_workflow_intent")
    if str(stage1_result.get("workflow_task_sense", "")) == "review_workflow_task":
        reasons.append("linked_review_workflow_task")
    if (
        _as_bool(stage1_result.get("weak_subject_review_specificity"))
        and not reasons
    ):
        return False, (
            "Weakly specific subject-review uncertainty was preserved as MAYBE "
            "without spending a second-model inference."
        )
    if _as_bool(stage1_result.get("relation_conflict")):
        reasons.append("relation_conflict_requires_verification")
    if _as_bool(stage1_result.get("external_domain_task_detected")):
        reasons.append("external_task_requires_verification")
    if _as_float(stage1_result.get("evidence_coverage_ratio")) >= 0.75:
        reasons.append("high_partial_evidence_coverage")

    if reasons:
        return True, "; ".join(reasons)
    return False, (
        "Stable low-information uncertainty was preserved as MAYBE without a "
        "second-model inference."
    )


def _write_local_checkpoint(results, output_path):
    output_dir = os.path.dirname(output_path) or "outputs"
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def _normalize_row_limit(value):
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


def screen_csv(
    csv_path,
    research_question,
    output_path="outputs/screened.csv",
    mode="local",
    model=DEFAULT_MODEL,
    progress_job_id=None,
    two_stage_enabled: bool = TWO_STAGE_SCREENING_ENABLED,
    first_stage_model: str = FIRST_STAGE_MODEL,
    second_stage_model: str = SECOND_STAGE_MODEL,
    max_rows: int | None = None,
    semantic_strategy: str = DEFAULT_SCREENING_STRATEGY,
    screening_engine: str | None = None,
    gemini_web_profile_dir: str | None = None,
    gemini_web_batch_size: int | None = None,
    gemini_api_key: str | None = None,
    inclusion_criteria: str = "",
    exclusion_criteria: str = "",
):
    semantic_strategy = normalize_screening_strategy(semantic_strategy)
    selected_engine = normalize_processing_engine(screening_engine or mode)
    pipeline_cfg = get_model_judge_config()
    pipeline_mode = pipeline_cfg["screening_pipeline_mode"]
    print_model_judge_config()
    initialize_semantic_frame_cache()
    profiler = PerformanceProfiler(
        enabled=True,
        output_dir=os.path.dirname(output_path) or "outputs",
    )
    progress_job_id = progress_job_id or f"direct-{uuid.uuid4()}"
    if not PROGRESS.start_job(progress_job_id):
        raise RuntimeError("Another screening job is already running.")

    df = pd.read_csv(csv_path)
    input_total_rows = len(df)
    explicit_row_limit = _normalize_row_limit(max_rows)
    dev_row_limit = _normalize_row_limit(DEV_SCREENING_ROW_LIMIT)
    effective_row_limit = explicit_row_limit if explicit_row_limit is not None else dev_row_limit

    # ---- PERFORMANCE TIMER START ----
    overall_start = time.perf_counter()
    # ---------------------------------

    # Auto-detect Abstract and Title columns across database export formats
    abstract_col = _find_col(df, [
        "Abstract", "abstract", "AB", "Abstracts", "Summary",
        "Author Abstract", "abstract_note", "Description"
    ])
    title_col = _find_col(df, [
        "Title", "title", "TI", "Article Title", "Document Title",
        "paper_title", "Name"
    ])

    if abstract_col is None:
        raise KeyError(
            f"No Abstract column found. Columns in your CSV: {list(df.columns)}"
        )
    if title_col is None:
        raise KeyError(
            f"No Title column found. Columns in your CSV: {list(df.columns)}"
        )

    valid_rows = df[df[abstract_col].notna()]
    valid_total_rows = len(valid_rows)
    row_limit_applied = effective_row_limit is not None
    if row_limit_applied:
        valid_rows = valid_rows.head(effective_row_limit)
    screened_total_rows = len(valid_rows)
    row_limit_value = effective_row_limit or ""

    print(f"Loaded rows: {input_total_rows}")
    print(f"Valid screening rows: {valid_total_rows}")
    print(f"Screening rows: {screened_total_rows}")
    print(f"Row limit applied: {'yes' if row_limit_applied else 'no'}")

    # ---------- Update PROGRESS after valid_rows ----------
    PROGRESS.begin_screening(progress_job_id, len(valid_rows))
    # -------------------------------------------------------

    web_config = (
        GeminiWebConfig(profile_dir=gemini_web_profile_dir)
        if gemini_web_profile_dir
        else None
    )

    if selected_engine == GEMINI_WEB_ENGINE:
        from config import GEMINI_WEB_BATCH_SIZE
        from gemini_web_screening import (
            GeminiWebScreeningOptions,
            screen_csv_with_gemini_web,
        )

        try:
            result = screen_csv_with_gemini_web(
                csv_path=csv_path,
                research_question=research_question,
                progress=PROGRESS,
                screening_session=SCREENING_SESSION,
                progress_job_id=progress_job_id,
                options=GeminiWebScreeningOptions(
                    batch_size=int(gemini_web_batch_size or GEMINI_WEB_BATCH_SIZE),
                    inclusion_criteria=inclusion_criteria,
                    exclusion_criteria=exclusion_criteria,
                    output_dir=os.path.dirname(output_path) or "outputs",
                    checkpoint_path=output_path,
                    browser=web_config or GeminiWebConfig(),
                ),
                max_rows=effective_row_limit,
            )
        except Exception as exc:
            PROGRESS.fail(progress_job_id, exc)
            raise

        rows = result.get("rows", [])
        SCREENING_SESSION.set_results(rows)
        PROGRESS.finish(progress_job_id)

        overall_time = time.perf_counter() - overall_start
        print("\n========== GEMINI WEB PERFORMANCE ==========")
        print(f"Total runtime: {overall_time:.2f} s")
        if len(valid_rows):
            print(f"Average per paper: {overall_time / len(valid_rows):.2f} s")
        print("===========================================\n")

        return {
            "keep": result["keep"],
            "maybe": result["maybe"],
            "reject": result["reject"],
            "parse_error": result["parse_error"],
            "output_file": result["output_file"],
            "two_stage_enabled": False,
            "stage1_model": "",
            "stage2_model": None,
            "semantic_strategy": semantic_strategy,
            "stage2_rerun_count": 0,
            "total_papers": result["total_papers"],
            "stage1_total": result["total_papers"],
            "stage2_total": 0,
            "input_total_rows": result.get("input_total_rows", input_total_rows),
            "screened_total_rows": result.get("screened_total_rows", result["total_papers"]),
            "row_limit_applied": result.get("row_limit_applied", row_limit_applied),
            "row_limit_value": result.get("row_limit_value", row_limit_value),
            "screening_engine": GEMINI_WEB_ENGINE,
            "batch_size": result["batch_size"],
            "output_dir": result["output_dir"],
        }

    with resolve_processing_engine(
        selected_engine,
        gemini_web_config=web_config,
        gemini_api_key=gemini_api_key,
    ) as inference_engine:
        # Stage-1 semantic frame is only needed by the LitSync workflow strategy.
        rq_frame_stage1 = None
        if strategy_requires_rq_frame(semantic_strategy):
            with profiler.measure("rq_extraction"):
                rq_frame_stage1 = extract_research_question_frame(
                    research_question=research_question,
                    model=first_stage_model,
                    inference_engine=inference_engine,
                )
                profiler.increment("rq_contract_created_count")
            with profiler.measure("corpus_profile"):
                corpus_profile = profile_corpus(
                    valid_rows,
                    title_col,
                    abstract_col,
                    rq_frame=rq_frame_stage1,
                )
                profiler.increment("corpus_profile_created_count")
                rq_frame_stage1 = enrich_research_question_frame_with_corpus(
                    rq_frame_stage1,
                    corpus_profile,
                )

        # Stage 2 re-extracts the RQ frame only when escalation is enabled.
        maybe_paper_indices = []

        keep_count = 0
        maybe_count = 0
        reject_count = 0
        parse_error_count = 0

        results = []
        included_results = []
        maybe_results = []
        excluded_results = []

        model_route_printed = {"stage1": False, "stage2": False}

        def _print_model_route(stage, result):
            if model_route_printed.get(stage):
                return
            cfg = get_model_judge_config()
            print("[MODEL JUDGE ROUTE]")
            print(f"stage={stage}")
            print(f"enabled={cfg['enable_model_judges']}")
            print(f"mode={cfg['model_judge_mode']}")
            print(f"hf_loading={cfg['enable_hf_model_loading']}")
            print(f"fusion_called={result.get('model_judges_enabled') is True}")
            print(f"fusion_action={result.get('model_fusion_action', '')}")
            print("[MODEL JUDGE SOURCE]", result.get("model_judge_runtime_source", ""))
            print("[MODEL JUDGE TIMING]", result.get("model_timing_seconds", 0.0))
            model_route_printed[stage] = True

        def record_stage1_result(result, current):
            nonlocal keep_count, maybe_count, reject_count
            _print_model_route("stage1", result)

            title = result["title"]
            abstract = result["abstract"]
            decision = result["decision"]
            reason = result["reason"]

            if decision == "KEEP":
                keep_count += 1
            elif decision == "MAYBE":
                maybe_count += 1
            elif decision == "REJECT":
                reject_count += 1

            PROGRESS.update_counts(
                progress_job_id,
                current,
                keep_count,
                maybe_count,
                reject_count,
            )

            results.append({
                "Title": title,
                "Abstract": abstract,
                "Decision": decision,
                "Reason": reason,
                "Confidence": result.get("confidence", ""),
                "Required_Evidence": result.get("required_evidence", ""),
                "Paper_Contribution": result.get("paper_contribution", ""),
                "processing_engine": selected_engine,
                "screening_strategy": semantic_strategy,
                "input_total_rows": input_total_rows,
                "valid_total_rows": valid_total_rows,
                "screened_total_rows": screened_total_rows,
                "row_limit_applied": row_limit_applied,
                "row_limit_value": row_limit_value,
                "rq_contract_created_count": 1 if rq_frame_stage1 else 0,
                "corpus_profile_created_count": 1 if rq_frame_stage1 else 0,
                "rq_contract_reuse_rate": 1.0 if rq_frame_stage1 else 0.0,
                **model_config_csv_fields(),
                **_fusion_decision_fields(result),
                **_result_semantic_fields(result, "stage1_"),
                **_stage2_not_run_fields(),
                "stage1_model": first_stage_model,
                "stage2_model": second_stage_model if two_stage_enabled else "",
                "stage2_rescreened": False,
                "stage1_decision": decision,
                "stage2_decision_raw": "",
                "stage2_promotion_confidence": 0.0,
                "stage2_promotion_blocked": False,
                "stage2_guard_reason": "",
                "stage2_demoted_to_reject": False,
                "stage2_demotion_reason": "",
                "stage2_uncertainty_preserved": False,
                "stage2_relation_conflict_detected": False,
                "stage2_explicit_contradiction": False,
                "stage2_task_sense_conflict": False,
                "stage2_task_object_mismatch": False,
                "stage2_workflow_intent_rescue": False,
                "stage2_preserved_due_to_ambiguous_task_sense": False,
                "stage2_workflow_intent_rescue_reason": "",
                "stage2_subject_review_direction_conflict": False,
                "stage2_relation_direction_used": "none",
                "stage2_directional_override_applied": False,
                "stage2_directional_override_reason": "",
                "stage2_directional_conflict": False,
                "stage2_directional_rescue": False,
                "stage1_processing_seconds": result.get("_processing_seconds", 0.0),
                "stage2_processing_seconds": 0.0,
                "stage2_rq_contract_reused": False,
                "stage2_escalation_selected": False,
                "stage2_escalation_reason": "",
            })

            if decision == "MAYBE":
                selected, selection_reason = _stage2_escalation_candidate(
                    rq_frame_stage1,
                    result,
                )
                results[-1]["stage2_escalation_selected"] = selected
                results[-1]["stage2_escalation_reason"] = selection_reason
                if selected:
                    maybe_paper_indices.append(len(results) - 1)

            if decision == "KEEP":
                included_results.append({
                    "Title": title,
                    "Abstract": abstract,
                    "Reason": reason,
                })
            elif decision == "MAYBE":
                maybe_results.append({
                    "Title": title,
                    "Abstract": abstract,
                    "Reason": reason,
                })
            elif decision == "REJECT":
                excluded_results.append({
                    "Title": title,
                    "Abstract": abstract,
                    "Reason": reason,
                })

            if current == len(valid_rows) or (
                LOCAL_CHECKPOINT_INTERVAL
                and current % int(LOCAL_CHECKPOINT_INTERVAL) == 0
            ):
                _write_local_checkpoint(results, output_path)

        pass1_started = time.perf_counter()
        for i, (source_index, row) in enumerate(valid_rows.iterrows(), start=1):
            paper_started_at = time.perf_counter()
            result = process_paper(
                row,
                title_col,
                abstract_col,
                research_question,
                rq_frame_stage1,
                selected_engine,
                model,
                semantic_strategy,
                inference_engine,
            )
            result["_processing_seconds"] = round(
                time.perf_counter() - paper_started_at,
                4,
            )
            profiler.add_row_seconds(result["_processing_seconds"])
            profiler.add_time("row_processing_total", result["_processing_seconds"])
            profiler.add_time("model_score_fusion", result.get("model_timing_seconds", 0.0))
            profiler.add_time("llm_structured_judge", result.get("llm_directional_timing_seconds", 0.0))
            record_stage1_result(result, i)
            results[-1]["source_row_index"] = source_index
        pass1_seconds = time.perf_counter() - pass1_started

        # ---------------------------------------------------------------------------

        # ---------------- Stage 2 (optional MAYBE escalation) ----------------
        stage2_rerun_count = 0
        final_keep_count = keep_count
        final_maybe_count = maybe_count
        final_reject_count = reject_count

        pass2_started = time.perf_counter()
        if two_stage_enabled and maybe_paper_indices:
            # Stage 2 only touches MAYBE items and replaces their decision.
            # KEEP/REJECT from stage 1 are not re-evaluated.
            PROGRESS.begin_stage2(progress_job_id, len(maybe_paper_indices))

            rq_frame_stage2 = None
            if strategy_requires_rq_frame(semantic_strategy):
                # The RQ contract is immutable across stages. Stage 2 spends the
                # stronger model only on ambiguous papers, not on re-parsing the RQ.
                rq_frame_stage2 = dict(rq_frame_stage1 or {})

            for idx in maybe_paper_indices:
                row = results[idx]
                title = row.get("Title", "")
                abstract = row.get("Abstract", "")

                stage2_rerun_count += 1
                stage2_started_at = time.perf_counter()
                stage2_result = screen_candidate(
                    title=title,
                    abstract=abstract,
                    research_question=research_question,
                    strategy=semantic_strategy,
                    rq_frame=rq_frame_stage2,
                    mode=selected_engine,
                    model=second_stage_model,
                    inference_engine=inference_engine,
                )
                _print_model_route("stage2", stage2_result)
                row["stage2_processing_seconds"] = round(
                    time.perf_counter() - stage2_started_at,
                    4,
                )
                row["stage2_rq_contract_reused"] = True

                raw_stage2_decision = stage2_result.get("decision", row.get("Decision", ""))
                stage2_reason = stage2_result.get("reason", row.get("Reason", ""))
                row["stage2_promotion_confidence"] = _promotion_confidence(stage2_result)
                promotion_allowed = True
                if raw_stage2_decision == "KEEP":
                    promotion_allowed, guard_reason = _stage2_keep_promotion_allowed(
                        rq_frame_stage2,
                        stage2_result,
                    )
                    row["stage2_guard_reason"] = guard_reason
                    if not promotion_allowed:
                        row["stage2_promotion_blocked"] = True

                arbitration = arbitrate_stage2(
                    row.get("stage1_decision", "MAYBE"),
                    stage2_result,
                    build_rq_contract(research_question, rq_frame_stage2 or {}),
                    keep_allowed=promotion_allowed,
                )
                stage2_decision = arbitration["decision"]
                stage2_reason = arbitration["stage2_override_reason"]
                row.update({
                    key: value
                    for key, value in arbitration.items()
                    if key != "decision"
                })
                row["stage2_demotion_reason"] = (
                    arbitration["stage2_override_reason"]
                    if arbitration["stage2_demoted_to_reject"]
                    else ""
                )
                row["stage2_relation_conflict_detected"] = _as_bool(
                    stage2_result.get("relation_conflict")
                )

                row["Decision"] = stage2_decision
                row["Reason"] = stage2_reason
                row["Confidence"] = stage2_result.get("confidence", row.get("Confidence", ""))
                row["Required_Evidence"] = stage2_result.get("required_evidence", "")
                row["Paper_Contribution"] = stage2_result.get("paper_contribution", "")
                row.update(_fusion_decision_fields(stage2_result))
                row.update(_result_semantic_fields(stage2_result, "stage2_"))
                row["Final_Fused_Decision"] = stage2_decision
                row["Final_Fused_Reason"] = stage2_reason
                row["stage2_decision_raw"] = stage2_result.get("decision", "")
                row["stage2_rescreened"] = True
                PROGRESS.update_stage2(progress_job_id, stage2_rerun_count)

            final_counts = SCREENING_SESSION.counts(results)

            final_keep_count = final_counts["keep"]
            final_maybe_count = final_counts["maybe"]
            final_reject_count = final_counts["reject"]

            PROGRESS.update_counts(
                progress_job_id,
                len(valid_rows),
                final_keep_count,
                final_maybe_count,
                final_reject_count,
            )

        pass2_seconds = time.perf_counter() - pass2_started
        with profiler.measure("final_adjudicator"):
            final_adjudication_changed = _apply_final_adjudication(results)
            if pipeline_mode == "two_pass_fast":
                final_adjudication_changed += _apply_fast_safety_wrapper(results)
        frame_stats = get_semantic_frame_cache_stats()
        frame_lookups = frame_stats.get("semantic_frame_cache_hits", 0) + frame_stats.get("semantic_frame_cache_misses", 0)
        expected_order = [index for index, _ in valid_rows.iterrows()]
        actual_order = [row.get("source_row_index") for row in results]
        for row in results:
            row["pipeline_mode"] = pipeline_mode
            row["pass1_seconds"] = round(pass1_seconds, 4)
            row["pass2_seconds"] = round(pass2_seconds, 4)
            row["pass1_rows"] = len(results)
            row["pass2_rows"] = len(maybe_paper_indices)
            row["llm_required_count"] = sum(str(item.get("stage1_llm_route", "")).startswith("llm_required") for item in results)
            row["llm_skipped_count"] = sum(str(item.get("stage1_llm_route", "")).startswith("skipped_") for item in results)
            row["deterministic_only_count"] = sum(not _as_bool(item.get("stage1_llm_directional_judge_used")) for item in results)
            row["semantic_frame_cache_hit_rate"] = round(frame_stats.get("semantic_frame_cache_hits", 0) / frame_lookups, 4) if frame_lookups else 0.0
            row["row_order_preserved"] = actual_order == expected_order
        if final_adjudication_changed or results:
            final_counts = SCREENING_SESSION.counts(results)
            final_keep_count = final_counts["keep"]
            final_maybe_count = final_counts["maybe"]
            final_reject_count = final_counts["reject"]
            PROGRESS.update_counts(
                progress_job_id,
                len(valid_rows),
                final_keep_count,
                final_maybe_count,
                final_reject_count,
            )

    _finalize_performance_profile(
        profiler,
        results,
        input_total_rows=input_total_rows,
        screened_total_rows=screened_total_rows,
        output_path=output_path,
    )
    SCREENING_SESSION.set_results(results)
    with profiler.measure("csv_writing"):
        _write_local_checkpoint(results, output_path)
    profile_path = profiler.write()
    profiler.extra["performance_profile_file"] = profile_path
    if profile_path:
        print(f"[PERFORMANCE PROFILE] {profile_path}")

    # ---------------- Stage 2 analytics + statistics ----------------
    overall_time = time.perf_counter() - overall_start

    stage1_maybe_count = maybe_count
    stage2_keep_gain = 0
    stage2_maybe_stayed = 0
    stage2_reject_count = 0

    if two_stage_enabled and maybe_paper_indices:
        # Compute transition counts from Stage1 decision to Stage2 decision (only for escalated MAYBE rows)
        # We still have `results` updated in-place for those MAYBE rows.
        # Stage1 MAYBE decision rows were stored with `stage1_decision == "MAYBE"`.
        escalated = [r for r in results if r.get("stage2_rescreened") is True and r.get("stage1_decision") == "MAYBE"]
        stage2_rerun_count = len(escalated)

        for r in escalated:
            final_decision = r.get("Decision")
            if final_decision == "KEEP":
                stage2_keep_gain += 1
            elif final_decision == "REJECT":
                stage2_reject_count += 1
            elif final_decision == "MAYBE":
                stage2_maybe_stayed += 1

        stage2_maybe_count_after = final_maybe_count

        print("\n========== TWO-STAGE SUMMARY ==========")
        print(f"Stage 1 Model:\n{first_stage_model}")
        print(f"Stage 2 Model:\n{second_stage_model}")
        print(f"Total Papers:\n{len(valid_rows)}")
        print("\nStage 1")
        print(f"KEEP:\n{keep_count}")
        print(f"MAYBE:\n{stage1_maybe_count}")
        print(f"REJECT:\n{reject_count}")
        print("\nStage 2")
        print(f"Re-screened:\n{stage2_rerun_count}")
        print(f"MAYBE â†’ KEEP:\n{stage2_keep_gain}")
        print(f"MAYBE â†’ REJECT:\n{stage2_reject_count}")
        print(f"MAYBE â†’ MAYBE:\n{stage2_maybe_stayed}")
        print("\nFinal")
        print(f"KEEP:\n{final_keep_count}")
        print(f"MAYBE:\n{final_maybe_count}")
        print(f"REJECT:\n{final_reject_count}")
        gain = stage2_keep_gain
        loss = stage2_reject_count
        print(f"\nTwo-stage Gain:\n+{gain} KEEP\n-{loss} MAYBE")
        print("\n===================================")

    PROGRESS.finish(progress_job_id)

    # ---- PERFORMANCE SUMMARY ----

    print("\n========== PERFORMANCE ==========")
    print(f"Total runtime: {overall_time:.2f} s")
    if len(valid_rows):
        print(f"Average per paper: {overall_time / len(valid_rows):.2f} s")
    print("=================================\n")
    # -----------------------------

    return {
        "keep": final_keep_count,
        "maybe": final_maybe_count,
        "reject": final_reject_count,
        "parse_error": parse_error_count,
        "output_file": output_path,
        "two_stage_enabled": two_stage_enabled,
        "stage1_model": first_stage_model,
        "stage2_model": second_stage_model if two_stage_enabled else None,
        "semantic_strategy": semantic_strategy,
        "stage2_rerun_count": stage2_rerun_count,
        "total_papers": len(valid_rows),
        "stage1_total": len(valid_rows),
        "input_total_rows": input_total_rows,
        "screened_total_rows": screened_total_rows,
        "row_limit_applied": row_limit_applied,
        "row_limit_value": row_limit_value,
        "rq_contract_created_count": 1 if rq_frame_stage1 else 0,
        "corpus_profile_created_count": 1 if rq_frame_stage1 else 0,
        "rq_contract_reuse_rate": 1.0 if rq_frame_stage1 else 0.0,
        "performance_profile_file": profiler.extra.get("performance_profile_file", ""),
        "stage2_total": stage2_rerun_count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local CSV screening. Defaults to full dataset screening.")
    parser.add_argument("csv_path", nargs="?", default="uploads/LitSync_Clean_Dataset_2026-06-07.csv")
    parser.add_argument(
        "--question",
        default="Can large language models help automate systematic literature reviews?",
        help="Research question used for screening.",
    )
    parser.add_argument("--output", default="outputs/screened.csv")
    parser.add_argument("--limit", type=int, default=None, help="Optional positive row limit for debug runs.")
    parser.add_argument("--full", action="store_true", help="Screen the full dataset. This is the default.")
    parser.add_argument("--pipeline", choices=["current", "two_pass_fast"], default=None)
    parser.add_argument("--batch-llm", action="store_true", help="Enable optional batch LLM judge for this process.")
    parser.add_argument("--workers", type=int, default=None, help="Set SCREENING_WORKERS for future parallel modes.")
    parser.add_argument("--ollama-concurrency", type=int, default=None, help="Set OLLAMA_MAX_CONCURRENT.")
    parser.add_argument("--cache-info", action="store_true", help="Print LLM directional cache stats and exit.")
    parser.add_argument("--clear-llm-cache", action="store_true", help="Clear persistent LLM directional cache and exit.")
    args = parser.parse_args()
    if args.clear_llm_cache:
        clear_llm_cache()
        print("LLM directional cache cleared.")
        raise SystemExit(0)
    if args.cache_info:
        print(get_cache_stats())
        raise SystemExit(0)
    if args.pipeline:
        os.environ["SCREENING_PIPELINE_MODE"] = args.pipeline
    if args.batch_llm:
        os.environ["ENABLE_BATCH_LLM_JUDGE"] = "true"
    if args.workers:
        os.environ["SCREENING_WORKERS"] = str(args.workers)
        os.environ["ENABLE_PARALLEL_SCREENING"] = "true"
    if args.ollama_concurrency:
        os.environ["OLLAMA_MAX_CONCURRENT"] = str(args.ollama_concurrency)
    summary = screen_csv(
        csv_path=args.csv_path,
        research_question=args.question,
        output_path=args.output,
        max_rows=None if args.full else args.limit,
    )
    print(summary)
