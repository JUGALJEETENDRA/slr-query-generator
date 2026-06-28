import os
import time
import pandas as pd
from screener import screen_paper
from fast_title_screener import title_score
from semantic_frame import extract_semantic_frame
from config import (
    DEFAULT_MODEL,
    HYBRID_SCREENING_ENABLED,
    FIRST_STAGE_MODEL,
    SECOND_STAGE_MODEL,
    HYBRID_ESCALATE_ON,
)

from concurrent.futures import ThreadPoolExecutor, as_completed  # Added for parallel processing
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

    def snapshot(self):
        with self._lock:
            state = dict(self._state)
            if state["status"] == "running" and self._started_at is not None:
                state["runtime_seconds"] = round(
                    time.perf_counter() - self._started_at,
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
):
    title = row[title_col]
    abstract = str(row[abstract_col])

    score = title_score(
        title=title,
        research_question=research_question,
        model=model,
    )

    if score < 10:
        return {
            "decision": "REJECT",
            "title": title,
            "abstract": abstract,
            "reason": (
                "Excluded at title screening because the title does not appear "
                "to address the review question closely enough to require "
                "abstract-level screening."
            ),
            "required_evidence": "",
            "paper_contribution": "",
        }

    result = screen_paper(
        title=title,
        abstract=abstract,
        research_question=research_question,
        rq_frame=rq_frame,
        mode=mode,
        model=model,
    )

    result["title"] = title
    result["abstract"] = abstract

    return result
# ---------------------------------------------------------------------------


def _result_semantic_fields(result, prefix=""):
    return {
        f"{prefix}paper_primary_subject": result.get("paper_primary_subject", ""),
        f"{prefix}paper_intervention_or_method": result.get("paper_intervention_or_method", ""),
        f"{prefix}paper_target_problem_or_task": result.get("paper_target_problem_or_task", ""),
        f"{prefix}paper_application_context": result.get("paper_application_context", ""),
        f"{prefix}paper_evidence_type": result.get("paper_evidence_type", ""),
        f"{prefix}paper_study_role": result.get("paper_study_role", ""),
        f"{prefix}paper_review_role": result.get("paper_review_role", ""),
        f"{prefix}technology_match": result.get("technology_match", 0.0),
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
    }


def _as_float(value):
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_score(value):
    return 1.0 if value is True or str(value).strip().lower() == "true" else 0.0


def _promotion_confidence(stage2_result):
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
    if paper_review_role == "technology_being_reviewed":
        return (
            False,
            "Stage 2 did not promote this paper because it appears to review the technology itself rather than provide direct evidence for the review question.",
        )

    confidence = _promotion_confidence(stage2_result)
    if confidence < 0.72:
        return (
            False,
            "Stage 2 did not promote this paper because the semantic evidence was not strong enough for automatic inclusion.",
        )

    return True, ""


def screen_csv(
    csv_path,
    research_question,
    output_path="outputs/screened.csv",
    mode="local",
    model=DEFAULT_MODEL,
    progress_job_id=None,
    hybrid_enabled: bool = HYBRID_SCREENING_ENABLED,
    first_stage_model: str = FIRST_STAGE_MODEL,
    second_stage_model: str = SECOND_STAGE_MODEL,
    max_rows: int | None = None,
):
    progress_job_id = progress_job_id or f"direct-{uuid.uuid4()}"
    if not PROGRESS.start_job(progress_job_id):
        raise RuntimeError("Another screening job is already running.")

    df = pd.read_csv(csv_path)

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

    valid_rows = df[df[abstract_col].notna()].head(100)
    if max_rows is not None:
        valid_rows = valid_rows.head(max_rows)

    # ---------- Update PROGRESS after valid_rows ----------
    PROGRESS.begin_screening(progress_job_id, len(valid_rows))
    # -------------------------------------------------------

    # Stage-1 semantic frame (baseline for paper-vs-RQ comparison)
    rq_frame_stage1 = extract_semantic_frame(
        title=research_question,
        abstract="",
        model=first_stage_model,
    )

    # Hybrid stage 2 will re-extract the RQ frame only if/when needed.
    maybe_paper_indices = []

    keep_count = 0
    maybe_count = 0
    reject_count = 0
    parse_error_count = 0

    results = []
    included_results = []
    maybe_results = []
    excluded_results = []

    # ---------- REPLACED the original for-loop with ThreadPoolExecutor ----------
    with ThreadPoolExecutor(max_workers=4) as executor:

        futures = [
            executor.submit(
                process_paper,
                row,
                title_col,
                abstract_col,
                research_question,
                rq_frame_stage1,
                mode,
                model,

            )
            for _, row in valid_rows.iterrows()
        ]

        for i, future in enumerate(as_completed(futures), start=1):

            result = future.result()

            # Track MAYBE papers for stage 2 escalation
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
                i,
                keep_count,
                maybe_count,
                reject_count,
            )


            results.append({
                "Title": title,
                "Abstract": abstract,
                "Decision": decision,
                "Reason": reason,
                "Required_Evidence": result.get("required_evidence", ""),
                "Paper_Contribution": result.get("paper_contribution", ""),
                **_result_semantic_fields(result, "stage1_"),
                **_result_semantic_fields({}, "stage2_"),
                "stage1_model": first_stage_model,
                "stage2_model": second_stage_model if hybrid_enabled else "",
                "hybrid_rescreened": False,
                "stage1_decision": decision,
                "stage2_decision_raw": "",
                "stage2_promotion_confidence": 0.0,
                "stage2_promotion_blocked": False,
                "stage2_guard_reason": "",
            })

            if decision == "MAYBE":
                # Store index into `results` for stage-2 replacement.
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
    # ---------------------------------------------------------------------------

    # ---------------- Stage 2 (optional hybrid escalation) ----------------
    stage2_rerun_count = 0
    final_keep_count = keep_count
    final_maybe_count = maybe_count
    final_reject_count = reject_count

    if hybrid_enabled and maybe_paper_indices:
        # Stage 2 only touches MAYBE items and replaces their decision.
        # IMPORTANT: KEEP/REJECT from stage 1 are not re-evaluated.
        PROGRESS.begin_stage2(progress_job_id, len(maybe_paper_indices))

        # Stage-2 RQ frame is extracted with the stronger model.
        rq_frame_stage2 = extract_semantic_frame(
            title=research_question,
            abstract="",
            model=second_stage_model,
        )

        # Re-evaluate each MAYBE entry.
        for idx in maybe_paper_indices:
            row = results[idx]
            title = row.get("Title", "")
            abstract = row.get("Abstract", "")

            stage2_rerun_count += 1
            # Reuse existing screener logic with the stage-2 model.
            stage2_result = screen_paper(
                title=title,
                abstract=abstract,
                research_question=research_question,
                rq_frame=rq_frame_stage2,
                mode=mode,
                model=second_stage_model,
            )

            stage2_decision = stage2_result.get("decision", row.get("Decision", ""))
            stage2_reason = stage2_result.get("reason", row.get("Reason", ""))
            row["stage2_promotion_confidence"] = _promotion_confidence(stage2_result)
            if stage2_decision == "KEEP":
                promotion_allowed, guard_reason = _stage2_keep_promotion_allowed(
                    rq_frame_stage2,
                    stage2_result,
                )
                if not promotion_allowed:
                    stage2_decision = "MAYBE"
                    stage2_reason = f"{stage2_reason}; {guard_reason}"
                    row["stage2_promotion_blocked"] = True
                    row["stage2_guard_reason"] = guard_reason

            row["Decision"] = stage2_decision
            row["Reason"] = stage2_reason
            row["Required_Evidence"] = stage2_result.get("required_evidence", "")
            row["Paper_Contribution"] = stage2_result.get("paper_contribution", "")
            row.update(_result_semantic_fields(stage2_result, "stage2_"))
            row["stage2_decision_raw"] = stage2_result.get("decision", "")
            row["hybrid_rescreened"] = True
            PROGRESS.update_stage2(progress_job_id, stage2_rerun_count)

        # Compute final counts
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

    SCREENING_SESSION.set_results(results)

    # ---------------- Stage 2 analytics + statistics ----------------
    overall_time = time.perf_counter() - overall_start

    stage1_maybe_count = maybe_count
    stage2_keep_gain = 0
    stage2_maybe_stayed = 0
    stage2_reject_count = 0

    if hybrid_enabled and maybe_paper_indices:
        # Compute transition counts from Stage1 decision to Stage2 decision (only for escalated MAYBE rows)
        # We still have `results` updated in-place for those MAYBE rows.
        # Stage1 MAYBE decision rows were stored with `stage1_decision == "MAYBE"`.
        escalated = [r for r in results if r.get("hybrid_rescreened") is True and r.get("stage1_decision") == "MAYBE"]
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

        print("\n========== HYBRID SUMMARY ==========")
        print(f"Stage 1 Model:\n{first_stage_model}")
        print(f"Stage 2 Model:\n{second_stage_model}")
        print(f"Total Papers:\n{len(valid_rows)}")
        print("\nStage 1")
        print(f"KEEP:\n{keep_count}")
        print(f"MAYBE:\n{stage1_maybe_count}")
        print(f"REJECT:\n{reject_count}")
        print("\nStage 2")
        print(f"Re-screened:\n{stage2_rerun_count}")
        print(f"MAYBE → KEEP:\n{stage2_keep_gain}")
        print(f"MAYBE → REJECT:\n{stage2_reject_count}")
        print(f"MAYBE → MAYBE:\n{stage2_maybe_stayed}")
        print("\nFinal")
        print(f"KEEP:\n{final_keep_count}")
        print(f"MAYBE:\n{final_maybe_count}")
        print(f"REJECT:\n{final_reject_count}")
        gain = stage2_keep_gain
        loss = stage2_reject_count
        print(f"\nHybrid Gain:\n+{gain} KEEP\n-{loss} MAYBE")
        print("\n===================================")

    PROGRESS.finish(progress_job_id)

    # ---- PERFORMANCE SUMMARY ----

    print("\n========== PERFORMANCE ==========")
    print(f"Total runtime: {overall_time:.2f} s")
    print(f"Average per paper: {overall_time / len(valid_rows):.2f} s")
    print("=================================\n")
    # -----------------------------

    return {
        "keep": final_keep_count,
        "maybe": final_maybe_count,
        "reject": final_reject_count,
        "parse_error": parse_error_count,
        "output_file": output_path,
        "hybrid_enabled": hybrid_enabled,
        "stage1_model": first_stage_model,
        "stage2_model": second_stage_model if hybrid_enabled else None,
        "stage2_rerun_count": stage2_rerun_count,
        "total_papers": len(valid_rows),
        "stage1_total": len(valid_rows),
        "stage2_total": stage2_rerun_count,
    }


if __name__ == "__main__":
    summary = screen_csv(
        csv_path="uploads/LitSync_Clean_Dataset_2026-06-07.csv",
        research_question="Can large language models help automate systematic literature reviews?"
    )
    print(summary)
