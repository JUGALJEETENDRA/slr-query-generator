"""LitSync Automated Benchmark Evaluation Engine.

Purpose:
- Analyze benchmark outputs (ground truth + LitSync screened output)
- Compute overall metrics
- Categorize failures using available telemetry fields
- Provide representative examples, likely root causes (by module), and
  recommendations ranked by observed failure frequencies.

This module is intentionally independent from the screening pipeline.
It must never influence KEEP/MAYBE/REJECT decisions.

Typical usage:
    from evaluation_engine import evaluate_benchmark
    print(evaluate_benchmark('benchmark/gold/gold_set.csv','outputs/comparator_results.csv'))
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from task_ontology import RESEARCH_TASK_ONTOLOGY


# -----------------------------
# Utilities
# -----------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_label(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip().lower()
    return s or None


def map_screened_decision_to_positive(screened_decision: Any) -> Optional[int]:
    """Map screened decision-ish values to {1,0}.

    Positive (1) => KEEP/INCLUDE/Relevent
    Negative (0) => REJECT/EXCLUDE/Irrelevant
    """
    d = _normalize_label(screened_decision)
    if d is None:
        return None

    if d in {"keep", "kept", "included", "include"}:
        return 1
    if d in {"maybe", "reject", "rejected", "exclude", "excluded"}:
        return 0

    if d in {"1", "true", "t", "yes", "y", "positive"}:
        return 1
    if d in {"0", "false", "f", "no", "n", "negative"}:
        return 0

    return None


def _safe_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        k = cand.lower()
        if k in lower_map:
            return lower_map[k]
    return None


def _choose_title_field(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_column(
        df,
        [
            "title",
            "paper_title",
            "paper",
            "name",
            "document_title",
            "Title",
            "Paper Title",
        ],
    )


def _matching_columns(row: pd.Series, candidates: Sequence[str]) -> List[str]:
    matches: List[str] = []
    for candidate in candidates:
        needle = candidate.lower()
        for col in row.index:
            haystack = str(col).lower()
            if needle == haystack or needle in haystack:
                matches.append(col)
    return matches


def _row_value(row: pd.Series, candidates: Sequence[str]) -> Any:
    for col in _matching_columns(row, candidates):
        value = row.get(col)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if str(value).strip() != "":
            return value
    return None


def _row_text(row: pd.Series, candidates: Sequence[str]) -> str:
    value = _row_value(row, candidates)
    if value is None:
        return ""
    return str(value).strip()


def _row_float(row: pd.Series, candidates: Sequence[str]) -> Optional[float]:
    return _safe_float(_row_value(row, candidates))


def _row_bool(row: pd.Series, candidates: Sequence[str]) -> Optional[bool]:
    value = _row_value(row, candidates)
    if value is None:
        return None
    normalized = _normalize_label(value)
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    return None


def _has_value(row: pd.Series, candidates: Sequence[str]) -> bool:
    return _row_value(row, candidates) is not None


def _known_canonical_task(value: str) -> bool:
    return str(value or "").strip() in RESEARCH_TASK_ONTOLOGY


def _compact_evidence(row: pd.Series) -> Dict[str, Any]:
    fields = [
        "canonical_task_left",
        "canonical_task_right",
        "task_identity_match",
        "task_match",
        "task_role_match",
        "technology_match",
        "context_match",
        "study_role_match",
        "review_role_match",
        "paper_intervention_or_method",
        "paper_target_problem_or_task",
        "paper_review_role",
        "paper_evidence_type",
    ]
    return {
        field: _row_value(row, [field])
        for field in fields
        if _row_value(row, [field]) is not None
    }


def _final_failure_name(predicted: Optional[int], ground_truth: Optional[int]) -> str:
    if predicted == 1 and ground_truth == 0:
        return "False Positive"
    if predicted == 0 and ground_truth == 1:
        return "False Negative"
    return "Decision Error"


# -----------------------------
# Failure categorization
# -----------------------------

@dataclass(frozen=True)
class FailureCategory:
    name: str
    description: str
    responsible_module: str = "evaluation_engine.py"
    expected_complexity: str = "Medium"
    expected_gain: str = "Medium"
    priority: int = 99
    suggestion: str = "Inspect representative failures and improve the responsible component."
    # Row predicates; category matches if all matchers return True.
    matchers: Tuple[Callable[[pd.Series], bool], ...] = ()


@dataclass
class FailureAnalysis:
    category: str
    root_cause: str
    confidence: float
    responsible_module: str
    suggested_fix: str
    origin: str
    propagation: str
    final_failure: str
    evidence: Dict[str, Any]
    expected_complexity: str = "Medium"
    expected_gain: str = "Medium"
    priority: int = 99


class FailureClassifier:
    """Classifies false positives/false negatives into failure categories.

    Extensible: add more FailureCategory objects with additional matchers.
    Schema-resilient: matchers only check for presence/values of common
    telemetry fields.

    Note: Matchers use generic telemetry semantics (low match/confidence,
    unknown ontology, etc.) and avoid domain-specific assumptions.
    """

    def __init__(self, categories: Optional[Sequence[FailureCategory]] = None):
        self.categories = list(categories) if categories is not None else self._default_categories()
        self.category_meta = {cat.name: cat for cat in self.categories}

    def classify_failure(self, row: pd.Series) -> List[str]:
        matched: List[str] = []
        for cat in self.categories:
            if cat.matchers and all(m(row) for m in cat.matchers):
                matched.append(cat.name)
            elif not cat.matchers and cat.name == "Other":
                # handled by fallback
                pass

        if not matched:
            return ["Other"]
        return matched

    def analyze_failure(
        self,
        row: pd.Series,
        predicted: Optional[int] = None,
        ground_truth: Optional[int] = None,
    ) -> List[FailureAnalysis]:
        analyses = self._semantic_failure_analyses(row, predicted, ground_truth)
        if analyses:
            return analyses

        return [
            self._analysis_from_category(
                category_name,
                row=row,
                predicted=predicted,
                ground_truth=ground_truth,
                confidence=0.55 if category_name != "Other" else 0.35,
            )
            for category_name in self.classify_failure(row)
        ]

    def _analysis_from_category(
        self,
        category_name: str,
        row: pd.Series,
        predicted: Optional[int],
        ground_truth: Optional[int],
        confidence: float,
        evidence: Optional[Dict[str, Any]] = None,
        root_cause: Optional[str] = None,
        origin: Optional[str] = None,
        propagation: Optional[str] = None,
    ) -> FailureAnalysis:
        meta = self.category_meta.get(
            category_name,
            FailureCategory(
                name=category_name,
                description="Unclassified failure.",
            ),
        )
        final_failure = _final_failure_name(predicted, ground_truth)
        return FailureAnalysis(
            category=category_name,
            root_cause=root_cause or meta.description,
            confidence=round(max(0.0, min(1.0, confidence)), 2),
            responsible_module=meta.responsible_module,
            suggested_fix=meta.suggestion,
            origin=origin or meta.responsible_module,
            propagation=propagation or "Telemetry indicated this component as the likely source of the final decision error.",
            final_failure=final_failure,
            evidence=evidence or _compact_evidence(row),
            expected_complexity=meta.expected_complexity,
            expected_gain=meta.expected_gain,
            priority=meta.priority,
        )

    def _semantic_failure_analyses(
        self,
        row: pd.Series,
        predicted: Optional[int],
        ground_truth: Optional[int],
    ) -> List[FailureAnalysis]:
        analyses: List[FailureAnalysis] = []

        canonical_left = _row_text(row, ["canonical_task_left"])
        canonical_right = _row_text(row, ["canonical_task_right"])
        task_identity_match = _row_bool(row, ["task_identity_match"])
        task_match = _row_float(row, ["task_match"])
        task_role_match = _row_float(row, ["task_role_match"])
        technology_match = _row_float(row, ["technology_match"])
        context_match = _row_float(row, ["context_match"])
        review_role_match = _row_bool(row, ["review_role_match"])
        study_role_match = _row_bool(row, ["study_role_match"])
        review_role = _row_text(row, ["paper_review_role", "review_role"])
        rq_task = _row_text(row, ["rq_target_problem_or_task", "required_task"])
        paper_task = _row_text(row, ["paper_target_problem_or_task", "target_problem_or_task"])
        rq_tech = _row_text(row, ["rq_intervention_or_method", "required_technology"])
        paper_tech = _row_text(row, ["paper_intervention_or_method", "intervention_or_method"])
        evidence_type = _row_text(row, ["paper_evidence_type", "evidence_type"])
        required_evidence = _row_text(row, ["required_evidence", "rq_evidence_type"])

        left_known = _known_canonical_task(canonical_left)
        right_known = _known_canonical_task(canonical_right)
        evidence = {
            "canonical_task_left": canonical_left,
            "canonical_task_right": canonical_right,
            "task_identity_match": task_identity_match,
            "task_match": task_match,
            "task_role_match": task_role_match,
            "technology_match": technology_match,
            "context_match": context_match,
            "review_role_match": review_role_match,
            "study_role_match": study_role_match,
            "review_role": review_role,
            "rq_task": rq_task,
            "paper_task": paper_task,
            "rq_technology": rq_tech,
            "paper_technology": paper_tech,
            "required_evidence": required_evidence,
            "paper_evidence_type": evidence_type,
        }

        if left_known and right_known and canonical_left != canonical_right:
            analyses.append(
                self._analysis_from_category(
                    "Canonical Task Mismatch",
                    row,
                    predicted,
                    ground_truth,
                    confidence=0.94,
                    evidence=evidence,
                    root_cause=(
                        f"The comparator identified different canonical research tasks: "
                        f"{canonical_left} vs {canonical_right}."
                    ),
                    origin="task_ontology.py",
                    propagation="Task identity conflict constrained task similarity before final classification.",
                )
            )

        if (
            technology_match is not None
            and technology_match < 0.50
            and (task_identity_match is True or (task_match is not None and task_match >= 0.55))
            and rq_tech
            and paper_tech
        ):
            analyses.append(
                self._analysis_from_category(
                    "Technology Hierarchy Missing",
                    row,
                    predicted,
                    ground_truth,
                    confidence=0.86,
                    evidence=evidence,
                    root_cause=(
                        f"Task evidence is related, but technology similarity is low for "
                        f"{rq_tech} vs {paper_tech}; this suggests a missing parent-child or sibling technology relation."
                    ),
                    origin="technology_ontology.py",
                    propagation="Comparator treated a likely technology relationship as weak similarity.",
                )
            )

        if review_role_match is False or review_role == "technology_being_reviewed":
            analyses.append(
                self._analysis_from_category(
                    "Review Role Conflict",
                    row,
                    predicted,
                    ground_truth,
                    confidence=0.88,
                    evidence=evidence,
                    root_cause="The paper's relationship to the review workflow conflicts with the required review role.",
                    origin="semantic_frame.py",
                    propagation="Review-role telemetry gated or weakened the comparator decision.",
                )
            )

        if study_role_match is False and _has_value(row, ["study_role"]):
            analyses.append(
                self._analysis_from_category(
                    "Study Role Conflict",
                    row,
                    predicted,
                    ground_truth,
                    confidence=0.72,
                    evidence=evidence,
                    root_cause="The extracted study role differs from the study role implied by the benchmark label.",
                    origin="semantic_frame.py",
                    propagation="Study-role mismatch reduced eligibility confidence.",
                )
            )

        if required_evidence and evidence_type and required_evidence.lower() != evidence_type.lower():
            analyses.append(
                self._analysis_from_category(
                    "Evidence Mismatch",
                    row,
                    predicted,
                    ground_truth,
                    confidence=0.80,
                    evidence=evidence,
                    root_cause=f"Required evidence ({required_evidence}) differs from paper evidence ({evidence_type}).",
                    origin="semantic_frame.py",
                    propagation="Evidence type mismatch made the final decision unreliable.",
                )
            )

        unknown_task = (
            (canonical_left and not left_known and canonical_left == rq_task)
            or (canonical_right and not right_known and canonical_right == paper_task)
        )
        if unknown_task and task_match is not None and task_match < 0.55:
            analyses.append(
                self._analysis_from_category(
                    "Ontology Miss",
                    row,
                    predicted,
                    ground_truth,
                    confidence=0.70,
                    evidence=evidence,
                    root_cause="At least one extracted task did not map to a canonical task identity.",
                    origin="task_ontology.py",
                    propagation="Comparator fell back to embedding similarity because canonical identity was unavailable.",
                )
            )

        if not analyses and (
            not paper_task
            or not canonical_right
            or (task_match is not None and task_match < 0.50 and not right_known)
        ):
            analyses.append(
                self._analysis_from_category(
                    "Extraction Ambiguity",
                    row,
                    predicted,
                    ground_truth,
                    confidence=0.62,
                    evidence=evidence,
                    root_cause="The extracted semantic frame lacks a clear task identity for reliable comparison.",
                    origin="semantic_frame.py",
                    propagation="Ambiguous extraction propagated into weak comparator telemetry.",
                )
            )

        return analyses

    @staticmethod
    def _default_categories() -> List[FailureCategory]:
        def col_value_lt(col_candidates: Sequence[str], threshold: float) -> Callable[[pd.Series], bool]:
            def _m(row: pd.Series) -> bool:
                for col in col_candidates:
                    # match exact or substring columns
                    for c in row.index:
                        cl = str(c).lower()
                        if col.lower() == cl or col.lower() in cl:
                            fv = _safe_float(row.get(c))
                            if fv is not None and fv < threshold:
                                return True
                return False

            return _m

        def col_label_in(col_candidates: Sequence[str], tokens: Sequence[str]) -> Callable[[pd.Series], bool]:
            tokens_set = {t.lower() for t in tokens}

            def _m(row: pd.Series) -> bool:
                for col in col_candidates:
                    for c in row.index:
                        cl = str(c).lower()
                        if col.lower() == cl or col.lower() in cl:
                            nv = _normalize_label(row.get(c))
                            if nv is not None and (nv in tokens_set or any(t in nv for t in tokens_set)):
                                return True
                return False

            return _m

        return [
            FailureCategory(
                name="Canonical Task Mismatch",
                description="Comparator found different canonical research tasks.",
                responsible_module="task_ontology.py",
                expected_complexity="Medium",
                expected_gain="High",
                priority=1,
                suggestion="Inspect repeated task-pair clusters and refine canonical task boundaries only when the relation is domain-independent.",
                matchers=(
                    col_label_in(["task_identity_match"], ["false"]),
                    col_label_in(["canonical_task_left", "canonical_task_right"], list(RESEARCH_TASK_ONTOLOGY)),
                ),
            ),
            FailureCategory(
                name="Technology Hierarchy Missing",
                description="Technology terms appear related through a hierarchy that telemetry cannot currently represent.",
                responsible_module="technology_ontology.py",
                expected_complexity="High",
                expected_gain="High",
                priority=2,
                suggestion="Introduce a domain-independent technology hierarchy or candidate parent-child suggestion layer; do not hardcode benchmark-specific relations.",
                matchers=(col_value_lt(["technology_match"], 0.5),),
            ),
            FailureCategory(
                name="Ontology Miss",
                description="Task or technology was not recognized or mapped to an ontology identity.",
                responsible_module="task_ontology.py",
                expected_complexity="Low",
                expected_gain="Medium",
                priority=3,
                suggestion="Review repeated unknown task phrases and add only general research-task synonyms.",
                matchers=(
                    col_label_in(
                        ["task_not_recognized", "task_unknown", "unknown_task", "task_identity_match"],
                        ["unknown", "not_recognized", "not_found", "false"],
                    ),
                ),
            ),
            FailureCategory(
                name="Low Confidence",
                description="Comparator/extractor confidence is weak.",
                responsible_module="semantic_comparator.py",
                expected_complexity="Medium",
                expected_gain="Medium",
                priority=7,
                suggestion="Improve confidence calibration and propagate extraction uncertainty into failure telemetry.",
                matchers=(
                    col_value_lt(
                        ["confidence", "comparator_confidence", "semantic_confidence", "low_confidence"],
                        0.5,
                    ),
                ),
            ),
            FailureCategory(
                name="Task Boundary",
                description="Mismatched task framing / boundary errors.",
                responsible_module="semantic_comparator.py",
                expected_complexity="Medium",
                expected_gain="High",
                priority=4,
                suggestion="Use canonical task identity and task-pair clusters before changing thresholds.",
                matchers=(col_value_lt(["task_match", "task_subject_match", "task_role_match"], 0.5),),
            ),
            FailureCategory(
                name="Technology Ontology",
                description="Technology ontology mapping failures.",
                responsible_module="technology_ontology.py",
                expected_complexity="High",
                expected_gain="High",
                priority=5,
                suggestion="Analyze repeated low-technology-match pairs as candidate hierarchy or synonym additions.",
                matchers=(col_value_lt(["technology_match"], 0.5),),
            ),
            FailureCategory(
                name="Study Role Conflict",
                description="Study role mismatch or misclassification.",
                responsible_module="semantic_frame.py",
                expected_complexity="Medium",
                expected_gain="Medium",
                priority=6,
                suggestion="Improve study-role extraction and distinguish empirical evaluation, review, dataset, tool, and framework papers.",
                matchers=(col_value_lt(["study_role_match"], 0.5),),
            ),
            FailureCategory(
                name="Review Role Conflict",
                description="Review role mismatch or misclassification.",
                responsible_module="semantic_frame.py",
                expected_complexity="Medium",
                expected_gain="High",
                priority=5,
                suggestion="Strengthen disentanglement between technology-being-reviewed and technology-used-for-review-workflow.",
                matchers=(col_value_lt(["review_role_match"], 0.5),),
            ),
            FailureCategory(
                name="Evidence Mismatch",
                description="Paper evidence type does not satisfy the evidence implied by the benchmark label or research question.",
                responsible_module="semantic_frame.py",
                expected_complexity="Medium",
                expected_gain="High",
                priority=4,
                suggestion="Make evidence_type a first-class evaluation signal and audit repeated evidence mismatch clusters.",
                matchers=(),
            ),
            FailureCategory(
                name="Extraction Ambiguity",
                description="Semantic frame extraction is missing or ambiguous enough to propagate a downstream comparator failure.",
                responsible_module="semantic_frame.py",
                expected_complexity="Medium",
                expected_gain="High",
                priority=3,
                suggestion="Audit extractor outputs for missing task, technology, evidence, or role fields before tuning comparator thresholds.",
                matchers=(),
            ),
            FailureCategory(
                name="Context Match",
                description="Context mismatch.",
                responsible_module="semantic_comparator.py",
                expected_complexity="Low",
                expected_gain="Medium",
                priority=8,
                suggestion="Use context as supporting evidence after task and role checks; inspect repeated context-pair clusters.",
                matchers=(col_value_lt(["context_match"], 0.5),),
            ),
            FailureCategory(
                name="Other",
                description="Fallback bucket when no category matches.",
                responsible_module="evaluation_engine.py",
                expected_complexity="Unknown",
                expected_gain="Unknown",
                priority=99,
                suggestion="Inspect examples and add a semantic failure category when a repeated pattern appears.",
                matchers=(),
            ),
        ]


# -----------------------------
# Reporting + History
# -----------------------------

@dataclass
class FailureExample:
    paper_title: str
    predicted: int
    ground_truth: int
    categories: List[str]
    row: Dict[str, Any]
    analyses: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkMetrics:
    total: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float


def _compute_binary_metrics(tp: int, fp: int, tn: int, fn: int) -> BenchmarkMetrics:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return BenchmarkMetrics(total, tp, fp, tn, fn, accuracy, precision, recall, f1)


class BenchmarkHistory:
    def __init__(self, history_path: str = "benchmark_history.json"):
        self.history_path = Path(history_path)

    def load(self) -> List[Dict[str, Any]]:
        if not self.history_path.exists():
            return []
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def append(self, record: Dict[str, Any]) -> None:
        history = self.load()
        history.append(record)
        self.history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


class BenchmarkReport:
    def __init__(
        self,
        metrics: BenchmarkMetrics,
        failure_breakdown: Dict[str, int],
        top_bottlenecks: List[Tuple[str, int]],
        examples_by_category: Dict[str, List[FailureExample]],
        recommendations: List[Dict[str, Any]],
        meta: Dict[str, Any],
        root_cause_breakdown: Optional[Dict[str, int]] = None,
        failure_clusters: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        ontology_suggestions: Optional[List[Dict[str, Any]]] = None,
        trend_deltas: Optional[Dict[str, Any]] = None,
    ):
        self.metrics = metrics
        self.failure_breakdown = failure_breakdown
        self.top_bottlenecks = top_bottlenecks
        self.examples_by_category = examples_by_category
        self.recommendations = recommendations
        self.meta = meta
        self.root_cause_breakdown = root_cause_breakdown or {}
        self.failure_clusters = failure_clusters or {}
        self.ontology_suggestions = ontology_suggestions or []
        self.trend_deltas = trend_deltas or {}

    def to_text(self, top_k_categories: int = 10) -> str:
        lines: List[str] = []
        lines.append("=" * 40)
        lines.append("\nLITSYNC BENCHMARK REPORT\n")
        lines.append("=" * 40)

        lines.append("\nOverall Accuracy")
        lines.append(f"Accuracy: {self.metrics.accuracy:.4f}")
        lines.append(f"Precision: {self.metrics.precision:.4f}")
        lines.append(f"Recall: {self.metrics.recall:.4f}")
        lines.append(f"F1 Score: {self.metrics.f1:.4f}")
        lines.append(f"False Positives: {self.metrics.fp}")
        lines.append(f"False Negatives: {self.metrics.fn}")

        lines.append("\n" + "-" * 40)
        lines.append("Failure Breakdown")

        for cat, count in sorted(self.failure_breakdown.items(), key=lambda x: x[1], reverse=True)[:top_k_categories]:
            lines.append(f"{cat}: {count}")

        if self.root_cause_breakdown:
            lines.append("\n" + "-" * 40)
            lines.append("Root Cause Breakdown")
            for root_cause, count in sorted(self.root_cause_breakdown.items(), key=lambda x: x[1], reverse=True)[:top_k_categories]:
                lines.append(f"{root_cause}: {count}")

        lines.append("\n" + "-" * 40)
        lines.append("Top Remaining Bottleneck")

        for i, (cat, count) in enumerate(self.top_bottlenecks[:5], start=1):
            lines.append(str(i))
            lines.append(cat)
            lines.append(f"Failures: {count}")

            ex = self.examples_by_category.get(cat, [])
            if ex:
                lines.append("Example Papers:")
                for e in ex[:3]:
                    lines.append(f"- {e.paper_title}")

            rec = next((r for r in self.recommendations if r.get("category") == cat), None)
            if rec:
                if rec.get("root_cause"):
                    lines.append("Likely Root Cause:")
                    lines.append(str(rec.get("root_cause", "")))
                if rec.get("confidence") is not None:
                    lines.append("Confidence:")
                    lines.append(str(rec.get("confidence", "")))
                lines.append("Suggested Improvement:")
                lines.append(str(rec.get("suggestion", "")))
                lines.append("Expected Impact:")
                lines.append(str(rec.get("expected_impact", "")))
                lines.append("Responsible Module:")
                lines.append(str(rec.get("responsible_module", "")))

            lines.append("")

        lines.append("\n" + "-" * 40)
        lines.append("Recommended Next Development Priority")
        for i, (cat, count) in enumerate(self.top_bottlenecks[:10], start=1):
            rec = next((r for r in self.recommendations if r.get("category") == cat), None)
            lines.append(f"{i}. {cat} ({count} failures)")
            if rec:
                if rec.get("responsible_module"):
                    lines.append(f"   Responsible Module: {rec['responsible_module']}")
                if rec.get("expected_impact"):
                    lines.append(f"   Expected Impact: {rec['expected_impact']}")
                if rec.get("estimated_complexity"):
                    lines.append(f"   Expected Complexity: {rec['estimated_complexity']}")
                if rec.get("suggestion"):
                    lines.append(f"   Recommendation: {rec['suggestion']}")

        if self.failure_clusters:
            lines.append("\n" + "-" * 40)
            lines.append("Failure Clusters")
            for cluster_name, clusters in self.failure_clusters.items():
                if not clusters:
                    continue
                lines.append(cluster_name)
                for cluster in clusters[:5]:
                    lines.append(
                        f"- {cluster.get('left', '')} -> {cluster.get('right', '')}: "
                        f"{cluster.get('count', 0)} failures"
                    )

        if self.ontology_suggestions:
            lines.append("\n" + "-" * 40)
            lines.append("Suggested Ontology Additions")
            for suggestion in self.ontology_suggestions[:10]:
                lines.append(
                    f"- {suggestion.get('left', '')} -> {suggestion.get('right', '')} "
                    f"({suggestion.get('count', 0)} failures): {suggestion.get('recommendation', '')}"
                )

        if self.trend_deltas:
            lines.append("\n" + "-" * 40)
            lines.append("Evolution Since Previous Benchmark")
            for key, value in self.trend_deltas.items():
                lines.append(f"{key}: {value}")

        return "\n".join(lines).strip() + "\n"


# -----------------------------
# Engine
# -----------------------------

class EvaluationEngine:
    def __init__(
        self,
        failure_classifier: Optional[FailureClassifier] = None,
        history_path: str = "benchmark_history.json",
    ):
        self.failure_classifier = failure_classifier or FailureClassifier()
        self.history = BenchmarkHistory(history_path)

    def _load_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def _infer_ground_truth_label_column(self, df: pd.DataFrame) -> Optional[str]:
        # Ground truth can be boolean, 0/1, label strings, etc.
        # We attempt common column names.
        for cand in [
            "ground_truth",
            "ground-truth",
            "label",
            "labels",
            "y_true",
            "true_label",
            "gold",
            "gold_label",
            "decision_ground_truth",
        ]:
            col = _first_existing_column(df, [cand])
            if col:
                return col
        # Fallback: if the GT CSV has only decision column.
        return _first_existing_column(df, ["decision", "label", "ground_truth"])

    def _infer_screened_prediction_column(self, df: pd.DataFrame) -> Optional[str]:
        return _first_existing_column(df, ["decision", "screened_decision", "predicted", "prediction"])

    def _infer_title_column(self, df: pd.DataFrame) -> Optional[str]:
        return _choose_title_field(df)

    def _infer_gt_positive(self, gt_value: Any) -> Optional[int]:
        # Prefer mapping from generic decision tokens.
        mapped = map_screened_decision_to_positive(gt_value)
        if mapped is not None:
            return mapped

        # Boolean-ish numeric.
        nv = _normalize_label(gt_value)
        if nv in {"1", "true", "t", "yes", "y", "positive"}:
            return 1
        if nv in {"0", "false", "f", "no", "n", "negative"}:
            return 0

        return None

    def _trend_deltas(
        self,
        benchmark_name: str,
        metrics: BenchmarkMetrics,
        failure_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        history = [
            record
            for record in self.history.load()
            if record.get("benchmark_name") == benchmark_name
        ]
        if not history:
            return {}

        previous = history[-1]
        previous_failures = previous.get("failure_counts", {}) or {}
        deltas: Dict[str, Any] = {
            "accuracy_delta": round(metrics.accuracy - float(previous.get("accuracy", 0.0)), 4),
            "f1_delta": round(metrics.f1 - float(previous.get("F1", 0.0)), 4),
        }

        for category, count in failure_counts.items():
            old_count = int(previous_failures.get(category, 0) or 0)
            deltas[f"{category}_delta"] = count - old_count

        return deltas

    @staticmethod
    def _build_failure_clusters(analyses: Sequence[FailureAnalysis]) -> Dict[str, List[Dict[str, Any]]]:
        task_pairs: Counter[Tuple[str, str]] = Counter()
        technology_pairs: Counter[Tuple[str, str]] = Counter()

        for analysis in analyses:
            evidence = analysis.evidence
            left_task = str(evidence.get("canonical_task_left") or evidence.get("rq_task") or "").strip()
            right_task = str(evidence.get("canonical_task_right") or evidence.get("paper_task") or "").strip()
            if left_task and right_task and left_task != right_task:
                task_pairs[(left_task, right_task)] += 1

            left_tech = str(evidence.get("rq_technology") or "").strip()
            right_tech = str(evidence.get("paper_technology") or "").strip()
            if left_tech and right_tech and left_tech.lower() != right_tech.lower():
                technology_pairs[(left_tech, right_tech)] += 1

        def _format(counter: Counter[Tuple[str, str]]) -> List[Dict[str, Any]]:
            return [
                {"left": left, "right": right, "count": count}
                for (left, right), count in counter.most_common(10)
            ]

        return {
            "Task Pairs": _format(task_pairs),
            "Technology Pairs": _format(technology_pairs),
        }

    @staticmethod
    def _build_ontology_suggestions(analyses: Sequence[FailureAnalysis]) -> List[Dict[str, Any]]:
        suggestions: List[Dict[str, Any]] = []
        clusters = EvaluationEngine._build_failure_clusters(analyses)

        for cluster in clusters.get("Technology Pairs", []):
            if cluster["count"] < 2:
                continue
            suggestions.append(
                {
                    **cluster,
                    "type": "technology_hierarchy_candidate",
                    "recommendation": (
                        "Review whether this repeated pair represents a parent-child, sibling, "
                        "or synonym technology relationship before adding ontology structure."
                    ),
                }
            )

        for cluster in clusters.get("Task Pairs", []):
            if cluster["count"] < 2:
                continue
            suggestions.append(
                {
                    **cluster,
                    "type": "task_boundary_candidate",
                    "recommendation": (
                        "Review whether this is a true research-task boundary or a missing "
                        "domain-independent synonym."
                    ),
                }
            )

        return suggestions[:10]

    @staticmethod
    def _recommendations_from_analyses(
        analyses: Sequence[FailureAnalysis],
        metrics: BenchmarkMetrics,
    ) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[FailureAnalysis]] = defaultdict(list)
        for analysis in analyses:
            grouped[analysis.category].append(analysis)

        recommendations: List[Dict[str, Any]] = []
        for category, items in grouped.items():
            strongest = max(items, key=lambda item: item.confidence)
            avg_confidence = sum(item.confidence for item in items) / len(items)
            recommendations.append(
                {
                    "category": category,
                    "failure_count": len(items),
                    "root_cause": strongest.root_cause,
                    "confidence": round(avg_confidence, 2),
                    "expected_impact": strongest.expected_gain
                    if len(items) < max(2, int(0.15 * metrics.total))
                    else "High",
                    "estimated_complexity": strongest.expected_complexity,
                    "suggestion": strongest.suggested_fix,
                    "responsible_module": strongest.responsible_module,
                    "priority": strongest.priority,
                    "origin": strongest.origin,
                    "propagation": strongest.propagation,
                }
            )

        recommendations.sort(
            key=lambda rec: (
                int(rec.get("priority", 99)),
                -int(rec.get("failure_count", 0)),
                -float(rec.get("confidence", 0.0)),
            )
        )
        return recommendations

    def evaluate(
        self,
        ground_truth_csv: str,
        screened_csv: str,
        benchmark_name: str = "benchmark",
        notes: str = "",
    ) -> BenchmarkReport:
        gt_df = self._load_csv(ground_truth_csv)
        sc_df = self._load_csv(screened_csv)

        if len(gt_df) == 0 or len(sc_df) == 0:
            raise ValueError("One of the CSVs is empty")

        gt_label_col = self._infer_ground_truth_label_column(gt_df)
        if not gt_label_col:
            raise ValueError("Could not infer ground-truth label column from ground truth CSV")

        pred_col = self._infer_screened_prediction_column(sc_df)
        if not pred_col:
            raise ValueError("Could not infer prediction/decision column from screened CSV")

        title_col_gt = self._infer_title_column(gt_df)
        title_col_sc = self._infer_title_column(sc_df)

        # Default alignment: row index.
        n = min(len(gt_df), len(sc_df))

        tp = fp = tn = fn = 0
        failure_counts: Dict[str, int] = {}
        root_cause_counts: Dict[str, int] = {}
        examples_by_category: Dict[str, List[FailureExample]] = {}
        all_analyses: List[FailureAnalysis] = []

        for i in range(n):
            gt_row = gt_df.iloc[i]
            sc_row = sc_df.iloc[i]

            gt_pos = self._infer_gt_positive(gt_row.get(gt_label_col))
            if gt_pos is None:
                continue

            pred_pos = map_screened_decision_to_positive(sc_row.get(pred_col))
            if pred_pos is None:
                continue

            if pred_pos == 1 and gt_pos == 1:
                tp += 1
                continue
            if pred_pos == 1 and gt_pos == 0:
                fp += 1
            elif pred_pos == 0 and gt_pos == 0:
                tn += 1
                continue
            elif pred_pos == 0 and gt_pos == 1:
                fn += 1

            # Error case: build merged row for telemetry-based failure classification.
            merged = pd.concat([gt_row.add_prefix("gt_"), sc_row.add_prefix("sc_")])
            analyses = self.failure_classifier.analyze_failure(
                merged,
                predicted=pred_pos,
                ground_truth=gt_pos,
            )
            all_analyses.extend(analyses)
            cats = [analysis.category for analysis in analyses]

            for analysis in analyses:
                cat = analysis.category
                failure_counts[cat] = failure_counts.get(cat, 0) + 1
                root_cause_counts[analysis.root_cause] = root_cause_counts.get(analysis.root_cause, 0) + 1
                if len(examples_by_category.setdefault(cat, [])) >= 5:
                    continue

                title = ""
                if title_col_sc:
                    title = str(sc_row.get(title_col_sc) or "")
                elif title_col_gt:
                    title = str(gt_row.get(title_col_gt) or "")

                examples_by_category[cat].append(
                    FailureExample(
                        paper_title=title,
                        predicted=pred_pos,
                        ground_truth=gt_pos,
                        categories=cats,
                        row=merged.to_dict(),
                        analyses=[
                            {
                                "category": item.category,
                                "root_cause": item.root_cause,
                                "confidence": item.confidence,
                                "responsible_module": item.responsible_module,
                                "suggested_fix": item.suggested_fix,
                                "origin": item.origin,
                                "propagation": item.propagation,
                                "final_failure": item.final_failure,
                                "evidence": item.evidence,
                            }
                            for item in analyses
                        ],
                    )
                )

        metrics = _compute_binary_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
        top_bottlenecks = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
        recommendations = self._recommendations_from_analyses(all_analyses, metrics)
        failure_clusters = self._build_failure_clusters(all_analyses)
        ontology_suggestions = self._build_ontology_suggestions(all_analyses)
        trend_deltas = self._trend_deltas(benchmark_name, metrics, failure_counts)

        report = BenchmarkReport(
            metrics=metrics,
            failure_breakdown=failure_counts,
            top_bottlenecks=top_bottlenecks,
            examples_by_category=examples_by_category,
            recommendations=recommendations,
            meta={
                "benchmark_name": benchmark_name,
                "ground_truth_csv": ground_truth_csv,
                "screened_csv": screened_csv,
                "timestamp": _utc_now_iso(),
                "notes": notes,
            },
            root_cause_breakdown=root_cause_counts,
            failure_clusters=failure_clusters,
            ontology_suggestions=ontology_suggestions,
            trend_deltas=trend_deltas,
        )

        # Persist history for trends.
        self.history.append(
            {
                "timestamp": report.meta["timestamp"],
                "benchmark_name": benchmark_name,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "F1": metrics.f1,
                "failure_counts": dict(top_bottlenecks),
                "top_bottleneck": top_bottlenecks[0][0] if top_bottlenecks else None,
                "notes": notes,
            }
        )

        return report


def evaluate_benchmark(
    ground_truth_csv: str,
    screened_csv: str,
    benchmark_name: str = "benchmark",
    notes: str = "",
    history_path: str = "benchmark_history.json",
) -> str:
    engine = EvaluationEngine(history_path=history_path)
    report = engine.evaluate(
        ground_truth_csv=ground_truth_csv,
        screened_csv=screened_csv,
        benchmark_name=benchmark_name,
        notes=notes,
    )
    return report.to_text()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LitSync automated benchmark evaluation engine")
    parser.add_argument("--ground_truth_csv", required=True)
    parser.add_argument("--screened_csv", required=True)
    parser.add_argument("--benchmark_name", default="benchmark")
    parser.add_argument("--notes", default="")
    parser.add_argument("--history_path", default="benchmark_history.json")
    args = parser.parse_args()

    print(
        evaluate_benchmark(
            ground_truth_csv=args.ground_truth_csv,
            screened_csv=args.screened_csv,
            benchmark_name=args.benchmark_name,
            notes=args.notes,
            history_path=args.history_path,
        )
    )


if __name__ == "__main__":
    main()

