from task_ontology import (
    RESEARCH_TASK_ONTOLOGY,
    canonicalize_task,
    compatible_task_families,
)
from method_ontology import compare_method_families
from relevance_policy import apply_relevance_policy
from collections import OrderedDict
from threading import Lock
from runtime_config import get_model_judge_config


MODEL_NAME = "all-MiniLM-L6-v2"

# Global model instance (None until first use)
MODEL = None
EMBEDDING_CACHE = OrderedDict()
EMBEDDING_CACHE_LOCK = Lock()
EMBEDDING_CACHE_MAX_SIZE = 4096


def _get_model():
    global MODEL

    if MODEL is not None:
        return MODEL

    if not get_model_judge_config()["enable_hf_model_loading"]:
        return None

    from sentence_transformers import SentenceTransformer

    MODEL = SentenceTransformer(MODEL_NAME)

    return MODEL


def _cosine_similarity(left_embedding, right_embedding):
    from sklearn.metrics.pairwise import cosine_similarity

    return float(cosine_similarity([left_embedding], [right_embedding])[0][0])


def _field(frame, name):
    value = frame.get(name, "")
    if value is None:
        return ""
    return str(value).strip()


def _semantic_unit(frame):
    parts = []
    task = _field(frame, "target_problem_or_task")
    study_role = _field(frame, "study_role")
    review_role = _field(frame, "review_role")

    if task:
        parts.append(f"task: {task}")
    if study_role:
        parts.append(f"study role: {study_role}")
    if review_role:
        parts.append(f"review role: {review_role}")

    return "\n".join(parts)


def _symbolic_match(left, right):
    return bool(left and right and left == right)


def _normalized_equal_or_missing(left, right):
    left = str(left or "").strip().lower()
    right = str(right or "").strip().lower()
    if not left or not right:
        return True
    return left == right


def _task_identities(value):
    return {
        item.strip()
        for item in str(value or "").split("|")
        if item.strip()
    }


def _is_known_task_identity(value):
    values = _task_identities(value)
    return bool(values) and all(item in RESEARCH_TASK_ONTOLOGY for item in values)


def _similarity(left, right):
    left = left.strip()
    right = right.strip()

    if not left or not right:
        return 0.0

    embeddings = _encode_by_index([left, right])
    return _pair_similarity(embeddings, 0, 1)


def _cached_embeddings(texts):
    unique_texts = list(dict.fromkeys(text for text in texts if text))
    found = {}
    missing = []
    with EMBEDDING_CACHE_LOCK:
        for text in unique_texts:
            cached = EMBEDDING_CACHE.get(text)
            if cached is None:
                missing.append(text)
            else:
                EMBEDDING_CACHE.move_to_end(text)
                found[text] = cached

    if missing:
        model = _get_model()
        if model is None:
            return found
        encoded = model.encode(missing)
        with EMBEDDING_CACHE_LOCK:
            for text, embedding in zip(missing, encoded):
                EMBEDDING_CACHE[text] = embedding
                EMBEDDING_CACHE.move_to_end(text)
                found[text] = embedding
            while len(EMBEDDING_CACHE) > EMBEDDING_CACHE_MAX_SIZE:
                EMBEDDING_CACHE.popitem(last=False)
    return found


def _encode_by_index(texts):
    indexed_texts = [(index, text) for index, text in enumerate(texts) if text]
    cached = _cached_embeddings([text for _, text in indexed_texts])
    return {
        index: cached[text]
        for index, text in indexed_texts
        if text in cached
    }


def clear_embedding_cache():
    with EMBEDDING_CACHE_LOCK:
        EMBEDDING_CACHE.clear()


def _pair_similarity(embeddings, left_index, right_index):
    if left_index not in embeddings or right_index not in embeddings:
        return 0.0
    return _cosine_similarity(embeddings[left_index], embeddings[right_index])


def _task_match_score(
    canonical_left,
    canonical_right,
    embedding_score,
    task_family_compatible=False,
):
    left_tasks = _task_identities(canonical_left)
    right_tasks = _task_identities(canonical_right)

    if left_tasks and right_tasks and left_tasks & right_tasks:
        return 1.0, True, False

    known_left = _is_known_task_identity(canonical_left)
    known_right = _is_known_task_identity(canonical_right)
    if known_left and known_right:
        if task_family_compatible:
            return max(embedding_score, 0.50), False, True
        return min(embedding_score, 0.49), False, True

    return embedding_score, False, False


def _review_workflow_gate_applies(rq_frame):
    question_type = _field(rq_frame, "question_type").lower()
    review_role = _field(rq_frame, "review_role").lower()
    return question_type == "review_workflow_automation" or review_role in {
        "screening",
        "study_selection",
        "search_strategy_generation",
        "deduplication",
        "data_extraction",
        "pico_extraction",
        "risk_of_bias",
        "evidence_synthesis",
        "review_assistance",
    }


def _compatible_evidence(rq_evidence_type, paper_evidence_type):
    rq_evidence_type = str(rq_evidence_type or "").strip().lower()
    paper_evidence_type = str(paper_evidence_type or "").strip().lower()
    if not rq_evidence_type or not paper_evidence_type:
        return True
    broad_markers = ("paper", "papers", "study", "studies", "evidence", "method", "methods")
    if any(marker in rq_evidence_type for marker in broad_markers):
        return True
    return rq_evidence_type == paper_evidence_type


def _hierarchical_decision(
    task_match,
    task_identity_match,
    task_identity_conflict,
    task_family_compatible,
    study_role_compatible,
    review_role_gate,
    evidence_compatible,
    technology_match,
    context_match,
    subject_match,
    task_family_score=0.0,
    method_family_compatible=False,
):
    decision, _ = _decision_details(
        task_match=task_match,
        task_identity_match=task_identity_match,
        task_identity_conflict=task_identity_conflict,
        task_family_compatible=task_family_compatible,
        task_family_score=task_family_score,
        method_family_compatible=method_family_compatible,
        study_role_compatible=study_role_compatible,
        review_role_gate=review_role_gate,
        evidence_compatible=evidence_compatible,
        technology_match=technology_match,
        context_match=context_match,
        subject_match=subject_match,
    )
    return decision


def _decision_details(
    task_match,
    task_identity_match,
    task_identity_conflict,
    task_family_compatible,
    task_family_score,
    study_role_compatible,
    review_role_gate,
    evidence_compatible,
    technology_match,
    context_match,
    subject_match,
    method_family_compatible=False,
):
    if not review_role_gate:
        return "REJECT", "review_workflow_gate_reject"

    if task_identity_conflict and task_family_compatible:
        domain_high = max(subject_match, context_match)
        domain_low = min(subject_match, context_match)
        strong_signals = (
            task_family_score >= 0.55
            and technology_match >= 0.50
            and domain_high >= 0.50
            and domain_low >= 0.35
        )
        medium_signals = (
            task_family_score >= 0.40
            and technology_match >= 0.35
            and domain_high >= 0.40
        )
        method_rescue_signals = (
            method_family_compatible
            and task_family_score >= 0.40
            and domain_high >= 0.40
        )
        if (
            method_rescue_signals
            and study_role_compatible
            and evidence_compatible
        ):
            return "KEEP", "task_family_method_rescue_keep"
        if method_family_compatible:
            if not study_role_compatible or not evidence_compatible:
                return "REJECT", "family_compatibility_role_mismatch_reject"
            if task_family_score >= 0.48 and domain_high >= 0.30:
                return "KEEP", "semantic_rescue_strong_keep"
            if domain_high >= 0.10:
                return "MAYBE", "semantic_rescue_floor_maybe"
            return "REJECT", "family_compatibility_domain_mismatch_reject"
        if (
            strong_signals
            and study_role_compatible
            and evidence_compatible
        ):
            return "KEEP", "task_family_strong_keep"
        if medium_signals and evidence_compatible:
            return "MAYBE", "task_family_medium_maybe"
        return "REJECT", "task_family_weak_reject"

    if task_identity_conflict:
        if technology_match >= 0.75 and context_match >= 0.65 and evidence_compatible:
            return "MAYBE", "canonical_conflict_manual_review"
        return "REJECT", "canonical_conflict_reject"

    if task_identity_match:
        if study_role_compatible and evidence_compatible:
            return "KEEP", "exact_task_strong_keep"
        return "MAYBE", "exact_task_weak_maybe"

    if (
        task_match >= 0.50
        and technology_match >= 0.55
        and (context_match >= 0.45 or subject_match >= 0.45)
        and study_role_compatible
        and evidence_compatible
    ):
        return "KEEP", "semantic_alignment_keep"

    if task_match >= 0.60 and study_role_compatible and evidence_compatible:
        return "KEEP", "task_similarity_keep"

    if task_match >= 0.50 or (
        task_match >= 0.45 and technology_match >= 0.65 and context_match >= 0.55
    ):
        return "MAYBE", "partial_semantic_alignment_maybe"

    return "REJECT", "weak_semantic_alignment_reject"


def _decision_reason(
    decision,
    decision_path,
    task_family_name="",
    technology_match=0.0,
    subject_match=0.0,
    context_match=0.0,
):
    family_label = task_family_name.replace("_", " ") or "related"
    if decision_path == "task_family_strong_keep":
        return (
            f"Kept because the paper has a compatible {family_label} task with strong "
            "method, subject, and application-context alignment."
        )
    if decision_path == "task_family_method_rescue_keep":
        return (
            f"Kept because a specific paper method belongs to the RQ's broader method "
            f"family and the {family_label} task and application-domain evidence align."
        )
    if decision_path == "semantic_rescue_strong_keep":
        return (
            "Included because the paper uses a specific method within the requested "
            "method family for a compatible target task with supporting domain evidence."
        )
    if decision_path == "semantic_rescue_floor_maybe":
        return (
            "Marked maybe because both method and task families are compatible, but "
            "the application-context evidence is not strong enough for automatic inclusion."
        )
    if decision_path == "family_compatibility_domain_mismatch_reject":
        return (
            "Rejected despite related task and method families because subject and "
            "application-context evidence indicate a clear domain mismatch."
        )
    if decision_path == "family_compatibility_role_mismatch_reject":
        return (
            "Rejected despite related task and method families because the study or "
            "evidence role is incompatible with the review question."
        )
    if decision_path == "task_family_medium_maybe":
        return (
            f"Marked maybe because the paper has a compatible {family_label} task and "
            "related semantic evidence, but the exact task or alignment strength differs."
        )
    if decision_path == "task_family_weak_reject":
        weakest = min(
            ("method", technology_match),
            ("subject", subject_match),
            ("application context", context_match),
            key=lambda item: item[1],
        )[0]
        return (
            f"Rejected despite {family_label} task compatibility because {weakest} "
            "alignment was too weak."
        )
    if decision_path == "review_workflow_gate_reject":
        return (
            "Rejected by the review-workflow role gate because the paper reviews the "
            "technology rather than applying it to an SLR workflow task."
        )
    if decision_path == "exact_task_strong_keep":
        return "Kept because the exact task and the method/domain evidence align."
    if decision_path == "exact_task_weak_maybe":
        return "Marked maybe because the exact task matches but method, domain, context, or evidence alignment is incomplete."
    if decision_path == "canonical_conflict_reject":
        return "Rejected because the canonical tasks conflict and method/context evidence does not overcome that conflict."
    if decision_path == "canonical_conflict_manual_review":
        return "Marked maybe because strong method and context evidence partially offsets a canonical task conflict."
    if decision == "KEEP":
        return "Kept because task, method, domain, and evidence signals align."
    if decision == "MAYBE":
        return "Marked maybe because semantic alignment is meaningful but incomplete."
    return "Rejected because the combined task, method, domain, and context evidence is too weak."


def compare_semantic_frames(rq_frame, paper_frame):
    # --- Batch Similarity Calculation ---
    # Prepare all pairs of text for a single batch encoding
    rq_tech = _field(rq_frame, "intervention_or_method")
    paper_tech = _field(paper_frame, "intervention_or_method")
    rq_task = _field(rq_frame, "target_problem_or_task")
    paper_task = _field(paper_frame, "target_problem_or_task")
    rq_subject = _field(rq_frame, "primary_subject")
    paper_subject = _field(paper_frame, "primary_subject")
    rq_context = _field(rq_frame, "application_context")
    # Edit 1: corrected to use paper_frame instead of rq_frame
    paper_context = _field(paper_frame, "application_context")
    rq_study_role = _field(rq_frame, "study_role")
    paper_study_role = _field(paper_frame, "study_role")
    rq_review_role = _field(rq_frame, "review_role")
    paper_review_role = _field(paper_frame, "review_role")
    rq_evidence_type = _field(rq_frame, "evidence_type")
    paper_evidence_type = _field(paper_frame, "evidence_type")
    canonical_task_left = canonicalize_task(rq_task)
    canonical_task_right = canonicalize_task(paper_task)
    method_family = compare_method_families(rq_tech, paper_tech, paper_task)
    family_context = " ".join(
        [rq_subject, paper_subject, rq_context, paper_context]
    )
    task_families = compatible_task_families(
        canonical_task_left,
        canonical_task_right,
        family_context,
    )
    task_family_compatible = bool(task_families)
    task_family_name = "|".join(sorted(task_families))
    rq_task_unit = _semantic_unit(rq_frame)
    paper_task_unit = _semantic_unit(paper_frame)

    # Texts to be encoded. Order matters for unpacking later.
    texts_to_encode = [
        rq_tech, paper_tech,
        rq_task, paper_task,
        rq_task, paper_subject,  # Diagnostic only; not used for decisions.
        rq_subject, paper_subject,
        rq_context, paper_context,
        rq_task_unit, paper_task_unit,
    ]

    # Filter out empty strings to avoid encoding them.
    valid_texts = [text for text in texts_to_encode if text]
    if not valid_texts:
        return {
            "technology_match": 0.0,
            "task_match": 0.0,
            "task_subject_match": 0.0,
            "task_role_match": 0.0,
            "subject_match": 0.0,
            "context_match": 0.0,
            "study_role_match": False,
            "review_role_match": False,
            "canonical_task_left": canonical_task_left,
            "canonical_task_right": canonical_task_right,
            "task_identity_match": False,
            "task_identity_conflict": False,
            "task_family_compatible": task_family_compatible,
            "task_family_match": task_family_name,
            "task_family_score": 0.0,
            **method_family,
            "effective_technology_match": 0.0,
            "decision_path": "no_semantic_text_reject",
            "semantic_rescue_applied": False,
            "semantic_rescue_reason": "",
            "reject_blocked_by_family_compatibility": False,
            "rejected_despite_task_family_compatibility": task_family_compatible,
            "review_workflow_gate_applied": _review_workflow_gate_applies(rq_frame),
            "comparison_diagnostic": "No semantic text was available to compare; check RQ and paper frame extraction.",
            "decision": "REJECT",
            "reason": "Rejected because there was no semantic text available to compare."
        }

    embeddings = _encode_by_index(texts_to_encode)

    # Calculate cosine similarities for each pair
    technology_match = _pair_similarity(embeddings, 0, 1)
    effective_technology_match = max(
        technology_match,
        method_family["method_family_confidence"]
        if method_family["method_family_compatible"]
        else 0.0,
    )
    task_vs_task_match = _pair_similarity(embeddings, 2, 3)
    task_vs_subject_match = _pair_similarity(embeddings, 4, 5)
    subject_match = _pair_similarity(embeddings, 6, 7)
    context_match = _pair_similarity(embeddings, 8, 9)
    task_role_match = _pair_similarity(embeddings, 10, 11)

    # Task match is task-to-task only. Related subjects are tracked separately
    # so they cannot hide a task boundary mismatch.
    task_match, task_identity_match, task_identity_conflict = _task_match_score(
        canonical_task_left,
        canonical_task_right,
        task_vs_task_match,
        task_family_compatible,
    )
    if task_identity_match:
        task_role_match = max(task_role_match, 1.0)
    elif task_identity_conflict and not task_family_compatible:
        task_role_match = min(task_role_match, 0.49)

    # New: review_role_match – symbolic equality, not embedding-based
    review_role_match = _symbolic_match(rq_review_role, paper_review_role)
    study_role_match = _symbolic_match(rq_study_role, paper_study_role)
    study_role_compatible = _normalized_equal_or_missing(rq_study_role, paper_study_role)
    evidence_compatible = _compatible_evidence(rq_evidence_type, paper_evidence_type)
    task_family_score = (
        (0.40 * effective_technology_match)
        + (0.30 * subject_match)
        + (0.30 * context_match)
    )

    # New: review-role gate – only papers that are NOT "technology_being_reviewed"
    review_workflow_gate_applied = _review_workflow_gate_applies(rq_frame)
    review_role_gate = (
        not review_workflow_gate_applied
        or paper_review_role != "technology_being_reviewed"
    )

    decision, decision_path = _decision_details(
        task_match=task_match,
        task_identity_match=task_identity_match,
        task_identity_conflict=task_identity_conflict,
        task_family_compatible=task_family_compatible,
        task_family_score=task_family_score,
        method_family_compatible=method_family["method_family_compatible"],
        study_role_compatible=study_role_compatible,
        review_role_gate=review_role_gate,
        evidence_compatible=evidence_compatible,
        technology_match=(
            effective_technology_match
            if task_family_compatible
            else technology_match
        ),
        context_match=context_match,
        subject_match=subject_match,
    )

    diagnostics = [f"Decision path: {decision_path}."]
    if not rq_task:
        diagnostics.append("RQ target task is empty; RQ frame extraction likely failed.")
    if not rq_tech:
        diagnostics.append("RQ method/technology is empty; technology match may be unreliable.")
    if not paper_task:
        diagnostics.append("Paper target task is empty; paper frame extraction likely failed.")
    if not review_role_gate:
        diagnostics.append("Review-workflow gate blocked a technology-review paper for a review-automation question.")
    if task_identity_conflict:
        if task_family_compatible:
            diagnostics.append(
                f"Canonical tasks differ but share compatible family {task_family_name}: "
                f"RQ={canonical_task_left}; paper={canonical_task_right}."
            )
        else:
            diagnostics.append(f"Canonical task mismatch: RQ={canonical_task_left}; paper={canonical_task_right}.")
    if task_family_compatible:
        diagnostics.append(
            f"Task-family score={task_family_score:.3f}; method={technology_match:.3f}; "
            f"subject={subject_match:.3f}; context={context_match:.3f}."
        )
    if method_family["method_family_compatible"]:
        diagnostics.append(
            f"Method-family match={method_family['method_family_match']}; "
            f"effective technology={effective_technology_match:.3f}. "
            f"{method_family['method_family_reason']}"
        )
    if decision_path in {"semantic_rescue_strong_keep", "semantic_rescue_floor_maybe"}:
        diagnostics.append(
            "Semantic rescue blocked a hard reject because task and method families "
            "are compatible and domain evidence is not clearly contradictory."
        )
    comparison_diagnostic = " ".join(diagnostics)
    semantic_rescue_applied = decision_path in {
        "semantic_rescue_strong_keep",
        "semantic_rescue_floor_maybe",
    }
    semantic_rescue_reason = (
        "A hard reject was blocked because task and method families are compatible "
        "and domain evidence is not clearly contradictory."
        if semantic_rescue_applied
        else ""
    )

    reason = _decision_reason(
        decision,
        decision_path,
        task_family_name,
        technology_match,
        subject_match,
        context_match,
    )

    result = {
        "technology_match": technology_match,
        "effective_technology_match": effective_technology_match,
        "task_match": task_match,
        "task_subject_match": task_vs_subject_match,
        "task_role_match": task_role_match,
        "subject_match": subject_match,
        "context_match": context_match,
        "study_role_match": study_role_match,
        "review_role_match": review_role_match,   # now a boolean (symbolic equality)
        "canonical_task_left": canonical_task_left,
        "canonical_task_right": canonical_task_right,
        "task_identity_match": task_identity_match,
        "task_identity_conflict": task_identity_conflict,
        "task_family_compatible": task_family_compatible,
        "task_family_match": task_family_name,
        "task_family_score": task_family_score,
        **method_family,
        "decision_path": decision_path,
        "semantic_rescue_applied": semantic_rescue_applied,
        "semantic_rescue_reason": semantic_rescue_reason,
        "reject_blocked_by_family_compatibility": semantic_rescue_applied,
        "rejected_despite_task_family_compatibility": (
            task_family_compatible
            and decision == "REJECT"
            and decision_path in {
                "task_family_weak_reject",
                "family_compatibility_domain_mismatch_reject",
                "family_compatibility_role_mismatch_reject",
            }
        ),
        "review_workflow_gate_applied": review_workflow_gate_applied,
        "comparison_diagnostic": comparison_diagnostic,
        "decision": decision,
        "reason": reason,
    }
    policy_result, policy_diagnostics = apply_relevance_policy(
        rq_frame,
        paper_frame,
        result,
    )
    result.update(policy_diagnostics)
    if policy_result is not None:
        result.update(policy_result)
    return result
