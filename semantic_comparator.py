from task_ontology import RESEARCH_TASK_ONTOLOGY, canonicalize_task


MODEL_NAME = "all-MiniLM-L6-v2"

# Global model instance (None until first use)
MODEL = None


def _get_model():
    global MODEL

    if MODEL is None:
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


def _is_known_task_identity(value):
    return value in RESEARCH_TASK_ONTOLOGY


def _similarity(left, right):
    left = left.strip()
    right = right.strip()

    if not left or not right:
        return 0.0

    model = _get_model()
    embeddings = model.encode([left, right])
    return _cosine_similarity(embeddings[0], embeddings[1])


def _encode_by_index(texts):
    model = _get_model()
    indexed_texts = [(index, text) for index, text in enumerate(texts) if text]
    embeddings = {}

    if indexed_texts:
        encoded = model.encode([text for _, text in indexed_texts])
        embeddings = {
            index: encoded[position]
            for position, (index, _) in enumerate(indexed_texts)
        }

    return embeddings


def _pair_similarity(embeddings, left_index, right_index):
    if left_index not in embeddings or right_index not in embeddings:
        return 0.0
    return _cosine_similarity(embeddings[left_index], embeddings[right_index])


def _task_match_score(canonical_left, canonical_right, embedding_score):
    if canonical_left and canonical_right and canonical_left == canonical_right:
        return 1.0, True, False

    known_left = _is_known_task_identity(canonical_left)
    known_right = _is_known_task_identity(canonical_right)
    if known_left and known_right:
        return min(embedding_score, 0.49), False, True

    return embedding_score, False, False


def _hierarchical_decision(
    task_match,
    task_identity_match,
    task_identity_conflict,
    study_role_compatible,
    review_role_gate,
    evidence_compatible,
    technology_match,
    context_match,
):
    """Decision policy: MAYBE is the exception.

    Keep ontology/identity checks and role gating, but rebalance thresholds so
    that confident semantic evidence yields KEEP/REJECT.
    """

    # Hard gate: if the extracted review role indicates the paper is primarily
    # reviewing the technology itself, it should not be treated as a clear
    # relevance match.
    if not review_role_gate:
        # Only promote to KEEP if task evidence is extremely strong.
        if task_identity_match and study_role_compatible and evidence_compatible:
            return "KEEP"
        if task_match >= 0.68 and technology_match >= 0.70 and context_match >= 0.60:
            return "KEEP"
        # Otherwise it's ambiguous or irrelevant.
        return "MAYBE" if task_match >= 0.55 else "REJECT"

    # If we have an explicit canonical conflict, prefer rejection unless there
    # is genuinely strong supporting evidence.
    if task_identity_conflict:
        strong_support = (
            technology_match >= 0.78
            and context_match >= 0.70
            and evidence_compatible
            and task_match >= 0.62
        )
        if strong_support:
            return "MAYBE"  # contradiction exists; don't override to KEEP

        # Contradiction + weak signals => REJECT.
        if task_match <= 0.60 or not (study_role_compatible and evidence_compatible):
            return "REJECT"

        # Otherwise only mildly ambiguous.
        return "MAYBE"

    # Canonical task identity match: this should usually be KEEP when roles
    # and evidence expectations are compatible.
    if task_identity_match:
        if study_role_compatible and evidence_compatible:
            return "KEEP"
        return "MAYBE"

    # No identity match: rely on calibrated semantic evidence.
    strong_relevance = (
        task_match >= 0.66
        and study_role_compatible
        and evidence_compatible
        and (technology_match >= 0.60 or context_match >= 0.60)
    )
    if strong_relevance:
        return "KEEP"

    strong_irrelevance = (
        task_match <= 0.46
        and not (study_role_compatible and evidence_compatible)
    )
    if strong_irrelevance:
        return "REJECT"

    # Ambiguous zone: use supporting signals to decide.
    if task_match >= 0.54 and study_role_compatible and evidence_compatible:
        # Mildly compatible, but not strong enough for KEEP.
        return "MAYBE"

    if task_match <= 0.52 and (not evidence_compatible or not study_role_compatible):
        return "REJECT"

    return "MAYBE"


def _decision_reason(decision, review_role_gate, task_identity_match, task_identity_conflict):
    if decision == "KEEP":
        if task_identity_match:
            return "The paper's canonical research task matches the review question and the role constraints are compatible."
        return "The paper's task, evidence role, and review role align with the review question."
    if decision == "MAYBE":
        if not review_role_gate:
            return "The paper is related, but it appears to review the technology itself rather than use it for the review task."
        if task_identity_conflict:
            return "The paper uses a different canonical research task, but other semantic signals are close enough for manual review."
        return "The paper is partially aligned with the review question but needs manual review."
    if task_identity_conflict:
        return "The paper was rejected because its canonical research task differs from the review question."
    return "The paper's task and review role do not sufficiently match the review question."


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
            "decision": "REJECT",
            "reason": "Rejected because there was no semantic text available to compare."
        }

    embeddings = _encode_by_index(texts_to_encode)

    # Calculate cosine similarities for each pair
    technology_match = _pair_similarity(embeddings, 0, 1)
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
    )
    if task_identity_match:
        task_role_match = max(task_role_match, 1.0)
    elif task_identity_conflict:
        task_role_match = min(task_role_match, 0.49)

    # New: review_role_match – symbolic equality, not embedding-based
    review_role_match = _symbolic_match(rq_review_role, paper_review_role)
    study_role_match = _symbolic_match(rq_study_role, paper_study_role)
    study_role_compatible = _normalized_equal_or_missing(rq_study_role, paper_study_role)
    evidence_compatible = _normalized_equal_or_missing(rq_evidence_type, paper_evidence_type)

    # New: review-role gate – only papers that are NOT "technology_being_reviewed"
    review_role_gate = (
        paper_review_role != "technology_being_reviewed"
    )

    decision = _hierarchical_decision(
        task_match=task_match,
        task_identity_match=task_identity_match,
        task_identity_conflict=task_identity_conflict,
        study_role_compatible=study_role_compatible,
        review_role_gate=review_role_gate,
        evidence_compatible=evidence_compatible,
        technology_match=technology_match,
        context_match=context_match,
    )
    reason = _decision_reason(
        decision,
        review_role_gate,
        task_identity_match,
        task_identity_conflict,
    )

    return {
        "technology_match": technology_match,
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
        "decision": decision,
        "reason": reason,
    }

