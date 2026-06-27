from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"

# Global model instance (None until first use)
MODEL = None


def _get_model():
    global MODEL

    if MODEL is None:
        MODEL = SentenceTransformer(MODEL_NAME)

    return MODEL


def _field(frame, name):
    value = frame.get(name, "")
    if value is None:
        return ""
    return str(value).strip()


def _similarity(left, right):
    left = left.strip()
    right = right.strip()

    if not left or not right:
        return 0.0

    model = _get_model()
    embeddings = model.encode([left, right])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])


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
    paper_context = _field(rq_frame, "application_context")  # note: this remains as in original

    # Texts to be encoded. Order matters for unpacking later.
    texts_to_encode = [
        rq_tech, paper_tech,
        rq_task, paper_task,
        rq_task, paper_subject,  # For the max() comparison
        rq_subject, paper_subject,
        rq_context, paper_context,
    ]

    # Filter out empty strings to avoid encoding them
    valid_texts = [text for text in texts_to_encode if text]
    if not valid_texts:
        return {
            "technology_match": 0.0,
            "task_match": 0.0,
            "subject_match": 0.0,
            "context_match": 0.0,
            "review_role_match": False,
            "decision": "REJECT",
            "reason": "No text to compare"
        }

    model = _get_model()
    embeddings = model.encode(texts_to_encode)  # <-- Target line

    # Calculate cosine similarities for each pair
    technology_match = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]) if rq_tech and paper_tech else 0.0
    task_vs_task_match = float(cosine_similarity([embeddings[2]], [embeddings[3]])[0][0]) if rq_task and paper_task else 0.0
    task_vs_subject_match = float(cosine_similarity([embeddings[4]], [embeddings[5]])[0][0]) if rq_task and paper_subject else 0.0
    subject_match = float(cosine_similarity([embeddings[6]], [embeddings[7]])[0][0]) if rq_subject and paper_subject else 0.0
    context_match = float(cosine_similarity([embeddings[8]], [embeddings[9]])[0][0]) if rq_context and paper_context else 0.0

    # Task match: best of target_problem_or_task vs target_problem_or_task or primary_subject
    task_match = max(task_vs_task_match, task_vs_subject_match)

    # New: review_role_match – symbolic equality, not embedding-based
    review_role_match = (
        _field(rq_frame, "review_role") == _field(paper_frame, "review_role")
    )

    # New: review-role gate – only papers that are NOT "technology_being_reviewed"
    review_role_gate = (
        _field(paper_frame, "review_role") != "technology_being_reviewed"
    )

    # Decision logic based on task_match and review_role_gate
    if task_match >= 0.60 and review_role_gate:
        decision = "KEEP"
        reason = "task+review_role_gate"
    elif task_match >= 0.50:
        decision = "MAYBE"
        reason = "task_match >= 0.45"
    else:
        decision = "REJECT"
        reason = "task_match below threshold"

    return {
        "technology_match": technology_match,
        "task_match": task_match,
        "subject_match": subject_match,
        "context_match": context_match,
        "review_role_match": review_role_match,   # now a boolean (symbolic equality)
        "decision": decision,
        "reason": reason,
    }


# Preload the model when the module is imported
_get_model()