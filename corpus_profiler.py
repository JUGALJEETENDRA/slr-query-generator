from collections import Counter

from domain_vocabulary import (
    CONTEXT_GROUPS,
    EVIDENCE_TERMS,
    METHOD_GROUPS,
    TASK_GROUPS,
    find_terms,
    join_terms,
    split_terms,
)
from review_workflow_ontology import classify_review_intent


def _top_terms(counter, limit=30):
    return join_terms(term for term, _ in counter.most_common(limit))


def profile_corpus(rows, title_col, abstract_col, sample_size=30, rq_frame=None):
    method_counter = Counter()
    task_counter = Counter()
    context_counter = Counter()
    evidence_counter = Counter()
    review_context_counter = Counter()
    workflow_counter = Counter()
    ai_tool_counter = Counter()
    external_domain_counter = Counter()
    technology_subject_counter = Counter()
    automation_intent_counter = Counter()
    relation_cluster_counter = Counter()
    tool_use_counter = Counter()

    for _, row in rows.head(sample_size).iterrows():
        text = f"{row.get(title_col, '')} {row.get(abstract_col, '')}"
        for terms in METHOD_GROUPS.values():
            method_counter.update(find_terms(text, terms))
        for terms in TASK_GROUPS.values():
            task_counter.update(find_terms(text, terms))
        for terms in CONTEXT_GROUPS.values():
            context_counter.update(find_terms(text, terms))
        evidence_counter.update(find_terms(text, EVIDENCE_TERMS))
        review = classify_review_intent({"primary_subject": text})
        review_context_counter.update(split_terms(review["review_context_terms"]))
        workflow_counter.update(split_terms(review["review_workflow_task_terms"]))
        ai_tool_counter.update(split_terms(review["ai_tool_terms"]))
        external_domain_counter.update(split_terms(review["external_domain_terms"]))
        automation_intent_counter.update(split_terms(review["review_automation_intent_terms"]))
        if review["technology_subject_review_detected"]:
            technology_subject_counter.update(split_terms(review["ai_tool_terms"]))
        relation_cluster_counter.update([review["review_intent_relation"]])
        if review["ai_tool_for_review_workflow"]:
            tool_use_counter.update(split_terms(review["review_workflow_task_terms"]))

    # Corpus terms may expand a dimension only when their family overlaps that
    # dimension in the RQ. This prevents unrelated frequent terms becoming RQ synonyms.
    if rq_frame:
        aligned = {
            "method": set(split_terms(rq_frame.get("method_synonyms", ""))),
            "task": set(split_terms(rq_frame.get("task_outcome_synonyms", ""))),
            "context": set(split_terms(rq_frame.get("context_synonyms", ""))),
        }
        method_counter = Counter({k: v for k, v in method_counter.items() if k in aligned["method"]})
        task_counter = Counter({k: v for k, v in task_counter.items() if k in aligned["task"]})
        context_counter = Counter({k: v for k, v in context_counter.items() if k in aligned["context"]})

    profile_terms = Counter()
    profile_terms.update(method_counter)
    profile_terms.update(task_counter)
    profile_terms.update(context_counter)
    profile_terms.update(evidence_counter)

    return {
        "corpus_profile_terms": _top_terms(profile_terms),
        "corpus_method_terms": _top_terms(method_counter),
        "corpus_task_terms": _top_terms(task_counter),
        "corpus_context_terms": _top_terms(context_counter),
        "corpus_evidence_terms": _top_terms(evidence_counter),
        "corpus_domain_specific_synonyms": _top_terms(
            method_counter + task_counter + context_counter
        ),
        "corpus_review_context_terms": _top_terms(review_context_counter),
        "corpus_workflow_task_terms": _top_terms(workflow_counter),
        "corpus_ai_tool_terms": _top_terms(ai_tool_counter),
        "corpus_external_domain_terms": _top_terms(external_domain_counter),
        "corpus_technology_subject_terms": _top_terms(technology_subject_counter),
        "corpus_automation_intent_terms": _top_terms(automation_intent_counter),
        "corpus_review_workflow_terms": _top_terms(workflow_counter + automation_intent_counter),
        "corpus_tool_use_terms": _top_terms(tool_use_counter),
        "corpus_subject_review_terms": _top_terms(technology_subject_counter),
        "corpus_relation_clusters": _top_terms(relation_cluster_counter),
    }
