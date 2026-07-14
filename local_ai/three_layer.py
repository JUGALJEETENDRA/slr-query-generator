from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .cache import JsonDiskCache, cache_key
from .contracts import (
    CriterionEvidence,
    EvidenceSpan,
    PaperAssessment,
    ReviewProtocol,
    SCHEMA_VERSION,
    ValidationReport,
    safe_maybe,
)
from .engine import GenerationResult, LocalAIError, OllamaStructuredEngine
from .evidence import build_evidence_units, evidence_lookup
from .hardware import RuntimeProfile, resolve_runtime_profile
from .prompts import protocol_critic_prompt, protocol_prompt
from .validator import validate_assessment


THREE_LAYER_PROMPT_VERSION = "local-resident-three-layer-v3.4"
TRIAGE_MODEL = os.getenv("LOCAL_TRIAGE_MODEL", "qwen2.5:3b")
DEEP_MODEL = os.getenv("LOCAL_DEEP_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
EDGE_MODEL = os.getenv("LOCAL_EDGE_MODEL", DEEP_MODEL)
TRIAGE_BATCH_SIZE = int(os.getenv("LOCAL_TRIAGE_BATCH_SIZE", "4"))
DEEP_BATCH_SIZE = int(os.getenv("LOCAL_DEEP_BATCH_SIZE", "4"))
EDGE_BATCH_SIZE = int(os.getenv("LOCAL_EDGE_BATCH_SIZE", "4"))
RISK = Literal["LOW", "BORDERLINE", "HIGH"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TriageItem(_Strict):
    p: str = Field(min_length=1, max_length=80)
    d: Literal["KEEP", "MAYBE", "REJECT"]
    k: RISK
    b: Literal["S", "X", "C", "U"]
    e: list[str] = Field(default_factory=list, max_length=2)


class TriageBatch(_Strict):
    items: list[TriageItem]


class CompactCriterion(_Strict):
    c: str = Field(min_length=1, max_length=80)
    v: Literal["MET", "NOT_MET", "UNCLEAR"]
    e: str = Field(default="", max_length=40)
    r: str = Field(min_length=1, max_length=180)


class AssessmentItem(_Strict):
    p: str = Field(min_length=1, max_length=80)
    d: Literal["KEEP", "MAYBE", "REJECT"]
    k: RISK
    r: str = Field(min_length=1, max_length=240)
    c: list[CompactCriterion]


class AssessmentBatch(_Strict):
    items: list[AssessmentItem]


class CriticBatch(AssessmentBatch):
    pass


@dataclass
class LayerResult:
    result: dict[str, Any]
    assessment: PaperAssessment | None
    validation: ValidationReport | None
    elapsed_seconds: float
    original_elapsed_seconds: float
    cache_hit: bool = False


def _run_protocol_id(
    question: str, inclusion: str, exclusion: str, research_context: str = ""
) -> str:
    payload = json.dumps(
        [THREE_LAYER_PROMPT_VERSION, question, research_context, inclusion, exclusion],
        ensure_ascii=False,
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def _paper_payload(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "p": str(paper["id"]),
        "u": build_evidence_units(str(paper.get("title", "")), str(paper.get("abstract", ""))),
    }


def triage_batch_prompt(
    research_question: str,
    inclusion_criteria: str,
    exclusion_criteria: str,
    papers: list[dict[str, Any]],
    research_context: str = "",
    protocol: ReviewProtocol | None = None,
) -> str:
    if protocol is not None:
        screening_contract = {
            "q": protocol.research_question,
            "objective": protocol.objective,
            "scope": protocol.scope_interpretation,
            "criteria": [
                {"c": c.id, "kind": c.kind, "required": c.required, "description": c.description}
                for c in protocol.criteria
            ],
            "relationships": protocol.expected_relationships,
        }
    else:
        screening_contract = {
            "q": research_question,
            "context": research_context,
            "include": inclusion_criteria,
            "exclude": exclusion_criteria,
        }
    payload = {
        "protocol": screening_contract,
        "required_p": [str(paper["id"]) for paper in papers],
        "papers": [_paper_payload(paper) for paper in papers],
    }
    return f"""You are the fast first-glance layer of a systematic-review screener.
Judge meaning and relationships, never keyword overlap. Apply the same supplied question and criteria to every
paper independently. KEEP only when direct evidence safely establishes every required relationship. REJECT only
when affirmative text establishes an explicit exclusion, contradiction, or clearly different relationship;
missing detail and silence are never enough. Otherwise MAYBE. k is decision risk: LOW only when the decision is
unmistakable, BORDERLINE when another careful reviewer might disagree, HIGH when evidence is weak.
Set b to S (direct support) for KEEP, X (explicit exclusion) or C (relationship contradiction) for REJECT,
and U (insufficient or ambiguous) for MAYBE. Use at most two exact evidence-unit IDs in e.
The p values are opaque identifiers. Copy every value from required_p exactly once; never rename, expand, prefix,
or renumber one.
Return JSON only and no hidden reasoning.

INPUT:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

OUTPUT SHAPE:
{{"items":[{{"p":"exact required_p value","d":"KEEP|MAYBE|REJECT","k":"LOW|BORDERLINE|HIGH","b":"S|X|C|U","e":["evidence_id"]}}]}}
"""


def assessment_batch_prompt(
    protocol: ReviewProtocol,
    papers: list[dict[str, Any]],
) -> str:
    compact_protocol = {
        "q": protocol.research_question,
        "criteria": [
            {"c": c.id, "kind": c.kind, "required": c.required, "description": c.description}
            for c in protocol.criteria
        ],
        "relationships": protocol.expected_relationships,
    }
    payload = {
        "protocol": compact_protocol,
        "required_p": [str(paper["id"]) for paper in papers],
        "papers": [_paper_payload(p) for p in papers],
    }
    return f"""You are the deep-review layer of a systematic-review screener.
Understand each title/abstract independently and apply every protocol criterion. Inclusion MET means required
evidence is present. Inclusion NOT_MET requires affirmative contradictory evidence; silence is UNCLEAR. Exclusion
MET means affirmative disqualifying evidence is present. Return one c item for every criterion, using at most one
exact evidence-unit ID in e (empty only when genuinely unclear). Resolve to KEEP or REJECT when evidence safely
supports it; otherwise MAYBE. k is LOW, BORDERLINE, or HIGH decision risk, not confidence. Keep the paper reason
to at most 18 words and each criterion rationale to at most 12 words.
The p values are opaque identifiers. Copy every value from required_p exactly once; never rename, expand, prefix,
or renumber one. Return JSON only, with no hidden reasoning.

INPUT:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

OUTPUT SHAPE:
{{"items":[{{"p":"exact required_p value","d":"KEEP|MAYBE|REJECT","k":"LOW|BORDERLINE|HIGH","r":"short reason","c":[{{"c":"criterion id","v":"MET|NOT_MET|UNCLEAR","e":"evidence id or empty","r":"short rationale"}}]}}]}}
"""


def critic_batch_prompt(
    protocol: ReviewProtocol,
    papers: list[dict[str, Any]],
    candidates: dict[str, LayerResult],
) -> str:
    compact_protocol = {
        "q": protocol.research_question,
        "criteria": [
            {"c": c.id, "kind": c.kind, "required": c.required, "description": c.description}
            for c in protocol.criteria
        ],
        "relationships": protocol.expected_relationships,
    }
    entries = []
    for paper in papers:
        paper_id = str(paper["id"])
        candidate = candidates[paper_id]
        entries.append({
            **_paper_payload(paper),
            "candidate": {
                "d": candidate.result.get("decision"),
                "k": candidate.result.get("decision_risk", "HIGH"),
                "r": candidate.result.get("reason", ""),
                "criteria": [
                    {"c": item.get("criterion_id"), "v": item.get("verdict"), "r": item.get("rationale")}
                    for item in candidate.result.get("criteria", [])
                ],
                "validation_errors": candidate.result.get("validation_errors", []),
            },
        })
    payload = {
        "protocol": compact_protocol,
        "required_p": [str(paper["id"]) for paper in papers],
        "papers": entries,
    }
    return f"""You are the final adversarial edge critic. Start fresh and try to falsify each candidate rather than
repeat it. For REJECT, demand affirmative evidence of exclusion or contradiction; absence of detail is not enough.
For KEEP, demand direct support for every required relationship rather than plausibility. Reverse an unsafe result
when evidence supports the other outcome. Use MAYBE when neither definitive outcome is evidence-safe. Return a
complete replacement with every criterion, at most one exact evidence-unit ID per criterion, and every requested p
exactly once. Keep the paper reason to at most 18 words and each criterion rationale to at most 12 words. k is LOW,
BORDERLINE, or HIGH final decision risk. The p values are opaque identifiers. Copy every value from required_p
exactly once; never rename, expand, prefix, or renumber one. JSON only; no hidden reasoning.

INPUT:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

OUTPUT SHAPE:
{{"items":[{{"p":"exact required_p value","d":"KEEP|MAYBE|REJECT","k":"LOW|BORDERLINE|HIGH","r":"short reason","c":[{{"c":"criterion id","v":"MET|NOT_MET|UNCLEAR","e":"evidence id or empty","r":"short rationale"}}]}}]}}
"""


class ThreeLayerLocalOrchestrator:
    """Throughput path: batched 3B triage -> batched 4B review -> batched 4B critic."""

    def __init__(
        self,
        profile: RuntimeProfile | None = None,
        triage_engine=None,
        deep_engine=None,
        cache: JsonDiskCache | None = None,
    ):
        self.profile = profile or resolve_runtime_profile()
        self.triage_profile = replace(
            self.profile, fast_model=TRIAGE_MODEL, strong_model=TRIAGE_MODEL,
            num_ctx=int(os.getenv("LOCAL_TRIAGE_CONTEXT", "4096")),
            concurrency=1, keep_alive="30m",
        )
        self.deep_profile = replace(
            self.profile, fast_model=DEEP_MODEL, strong_model=EDGE_MODEL,
            num_ctx=4096, concurrency=1, keep_alive="30m",
        )
        self.triage_engine = triage_engine or OllamaStructuredEngine(self.triage_profile)
        self.deep_engine = deep_engine or OllamaStructuredEngine(self.deep_profile)
        self._deep_model_active = False
        self.cache = cache or JsonDiskCache(os.getenv("LOCAL_AI_CACHE_PATH", "outputs/cache/local_ai"))

    def run_protocol_id(
        self, question: str, inclusion: str = "", exclusion: str = "", research_context: str = ""
    ) -> str:
        return _run_protocol_id(question, inclusion, exclusion, research_context)

    @staticmethod
    def _metrics(
        generation: GenerationResult | None,
        layer: str,
        batch_size: int,
        retry: int,
        error: str = "",
    ) -> dict[str, Any]:
        return {
            "layer": layer,
            "batch_size": batch_size,
            "retry": retry,
            "prompt_tokens": generation.prompt_tokens if generation else 0,
            "output_tokens": generation.output_tokens if generation else 0,
            "tokens_per_second": generation.tokens_per_second if generation else 0.0,
            "model_duration_seconds": generation.model_duration_seconds if generation else 0.0,
            "total_duration_seconds": generation.total_duration_seconds if generation else 0.0,
            "wall_seconds": generation.elapsed_seconds if generation else 0.0,
            "cache_status": "miss",
            "error": error[:300],
        }

    def _execute_batches(
        self,
        *,
        papers: list[dict[str, Any]],
        batch_size: int,
        layer: str,
        model: str,
        engine: Any,
        schema: type[BaseModel],
        prompt_factory: Callable[[list[dict[str, Any]]], str],
        normalize: Callable[[dict[str, Any], BaseModel, dict[str, Any], list[dict[str, Any]]], LayerResult],
        safe: Callable[[dict[str, Any], str, list[dict[str, Any]]], LayerResult],
        on_batch: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, LayerResult], list[dict[str, Any]]]:
        results: dict[str, LayerResult] = {}
        batch_events: list[dict[str, Any]] = []

        def run(
            group: list[dict[str, Any]], retry: int = 0,
            history: list[dict[str, Any]] | None = None,
        ) -> None:
            history = list(history or [])
            expected = {str(p["id"]): p for p in group}
            generation: GenerationResult | None = None
            error = ""
            parsed: dict[str, BaseModel] = {}
            prepared_prompt = prompt_factory(group)
            max_prompt_chars = int(os.getenv("LOCAL_AI_MAX_BATCH_PROMPT_CHARS", "14000"))
            if len(group) > 1 and len(prepared_prompt) > max_prompt_chars:
                midpoint = max(1, len(group) // 2)
                run(group[:midpoint], retry, history)
                run(group[midpoint:], retry, history)
                return
            try:
                generation = engine.generate(model, prepared_prompt, schema)
                container = schema.model_validate(generation.value)
                items = list(container.items)
                counts: dict[str, int] = {}
                for item in items:
                    item_id = str(item.p)
                    counts[item_id] = counts.get(item_id, 0) + 1
                for item in items:
                    item_id = str(item.p)
                    if item_id in expected and counts[item_id] == 1:
                        parsed[item_id] = item
                unknown = sorted({str(item.p) for item in items} - set(expected))
                duplicate = sorted(key for key, value in counts.items() if value > 1)
                missing = sorted(set(expected) - set(parsed))
                if unknown or duplicate or missing:
                    error = f"batch ids invalid; unknown={unknown}, duplicate={duplicate}, missing={missing}"
            except (LocalAIError, ValidationError, ValueError, TypeError) as exc:
                error = str(exc)
            metric = self._metrics(generation, layer, len(group), retry, error)
            metric["batch_id"] = f"{layer}-{len(batch_events) + 1}"
            batch_events.append(metric)

            invalid: list[dict[str, Any]] = []
            completed_decisions = {"KEEP": 0, "MAYBE": 0, "REJECT": 0}
            for paper_id, paper in expected.items():
                item = parsed.get(paper_id)
                if item is None:
                    invalid.append(paper)
                    continue
                try:
                    results[paper_id] = normalize(paper, item, metric, history + [metric])
                    decision = str(results[paper_id].result.get("decision") or "MAYBE")
                    completed_decisions[decision] = completed_decisions.get(decision, 0) + 1
                except (ValidationError, ValueError, KeyError) as exc:
                    invalid.append(paper)
                    metric["error"] = (metric["error"] + "; " + str(exc)).strip("; ")[:300]

            if on_batch:
                on_batch({
                    **dict(metric),
                    "completed_papers": sum(completed_decisions.values()),
                    "invalid_papers": len(invalid),
                    "decision_counts": completed_decisions,
                })

            if not invalid:
                return
            if len(invalid) == 1:
                if retry == 0:
                    run(invalid, retry + 1, history + [metric])
                    return
                results[str(invalid[0]["id"])] = safe(
                    invalid[0], error or metric["error"], history + [metric]
                )
                if on_batch:
                    on_batch({
                        **dict(metric), "completed_papers": 1, "invalid_papers": 0,
                        "decision_counts": {"KEEP": 0, "MAYBE": 1, "REJECT": 0},
                        "safe_failure": True,
                    })
                return
            midpoint = max(1, len(invalid) // 2)
            run(invalid[:midpoint], retry + 1, history + [metric])
            run(invalid[midpoint:], retry + 1, history + [metric])

        for start in range(0, len(papers), max(1, batch_size)):
            run(papers[start:start + max(1, batch_size)])
        return results, batch_events

    def triage_batch(
        self,
        question: str,
        papers: list[dict[str, Any]],
        inclusion: str = "",
        exclusion: str = "",
        research_context: str = "",
        protocol: ReviewProtocol | None = None,
        on_batch: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, LayerResult], list[dict[str, Any]]]:
        run_id = self.run_protocol_id(question, inclusion, exclusion, research_context)

        def normalize(paper, item: TriageItem, metric, paper_metrics):
            lookup = evidence_lookup(str(paper.get("title", "")), str(paper.get("abstract", "")))
            bad = [evidence_id for evidence_id in item.e if evidence_id not in lookup]
            if bad:
                raise ValueError("unknown evidence IDs: " + ", ".join(bad))
            if item.d in {"KEEP", "REJECT"} and not item.e:
                raise ValueError("definitive triage decision lacks evidence")
            allowed_basis = {
                "KEEP": {"S"},
                "REJECT": {"X", "C"},
                "MAYBE": {"U"},
            }
            if item.b not in allowed_basis[item.d]:
                raise ValueError(f"{item.d} conflicts with triage basis {item.b}")
            public_basis = {
                "S": "DIRECT_SUPPORT",
                "X": "EXPLICIT_EXCLUSION",
                "C": "RELATIONSHIP_CONTRADICTION",
                "U": "INSUFFICIENT_OR_AMBIGUOUS",
            }[item.b]
            reason = {
                "S": "Quick triage found direct evidence supporting the required relationship.",
                "X": "Quick triage found affirmative evidence of an explicit exclusion.",
                "C": "Quick triage found affirmative evidence of a contradictory relationship.",
                "U": "Quick triage found insufficient or ambiguous evidence for a safe decision.",
            }[item.b]
            evidence = [
                {"criterion_id": "quick_relevance", "source": lookup[e]["source"],
                 "quote": lookup[e]["text"], "evidence_id": e}
                for e in item.e
            ]
            transition = "final" if item.d in {"KEEP", "REJECT"} and item.k == "LOW" else "deep_review"
            recorded_metrics = [
                {**dict(value), "queue_transition": transition} for value in paper_metrics
            ]
            allocated = sum(
                float(value.get("wall_seconds") or 0.0) /
                max(1, int(value.get("batch_size") or 1))
                for value in paper_metrics
            )
            result = {
                "schema_version": SCHEMA_VERSION, "decision": item.d, "reason": reason,
                "confidence": {"LOW": 0.95, "BORDERLINE": 0.75, "HIGH": 0.5}[item.k],
                "decision_risk": item.k, "triage_basis": public_basis, "protocol_id": run_id,
                "criteria": [{"criterion_id": "quick_relevance",
                              "verdict": "MET" if item.d == "KEEP" else ("NOT_MET" if item.d == "REJECT" else "UNCLEAR"),
                              "rationale": reason,
                              "evidence": [{k: v for k, v in span.items() if k != "criterion_id"} for span in evidence]}],
                "evidence": evidence, "summary": reason,
                "uncertainty": [] if item.d != "MAYBE" else ["Deep review required."],
                "missing_information": [], "contradictions": [],
                "validation_status": "validated", "validation_errors": [], "validation_warnings": [],
                "escalated": False, "model": TRIAGE_MODEL, "model_tier": "resident_three_layer_local",
                "prompt_version": THREE_LAYER_PROMPT_VERSION, "attempts": 1,
                "cache_hit": False, "processing_seconds": round(allocated, 4),
                "original_processing_seconds": round(allocated, 4), "runtime_downgrades": [],
                "layer_trace": [{"layer": 1, "name": "quick_triage", "model": TRIAGE_MODEL,
                                 "decision": item.d, "risk": item.k, "basis": public_basis,
                                 "validation_status": "validated"}],
                "layer_metrics": recorded_metrics,
            }
            return LayerResult(result, None, ValidationReport(valid=True), allocated, allocated)

        def safe(paper, error, metrics):
            allocated = sum(float(m.get("wall_seconds") or 0.0) / max(1, int(m.get("batch_size") or 1)) for m in metrics)
            result = {
                "schema_version": SCHEMA_VERSION, "decision": "MAYBE",
                "reason": "Quick triage could not produce a safe structured result.",
                "confidence": 0.0, "decision_risk": "HIGH", "protocol_id": run_id,
                "triage_basis": "INSUFFICIENT_OR_AMBIGUOUS",
                "criteria": [], "evidence": [], "summary": "Quick triage failed safely.",
                "uncertainty": [error or "Malformed batch output."], "missing_information": [],
                "contradictions": [], "validation_status": "unresolved",
                "validation_errors": [error or "Malformed batch output."], "validation_warnings": [],
                "escalated": False, "model": TRIAGE_MODEL, "model_tier": "resident_three_layer_local",
                "prompt_version": THREE_LAYER_PROMPT_VERSION, "attempts": len(metrics),
                "cache_hit": False, "processing_seconds": round(allocated, 4),
                "original_processing_seconds": round(allocated, 4), "runtime_downgrades": [],
                "layer_trace": [{"layer": 1, "name": "quick_triage", "model": TRIAGE_MODEL,
                                 "decision": "MAYBE", "risk": "HIGH",
                                 "basis": "INSUFFICIENT_OR_AMBIGUOUS",
                                 "validation_status": "unresolved"}],
                "layer_metrics": [
                    {**dict(value), "queue_transition": "deep_review"} for value in metrics
                ],
            }
            return LayerResult(result, None, ValidationReport(valid=False, errors=result["validation_errors"]), allocated, allocated)

        return self._execute_batches(
            papers=papers, batch_size=TRIAGE_BATCH_SIZE, layer="quick_triage", model=TRIAGE_MODEL,
            engine=self.triage_engine, schema=TriageBatch,
            prompt_factory=lambda group: triage_batch_prompt(
                question, inclusion, exclusion, group, research_context, protocol
            ),
            normalize=normalize, safe=safe,
            on_batch=on_batch,
        )

    def unload_triage(self) -> None:
        if hasattr(self.triage_engine, "unload"):
            self.triage_engine.unload(TRIAGE_MODEL)

    def unload_deep(self) -> None:
        if self._deep_model_active and hasattr(self.deep_engine, "unload"):
            self.deep_engine.unload(DEEP_MODEL)
        self._deep_model_active = False

    def prepare_edge_critic(self) -> None:
        if EDGE_MODEL != DEEP_MODEL:
            self.unload_deep()

    def compile_protocol(
        self, question: str, inclusion: str = "", exclusion: str = "", research_context: str = ""
    ) -> ReviewProtocol:
        key = cache_key(
            question, research_context, inclusion, exclusion,
            DEEP_MODEL, THREE_LAYER_PROMPT_VERSION, "protocol",
        )
        cached = self.cache.get("three_layer", key)
        if cached:
            protocol = ReviewProtocol.model_validate(cached)
            if protocol.prompt_version == THREE_LAYER_PROMPT_VERSION:
                return protocol
        initial = self.deep_engine.generate(
            DEEP_MODEL,
            protocol_prompt(question, inclusion, exclusion, research_context),
            ReviewProtocol,
        )
        self._deep_model_active = True
        protocol = ReviewProtocol.model_validate(
            self._normalize_protocol_provenance(
                initial.value, allow_user=bool(inclusion.strip() or exclusion.strip())
            )
        ).model_copy(update={
            "research_question": question, "research_context": research_context,
            "model": DEEP_MODEL, "prompt_version": THREE_LAYER_PROMPT_VERSION,
        })
        criticised = self.deep_engine.generate(
            DEEP_MODEL,
            protocol_critic_prompt(protocol, inclusion, exclusion, research_context),
            ReviewProtocol,
        )
        protocol = ReviewProtocol.model_validate(
            self._normalize_protocol_provenance(
                criticised.value, allow_user=bool(inclusion.strip() or exclusion.strip())
            )
        ).model_copy(update={
            "research_question": question, "research_context": research_context,
            "model": DEEP_MODEL, "prompt_version": THREE_LAYER_PROMPT_VERSION,
        })
        self._validate_protocol(protocol, inclusion, exclusion)
        protocol = protocol.with_identity()
        self.cache.set("three_layer", key, protocol.model_dump(mode="json"))
        return protocol

    @staticmethod
    def _normalize_protocol_provenance(
        value: dict[str, Any], *, allow_user: bool = True
    ) -> dict[str, Any]:
        """Normalize free model wording for the closed provenance enum."""
        normalized = dict(value)
        criteria = []
        for raw in value.get("criteria") or []:
            criterion = dict(raw)
            if criterion.get("source") != "user" or not allow_user:
                criterion["source"] = "research_question"
            if criterion.get("kind") == "exclusion" and criterion["source"] != "user":
                continue
            criteria.append(criterion)
        normalized["criteria"] = criteria
        return normalized

    @staticmethod
    def _validate_protocol(protocol: ReviewProtocol, inclusion: str, exclusion: str) -> None:
        if any(c.source == "research_question" and c.kind == "exclusion" for c in protocol.criteria):
            raise ValueError("RQ scope must use positive inclusion criteria")
        entries = lambda value: [
            item.strip(" -*\t") for line in value.splitlines() for item in line.split(";")
            if item.strip(" -*\t")
        ]
        expected = (len(entries(inclusion)), len(entries(exclusion)))
        actual = (
            sum(c.source == "user" and c.kind == "inclusion" for c in protocol.criteria),
            sum(c.source == "user" and c.kind == "exclusion" for c in protocol.criteria),
        )
        if expected != actual:
            raise ValueError("compiled protocol omitted authoritative user criteria")

    @staticmethod
    def _assessment_from_compact(item: AssessmentItem) -> PaperAssessment:
        criteria = []
        for compact in item.c:
            evidence = []
            if compact.e:
                source = "title" if compact.e.startswith("title_") else "abstract"
                evidence = [EvidenceSpan(source=source, evidence_id=compact.e)]
            criteria.append(CriterionEvidence(
                criterion_id=compact.c, verdict=compact.v, rationale=compact.r, evidence=evidence
            ))
        return PaperAssessment(
            summary=item.r, criteria=criteria, contradictions=[],
            missing_information=["Title/abstract evidence is insufficient."] if item.d == "MAYBE" else [],
            decision=item.d,
            confidence={"LOW": 0.95, "BORDERLINE": 0.75, "HIGH": 0.5}[item.k],
            reason=item.r,
            uncertainty=["The available title/abstract does not support a safe definitive decision."] if item.d == "MAYBE" else [],
        )

    def _assessment_batches(
        self,
        *,
        protocol: ReviewProtocol,
        run_id: str,
        papers: list[dict[str, Any]],
        layer: Literal["deep_review", "edge_critic"],
        candidates: dict[str, LayerResult] | None = None,
        on_batch: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, LayerResult], list[dict[str, Any]]]:
        schema: type[BaseModel] = AssessmentBatch if layer == "deep_review" else CriticBatch
        batch_size = DEEP_BATCH_SIZE if layer == "deep_review" else EDGE_BATCH_SIZE
        prior = candidates or {}

        def normalize(paper, item: AssessmentItem, metric, paper_metrics):
            assessment = self._assessment_from_compact(item)
            validation = validate_assessment(
                assessment, protocol, str(paper.get("title", "")), str(paper.get("abstract", ""))
            )
            if not validation.valid:
                return safe(paper, "; ".join(validation.errors), paper_metrics)
            allocated = sum(
                float(value.get("wall_seconds") or 0.0) /
                max(1, int(value.get("batch_size") or 1))
                for value in paper_metrics
            )
            previous = prior.get(str(paper["id"]))
            result = self._assessment_public(
                assessment, validation, protocol, run_id, paper, layer,
                EDGE_MODEL if layer == "edge_critic" else DEEP_MODEL, item.k, allocated,
                previous,
            )
            transition = (
                "final" if layer == "edge_critic"
                else ("final" if item.d != "MAYBE" and item.k == "LOW" else "edge_critic")
            )
            result["layer_metrics"] = (
                list(previous.result.get("layer_metrics", [])) if previous else []
            ) + [{**dict(value), "queue_transition": transition} for value in paper_metrics]
            return LayerResult(result, assessment, validation, allocated, allocated)

        def safe(paper, error, metrics):
            assessment = safe_maybe(f"{layer} could not produce an evidence-safe result: {error}")
            validation = ValidationReport(valid=False, errors=[error or "Malformed batch output."])
            allocated = sum(float(m.get("wall_seconds") or 0.0) / max(1, int(m.get("batch_size") or 1)) for m in metrics)
            previous = prior.get(str(paper["id"]))
            result = self._assessment_public(
                assessment, validation, protocol, run_id, paper, layer,
                EDGE_MODEL if layer == "edge_critic" else DEEP_MODEL, "HIGH", allocated,
                previous,
            )
            result.update(decision="MAYBE", validation_status="unresolved", validation_errors=validation.errors)
            transition = "final" if layer == "edge_critic" else "edge_critic"
            result["layer_metrics"] = (
                list(previous.result.get("layer_metrics", [])) if previous else []
            ) + [{**dict(value), "queue_transition": transition} for value in metrics]
            return LayerResult(result, assessment, validation, allocated, allocated)

        prompt_factory = (
            (lambda group: assessment_batch_prompt(protocol, group))
            if layer == "deep_review"
            else (lambda group: critic_batch_prompt(protocol, group, prior))
        )
        return self._execute_batches(
            papers=papers, batch_size=batch_size, layer=layer,
            model=DEEP_MODEL if layer == "deep_review" else EDGE_MODEL,
            engine=self.deep_engine, schema=schema, prompt_factory=prompt_factory,
            normalize=normalize, safe=safe,
            on_batch=on_batch,
        )

    def _assessment_public(
        self, assessment, validation, protocol, run_id, paper, layer, model, risk, elapsed, previous,
    ) -> dict[str, Any]:
        units = evidence_lookup(str(paper.get("title", "")), str(paper.get("abstract", "")))
        criteria, evidence = [], []
        for item in assessment.criteria:
            spans = []
            for reference in item.evidence:
                unit = units.get(reference.evidence_id)
                if unit and unit["source"] == reference.source:
                    span = {"source": unit["source"], "quote": unit["text"], "evidence_id": reference.evidence_id}
                    spans.append(span)
                    evidence.append({"criterion_id": item.criterion_id, **span})
            criteria.append({**item.model_dump(mode="json"), "evidence": spans})
        prior_trace = list(previous.result.get("layer_trace", [])) if previous else []
        trace = prior_trace + [{
            "layer": 2 if layer == "deep_review" else 3, "name": layer, "model": model,
            "decision": assessment.decision, "risk": risk,
            "validation_status": "validated" if validation.valid else "unresolved",
        }]
        prior_seconds = float(previous.result.get("processing_seconds") or 0.0) if previous else 0.0
        return {
            "schema_version": SCHEMA_VERSION, "decision": assessment.decision,
            "reason": assessment.reason, "confidence": assessment.confidence, "decision_risk": risk,
            "protocol_id": run_id, "criteria": criteria, "evidence": evidence,
            "summary": assessment.summary, "uncertainty": assessment.uncertainty,
            "missing_information": assessment.missing_information, "contradictions": assessment.contradictions,
            "validation_status": "validated" if validation.valid else "unresolved",
            "validation_errors": validation.errors, "validation_warnings": validation.warnings,
            "escalated": True, "model": model, "model_tier": "resident_three_layer_local",
            "prompt_version": THREE_LAYER_PROMPT_VERSION, "attempts": len(trace), "cache_hit": False,
            "processing_seconds": round(prior_seconds + elapsed, 4),
            "original_processing_seconds": round(prior_seconds + elapsed, 4),
            "runtime_downgrades": [], "layer_trace": trace, "internal_protocol_id": protocol.protocol_id,
        }

    def deep_review_batch(self, protocol, run_id, papers, triage_results, on_batch=None):
        self._deep_model_active = True
        return self._assessment_batches(
            protocol=protocol, run_id=run_id, papers=papers,
            layer="deep_review", candidates=triage_results, on_batch=on_batch,
        )

    def edge_critic_batch(self, protocol, run_id, papers, deep_results, on_batch=None):
        return self._assessment_batches(
            protocol=protocol, run_id=run_id, papers=papers,
            layer="edge_critic", candidates=deep_results, on_batch=on_batch,
        )

    @staticmethod
    def needs_deep_review(layer: LayerResult) -> bool:
        return (
            layer.result.get("decision") in {"MAYBE", "REJECT"}
            or layer.result.get("decision_risk") != "LOW"
            or layer.result.get("validation_status") != "validated"
        )

    @staticmethod
    def needs_edge_critic(layer: LayerResult) -> bool:
        return (
            layer.result.get("decision") == "MAYBE"
            or layer.result.get("decision_risk") != "LOW"
            or layer.result.get("validation_status") != "validated"
        )

    # Single-paper compatibility for /screen. It uses the same batch contracts with a batch of one.
    def screen_paper(
        self, research_question, title, abstract, inclusion_criteria="", exclusion_criteria="",
        research_context="",
    ):
        run_id = self.run_protocol_id(
            research_question, inclusion_criteria, exclusion_criteria, research_context
        )
        try:
            protocol = self.compile_protocol(
                research_question, inclusion_criteria, exclusion_criteria, research_context
            )
        except (LocalAIError, ValidationError, ValueError) as exc:
            return {
                "schema_version": SCHEMA_VERSION, "decision": "MAYBE",
                "reason": f"Research protocol setup failed: {str(exc)[:300]}",
                "confidence": 0.0, "decision_risk": "HIGH", "protocol_id": run_id,
                "criteria": [], "evidence": [], "uncertainty": [str(exc)],
                "validation_status": "unresolved", "validation_errors": [str(exc)],
                "model_tier": "resident_three_layer_local", "model": DEEP_MODEL,
                "prompt_version": THREE_LAYER_PROMPT_VERSION, "processing_seconds": 0.0,
                "original_processing_seconds": 0.0, "cache_hit": False,
                "runtime_downgrades": [], "layer_trace": [], "layer_metrics": [],
                "escalated": False,
            }
        self.unload_deep()
        paper = {"id": "paper_1", "title": title, "abstract": abstract}
        triage, _ = self.triage_batch(
            research_question, [paper], inclusion_criteria, exclusion_criteria,
            research_context, protocol,
        )
        first = triage["paper_1"]
        if not self.needs_deep_review(first):
            return first.result
        self.unload_triage()
        try:
            deep, _ = self.deep_review_batch(protocol, run_id, [paper], triage)
            second = deep["paper_1"]
            if not self.needs_edge_critic(second):
                return second.result
            self.prepare_edge_critic()
            edge, _ = self.edge_critic_batch(protocol, run_id, [paper], deep)
            return edge["paper_1"].result
        except (LocalAIError, ValidationError, ValueError) as exc:
            result = dict(first.result)
            result.update(
                decision="MAYBE", reason=f"4B review unavailable: {str(exc)[:300]}",
                validation_status="unresolved", validation_errors=[str(exc)],
            )
            return result
