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
    ScreeningRQFrame,
    SCHEMA_VERSION,
    ValidationReport,
    safe_maybe,
)
from .engine import GenerationResult, LocalAIError, OllamaStructuredEngine
from .evidence import build_evidence_units, evidence_lookup
from .hardware import RuntimeProfile, resolve_runtime_profile
from .prompts import protocol_critic_prompt, protocol_prompt
from .profiles import resolve_local_screening_profile
from .validator import validate_assessment


THREE_LAYER_PROMPT_VERSION = "local-semantic-boundary-v3.12"
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


class GroundedCompactCriterion(_Strict):
    c: str = Field(min_length=1, max_length=80)
    v: Literal["MET", "NOT_MET", "UNCLEAR"]
    e: list[str] = Field(default_factory=list, max_length=2)
    r: str = Field(min_length=1, max_length=180)


class GroundedAssessmentItem(_Strict):
    p: str = Field(min_length=1, max_length=80)
    d: Literal["KEEP", "MAYBE", "REJECT"]
    k: RISK
    r: str = Field(min_length=1, max_length=240)
    c: list[GroundedCompactCriterion]


class GroundedAssessmentBatch(_Strict):
    items: list[GroundedAssessmentItem]


class GroundedCriticBatch(GroundedAssessmentBatch):
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
    question: str, inclusion: str, exclusion: str, research_context: str = "",
    *, frame_id: str = "", profile_name: str = "baseline-v3.12",
) -> str:
    identity = THREE_LAYER_PROMPT_VERSION if profile_name == "baseline-v3.12" and not frame_id else profile_name
    payload = json.dumps(
        ([identity, question, research_context, inclusion, exclusion]
         if not frame_id else [identity, question, research_context, inclusion, exclusion, frame_id]),
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
    rq_frame: ScreeningRQFrame | None = None,
) -> str:
    if protocol is not None:
        screening_contract = {
            "q": protocol.research_question,
            "criteria": [
                {"c": c.id, "kind": c.kind, "required": c.required, "description": c.description}
                for c in protocol.criteria
            ],
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
        "rq_frame": rq_frame.compact_prompt_payload(triage=True) if rq_frame else None,
        "required_p": [str(paper["id"]) for paper in papers],
        "papers": [_paper_payload(paper) for paper in papers],
    }
    return f"""You are the fast first-glance layer of a systematic-review screener.
Judge meaning and relationships, never keyword overlap. Apply the same supplied question and criteria to every
paper independently. Before deciding, internally test every required criterion against this paper alone. A concept
mentioned as background, motivation, or general context does not establish that the paper studies or applies it.
Concepts mentioned separately do not establish the required relationship. Never substitute a related technology,
setting, population, task, or outcome for the one required by the protocol.
The original q and explicit user criteria are authoritative. Boundaries are advisory near-miss checks and must not
add any requirement that q or an explicit user criterion does not contain.
Before KEEP, identify q's central phenomenon, intervention, or subject, then reread the selected e units alone.
At least one selected unit must directly identify that central subject and the selected evidence must support q's
requested scope or relationship. Evidence about a similar method, capability, outcome, or setting does not qualify.
Never infer identity from an acronym or framework name. A central subject mentioned only as background, motivation,
limitation, or future work is not part of the paper's own contribution and cannot support KEEP.
When rq_frame is present, its required source spans clarify the RQ structure. Its advisory concepts and allowed
variants may clarify equivalence but can never broaden eligibility or replace the required relationship.
KEEP only when direct evidence safely establishes every required relationship as part of the paper's own aim,
method, result, application, or explicit review scope. For KEEP, choose evidence units that support the hardest and
most specific required relationship, not merely the broad topic. REJECT only
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
    rq_frame: ScreeningRQFrame | None = None,
) -> str:
    compact_protocol = {
        "q": protocol.research_question,
        "criteria": [
            {"c": c.id, "kind": c.kind, "required": c.required, "description": c.description}
            for c in protocol.criteria
        ],
        "advisory_near_miss_checks": protocol.semantic_boundaries,
    }
    payload = {
        "protocol": compact_protocol,
        "rq_frame": rq_frame.compact_prompt_payload() if rq_frame else None,
        "required_p": [str(paper["id"]) for paper in papers],
        "papers": [_paper_payload(p) for p in papers],
    }
    if rq_frame is None or rq_frame.frame_version != "local-rq-frame-v2":
        return f"""You are the deep-review layer of a systematic-review screener.
Understand each title/abstract independently and apply every protocol criterion. Test the most specific criterion
first. Inclusion MET means the selected evidence unit, read with the title, semantically entails that criterion as
part of the paper's own contribution or explicit review scope. Topical association, background discussion,
co-occurring concepts, or a neighboring technology/setting/relationship is not entailment. Inclusion NOT_MET
requires affirmative contradictory evidence; silence is UNCLEAR. Exclusion
MET means affirmative disqualifying evidence is present.
The original q and explicit user criteria are authoritative. Boundaries are advisory near-miss checks and cannot
create additional inclusion or exclusion requirements.
When rq_frame is present, preserve its required group relationships. Allowed variants are advisory interpretation
aids only and never independent evidence that the paper satisfies the RQ.
For rq_core_relationship MET, the cited unit must directly identify q's central phenomenon, intervention, or
subject and support its requested scope or relationship. Do not infer identity from similar functionality, an
acronym, a named framework, or a neighboring technique. Use only ids listed in protocol.criteria for c; never use
an advisory_near_miss_checks field name as a criterion id.
Return one c item for every criterion, using at most one exact evidence-unit ID in e. Every MET or NOT_MET must
have criterion-specific evidence; only UNCLEAR may use an
empty e. Resolve to KEEP or REJECT when evidence safely
supports it; otherwise MAYBE. k is LOW, BORDERLINE, or HIGH decision risk, not confidence. Keep the paper reason
to at most 18 words and each criterion rationale to at most 12 words.
The p values are opaque identifiers. Copy every value from required_p exactly once; never rename, expand, prefix,
or renumber one. Return JSON only, with no hidden reasoning.

INPUT:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

OUTPUT SHAPE:
{{"items":[{{"p":"exact required_p value","d":"KEEP|MAYBE|REJECT","k":"LOW|BORDERLINE|HIGH","r":"short reason","c":[{{"c":"criterion id","v":"MET|NOT_MET|UNCLEAR","e":"evidence id or empty","r":"short rationale"}}]}}]}}
"""
    evidence_rule = (
        "Use up to two exact evidence-unit IDs in e so the required concepts and their relationship may be "
        "grounded across adjacent units. Every MET or NOT_MET must have at least one ID; only UNCLEAR may use []."
        if rq_frame is not None and rq_frame.frame_version == "local-rq-frame-v2" else
        "Use at most one exact evidence-unit ID in e. Every MET or NOT_MET must have evidence; only UNCLEAR may use an empty e."
    )
    output_shape = (
        '{{"items":[{{"p":"exact required_p value","d":"KEEP|MAYBE|REJECT","k":"LOW|BORDERLINE|HIGH","r":"short reason","c":[{{"c":"criterion id","v":"MET|NOT_MET|UNCLEAR","e":["evidence_id"],"r":"short rationale"}}]}}]}}'
        if rq_frame is not None and rq_frame.frame_version == "local-rq-frame-v2" else
        '{{"items":[{{"p":"exact required_p value","d":"KEEP|MAYBE|REJECT","k":"LOW|BORDERLINE|HIGH","r":"short reason","c":[{{"c":"criterion id","v":"MET|NOT_MET|UNCLEAR","e":"evidence id or empty","r":"short rationale"}}]}}]}}'
    )
    return f"""You are the deep-review layer of a systematic-review screener.
Understand each title/abstract independently and apply every protocol criterion. Test the most specific criterion
first. Inclusion MET means the selected evidence unit, read with the title, semantically entails that criterion as
part of the paper's own contribution or explicit review scope. Topical association, background discussion,
co-occurring concepts, or a neighboring technology/setting/relationship is not entailment. Inclusion NOT_MET
requires affirmative contradictory evidence; silence is UNCLEAR. Exclusion
MET means affirmative disqualifying evidence is present.
The original q and explicit user criteria are authoritative. Boundaries are advisory near-miss checks and cannot
create additional inclusion or exclusion requirements.
When rq_frame is present, preserve its required group relationships. Allowed variants are advisory interpretation
aids only and never independent evidence that the paper satisfies the RQ.
For rq_core_relationship MET, the cited unit must directly identify q's central phenomenon, intervention, or
subject and support its requested scope or relationship. Do not infer identity from similar functionality, an
acronym, a named framework, or a neighboring technique. Use only ids listed in protocol.criteria for c; never use
an advisory_near_miss_checks field name as a criterion id.
Return one c item for every criterion. {evidence_rule} Resolve to KEEP or REJECT when evidence safely
supports it; otherwise MAYBE. k is LOW, BORDERLINE, or HIGH decision risk, not confidence. Keep the paper reason
to at most 18 words and each criterion rationale to at most 12 words.
The p values are opaque identifiers. Copy every value from required_p exactly once; never rename, expand, prefix,
or renumber one. Return JSON only, with no hidden reasoning.

INPUT:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

OUTPUT SHAPE:
{output_shape}
"""


def critic_batch_prompt(
    protocol: ReviewProtocol,
    papers: list[dict[str, Any]],
    candidates: dict[str, LayerResult],
    rq_frame: ScreeningRQFrame | None = None,
) -> str:
    compact_protocol = {
        "q": protocol.research_question,
        "criteria": [
            {"c": c.id, "kind": c.kind, "required": c.required, "description": c.description}
            for c in protocol.criteria
        ],
        "advisory_near_miss_checks": protocol.semantic_boundaries,
    }
    entries = []
    for paper in papers:
        paper_id = str(paper["id"])
        candidate = candidates[paper_id]
        entries.append({
            **_paper_payload(paper),
            "review_trigger": {
                "risk": candidate.result.get("decision_risk", "HIGH"),
                # Do not leak decision-specific validator wording (for example,
                # "KEEP lacks...") into the independent adjudication prompt.
                "needs_independent_validation": bool(
                    candidate.result.get("validation_errors", [])
                ),
            },
        })
    payload = {
        "protocol": compact_protocol,
        "rq_frame": rq_frame.compact_prompt_payload() if rq_frame else None,
        "required_p": [str(paper["id"]) for paper in papers],
        "papers": entries,
    }
    if rq_frame is None or rq_frame.frame_version != "local-rq-frame-v2":
        return f"""You are the final prediction-blind adjudicator. No earlier decision is supplied. Independently rebuild
the assessment from the protocol and this paper's evidence. Test the hardest semantic boundary and most specific
criterion first. A background mention, topical association, or separately mentioned concepts cannot establish the
required relationship. For REJECT, demand affirmative evidence of exclusion or contradiction; absence of detail is
not enough. For KEEP, demand criterion-specific support for every required relationship as part of the paper's own
contribution or explicit review scope. Use MAYBE when neither definitive outcome is evidence-safe.
The original q and explicit user criteria are authoritative. Boundaries may identify near misses but cannot narrow
or expand q.
When rq_frame is present, independently apply its required group relationships and forbidden-broadening warnings.
Allowed variants cannot replace evidence of the paper's actual contribution and requested relationship.
For rq_core_relationship MET, require cited evidence that directly identifies q's central phenomenon,
intervention, or subject and supports its requested scope or relationship. Shared functionality, an acronym, a
framework name, or a related technique cannot establish identity. Use only ids from protocol.criteria for c;
advisory_near_miss_checks is never a criterion id.
Return a complete replacement with every criterion, at most one exact evidence-unit ID per criterion, and every requested p
exactly once. Every MET or NOT_MET must cite evidence; only UNCLEAR may use an empty evidence ID. Keep the paper
reason to at most 18 words and each criterion rationale to at most 12 words. k is LOW,
BORDERLINE, or HIGH final decision risk. The p values are opaque identifiers. Copy every value from required_p
exactly once; never rename, expand, prefix, or renumber one. JSON only; no hidden reasoning.

INPUT:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

OUTPUT SHAPE:
{{"items":[{{"p":"exact required_p value","d":"KEEP|MAYBE|REJECT","k":"LOW|BORDERLINE|HIGH","r":"short reason","c":[{{"c":"criterion id","v":"MET|NOT_MET|UNCLEAR","e":"evidence id or empty","r":"short rationale"}}]}}]}}
"""
    evidence_rule = (
        "Use up to two exact evidence-unit IDs per criterion; every MET or NOT_MET needs at least one ID, and only UNCLEAR may use []."
        if rq_frame is not None and rq_frame.frame_version == "local-rq-frame-v2" else
        "Use at most one exact evidence-unit ID per criterion; every MET or NOT_MET needs evidence, and only UNCLEAR may use an empty ID."
    )
    output_shape = (
        '{{"items":[{{"p":"exact required_p value","d":"KEEP|MAYBE|REJECT","k":"LOW|BORDERLINE|HIGH","r":"short reason","c":[{{"c":"criterion id","v":"MET|NOT_MET|UNCLEAR","e":["evidence_id"],"r":"short rationale"}}]}}]}}'
        if rq_frame is not None and rq_frame.frame_version == "local-rq-frame-v2" else
        '{{"items":[{{"p":"exact required_p value","d":"KEEP|MAYBE|REJECT","k":"LOW|BORDERLINE|HIGH","r":"short reason","c":[{{"c":"criterion id","v":"MET|NOT_MET|UNCLEAR","e":"evidence id or empty","r":"short rationale"}}]}}]}}'
    )
    return f"""You are the final prediction-blind adjudicator. No earlier decision is supplied. Independently rebuild
the assessment from the protocol and this paper's evidence. Test the hardest semantic boundary and most specific
criterion first. A background mention, topical association, or separately mentioned concepts cannot establish the
required relationship. For REJECT, demand affirmative evidence of exclusion or contradiction; absence of detail is
not enough. For KEEP, demand criterion-specific support for every required relationship as part of the paper's own
contribution or explicit review scope. Use MAYBE when neither definitive outcome is evidence-safe.
The original q and explicit user criteria are authoritative. Boundaries may identify near misses but cannot narrow
or expand q.
When rq_frame is present, independently apply its required group relationships and forbidden-broadening warnings.
Allowed variants cannot replace evidence of the paper's actual contribution and requested relationship.
For rq_core_relationship MET, require cited evidence that directly identifies q's central phenomenon,
intervention, or subject and supports its requested scope or relationship. Shared functionality, an acronym, a
framework name, or a related technique cannot establish identity. Use only ids from protocol.criteria for c;
advisory_near_miss_checks is never a criterion id.
Return a complete replacement with every criterion and every requested p exactly once. {evidence_rule} Keep the paper
reason to at most 18 words and each criterion rationale to at most 12 words. k is LOW,
BORDERLINE, or HIGH final decision risk. The p values are opaque identifiers. Copy every value from required_p
exactly once; never rename, expand, prefix, or renumber one. JSON only; no hidden reasoning.

INPUT:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

OUTPUT SHAPE:
{output_shape}
"""


class ThreeLayerLocalOrchestrator:
    """Throughput path: batched 3B triage -> batched 4B review -> batched 4B critic."""

    def __init__(
        self,
        profile: RuntimeProfile | None = None,
        triage_engine=None,
        deep_engine=None,
        cache: JsonDiskCache | None = None,
        screening_profile: str | None = None,
    ):
        self.profile = profile or resolve_runtime_profile()
        self.screening_profile = resolve_local_screening_profile(screening_profile)
        selected = self.screening_profile
        self.triage_model = selected.triage_model
        self.protocol_model = selected.protocol_model
        self.deep_model = selected.deep_model
        self.edge_model = selected.edge_model
        self.prompt_version = selected.prompt_version
        self.triage_profile = replace(
            self.profile, fast_model=self.triage_model, strong_model=self.triage_model,
            num_ctx=int(os.getenv("LOCAL_TRIAGE_CONTEXT", "4096")),
            concurrency=1, keep_alive="30m",
        )
        self.deep_profile = replace(
            self.profile, fast_model=self.deep_model, strong_model=self.edge_model,
            num_ctx=4096, concurrency=1, keep_alive="30m",
        )
        self.triage_engine = triage_engine or OllamaStructuredEngine(self.triage_profile)
        self.deep_engine = deep_engine or OllamaStructuredEngine(self.deep_profile)
        self._deep_model_active = False
        self._active_deep_model: str | None = None
        self.cache = cache or JsonDiskCache(os.getenv("LOCAL_AI_CACHE_PATH", "outputs/cache/local_ai"))

    def require_profile_models(self) -> None:
        installed = self.profile.hardware.installed_models
        if not installed:
            raise LocalAIError("Ollama is unavailable or returned no installed local models.")
        required = {
            self.triage_model, self.protocol_model, self.deep_model, self.edge_model,
        }
        missing = sorted(model for model in required if model not in installed)
        if missing:
            raise LocalAIError(
                f"Local profile {self.screening_profile.name!r} requires missing model(s): "
                + ", ".join(missing)
                + ". No automatic downgrade was applied."
            )

    def run_protocol_id(
        self, question: str, inclusion: str = "", exclusion: str = "", research_context: str = "",
        rq_frame: ScreeningRQFrame | None = None,
    ) -> str:
        return _run_protocol_id(
            question, inclusion, exclusion, research_context,
            frame_id=rq_frame.frame_id if rq_frame and self.screening_profile.structured_rq else "",
            profile_name=self.screening_profile.name,
        )

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
        rq_frame: ScreeningRQFrame | None = None,
        on_batch: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, LayerResult], list[dict[str, Any]]]:
        active_frame = rq_frame if self.screening_profile.structured_rq else None
        run_id = self.run_protocol_id(question, inclusion, exclusion, research_context, active_frame)

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
            transition = (
                "final"
                if not self.screening_profile.require_deep_review
                and item.d in {"KEEP", "REJECT"} and item.k == "LOW"
                else "deep_review"
            )
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
                "escalated": False, "model": self.triage_model, "model_tier": "resident_three_layer_local",
                "prompt_version": self.prompt_version, "attempts": 1,
                "cache_hit": False, "processing_seconds": round(allocated, 4),
                "original_processing_seconds": round(allocated, 4), "runtime_downgrades": [],
                "layer_trace": [{"layer": 1, "name": "quick_triage", "model": self.triage_model,
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
                "escalated": False, "model": self.triage_model, "model_tier": "resident_three_layer_local",
                "prompt_version": self.prompt_version, "attempts": len(metrics),
                "cache_hit": False, "processing_seconds": round(allocated, 4),
                "original_processing_seconds": round(allocated, 4), "runtime_downgrades": [],
                "layer_trace": [{"layer": 1, "name": "quick_triage", "model": self.triage_model,
                                 "decision": "MAYBE", "risk": "HIGH",
                                 "basis": "INSUFFICIENT_OR_AMBIGUOUS",
                                 "validation_status": "unresolved"}],
                "layer_metrics": [
                    {**dict(value), "queue_transition": "deep_review"} for value in metrics
                ],
            }
            return LayerResult(result, None, ValidationReport(valid=False, errors=result["validation_errors"]), allocated, allocated)

        return self._execute_batches(
            papers=papers, batch_size=TRIAGE_BATCH_SIZE, layer="quick_triage", model=self.triage_model,
            engine=self.triage_engine, schema=TriageBatch,
            prompt_factory=lambda group: triage_batch_prompt(
                question, inclusion, exclusion, group, research_context, protocol, active_frame
            ),
            normalize=normalize, safe=safe,
            on_batch=on_batch,
        )

    def unload_triage(self) -> None:
        if hasattr(self.triage_engine, "unload"):
            self.triage_engine.unload(self.triage_model)

    def unload_deep(self) -> None:
        if self._deep_model_active and hasattr(self.deep_engine, "unload"):
            self.deep_engine.unload(self._active_deep_model or self.deep_model)
        self._deep_model_active = False
        self._active_deep_model = None

    def prepare_edge_critic(self) -> None:
        if self.edge_model != self.deep_model:
            self.unload_deep()

    def compile_protocol(
        self, question: str, inclusion: str = "", exclusion: str = "", research_context: str = "",
        rq_frame: ScreeningRQFrame | None = None,
    ) -> ReviewProtocol:
        active_frame = rq_frame if self.screening_profile.structured_rq else None
        key_parts = [
            question, research_context, inclusion, exclusion,
            self.protocol_model, self.prompt_version,
        ]
        if active_frame:
            key_parts.append(active_frame.frame_id)
        key = cache_key(*key_parts, "protocol")
        cached = self.cache.get("three_layer", key)
        if cached:
            protocol = ReviewProtocol.model_validate(cached)
            if protocol.prompt_version == self.prompt_version:
                return protocol
        initial = self.deep_engine.generate(
            self.protocol_model,
            protocol_prompt(question, inclusion, exclusion, research_context, active_frame),
            ReviewProtocol,
        )
        self._deep_model_active = True
        self._active_deep_model = self.protocol_model
        protocol = ReviewProtocol.model_validate(
            self._anchor_rq_contract(self._normalize_protocol_provenance(
                initial.value, allow_user=bool(inclusion.strip() or exclusion.strip())
            ), question, self.screening_profile.structured_rq, active_frame,
                self.screening_profile.evidence_grounded, inclusion, exclusion)
        ).model_copy(update={
            "research_question": question, "research_context": research_context,
            "model": self.protocol_model, "prompt_version": self.prompt_version,
        })
        criticised = self.deep_engine.generate(
            self.protocol_model,
            protocol_critic_prompt(protocol, inclusion, exclusion, research_context, active_frame),
            ReviewProtocol,
        )
        protocol = ReviewProtocol.model_validate(
            self._anchor_rq_contract(self._normalize_protocol_provenance(
                criticised.value, allow_user=bool(inclusion.strip() or exclusion.strip())
            ), question, self.screening_profile.structured_rq, active_frame,
                self.screening_profile.evidence_grounded, inclusion, exclusion)
        ).model_copy(update={
            "research_question": question, "research_context": research_context,
            "model": self.protocol_model, "prompt_version": self.prompt_version,
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
    def _anchor_rq_contract(
        value: dict[str, Any], question: str = "", preserve_semantics: bool = False,
        rq_frame: ScreeningRQFrame | None = None, evidence_grounded: bool = False,
        inclusion: str = "", exclusion: str = "",
    ) -> dict[str, Any]:
        """Normalize the single RQ gate without discarding its semantic description."""
        normalized = dict(value)
        user_criteria = [
            dict(item) for item in value.get("criteria") or []
            if item.get("source") == "user"
        ]
        if evidence_grounded:
            entries = lambda text: [
                item.strip(" -*\t") for line in text.splitlines() for item in line.split(";")
                if item.strip(" -*\t")
            ]
            user_criteria = [
                {
                    "id": f"user_inclusion_{index}", "kind": "inclusion",
                    "description": description, "required": True,
                    "expected_evidence": "Direct title or abstract evidence satisfying this explicit user criterion.",
                    "source": "user",
                }
                for index, description in enumerate(entries(inclusion), 1)
            ] + [
                {
                    "id": f"user_exclusion_{index}", "kind": "exclusion",
                    "description": description, "required": True,
                    "expected_evidence": "Direct title or abstract evidence establishing this explicit user exclusion.",
                    "source": "user",
                }
                for index, description in enumerate(entries(exclusion), 1)
            ]
        candidate = next((
            dict(item) for item in value.get("criteria") or []
            if item.get("source") == "research_question"
        ), {})
        if evidence_grounded and rq_frame is not None:
            clauses = []
            for group in rq_frame.groups:
                if not group.required or group.group_relationship == "ADVISORY":
                    continue
                alternatives = " OR ".join(f'"{span}"' for span in group.source_spans)
                clauses.append(f"{group.role}: ({alternatives})")
            relationship = " AND ".join(f"({clause})" for clause in clauses)
            description = (
                f'Original research question: "{rq_frame.question}" Required source-linked groups: '
                f"{relationship}. The paper must address the relationship asked by the original question."
            )
            expected_evidence = (
                "Direct title or abstract evidence jointly covering every required group and the relationship "
                "expressed by the original research question."
            )
        else:
            description = (
            str(candidate.get("description") or question).strip()
            if preserve_semantics else
            "The paper's own contribution or explicit review scope directly addresses "
            "the complete entities, scope, and relationships stated in the research question."
            )
            expected_evidence = (
                str(candidate.get("expected_evidence") or "").strip()
                if preserve_semantics else ""
            ) or "Direct title or abstract evidence responsive to the original research question."
        normalized["criteria"] = [{
            "id": "rq_core_relationship",
            "kind": "inclusion",
            "required": True,
            "description": description,
            "expected_evidence": expected_evidence,
            "source": "research_question",
        }, *user_criteria]
        if not preserve_semantics or not normalized.get("expected_relationships"):
            normalized["expected_relationships"] = [
                "Use the complete relationship in the original research question without inferred requirements."
            ]
        if evidence_grounded and rq_frame is not None:
            normalized["expected_relationships"] = [
                "Preserve the complete relationship in the verbatim original research question."
            ]
            normalized["semantic_boundaries"] = list(rq_frame.forbidden_broadening_warnings[:6])
        return normalized

    @staticmethod
    def _validate_protocol(protocol: ReviewProtocol, inclusion: str, exclusion: str) -> None:
        if any(c.source == "research_question" and c.kind == "exclusion" for c in protocol.criteria):
            raise ValueError("RQ scope must use positive inclusion criteria")
        rq_criteria = [c for c in protocol.criteria if c.source == "research_question"]
        if len(rq_criteria) != 1 or rq_criteria[0].id != "rq_core_relationship" or not rq_criteria[0].required:
            raise ValueError("protocol must contain one required composite RQ relationship")
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
            evidence_ids = compact.e if isinstance(compact.e, list) else ([compact.e] if compact.e else [])
            for evidence_id in evidence_ids:
                source = "title" if evidence_id.startswith("title_") else "abstract"
                evidence.append(EvidenceSpan(source=source, evidence_id=evidence_id))
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
        rq_frame: ScreeningRQFrame | None = None,
        on_batch: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, LayerResult], list[dict[str, Any]]]:
        if self.screening_profile.evidence_grounded:
            schema: type[BaseModel] = (
                GroundedAssessmentBatch if layer == "deep_review" else GroundedCriticBatch
            )
        else:
            schema = AssessmentBatch if layer == "deep_review" else CriticBatch
        batch_size = DEEP_BATCH_SIZE if layer == "deep_review" else EDGE_BATCH_SIZE
        prior = candidates or {}

        def normalize(paper, item: AssessmentItem, metric, paper_metrics):
            assessment = self._assessment_from_compact(item)
            validation = validate_assessment(
                assessment, protocol, str(paper.get("title", "")), str(paper.get("abstract", "")),
                rq_frame if self.screening_profile.evidence_grounded else None,
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
                self.edge_model if layer == "edge_critic" else self.deep_model, item.k, allocated,
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
            previous = prior.get(str(paper["id"]))
            allocated = sum(
                float(m.get("wall_seconds") or 0.0) /
                max(1, int(m.get("batch_size") or 1))
                for m in metrics
            )
            if (
                layer == "edge_critic" and previous is not None
                and previous.result.get("validation_status") == "validated"
            ):
                # Validation, not subject-matter code, decides this fallback:
                # an invalid critic cannot replace a valid deep assessment.
                result = dict(previous.result)
                result["layer_trace"] = list(previous.result.get("layer_trace", [])) + [{
                    "layer": 3,
                    "name": "edge_critic",
                    "model": self.edge_model,
                    "decision": previous.result.get("decision", "MAYBE"),
                    "risk": previous.result.get("decision_risk", "HIGH"),
                    "validation_status": "invalid_fallback_to_deep",
                }]
                result["layer_metrics"] = list(previous.result.get("layer_metrics", [])) + [
                    {**dict(value), "queue_transition": "final_valid_deep_fallback"}
                    for value in metrics
                ]
                result["processing_seconds"] = round(
                    float(previous.result.get("processing_seconds") or 0.0) + allocated, 4
                )
                result["original_processing_seconds"] = round(
                    float(previous.result.get("original_processing_seconds") or 0.0) + allocated, 4
                )
                result["attempts"] = len(result["layer_trace"])
                warnings = list(previous.result.get("validation_warnings", []))
                warnings.append(f"Invalid edge critic ignored; retained validated deep assessment: {error}")
                result["validation_warnings"] = warnings
                return LayerResult(
                    result, previous.assessment, previous.validation,
                    previous.elapsed_seconds + allocated,
                    previous.original_elapsed_seconds + allocated,
                )
            assessment = safe_maybe(f"{layer} could not produce an evidence-safe result: {error}")
            validation = ValidationReport(valid=False, errors=[error or "Malformed batch output."])
            result = self._assessment_public(
                assessment, validation, protocol, run_id, paper, layer,
                self.edge_model if layer == "edge_critic" else self.deep_model, "HIGH", allocated,
                previous,
            )
            result.update(decision="MAYBE", validation_status="unresolved", validation_errors=validation.errors)
            transition = "final" if layer == "edge_critic" else "edge_critic"
            result["layer_metrics"] = (
                list(previous.result.get("layer_metrics", [])) if previous else []
            ) + [{**dict(value), "queue_transition": transition} for value in metrics]
            return LayerResult(result, assessment, validation, allocated, allocated)

        prompt_factory = (
            (lambda group: assessment_batch_prompt(protocol, group, rq_frame))
            if layer == "deep_review"
            else (lambda group: critic_batch_prompt(protocol, group, prior, rq_frame))
        )
        return self._execute_batches(
            papers=papers, batch_size=batch_size, layer=layer,
            model=self.deep_model if layer == "deep_review" else self.edge_model,
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
            "rq_group_coverage": validation.rq_group_coverage,
            "escalated": True, "model": model, "model_tier": "resident_three_layer_local",
            "prompt_version": self.prompt_version, "attempts": len(trace), "cache_hit": False,
            "processing_seconds": round(prior_seconds + elapsed, 4),
            "original_processing_seconds": round(prior_seconds + elapsed, 4),
            "runtime_downgrades": [], "layer_trace": trace, "internal_protocol_id": protocol.protocol_id,
        }

    def deep_review_batch(self, protocol, run_id, papers, triage_results, on_batch=None, rq_frame=None):
        self._deep_model_active = True
        self._active_deep_model = self.deep_model
        return self._assessment_batches(
            protocol=protocol, run_id=run_id, papers=papers,
            layer="deep_review", candidates=triage_results, on_batch=on_batch, rq_frame=rq_frame,
        )

    def edge_critic_batch(self, protocol, run_id, papers, deep_results, on_batch=None, rq_frame=None):
        self._deep_model_active = True
        self._active_deep_model = self.edge_model
        return self._assessment_batches(
            protocol=protocol, run_id=run_id, papers=papers,
            layer="edge_critic", candidates=deep_results, on_batch=on_batch, rq_frame=rq_frame,
        )

    @staticmethod
    def needs_deep_review(layer: LayerResult) -> bool:
        return (
            layer.result.get("decision") in {"MAYBE", "REJECT"}
            or layer.result.get("decision_risk") != "LOW"
            or layer.result.get("validation_status") != "validated"
        )

    def requires_deep_review(self, layer: LayerResult) -> bool:
        return self.screening_profile.require_deep_review or self.needs_deep_review(layer)

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
        research_context="", rq_frame: ScreeningRQFrame | None = None,
    ):
        active_frame = rq_frame if self.screening_profile.structured_rq else None
        run_id = self.run_protocol_id(
            research_question, inclusion_criteria, exclusion_criteria, research_context, active_frame
        )
        try:
            protocol = self.compile_protocol(
                research_question, inclusion_criteria, exclusion_criteria, research_context, active_frame
            )
        except (LocalAIError, ValidationError, ValueError) as exc:
            return {
                "schema_version": SCHEMA_VERSION, "decision": "MAYBE",
                "reason": f"Research protocol setup failed: {str(exc)[:300]}",
                "confidence": 0.0, "decision_risk": "HIGH", "protocol_id": run_id,
                "criteria": [], "evidence": [], "uncertainty": [str(exc)],
                "validation_status": "unresolved", "validation_errors": [str(exc)],
                "model_tier": "resident_three_layer_local", "model": self.protocol_model,
                "prompt_version": self.prompt_version, "processing_seconds": 0.0,
                "original_processing_seconds": 0.0, "cache_hit": False,
                "runtime_downgrades": [], "layer_trace": [], "layer_metrics": [],
                "escalated": False,
            }
        self.unload_deep()
        paper = {"id": "paper_1", "title": title, "abstract": abstract}
        triage, _ = self.triage_batch(
            research_question, [paper], inclusion_criteria, exclusion_criteria,
            research_context, protocol, active_frame,
        )
        first = triage["paper_1"]
        if not self.requires_deep_review(first):
            return first.result
        self.unload_triage()
        try:
            deep, _ = self.deep_review_batch(protocol, run_id, [paper], triage, rq_frame=active_frame)
            second = deep["paper_1"]
            if not self.needs_edge_critic(second):
                return second.result
            self.prepare_edge_critic()
            edge, _ = self.edge_critic_batch(protocol, run_id, [paper], deep, rq_frame=active_frame)
            return edge["paper_1"].result
        except (LocalAIError, ValidationError, ValueError) as exc:
            result = dict(first.result)
            result.update(
                decision="MAYBE", reason=f"4B review unavailable: {str(exc)[:300]}",
                validation_status="unresolved", validation_errors=[str(exc)],
            )
            return result
