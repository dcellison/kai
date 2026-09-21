"""Private, policy-bounded model comparison for semantic-memory roles.

The evaluator replays a sealed production corpus without writing memories.
Model output is captured in an immutable private artifact, then exposed through
a blind review template whose arm identifiers do not reveal model names.  A
separate sealed human review drives the final metrics and promotion decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from kai.config import Config, ModelRole, canonicalize_model_for_backend, validate_model_for_backend_policy
from kai.memory import MemoryResult
from kai.memory_quality_corpus import (
    QUALITY_LABELS,
    MemoryQualityCorpusError,
    _assert_outside_source_tree,
    _document_digest,
    _load_json,
    _now,
    _write_private_json,
    score_review,
    validate_sealed_review,
    validate_snapshot,
)
from kai.workshop.runtime_profiles import ProtectedRuntimeProfile, WorkshopRuntimeProfileRegistry

PLAN_VERSION = 1
EVALUATION_VERSION = 1
EVALUATION_REVIEW_VERSION = 1
EVALUATION_SCORE_VERSION = 1

FACT_ROLE = "fact_extraction"
EPISODE_ROLE = "episode_generation"
ROLES = frozenset({FACT_ROLE, EPISODE_ROLE})

_ROLE_MODEL_KEYS = {
    FACT_ROLE: ModelRole.MEMORY_EXTRACTION.value,
    EPISODE_ROLE: ModelRole.MEMORY_EPISODE.value,
}
_BAD_SUPPORT_LABELS = frozenset({"incorrect", "unsupported", "stale_on_arrival", "wrong_speaker"})
_OUTPUT_VERDICTS = frozenset({"useful", "not_useful", "not_applicable"})
_SUCCESS_OUTCOMES = frozenset({"succeeded", "validation_rejection"})
_PROVIDER_FAILURE_OUTCOMES = frozenset(
    {"authentication_failure", "provider_failure", "timeout", "policy_rejection", "parsing_failure"}
)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "minimum_precision": 0.95,
    "minimum_useful_recall": 0.80,
    "maximum_over_extraction_rate": 0.10,
    "maximum_fragmentation_rate": 0.10,
    "minimum_scope_accuracy": 0.90,
    "minimum_consolidation_accuracy": 0.80,
    "minimum_episode_classifier_accuracy": 0.90,
    "minimum_episode_useful_rate": 0.80,
    "minimum_structured_output_reliability": 0.98,
    "maximum_provider_failure_rate": 0.02,
    "maximum_latency_multiplier": 3.0,
    "maximum_precision_regression": 0.0,
    "maximum_useful_recall_regression": 0.0,
}

FactRunner = Callable[[dict[str, object], dict[str, object], str, str | None, Config], Awaitable[dict[str, object]]]
EpisodeRunner = Callable[[dict[str, object], dict[str, object], str, str | None, Config], Awaitable[dict[str, object]]]


def build_plan_template(
    snapshot: dict[str, object],
    review: dict[str, object],
    *,
    runtime_profile_id: str,
) -> dict[str, object]:
    """Create an editable plan seeded from models observed in the corpus."""
    validate_snapshot(snapshot)
    validate_sealed_review(review, snapshot)
    if snapshot.get("version") != 2:
        raise MemoryQualityCorpusError("Model evaluation requires a version-2 snapshot with replay context")
    candidates: list[dict[str, object]] = []
    for role in sorted(ROLES):
        matching: list[tuple[str, str, str, str]] = []
        for case in _cases(snapshot):
            provenance = _mapping(case.get("provenance"), "case provenance")
            if provenance.get("runtime_profile_id") != runtime_profile_id or case.get("extraction_role") != role:
                continue
            pipeline = _mapping(case.get("pipeline"), "case pipeline")
            matching.append(
                (
                    str(provenance.get("created_at") or ""),
                    _required_string(pipeline, "backend"),
                    _required_string(pipeline, "provider"),
                    _required_string(pipeline, "model"),
                )
            )
        if not matching:
            raise MemoryQualityCorpusError(f"Snapshot has no {role} cases for runtime profile {runtime_profile_id}")
        _, backend, provider, model = max(matching)
        candidates.append(
            {
                "role": role,
                "backend": backend,
                "provider": provider,
                "model": model,
                "baseline": True,
            }
        )
    return {
        "artifact": "kai_memory_model_evaluation_plan",
        "version": PLAN_VERSION,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "review_sha256": review["review_sha256"],
        "runtime_profile_id": runtime_profile_id,
        "created_at": _now(),
        "instructions": {
            "candidate": "Add at least one stronger candidate for each role; keep exactly one baseline per role.",
            "roles": sorted(ROLES),
            "authority": "Every candidate must use the baseline backend/provider and an allowed runtime-profile model.",
        },
        "candidates": candidates,
        "thresholds": dict(DEFAULT_THRESHOLDS),
    }


def seal_plan(
    plan: dict[str, object],
    snapshot: dict[str, object],
    review: dict[str, object],
    registry: WorkshopRuntimeProfileRegistry,
) -> tuple[dict[str, object], ProtectedRuntimeProfile]:
    """Validate candidate policy and return an immutable execution plan."""
    validate_snapshot(snapshot)
    validate_sealed_review(review, snapshot)
    if snapshot.get("version") != 2:
        raise MemoryQualityCorpusError("Model evaluation requires a version-2 snapshot with replay context")
    if plan.get("artifact") != "kai_memory_model_evaluation_plan" or plan.get("version") != PLAN_VERSION:
        raise MemoryQualityCorpusError("Unsupported memory-model evaluation plan")
    if plan.get("snapshot_sha256") != snapshot.get("snapshot_sha256"):
        raise MemoryQualityCorpusError("Evaluation plan belongs to a different snapshot")
    if plan.get("review_sha256") != review.get("review_sha256"):
        raise MemoryQualityCorpusError("Evaluation plan belongs to a different sealed review")
    runtime_profile_id = _required_string(plan, "runtime_profile_id")
    try:
        profile = registry.resolve(runtime_profile_id)
    except Exception as exc:
        raise MemoryQualityCorpusError("Evaluation runtime profile is unavailable") from exc

    raw_candidates = plan.get("candidates")
    if not isinstance(raw_candidates, list):
        raise MemoryQualityCorpusError("Evaluation candidates must be a list")
    sealed_candidates: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    baselines: dict[str, tuple[str, str]] = {}
    counts = {role: 0 for role in ROLES}
    for raw in raw_candidates:
        candidate = _mapping(raw, "evaluation candidate")
        role = _required_string(candidate, "role")
        backend = _required_string(candidate, "backend")
        provider = _required_string(candidate, "provider")
        model = canonicalize_model_for_backend(_required_string(candidate, "model"), backend)
        baseline = candidate.get("baseline")
        if role not in ROLES or not isinstance(baseline, bool):
            raise MemoryQualityCorpusError("Evaluation candidate role or baseline flag is invalid")
        key = (role, backend, provider, model)
        if key in seen:
            raise MemoryQualityCorpusError("Evaluation candidates must be unique")
        seen.add(key)
        counts[role] += 1
        try:
            option = profile.backend_option(f"{backend}:{provider}")
        except Exception as exc:
            raise MemoryQualityCorpusError(
                f"Candidate route {backend}:{provider} is not authorized for {runtime_profile_id}"
            ) from exc
        if not validate_model_for_backend_policy(
            model,
            backend,
            provider,
            allowed_models=option.allowed_models,
        ):
            raise MemoryQualityCorpusError(
                f"Candidate model {model!r} is outside runtime authority for {backend}:{provider}"
            )
        if baseline:
            if role in baselines:
                raise MemoryQualityCorpusError(f"Role {role} must have exactly one baseline")
            baselines[role] = (backend, provider)
        candidate_id = _opaque_id("candidate", "\0".join(key))
        sealed_candidates.append(
            {
                "candidate_id": candidate_id,
                "role": role,
                "backend": backend,
                "provider": provider,
                "model": model,
                "baseline": baseline,
            }
        )
    if set(baselines) != ROLES or any(counts[role] < 2 for role in ROLES):
        raise MemoryQualityCorpusError("Each evaluation role needs exactly one baseline and at least one candidate")
    for candidate in sealed_candidates:
        role = str(candidate["role"])
        if (candidate["backend"], candidate["provider"]) != baselines[role]:
            raise MemoryQualityCorpusError(
                f"Role {role} candidates must use the baseline backend/provider so promotion cannot change conversations"
            )

    thresholds = _validate_thresholds(plan.get("thresholds"))
    baseline_score = score_review(snapshot, review)
    if baseline_score.get("qualification_ready") is not True:
        raise MemoryQualityCorpusError("The sealed corpus review is not qualification-ready")
    sealed: dict[str, object] = {
        "artifact": "kai_memory_model_evaluation_sealed_plan",
        "version": PLAN_VERSION,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "review_sha256": review["review_sha256"],
        "runtime_profile_id": runtime_profile_id,
        "sealed_at": _now(),
        "candidates": sealed_candidates,
        "thresholds": thresholds,
    }
    sealed["plan_sha256"] = _document_digest(sealed, "plan_sha256")
    return sealed, profile


async def run_evaluation(
    snapshot: dict[str, object],
    review: dict[str, object],
    sealed_plan: dict[str, object],
    profile: ProtectedRuntimeProfile,
    config: Config,
    *,
    fact_runner: FactRunner | None = None,
    episode_runner: EpisodeRunner | None = None,
) -> dict[str, object]:
    """Replay every eligible case sequentially without writing memory."""
    _validate_sealed_plan(sealed_plan, snapshot, review)
    runtime_profile_id = str(sealed_plan["runtime_profile_id"])
    if str(profile.profile_id) != runtime_profile_id:
        raise MemoryQualityCorpusError("Execution profile does not match the sealed evaluation plan")
    principal_id = _required_string(snapshot, "principal_id")
    run_fact = fact_runner or _run_fact_candidate
    run_episode = episode_runner or _run_episode_candidate
    arms: list[dict[str, object]] = []
    raw_candidates = sealed_plan["candidates"]
    assert isinstance(raw_candidates, list)
    eligible = [
        case
        for case in _cases(snapshot)
        if _mapping(case.get("provenance"), "case provenance").get("runtime_profile_id") == runtime_profile_id
    ]
    for raw_candidate in raw_candidates:
        candidate = _mapping(raw_candidate, "sealed candidate")
        role = str(candidate["role"])
        arm_id = _opaque_id("arm", f"{sealed_plan['plan_sha256']}\0{candidate['candidate_id']}")
        results: list[dict[str, object]] = []
        for case in sorted(eligible, key=lambda value: str(value.get("case_id"))):
            if case.get("extraction_role") != role:
                continue
            if role == FACT_ROLE:
                output = await run_fact(case, candidate, principal_id, profile.os_user, config)
            else:
                output = await run_episode(case, candidate, principal_id, profile.os_user, config)
            output = dict(output)
            output["case_id"] = _required_string(case, "case_id")
            results.append(output)
        arms.append(
            {
                "arm_id": arm_id,
                "candidate_id": candidate["candidate_id"],
                "role": role,
                "backend": candidate["backend"],
                "provider": candidate["provider"],
                "model": candidate["model"],
                "baseline": candidate["baseline"],
                "results": results,
            }
        )
    evaluation: dict[str, object] = {
        "artifact": "kai_memory_model_evaluation",
        "version": EVALUATION_VERSION,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "review_sha256": review["review_sha256"],
        "plan_sha256": sealed_plan["plan_sha256"],
        "runtime_profile_id": runtime_profile_id,
        "created_at": _now(),
        "arms": arms,
    }
    evaluation["evaluation_sha256"] = _document_digest(evaluation, "evaluation_sha256")
    return evaluation


async def _run_fact_candidate(
    case: dict[str, object],
    candidate: dict[str, object],
    principal_id: str,
    os_user: str | None,
    config: Config,
) -> dict[str, object]:
    from kai import memory_extraction

    conversation = _mapping(case.get("conversation"), "case conversation")
    pipeline = _mapping(case.get("pipeline"), "case pipeline")
    user_text = _required_string(conversation, "user")
    assistant_text = _required_string(conversation, "assistant")
    prior_pairs = _prior_pairs(conversation)
    memories = _candidate_memories(pipeline)
    candidate_ids = {memory.id for memory in memories}
    candidate_metadata = {memory.id: memory.metadata for memory in memories}
    payload = memory_extraction._build_extraction_payload(
        user_text,
        assistant_text,
        candidates=memories,
        prior_pairs=prior_pairs,
    )
    started = time.monotonic()
    result = await memory_extraction._run_extractor(
        payload,
        config,
        candidate_ids=candidate_ids,
        candidate_metadata=candidate_metadata,
        user_id=principal_id,
        effective_backend=str(candidate["backend"]),
        effective_provider=str(candidate["provider"]),
        os_user=os_user,
        user_window_text=" ".join([pair[0] for pair in prior_pairs] + [user_text]),
        assistant_window_text=" ".join([pair[1] for pair in prior_pairs] + [assistant_text]),
        resolved_model=str(candidate["model"]),
    )
    return {
        "outcome": result.outcome,
        "duration_ms": round((time.monotonic() - started) * 1000),
        "raw_fact_count": result.raw_fact_count,
        "has_episode": result.has_episode,
        "facts": result.facts,
    }


async def _run_episode_candidate(
    case: dict[str, object],
    candidate: dict[str, object],
    principal_id: str,
    os_user: str | None,
    config: Config,
) -> dict[str, object]:
    from kai import memory_extraction

    conversation = _mapping(case.get("conversation"), "case conversation")
    payload = memory_extraction._build_episode_payload(
        _required_string(conversation, "user"),
        _required_string(conversation, "assistant"),
    )
    failures: list[str] = []
    started = time.monotonic()
    episode, reason = await memory_extraction._run_episode_extractor(
        payload,
        config,
        user_id=principal_id,
        effective_backend=str(candidate["backend"]),
        effective_provider=str(candidate["provider"]),
        os_user=os_user,
        resolved_model=str(candidate["model"]),
        failure_category_out=failures,
    )
    outcome = "succeeded" if episode is not None else (failures[-1] if failures else reason or "provider_failure")
    return {
        "outcome": outcome,
        "duration_ms": round((time.monotonic() - started) * 1000),
        "episode": episode,
    }


def build_blind_review_template(
    snapshot: dict[str, object],
    review: dict[str, object],
    evaluation: dict[str, object],
) -> dict[str, object]:
    """Create a model-blind review surface ordered independently of models."""
    _validate_evaluation(evaluation, snapshot, review)
    cases_by_id = {str(case["case_id"]): case for case in _cases(snapshot)}
    blind_cases: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    arms = evaluation["arms"]
    assert isinstance(arms, list)
    ordered_arms = sorted(
        (_mapping(arm, "evaluation arm") for arm in arms),
        key=lambda arm: hashlib.sha256(f"{evaluation['evaluation_sha256']}\0{arm['arm_id']}".encode()).hexdigest(),
    )
    for arm in ordered_arms:
        role = str(arm["role"])
        results = arm.get("results")
        assert isinstance(results, list)
        for raw_result in results:
            result = _mapping(raw_result, "evaluation result")
            case_id = _required_string(result, "case_id")
            case = cases_by_id[case_id]
            conversation = _mapping(case.get("conversation"), "case conversation")
            output_reviews: list[dict[str, object]] = []
            facts = result.get("facts", [])
            if isinstance(facts, list):
                for index, fact in enumerate(facts):
                    if not isinstance(fact, dict):
                        continue
                    output_reviews.append(
                        {
                            "output_index": index,
                            "verdict": "pending",
                            "labels": [],
                            "scope_correct": None,
                            "consolidation_correct": None,
                        }
                    )
            episode_review: dict[str, object] | None = None
            if isinstance(result.get("episode"), dict):
                episode_review = {"verdict": "pending", "labels": []}
            blind_cases.append(
                {
                    "case_id": case_id,
                    "arm_id": arm["arm_id"],
                    "role": role,
                    "conversation": {
                        "prior_pairs": conversation.get("prior_pairs", []),
                        "user": conversation.get("user"),
                        "assistant": conversation.get("assistant"),
                    },
                    "outcome": result.get("outcome"),
                    "has_episode": result.get("has_episode"),
                    "facts": facts,
                    "episode": result.get("episode"),
                }
            )
            decisions.append(
                {
                    "case_id": case_id,
                    "arm_id": arm["arm_id"],
                    "review_status": "pending",
                    "outputs": output_reviews,
                    "episode": episode_review,
                    "note": "",
                }
            )
    return {
        "artifact": "kai_memory_model_evaluation_review",
        "version": EVALUATION_REVIEW_VERSION,
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "created_at": _now(),
        "instructions": {
            "verdicts": sorted(_OUTPUT_VERDICTS),
            "quality_labels": [
                "useful",
                "incorrect",
                "unsupported",
                "transient",
                "stale_on_arrival",
                "redundant",
                "fragmented",
                "wrongly_scoped",
                "wrong_speaker",
            ],
            "blindness": "Arm identifiers are opaque; model identities exist only in the immutable evaluation artifact.",
        },
        "cases": blind_cases,
        "decisions": decisions,
    }


def seal_evaluation_review(
    template: dict[str, object],
    evaluation: dict[str, object],
    *,
    reviewer: str,
) -> dict[str, object]:
    if template.get("artifact") != "kai_memory_model_evaluation_review":
        raise MemoryQualityCorpusError("Unsupported model-evaluation review")
    if template.get("version") != EVALUATION_REVIEW_VERSION:
        raise MemoryQualityCorpusError("Unsupported model-evaluation review version")
    if template.get("evaluation_sha256") != evaluation.get("evaluation_sha256"):
        raise MemoryQualityCorpusError("Evaluation review belongs to a different run")
    if not reviewer.strip():
        raise MemoryQualityCorpusError("Reviewer must be non-empty")
    expected = _evaluation_output_bindings(evaluation)
    decisions = template.get("decisions")
    if not isinstance(decisions, list):
        raise MemoryQualityCorpusError("Evaluation review decisions must be a list")
    actual: set[tuple[str, str]] = set()
    sealed_decisions: list[dict[str, object]] = []
    for raw in decisions:
        decision = _mapping(raw, "evaluation review decision")
        binding = (_required_string(decision, "arm_id"), _required_string(decision, "case_id"))
        if binding in actual or binding not in expected:
            raise MemoryQualityCorpusError("Evaluation review has duplicate or unknown output bindings")
        if decision.get("review_status") != "complete":
            raise MemoryQualityCorpusError("Every evaluation decision must be marked complete")
        _validate_review_outputs(decision, expected[binding])
        actual.add(binding)
        sealed_decisions.append(dict(decision))
    if actual != set(expected):
        raise MemoryQualityCorpusError("Evaluation review must cover every arm and case exactly once")
    sealed: dict[str, object] = {
        "artifact": "kai_memory_model_evaluation_sealed_review",
        "version": EVALUATION_REVIEW_VERSION,
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "reviewer": reviewer.strip(),
        "sealed_at": _now(),
        "decisions": sealed_decisions,
    }
    sealed["evaluation_review_sha256"] = _document_digest(sealed, "evaluation_review_sha256")
    return sealed


def score_evaluation(
    snapshot: dict[str, object],
    review: dict[str, object],
    sealed_plan: dict[str, object],
    evaluation: dict[str, object],
    evaluation_review: dict[str, object],
) -> dict[str, object]:
    _validate_sealed_plan(sealed_plan, snapshot, review)
    _validate_evaluation(evaluation, snapshot, review)
    if evaluation.get("plan_sha256") != sealed_plan.get("plan_sha256"):
        raise MemoryQualityCorpusError("Evaluation belongs to a different sealed plan")
    _validate_evaluation_review(evaluation_review, evaluation)
    source_decisions = {
        str(item["case_id"]): item
        for item in review["decisions"]  # type: ignore[index]
        if isinstance(item, dict)
    }
    judged = {
        (str(item["arm_id"]), str(item["case_id"])): item
        for item in evaluation_review["decisions"]  # type: ignore[index]
        if isinstance(item, dict)
    }
    arm_scores: list[dict[str, object]] = []
    arms = evaluation["arms"]
    assert isinstance(arms, list)
    for raw_arm in arms:
        arm = _mapping(raw_arm, "evaluation arm")
        metrics = _score_arm(arm, source_decisions, judged)
        arm_scores.append(
            {
                "arm_id": arm["arm_id"],
                "candidate_id": arm["candidate_id"],
                "role": arm["role"],
                "backend": arm["backend"],
                "provider": arm["provider"],
                "model": arm["model"],
                "baseline": arm["baseline"],
                "metrics": metrics,
            }
        )
    thresholds = _mapping(sealed_plan.get("thresholds"), "evaluation thresholds")
    baselines = {str(score["role"]): score for score in arm_scores if score["baseline"] is True}
    recommendations: list[dict[str, object]] = []
    for score in arm_scores:
        if score["baseline"] is True:
            continue
        role = str(score["role"])
        gates = _promotion_gates(
            role,
            _mapping(score["metrics"], "candidate metrics"),
            _mapping(baselines[role]["metrics"], "baseline metrics"),
            thresholds,
        )
        recommendations.append(
            {
                "candidate_id": score["candidate_id"],
                "role": role,
                "promote": all(gates.values()),
                "gates": gates,
                "role_model_key": _ROLE_MODEL_KEYS[role],
                "promoted_model": score["model"],
                "rollback_model": baselines[role]["model"],
            }
        )
    report: dict[str, object] = {
        "artifact": "kai_memory_model_evaluation_score",
        "version": EVALUATION_SCORE_VERSION,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "review_sha256": review["review_sha256"],
        "plan_sha256": sealed_plan["plan_sha256"],
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "evaluation_review_sha256": evaluation_review["evaluation_review_sha256"],
        "generated_at": _now(),
        "thresholds": thresholds,
        "arms": arm_scores,
        "recommendations": recommendations,
        "automatic_mutation": False,
        "historical_provenance_preserved": True,
    }
    report["score_sha256"] = _document_digest(report, "score_sha256")
    return report


def _score_arm(
    arm: dict[str, object],
    source_decisions: Mapping[str, dict[str, object]],
    judged: Mapping[tuple[str, str], dict[str, object]],
) -> dict[str, object]:
    results = arm.get("results")
    assert isinstance(results, list)
    reviewed = useful = supported = fragmented = 0
    expected_facts = useful_for_recall = fact_outputs = excess_facts = 0
    invalid_fact_outputs = 0
    scope_total = scope_correct = consolidation_total = consolidation_correct = 0
    classifier_total = classifier_correct = 0
    episode_outputs = useful_episodes = 0
    successful = provider_failures = 0
    latencies: list[int] = []
    for raw_result in results:
        result = _mapping(raw_result, "evaluation result")
        case_id = _required_string(result, "case_id")
        decision = judged[(str(arm["arm_id"]), case_id)]
        source = source_decisions[case_id]
        outcome = str(result.get("outcome") or "provider_failure")
        successful += int(outcome in _SUCCESS_OUTCOMES)
        provider_failures += int(outcome in _PROVIDER_FAILURE_OUTCOMES)
        duration = result.get("duration_ms")
        if isinstance(duration, int) and not isinstance(duration, bool) and duration >= 0:
            latencies.append(duration)
        outputs = decision.get("outputs")
        assert isinstance(outputs, list)
        expected = source.get("expected_fact_count")
        expected_count = expected if isinstance(expected, int) and not isinstance(expected, bool) else 0
        expected_facts += expected_count
        raw_fact_count = result.get("raw_fact_count")
        produced_count = (
            raw_fact_count
            if isinstance(raw_fact_count, int)
            and not isinstance(raw_fact_count, bool)
            and raw_fact_count >= len(outputs)
            else len(outputs)
        )
        rejected_count = produced_count - len(outputs)
        invalid_fact_outputs += rejected_count
        reviewed += rejected_count
        fact_outputs += produced_count
        excess_facts += max(0, produced_count - expected_count)
        useful_in_case = 0
        for raw_output in outputs:
            output = _mapping(raw_output, "reviewed evaluation output")
            if output.get("verdict") == "not_applicable":
                continue
            reviewed += 1
            useful_now = output.get("verdict") == "useful"
            useful += int(useful_now)
            useful_in_case += int(useful_now)
            labels = {str(label) for label in output.get("labels", [])}
            supported += int(not labels.intersection(_BAD_SUPPORT_LABELS))
            fragmented += int("fragmented" in labels)
            if output.get("scope_correct") is not None:
                scope_total += 1
                scope_correct += int(output.get("scope_correct") is True)
            if output.get("consolidation_correct") is not None:
                consolidation_total += 1
                consolidation_correct += int(output.get("consolidation_correct") is True)
        useful_for_recall += min(useful_in_case, expected_count)
        expected_episode = source.get("episode_expected")
        if isinstance(expected_episode, bool) and "has_episode" in result:
            classifier_total += 1
            classifier_correct += int(result.get("has_episode") is expected_episode)
        episode_review = decision.get("episode")
        if isinstance(episode_review, dict):
            episode_outputs += 1
            if episode_review.get("verdict") != "not_applicable":
                reviewed += 1
                useful_now = episode_review.get("verdict") == "useful"
                useful += int(useful_now)
                useful_episodes += int(useful_now)
                labels = {str(label) for label in episode_review.get("labels", [])}
                supported += int(not labels.intersection(_BAD_SUPPORT_LABELS))
                fragmented += int("fragmented" in labels)
    return {
        "cases": len(results),
        "precision": _ratio(supported, reviewed),
        "useful_memory_rate": _ratio(useful, reviewed),
        "useful_recall": _ratio(useful_for_recall, expected_facts),
        "over_extraction_rate": _ratio(excess_facts, fact_outputs),
        "invalid_fact_output_rate": _ratio(invalid_fact_outputs, fact_outputs),
        "fragmentation_rate": _ratio(fragmented, reviewed),
        "scope_accuracy": _ratio(scope_correct, scope_total),
        "consolidation_accuracy": _ratio(consolidation_correct, consolidation_total),
        "episode_classifier_accuracy": _ratio(classifier_correct, classifier_total),
        "episode_useful_rate": _ratio(useful_episodes, episode_outputs),
        "structured_output_reliability": _ratio(successful, len(results)),
        "provider_failure_rate": _ratio(provider_failures, len(results)),
        "median_latency_ms": round(statistics.median(latencies)) if latencies else None,
    }


def _promotion_gates(
    role: str,
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> dict[str, bool]:
    gates = {
        "precision": _at_least(candidate.get("precision"), thresholds["minimum_precision"]),
        "structured_output_reliability": _at_least(
            candidate.get("structured_output_reliability"),
            thresholds["minimum_structured_output_reliability"],
        ),
        "provider_failure_rate": _at_most(
            candidate.get("provider_failure_rate"), thresholds["maximum_provider_failure_rate"]
        ),
        "latency": _latency_within(
            candidate,
            baseline,
            _number(thresholds["maximum_latency_multiplier"], "maximum latency multiplier"),
        ),
        "precision_regression": _regression_within(
            candidate.get("precision"), baseline.get("precision"), thresholds["maximum_precision_regression"]
        ),
    }
    if role == FACT_ROLE:
        gates.update(
            {
                "useful_recall": _at_least(candidate.get("useful_recall"), thresholds["minimum_useful_recall"]),
                "useful_recall_regression": _regression_within(
                    candidate.get("useful_recall"),
                    baseline.get("useful_recall"),
                    thresholds["maximum_useful_recall_regression"],
                ),
                "over_extraction_rate": _at_most(
                    candidate.get("over_extraction_rate"), thresholds["maximum_over_extraction_rate"]
                ),
                "fragmentation_rate": _at_most(
                    candidate.get("fragmentation_rate"), thresholds["maximum_fragmentation_rate"]
                ),
                "scope_accuracy": _at_least(candidate.get("scope_accuracy"), thresholds["minimum_scope_accuracy"]),
                "consolidation_accuracy": _at_least(
                    candidate.get("consolidation_accuracy"), thresholds["minimum_consolidation_accuracy"]
                ),
                "episode_classifier_accuracy": _at_least(
                    candidate.get("episode_classifier_accuracy"),
                    thresholds["minimum_episode_classifier_accuracy"],
                ),
            }
        )
    else:
        gates["episode_useful_rate"] = _at_least(
            candidate.get("episode_useful_rate"), thresholds["minimum_episode_useful_rate"]
        )
    return gates


def render_report(report: dict[str, object]) -> str:
    lines = [
        "# Kai memory model evaluation",
        "",
        f"- Snapshot: `{report['snapshot_sha256']}`",
        f"- Evaluation: `{report['evaluation_sha256']}`",
        "- Automatic production mutation: no",
        "",
        "## Arms",
        "",
        "| Role | Model | Baseline | Precision | Useful recall | Episode quality | Reliability | Median latency |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for raw in report["arms"]:  # type: ignore[index]
        arm = _mapping(raw, "score arm")
        metrics = _mapping(arm.get("metrics"), "score metrics")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(arm["role"]),
                    str(arm["model"]),
                    "yes" if arm["baseline"] else "no",
                    _percent(metrics.get("precision")),
                    _percent(metrics.get("useful_recall")),
                    _percent(metrics.get("episode_useful_rate")),
                    _percent(metrics.get("structured_output_reliability")),
                    f"{metrics.get('median_latency_ms')} ms" if metrics.get("median_latency_ms") is not None else "n/a",
                )
            )
            + " |"
        )
    lines.extend(["", "## Promotion decisions", ""])
    for raw in report["recommendations"]:  # type: ignore[index]
        recommendation = _mapping(raw, "promotion recommendation")
        failed = [name for name, passed in _mapping(recommendation["gates"], "promotion gates").items() if not passed]
        lines.append(
            f"- `{recommendation['role_model_key']}` → `{recommendation['promoted_model']}`: "
            f"{'PROMOTE' if recommendation['promote'] else 'DO NOT PROMOTE'}"
            + (f" (failed: {', '.join(failed)})" if failed else "")
            + f"; rollback `{recommendation['rollback_model']}`"
        )
    lines.append("")
    return "\n".join(lines)


def load_plan(path: Path) -> dict[str, object]:
    return _load_json(path)


def load_evaluation(path: Path) -> dict[str, object]:
    return _load_json(path)


def load_evaluation_review(path: Path) -> dict[str, object]:
    return _load_json(path)


def write_plan_template(path: Path, document: dict[str, object]) -> None:
    _write_private_json(path, document, immutable=False)


def write_sealed_plan(path: Path, document: dict[str, object]) -> None:
    _write_private_json(path, document, immutable=True)


def write_evaluation(path: Path, document: dict[str, object]) -> None:
    _write_private_json(path, document, immutable=True)


def write_evaluation_review_template(path: Path, document: dict[str, object]) -> None:
    _write_private_json(path, document, immutable=False)


def write_sealed_evaluation_review(path: Path, document: dict[str, object]) -> None:
    _write_private_json(path, document, immutable=True)


def write_score(path: Path, report: dict[str, object]) -> None:
    markdown = path.with_suffix(".md")
    _assert_outside_source_tree(path)
    _assert_outside_source_tree(markdown)
    if path.exists():
        raise MemoryQualityCorpusError(f"Refusing to overwrite existing artifact: {path}")
    if markdown.exists():
        raise MemoryQualityCorpusError(f"Refusing to overwrite existing artifact: {markdown}")
    _write_private_json(path, report, immutable=True)
    markdown.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(markdown.parent, 0o700)
    descriptor = os.open(markdown, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(render_report(report))
    except Exception:
        path.unlink(missing_ok=True)
        markdown.unlink(missing_ok=True)
        raise
    markdown.chmod(0o400)


def _validate_sealed_plan(plan: dict[str, object], snapshot: dict[str, object], review: dict[str, object]) -> None:
    if plan.get("artifact") != "kai_memory_model_evaluation_sealed_plan" or plan.get("version") != PLAN_VERSION:
        raise MemoryQualityCorpusError("Unsupported sealed evaluation plan")
    if plan.get("snapshot_sha256") != snapshot.get("snapshot_sha256"):
        raise MemoryQualityCorpusError("Sealed plan belongs to a different snapshot")
    if plan.get("review_sha256") != review.get("review_sha256"):
        raise MemoryQualityCorpusError("Sealed plan belongs to a different review")
    if plan.get("plan_sha256") != _document_digest(plan, "plan_sha256"):
        raise MemoryQualityCorpusError("Sealed evaluation plan digest does not match")


def _validate_evaluation(evaluation: dict[str, object], snapshot: dict[str, object], review: dict[str, object]) -> None:
    if evaluation.get("artifact") != "kai_memory_model_evaluation" or evaluation.get("version") != EVALUATION_VERSION:
        raise MemoryQualityCorpusError("Unsupported memory-model evaluation")
    if evaluation.get("snapshot_sha256") != snapshot.get("snapshot_sha256"):
        raise MemoryQualityCorpusError("Evaluation belongs to a different snapshot")
    if evaluation.get("review_sha256") != review.get("review_sha256"):
        raise MemoryQualityCorpusError("Evaluation belongs to a different review")
    if evaluation.get("evaluation_sha256") != _document_digest(evaluation, "evaluation_sha256"):
        raise MemoryQualityCorpusError("Evaluation digest does not match")


def _validate_evaluation_review(review: dict[str, object], evaluation: dict[str, object]) -> None:
    if review.get("artifact") != "kai_memory_model_evaluation_sealed_review":
        raise MemoryQualityCorpusError("Evaluation review has not been sealed")
    if review.get("evaluation_sha256") != evaluation.get("evaluation_sha256"):
        raise MemoryQualityCorpusError("Evaluation review belongs to a different run")
    if review.get("evaluation_review_sha256") != _document_digest(review, "evaluation_review_sha256"):
        raise MemoryQualityCorpusError("Evaluation review digest does not match")


def _evaluation_output_bindings(
    evaluation: dict[str, object],
) -> dict[tuple[str, str], tuple[int, bool]]:
    expected: dict[tuple[str, str], tuple[int, bool]] = {}
    arms = evaluation.get("arms")
    if not isinstance(arms, list):
        raise MemoryQualityCorpusError("Evaluation arms are malformed")
    for raw_arm in arms:
        arm = _mapping(raw_arm, "evaluation arm")
        results = arm.get("results")
        if not isinstance(results, list):
            raise MemoryQualityCorpusError("Evaluation results are malformed")
        for raw_result in results:
            result = _mapping(raw_result, "evaluation result")
            binding = (_required_string(arm, "arm_id"), _required_string(result, "case_id"))
            facts = result.get("facts", [])
            if not isinstance(facts, list) or binding in expected:
                raise MemoryQualityCorpusError("Evaluation output bindings are malformed")
            expected[binding] = (len(facts), isinstance(result.get("episode"), dict))
    return expected


def _validate_review_outputs(decision: Mapping[str, object], expected: tuple[int, bool]) -> None:
    outputs = decision.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != expected[0]:
        raise MemoryQualityCorpusError("Reviewed fact outputs do not match the evaluation")
    indexes: set[int] = set()
    for raw_output in outputs:
        output = _mapping(raw_output, "reviewed fact output")
        index = output.get("output_index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= expected[0]
            or index in indexes
        ):
            raise MemoryQualityCorpusError("Reviewed fact output indexes are malformed")
        indexes.add(index)
        if output.get("verdict") not in _OUTPUT_VERDICTS:
            raise MemoryQualityCorpusError("Every fact output needs a final verdict")
        labels = output.get("labels")
        if (
            not isinstance(labels, list)
            or any(not isinstance(label, str) for label in labels)
            or not set(labels).issubset(QUALITY_LABELS)
        ):
            raise MemoryQualityCorpusError("Reviewed fact output labels are malformed")
        for field in ("scope_correct", "consolidation_correct"):
            if not isinstance(output.get(field), (bool, type(None))):
                raise MemoryQualityCorpusError(f"Reviewed fact output {field} is malformed")
    episode = decision.get("episode")
    if expected[1]:
        if not isinstance(episode, dict) or episode.get("verdict") not in _OUTPUT_VERDICTS:
            raise MemoryQualityCorpusError("Generated episode needs a final verdict")
        labels = episode.get("labels")
        if (
            not isinstance(labels, list)
            or any(not isinstance(label, str) for label in labels)
            or not set(labels).issubset(QUALITY_LABELS)
        ):
            raise MemoryQualityCorpusError("Generated episode labels are malformed")
    elif episode is not None:
        raise MemoryQualityCorpusError("Review cannot add an episode the evaluation did not produce")


def _validate_thresholds(raw: object) -> dict[str, float]:
    values = _mapping(raw, "evaluation thresholds")
    if set(values) != set(DEFAULT_THRESHOLDS):
        raise MemoryQualityCorpusError("Evaluation thresholds must contain the complete versioned threshold set")
    checked: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise MemoryQualityCorpusError(f"Evaluation threshold {name} is invalid")
        number = float(value)
        if name == "maximum_latency_multiplier":
            if number < 1.0:
                raise MemoryQualityCorpusError("Maximum latency multiplier must be at least 1")
        elif not 0.0 <= number <= 1.0:
            raise MemoryQualityCorpusError(f"Evaluation threshold {name} must be between 0 and 1")
        checked[name] = number
    return checked


def _candidate_memories(pipeline: Mapping[str, object]) -> list[MemoryResult]:
    raw_context = pipeline.get("candidate_context")
    if not isinstance(raw_context, list):
        raise MemoryQualityCorpusError("Evaluation case is missing candidate context")
    memories: list[MemoryResult] = []
    for raw in raw_context:
        candidate = _mapping(raw, "candidate context")
        if candidate.get("state") != "present":
            continue
        metadata = _mapping(candidate.get("metadata"), "candidate metadata")
        memories.append(
            MemoryResult(
                id=_required_string(candidate, "memory_id"),
                text=_required_string(candidate, "text"),
                score=0.0,
                memory_type=str(candidate.get("memory_type") or "fact"),
                metadata=dict(metadata),
                created_at=str(candidate.get("created_at") or ""),
                updated_at=str(candidate.get("updated_at") or ""),
            )
        )
    return memories


def _prior_pairs(conversation: Mapping[str, object]) -> list[tuple[str, str]]:
    raw_pairs = conversation.get("prior_pairs")
    if not isinstance(raw_pairs, list):
        raise MemoryQualityCorpusError("Evaluation case is missing prior conversation context")
    pairs: list[tuple[str, str]] = []
    for raw in raw_pairs:
        pair = _mapping(raw, "prior conversation pair")
        pairs.append((_required_string(pair, "user"), _required_string(pair, "assistant")))
    return pairs


def _cases(snapshot: Mapping[str, object]) -> list[dict[str, object]]:
    raw = snapshot.get("cases")
    if not isinstance(raw, list):
        raise MemoryQualityCorpusError("Snapshot cases are malformed")
    return [_mapping(case, "snapshot case") for case in raw]


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MemoryQualityCorpusError(f"{label.capitalize()} must be an object")
    return value


def _required_string(value: Mapping[str, object], field: str) -> str:
    raw = value.get(field)
    if not isinstance(raw, str) or not raw.strip():
        raise MemoryQualityCorpusError(f"{field} must be a non-empty string")
    return raw.strip()


def _opaque_id(kind: str, value: str) -> str:
    digest = hashlib.sha256(f"kai-memory-eval:{kind}\0{value}".encode()).hexdigest()
    return f"{kind[:3]}_{digest[:32]}"


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _at_least(value: object, threshold: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) >= _number(threshold, "threshold")
    )


def _at_most(value: object, threshold: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) <= _number(threshold, "threshold")
    )


def _regression_within(value: object, baseline: object, maximum: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if not isinstance(baseline, (int, float)) or isinstance(baseline, bool):
        return False
    return float(baseline) - float(value) <= _number(maximum, "maximum regression")


def _latency_within(candidate: Mapping[str, object], baseline: Mapping[str, object], multiplier: float) -> bool:
    candidate_latency = candidate.get("median_latency_ms")
    baseline_latency = baseline.get("median_latency_ms")
    if not isinstance(candidate_latency, (int, float)) or isinstance(candidate_latency, bool):
        return False
    if not isinstance(baseline_latency, (int, float)) or isinstance(baseline_latency, bool):
        return False
    return float(candidate_latency) <= max(1.0, float(baseline_latency)) * multiplier


def _percent(value: object) -> str:
    return "n/a" if value is None else f"{_number(value, 'metric'):.1%}"


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MemoryQualityCorpusError(f"{label.capitalize()} must be numeric")
    return float(value)


def dump_canonical_json(value: dict[str, object]) -> str:
    """Expose stable JSON for operator-side plan inspection and tests."""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
