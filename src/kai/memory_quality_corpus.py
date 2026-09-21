"""Private, operator-reviewed corpus tooling for production memory quality.

The corpus deliberately lives outside the source tree.  A snapshot contains
private canonical conversation text and the memory rows attributable to each
production extraction receipt, so every writer in this module creates 0700
directories and 0600 files.  Hashes cover snapshot inputs and sealed reviewer
decisions independently; scoring refuses drifted artifacts.

This module never invokes a model.  It establishes the human-reviewed baseline
that later model-comparison work can consume without changing the evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kai.memory import MemoryResult

CORPUS_VERSION = 2
SUPPORTED_CORPUS_VERSIONS = frozenset({1, CORPUS_VERSION})
REVIEW_VERSION = 1

QUALITY_LABELS = frozenset(
    {
        "useful",
        "incorrect",
        "unsupported",
        "transient",
        "stale_on_arrival",
        "redundant",
        "fragmented",
        "wrongly_scoped",
        "wrong_speaker",
        "missed_update",
        "episode_false_positive",
        "episode_false_negative",
    }
)

SCENARIO_TAGS = frozenset(
    {
        "changed_value",
        "refinement",
        "repeat",
        "negation",
        "preference_reversal",
        "renamed_resource",
        "completed_workflow",
        "routine_acknowledgment",
        "multi_agent",
        "project_scope",
        "global_scope",
    }
)

_VERDICTS = frozenset({"useful", "not_useful", "not_applicable"})
_BOOL_OR_NONE = (bool, type(None))
_MIN_READY_CASES = 30


class MemoryQualityCorpusError(RuntimeError):
    """A corpus artifact is malformed, unauthorized, or has drifted."""


@dataclass(frozen=True, slots=True)
class ProductionReceipt:
    receipt_id: str
    principal_id: str
    runtime_profile_id: str
    run_id: str
    source_message_id: str
    result_message_id: str
    extraction_role: str
    backend: str
    provider: str
    model: str
    prompt_version: str
    schema_version: str
    policy_version: str
    status: str
    decision_outcome: str | None
    classifier_result: bool | None
    proposed_intents: tuple[dict[str, object], ...]
    validation_outcome: dict[str, object]
    storage_outcome: dict[str, object]
    memory_scopes: tuple[dict[str, object], ...]
    candidate_ids: tuple[str, ...]
    created_at: str
    completed_at: str | None
    channel_id: str
    channel_kind: str
    channel_name: str | None
    agent_handle: str | None
    agent_display_name: str
    source_body: str
    result_body: str
    prior_pairs: tuple[tuple[str, str], ...] = ()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _document_digest(document: dict[str, object], field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _case_id(receipt_id: str) -> str:
    digest = hashlib.sha256(f"memory-quality\0{receipt_id}".encode()).hexdigest()
    return f"mqc_{digest[:32]}"


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MemoryQualityCorpusError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MemoryQualityCorpusError(f"{path} must contain one JSON object")
    return value


def _write_private_json(path: Path, value: dict[str, object], *, immutable: bool) -> None:
    _assert_outside_source_tree(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists():
        raise MemoryQualityCorpusError(f"Refusing to overwrite existing artifact: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    os.chmod(path, 0o400 if immutable else 0o600)


def _assert_outside_source_tree(path: Path) -> None:
    from kai.config import PROJECT_ROOT

    resolved = path.resolve(strict=False)
    project_root = PROJECT_ROOT.resolve()
    if resolved == project_root or resolved.is_relative_to(project_root):
        raise MemoryQualityCorpusError("Private memory-quality artifacts cannot be written inside the source tree")


def resolve_human_principal(connection: sqlite3.Connection, selector: str) -> str:
    """Resolve a canonical principal ID or an unambiguous external subject."""
    direct = connection.execute(
        "SELECT id FROM principals WHERE id = ? AND kind = 'human'",
        (selector,),
    ).fetchall()
    if len(direct) == 1:
        return str(direct[0][0])
    external = connection.execute(
        "SELECT DISTINCT p.id FROM principals p "
        "JOIN external_identities e ON e.principal_id = p.id "
        "WHERE p.kind = 'human' AND e.external_subject = ?",
        (selector,),
    ).fetchall()
    if len(external) != 1:
        raise MemoryQualityCorpusError("Principal selector is missing, ambiguous, or not a human")
    return str(external[0][0])


def load_production_receipts(
    connection: sqlite3.Connection,
    principal_id: str,
    *,
    limit: int,
    seed: int,
    context_turns: int = 3,
) -> list[ProductionReceipt]:
    """Load a deterministic, role-balanced sample owned by one principal."""
    if limit < 1 or limit > 1000:
        raise MemoryQualityCorpusError("Sample limit must be between 1 and 1000")
    if isinstance(context_turns, bool) or not isinstance(context_turns, int) or not 0 <= context_turns <= 10:
        raise MemoryQualityCorpusError("Context turns must be between 0 and 10")
    rows = connection.execute(
        "SELECT receipt.receipt_id, receipt.principal_id, receipt.runtime_profile_id, "
        "receipt.run_id, receipt.source_message_id, receipt.result_message_id, "
        "receipt.extraction_role, receipt.backend, receipt.provider, receipt.model, "
        "receipt.prompt_version, receipt.schema_version, receipt.policy_version, "
        "receipt.status, receipt.decision_outcome, receipt.classifier_result, "
        "receipt.proposed_intents_json, receipt.validation_outcome_json, "
        "receipt.storage_outcome_json, receipt.memory_scope_json, receipt.candidate_ids_json, "
        "receipt.created_at, receipt.completed_at, run.channel_id, channel.kind, channel.name, definition.handle, "
        "agent_principal.display_name, source.body, result.body "
        "FROM memory_extraction_receipts receipt "
        "JOIN runtime_profile_owners owner "
        "ON owner.runtime_profile_id = receipt.runtime_profile_id "
        "AND owner.principal_id = receipt.principal_id "
        "JOIN runs run ON run.id = receipt.run_id "
        "AND run.requested_by_principal_id = receipt.principal_id "
        "JOIN channels channel ON channel.id = run.channel_id "
        "JOIN agents agent ON agent.id = run.agent_id "
        "JOIN principals agent_principal ON agent_principal.id = agent.principal_id "
        "LEFT JOIN agent_definitions definition ON definition.agent_id = agent.id "
        "JOIN messages source ON source.id = receipt.source_message_id "
        "JOIN messages result ON result.id = receipt.result_message_id "
        "WHERE receipt.principal_id = ? AND receipt.status IN ('completed', 'failed') "
        "ORDER BY receipt.created_at DESC LIMIT 5000",
        (principal_id,),
    ).fetchall()
    parsed = [_receipt_from_row(row) for row in rows]
    rng = random.Random(seed)
    by_role: dict[str, list[ProductionReceipt]] = {}
    for receipt in parsed:
        by_role.setdefault(receipt.extraction_role, []).append(receipt)
    for receipts in by_role.values():
        rng.shuffle(receipts)
    selected: list[ProductionReceipt] = []
    roles = sorted(by_role)
    while roles and len(selected) < limit:
        next_roles: list[str] = []
        for role in roles:
            receipts = by_role[role]
            if receipts and len(selected) < limit:
                selected.append(receipts.pop())
            if receipts:
                next_roles.append(role)
        roles = next_roles
    return [
        replace(
            receipt,
            prior_pairs=_load_prior_pairs(connection, receipt.run_id, limit=context_turns),
        )
        for receipt in selected
    ]


def _load_prior_pairs(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    limit: int,
) -> tuple[tuple[str, str], ...]:
    if limit == 0:
        return ()
    rows = connection.execute(
        "SELECT prior_source.body, prior_result.body FROM runs current "
        "JOIN messages current_source ON current_source.id = current.inbound_message_id "
        "JOIN runs prior ON prior.channel_id = current.channel_id "
        "AND prior.agent_id = current.agent_id "
        "AND prior.requested_by_principal_id = current.requested_by_principal_id "
        "JOIN messages prior_source ON prior_source.id = prior.inbound_message_id "
        "JOIN messages prior_result ON prior_result.id = prior.result_message_id "
        "WHERE current.id = ? AND prior.status = 'completed' "
        "AND prior_source.created_event_position < current_source.created_event_position "
        "ORDER BY prior_source.created_event_position DESC, prior.id DESC LIMIT ?",
        (run_id, limit),
    ).fetchall()
    return tuple((str(row[0]), str(row[1])) for row in reversed(rows))


def _receipt_from_row(row: Iterable[object]) -> ProductionReceipt:
    values = tuple(row)
    return ProductionReceipt(
        receipt_id=str(values[0]),
        principal_id=str(values[1]),
        runtime_profile_id=str(values[2]),
        run_id=str(values[3]),
        source_message_id=str(values[4]),
        result_message_id=str(values[5]),
        extraction_role=str(values[6]),
        backend=str(values[7]),
        provider=str(values[8]),
        model=str(values[9]),
        prompt_version=str(values[10]),
        schema_version=str(values[11]),
        policy_version=str(values[12]),
        status=str(values[13]),
        decision_outcome=str(values[14]) if values[14] is not None else None,
        classifier_result=None if values[15] is None else bool(values[15]),
        proposed_intents=tuple(json.loads(str(values[16]))),
        validation_outcome=dict(json.loads(str(values[17]))),
        storage_outcome=dict(json.loads(str(values[18]))),
        memory_scopes=tuple(json.loads(str(values[19]))),
        candidate_ids=tuple(str(value) for value in json.loads(str(values[20]))),
        created_at=str(values[21]),
        completed_at=str(values[22]) if values[22] is not None else None,
        channel_id=str(values[23]),
        channel_kind=str(values[24]),
        channel_name=str(values[25]) if values[25] is not None else None,
        agent_handle=str(values[26]) if values[26] is not None else None,
        agent_display_name=str(values[27]),
        source_body=str(values[28]),
        result_body=str(values[29]),
    )


def build_snapshot(
    *,
    principal_id: str,
    receipts: list[ProductionReceipt],
    memory_lookup: Callable[[str, str, str], MemoryResult | None],
    seed: int,
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for receipt in receipts:
        decisions = receipt.storage_outcome.get("decisions", [])
        encoded_decisions: list[dict[str, object]] = []
        if isinstance(decisions, list):
            for raw in decisions:
                if not isinstance(raw, dict):
                    continue
                decision = dict(raw)
                for key in ("new_memory_id", "replaced_memory_id"):
                    memory_id = raw.get(key)
                    if isinstance(memory_id, str) and memory_id:
                        memory_row = memory_lookup(principal_id, receipt.runtime_profile_id, memory_id)
                        decision[f"{key}_snapshot"] = _encode_memory(memory_row, memory_id)
                encoded_decisions.append(decision)
        case: dict[str, object] = {
            "case_id": _case_id(receipt.receipt_id),
            "receipt_id": receipt.receipt_id,
            "extraction_role": receipt.extraction_role,
            "provenance": {
                "run_id": receipt.run_id,
                "source_message_id": receipt.source_message_id,
                "result_message_id": receipt.result_message_id,
                "runtime_profile_id": receipt.runtime_profile_id,
                "created_at": receipt.created_at,
                "completed_at": receipt.completed_at,
            },
            "conversation": {
                "channel_id": receipt.channel_id,
                "channel_kind": receipt.channel_kind,
                "channel_name": receipt.channel_name,
                "agent_handle": receipt.agent_handle,
                "agent_display_name": receipt.agent_display_name,
                "user": receipt.source_body,
                "assistant": receipt.result_body,
                "prior_pairs": [
                    {"user": user_text, "assistant": assistant_text}
                    for user_text, assistant_text in receipt.prior_pairs
                ],
            },
            "pipeline": {
                "backend": receipt.backend,
                "provider": receipt.provider,
                "model": receipt.model,
                "prompt_version": receipt.prompt_version,
                "schema_version": receipt.schema_version,
                "policy_version": receipt.policy_version,
                "status": receipt.status,
                "decision_outcome": receipt.decision_outcome,
                "classifier_result": receipt.classifier_result,
                "proposed_intents": list(receipt.proposed_intents),
                "validation_outcome": receipt.validation_outcome,
                "memory_scopes": list(receipt.memory_scopes),
                "candidate_context": [
                    _encode_memory(
                        memory_lookup(principal_id, receipt.runtime_profile_id, memory_id),
                        memory_id,
                    )
                    for memory_id in receipt.candidate_ids
                ],
                "storage_decisions": encoded_decisions,
            },
        }
        case["input_sha256"] = hashlib.sha256(
            _canonical_json({"conversation": case["conversation"], "pipeline": case["pipeline"]}).encode("utf-8")
        ).hexdigest()
        cases.append(case)
    document: dict[str, object] = {
        "artifact": "kai_memory_quality_snapshot",
        "version": CORPUS_VERSION,
        "principal_id": principal_id,
        "created_at": _now(),
        "selection": {"seed": seed, "requested": len(receipts)},
        "case_count": len(cases),
        "cases": cases,
    }
    document["snapshot_sha256"] = _document_digest(document, "snapshot_sha256")
    return document


def _encode_memory(row: MemoryResult | None, memory_id: str) -> dict[str, object]:
    if row is None:
        return {"memory_id": memory_id, "state": "missing"}
    return {
        "memory_id": row.id,
        "state": "present",
        "text": row.text,
        "memory_type": row.memory_type,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "metadata": row.metadata,
    }


def validate_snapshot(document: dict[str, object]) -> None:
    if (
        document.get("artifact") != "kai_memory_quality_snapshot"
        or document.get("version") not in SUPPORTED_CORPUS_VERSIONS
    ):
        raise MemoryQualityCorpusError("Unsupported memory-quality snapshot")
    expected = document.get("snapshot_sha256")
    if not isinstance(expected, str) or expected != _document_digest(document, "snapshot_sha256"):
        raise MemoryQualityCorpusError("Snapshot digest does not match its immutable inputs")
    cases = document.get("cases")
    if not isinstance(cases, list) or document.get("case_count") != len(cases):
        raise MemoryQualityCorpusError("Snapshot case count is malformed")
    ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(ids) != len(cases) or len(set(ids)) != len(ids) or not all(isinstance(value, str) for value in ids):
        raise MemoryQualityCorpusError("Snapshot case identities are malformed")


def build_review_template(snapshot: dict[str, object]) -> dict[str, object]:
    validate_snapshot(snapshot)
    cases = snapshot["cases"]
    assert isinstance(cases, list)
    return {
        "artifact": "kai_memory_quality_review",
        "version": REVIEW_VERSION,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "principal_id": snapshot["principal_id"],
        "created_at": _now(),
        "instructions": {
            "verdicts": sorted(_VERDICTS),
            "quality_labels": sorted(QUALITY_LABELS),
            "scenario_tags": sorted(SCENARIO_TAGS),
        },
        "decisions": [_empty_decision(case) for case in cases if isinstance(case, dict)],
    }


def _empty_decision(case: dict[str, object]) -> dict[str, object]:
    pipeline = case.get("pipeline")
    decisions = pipeline.get("storage_decisions", []) if isinstance(pipeline, dict) else []
    output_reviews: list[dict[str, object]] = []
    if isinstance(decisions, list):
        for index, decision in enumerate(decisions):
            if not isinstance(decision, dict):
                continue
            snapshot = decision.get("new_memory_id_snapshot")
            output_reviews.append(
                {
                    "decision_index": decision.get("index", index),
                    "memory_id": snapshot.get("memory_id") if isinstance(snapshot, dict) else None,
                    "verdict": "pending",
                    "labels": [],
                    "scope_correct": None,
                    "consolidation_correct": None,
                    "note": "",
                }
            )
    return {
        "case_id": case["case_id"],
        "review_status": "pending",
        "scenario_tags": [],
        "case_labels": [],
        "expected_fact_count": None,
        "episode_expected": None,
        "update_expected": None,
        "update_detected": None,
        "outputs": output_reviews,
        "note": "",
    }


def seal_review(snapshot: dict[str, object], review: dict[str, object], *, reviewer: str) -> dict[str, object]:
    validate_snapshot(snapshot)
    if review.get("artifact") != "kai_memory_quality_review" or review.get("version") != REVIEW_VERSION:
        raise MemoryQualityCorpusError("Unsupported memory-quality review")
    if review.get("snapshot_sha256") != snapshot.get("snapshot_sha256"):
        raise MemoryQualityCorpusError("Review belongs to a different snapshot")
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise MemoryQualityCorpusError("Review decisions must be a list")
    cases = snapshot["cases"]
    assert isinstance(cases, list)
    cases_by_id = {str(case["case_id"]): case for case in cases if isinstance(case, dict)}
    case_ids = set(cases_by_id)
    decision_ids: set[str] = set()
    for decision in decisions:
        _validate_decision(decision)
        assert isinstance(decision, dict)
        case_id = decision.get("case_id")
        assert isinstance(case_id, str)
        if case_id in decision_ids or case_id not in case_ids:
            raise MemoryQualityCorpusError("Review has duplicate or unknown case identities")
        _validate_output_bindings(cases_by_id[case_id], decision)
        decision_ids.add(case_id)
    if decision_ids != case_ids:
        raise MemoryQualityCorpusError("Review must contain every snapshot case exactly once")
    if not reviewer.strip():
        raise MemoryQualityCorpusError("Reviewer must be non-empty")
    sealed = dict(review)
    sealed.pop("instructions", None)
    sealed["artifact"] = "kai_memory_quality_sealed_review"
    sealed["reviewer"] = reviewer.strip()
    sealed["sealed_at"] = _now()
    sealed["review_sha256"] = _document_digest(sealed, "review_sha256")
    return sealed


def _validate_output_bindings(case: dict[str, object], decision: dict[str, object]) -> None:
    pipeline = case.get("pipeline")
    raw_storage = pipeline.get("storage_decisions", []) if isinstance(pipeline, dict) else []
    expected: set[tuple[object, object]] = set()
    if isinstance(raw_storage, list):
        for fallback, storage in enumerate(raw_storage):
            if not isinstance(storage, dict):
                continue
            memory_snapshot = storage.get("new_memory_id_snapshot")
            memory_id = memory_snapshot.get("memory_id") if isinstance(memory_snapshot, dict) else None
            expected.add((storage.get("index", fallback), memory_id))
    raw_outputs = decision.get("outputs")
    assert isinstance(raw_outputs, list)
    actual = {
        (output.get("decision_index"), output.get("memory_id")) for output in raw_outputs if isinstance(output, dict)
    }
    if len(actual) != len(raw_outputs) or actual != expected:
        raise MemoryQualityCorpusError("Review outputs do not match the immutable snapshot decisions")


def _validate_decision(value: object) -> None:
    if not isinstance(value, dict) or value.get("review_status") != "complete":
        raise MemoryQualityCorpusError("Every review decision must be marked complete")
    _validate_string_set(value.get("scenario_tags"), SCENARIO_TAGS, "scenario tags")
    _validate_string_set(value.get("case_labels"), QUALITY_LABELS, "case labels")
    for field in ("episode_expected", "update_expected", "update_detected"):
        if not isinstance(value.get(field), _BOOL_OR_NONE):
            raise MemoryQualityCorpusError(f"Invalid {field}")
    fact_count = value.get("expected_fact_count")
    if fact_count is not None and (isinstance(fact_count, bool) or not isinstance(fact_count, int) or fact_count < 0):
        raise MemoryQualityCorpusError("Invalid expected fact count")
    outputs = value.get("outputs")
    if not isinstance(outputs, list):
        raise MemoryQualityCorpusError("Output reviews must be a list")
    for output in outputs:
        if not isinstance(output, dict) or output.get("verdict") not in _VERDICTS:
            raise MemoryQualityCorpusError("Every output needs a final verdict")
        _validate_string_set(output.get("labels"), QUALITY_LABELS, "output labels")
        for field in ("scope_correct", "consolidation_correct"):
            if not isinstance(output.get(field), _BOOL_OR_NONE):
                raise MemoryQualityCorpusError(f"Invalid output {field}")


def _validate_string_set(value: object, allowed: frozenset[str], label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) or item not in allowed for item in value):
        raise MemoryQualityCorpusError(f"Invalid {label}")
    if len(value) != len(set(value)):
        raise MemoryQualityCorpusError(f"Duplicate {label}")


def validate_sealed_review(review: dict[str, object], snapshot: dict[str, object]) -> None:
    if review.get("artifact") != "kai_memory_quality_sealed_review":
        raise MemoryQualityCorpusError("Review has not been sealed")
    expected = review.get("review_sha256")
    if not isinstance(expected, str) or expected != _document_digest(review, "review_sha256"):
        raise MemoryQualityCorpusError("Sealed review digest does not match its decisions")
    if review.get("snapshot_sha256") != snapshot.get("snapshot_sha256"):
        raise MemoryQualityCorpusError("Sealed review belongs to a different snapshot")
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise MemoryQualityCorpusError("Review decisions must be a list")
    cases = snapshot.get("cases")
    assert isinstance(cases, list)
    cases_by_id = {str(case["case_id"]): case for case in cases if isinstance(case, dict)}
    seen: set[str] = set()
    for decision in decisions:
        _validate_decision(decision)
        assert isinstance(decision, dict)
        case_id = decision.get("case_id")
        if not isinstance(case_id, str) or case_id not in cases_by_id or case_id in seen:
            raise MemoryQualityCorpusError("Review has duplicate or unknown case identities")
        _validate_output_bindings(cases_by_id[case_id], decision)
        seen.add(case_id)
    if seen != set(cases_by_id):
        raise MemoryQualityCorpusError("Review must contain every snapshot case exactly once")


def score_review(snapshot: dict[str, object], review: dict[str, object]) -> dict[str, object]:
    validate_snapshot(snapshot)
    validate_sealed_review(review, snapshot)
    decisions = review["decisions"]
    assert isinstance(decisions, list)
    reviewed_outputs = useful_outputs = supported_outputs = 0
    duplicated = fragmented = 0
    scope_total = scope_correct = consolidation_total = consolidation_correct = 0
    update_expected = update_detected = 0
    episode_expected = episode_classifier_correct = episode_outputs = useful_episode_outputs = 0
    scenario_coverage: set[str] = set()
    case_by_id = {case["case_id"]: case for case in snapshot["cases"] if isinstance(case, dict)}  # type: ignore[index]
    for decision in decisions:
        assert isinstance(decision, dict)
        scenario_coverage.update(str(value) for value in decision["scenario_tags"])
        if decision.get("update_expected") is True:
            update_expected += 1
            update_detected += int(decision.get("update_detected") is True)
        case = case_by_id[str(decision["case_id"])]
        pipeline = case["pipeline"]
        assert isinstance(pipeline, dict)
        if decision.get("episode_expected") is not None:
            episode_expected += 1
            episode_classifier_correct += int(bool(pipeline.get("classifier_result")) is decision["episode_expected"])
        for output in decision["outputs"]:
            assert isinstance(output, dict)
            verdict = output["verdict"]
            if verdict == "not_applicable":
                continue
            reviewed_outputs += 1
            useful_outputs += int(verdict == "useful")
            labels = set(output["labels"])
            supported_outputs += int(
                not labels.intersection({"incorrect", "unsupported", "stale_on_arrival", "wrong_speaker"})
            )
            duplicated += int("redundant" in labels)
            fragmented += int("fragmented" in labels)
            if output.get("scope_correct") is not None:
                scope_total += 1
                scope_correct += int(output["scope_correct"] is True)
            if output.get("consolidation_correct") is not None:
                consolidation_total += 1
                consolidation_correct += int(output["consolidation_correct"] is True)
            if case.get("extraction_role") == "episode_generation":
                episode_outputs += 1
                useful_episode_outputs += int(verdict == "useful")
    role_coverage = sorted({str(case.get("extraction_role")) for case in case_by_id.values()})
    report: dict[str, object] = {
        "artifact": "kai_memory_quality_score",
        "version": 1,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "review_sha256": review["review_sha256"],
        "generated_at": _now(),
        "counts": {
            "cases": len(decisions),
            "reviewed_outputs": reviewed_outputs,
            "useful_outputs": useful_outputs,
            "supported_outputs": supported_outputs,
            "update_cases": update_expected,
            "episode_expectations": episode_expected,
            "episode_outputs": episode_outputs,
        },
        "metrics": {
            "precision": _ratio(supported_outputs, reviewed_outputs),
            "useful_memory_rate": _ratio(useful_outputs, reviewed_outputs),
            "duplication_rate": _ratio(duplicated, reviewed_outputs),
            "fragmentation_rate": _ratio(fragmented, reviewed_outputs),
            "update_detection_rate": _ratio(update_detected, update_expected),
            "episode_classifier_accuracy": _ratio(episode_classifier_correct, episode_expected),
            "episode_useful_rate": _ratio(useful_episode_outputs, episode_outputs),
            "scope_accuracy": _ratio(scope_correct, scope_total),
            "consolidation_accuracy": _ratio(consolidation_correct, consolidation_total),
        },
        "coverage": {
            "roles": role_coverage,
            "scenario_tags": sorted(scenario_coverage),
            "missing_scenario_tags": sorted(SCENARIO_TAGS - scenario_coverage),
        },
    }
    coverage = report["coverage"]
    assert isinstance(coverage, dict)
    report["qualification_ready"] = (
        len(decisions) >= _MIN_READY_CASES
        and set(role_coverage) == {"episode_generation", "fact_extraction"}
        and not coverage["missing_scenario_tags"]
        and reviewed_outputs > 0
        and update_expected > 0
        and episode_expected > 0
        and episode_outputs > 0
        and scope_total > 0
        and consolidation_total > 0
    )
    return report


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def render_markdown_report(report: dict[str, object]) -> str:
    counts = report["counts"]
    metrics = report["metrics"]
    coverage = report["coverage"]
    assert isinstance(counts, dict) and isinstance(metrics, dict) and isinstance(coverage, dict)
    lines = [
        "# Kai memory-quality baseline",
        "",
        f"- Snapshot: `{report['snapshot_sha256']}`",
        f"- Sealed review: `{report['review_sha256']}`",
        f"- Cases: {counts['cases']}",
        f"- Qualification ready: {'yes' if report['qualification_ready'] else 'no'}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for name, value in metrics.items():
        rendered = "not measured" if value is None else f"{float(value):.1%}"
        lines.append(f"| {str(name).replace('_', ' ').title()} | {rendered} |")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Roles: {', '.join(coverage['roles']) or 'none'}",
            f"- Scenario tags: {', '.join(coverage['scenario_tags']) or 'none'}",
            f"- Missing scenario tags: {', '.join(coverage['missing_scenario_tags']) or 'none'}",
            "",
        ]
    )
    return "\n".join(lines)


def load_snapshot(path: Path) -> dict[str, object]:
    document = _load_json(path)
    validate_snapshot(document)
    return document


def load_review(path: Path) -> dict[str, object]:
    return _load_json(path)


def write_snapshot(path: Path, document: dict[str, object]) -> None:
    validate_snapshot(document)
    _write_private_json(path, document, immutable=True)


def write_review_template(path: Path, document: dict[str, object]) -> None:
    _write_private_json(path, document, immutable=False)


def write_sealed_review(path: Path, document: dict[str, object], snapshot: dict[str, object]) -> None:
    validate_sealed_review(document, snapshot)
    _write_private_json(path, document, immutable=True)


def write_score(path: Path, report: dict[str, object]) -> None:
    markdown_path = path.with_suffix(".md")
    _assert_outside_source_tree(path)
    _assert_outside_source_tree(markdown_path)
    if path.exists():
        raise MemoryQualityCorpusError(f"Refusing to overwrite existing artifact: {path}")
    if markdown_path.exists():
        raise MemoryQualityCorpusError(f"Refusing to overwrite existing artifact: {markdown_path}")
    _write_private_json(path, report, immutable=True)
    descriptor = os.open(markdown_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(render_markdown_report(report))
    except Exception:
        path.unlink(missing_ok=True)
        markdown_path.unlink(missing_ok=True)
        raise
    os.chmod(markdown_path, 0o400)
