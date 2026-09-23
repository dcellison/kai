"""Canonical temporal fact and episode lifecycle vocabulary and projection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import aiosqlite

from kai.workshop.domain import (
    MemoryClaimId,
    MemoryEpisodeId,
    MemoryRevisionId,
    PrincipalId,
    RuntimeProfileId,
    WorkshopEventType,
)
from kai.workshop.store import StoredEvent

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FactLifecycleState(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    EXPIRED = "expired"
    UNRESOLVED_CONFLICT = "unresolved_conflict"


class FactLifecycleTransition(StrEnum):
    RECORDED = "recorded"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    EXPIRED = "expired"
    CONFLICT_OPENED = "conflict_opened"
    CONFLICT_RESOLVED = "conflict_resolved"


class EpisodeFollowupRelationship(StrEnum):
    REVISITED = "revisited"
    OUTCOME_CHANGED = "outcome_changed"
    INVALIDATED_CONCLUSION = "invalidated_conclusion"
    REPEATED = "repeated"
    RESOLVED_BY = "resolved_by"


class LegacyMemoryClassification(StrEnum):
    CANONICAL = "canonical"
    LEGACY_COMPLETE = "legacy_complete"
    LEGACY_INCOMPLETE = "legacy_incomplete"
    LEGACY_QUARANTINED = "legacy_quarantined"


class MemoryAdmissionAuthority(StrEnum):
    PROVENANCE_VERIFIED = "provenance_verified"
    OPERATOR_REVIEW = "operator_review"
    QUARANTINED = "quarantined"


class LegacyTemporalGap(StrEnum):
    ASSERTION_TIME = "assertion_time"
    OBSERVATION_TIME = "observation_time"
    OCCURRENCE_TIME = "occurrence_time"
    EVIDENCE = "evidence"
    PROVENANCE = "provenance"
    SCOPE = "scope"


@dataclass(frozen=True, slots=True)
class LegacyTemporalClassification:
    classification: LegacyMemoryClassification
    gaps: tuple[LegacyTemporalGap, ...]


def classify_legacy_temporal_metadata(
    metadata: dict[str, object],
    *,
    memory_kind: str,
) -> LegacyTemporalClassification:
    """Classify legacy rows without inventing missing temporal authority."""
    if memory_kind not in {"fact", "episode"}:
        return LegacyTemporalClassification(
            LegacyMemoryClassification.LEGACY_QUARANTINED,
            (LegacyTemporalGap.PROVENANCE,),
        )
    gaps: list[LegacyTemporalGap] = []
    if memory_kind == "fact":
        if not metadata.get("asserted_at"):
            gaps.append(LegacyTemporalGap.ASSERTION_TIME)
        if not metadata.get("observed_at"):
            gaps.append(LegacyTemporalGap.OBSERVATION_TIME)
    elif not metadata.get("occurred_from"):
        gaps.append(LegacyTemporalGap.OCCURRENCE_TIME)
    if not metadata.get("evidence"):
        gaps.append(LegacyTemporalGap.EVIDENCE)
    if not metadata.get("scope") or (metadata.get("scope") == "project" and not metadata.get("project_id")):
        gaps.append(LegacyTemporalGap.SCOPE)
    if not metadata.get("source") or not (metadata.get("model") or metadata.get("operator_principal_id")):
        gaps.append(LegacyTemporalGap.PROVENANCE)
    return LegacyTemporalClassification(
        LegacyMemoryClassification.LEGACY_COMPLETE if not gaps else LegacyMemoryClassification.LEGACY_INCOMPLETE,
        tuple(gaps),
    )


def validate_fact_transition(
    current: FactLifecycleState,
    transition: FactLifecycleTransition,
) -> FactLifecycleState:
    """Return the deterministic next state or reject an invalid transition."""
    allowed = {
        (FactLifecycleState.ACTIVE, FactLifecycleTransition.SUPERSEDED): FactLifecycleState.SUPERSEDED,
        (FactLifecycleState.ACTIVE, FactLifecycleTransition.RETRACTED): FactLifecycleState.RETRACTED,
        (FactLifecycleState.ACTIVE, FactLifecycleTransition.EXPIRED): FactLifecycleState.EXPIRED,
        (
            FactLifecycleState.ACTIVE,
            FactLifecycleTransition.CONFLICT_OPENED,
        ): FactLifecycleState.UNRESOLVED_CONFLICT,
        (
            FactLifecycleState.UNRESOLVED_CONFLICT,
            FactLifecycleTransition.CONFLICT_RESOLVED,
        ): FactLifecycleState.ACTIVE,
    }
    try:
        return allowed[(current, transition)]
    except KeyError as exc:
        raise ValueError(f"Invalid memory fact transition: {current.value} -> {transition.value}") from exc


def _required_text(value: object, *, field: str, maximum: int = 16384) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"Temporal memory requires non-empty {field}")
    return value


def _optional_text(value: object, *, field: str, maximum: int = 256) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"Temporal memory {field} is invalid")
    return value


def _timestamp(value: object, *, field: str, required: bool = False) -> str | None:
    text = _optional_text(value, field=field, maximum=64)
    if text is None:
        if required:
            raise ValueError(f"Temporal memory requires {field}")
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Temporal memory {field} is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Temporal memory {field} must include a timezone")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds")


def _scope(payload: dict[str, Any]) -> tuple[str, str]:
    kind = _required_text(payload.get("scope_kind"), field="scope_kind", maximum=16)
    key_value = payload.get("scope_key")
    if not isinstance(key_value, str):
        raise ValueError("Temporal memory scope_key must be a string")
    if kind == "global" and key_value:
        raise ValueError("Global temporal memory cannot carry a scope key")
    if kind == "project" and (not key_value or len(key_value) > 256):
        raise ValueError("Project temporal memory requires a bounded scope key")
    if kind not in {"global", "project"}:
        raise ValueError("Temporal memory scope kind is invalid")
    return kind, key_value


def _migration(payload: dict[str, Any]) -> tuple[str, str]:
    classification = LegacyMemoryClassification(
        _required_text(payload.get("migration_classification"), field="migration_classification", maximum=32)
    )
    raw_gaps = payload.get("migration_gaps")
    if not isinstance(raw_gaps, list) or len(raw_gaps) > 16:
        raise ValueError("Temporal memory migration_gaps must be a bounded list")
    gaps = tuple(LegacyTemporalGap(_required_text(item, field="migration gap", maximum=32)) for item in raw_gaps)
    if len(set(gaps)) != len(gaps):
        raise ValueError("Temporal memory migration gaps must be unique")
    if classification in {LegacyMemoryClassification.CANONICAL, LegacyMemoryClassification.LEGACY_COMPLETE}:
        if gaps:
            raise ValueError("Complete temporal memory cannot carry migration gaps")
    elif not gaps:
        raise ValueError("Incomplete temporal memory must identify its migration gaps")
    return classification.value, json.dumps([gap.value for gap in gaps], separators=(",", ":"))


def resolve_memory_admission(
    migration_classification: str,
    admission_authority: object = None,
) -> MemoryAdmissionAuthority:
    """Resolve explicit admission separately from immutable provenance quality."""
    classification = LegacyMemoryClassification(migration_classification)
    if admission_authority is None:
        return (
            MemoryAdmissionAuthority.PROVENANCE_VERIFIED
            if classification in {LegacyMemoryClassification.CANONICAL, LegacyMemoryClassification.LEGACY_COMPLETE}
            else MemoryAdmissionAuthority.QUARANTINED
        )
    authority = MemoryAdmissionAuthority(_required_text(admission_authority, field="admission_authority", maximum=32))
    if authority == MemoryAdmissionAuthority.PROVENANCE_VERIFIED and classification not in {
        LegacyMemoryClassification.CANONICAL,
        LegacyMemoryClassification.LEGACY_COMPLETE,
    }:
        raise ValueError("Incomplete temporal memory cannot claim provenance-verified admission")
    if authority == MemoryAdmissionAuthority.QUARANTINED and classification in {
        LegacyMemoryClassification.CANONICAL,
        LegacyMemoryClassification.LEGACY_COMPLETE,
    }:
        raise ValueError("Complete temporal memory cannot be quarantined by provenance")
    if (
        authority == MemoryAdmissionAuthority.OPERATOR_REVIEW
        and classification == LegacyMemoryClassification.LEGACY_QUARANTINED
    ):
        raise ValueError("Quarantined legacy memory cannot be admitted without correction")
    return authority


def _evidence(payload: dict[str, Any], *, canonical: bool) -> str:
    raw = payload.get("evidence")
    if not isinstance(raw, list) or len(raw) > 64:
        raise ValueError("Temporal memory evidence must be a bounded list")
    normalized: list[dict[str, str | None]] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"kind", "reference_id", "sha256"}:
            raise ValueError("Temporal memory evidence entries have an invalid shape")
        kind = _required_text(item.get("kind"), field="evidence kind", maximum=32)
        # `agent` names a fact the owner's agent saved deliberately through
        # the internal memory API; that save has no message or run id of its
        # own, so it is cited by the save's id instead.
        if kind not in {"message", "run", "operator", "legacy", "agent"}:
            raise ValueError("Temporal memory evidence kind is invalid")
        reference_id = _required_text(item.get("reference_id"), field="evidence reference", maximum=128)
        digest = item.get("sha256")
        if digest is not None and (not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest)):
            raise ValueError("Temporal memory evidence digest is invalid")
        normalized.append({"kind": kind, "reference_id": reference_id, "sha256": digest})
    if canonical and not normalized:
        raise ValueError("Canonical temporal memory requires evidence")
    return json.dumps(normalized, separators=(",", ":"), sort_keys=True)


def _vector_metadata(payload: dict[str, Any]) -> str:
    raw = payload.get("vector_metadata", {})
    if not isinstance(raw, dict) or any(not isinstance(key, str) or not key for key in raw):
        raise ValueError("Temporal memory vector metadata must be an object with string keys")
    try:
        encoded = json.dumps(raw, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Temporal memory vector metadata is not JSON serializable") from exc
    if len(encoded) > 65536:
        raise ValueError("Temporal memory vector metadata is too large")
    return encoded


def _string_list(value: object, *, field: str, maximum_items: int, maximum_length: int) -> str:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum_items:
        raise ValueError(f"Temporal episode {field} must be a bounded non-empty list")
    normalized = [_required_text(item, field=f"{field} item", maximum=maximum_length) for item in value]
    return json.dumps(normalized, separators=(",", ":"))


async def _require_owner_runtime(
    connection: aiosqlite.Connection,
    *,
    workshop_id: str,
    principal_id: PrincipalId,
    runtime_profile_id: RuntimeProfileId,
) -> None:
    async with connection.execute(
        "SELECT p.kind, EXISTS(SELECT 1 FROM workshop_memberships wm "
        "WHERE wm.workshop_id = ? AND wm.principal_id = p.id), "
        "EXISTS(SELECT 1 FROM runtime_profile_owners rpo "
        "WHERE rpo.runtime_profile_id = ? AND rpo.principal_id = p.id) "
        "FROM principals p WHERE p.id = ?",
        (workshop_id, runtime_profile_id, principal_id),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None or str(row[0]) != "human" or not bool(row[1]) or not bool(row[2]):
        raise ValueError("Temporal memory owner and runtime authority do not match")


def _revision_values(payload: dict[str, Any]) -> tuple[object, ...]:
    expected = {
        "revision_id",
        "content",
        "asserted_at",
        "observed_at",
        "stored_at",
        "valid_from",
        "valid_until",
        "reason",
        "evidence",
        "source_receipt_id",
        "source_run_id",
        "source_message_id",
        "result_message_id",
        "backend",
        "provider",
        "model",
        "prompt_version",
        "schema_version",
        "supersedes_revision_id",
        "migration_classification",
        "migration_gaps",
        "admission_authority",
        "vector_metadata",
    }
    required = expected - {"admission_authority", "vector_metadata"}
    if not required.issubset(payload) or not set(payload).issubset(expected):
        raise ValueError("Temporal fact revision payload has an invalid shape")
    revision_id = MemoryRevisionId(_required_text(payload.get("revision_id"), field="revision_id", maximum=128))
    classification, gaps_json = _migration(payload)
    evidence_json = _evidence(payload, canonical=classification == LegacyMemoryClassification.CANONICAL.value)
    valid_from = _timestamp(payload.get("valid_from"), field="valid_from")
    valid_until = _timestamp(payload.get("valid_until"), field="valid_until")
    if valid_from is not None and valid_until is not None and valid_until <= valid_from:
        raise ValueError("Temporal fact validity interval is invalid")
    supersedes_raw = payload.get("supersedes_revision_id")
    supersedes = (
        None
        if supersedes_raw is None
        else MemoryRevisionId(_required_text(supersedes_raw, field="supersedes_revision_id", maximum=128))
    )
    return (
        revision_id,
        _required_text(payload.get("content"), field="content"),
        _timestamp(payload.get("asserted_at"), field="asserted_at"),
        _timestamp(payload.get("observed_at"), field="observed_at"),
        _timestamp(payload.get("stored_at"), field="stored_at", required=True),
        valid_from,
        valid_until,
        _required_text(payload.get("reason"), field="reason", maximum=2048),
        evidence_json,
        _optional_text(payload.get("source_receipt_id"), field="source_receipt_id", maximum=128),
        _optional_text(payload.get("source_run_id"), field="source_run_id", maximum=128),
        _optional_text(payload.get("source_message_id"), field="source_message_id", maximum=128),
        _optional_text(payload.get("result_message_id"), field="result_message_id", maximum=128),
        _optional_text(payload.get("backend"), field="backend", maximum=64),
        _optional_text(payload.get("provider"), field="provider", maximum=64),
        _optional_text(payload.get("model"), field="model"),
        _optional_text(payload.get("prompt_version"), field="prompt_version", maximum=64),
        _optional_text(payload.get("schema_version"), field="schema_version", maximum=64),
        supersedes,
        classification,
        gaps_json,
        resolve_memory_admission(classification, payload.get("admission_authority")).value,
        _vector_metadata(payload),
    )


async def _insert_revision(
    connection: aiosqlite.Connection,
    *,
    claim_id: MemoryClaimId,
    payload: dict[str, Any],
    event: StoredEvent,
    state: FactLifecycleState,
    transition: FactLifecycleTransition,
) -> MemoryRevisionId:
    values = _revision_values(payload)
    revision_id = MemoryRevisionId(str(values[0]))
    await connection.execute(
        "INSERT INTO memory_fact_revisions ("
        "revision_id, claim_id, content, asserted_at, observed_at, stored_at, valid_from, valid_until, "
        "reason, evidence_json, source_receipt_id, source_run_id, source_message_id, result_message_id, "
        "backend, provider, model, prompt_version, schema_version, supersedes_revision_id, "
        "migration_classification, migration_gaps_json, admission_authority, vector_metadata_json, "
        "created_event_position"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (values[0], claim_id, *values[1:], event.position),
    )
    reason = str(values[7])
    occurred_at = event.envelope.occurred_at.isoformat()
    await connection.execute(
        "INSERT INTO memory_fact_revision_states "
        "(revision_id, claim_id, state, state_reason, state_event_position, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (revision_id, claim_id, state.value, reason, event.position, occurred_at),
    )
    await _record_fact_history(
        connection,
        event=event,
        claim_id=claim_id,
        revision_id=revision_id,
        transition=transition,
        previous_state=None,
        new_state=state,
        reason=reason,
    )
    return revision_id


async def _queue_vector_operation(
    connection: aiosqlite.Connection,
    *,
    event: StoredEvent,
    claim_id: MemoryClaimId,
    revision_id: MemoryRevisionId,
    operation: str,
    prior_revision_id: MemoryRevisionId | None = None,
) -> None:
    if operation not in {"upsert", "replace", "delete"}:
        raise ValueError("Temporal memory vector operation is invalid")
    occurred_at = event.envelope.occurred_at.isoformat()
    await connection.execute(
        "INSERT INTO memory_fact_vector_operations "
        "(event_position, claim_id, revision_id, prior_revision_id, operation, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
        (event.position, claim_id, revision_id, prior_revision_id, operation, occurred_at, occurred_at),
    )


async def _record_fact_history(
    connection: aiosqlite.Connection,
    *,
    event: StoredEvent,
    claim_id: MemoryClaimId,
    revision_id: MemoryRevisionId,
    transition: FactLifecycleTransition,
    previous_state: FactLifecycleState | None,
    new_state: FactLifecycleState,
    reason: str,
) -> None:
    await connection.execute(
        "INSERT INTO memory_fact_lifecycle_events "
        "(event_position, claim_id, revision_id, transition, previous_state, new_state, reason, occurred_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            event.position,
            claim_id,
            revision_id,
            transition.value,
            None if previous_state is None else previous_state.value,
            new_state.value,
            reason,
            event.envelope.occurred_at.isoformat(),
        ),
    )


async def _state(
    connection: aiosqlite.Connection,
    *,
    claim_id: MemoryClaimId,
    revision_id: MemoryRevisionId,
) -> FactLifecycleState:
    async with connection.execute(
        "SELECT state FROM memory_fact_revision_states WHERE claim_id = ? AND revision_id = ?",
        (claim_id, revision_id),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        raise ValueError("Temporal fact revision does not exist in its claim")
    return FactLifecycleState(str(row[0]))


async def _set_state(
    connection: aiosqlite.Connection,
    *,
    event: StoredEvent,
    claim_id: MemoryClaimId,
    revision_id: MemoryRevisionId,
    transition: FactLifecycleTransition,
    expected: FactLifecycleState,
    new_state: FactLifecycleState,
    reason: str,
) -> None:
    current = await _state(connection, claim_id=claim_id, revision_id=revision_id)
    if current != expected:
        raise ValueError("Temporal fact transition has a stale or invalid source state")
    if validate_fact_transition(current, transition) != new_state:
        raise ValueError("Temporal fact transition does not produce the requested state")
    await connection.execute(
        "UPDATE memory_fact_revision_states SET state = ?, state_reason = ?, "
        "state_event_position = ?, updated_at = ? WHERE revision_id = ? AND claim_id = ?",
        (new_state.value, reason, event.position, event.envelope.occurred_at.isoformat(), revision_id, claim_id),
    )
    await _record_fact_history(
        connection,
        event=event,
        claim_id=claim_id,
        revision_id=revision_id,
        transition=transition,
        previous_state=current,
        new_state=new_state,
        reason=reason,
    )


async def _apply_fact_event(connection: aiosqlite.Connection, event: StoredEvent) -> None:
    envelope = event.envelope
    if envelope.aggregate_type != "memory_fact_claim" or envelope.event_version != 1:
        raise ValueError("Temporal fact event requires a version-one memory fact claim aggregate")
    claim_id = MemoryClaimId(str(envelope.aggregate_id))
    payload = envelope.payload
    if envelope.event_type == WorkshopEventType.MEMORY_FACT_RECORDED:
        expected = {
            "owner_principal_id",
            "runtime_profile_id",
            "scope_kind",
            "scope_key",
            "claim_identity_sha256",
            "revision",
        }
        if set(payload) != expected or not isinstance(payload.get("revision"), dict):
            raise ValueError("Temporal fact creation payload has an invalid shape")
        owner = PrincipalId(_required_text(payload.get("owner_principal_id"), field="owner_principal_id", maximum=128))
        runtime = RuntimeProfileId(
            _required_text(payload.get("runtime_profile_id"), field="runtime_profile_id", maximum=128)
        )
        if envelope.actor_principal_id != owner:
            raise ValueError("Temporal fact creation must be attributed to its owner")
        await _require_owner_runtime(
            connection,
            workshop_id=str(envelope.workshop_id),
            principal_id=owner,
            runtime_profile_id=runtime,
        )
        scope_kind, scope_key = _scope(payload)
        digest = _required_text(payload.get("claim_identity_sha256"), field="claim_identity_sha256", maximum=64)
        if not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError("Temporal fact claim identity digest is invalid")
        await connection.execute(
            "INSERT INTO memory_fact_claims "
            "(claim_id, workshop_id, owner_principal_id, runtime_profile_id, scope_kind, scope_key, "
            "claim_identity_sha256, created_at, created_event_position) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                claim_id,
                envelope.workshop_id,
                owner,
                runtime,
                scope_kind,
                scope_key,
                digest,
                envelope.occurred_at.isoformat(),
                event.position,
            ),
        )
        revision_payload = dict(payload["revision"])
        if revision_payload.get("supersedes_revision_id") is not None:
            raise ValueError("An initial temporal fact revision cannot supersede another revision")
        await _insert_revision(
            connection,
            claim_id=claim_id,
            payload=revision_payload,
            event=event,
            state=FactLifecycleState.ACTIVE,
            transition=FactLifecycleTransition.RECORDED,
        )
        await _queue_vector_operation(
            connection,
            event=event,
            claim_id=claim_id,
            revision_id=MemoryRevisionId(str(revision_payload["revision_id"])),
            operation="upsert",
        )
        return

    async with connection.execute(
        "SELECT owner_principal_id FROM memory_fact_claims WHERE claim_id = ? AND workshop_id = ?",
        (claim_id, envelope.workshop_id),
    ) as cursor:
        claim = await cursor.fetchone()
    if claim is None or envelope.actor_principal_id != PrincipalId(str(claim[0])):
        raise ValueError("Temporal fact transition must target an owned claim")

    reason = _required_text(payload.get("reason"), field="reason", maximum=2048)
    if envelope.event_type == WorkshopEventType.MEMORY_FACT_SUPERSEDED:
        expected = {"prior_revision_id", "reason", "revision", "scope_kind", "scope_key"}
        if set(payload) not in (expected, expected - {"scope_kind", "scope_key"}) or not isinstance(
            payload.get("revision"), dict
        ):
            raise ValueError("Temporal fact supersession payload has an invalid shape")
        prior = MemoryRevisionId(
            _required_text(payload.get("prior_revision_id"), field="prior_revision_id", maximum=128)
        )
        revision_payload = dict(payload["revision"])
        if revision_payload.get("supersedes_revision_id") != str(prior):
            raise ValueError("Temporal fact successor must link to the superseded revision")
        await _set_state(
            connection,
            event=event,
            claim_id=claim_id,
            revision_id=prior,
            transition=FactLifecycleTransition.SUPERSEDED,
            expected=FactLifecycleState.ACTIVE,
            new_state=FactLifecycleState.SUPERSEDED,
            reason=reason,
        )
        successor = await _insert_revision(
            connection,
            claim_id=claim_id,
            payload=revision_payload,
            event=event,
            state=FactLifecycleState.ACTIVE,
            transition=FactLifecycleTransition.RECORDED,
        )
        await _queue_vector_operation(
            connection,
            event=event,
            claim_id=claim_id,
            revision_id=successor,
            prior_revision_id=prior,
            operation="replace",
        )
        if "scope_kind" in payload:
            scope_kind, scope_key = _scope(payload)
            await connection.execute(
                "UPDATE memory_fact_claims SET scope_kind = ?, scope_key = ? WHERE claim_id = ?",
                (scope_kind, scope_key, claim_id),
            )
        return

    if envelope.event_type in {WorkshopEventType.MEMORY_FACT_RETRACTED, WorkshopEventType.MEMORY_FACT_EXPIRED}:
        if set(payload) != {"revision_id", "reason"}:
            raise ValueError("Temporal fact terminal transition payload has an invalid shape")
        revision_id = MemoryRevisionId(_required_text(payload.get("revision_id"), field="revision_id", maximum=128))
        transition = (
            FactLifecycleTransition.RETRACTED
            if envelope.event_type == WorkshopEventType.MEMORY_FACT_RETRACTED
            else FactLifecycleTransition.EXPIRED
        )
        await _set_state(
            connection,
            event=event,
            claim_id=claim_id,
            revision_id=revision_id,
            transition=transition,
            expected=FactLifecycleState.ACTIVE,
            new_state=(
                FactLifecycleState.RETRACTED
                if transition == FactLifecycleTransition.RETRACTED
                else FactLifecycleState.EXPIRED
            ),
            reason=reason,
        )
        await _queue_vector_operation(
            connection,
            event=event,
            claim_id=claim_id,
            revision_id=revision_id,
            operation="delete",
        )
        return

    if envelope.event_type == WorkshopEventType.MEMORY_FACT_CONFLICT_OPENED:
        if set(payload) != {"active_revision_id", "reason", "revision"} or not isinstance(
            payload.get("revision"), dict
        ):
            raise ValueError("Temporal fact conflict payload has an invalid shape")
        active = MemoryRevisionId(
            _required_text(payload.get("active_revision_id"), field="active_revision_id", maximum=128)
        )
        await _set_state(
            connection,
            event=event,
            claim_id=claim_id,
            revision_id=active,
            transition=FactLifecycleTransition.CONFLICT_OPENED,
            expected=FactLifecycleState.ACTIVE,
            new_state=FactLifecycleState.UNRESOLVED_CONFLICT,
            reason=reason,
        )
        await _insert_revision(
            connection,
            claim_id=claim_id,
            payload=dict(payload["revision"]),
            event=event,
            state=FactLifecycleState.UNRESOLVED_CONFLICT,
            transition=FactLifecycleTransition.CONFLICT_OPENED,
        )
        await _queue_vector_operation(
            connection,
            event=event,
            claim_id=claim_id,
            revision_id=active,
            operation="delete",
        )
        return

    if envelope.event_type == WorkshopEventType.MEMORY_FACT_CONFLICT_RESOLVED:
        if set(payload) != {"winner_revision_id", "loser_revision_ids", "reason"}:
            raise ValueError("Temporal fact conflict resolution payload has an invalid shape")
        winner = MemoryRevisionId(
            _required_text(payload.get("winner_revision_id"), field="winner_revision_id", maximum=128)
        )
        raw_losers = payload.get("loser_revision_ids")
        if not isinstance(raw_losers, list) or not raw_losers or len(raw_losers) > 32:
            raise ValueError("Temporal fact conflict resolution requires bounded losers")
        losers = tuple(
            MemoryRevisionId(_required_text(item, field="loser revision", maximum=128)) for item in raw_losers
        )
        if len(set(losers)) != len(losers) or winner in losers:
            raise ValueError("Temporal fact conflict resolution revisions are invalid")
        async with connection.execute(
            "SELECT revision_id FROM memory_fact_revision_states WHERE claim_id = ? AND state = 'unresolved_conflict'",
            (claim_id,),
        ) as cursor:
            unresolved = {MemoryRevisionId(str(row[0])) for row in await cursor.fetchall()}
        if unresolved != {winner, *losers}:
            raise ValueError("Temporal fact conflict resolution must settle every unresolved revision")
        for loser in losers:
            current = await _state(connection, claim_id=claim_id, revision_id=loser)
            if current != FactLifecycleState.UNRESOLVED_CONFLICT:
                raise ValueError("Temporal fact conflict loser is not unresolved")
            await connection.execute(
                "UPDATE memory_fact_revision_states SET state = 'superseded', state_reason = ?, "
                "state_event_position = ?, updated_at = ? WHERE revision_id = ? AND claim_id = ?",
                (reason, event.position, envelope.occurred_at.isoformat(), loser, claim_id),
            )
            await _record_fact_history(
                connection,
                event=event,
                claim_id=claim_id,
                revision_id=loser,
                transition=FactLifecycleTransition.CONFLICT_RESOLVED,
                previous_state=current,
                new_state=FactLifecycleState.SUPERSEDED,
                reason=reason,
            )
        await _set_state(
            connection,
            event=event,
            claim_id=claim_id,
            revision_id=winner,
            transition=FactLifecycleTransition.CONFLICT_RESOLVED,
            expected=FactLifecycleState.UNRESOLVED_CONFLICT,
            new_state=FactLifecycleState.ACTIVE,
            reason=reason,
        )
        await _queue_vector_operation(
            connection,
            event=event,
            claim_id=claim_id,
            revision_id=winner,
            operation="upsert",
        )
        return

    if envelope.event_type == WorkshopEventType.MEMORY_FACT_RESTORED:
        expected = {"prior_revision_id", "reason", "revision", "scope_kind", "scope_key"}
        if set(payload) not in (expected, expected - {"scope_kind", "scope_key"}) or not isinstance(
            payload.get("revision"), dict
        ):
            raise ValueError("Temporal fact restoration payload has an invalid shape")
        prior = MemoryRevisionId(
            _required_text(payload.get("prior_revision_id"), field="prior_revision_id", maximum=128)
        )
        prior_state = await _state(connection, claim_id=claim_id, revision_id=prior)
        if prior_state not in {
            FactLifecycleState.SUPERSEDED,
            FactLifecycleState.RETRACTED,
            FactLifecycleState.EXPIRED,
        }:
            raise ValueError("Only an inactive temporal fact revision can be restored")
        async with connection.execute(
            "SELECT COUNT(*) FROM memory_fact_revision_states "
            "WHERE claim_id = ? AND state IN ('active', 'unresolved_conflict')",
            (claim_id,),
        ) as cursor:
            active = await cursor.fetchone()
        if active is None or int(active[0]) != 0:
            raise ValueError("A temporal fact cannot be restored while its claim has current truth")
        revision_payload = dict(payload["revision"])
        if revision_payload.get("supersedes_revision_id") != str(prior):
            raise ValueError("A restored temporal fact must link to its prior revision")
        restored = await _insert_revision(
            connection,
            claim_id=claim_id,
            payload=revision_payload,
            event=event,
            state=FactLifecycleState.ACTIVE,
            transition=FactLifecycleTransition.RECORDED,
        )
        await _queue_vector_operation(
            connection,
            event=event,
            claim_id=claim_id,
            revision_id=restored,
            prior_revision_id=prior,
            operation="replace",
        )
        if "scope_kind" in payload:
            scope_kind, scope_key = _scope(payload)
            await connection.execute(
                "UPDATE memory_fact_claims SET scope_kind = ?, scope_key = ? WHERE claim_id = ?",
                (scope_kind, scope_key, claim_id),
            )
        return
    raise ValueError("Unsupported temporal fact event type")


_EPISODE_BASE_FIELDS = frozenset(
    {
        "owner_principal_id",
        "runtime_profile_id",
        "scope_kind",
        "scope_key",
        "content",
        "occurred_from",
        "occurred_until",
        "observed_at",
        "stored_at",
        "reason",
        "evidence",
        "source_receipt_id",
        "source_run_id",
        "source_message_id",
        "result_message_id",
        "backend",
        "provider",
        "model",
        "prompt_version",
        "schema_version",
        "migration_classification",
        "migration_gaps",
    }
)
_EPISODE_STRUCTURED_FIELDS = frozenset(
    {
        "goal",
        "context",
        "approach",
        "outcome",
        "outcome_quality",
        "lessons",
        "tags",
        "actors",
        "vector_metadata",
        "similarity_fingerprint",
    }
)


def _episode_values(payload: dict[str, Any], *, event_version: int) -> dict[str, object]:
    """
    Validate an episode-recorded payload and return its stored column values.

    This is every check the projection makes that does not need the
    database; ownership of the runtime profile is checked separately by
    the caller. Keeping it pure lets reconciliation ask "would this
    episode be accepted?" before it writes anything, with the exact rules
    the projection enforces rather than a copy that could drift.
    """
    required = _EPISODE_BASE_FIELDS | _EPISODE_STRUCTURED_FIELDS if event_version == 2 else _EPISODE_BASE_FIELDS
    if set(payload) not in {frozenset(required), frozenset(required | {"admission_authority"})}:
        raise ValueError("Temporal episode payload has an invalid shape")
    owner = PrincipalId(_required_text(payload.get("owner_principal_id"), field="owner_principal_id", maximum=128))
    runtime = RuntimeProfileId(
        _required_text(payload.get("runtime_profile_id"), field="runtime_profile_id", maximum=128)
    )
    scope_kind, scope_key = _scope(payload)
    classification, gaps_json = _migration(payload)
    admission_authority = resolve_memory_admission(classification, payload.get("admission_authority")).value
    evidence_json = _evidence(payload, canonical=classification == LegacyMemoryClassification.CANONICAL.value)
    occurred_from = _timestamp(payload.get("occurred_from"), field="occurred_from")
    occurred_until = _timestamp(payload.get("occurred_until"), field="occurred_until")
    if occurred_from is not None and occurred_until is not None and occurred_until < occurred_from:
        raise ValueError("Temporal episode occurrence interval is invalid")
    values: dict[str, object] = {
        "owner_principal_id": owner,
        "runtime_profile_id": runtime,
        "scope_kind": scope_kind,
        "scope_key": scope_key,
        "content": _required_text(payload.get("content"), field="content", maximum=65536),
        "occurred_from": occurred_from,
        "occurred_until": occurred_until,
        "observed_at": _timestamp(payload.get("observed_at"), field="observed_at"),
        "stored_at": _timestamp(payload.get("stored_at"), field="stored_at", required=True),
        "reason": _required_text(payload.get("reason"), field="reason", maximum=2048),
        "evidence_json": evidence_json,
        "source_receipt_id": _optional_text(payload.get("source_receipt_id"), field="source_receipt_id", maximum=128),
        "source_run_id": _optional_text(payload.get("source_run_id"), field="source_run_id", maximum=128),
        "source_message_id": _optional_text(payload.get("source_message_id"), field="source_message_id", maximum=128),
        "result_message_id": _optional_text(payload.get("result_message_id"), field="result_message_id", maximum=128),
        "backend": _optional_text(payload.get("backend"), field="backend", maximum=64),
        "provider": _optional_text(payload.get("provider"), field="provider", maximum=64),
        "model": _optional_text(payload.get("model"), field="model"),
        "prompt_version": _optional_text(payload.get("prompt_version"), field="prompt_version", maximum=64),
        "schema_version": _optional_text(payload.get("schema_version"), field="schema_version", maximum=64),
        "migration_classification": classification,
        "migration_gaps_json": gaps_json,
        "admission_authority": admission_authority,
        "goal": None,
        "context": None,
        "approach": None,
        "outcome": None,
        "outcome_quality": None,
        "lessons": None,
        "tags_json": "[]",
        "actors_json": "[]",
        "vector_metadata_json": "{}",
        "similarity_fingerprint": None,
    }
    if event_version == 2:
        outcome_quality = _required_text(payload.get("outcome_quality"), field="outcome_quality", maximum=16)
        if outcome_quality not in {"success", "partial", "failure"}:
            raise ValueError("Temporal episode outcome_quality is invalid")
        similarity_fingerprint = _required_text(
            payload.get("similarity_fingerprint"), field="similarity_fingerprint", maximum=64
        )
        if not _SHA256_PATTERN.fullmatch(similarity_fingerprint):
            raise ValueError("Temporal episode similarity fingerprint is invalid")
        values.update(
            {
                "goal": _required_text(payload.get("goal"), field="goal", maximum=300),
                "context": _required_text(payload.get("context"), field="context", maximum=500),
                "approach": _required_text(payload.get("approach"), field="approach", maximum=500),
                "outcome": _required_text(payload.get("outcome"), field="outcome", maximum=500),
                "outcome_quality": outcome_quality,
                "lessons": _optional_text(payload.get("lessons"), field="lessons", maximum=500),
                "tags_json": _string_list(payload.get("tags"), field="tags", maximum_items=5, maximum_length=50),
                "actors_json": _string_list(
                    payload.get("actors"), field="actors", maximum_items=10, maximum_length=100
                ),
                "vector_metadata_json": _vector_metadata(payload),
                "similarity_fingerprint": similarity_fingerprint,
            }
        )
    return values


def validate_episode_payload(payload: dict[str, Any]) -> None:
    """
    Raise ValueError when a current-version episode payload would be rejected.

    Reconciliation and triage call this before offering or applying an
    episode, so a record that cannot be stored is caught while nothing
    has been written yet.
    """
    _episode_values(payload, event_version=2)


def validate_fact_revision_payload(payload: dict[str, Any]) -> None:
    """
    Raise ValueError when a fact revision payload would be rejected.

    The same pure checks the projection applies to every created,
    superseding, or restoring revision.
    """
    _revision_values(payload)


async def _apply_episode_event(connection: aiosqlite.Connection, event: StoredEvent) -> None:
    envelope = event.envelope
    if envelope.aggregate_type != "memory_episode" or envelope.event_version not in {1, 2}:
        raise ValueError("Temporal episode event requires a supported memory episode aggregate")
    episode_id = MemoryEpisodeId(str(envelope.aggregate_id))
    payload = envelope.payload
    if envelope.event_type == WorkshopEventType.MEMORY_EPISODE_RECORDED:
        values = _episode_values(payload, event_version=envelope.event_version)
        owner = PrincipalId(str(values["owner_principal_id"]))
        runtime = RuntimeProfileId(str(values["runtime_profile_id"]))
        if envelope.actor_principal_id != owner:
            raise ValueError("Temporal episode creation must be attributed to its owner")
        await _require_owner_runtime(
            connection,
            workshop_id=str(envelope.workshop_id),
            principal_id=owner,
            runtime_profile_id=runtime,
        )
        await connection.execute(
            "INSERT INTO memory_episodes ("
            "episode_id, workshop_id, owner_principal_id, runtime_profile_id, scope_kind, scope_key, "
            "content, occurred_from, occurred_until, observed_at, stored_at, reason, evidence_json, "
            "source_receipt_id, source_run_id, source_message_id, result_message_id, backend, provider, "
            "model, prompt_version, schema_version, migration_classification, migration_gaps_json, "
            "admission_authority, created_event_position, goal, context, approach, outcome, outcome_quality, lessons, tags_json, "
            "actors_json, vector_metadata_json, similarity_fingerprint) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode_id,
                envelope.workshop_id,
                owner,
                runtime,
                values["scope_kind"],
                values["scope_key"],
                values["content"],
                values["occurred_from"],
                values["occurred_until"],
                values["observed_at"],
                values["stored_at"],
                values["reason"],
                values["evidence_json"],
                values["source_receipt_id"],
                values["source_run_id"],
                values["source_message_id"],
                values["result_message_id"],
                values["backend"],
                values["provider"],
                values["model"],
                values["prompt_version"],
                values["schema_version"],
                values["migration_classification"],
                values["migration_gaps_json"],
                values["admission_authority"],
                event.position,
                values["goal"],
                values["context"],
                values["approach"],
                values["outcome"],
                values["outcome_quality"],
                values["lessons"],
                values["tags_json"],
                values["actors_json"],
                values["vector_metadata_json"],
                values["similarity_fingerprint"],
            ),
        )
        if envelope.event_version == 2:
            occurred_at = envelope.occurred_at.isoformat()
            await connection.execute(
                "INSERT INTO memory_episode_vector_operations "
                "(event_position, episode_id, status, created_at, updated_at) "
                "VALUES (?, ?, 'pending', ?, ?)",
                (event.position, episode_id, occurred_at, occurred_at),
            )
        return

    if envelope.event_type != WorkshopEventType.MEMORY_EPISODE_FOLLOWUP_RECORDED:
        raise ValueError("Unsupported temporal episode event type")
    if set(payload) != {"target_episode_id", "relationship", "reason"}:
        raise ValueError("Temporal episode follow-up payload has an invalid shape")
    target = MemoryEpisodeId(_required_text(payload.get("target_episode_id"), field="target_episode_id", maximum=128))
    relationship = EpisodeFollowupRelationship(
        _required_text(payload.get("relationship"), field="relationship", maximum=32)
    )
    async with connection.execute(
        "SELECT workshop_id, owner_principal_id, runtime_profile_id, scope_kind, scope_key "
        "FROM memory_episodes WHERE episode_id = ?",
        (episode_id,),
    ) as cursor:
        source = await cursor.fetchone()
    if source is None or envelope.actor_principal_id != PrincipalId(str(source[1])):
        raise ValueError("Temporal episode follow-up must target an owned source episode")
    async with connection.execute(
        "SELECT workshop_id, owner_principal_id, runtime_profile_id, scope_kind, scope_key "
        "FROM memory_episodes WHERE episode_id = ?",
        (target,),
    ) as cursor:
        target_owner = await cursor.fetchone()
    if target_owner is None or tuple(target_owner) != tuple(source):
        raise ValueError("Temporal episode follow-up cannot cross owner, runtime, or scope authority")
    await connection.execute(
        "INSERT INTO memory_episode_followups "
        "(source_episode_id, target_episode_id, workshop_id, owner_principal_id, runtime_profile_id, "
        "scope_kind, scope_key, relationship, reason, created_at, created_event_position) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            episode_id,
            target,
            *source,
            relationship.value,
            _required_text(payload.get("reason"), field="reason", maximum=2048),
            envelope.occurred_at.isoformat(),
            event.position,
        ),
    )


TEMPORAL_MEMORY_EVENT_TYPES = frozenset(
    {
        WorkshopEventType.MEMORY_FACT_RECORDED,
        WorkshopEventType.MEMORY_FACT_SUPERSEDED,
        WorkshopEventType.MEMORY_FACT_RETRACTED,
        WorkshopEventType.MEMORY_FACT_EXPIRED,
        WorkshopEventType.MEMORY_FACT_CONFLICT_OPENED,
        WorkshopEventType.MEMORY_FACT_CONFLICT_RESOLVED,
        WorkshopEventType.MEMORY_FACT_RESTORED,
        WorkshopEventType.MEMORY_EPISODE_RECORDED,
        WorkshopEventType.MEMORY_EPISODE_FOLLOWUP_RECORDED,
    }
)


async def apply_temporal_memory_event(connection: aiosqlite.Connection, event: StoredEvent) -> None:
    """Project one validated temporal-memory event without touching vector storage."""
    if event.envelope.event_type not in TEMPORAL_MEMORY_EVENT_TYPES:
        raise ValueError("Unsupported temporal memory event")
    if str(event.envelope.event_type).startswith("memory_fact."):
        await _apply_fact_event(connection, event)
    else:
        await _apply_episode_event(connection, event)
