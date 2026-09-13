"""Bounded backend context assembled from canonical Workshop messages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from kai.prompt_utils import render_untrusted_json_block
from kai.workshop.domain import MessageId
from kai.workshop.run_lifecycle import DurableRun
from kai.workshop.store import WorkshopEventStore

_MAX_CONTEXT_MESSAGES = 50
_MAX_CONTEXT_CHARACTERS = 24_000
_MAX_MESSAGE_CHARACTERS = 6_000
_MAX_CONTEXT_AGE = timedelta(days=30)


@dataclass(frozen=True, slots=True)
class CanonicalConversationContext:
    text: str
    message_count: int
    through_event_position: int
    from_event_position: int = 0
    eligible_message_count: int = 0
    omitted_message_count: int = 0
    truncated: bool = False
    revision: str = ""
    records: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class CanonicalConversationDelivery:
    """Fresh-session snapshot and live-session delta for one run."""

    snapshot: CanonicalConversationContext
    delta: CanonicalConversationContext
    cursor_event_position: int


@dataclass(frozen=True, slots=True)
class ConversationObservationSettlement:
    channel_id: str
    agent_id: str
    runtime_profile_id: str
    run_id: str
    inbound_message_id: str


@dataclass(frozen=True, slots=True)
class ConversationObservationSettlementResult:
    changed: bool
    observed_through_event_position: int


def _bounded_body(body: str) -> tuple[str, bool]:
    if len(body) <= _MAX_MESSAGE_CHARACTERS:
        return body, False
    return body[:_MAX_MESSAGE_CHARACTERS], True


def render_canonical_message_data(
    rows: Sequence[Any],
    *,
    mode: str,
    scope_kind: str,
    scope_id: str,
    after_event_position: int,
    before_event_position: int,
    eligible_message_count: int | None = None,
    age_omitted_message_count: int = 0,
) -> CanonicalConversationContext:
    """Render typed canonical messages inside a randomized untrusted envelope."""
    records: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    characters = 0
    body_truncated = False
    # Callers supply newest-first rows. Keep the newest messages when the
    # character bound is tighter than the message-count bound, then restore
    # chronological order for delivery.
    for row in rows:
        body, truncated = _bounded_body(str(row[4]))
        relation = (
            "thread_reply" if row[7] is not None else "thread_root" if scope_kind == "thread" else "channel_message"
        )
        record: dict[str, object] = {
            "record_type": "canonical_message",
            "message_id": str(row[0]),
            "author_kind": str(row[2]),
            "author_display_name": str(row[3]).strip() or str(row[2]).title(),
            "thread_relation": relation,
            "reply_to_message_id": None if row[6] is None else str(row[6]),
            "thread_root_message_id": None if row[7] is None else str(row[7]),
            "event_position": int(str(row[1])),
            "body": body,
            "body_truncated": truncated,
            "original_character_count": len(str(row[4])),
        }
        encoded_size = len(json.dumps(record, ensure_ascii=False, sort_keys=True))
        if selected and characters + encoded_size > _MAX_CONTEXT_CHARACTERS:
            break
        selected.append(record)
        characters += encoded_size
        body_truncated = body_truncated or truncated
    selected.reverse()
    eligible = len(rows) if eligible_message_count is None else eligible_message_count
    omitted = max(0, eligible - len(selected))
    reasons: list[str] = []
    if age_omitted_message_count:
        reasons.append("age_limit")
    if omitted > age_omitted_message_count:
        reasons.append("message_or_character_limit")
    if body_truncated:
        reasons.append("message_body_limit")
    metadata: dict[str, object] = {
        "record_type": "canonical_conversation_window",
        "mode": mode,
        "scope_kind": scope_kind,
        "scope_id": scope_id,
        "after_event_position": after_event_position,
        "before_event_position": before_event_position,
        "selected_message_count": len(selected),
        "eligible_message_count": eligible,
        "omitted_message_count": omitted,
        "age_omitted_message_count": age_omitted_message_count,
        "truncated": bool(reasons),
        "truncation_reasons": reasons,
        "limits": {
            "max_messages": _MAX_CONTEXT_MESSAGES,
            "max_characters": _MAX_CONTEXT_CHARACTERS,
            "max_message_characters": _MAX_MESSAGE_CHARACTERS,
            "max_age_seconds": int(_MAX_CONTEXT_AGE.total_seconds()),
        },
    }
    records = [metadata, *selected]
    canonical = "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
    revision = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    text = render_untrusted_json_block("CANONICAL CONVERSATION DATA", records)
    through = int(str(selected[-1]["event_position"])) if selected else after_event_position
    start = int(str(selected[0]["event_position"])) if selected else 0
    return CanonicalConversationContext(
        text=text,
        message_count=len(selected),
        through_event_position=through,
        from_event_position=start,
        eligible_message_count=eligible,
        omitted_message_count=omitted,
        truncated=bool(reasons),
        revision=revision,
        records=tuple(records),
    )


def with_canonical_transcript_pointer(
    context: CanonicalConversationContext,
    transcript_path: object,
) -> CanonicalConversationContext:
    """Add the derived full-history pointer inside the untrusted envelope."""
    transcript_record: dict[str, object] = {
        "record_type": "canonical_transcript_pointer",
        "format": "canonical_jsonl_v1",
        "path": str(transcript_path),
        "authority": "derived_from_canonical_messages",
    }
    records = (
        *context.records,
        transcript_record,
    )
    canonical = "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
    return CanonicalConversationContext(
        text=render_untrusted_json_block("CANONICAL CONVERSATION DATA", list(records)),
        message_count=context.message_count,
        through_event_position=context.through_event_position,
        from_event_position=context.from_event_position,
        eligible_message_count=context.eligible_message_count,
        omitted_message_count=context.omitted_message_count,
        truncated=context.truncated,
        revision=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        records=records,
    )


async def _scope(store: WorkshopEventStore, run: DurableRun) -> tuple[int, str, str, str]:
    async with store.connection.execute(
        "SELECT created_event_position, created_at, thread_root_id FROM messages WHERE id = ? AND channel_id = ?",
        (run.inbound_message_id, run.channel_id),
    ) as cursor:
        inbound = await cursor.fetchone()
    if inbound is None:
        raise RuntimeError("Canonical inbound message no longer exists")
    inbound_position = int(inbound[0])
    if run.kind.value == "observe":
        if run.observed_from_event_position is None or run.observation_scope_kind is None:
            raise RuntimeError("Standing observation context is missing its immutable boundary")
        inbound_position = run.observed_from_event_position
        if run.observation_scope_kind == "channel":
            return inbound_position, str(inbound[1]), "channel", str(run.channel_id)
        if run.observation_scope_kind == "thread" and run.observation_scope_id is not None:
            return inbound_position, str(inbound[1]), "thread", str(run.observation_scope_id)
        raise RuntimeError("Standing observation context has an invalid scope")
    thread_root_id = None if inbound[2] is None else str(inbound[2])
    if thread_root_id is None:
        return inbound_position, str(inbound[1]), "channel", str(run.channel_id)
    return inbound_position, str(inbound[1]), "thread", thread_root_id


async def _cursor_position(store: WorkshopEventStore, run: DurableRun, scope_kind: str, scope_id: str) -> int:
    if run.runtime_profile_id is None:
        return 0
    async with store.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'channel_agent_conversation_observations'"
    ) as cursor:
        if await cursor.fetchone() is None:
            return 0
    async with store.connection.execute(
        "SELECT observed_through_event_position FROM channel_agent_conversation_observations "
        "WHERE channel_id = ? AND agent_id = ? AND runtime_profile_id = ? AND scope_kind = ? AND scope_id = ?",
        (run.channel_id, run.agent_id, run.runtime_profile_id, scope_kind, scope_id),
    ) as cursor:
        row = await cursor.fetchone()
    ordinary = int(row[0]) if row is not None else 0
    async with store.connection.execute(
        "SELECT MAX(delivered_through_event_position) FROM channel_agent_observation_states "
        "WHERE channel_id = ? AND agent_id = ? AND scope_kind = ? AND scope_id = ?",
        (run.channel_id, run.agent_id, scope_kind, scope_id),
    ) as cursor:
        standing = await cursor.fetchone()
    return max(ordinary, int(standing[0]) if standing is not None and standing[0] is not None else 0)


async def _assemble_window(
    store: WorkshopEventStore,
    run: DurableRun,
    *,
    mode: str,
    after_event_position: int,
) -> CanonicalConversationContext:
    inbound_position, inbound_at, scope_kind, scope_id = await _scope(store, run)
    scope_clause = "m.thread_root_id IS NULL"
    scope_parameters: list[object] = []
    if scope_kind == "thread":
        scope_clause = "(m.id = ? OR m.thread_root_id = ?)"
        scope_parameters.extend((scope_id, scope_id))
    author_clause = ""
    author_parameters: list[object] = []
    if mode == "delta":
        author_clause = " AND m.author_principal_id != (SELECT principal_id FROM agents WHERE id = ?)"
        author_parameters.append(run.agent_id)
    cutoff = (datetime.fromisoformat(inbound_at).astimezone(UTC) - _MAX_CONTEXT_AGE).isoformat()
    range_parameters: list[object] = [
        run.channel_id,
        *scope_parameters,
        after_event_position,
        inbound_position,
        *author_parameters,
    ]
    async with store.connection.execute(
        "SELECT COUNT(*) FROM messages m WHERE m.channel_id = ? AND "
        + scope_clause
        + " AND m.created_event_position > ? AND m.created_event_position < ?"
        + author_clause,
        tuple(range_parameters),
    ) as cursor:
        count_row = await cursor.fetchone()
    eligible = int(count_row[0]) if count_row is not None else 0
    async with store.connection.execute(
        "SELECT COUNT(*) FROM messages m WHERE m.channel_id = ? AND "
        + scope_clause
        + " AND m.created_event_position > ? AND m.created_event_position < ?"
        + author_clause
        + " AND m.created_at >= ?",
        tuple([*range_parameters, cutoff]),
    ) as cursor:
        recent_count_row = await cursor.fetchone()
    recent_count = int(recent_count_row[0]) if recent_count_row is not None else 0
    parameters = [*range_parameters, cutoff, _MAX_CONTEXT_MESSAGES]
    async with store.connection.execute(
        "SELECT m.id, m.created_event_position, p.kind, p.display_name, m.body, "
        "m.author_principal_id, m.reply_to_message_id, m.thread_root_id "
        "FROM messages m JOIN principals p ON p.id = m.author_principal_id "
        "WHERE m.channel_id = ? AND "
        + scope_clause
        + " AND m.created_event_position > ? AND m.created_event_position < ?"
        + author_clause
        + " AND m.created_at >= ? "
        "ORDER BY m.created_event_position DESC, m.id DESC LIMIT ?",
        tuple(parameters),
    ) as cursor:
        rows = list(await cursor.fetchall())
    return render_canonical_message_data(
        rows,
        mode=mode,
        scope_kind=scope_kind,
        scope_id=scope_id,
        after_event_position=after_event_position,
        before_event_position=inbound_position,
        eligible_message_count=eligible,
        age_omitted_message_count=max(0, eligible - recent_count),
    )


async def assemble_canonical_conversation_context(
    store: WorkshopEventStore,
    run: DurableRun,
) -> CanonicalConversationContext:
    """Return a structured fresh-session snapshot strictly before the prompt."""
    if not isinstance(run.inbound_message_id, MessageId):
        raise TypeError("run must identify a typed inbound message")
    return await _assemble_window(store, run, mode="snapshot", after_event_position=0)


async def assemble_canonical_conversation_delivery(
    store: WorkshopEventStore,
    run: DurableRun,
) -> CanonicalConversationDelivery:
    """Return the fresh snapshot and live delta for one protected dispatch."""
    snapshot = await _assemble_window(store, run, mode="snapshot", after_event_position=0)
    _, _, scope_kind, scope_id = await _scope(store, run)
    cursor_position = await _cursor_position(store, run, scope_kind, scope_id)
    delta = await _assemble_window(store, run, mode="delta", after_event_position=cursor_position)
    return CanonicalConversationDelivery(snapshot, delta, cursor_position)


async def settle_conversation_observation_in_transaction(
    store: WorkshopEventStore,
    settlement: ConversationObservationSettlement,
    *,
    occurred_at: datetime,
) -> ConversationObservationSettlementResult:
    """Advance ordinary and matching standing cursors after successful dispatch."""
    async with store.connection.execute(
        "SELECT created_event_position, thread_root_id FROM messages WHERE id = ? AND channel_id = ?",
        (settlement.inbound_message_id, settlement.channel_id),
    ) as cursor:
        source = await cursor.fetchone()
    if source is None:
        raise RuntimeError("Conversation observation source message is missing")
    through = int(source[0])
    scope_kind = "thread" if source[1] is not None else "channel"
    scope_id = str(source[1]) if source[1] is not None else settlement.channel_id
    async with store.connection.execute(
        "SELECT observed_through_event_position, last_run_id FROM channel_agent_conversation_observations "
        "WHERE channel_id = ? AND agent_id = ? AND runtime_profile_id = ? AND scope_kind = ? AND scope_id = ?",
        (settlement.channel_id, settlement.agent_id, settlement.runtime_profile_id, scope_kind, scope_id),
    ) as cursor:
        prior = await cursor.fetchone()
    if prior is not None and int(prior[0]) >= through:
        return ConversationObservationSettlementResult(False, int(prior[0]))
    now = occurred_at.astimezone(UTC).isoformat()
    await store.connection.execute(
        "INSERT INTO channel_agent_conversation_observations (channel_id, agent_id, runtime_profile_id, "
        "scope_kind, scope_id, observed_through_event_position, last_run_id, last_inbound_message_id, "
        "state_version, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?) "
        "ON CONFLICT(channel_id, agent_id, runtime_profile_id, scope_id) DO UPDATE SET "
        "observed_through_event_position = excluded.observed_through_event_position, "
        "last_run_id = excluded.last_run_id, last_inbound_message_id = excluded.last_inbound_message_id, "
        "state_version = channel_agent_conversation_observations.state_version + 1, updated_at = excluded.updated_at",
        (
            settlement.channel_id,
            settlement.agent_id,
            settlement.runtime_profile_id,
            scope_kind,
            scope_id,
            through,
            settlement.run_id,
            settlement.inbound_message_id,
            now,
            now,
        ),
    )
    async with store.connection.execute(
        "SELECT delivered_through_event_position, pending_through_event_position, "
        "considered_through_event_position, not_before, lifecycle_state "
        "FROM channel_agent_observation_states WHERE channel_id = ? AND agent_id = ? "
        "AND scope_kind = ? AND scope_id = ?",
        (settlement.channel_id, settlement.agent_id, scope_kind, scope_id),
    ) as cursor:
        standing = await cursor.fetchone()
    if standing is not None and str(standing[4]) != "paused_overflow" and int(standing[0]) < through:
        pending_through = max(through, int(standing[1]))
        scope_clause = "m.thread_root_id IS NULL" if scope_kind == "channel" else "m.thread_root_id = ?"
        parameters: list[object] = [settlement.agent_id, settlement.channel_id]
        if scope_kind == "thread":
            parameters.append(scope_id)
        parameters.extend((through, pending_through))
        async with store.connection.execute(
            "SELECT MIN(m.created_event_position), COUNT(*) FROM messages m "
            "WHERE m.author_principal_id != (SELECT principal_id FROM agents WHERE id = ?) "
            "AND m.channel_id = ? AND "
            + scope_clause
            + " AND m.created_event_position > ? AND m.created_event_position <= ?",
            tuple(parameters),
        ) as cursor:
            remaining = await cursor.fetchone()
        remaining_count = int(remaining[1]) if remaining is not None else 0
        oldest = int(remaining[0]) if remaining is not None and remaining[0] is not None else None
        await store.connection.execute(
            "UPDATE channel_agent_observation_states SET delivered_through_event_position = ?, "
            "pending_through_event_position = ?, considered_through_event_position = MAX(?, ?), "
            "oldest_pending_event_position = ?, pending_message_count = ?, "
            "not_before = CASE WHEN ? = 0 THEN NULL ELSE not_before END, "
            "lifecycle_state = CASE WHEN ? = 0 THEN 'idle' ELSE 'pending' END, "
            "state_version = state_version + 1 WHERE channel_id = ? AND agent_id = ? "
            "AND scope_kind = ? AND scope_id = ?",
            (
                through,
                pending_through,
                int(standing[2]),
                through,
                oldest,
                remaining_count,
                remaining_count,
                remaining_count,
                settlement.channel_id,
                settlement.agent_id,
                scope_kind,
                scope_id,
            ),
        )
    return ConversationObservationSettlementResult(True, through)


async def conversation_observation_available(store: WorkshopEventStore) -> bool:
    async with store.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'channel_agent_conversation_observations'"
    ) as cursor:
        return await cursor.fetchone() is not None


async def assemble_canonical_prior_pairs(
    store: WorkshopEventStore,
    run: DurableRun,
    *,
    limit: int,
) -> tuple[tuple[str, str], ...]:
    """Return completed exchanges before ``run`` in its exact owner lane.

    Pairing follows durable run lineage rather than adjacent channel messages.
    That prevents another human, another agent, notifications, failures, or an
    interleaved run from becoming semantic-memory episode context.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if limit == 0:
        return ()
    async with store.connection.execute(
        "SELECT created_event_position FROM messages WHERE id = ? AND channel_id = ?",
        (run.inbound_message_id, run.channel_id),
    ) as cursor:
        inbound = await cursor.fetchone()
    if inbound is None:
        raise RuntimeError("Canonical inbound message no longer exists")

    async with store.connection.execute(
        "SELECT source.body, result.body FROM runs prior "
        "JOIN messages source ON source.id = prior.inbound_message_id "
        "JOIN messages result ON result.id = prior.result_message_id "
        "WHERE prior.channel_id = ? AND prior.agent_id = ? "
        "AND prior.requested_by_principal_id = ? AND prior.status = 'completed' "
        "AND source.created_event_position < ? "
        "ORDER BY source.created_event_position DESC, prior.id DESC LIMIT ?",
        (
            run.channel_id,
            run.agent_id,
            run.requested_by_principal_id,
            int(inbound[0]),
            limit,
        ),
    ) as cursor:
        rows = list(await cursor.fetchall())
    return tuple((str(row[0]), str(row[1])) for row in reversed(rows))
