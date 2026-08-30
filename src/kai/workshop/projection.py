"""Canonical collaboration projection for Workshop events."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from kai.workshop.agent_definitions import (
    MAX_AGENT_DESCRIPTION,
    MAX_AGENT_DISPLAY_NAME,
    MAX_AGENT_INSTRUCTIONS,
    MAX_AGENT_PURPOSE,
    normalize_agent_handle,
    validate_agent_capabilities,
    validate_agent_presentation,
    validate_agent_text,
)
from kai.workshop.domain import (
    AgentDefinitionId,
    AgentDefinitionRevisionId,
    AgentDelegationId,
    AgentEnablementId,
    AgentId,
    ChannelId,
    MessageId,
    PrincipalId,
    RunAttemptId,
    RunExecutionOwnerId,
    RunId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.store import StoredEvent

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_MEDIA_TYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_RUN_TERMINAL_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_EXECUTION_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_MODEL_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Workshop event payload requires non-empty {key!r}")
    return value


def _parse_projection_timestamp(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _require_exact_payload(payload: dict[str, Any], keys: set[str]) -> None:
    if set(payload) != keys:
        raise ValueError(f"Workshop event payload must contain exactly {sorted(keys)!r}")


async def _table_has_column(
    connection: aiosqlite.Connection,
    table: str,
    column: str,
) -> bool:
    """Support historical-schema qualification without weakening current replay."""
    async with connection.execute(f"PRAGMA table_info({table})") as cursor:
        return any(str(row[1]) == column for row in await cursor.fetchall())


async def _message_mentions_json(
    connection: aiosqlite.Connection,
    channel_id: ChannelId,
    body: str,
    raw_mentions: object,
) -> str:
    if not isinstance(raw_mentions, list):
        raise ValueError("Workshop message mentions must be a list")
    if not raw_mentions:
        return "[]"
    async with connection.execute(
        "SELECT p.id, p.kind, COALESCE(ad.handle, p.display_name) "
        "FROM channel_memberships cm "
        "JOIN principals p ON p.id = cm.principal_id "
        "LEFT JOIN agents a ON a.principal_id = p.id "
        "LEFT JOIN agent_definitions ad ON ad.agent_id = a.id "
        "WHERE cm.channel_id = ?",
        (channel_id,),
    ) as cursor:
        members = {PrincipalId(str(row[0])): (str(row[1]), str(row[2])) for row in await cursor.fetchall()}
    normalized: list[dict[str, object]] = []
    previous_end = 0
    for raw in raw_mentions:
        if not isinstance(raw, dict) or set(raw) != {"principal_id", "kind", "start", "length"}:
            raise ValueError("Workshop message mention must have the canonical fields")
        principal_id = PrincipalId(_required_text(raw, "principal_id"))
        kind = _required_text(raw, "kind")
        start = raw.get("start")
        length = raw.get("length")
        if kind not in {"human", "agent"}:
            raise ValueError("Workshop message mention kind must be human or agent")
        if not isinstance(start, int) or isinstance(start, bool) or start < previous_end:
            raise ValueError("Workshop message mentions must be ordered and non-overlapping")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 1:
            raise ValueError("Workshop message mention length is invalid")
        end = start + length
        if end > len(body) or body[start : start + 1] != "@":
            raise ValueError("Workshop message mention span is outside the body")
        member = members.get(principal_id)
        if member is None or member[0] != kind:
            raise ValueError("Workshop message mention must resolve to a channel member")
        if body[start + 1 : end].casefold() != member[1].casefold():
            raise ValueError("Workshop message mention span must match the member display name")
        normalized.append(
            {
                "kind": kind,
                "length": length,
                "principal_id": principal_id,
                "start": start,
            }
        )
        previous_end = end
    return json.dumps(normalized, separators=(",", ":"), sort_keys=True)


async def _apply_run_event(connection: aiosqlite.Connection, event: StoredEvent) -> None:
    envelope = event.envelope
    if not isinstance(envelope.aggregate_id, RunId) or envelope.aggregate_type != "run":
        raise ValueError("Workshop run events require a typed run aggregate")

    occurred_at = envelope.occurred_at.isoformat()
    payload = envelope.payload
    if envelope.event_type == WorkshopEventType.RUN_ACCEPTED:
        if envelope.event_version not in {1, 2, 3, 4}:
            raise ValueError("Unsupported Workshop run acceptance event version")
        keys = {"inbound_message_id", "channel_id", "requested_by_principal_id", "agent_id"}
        if envelope.event_version in {2, 3, 4}:
            keys.add("agent_definition_revision_id")
        if envelope.event_version in {3, 4}:
            keys.update({"runtime_profile_id", "sponsor_principal_id"})
        if envelope.event_version == 4:
            keys.update({"parent_run_id", "delegation_id"})
        _require_exact_payload(
            payload,
            keys,
        )
        inbound_message_id = MessageId(_required_text(payload, "inbound_message_id"))
        channel_id = ChannelId(_required_text(payload, "channel_id"))
        requested_by = PrincipalId(_required_text(payload, "requested_by_principal_id"))
        agent_id = AgentId(_required_text(payload, "agent_id"))
        revision_id = (
            AgentDefinitionRevisionId(_required_text(payload, "agent_definition_revision_id"))
            if envelope.event_version in {2, 3, 4}
            else None
        )
        runtime_profile_id = _required_text(payload, "runtime_profile_id") if envelope.event_version in {3, 4} else None
        sponsor_principal_id = (
            PrincipalId(_required_text(payload, "sponsor_principal_id")) if envelope.event_version in {3, 4} else None
        )
        parent_run_id = RunId(_required_text(payload, "parent_run_id")) if envelope.event_version == 4 else None
        delegation_id = (
            AgentDelegationId(_required_text(payload, "delegation_id")) if envelope.event_version == 4 else None
        )
        has_attachment_lifecycle = await _table_has_column(
            connection,
            "channel_agents",
            "detached_at",
        )
        active_attachment_clause = " AND ca.detached_at IS NULL" if has_attachment_lifecycle else ""
        if envelope.event_version == 4:
            async with connection.execute(
                "SELECT c.workshop_id, m.channel_id, parent.requested_by_principal_id, "
                "target.agent_id, m.created_at, caller.principal_id "
                "FROM messages m "
                "JOIN principals author ON author.id = m.author_principal_id AND author.kind = 'agent' "
                "JOIN agents caller ON caller.principal_id = author.id "
                "JOIN runs parent ON parent.id = ? AND parent.agent_id = caller.id "
                "AND parent.channel_id = m.channel_id "
                "JOIN channels c ON c.id = m.channel_id AND c.kind = 'group' "
                "JOIN channel_agents target ON target.channel_id = m.channel_id AND target.agent_id = ?"
                + active_attachment_clause.replace("ca.", "target.")
                + " "
                "WHERE m.id = ?",
                (parent_run_id, agent_id, inbound_message_id),
            ) as cursor:
                rows = list(await cursor.fetchall())
        else:
            async with connection.execute(
                "SELECT c.workshop_id, m.channel_id, m.author_principal_id, ca.agent_id, m.created_at "
                "FROM messages m "
                "JOIN principals p ON p.id = m.author_principal_id AND p.kind = 'human' "
                "JOIN channels c ON c.id = m.channel_id "
                "JOIN channel_memberships cm ON cm.channel_id = m.channel_id "
                "AND cm.principal_id = m.author_principal_id "
                "JOIN channel_agents ca ON ca.channel_id = m.channel_id AND ca.agent_id = ?"
                + active_attachment_clause
                + " "
                "JOIN agents a ON a.id = ca.agent_id AND a.workshop_id = c.workshop_id "
                "WHERE m.id = ?",
                (agent_id, inbound_message_id),
            ) as cursor:
                rows = list(await cursor.fetchall())
        expected = (envelope.workshop_id, channel_id, requested_by, agent_id)
        if len(rows) != 1 or tuple(rows[0][:4]) != expected:
            raise ValueError("Workshop run must match one human message and attached channel agent")
        inbound_created_at = _parse_projection_timestamp(rows[0][4])
        if envelope.occurred_at < inbound_created_at:
            raise ValueError("Workshop run cannot be accepted before its inbound message")
        expected_actor = PrincipalId(str(rows[0][5])) if envelope.event_version == 4 and rows else requested_by
        if envelope.actor_principal_id != expected_actor:
            raise ValueError("Workshop run acceptance actor does not match its canonical requester")
        if revision_id is not None:
            async with connection.execute(
                "SELECT d.agent_id, d.lifecycle_state, d.active_revision_id "
                "FROM agent_definition_revisions r "
                "JOIN agent_definitions d ON d.id = r.agent_definition_id "
                "WHERE r.id = ?",
                (revision_id,),
            ) as cursor:
                definition_row = await cursor.fetchone()
            if definition_row is None or tuple(definition_row) != (agent_id, "active", revision_id):
                raise ValueError("Workshop run must bind the agent's active definition revision")
        if envelope.event_version in {3, 4}:
            async with connection.execute(
                "SELECT COALESCE(ca.sponsor_principal_id, CASE WHEN c.kind = 'direct' "
                "THEN ? ELSE NULL END), ca.sponsored_runtime_profile_id, "
                "ra.runtime_profile_id FROM channel_agents ca "
                "JOIN channels c ON c.id = ca.channel_id "
                "JOIN channel_agent_runtime_assignments ra ON ra.channel_id = ca.channel_id "
                "AND ra.agent_id = ca.agent_id WHERE ca.channel_id = ? AND ca.agent_id = ? "
                "AND ca.detached_at IS NULL",
                (requested_by, channel_id, agent_id),
            ) as cursor:
                sponsorship_row = await cursor.fetchone()
            if sponsorship_row is None or tuple(str(value) for value in sponsorship_row) != (
                str(sponsor_principal_id),
                runtime_profile_id,
                runtime_profile_id,
            ):
                raise ValueError("Workshop run must snapshot its active runtime sponsorship")
        if envelope.event_version == 1:
            await connection.execute(
                "INSERT INTO runs "
                "(id, workshop_id, channel_id, requested_by_principal_id, agent_id, "
                "inbound_message_id, status, accepted_at, last_event_position) "
                "VALUES (?, ?, ?, ?, ?, ?, 'accepted', ?, ?)",
                (
                    envelope.aggregate_id,
                    envelope.workshop_id,
                    channel_id,
                    requested_by,
                    agent_id,
                    inbound_message_id,
                    occurred_at,
                    event.position,
                ),
            )
        elif envelope.event_version == 2:
            await connection.execute(
                "INSERT INTO runs "
                "(id, workshop_id, channel_id, requested_by_principal_id, agent_id, "
                "inbound_message_id, agent_definition_revision_id, status, accepted_at, "
                "last_event_position) VALUES (?, ?, ?, ?, ?, ?, ?, 'accepted', ?, ?)",
                (
                    envelope.aggregate_id,
                    envelope.workshop_id,
                    channel_id,
                    requested_by,
                    agent_id,
                    inbound_message_id,
                    revision_id,
                    occurred_at,
                    event.position,
                ),
            )
        elif envelope.event_version == 3:
            await connection.execute(
                "INSERT INTO runs "
                "(id, workshop_id, channel_id, requested_by_principal_id, agent_id, "
                "inbound_message_id, agent_definition_revision_id, runtime_profile_id, "
                "sponsor_principal_id, status, accepted_at, last_event_position) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted', ?, ?)",
                (
                    envelope.aggregate_id,
                    envelope.workshop_id,
                    channel_id,
                    requested_by,
                    agent_id,
                    inbound_message_id,
                    revision_id,
                    runtime_profile_id,
                    sponsor_principal_id,
                    occurred_at,
                    event.position,
                ),
            )
        else:
            await connection.execute(
                "INSERT INTO runs "
                "(id, workshop_id, channel_id, requested_by_principal_id, agent_id, "
                "inbound_message_id, agent_definition_revision_id, runtime_profile_id, "
                "sponsor_principal_id, parent_run_id, delegation_id, status, accepted_at, "
                "last_event_position) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted', ?, ?)",
                (
                    envelope.aggregate_id,
                    envelope.workshop_id,
                    channel_id,
                    requested_by,
                    agent_id,
                    inbound_message_id,
                    revision_id,
                    runtime_profile_id,
                    sponsor_principal_id,
                    parent_run_id,
                    delegation_id,
                    occurred_at,
                    event.position,
                ),
            )
        return

    async with connection.execute(
        "SELECT r.workshop_id, r.status, r.requested_by_principal_id, a.principal_id, "
        "r.accepted_at, r.started_at, r.inbound_message_id, "
        "r.cancellation_requested_at, r.cancellation_code, r.channel_id "
        "FROM runs r JOIN agents a ON a.id = r.agent_id WHERE r.id = ?",
        (envelope.aggregate_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None or WorkshopId(str(row[0])) != envelope.workshop_id:
        raise ValueError("Workshop run transition references an unknown run")
    status = str(row[1])
    requested_by = PrincipalId(str(row[2]))
    agent_principal = PrincipalId(str(row[3]))
    accepted_at = _parse_projection_timestamp(row[4])
    started_at = None if row[5] is None else _parse_projection_timestamp(row[5])
    inbound_message_id = MessageId(str(row[6]))
    cancellation_requested_at = row[7]

    if envelope.event_type == WorkshopEventType.RUN_CANCELLATION_REQUESTED:
        _require_exact_payload(payload, {"cancellation_code"})
        cancellation_code = _required_text(payload, "cancellation_code")
        if not _RUN_TERMINAL_CODE_PATTERN.fullmatch(cancellation_code):
            raise ValueError("Workshop run cancellation_code must be a lowercase identifier")
        if status not in {"accepted", "started"} or envelope.actor_principal_id != requested_by:
            raise ValueError("Workshop cancellation can be requested only for a nonterminal run by its human")
        lifecycle_floor = started_at if started_at is not None else accepted_at
        if envelope.occurred_at < lifecycle_floor:
            raise ValueError("Workshop cancellation request cannot precede current run state")
        if cancellation_requested_at is not None:
            raise ValueError("Workshop run already has a cancellation request")
        await connection.execute(
            "UPDATE runs SET cancellation_requested_at = ?, cancellation_code = ?, "
            "last_event_position = ? WHERE id = ?",
            (occurred_at, cancellation_code, event.position, envelope.aggregate_id),
        )
        return

    if envelope.event_type == WorkshopEventType.RUN_STARTED:
        if envelope.event_version == 1:
            _require_exact_payload(payload, set())
        elif envelope.event_version == 2:
            _require_exact_payload(payload, {"attempt_id"})
            attempt_transition_at = await _require_run_attempt(
                connection,
                RunAttemptId(_required_text(payload, "attempt_id")),
                envelope.aggregate_id,
                {"started"},
            )
            if envelope.occurred_at < attempt_transition_at:
                raise ValueError("Workshop run cannot start before its execution attempt")
        else:
            raise ValueError("Unsupported Workshop run.started event version")
        if status != "accepted" or envelope.actor_principal_id != agent_principal or envelope.occurred_at < accepted_at:
            raise ValueError("Workshop run can start only once through its attached agent")
        await connection.execute(
            "UPDATE runs SET status = 'started', started_at = ?, last_event_position = ? WHERE id = ?",
            (occurred_at, event.position, envelope.aggregate_id),
        )
        return

    if envelope.event_type == WorkshopEventType.RUN_COMPLETED:
        result_message_id: MessageId | None = None
        if envelope.event_version == 1:
            _require_exact_payload(payload, set())
        elif envelope.event_version == 2:
            _require_exact_payload(payload, {"attempt_id", "result_message_id"})
            attempt_transition_at = await _require_run_attempt(
                connection,
                RunAttemptId(_required_text(payload, "attempt_id")),
                envelope.aggregate_id,
                {"completed"},
            )
            if envelope.occurred_at < attempt_transition_at:
                raise ValueError("Workshop run cannot complete before its execution attempt")
            result_message_id = MessageId(_required_text(payload, "result_message_id"))
            async with connection.execute(
                "SELECT COUNT(*) FROM messages WHERE id = ? AND channel_id = ? "
                "AND author_principal_id = ? AND reply_to_message_id = ?",
                (result_message_id, ChannelId(str(row[9])), agent_principal, inbound_message_id),
            ) as cursor:
                result_row = await cursor.fetchone()
            if result_row is None or int(result_row[0]) != 1:
                raise ValueError("Workshop completed run must reference its canonical agent result")
        else:
            raise ValueError("Unsupported Workshop run.completed event version")
        if (
            status != "started"
            or envelope.actor_principal_id != agent_principal
            or started_at is None
            or envelope.occurred_at < started_at
        ):
            raise ValueError("Workshop run can complete only from started through its attached agent")
        await connection.execute(
            "UPDATE runs SET status = 'completed', terminal_at = ?, result_message_id = ?, "
            "last_event_position = ? WHERE id = ?",
            (occurred_at, result_message_id, event.position, envelope.aggregate_id),
        )
        return

    if envelope.event_type == WorkshopEventType.RUN_FAILED:
        expected_keys = {"failure_code"} if envelope.event_version == 1 else {"attempt_id", "failure_code"}
        if envelope.event_version not in {1, 2}:
            raise ValueError("Unsupported Workshop run.failed event version")
        _require_exact_payload(payload, expected_keys)
        if envelope.event_version == 2:
            attempt_transition_at = await _require_run_attempt(
                connection,
                RunAttemptId(_required_text(payload, "attempt_id")),
                envelope.aggregate_id,
                {"failed", "interrupted"},
            )
            if envelope.occurred_at < attempt_transition_at:
                raise ValueError("Workshop run cannot fail before its execution attempt")
        failure_code = _required_text(payload, "failure_code")
        if not _RUN_TERMINAL_CODE_PATTERN.fullmatch(failure_code):
            raise ValueError("Workshop run failure_code must be a lowercase identifier")
        if (
            status != "started"
            or envelope.actor_principal_id != agent_principal
            or started_at is None
            or envelope.occurred_at < started_at
        ):
            raise ValueError("Workshop run can fail only from started through its attached agent")
        await connection.execute(
            "UPDATE runs SET status = 'failed', terminal_at = ?, terminal_code = ?, "
            "last_event_position = ? WHERE id = ?",
            (occurred_at, failure_code, event.position, envelope.aggregate_id),
        )
        return

    if envelope.event_type == WorkshopEventType.RUN_CANCELLED:
        expected_keys = {"cancellation_code"} if envelope.event_version == 1 else {"attempt_id", "cancellation_code"}
        if envelope.event_version not in {1, 2}:
            raise ValueError("Unsupported Workshop run.cancelled event version")
        _require_exact_payload(payload, expected_keys)
        cancellation_code = _required_text(payload, "cancellation_code")
        if not _RUN_TERMINAL_CODE_PATTERN.fullmatch(cancellation_code):
            raise ValueError("Workshop run cancellation_code must be a lowercase identifier")
        lifecycle_floor = started_at if started_at is not None else accepted_at
        if envelope.event_version == 1:
            valid_actor = envelope.actor_principal_id == requested_by
        else:
            attempt_transition_at = await _require_run_attempt(
                connection,
                RunAttemptId(_required_text(payload, "attempt_id")),
                envelope.aggregate_id,
                {"cancelled"},
            )
            if envelope.occurred_at < attempt_transition_at:
                raise ValueError("Workshop run cannot cancel before its execution attempt")
            valid_actor = envelope.actor_principal_id == agent_principal and cancellation_requested_at is not None
            if cancellation_code != str(row[8]):
                raise ValueError("Workshop cancellation acknowledgement must match its request code")
        if status not in {"accepted", "started"} or not valid_actor or envelope.occurred_at < lifecycle_floor:
            raise ValueError("Workshop run cancellation acknowledgement is not authorized")
        await connection.execute(
            "UPDATE runs SET status = 'cancelled', terminal_at = ?, terminal_code = ?, "
            "last_event_position = ? WHERE id = ?",
            (occurred_at, cancellation_code, event.position, envelope.aggregate_id),
        )
        return

    raise ValueError(f"Unsupported Workshop run event: {envelope.event_type}")


async def _require_run_attempt(
    connection: aiosqlite.Connection,
    attempt_id: RunAttemptId,
    run_id: RunId,
    statuses: set[str],
) -> datetime:
    async with connection.execute(
        "SELECT status, started_at, terminal_at FROM run_attempts WHERE id = ? AND run_id = ?",
        (attempt_id, run_id),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None or str(row[0]) not in statuses:
        raise ValueError("Workshop run transition does not match its execution attempt")
    transition_at = row[2] if row[2] is not None else row[1]
    if transition_at is None:
        raise ValueError("Workshop run attempt is missing its transition timestamp")
    return _parse_projection_timestamp(transition_at)


def _execution_text(payload: dict[str, Any], key: str, *, pattern: re.Pattern[str] | None = None) -> str:
    value = _required_text(payload, key)
    if (
        len(value) > 128
        or value != value.strip()
        or any(character.isspace() and character != " " for character in value)
    ):
        raise ValueError(f"Workshop execution {key} is not bounded text")
    if pattern is not None and not pattern.fullmatch(value):
        raise ValueError(f"Workshop execution {key} is not a valid identifier")
    return value


async def _apply_run_attempt_event(connection: aiosqlite.Connection, event: StoredEvent) -> None:
    envelope = event.envelope
    if not isinstance(envelope.aggregate_id, RunAttemptId) or envelope.aggregate_type != "run_attempt":
        raise ValueError("Workshop run-attempt events require a typed attempt aggregate")
    if envelope.event_version != 1:
        raise ValueError("Unsupported Workshop run-attempt event version")
    payload = envelope.payload
    occurred_at = envelope.occurred_at
    occurred_text = occurred_at.isoformat()

    if envelope.event_type == WorkshopEventType.RUN_ATTEMPT_GRANTED:
        _require_exact_payload(
            payload,
            {
                "run_id",
                "attempt_sequence",
                "owner_id",
                "fence_token",
                "backend",
                "provider",
                "model",
                "execution_contract",
                "lease_version",
                "lease_expires_at",
            },
        )
        run_id = RunId(_required_text(payload, "run_id"))
        owner_id = RunExecutionOwnerId(_required_text(payload, "owner_id"))
        sequence = payload["attempt_sequence"]
        fence_token = payload["fence_token"]
        lease_version = payload["lease_version"]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (sequence, fence_token, lease_version)
        ):
            raise ValueError("Workshop attempt sequence, fence, and lease version must be positive integers")
        if sequence != fence_token or lease_version != 1:
            raise ValueError("Initial Workshop attempt fencing state is invalid")
        backend = _execution_text(payload, "backend", pattern=_EXECUTION_IDENTIFIER_PATTERN)
        provider_value = payload["provider"]
        if provider_value is not None and (
            not isinstance(provider_value, str) or not _EXECUTION_IDENTIFIER_PATTERN.fullmatch(provider_value)
        ):
            raise ValueError("Workshop execution provider is not a valid identifier")
        provider = str(provider_value) if provider_value is not None else None
        model = _execution_text(payload, "model", pattern=_MODEL_IDENTIFIER_PATTERN)
        execution_contract = _execution_text(payload, "execution_contract", pattern=_EXECUTION_IDENTIFIER_PATTERN)
        lease_expires_at = _parse_projection_timestamp(payload["lease_expires_at"])
        if lease_expires_at <= occurred_at:
            raise ValueError("Workshop attempt lease must expire after it is granted")
        async with connection.execute(
            "SELECT r.workshop_id, r.status, r.accepted_at, r.cancellation_requested_at, a.principal_id, "
            "COALESCE(MAX(ra.attempt_sequence), 0) "
            "FROM runs r JOIN agents a ON a.id = r.agent_id "
            "LEFT JOIN run_attempts ra ON ra.run_id = r.id WHERE r.id = ? GROUP BY r.id",
            (run_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if (
            row is None
            or WorkshopId(str(row[0])) != envelope.workshop_id
            or str(row[1]) != "accepted"
            or row[3] is not None
            or envelope.actor_principal_id != PrincipalId(str(row[4]))
            or occurred_at < _parse_projection_timestamp(row[2])
            or sequence != int(row[5]) + 1
        ):
            raise ValueError("Workshop attempt grant is not authorized for this accepted run")
        await connection.execute(
            "INSERT INTO run_attempts "
            "(id, run_id, attempt_sequence, owner_id, fence_token, status, backend, provider, model, "
            "execution_contract, lease_version, granted_at, lease_expires_at, last_event_position) "
            "VALUES (?, ?, ?, ?, ?, 'granted', ?, ?, ?, ?, 1, ?, ?, ?)",
            (
                envelope.aggregate_id,
                run_id,
                sequence,
                owner_id,
                fence_token,
                backend,
                provider,
                model,
                execution_contract,
                occurred_text,
                lease_expires_at.isoformat(),
                event.position,
            ),
        )
        return

    run_id = RunId(_required_text(payload, "run_id"))
    owner_id = RunExecutionOwnerId(_required_text(payload, "owner_id"))
    fence_token = payload.get("fence_token")
    lease_version = payload.get("lease_version")
    if (
        isinstance(fence_token, bool)
        or not isinstance(fence_token, int)
        or isinstance(lease_version, bool)
        or not isinstance(lease_version, int)
    ):
        raise ValueError("Workshop attempt fence and lease version must be integers")
    async with connection.execute(
        "SELECT ra.run_id, ra.status, ra.owner_id, ra.fence_token, ra.lease_version, "
        "ra.lease_expires_at, ra.granted_at, ra.started_at, r.workshop_id, "
        "r.cancellation_requested_at, a.principal_id "
        "FROM run_attempts ra JOIN runs r ON r.id = ra.run_id "
        "JOIN agents a ON a.id = r.agent_id WHERE ra.id = ?",
        (envelope.aggregate_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if (
        row is None
        or RunId(str(row[0])) != run_id
        or RunExecutionOwnerId(str(row[2])) != owner_id
        or int(row[3]) != fence_token
        or WorkshopId(str(row[8])) != envelope.workshop_id
        or envelope.actor_principal_id != PrincipalId(str(row[10]))
    ):
        raise ValueError("Workshop attempt event does not match its fenced owner")
    status = str(row[1])
    current_lease_version = int(row[4])
    lease_expires_at = _parse_projection_timestamp(row[5])
    granted_at = _parse_projection_timestamp(row[6])
    attempt_started_at = None if row[7] is None else _parse_projection_timestamp(row[7])

    if envelope.event_type == WorkshopEventType.RUN_ATTEMPT_STARTED:
        _require_exact_payload(payload, {"run_id", "owner_id", "fence_token", "lease_version"})
        if (
            status != "granted"
            or lease_version != current_lease_version
            or occurred_at < granted_at
            or occurred_at >= lease_expires_at
            or row[9] is not None
        ):
            raise ValueError("Workshop attempt cannot start without a current grant")
        await connection.execute(
            "UPDATE run_attempts SET status = 'started', started_at = ?, last_event_position = ? WHERE id = ?",
            (occurred_text, event.position, envelope.aggregate_id),
        )
        return

    if envelope.event_type == WorkshopEventType.RUN_ATTEMPT_LEASE_RENEWED:
        _require_exact_payload(
            payload,
            {"run_id", "owner_id", "fence_token", "lease_version", "lease_expires_at"},
        )
        new_expiry = _parse_projection_timestamp(payload["lease_expires_at"])
        if (
            status not in {"granted", "started"}
            or lease_version != current_lease_version + 1
            or occurred_at < (attempt_started_at or granted_at)
            or occurred_at >= lease_expires_at
            or new_expiry <= lease_expires_at
        ):
            raise ValueError("Workshop attempt lease renewal is stale or does not extend authority")
        await connection.execute(
            "UPDATE run_attempts SET lease_version = ?, lease_expires_at = ?, last_event_position = ? WHERE id = ?",
            (lease_version, new_expiry.isoformat(), event.position, envelope.aggregate_id),
        )
        return

    terminal_specs: dict[str, tuple[set[str], str, str | None, bool]] = {
        WorkshopEventType.RUN_ATTEMPT_EXPIRED: ({"granted"}, "expired", "lease_expired", True),
        WorkshopEventType.RUN_ATTEMPT_INTERRUPTED: ({"started"}, "interrupted", "execution_interrupted", True),
        WorkshopEventType.RUN_ATTEMPT_COMPLETED: ({"started"}, "completed", None, False),
        WorkshopEventType.RUN_ATTEMPT_FAILED: ({"started"}, "failed", None, False),
        WorkshopEventType.RUN_ATTEMPT_CANCELLED: ({"granted", "started"}, "cancelled", None, False),
    }
    spec = terminal_specs.get(envelope.event_type)
    if spec is None:
        raise ValueError(f"Unsupported Workshop run-attempt event: {envelope.event_type}")
    allowed, terminal_status, fixed_code, requires_expiry = spec
    expected = {"run_id", "owner_id", "fence_token", "lease_version"}
    if fixed_code is not None or terminal_status in {"failed", "cancelled"}:
        expected.add("terminal_code")
    _require_exact_payload(payload, expected)
    terminal_code = fixed_code
    if "terminal_code" in payload:
        terminal_code = _required_text(payload, "terminal_code")
    if terminal_code is not None and not _RUN_TERMINAL_CODE_PATTERN.fullmatch(terminal_code):
        raise ValueError("Workshop attempt terminal_code must be a lowercase identifier")
    if status not in allowed or lease_version != current_lease_version:
        raise ValueError("Workshop attempt terminal transition is stale")
    if occurred_at < (attempt_started_at or granted_at):
        raise ValueError("Workshop attempt terminal transition cannot precede its current state")
    if requires_expiry and occurred_at < lease_expires_at:
        raise ValueError("Workshop attempt cannot expire before its lease")
    if not requires_expiry and occurred_at >= lease_expires_at:
        raise ValueError("Workshop attempt owner cannot settle an expired lease")
    if terminal_status == "cancelled" and row[9] is None:
        raise ValueError("Workshop attempt cannot confirm cancellation without durable human intent")
    await connection.execute(
        "UPDATE run_attempts SET status = ?, terminal_at = ?, terminal_code = ?, last_event_position = ? WHERE id = ?",
        (terminal_status, occurred_text, terminal_code, event.position, envelope.aggregate_id),
    )


async def _apply_agent_delegation_event(
    connection: aiosqlite.Connection,
    event: StoredEvent,
) -> None:
    envelope = event.envelope
    if (
        not isinstance(envelope.aggregate_id, AgentDelegationId)
        or envelope.aggregate_type != "agent_delegation"
        or envelope.event_version != 1
    ):
        raise ValueError("Workshop delegation events require a typed v1 delegation aggregate")
    payload = envelope.payload
    occurred_at = envelope.occurred_at.isoformat()

    if envelope.event_type == WorkshopEventType.AGENT_DELEGATION_REQUESTED:
        _require_exact_payload(
            payload,
            {
                "channel_id",
                "thread_root_id",
                "root_run_id",
                "parent_run_id",
                "parent_delegation_id",
                "child_run_id",
                "requesting_principal_id",
                "caller_agent_id",
                "target_agent_id",
                "caller_sponsor_principal_id",
                "caller_runtime_profile_id",
                "target_sponsor_principal_id",
                "target_runtime_profile_id",
                "caller_definition_revision_id",
                "target_definition_revision_id",
                "request_message_id",
                "task",
                "context",
                "request_hash",
                "depth",
            },
        )
        channel_id = ChannelId(_required_text(payload, "channel_id"))
        thread_value = payload.get("thread_root_id")
        thread_root_id = None if thread_value is None else MessageId(str(thread_value))
        root_run_id = RunId(_required_text(payload, "root_run_id"))
        parent_run_id = RunId(_required_text(payload, "parent_run_id"))
        parent_delegation_value = payload.get("parent_delegation_id")
        parent_delegation_id = (
            None if parent_delegation_value is None else AgentDelegationId(str(parent_delegation_value))
        )
        child_run_id = RunId(_required_text(payload, "child_run_id"))
        requesting_principal_id = PrincipalId(_required_text(payload, "requesting_principal_id"))
        caller_agent_id = AgentId(_required_text(payload, "caller_agent_id"))
        target_agent_id = AgentId(_required_text(payload, "target_agent_id"))
        caller_sponsor = PrincipalId(_required_text(payload, "caller_sponsor_principal_id"))
        caller_runtime = _required_text(payload, "caller_runtime_profile_id")
        target_sponsor = PrincipalId(_required_text(payload, "target_sponsor_principal_id"))
        target_runtime = _required_text(payload, "target_runtime_profile_id")
        caller_revision = AgentDefinitionRevisionId(_required_text(payload, "caller_definition_revision_id"))
        target_revision = AgentDefinitionRevisionId(_required_text(payload, "target_definition_revision_id"))
        request_message_id = MessageId(_required_text(payload, "request_message_id"))
        task = _required_text(payload, "task")
        context = payload.get("context")
        request_hash = _required_text(payload, "request_hash")
        depth = payload.get("depth")
        if not isinstance(context, dict) or set(context) != {"summary", "message_ids"}:
            raise ValueError("Workshop delegation context must contain only summary and message_ids")
        summary = context.get("summary")
        raw_message_ids = context.get("message_ids")
        if not isinstance(summary, str) or len(summary) > 4_000:
            raise ValueError("Workshop delegation context summary is invalid")
        if not isinstance(raw_message_ids, list) or len(raw_message_ids) > 12:
            raise ValueError("Workshop delegation context message references are invalid")
        try:
            context_message_ids = tuple(MessageId(str(item)) for item in raw_message_ids)
        except ValueError as exc:
            raise ValueError("Workshop delegation context message references are invalid") from exc
        if len(set(context_message_ids)) != len(context_message_ids):
            raise ValueError("Workshop delegation context message references must be unique")
        if len(task) > 6_000:
            raise ValueError("Workshop delegation task is too large")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
            raise ValueError("Workshop delegation depth must be positive")
        if not _SHA256_PATTERN.fullmatch(request_hash):
            raise ValueError("Workshop delegation request hash is invalid")
        async with connection.execute(
            "SELECT parent.workshop_id, parent.channel_id, parent.requested_by_principal_id, "
            "parent.agent_id, parent.runtime_profile_id, parent.sponsor_principal_id, "
            "parent.agent_definition_revision_id, child.channel_id, child.requested_by_principal_id, "
            "child.agent_id, child.runtime_profile_id, child.sponsor_principal_id, "
            "child.agent_definition_revision_id, child.parent_run_id, child.delegation_id, "
            "message.author_principal_id, caller.principal_id, message.thread_root_id, "
            "message.body, parent.delegation_id, target_definition.handle "
            "FROM runs parent JOIN runs child ON child.id = ? "
            "JOIN messages message ON message.id = ? AND message.channel_id = parent.channel_id "
            "JOIN agents caller ON caller.id = parent.agent_id "
            "JOIN agent_definitions target_definition ON target_definition.agent_id = child.agent_id "
            "WHERE parent.id = ?",
            (child_run_id, request_message_id, parent_run_id),
        ) as cursor:
            row = await cursor.fetchone()
        expected = (
            envelope.workshop_id,
            channel_id,
            requesting_principal_id,
            caller_agent_id,
            caller_runtime,
            caller_sponsor,
            caller_revision,
            channel_id,
            requesting_principal_id,
            target_agent_id,
            target_runtime,
            target_sponsor,
            target_revision,
            parent_run_id,
            envelope.aggregate_id,
        )
        if row is None or tuple(row[:15]) != expected or row[15] != row[16]:
            raise ValueError("Workshop delegation does not match its canonical runs and request message")
        expected_parent_delegation = None if row[19] is None else AgentDelegationId(str(row[19]))
        if parent_delegation_id != expected_parent_delegation:
            raise ValueError("Workshop delegation parent identity does not match its parent run")
        if parent_delegation_id is None:
            if root_run_id != parent_run_id or depth != 1:
                raise ValueError("Workshop root delegation lineage is invalid")
        else:
            async with connection.execute(
                "SELECT root_run_id, depth FROM agent_delegations WHERE id = ?",
                (parent_delegation_id,),
            ) as cursor:
                parent_delegation_row = await cursor.fetchone()
            if (
                parent_delegation_row is None
                or root_run_id != RunId(str(parent_delegation_row[0]))
                or depth != int(parent_delegation_row[1]) + 1
            ):
                raise ValueError("Workshop nested delegation lineage is invalid")
        expected_thread = None if row[17] is None else MessageId(str(row[17]))
        if thread_root_id != expected_thread:
            raise ValueError("Workshop delegation thread does not match its request message")
        if context_message_ids:
            placeholders = ",".join("?" for _ in context_message_ids)
            async with connection.execute(
                f"SELECT id FROM messages WHERE channel_id = ? AND id IN ({placeholders})",
                (channel_id, *context_message_ids),
            ) as cursor:
                found_context = {MessageId(str(item[0])) for item in await cursor.fetchall()}
            if found_context != set(context_message_ids):
                raise ValueError("Workshop delegation context references leave the shared channel")
        normalized_context = {
            "summary": summary,
            "message_ids": [str(item) for item in context_message_ids],
        }
        target_handle = str(row[20])
        encoded_request = json.dumps(
            {
                "target_handle": target_handle,
                "task": task,
                "context": normalized_context,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if hashlib.sha256(encoded_request.encode()).hexdigest() != request_hash:
            raise ValueError("Workshop delegation request hash does not match its bounded input")
        expected_body_lines = [
            f"Delegation request from the current agent to @{target_handle}.",
            "",
            "Task:",
            task,
        ]
        if summary:
            expected_body_lines.extend(("", "Bounded shared context:", summary))
        if context_message_ids:
            expected_body_lines.extend(
                (
                    "",
                    "Canonical shared-channel references:",
                    ", ".join(str(item) for item in context_message_ids),
                )
            )
        expected_body_lines.extend(
            (
                "",
                f"Parent run: {parent_run_id}",
                "Return the requested result to the calling agent. Do not expose credentials or private memory.",
            )
        )
        if str(row[18]) != "\n".join(expected_body_lines):
            raise ValueError("Workshop delegation request message does not match its bounded input")
        if envelope.actor_principal_id != PrincipalId(str(row[16])):
            raise ValueError("Workshop delegation request actor must be its caller agent")
        async with connection.execute(
            "SELECT capabilities_json FROM agent_definition_revisions WHERE id = ?",
            (caller_revision,),
        ) as cursor:
            capability_row = await cursor.fetchone()
        if capability_row is None or "agent_delegation" not in json.loads(str(capability_row[0])):
            raise ValueError("Workshop delegation caller revision lacks delegation authority")
        await connection.execute(
            "INSERT INTO agent_delegations "
            "(id, workshop_id, channel_id, thread_root_id, root_run_id, parent_run_id, "
            "parent_delegation_id, child_run_id, requesting_principal_id, caller_agent_id, "
            "target_agent_id, caller_sponsor_principal_id, caller_runtime_profile_id, "
            "target_sponsor_principal_id, target_runtime_profile_id, "
            "caller_definition_revision_id, target_definition_revision_id, request_message_id, "
            "task, context_json, request_hash, depth, status, created_at, "
            "created_event_position, last_event_position) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "'requested', ?, ?, ?)",
            (
                envelope.aggregate_id,
                envelope.workshop_id,
                channel_id,
                thread_root_id,
                root_run_id,
                parent_run_id,
                parent_delegation_id,
                child_run_id,
                requesting_principal_id,
                caller_agent_id,
                target_agent_id,
                caller_sponsor,
                caller_runtime,
                target_sponsor,
                target_runtime,
                caller_revision,
                target_revision,
                request_message_id,
                task,
                json.dumps(context, separators=(",", ":"), sort_keys=True),
                request_hash,
                depth,
                occurred_at,
                event.position,
                event.position,
            ),
        )
        return

    async with connection.execute(
        "SELECT d.status, d.child_run_id, d.target_agent_id, a.principal_id, r.status, "
        "r.result_message_id FROM agent_delegations d "
        "JOIN agents a ON a.id = d.target_agent_id "
        "JOIN runs r ON r.id = d.child_run_id WHERE d.id = ?",
        (envelope.aggregate_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None or envelope.actor_principal_id != PrincipalId(str(row[3])):
        raise ValueError("Workshop delegation transition actor must be its target agent")
    if RunId(_required_text(payload, "child_run_id")) != RunId(str(row[1])):
        raise ValueError("Workshop delegation transition identifies the wrong child run")
    if envelope.event_type == WorkshopEventType.AGENT_DELEGATION_STARTED:
        _require_exact_payload(payload, {"child_run_id"})
        if str(row[0]) != "requested" or str(row[4]) not in {"accepted", "started"}:
            raise ValueError("Workshop delegation can start only once with a nonterminal child run")
        await connection.execute(
            "UPDATE agent_delegations SET status = 'executing', started_at = ?, last_event_position = ? WHERE id = ?",
            (occurred_at, event.position, envelope.aggregate_id),
        )
        return

    if envelope.event_type not in {
        WorkshopEventType.AGENT_DELEGATION_COMPLETED,
        WorkshopEventType.AGENT_DELEGATION_FAILED,
        WorkshopEventType.AGENT_DELEGATION_CANCELLED,
    }:
        raise ValueError("Unsupported Workshop delegation event type")
    _require_exact_payload(payload, {"child_run_id", "outcome_code", "response_message_id"})
    outcome_code = _required_text(payload, "outcome_code")
    response_value = payload.get("response_message_id")
    response_message_id = None if response_value is None else MessageId(str(response_value))
    terminal_status = {
        WorkshopEventType.AGENT_DELEGATION_COMPLETED.value: "completed",
        WorkshopEventType.AGENT_DELEGATION_FAILED.value: "failed",
        WorkshopEventType.AGENT_DELEGATION_CANCELLED.value: "cancelled",
    }[str(envelope.event_type)]
    if str(row[0]) not in {"requested", "executing"} or str(row[4]) != terminal_status:
        raise ValueError("Workshop delegation terminal state does not match its child run")
    expected_response = None if row[5] is None else MessageId(str(row[5]))
    if response_message_id != expected_response:
        raise ValueError("Workshop delegation response does not match its child run result")
    await connection.execute(
        "UPDATE agent_delegations SET status = ?, outcome_code = ?, response_message_id = ?, "
        "terminal_at = ?, last_event_position = ? WHERE id = ?",
        (
            terminal_status,
            outcome_code,
            response_message_id,
            occurred_at,
            event.position,
            envelope.aggregate_id,
        ),
    )


class CanonicalConversationProjection:
    """Rebuild the initial Workshop collaboration records from events."""

    name = "canonical_conversations"
    # Agent definitions project durable draft, active, and archived lifecycle state.
    version = 15

    async def reset(self, connection: aiosqlite.Connection) -> None:
        async with connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'") as cursor:
            existing_tables = {str(row[0]) for row in await cursor.fetchall()}
        if "agent_definitions" in existing_tables:
            # Break the active-revision pointer before deleting immutable
            # revisions. Replay restores it from the activation event.
            await connection.execute("UPDATE agent_definitions SET active_revision_id = NULL")
        if "runs" in existing_tables and await _table_has_column(
            connection,
            "runs",
            "parent_run_id",
        ):
            await connection.execute("UPDATE runs SET parent_run_id = NULL")
        for table in (
            "deliveries",
            "artifacts",
            "agent_delegations",
            "run_attempts",
            "runs",
            "message_reactions",
            "messages",
            "principal_agent_enablements",
            "channel_agent_dismissals",
            "channel_agent_runtime_assignments",
            "channel_agents",
            "channel_bindings",
            "channel_memberships",
            "channels",
            "agent_definition_revisions",
            "agent_definitions",
            "agents",
            "workshop_memberships",
            "external_identities",
            "principals",
            "workshops",
        ):
            if table in existing_tables:
                await connection.execute(f"DELETE FROM {table}")

    async def apply(self, connection: aiosqlite.Connection, event: StoredEvent) -> None:
        envelope = event.envelope
        payload = envelope.payload
        occurred_at = envelope.occurred_at.isoformat()

        if envelope.event_type in {
            WorkshopEventType.RUN_ACCEPTED,
            WorkshopEventType.RUN_CANCELLATION_REQUESTED,
            WorkshopEventType.RUN_STARTED,
            WorkshopEventType.RUN_COMPLETED,
            WorkshopEventType.RUN_FAILED,
            WorkshopEventType.RUN_CANCELLED,
        }:
            await _apply_run_event(connection, event)
            return

        if envelope.event_type in {
            WorkshopEventType.AGENT_DELEGATION_REQUESTED,
            WorkshopEventType.AGENT_DELEGATION_STARTED,
            WorkshopEventType.AGENT_DELEGATION_COMPLETED,
            WorkshopEventType.AGENT_DELEGATION_FAILED,
            WorkshopEventType.AGENT_DELEGATION_CANCELLED,
        }:
            await _apply_agent_delegation_event(connection, event)
            return

        if envelope.event_type in {
            WorkshopEventType.RUN_ATTEMPT_GRANTED,
            WorkshopEventType.RUN_ATTEMPT_STARTED,
            WorkshopEventType.RUN_ATTEMPT_LEASE_RENEWED,
            WorkshopEventType.RUN_ATTEMPT_EXPIRED,
            WorkshopEventType.RUN_ATTEMPT_INTERRUPTED,
            WorkshopEventType.RUN_ATTEMPT_COMPLETED,
            WorkshopEventType.RUN_ATTEMPT_FAILED,
            WorkshopEventType.RUN_ATTEMPT_CANCELLED,
        }:
            await _apply_run_attempt_event(connection, event)
            return

        if envelope.event_type == WorkshopEventType.WORKSHOP_CREATED:
            await connection.execute(
                "INSERT INTO workshops (id, name, created_at) VALUES (?, ?, ?)",
                (envelope.aggregate_id, _required_text(payload, "name"), occurred_at),
            )
        elif envelope.event_type == WorkshopEventType.PRINCIPAL_CREATED:
            await connection.execute(
                "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    _required_text(payload, "kind"),
                    _required_text(payload, "display_name"),
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.EXTERNAL_IDENTITY_BOUND:
            await connection.execute(
                "INSERT INTO external_identities "
                "(id, principal_id, provider, external_subject, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    _required_text(payload, "principal_id"),
                    _required_text(payload, "provider"),
                    _required_text(payload, "external_subject"),
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.WORKSHOP_MEMBER_ADDED:
            await connection.execute(
                "INSERT INTO workshop_memberships "
                "(id, workshop_id, principal_id, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    envelope.workshop_id,
                    _required_text(payload, "principal_id"),
                    _required_text(payload, "role"),
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.CHANNEL_CREATED:
            name = payload.get("name")
            if name is not None and not isinstance(name, str):
                raise ValueError("Workshop channel name must be a string or null")
            await connection.execute(
                "INSERT INTO channels (id, workshop_id, kind, name, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    envelope.workshop_id,
                    _required_text(payload, "kind"),
                    name,
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.CHANNEL_MEMBER_ADDED:
            role = _required_text(payload, "role")
            if role not in {"owner", "participant"}:
                raise ValueError("Workshop channel member role must be 'owner' or 'participant'")
            await connection.execute(
                "INSERT INTO channel_memberships "
                "(id, channel_id, principal_id, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    _required_text(payload, "channel_id"),
                    _required_text(payload, "principal_id"),
                    role,
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.TRANSPORT_CHANNEL_BOUND:
            await connection.execute(
                "INSERT INTO channel_bindings "
                "(id, channel_id, transport, external_channel_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    _required_text(payload, "channel_id"),
                    _required_text(payload, "transport"),
                    _required_text(payload, "external_channel_id"),
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.AGENT_CREATED:
            await connection.execute(
                "INSERT INTO agents (id, workshop_id, principal_id, name, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    envelope.workshop_id,
                    _required_text(payload, "principal_id"),
                    _required_text(payload, "name"),
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.AGENT_DEFINITION_CREATED:
            if (
                not isinstance(envelope.aggregate_id, AgentDefinitionId)
                or envelope.aggregate_type != "agent_definition"
            ):
                raise ValueError("Workshop agent definition creation requires a typed definition aggregate")
            _require_exact_payload(
                payload,
                {"agent_id", "handle", "display_name", "description", "presentation", "lifecycle_state"},
            )
            agent_id = AgentId(_required_text(payload, "agent_id"))
            handle = normalize_agent_handle(payload.get("handle"))
            display_name = validate_agent_text(
                payload.get("display_name"), field="display_name", maximum=MAX_AGENT_DISPLAY_NAME
            )
            description = validate_agent_text(
                payload.get("description"),
                field="description",
                maximum=MAX_AGENT_DESCRIPTION,
                allow_empty=True,
            )
            lifecycle_state = _required_text(payload, "lifecycle_state")
            if lifecycle_state not in {"draft", "active", "archived"}:
                raise ValueError("Workshop agent definition lifecycle_state is invalid")
            presentation_json = validate_agent_presentation(payload.get("presentation"))
            async with connection.execute("SELECT workshop_id FROM agents WHERE id = ?", (agent_id,)) as cursor:
                agent_row = await cursor.fetchone()
            if agent_row is None or WorkshopId(str(agent_row[0])) != envelope.workshop_id:
                raise ValueError("Workshop agent definition must reference an agent in its workshop")
            await connection.execute(
                "INSERT INTO agent_definitions "
                "(id, workshop_id, agent_id, handle, display_name, description, "
                "presentation_json, lifecycle_state, active_revision_id, created_at, "
                "created_event_position) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                (
                    envelope.aggregate_id,
                    envelope.workshop_id,
                    agent_id,
                    handle,
                    display_name,
                    description,
                    presentation_json,
                    lifecycle_state,
                    occurred_at,
                    event.position,
                ),
            )
        elif envelope.event_type == WorkshopEventType.AGENT_DEFINITION_REVISION_ADDED:
            if (
                not isinstance(envelope.aggregate_id, AgentDefinitionRevisionId)
                or envelope.aggregate_type != "agent_definition_revision"
            ):
                raise ValueError("Workshop agent revision creation requires a typed revision aggregate")
            _require_exact_payload(
                payload,
                {"definition_id", "revision_number", "purpose", "instructions", "capabilities"},
            )
            definition_id = AgentDefinitionId(_required_text(payload, "definition_id"))
            revision_number = payload.get("revision_number")
            if not isinstance(revision_number, int) or isinstance(revision_number, bool) or revision_number < 1:
                raise ValueError("Workshop agent revision_number must be positive")
            purpose = validate_agent_text(payload.get("purpose"), field="purpose", maximum=MAX_AGENT_PURPOSE)
            instructions = validate_agent_text(
                payload.get("instructions"), field="instructions", maximum=MAX_AGENT_INSTRUCTIONS
            )
            capabilities = validate_agent_capabilities(payload.get("capabilities"))
            async with connection.execute(
                "SELECT workshop_id FROM agent_definitions WHERE id = ?", (definition_id,)
            ) as cursor:
                definition_row = await cursor.fetchone()
            if definition_row is None or WorkshopId(str(definition_row[0])) != envelope.workshop_id:
                raise ValueError("Workshop agent revision must reference a definition in its workshop")
            async with connection.execute(
                "SELECT COALESCE(MAX(revision_number), 0) FROM agent_definition_revisions "
                "WHERE agent_definition_id = ?",
                (definition_id,),
            ) as cursor:
                revision_count_row = await cursor.fetchone()
            assert revision_count_row is not None
            expected_number = int(revision_count_row[0]) + 1
            if revision_number != expected_number:
                raise ValueError("Workshop agent revisions must be sequential")
            await connection.execute(
                "INSERT INTO agent_definition_revisions "
                "(id, agent_definition_id, revision_number, purpose, instructions, "
                "capabilities_json, created_at, created_event_position) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    definition_id,
                    revision_number,
                    purpose,
                    instructions,
                    json.dumps(capabilities, separators=(",", ":")),
                    occurred_at,
                    event.position,
                ),
            )
        elif envelope.event_type == WorkshopEventType.AGENT_DEFINITION_REVISION_ACTIVATED:
            if (
                not isinstance(envelope.aggregate_id, AgentDefinitionId)
                or envelope.aggregate_type != "agent_definition"
            ):
                raise ValueError("Workshop agent revision activation requires a typed definition aggregate")
            _require_exact_payload(payload, {"revision_id"})
            revision_id = AgentDefinitionRevisionId(_required_text(payload, "revision_id"))
            async with connection.execute(
                "SELECT d.workshop_id, d.lifecycle_state, r.agent_definition_id "
                "FROM agent_definitions d JOIN agent_definition_revisions r ON r.id = ? "
                "WHERE d.id = ?",
                (revision_id, envelope.aggregate_id),
            ) as cursor:
                revision_row = await cursor.fetchone()
            if (
                revision_row is None
                or WorkshopId(str(revision_row[0])) != envelope.workshop_id
                or str(revision_row[1]) == "archived"
                or AgentDefinitionId(str(revision_row[2])) != envelope.aggregate_id
            ):
                raise ValueError("Workshop agent revision activation is invalid")
            await connection.execute(
                "UPDATE agent_definitions SET lifecycle_state = 'active', active_revision_id = ? WHERE id = ?",
                (revision_id, envelope.aggregate_id),
            )
        elif envelope.event_type == WorkshopEventType.AGENT_DEFINITION_ARCHIVED:
            if (
                not isinstance(envelope.aggregate_id, AgentDefinitionId)
                or envelope.aggregate_type != "agent_definition"
            ):
                raise ValueError("Workshop agent archival requires a typed definition aggregate")
            _require_exact_payload(payload, set())
            async with connection.execute(
                "SELECT workshop_id, lifecycle_state FROM agent_definitions WHERE id = ?",
                (envelope.aggregate_id,),
            ) as cursor:
                definition_row = await cursor.fetchone()
            if (
                definition_row is None
                or WorkshopId(str(definition_row[0])) != envelope.workshop_id
                or str(definition_row[1]) == "archived"
            ):
                raise ValueError("Workshop agent archival is invalid")
            await connection.execute(
                "UPDATE agent_definitions SET lifecycle_state = 'archived' WHERE id = ?",
                (envelope.aggregate_id,),
            )
        elif envelope.event_type == WorkshopEventType.PRINCIPAL_AGENT_ENABLED:
            if not isinstance(envelope.aggregate_id, AgentEnablementId):
                raise ValueError("Workshop agent enablement requires a typed enablement aggregate")
            _require_exact_payload(
                payload,
                {
                    "principal_id",
                    "agent_definition_id",
                    "agent_id",
                    "direct_channel_id",
                    "runtime_profile_id",
                },
            )
            principal_id = _required_text(payload, "principal_id")
            definition_id = _required_text(payload, "agent_definition_id")
            agent_id = _required_text(payload, "agent_id")
            channel_id = _required_text(payload, "direct_channel_id")
            runtime_profile_id = _required_text(payload, "runtime_profile_id")
            async with connection.execute(
                "SELECT d.workshop_id, d.agent_id, d.lifecycle_state, a.principal_id, "
                "c.workshop_id, c.kind, ra.runtime_profile_id FROM agent_definitions d "
                "JOIN agents a ON a.id = d.agent_id JOIN channels c ON c.id = ? "
                "JOIN channel_memberships hm ON hm.channel_id = c.id AND hm.principal_id = ? "
                "AND hm.role = 'owner' JOIN channel_agents ca ON ca.channel_id = c.id "
                "AND ca.agent_id = ? AND ca.detached_at IS NULL "
                "JOIN channel_agent_runtime_assignments ra "
                "ON ra.channel_id = c.id AND ra.agent_id = ca.agent_id WHERE d.id = ?",
                (channel_id, principal_id, agent_id, definition_id),
            ) as cursor:
                authority = await cursor.fetchone()
            if (
                authority is None
                or tuple(authority[:3])
                != (
                    envelope.workshop_id,
                    agent_id,
                    "active",
                )
                or tuple(authority[4:])
                != (
                    envelope.workshop_id,
                    "direct",
                    runtime_profile_id,
                )
            ):
                raise ValueError("Workshop agent enablement authority is invalid")
            agent_principal_id = str(authority[3])
            async with connection.execute(
                "SELECT COUNT(*), SUM(CASE WHEN principal_id = ? THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN principal_id = ? THEN 1 ELSE 0 END) "
                "FROM channel_memberships WHERE channel_id = ?",
                (principal_id, agent_principal_id, channel_id),
            ) as cursor:
                membership_counts = await cursor.fetchone()
            if membership_counts is None or tuple(membership_counts) != (2, 1, 1):
                raise ValueError("Workshop agent enablement direct channel must have exactly two members")
            await connection.execute(
                "INSERT INTO principal_agent_enablements "
                "(id, workshop_id, principal_id, agent_definition_id, agent_id, "
                "direct_channel_id, runtime_profile_id, lifecycle_state, created_at, "
                "updated_at, created_event_position, last_event_position) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'enabled', ?, ?, ?, ?) "
                "ON CONFLICT(principal_id, agent_definition_id) DO UPDATE SET "
                "lifecycle_state = 'enabled', updated_at = excluded.updated_at, "
                "last_event_position = excluded.last_event_position",
                (
                    envelope.aggregate_id,
                    envelope.workshop_id,
                    principal_id,
                    definition_id,
                    agent_id,
                    channel_id,
                    runtime_profile_id,
                    occurred_at,
                    occurred_at,
                    event.position,
                    event.position,
                ),
            )
        elif envelope.event_type == WorkshopEventType.PRINCIPAL_AGENT_DISABLED:
            if not isinstance(envelope.aggregate_id, AgentEnablementId):
                raise ValueError("Workshop agent disablement requires a typed enablement aggregate")
            _require_exact_payload(payload, set())
            cursor = await connection.execute(
                "UPDATE principal_agent_enablements SET lifecycle_state = 'disabled', "
                "updated_at = ?, last_event_position = ? WHERE id = ? "
                "AND workshop_id = ? AND lifecycle_state = 'enabled'",
                (occurred_at, event.position, envelope.aggregate_id, envelope.workshop_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Workshop agent disablement has no enabled principal agent")
        elif envelope.event_type == WorkshopEventType.PRINCIPAL_AGENT_RUNTIME_CHANGED:
            if not isinstance(envelope.aggregate_id, AgentEnablementId):
                raise ValueError("Workshop agent runtime change requires a typed enablement aggregate")
            _require_exact_payload(payload, {"runtime_profile_id"})
            runtime_profile_id = _required_text(payload, "runtime_profile_id")
            async with connection.execute(
                "SELECT e.direct_channel_id, e.agent_id, ra.runtime_profile_id "
                "FROM principal_agent_enablements e "
                "JOIN channel_agent_runtime_assignments ra "
                "ON ra.channel_id = e.direct_channel_id AND ra.agent_id = e.agent_id "
                "WHERE e.id = ? AND e.workshop_id = ?",
                (envelope.aggregate_id, envelope.workshop_id),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None or str(row[2]) != runtime_profile_id:
                raise ValueError("Workshop agent runtime change does not match its channel assignment")
            cursor = await connection.execute(
                "UPDATE principal_agent_enablements SET runtime_profile_id = ?, "
                "updated_at = ?, last_event_position = ? WHERE id = ?",
                (runtime_profile_id, occurred_at, event.position, envelope.aggregate_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Workshop agent runtime change has no principal agent")
        elif envelope.event_type == WorkshopEventType.CHANNEL_AGENT_ATTACHED:
            if envelope.event_version not in {1, 2}:
                raise ValueError("Unsupported Workshop channel-agent attachment event version")
            keys = {"channel_id", "agent_id"}
            if envelope.event_version == 2:
                keys.update({"sponsor_principal_id", "runtime_profile_id"})
            _require_exact_payload(payload, keys)
            channel_id = ChannelId(_required_text(payload, "channel_id"))
            agent_id = AgentId(_required_text(payload, "agent_id"))
            sponsor_principal_id = (
                PrincipalId(_required_text(payload, "sponsor_principal_id")) if envelope.event_version == 2 else None
            )
            runtime_profile_id = _required_text(payload, "runtime_profile_id") if envelope.event_version == 2 else None
            if envelope.event_version == 2:
                if envelope.actor_principal_id is None:
                    raise ValueError("Workshop channel-agent attachment requires a human actor")
                async with connection.execute(
                    "SELECT c.workshop_id FROM channels c "
                    "JOIN channel_memberships owner ON owner.channel_id = c.id "
                    "AND owner.principal_id = ? AND owner.role = 'owner' "
                    "JOIN principal_agent_enablements pae ON pae.principal_id = ? "
                    "AND pae.agent_id = ? AND pae.runtime_profile_id = ? "
                    "AND pae.lifecycle_state = 'enabled' "
                    "JOIN channels direct_channel ON direct_channel.id = pae.direct_channel_id "
                    "AND direct_channel.kind = 'direct' AND direct_channel.workshop_id = c.workshop_id "
                    "WHERE c.id = ? AND c.kind = 'group'",
                    (
                        envelope.actor_principal_id,
                        sponsor_principal_id,
                        agent_id,
                        runtime_profile_id,
                        channel_id,
                    ),
                ) as cursor:
                    attachment_row = await cursor.fetchone()
                if (
                    attachment_row is None
                    or str(attachment_row[0]) != str(envelope.workshop_id)
                    or envelope.actor_principal_id != sponsor_principal_id
                ):
                    raise ValueError("Workshop channel-agent attachment has no enabled sponsorship")
            has_attachment_lifecycle = await _table_has_column(
                connection,
                "channel_agents",
                "detached_at",
            )
            attachment_columns = "id, detached_at" if has_attachment_lifecycle else "id, NULL"
            async with connection.execute(
                f"SELECT {attachment_columns} FROM channel_agents WHERE channel_id = ? AND agent_id = ?",
                (channel_id, agent_id),
            ) as cursor:
                existing_attachment = await cursor.fetchone()
            if existing_attachment is None:
                if has_attachment_lifecycle:
                    await connection.execute(
                        "INSERT INTO channel_agents "
                        "(id, channel_id, agent_id, created_at, sponsor_principal_id, "
                        "sponsored_runtime_profile_id, attached_event_position) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            envelope.aggregate_id,
                            channel_id,
                            agent_id,
                            occurred_at,
                            sponsor_principal_id,
                            runtime_profile_id,
                            event.position,
                        ),
                    )
                else:
                    await connection.execute(
                        "INSERT INTO channel_agents (id, channel_id, agent_id, created_at) VALUES (?, ?, ?, ?)",
                        (envelope.aggregate_id, channel_id, agent_id, occurred_at),
                    )
            elif existing_attachment[1] is not None and envelope.event_version == 2:
                cursor = await connection.execute(
                    "UPDATE channel_agents SET sponsor_principal_id = ?, "
                    "sponsored_runtime_profile_id = ?, attached_event_position = ?, "
                    "detached_at = NULL, detached_event_position = NULL "
                    "WHERE id = ? AND channel_id = ? AND agent_id = ?",
                    (
                        sponsor_principal_id,
                        runtime_profile_id,
                        event.position,
                        envelope.aggregate_id,
                        channel_id,
                        agent_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError("Workshop channel-agent reattachment does not match its history")
            else:
                raise ValueError("Workshop channel-agent is already attached")
        elif envelope.event_type == WorkshopEventType.CHANNEL_AGENT_DETACHED:
            if envelope.event_version != 1:
                raise ValueError("Unsupported Workshop channel-agent detachment event version")
            _require_exact_payload(
                payload,
                {"channel_id", "agent_id", "sponsor_principal_id", "runtime_profile_id"},
            )
            channel_id = ChannelId(_required_text(payload, "channel_id"))
            agent_id = AgentId(_required_text(payload, "agent_id"))
            sponsor_principal_id = PrincipalId(_required_text(payload, "sponsor_principal_id"))
            runtime_profile_id = _required_text(payload, "runtime_profile_id")
            if envelope.actor_principal_id is None:
                raise ValueError("Workshop channel-agent detachment requires a human actor")
            cursor = await connection.execute(
                "UPDATE channel_agents SET detached_at = ?, detached_event_position = ? "
                "WHERE id = ? AND channel_id = ? AND agent_id = ? "
                "AND sponsor_principal_id = ? AND sponsored_runtime_profile_id = ? "
                "AND detached_at IS NULL AND EXISTS ("
                "SELECT 1 FROM channels c JOIN channel_memberships owner "
                "ON owner.channel_id = c.id AND owner.principal_id = ? AND owner.role = 'owner' "
                "WHERE c.id = channel_agents.channel_id AND c.kind = 'group' "
                "AND c.workshop_id = ?)",
                (
                    occurred_at,
                    event.position,
                    envelope.aggregate_id,
                    channel_id,
                    agent_id,
                    sponsor_principal_id,
                    runtime_profile_id,
                    envelope.actor_principal_id,
                    envelope.workshop_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Workshop channel-agent detachment has no matching active attachment")
        elif envelope.event_type == WorkshopEventType.CHANNEL_AGENT_DISMISSED:
            _require_exact_payload(payload, {"agent_id", "thread_root_message_id"})
            agent_id = AgentId(_required_text(payload, "agent_id"))
            thread_root = payload.get("thread_root_message_id")
            if thread_root is not None:
                MessageId(str(thread_root))
            if envelope.actor_principal_id is None:
                raise ValueError("Workshop agent dismissal requires a human actor")
            async with connection.execute(
                "SELECT c.kind, c.workshop_id FROM channels c "
                "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.principal_id = ? "
                "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
                "JOIN channel_agents ca ON ca.channel_id = c.id AND ca.agent_id = ? "
                "WHERE c.id = ?",
                (envelope.actor_principal_id, agent_id, envelope.aggregate_id),
            ) as cursor:
                dismissal_row = await cursor.fetchone()
            if dismissal_row is None or tuple(dismissal_row) != ("group", envelope.workshop_id):
                raise ValueError("Workshop agent dismissal must target an attached group agent")
            if thread_root is not None:
                async with connection.execute(
                    "SELECT 1 FROM messages WHERE id = ? AND channel_id = ? AND thread_root_id IS NULL",
                    (thread_root, envelope.aggregate_id),
                ) as cursor:
                    if await cursor.fetchone() is None:
                        raise ValueError("Workshop agent dismissal must target a root message in its channel")
            await connection.execute(
                "INSERT INTO channel_agent_dismissals "
                "(id, channel_id, agent_id, dismissed_by_principal_id, "
                "thread_root_message_id, dismissed_at, created_event_position) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    envelope.event_id,
                    envelope.aggregate_id,
                    agent_id,
                    envelope.actor_principal_id,
                    thread_root,
                    occurred_at,
                    event.position,
                ),
            )
        elif envelope.event_type == WorkshopEventType.RUNTIME_PROFILE_ASSIGNED:
            await connection.execute(
                "INSERT INTO channel_agent_runtime_assignments "
                "(id, channel_id, agent_id, runtime_profile_id, created_at, created_event_position) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    _required_text(payload, "channel_id"),
                    _required_text(payload, "agent_id"),
                    _required_text(payload, "runtime_profile_id"),
                    occurred_at,
                    event.position,
                ),
            )
            if await _table_has_column(
                connection,
                "channel_agents",
                "sponsored_runtime_profile_id",
            ):
                await connection.execute(
                    "UPDATE channel_agents SET sponsored_runtime_profile_id = "
                    "COALESCE(sponsored_runtime_profile_id, ?), sponsor_principal_id = "
                    "COALESCE(sponsor_principal_id, ("
                    "SELECT owner.principal_id FROM channel_agent_runtime_assignments direct_ra "
                    "JOIN channels direct_channel ON direct_channel.id = direct_ra.channel_id "
                    "AND direct_channel.kind = 'direct' "
                    "JOIN channel_memberships owner ON owner.channel_id = direct_channel.id "
                    "AND owner.role = 'owner' "
                    "WHERE direct_ra.runtime_profile_id = ? AND direct_ra.agent_id = ? "
                    "ORDER BY direct_ra.created_event_position LIMIT 1)) "
                    "WHERE channel_id = ? AND agent_id = ?",
                    (
                        _required_text(payload, "runtime_profile_id"),
                        _required_text(payload, "runtime_profile_id"),
                        _required_text(payload, "agent_id"),
                        _required_text(payload, "channel_id"),
                        _required_text(payload, "agent_id"),
                    ),
                )
        elif envelope.event_type == WorkshopEventType.RUNTIME_PROFILE_REASSIGNED:
            channel_id = _required_text(payload, "channel_id")
            agent_id = _required_text(payload, "agent_id")
            runtime_profile_id = _required_text(payload, "runtime_profile_id")
            cursor = await connection.execute(
                "UPDATE channel_agent_runtime_assignments "
                "SET runtime_profile_id = ?, created_event_position = ? "
                "WHERE id = ? AND channel_id = ? AND agent_id = ?",
                (
                    runtime_profile_id,
                    event.position,
                    envelope.aggregate_id,
                    channel_id,
                    agent_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Workshop runtime reassignment has no matching channel-agent assignment")
            if await _table_has_column(
                connection,
                "channel_agents",
                "sponsored_runtime_profile_id",
            ):
                await connection.execute(
                    "UPDATE channel_agents SET sponsored_runtime_profile_id = ? WHERE channel_id = ? AND agent_id = ?",
                    (runtime_profile_id, channel_id, agent_id),
                )
        elif envelope.event_type == WorkshopEventType.MESSAGE_CREATED:
            reply_to = payload.get("reply_to_message_id")
            if reply_to is not None and not isinstance(reply_to, str):
                raise ValueError("Workshop reply_to_message_id must be a string or null")
            thread_root = payload.get("thread_root_id")
            if thread_root is not None and not isinstance(thread_root, str):
                raise ValueError("Workshop thread_root_id must be a string or null")
            channel_id = ChannelId(_required_text(payload, "channel_id"))
            if thread_root is not None:
                async with connection.execute(
                    "SELECT m.channel_id, m.thread_root_id, c.kind FROM messages m "
                    "JOIN channels c ON c.id = m.channel_id WHERE m.id = ?",
                    (thread_root,),
                ) as cursor:
                    root_row = await cursor.fetchone()
                if root_row is None or tuple(root_row) != (channel_id, None, "group"):
                    raise ValueError("Workshop thread root must be a top-level message in the same group channel")
            body = _required_text(payload, "body")
            mentions_json = await _message_mentions_json(
                connection,
                channel_id,
                body,
                payload.get("mentions", []),
            )
            await connection.execute(
                "INSERT INTO messages "
                "(id, channel_id, author_principal_id, reply_to_message_id, body, "
                "created_event_position, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    channel_id,
                    _required_text(payload, "author_principal_id"),
                    reply_to,
                    body,
                    event.position,
                    occurred_at,
                ),
            )
            # Empty mentions use the column default, which keeps the current
            # projection executable in migration tests frozen before schema
            # v41. Accepted mention spans require the v41 authority column.
            if mentions_json != "[]":
                await connection.execute(
                    "UPDATE messages SET mentions_json = ? WHERE id = ?",
                    (mentions_json, envelope.aggregate_id),
                )
            # Like mentions, the nullable thread field is updated separately
            # so old events remain projectable in migration tests frozen
            # before the v43 column exists.
            if thread_root is not None:
                await connection.execute(
                    "UPDATE messages SET thread_root_id = ? WHERE id = ?",
                    (thread_root, envelope.aggregate_id),
                )
        elif envelope.event_type in {
            WorkshopEventType.MESSAGE_REACTION_ADDED,
            WorkshopEventType.MESSAGE_REACTION_REMOVED,
        }:
            if not isinstance(envelope.aggregate_id, MessageId) or envelope.aggregate_type != "message":
                raise ValueError("Workshop message reaction requires a typed message aggregate")
            _require_exact_payload(payload, {"channel_id", "principal_id", "reaction"})
            channel_id = ChannelId(_required_text(payload, "channel_id"))
            principal_id = PrincipalId(_required_text(payload, "principal_id"))
            reaction = _required_text(payload, "reaction")
            if reaction not in {"thumbs_up", "heart", "laugh", "celebrate", "eyes", "check"}:
                raise ValueError("Unsupported Workshop message reaction")
            if envelope.actor_principal_id != principal_id:
                raise ValueError("Workshop message reaction actor must match its principal")
            async with connection.execute(
                "SELECT 1 FROM messages m "
                "JOIN channel_memberships cm ON cm.channel_id = m.channel_id "
                "AND cm.principal_id = ? "
                "WHERE m.id = ? AND m.channel_id = ?",
                (principal_id, envelope.aggregate_id, channel_id),
            ) as cursor:
                if await cursor.fetchone() is None:
                    raise ValueError("Workshop message reaction requires channel membership")
            if envelope.event_type == WorkshopEventType.MESSAGE_REACTION_ADDED:
                await connection.execute(
                    "INSERT INTO message_reactions "
                    "(message_id, principal_id, reaction, created_at, created_event_position) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (envelope.aggregate_id, principal_id, reaction, occurred_at, event.position),
                )
            else:
                cursor = await connection.execute(
                    "DELETE FROM message_reactions WHERE message_id = ? AND principal_id = ? AND reaction = ?",
                    (envelope.aggregate_id, principal_id, reaction),
                )
                if cursor.rowcount != 1:
                    raise ValueError("Workshop message reaction removal has no matching reaction")
        elif envelope.event_type == WorkshopEventType.ARTIFACT_CREATED:
            if envelope.event_version not in {1, 2}:
                raise ValueError("Unsupported Workshop artifact event version")
            created_by = _required_text(payload, "created_by_principal_id")
            if envelope.actor_principal_id != created_by:
                raise ValueError("Workshop artifact actor must match created_by_principal_id")
            channel_id = _required_text(payload, "channel_id")
            message_id = _required_text(payload, "message_id")
            async with connection.execute(
                "SELECT c.workshop_id, m.channel_id, m.author_principal_id, p.kind "
                "FROM messages m JOIN channels c ON c.id = m.channel_id "
                "JOIN principals p ON p.id = m.author_principal_id WHERE m.id = ?",
                (message_id,),
            ) as cursor:
                message_row = await cursor.fetchone()
            expected_author_kind = "human" if envelope.event_version == 1 else "agent"
            if message_row is None or tuple(message_row) != (
                envelope.workshop_id,
                channel_id,
                created_by,
                expected_author_kind,
            ):
                raise ValueError("Workshop artifact must belong to a message with the event-version author kind")
            kind = _required_text(payload, "kind")
            if kind not in {"photo", "document", "voice"}:
                raise ValueError("Workshop artifact kind is unsupported")
            byte_size = payload.get("byte_size")
            if not isinstance(byte_size, int) or isinstance(byte_size, bool) or byte_size < 0:
                raise ValueError("Workshop artifact byte_size must be a non-negative integer")
            content_sha256 = _required_text(payload, "content_sha256")
            if not _SHA256_PATTERN.fullmatch(content_sha256):
                raise ValueError("Workshop artifact content_sha256 must be lowercase SHA-256")
            original_filename = payload.get("original_filename")
            if original_filename is not None and (
                not isinstance(original_filename, str)
                or not original_filename
                or original_filename != original_filename.strip()
                or len(original_filename) > 255
                or original_filename in {".", ".."}
                or "/" in original_filename
                or "\\" in original_filename
                or "\0" in original_filename
            ):
                raise ValueError("Workshop artifact original_filename must be a bounded string or null")
            media_type = _required_text(payload, "media_type")
            if not _MEDIA_TYPE_PATTERN.fullmatch(media_type):
                raise ValueError("Workshop artifact media_type must be a lowercase MIME type")
            storage_path = _required_text(payload, "storage_path")
            if not Path(storage_path).is_absolute():
                raise ValueError("Workshop artifact storage_path must be absolute")
            source_transport = _required_text(payload, "source_transport")
            if not _IDENTIFIER_PATTERN.fullmatch(source_transport):
                raise ValueError("Workshop artifact source_transport must be a lowercase identifier")
            source_unique_id = _required_text(payload, "source_unique_id")
            if source_unique_id != source_unique_id.strip() or len(source_unique_id) > 512:
                raise ValueError("Workshop artifact source_unique_id must be bounded")
            await connection.execute(
                "INSERT INTO artifacts "
                "(id, workshop_id, channel_id, message_id, created_by_principal_id, kind, "
                "media_type, byte_size, content_sha256, original_filename, storage_path, "
                "source_transport, source_unique_id, created_event_position, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    envelope.workshop_id,
                    channel_id,
                    message_id,
                    created_by,
                    kind,
                    media_type,
                    byte_size,
                    content_sha256,
                    original_filename,
                    storage_path,
                    source_transport,
                    source_unique_id,
                    event.position,
                    occurred_at,
                ),
            )
        elif envelope.event_type in {
            WorkshopEventType.DELIVERY_SUCCEEDED,
            WorkshopEventType.DELIVERY_FAILED,
        }:
            status = "succeeded" if envelope.event_type == WorkshopEventType.DELIVERY_SUCCEEDED else "failed"
            message_id = _required_text(payload, "message_id")
            channel_id = _required_text(payload, "channel_id")
            transport = _required_text(payload, "transport")
            if envelope.event_version == 1:
                channel_binding_id = None
                async with connection.execute(
                    "SELECT 1 FROM messages WHERE id = ? AND channel_id = ?",
                    (message_id, channel_id),
                ) as cursor:
                    message_row = await cursor.fetchone()
                if message_row is None:
                    raise ValueError("Workshop delivery message must belong to its channel")
            elif envelope.event_version in {2, 3}:
                channel_binding_id = _required_text(payload, "channel_binding_id")
                async with connection.execute(
                    "SELECT 1 FROM messages m JOIN channel_bindings cb ON cb.channel_id = m.channel_id "
                    "WHERE m.id = ? AND m.channel_id = ? AND cb.id = ? AND cb.transport = ?",
                    (message_id, channel_id, channel_binding_id, transport),
                ) as cursor:
                    binding_row = await cursor.fetchone()
                if binding_row is None:
                    raise ValueError("Workshop delivery message and binding must belong to the same channel")
            else:
                raise ValueError("Workshop delivery event version is unsupported")
            await connection.execute(
                "INSERT INTO deliveries "
                "(id, message_id, channel_id, channel_binding_id, transport, mode, status, "
                "created_at, updated_at, last_event_position) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "status = excluded.status, updated_at = excluded.updated_at, "
                "last_event_position = excluded.last_event_position",
                (
                    envelope.aggregate_id,
                    message_id,
                    channel_id,
                    channel_binding_id,
                    transport,
                    _required_text(payload, "mode"),
                    status,
                    occurred_at,
                    occurred_at,
                    event.position,
                ),
            )
