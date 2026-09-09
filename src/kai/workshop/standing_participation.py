"""Canonical policy and subscription authority for standing Workshop agents."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import aiosqlite

from kai.workshop.agent_definitions import validate_collaboration_operations
from kai.workshop.collaboration_authority import (
    CollaborationHostPolicy,
    CollaborationOperation,
)
from kai.workshop.domain import (
    AgentDefinitionId,
    AgentDefinitionRevisionId,
    AgentId,
    ChannelId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.store import IdempotencyConflictError, Projection, StoredEvent, WorkshopEventStore

_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_END_REASONS = frozenset(
    {
        "dismissed",
        "detached",
        "definition_archived",
        "channel_archived",
        "channel_policy_disabled",
        "owner_policy_revoked",
        "host_policy_revoked",
        "quiet_expired",
        "overflow",
        "access_removed",
    }
)


class WorkshopStandingParticipationError(RuntimeError):
    """Base error for standing-participation authority."""


class WorkshopStandingParticipationAccessDenied(WorkshopStandingParticipationError):
    """The principal cannot inspect or mutate the requested channel."""


class WorkshopStandingParticipationValidationError(WorkshopStandingParticipationError):
    """The proposed standing-participation mutation is invalid."""


class WorkshopStandingParticipationConflict(WorkshopStandingParticipationError):
    """Canonical standing state changed or an idempotency key was reused."""


class WorkshopStandingParticipationStorageError(WorkshopStandingParticipationError):
    """Canonical standing state could not be persisted."""


@dataclass(frozen=True, slots=True)
class ChannelStandingPolicy:
    channel_id: ChannelId
    enabled: bool
    policy_version: int
    can_manage: bool
    host_enabled: bool
    host_policy_version: int
    max_agents_per_channel: int
    coalescing_grace_seconds: int
    max_messages_per_observe_run: int
    max_pending_messages_per_scope: int
    max_pending_age_seconds: int
    max_observe_runs_per_hour: int
    max_unsolicited_messages_per_hour: int
    minimum_unsolicited_interval_seconds: int
    quiet_expiry_seconds: int


@dataclass(frozen=True, slots=True)
class StandingSubscription:
    channel_id: ChannelId
    agent_id: AgentId
    agent_definition_id: AgentDefinitionId
    agent_definition_revision_id: AgentDefinitionRevisionId
    lifecycle_state: str
    started_by_principal_id: PrincipalId
    started_by_message_id: MessageId
    owner_policy_version: int
    channel_policy_version: int
    host_policy_version: int
    quiet_expires_at: datetime | None
    state_version: int
    end_reason: str | None
    agent_display_name: str
    agent_handle: str
    started_at: datetime


@dataclass(frozen=True, slots=True)
class StandingObservationSummary:
    agent_id: AgentId
    scope_kind: str
    scope_id: str
    delivered_through_event_position: int
    pending_through_event_position: int
    considered_through_event_position: int
    pending_message_count: int
    not_before: datetime | None
    lifecycle_state: str
    overflowed_at: datetime | None
    overflow_reason: str | None
    overflow_from_event_position: int | None
    overflow_through_event_position: int | None
    state_version: int


@dataclass(frozen=True, slots=True)
class StandingRunSummary:
    run_id: str
    agent_id: AgentId
    agent_display_name: str
    status: str
    accepted_at: datetime
    started_at: datetime | None
    terminal_at: datetime | None
    terminal_code: str | None
    scope_kind: str
    scope_id: str
    observed_from_event_position: int
    observed_through_event_position: int
    observed_message_count: int
    human_anchor_message_id: MessageId
    collaboration_grant_id: str | None
    outcome: str | None


@dataclass(frozen=True, slots=True)
class ChannelStandingSnapshot:
    policy: ChannelStandingPolicy
    subscriptions: tuple[StandingSubscription, ...]
    observations: tuple[StandingObservationSummary, ...] = ()
    recent_runs: tuple[StandingRunSummary, ...] = ()
    can_inspect_silent: bool = False


@dataclass(frozen=True, slots=True)
class ChannelStandingPolicyMutation:
    snapshot: ChannelStandingSnapshot
    changed: bool
    replayed: bool


def _request_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(UTC)


async def apply_standing_participation_event(
    connection: aiosqlite.Connection,
    event: StoredEvent,
) -> None:
    """Apply one validated standing-participation event to current projections."""
    envelope = event.envelope
    payload = envelope.payload
    occurred_at = envelope.occurred_at.isoformat()
    if not isinstance(envelope.aggregate_id, ChannelId) or envelope.aggregate_type != "channel":
        raise ValueError("Standing-participation events require a channel aggregate")
    if envelope.event_version != 1:
        raise ValueError("Unsupported standing-participation event version")
    channel_id = envelope.aggregate_id

    if envelope.event_type == WorkshopEventType.CHANNEL_STANDING_PARTICIPATION_POLICY_SET:
        if set(payload) != {"enabled", "expected_policy_version", "policy_version", "host_policy_version"}:
            raise ValueError("Standing-participation policy payload is invalid")
        enabled = payload["enabled"]
        expected = payload["expected_policy_version"]
        version = payload["policy_version"]
        host_version = payload["host_policy_version"]
        if (
            not isinstance(enabled, bool)
            or not isinstance(expected, int)
            or isinstance(expected, bool)
            or expected < 0
            or not isinstance(version, int)
            or isinstance(version, bool)
            or version != expected + 1
            or not isinstance(host_version, int)
            or isinstance(host_version, bool)
            or host_version < 1
        ):
            raise ValueError("Standing-participation policy values are invalid")
        async with connection.execute(
            "SELECT 1 FROM channels c JOIN channel_memberships cm ON cm.channel_id = c.id "
            "AND cm.principal_id = ? AND cm.role = 'owner' JOIN principals p ON p.id = cm.principal_id "
            "AND p.kind = 'human' WHERE c.id = ? AND c.workshop_id = ? AND c.kind = 'group'",
            (envelope.actor_principal_id, channel_id, envelope.workshop_id),
        ) as cursor:
            if await cursor.fetchone() is None:
                raise ValueError("Standing-participation policy must be set by a channel owner")
        async with connection.execute(
            "SELECT policy_version FROM channel_standing_participation_policies WHERE channel_id = ?",
            (channel_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if (int(row[0]) if row else 0) != expected:
            raise ValueError("Standing-participation policy changed concurrently")
        await connection.execute(
            "INSERT INTO channel_standing_participation_policies "
            "(channel_id, enabled, policy_version, updated_by_principal_id, host_policy_version, "
            "updated_at, updated_event_position) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(channel_id) DO UPDATE SET enabled = excluded.enabled, "
            "policy_version = excluded.policy_version, updated_by_principal_id = excluded.updated_by_principal_id, "
            "host_policy_version = excluded.host_policy_version, updated_at = excluded.updated_at, "
            "updated_event_position = excluded.updated_event_position",
            (channel_id, int(enabled), version, envelope.actor_principal_id, host_version, occurred_at, event.position),
        )
        return

    if envelope.event_type == WorkshopEventType.CHANNEL_AGENT_STANDING_STARTED:
        required = {
            "agent_id",
            "agent_definition_id",
            "agent_revision_id",
            "started_by_principal_id",
            "started_by_message_id",
            "owner_policy_version",
            "channel_policy_version",
            "host_policy_version",
            "quiet_expires_at",
        }
        if set(payload) != required:
            raise ValueError("Standing-subscription start payload is invalid")
        agent_id = AgentId(str(payload["agent_id"]))
        definition_id = AgentDefinitionId(str(payload["agent_definition_id"]))
        revision_id = AgentDefinitionRevisionId(str(payload["agent_revision_id"]))
        starter = PrincipalId(str(payload["started_by_principal_id"]))
        message_id = MessageId(str(payload["started_by_message_id"]))
        if envelope.actor_principal_id != starter:
            raise ValueError("Standing-subscription starter does not match the event actor")
        for field in ("owner_policy_version", "channel_policy_version", "host_policy_version"):
            value = payload[field]
            minimum = 0 if field == "owner_policy_version" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError("Standing-subscription policy evidence is invalid")
        quiet = payload["quiet_expires_at"]
        if quiet is not None:
            _timestamp(quiet, field="quiet_expires_at")
        async with connection.execute(
            "SELECT m.mentions_json FROM messages m JOIN principals p ON p.id = m.author_principal_id "
            "AND p.kind = 'human' WHERE m.id = ? AND m.channel_id = ? AND m.author_principal_id = ?",
            (message_id, channel_id, starter),
        ) as cursor:
            message_row = await cursor.fetchone()
            if message_row is None:
                raise ValueError("Standing participation must start from a human message in the channel")
        async with connection.execute(
            "SELECT a.principal_id, r.collaboration_operations_json FROM channels c "
            "JOIN channel_agents ca ON ca.channel_id = c.id AND ca.agent_id = ? "
            "JOIN agents a ON a.id = ca.agent_id "
            "AND ca.detached_at IS NULL JOIN agent_definitions d ON d.id = ? AND d.agent_id = ca.agent_id "
            "AND d.lifecycle_state = 'active' AND d.active_revision_id = ? "
            "JOIN agent_definition_revisions r ON r.id = d.active_revision_id WHERE c.id = ? "
            "AND c.workshop_id = ? AND c.kind = 'group' AND c.archived_at IS NULL",
            (agent_id, definition_id, revision_id, channel_id, envelope.workshop_id),
        ) as cursor:
            agent_row = await cursor.fetchone()
            if agent_row is None:
                raise ValueError("Standing subscription requires an active attached agent revision")
        mentions = json.loads(str(message_row[0]))
        if not any(
            isinstance(item, dict)
            and item.get("kind") == "agent"
            and str(item.get("principal_id")) == str(agent_row[0])
            for item in mentions
        ):
            raise ValueError("Standing participation must start from an explicit agent mention")
        if "standing_participation" not in validate_collaboration_operations(json.loads(str(agent_row[1]))):
            raise ValueError("Standing agent revision does not request standing participation")
        async with connection.execute(
            "SELECT enabled, policy_version FROM channel_standing_participation_policies WHERE channel_id = ?",
            (channel_id,),
        ) as cursor:
            channel_policy = await cursor.fetchone()
        if (
            channel_policy is None
            or not bool(channel_policy[0])
            or int(channel_policy[1]) != payload["channel_policy_version"]
        ):
            raise ValueError("Standing channel policy evidence is not current")
        async with connection.execute(
            "SELECT policy_version, allowed_operations_json FROM agent_collaboration_owner_policies "
            "WHERE agent_definition_id = ?",
            (definition_id,),
        ) as cursor:
            owner_policy = await cursor.fetchone()
        if (
            owner_policy is None
            or int(owner_policy[0]) != payload["owner_policy_version"]
            or "standing_participation" not in validate_collaboration_operations(json.loads(str(owner_policy[1])))
        ):
            raise ValueError("Standing owner-policy evidence is not current")
        await connection.execute(
            "INSERT INTO channel_agent_standings "
            "(channel_id, agent_id, agent_definition_id, agent_definition_revision_id, "
            "started_by_principal_id, started_by_message_id, owner_policy_version, channel_policy_version, "
            "host_policy_version, lifecycle_state, quiet_expires_at, started_at, started_event_position, "
            "ended_at, end_reason, cause_event_id, ended_event_position, state_version, last_event_position) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, NULL, NULL, NULL, NULL, 1, ?) "
            "ON CONFLICT(channel_id, agent_id) DO UPDATE SET agent_definition_id = excluded.agent_definition_id, "
            "agent_definition_revision_id = excluded.agent_definition_revision_id, "
            "started_by_principal_id = excluded.started_by_principal_id, "
            "started_by_message_id = excluded.started_by_message_id, owner_policy_version = excluded.owner_policy_version, "
            "channel_policy_version = excluded.channel_policy_version, host_policy_version = excluded.host_policy_version, "
            "lifecycle_state = 'active', quiet_expires_at = excluded.quiet_expires_at, "
            "started_at = excluded.started_at, started_event_position = excluded.started_event_position, "
            "ended_at = NULL, end_reason = NULL, cause_event_id = NULL, ended_event_position = NULL, "
            "state_version = channel_agent_standings.state_version + 1, last_event_position = excluded.last_event_position",
            (
                channel_id,
                agent_id,
                definition_id,
                revision_id,
                starter,
                message_id,
                payload["owner_policy_version"],
                payload["channel_policy_version"],
                payload["host_policy_version"],
                quiet,
                occurred_at,
                event.position,
                event.position,
            ),
        )
        return

    if envelope.event_type == WorkshopEventType.CHANNEL_AGENT_STANDING_ENDED:
        if set(payload) != {"agent_id", "reason", "cause_event_id", "ended_by_principal_id"}:
            raise ValueError("Standing-subscription end payload is invalid")
        agent_id = AgentId(str(payload["agent_id"]))
        reason = str(payload["reason"])
        if reason not in _END_REASONS:
            raise ValueError("Standing-subscription end reason is invalid")
        ended_by = payload["ended_by_principal_id"]
        if ended_by is not None and PrincipalId(str(ended_by)) != envelope.actor_principal_id:
            raise ValueError("Standing-subscription end actor is invalid")
        async with connection.execute(
            "SELECT lifecycle_state FROM channel_agent_standings WHERE channel_id = ? AND agent_id = ?",
            (channel_id, agent_id),
        ) as cursor:
            current = await cursor.fetchone()
        if current is None or str(current[0]) != "active":
            raise ValueError("Only an active standing subscription may end")
        lifecycle_state = "paused_overflow" if reason == "overflow" else "ended"
        await connection.execute(
            "UPDATE channel_agent_standings SET lifecycle_state = ?, ended_at = ?, end_reason = ?, "
            "cause_event_id = ?, ended_event_position = ?, state_version = state_version + 1, "
            "last_event_position = ? WHERE channel_id = ? AND agent_id = ?",
            (
                lifecycle_state,
                occurred_at,
                reason,
                payload["cause_event_id"],
                event.position,
                event.position,
                channel_id,
                agent_id,
            ),
        )
        return
    if envelope.event_type == WorkshopEventType.CHANNEL_AGENT_STANDING_OBSERVATION_RESUMED:
        required = {
            "agent_id",
            "scope_id",
            "expected_state_version",
            "state_version",
            "resumed_through_event_position",
        }
        if set(payload) != required:
            raise ValueError("Standing observation resume payload is invalid")
        agent_id = AgentId(str(payload["agent_id"]))
        scope_id = str(payload["scope_id"])
        expected = payload["expected_state_version"]
        version = payload["state_version"]
        through = payload["resumed_through_event_position"]
        if (
            not scope_id
            or not isinstance(expected, int)
            or isinstance(expected, bool)
            or expected < 1
            or not isinstance(version, int)
            or isinstance(version, bool)
            or version != expected + 1
            or not isinstance(through, int)
            or isinstance(through, bool)
            or through < 0
        ):
            raise ValueError("Standing observation resume values are invalid")
        async with connection.execute(
            "SELECT o.lifecycle_state, o.state_version, o.considered_through_event_position "
            "FROM channel_agent_observation_states o JOIN channel_agent_standings s "
            "ON s.channel_id = o.channel_id AND s.agent_id = o.agent_id "
            "JOIN channel_memberships cm ON cm.channel_id = o.channel_id "
            "AND cm.principal_id = ? JOIN principals p ON p.id = cm.principal_id "
            "AND p.kind = 'human' WHERE o.channel_id = ? AND o.agent_id = ? AND o.scope_id = ? "
            "AND s.lifecycle_state = 'active'",
            (envelope.actor_principal_id, channel_id, agent_id, scope_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None or str(row[0]) != "paused_overflow" or int(row[1]) != expected or int(row[2]) != through:
            raise ValueError("Standing observation resume state is stale")
        await connection.execute(
            "UPDATE channel_agent_observation_states SET delivered_through_event_position = ?, "
            "pending_through_event_position = ?, oldest_pending_event_position = NULL, "
            "pending_message_count = 0, not_before = NULL, lifecycle_state = 'idle', "
            "overflowed_at = NULL, overflow_reason = NULL, overflow_from_event_position = NULL, "
            "overflow_through_event_position = NULL, state_version = ?, last_event_position = ? "
            "WHERE channel_id = ? AND agent_id = ? AND scope_id = ?",
            (through, through, version, event.position, channel_id, agent_id, scope_id),
        )
        return
    raise ValueError("Unsupported standing-participation event")


class WorkshopStandingParticipationService:
    """Manage channel opt-in and durable standing subscriptions."""

    def __init__(
        self,
        store: WorkshopEventStore,
        host_policy: CollaborationHostPolicy,
    ) -> None:
        self._store = store
        self._host_policy = host_policy

    async def synchronize_host_policy(self) -> None:
        """Publish current host limits for the universal message projection."""
        from kai.workshop.standing_observation import (
            synchronize_standing_observation_host_policy_in_transaction,
        )

        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await synchronize_standing_observation_host_policy_in_transaction(
                connection,
                self._host_policy,
                occurred_at=datetime.now(UTC),
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

    @staticmethod
    def _projection() -> Projection:
        from kai.workshop.projection import CanonicalConversationProjection

        return CanonicalConversationProjection()

    async def inspect(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        *,
        include_silent: bool = False,
    ) -> ChannelStandingSnapshot:
        if not isinstance(include_silent, bool):
            raise WorkshopStandingParticipationValidationError("include_silent must be a boolean")
        workshop_id, can_manage = await self._channel_access_row(principal_id, channel_id)
        await self._reconcile_current(channel_id)
        async with self._store.connection.execute(
            "SELECT enabled, policy_version FROM channel_standing_participation_policies WHERE channel_id = ?",
            (channel_id,),
        ) as cursor:
            policy_row = await cursor.fetchone()
        host = self._host_policy.standing_participation
        policy = ChannelStandingPolicy(
            channel_id=channel_id,
            enabled=bool(policy_row[0]) if policy_row else False,
            policy_version=int(policy_row[1]) if policy_row else 0,
            can_manage=can_manage,
            host_enabled=host.enabled,
            host_policy_version=self._host_policy.version,
            max_agents_per_channel=host.max_agents_per_channel,
            coalescing_grace_seconds=host.coalescing_grace_seconds,
            max_messages_per_observe_run=host.max_messages_per_observe_run,
            max_pending_messages_per_scope=host.max_pending_messages_per_scope,
            max_pending_age_seconds=host.max_pending_age_seconds,
            max_observe_runs_per_hour=host.max_observe_runs_per_hour,
            max_unsolicited_messages_per_hour=host.max_unsolicited_messages_per_hour,
            minimum_unsolicited_interval_seconds=host.minimum_unsolicited_interval_seconds,
            quiet_expiry_seconds=host.quiet_expiry_seconds,
        )
        async with self._store.connection.execute(
            "SELECT s.agent_id, s.agent_definition_id, s.agent_definition_revision_id, s.lifecycle_state, "
            "s.started_by_principal_id, s.started_by_message_id, s.owner_policy_version, s.channel_policy_version, "
            "s.host_policy_version, s.quiet_expires_at, s.state_version, s.end_reason, "
            "d.display_name, d.handle, s.started_at "
            "FROM channel_agent_standings s JOIN agent_definitions d ON d.id = s.agent_definition_id "
            "WHERE s.channel_id = ? ORDER BY s.started_event_position, s.agent_id",
            (channel_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        subscriptions = tuple(
            StandingSubscription(
                channel_id,
                AgentId(str(row[0])),
                AgentDefinitionId(str(row[1])),
                AgentDefinitionRevisionId(str(row[2])),
                str(row[3]),
                PrincipalId(str(row[4])),
                MessageId(str(row[5])),
                int(row[6]),
                int(row[7]),
                int(row[8]),
                _timestamp(row[9], field="quiet_expires_at") if row[9] is not None else None,
                int(row[10]),
                str(row[11]) if row[11] is not None else None,
                str(row[12]),
                str(row[13]),
                _timestamp(row[14], field="started_at"),
            )
            for row in rows
        )
        async with self._store.connection.execute(
            "SELECT agent_id, scope_kind, scope_id, delivered_through_event_position, "
            "pending_through_event_position, considered_through_event_position, pending_message_count, "
            "not_before, lifecycle_state, overflowed_at, overflow_reason, overflow_from_event_position, "
            "overflow_through_event_position, state_version FROM channel_agent_observation_states "
            "WHERE channel_id = ? ORDER BY agent_id, scope_kind, scope_id",
            (channel_id,),
        ) as cursor:
            observation_rows = list(await cursor.fetchall())
        observations = tuple(
            StandingObservationSummary(
                AgentId(str(row[0])),
                str(row[1]),
                str(row[2]),
                int(row[3]),
                int(row[4]),
                int(row[5]),
                int(row[6]),
                _timestamp(row[7], field="not_before") if row[7] is not None else None,
                str(row[8]),
                _timestamp(row[9], field="overflowed_at") if row[9] is not None else None,
                str(row[10]) if row[10] is not None else None,
                int(row[11]) if row[11] is not None else None,
                int(row[12]) if row[12] is not None else None,
                int(row[13]),
            )
            for row in observation_rows
        )
        async with self._store.connection.execute(
            "SELECT wm.role = 'admin' FROM workshop_memberships wm WHERE wm.workshop_id = ? AND wm.principal_id = ?",
            (workshop_id, principal_id),
        ) as cursor:
            admin_row = await cursor.fetchone()
        is_admin = bool(admin_row[0]) if admin_row is not None else False
        async with self._store.connection.execute(
            "SELECT EXISTS(SELECT 1 FROM channel_agents ca JOIN agent_definitions d ON d.agent_id = ca.agent_id "
            "WHERE ca.channel_id = ? AND ca.detached_at IS NULL AND d.owner_principal_id = ?)",
            (channel_id, principal_id),
        ) as cursor:
            owner_row = await cursor.fetchone()
        can_inspect_silent = is_admin or bool(owner_row[0] if owner_row is not None else False)
        async with self._store.connection.execute(
            "SELECT r.id, r.agent_id, d.display_name, r.status, r.accepted_at, r.started_at, r.terminal_at, "
            "r.terminal_code, r.observation_scope_kind, r.observation_scope_id, "
            "r.observed_from_event_position, r.observed_through_event_position, "
            "json_array_length(r.observed_message_ids_json), r.human_anchor_message_id, "
            "(SELECT g.id FROM collaboration_grants g WHERE g.run_id = r.id "
            "ORDER BY g.issued_event_position DESC LIMIT 1), r.standing_outcome, d.owner_principal_id "
            "FROM runs r JOIN agents a ON a.id = r.agent_id "
            "JOIN agent_definition_revisions rev ON rev.id = r.agent_definition_revision_id "
            "JOIN agent_definitions d ON d.id = rev.agent_definition_id AND d.agent_id = a.id "
            "WHERE r.channel_id = ? AND r.kind = 'observe' ORDER BY r.accepted_at DESC, r.id DESC LIMIT 50",
            (channel_id,),
        ) as cursor:
            run_rows = list(await cursor.fetchall())
        recent_runs = tuple(
            StandingRunSummary(
                run_id=str(row[0]),
                agent_id=AgentId(str(row[1])),
                agent_display_name=str(row[2]),
                status=str(row[3]),
                accepted_at=_timestamp(row[4], field="accepted_at"),
                started_at=_timestamp(row[5], field="started_at") if row[5] is not None else None,
                terminal_at=_timestamp(row[6], field="terminal_at") if row[6] is not None else None,
                terminal_code=str(row[7]) if row[7] is not None else None,
                scope_kind=str(row[8]),
                scope_id=str(row[9]),
                observed_from_event_position=int(row[10]),
                observed_through_event_position=int(row[11]),
                observed_message_count=int(row[12]),
                human_anchor_message_id=MessageId(str(row[13])),
                collaboration_grant_id=str(row[14]) if row[14] is not None else None,
                outcome=str(row[15]) if row[15] is not None else None,
            )
            for row in run_rows
            if str(row[15]) != "silent" or (include_silent and (is_admin or str(row[16]) == str(principal_id)))
        )
        return ChannelStandingSnapshot(policy, subscriptions, observations, recent_runs, can_inspect_silent)

    async def resume_observation(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        agent_id: AgentId,
        *,
        scope_id: object,
        expected_state_version: object,
        client_operation_id: object,
    ) -> ChannelStandingSnapshot:
        if not isinstance(scope_id, str) or not scope_id or len(scope_id) > 128:
            raise WorkshopStandingParticipationValidationError("scope_id is invalid")
        if (
            not isinstance(expected_state_version, int)
            or isinstance(expected_state_version, bool)
            or expected_state_version < 1
        ):
            raise WorkshopStandingParticipationValidationError("expected_state_version is invalid")
        if not isinstance(client_operation_id, str) or not _OPERATION_ID.fullmatch(client_operation_id):
            raise WorkshopStandingParticipationValidationError("client_operation_id is invalid")
        workshop_id, _can_manage = await self._channel_access_row(principal_id, channel_id)
        connection = self._store.connection
        request = {
            "agent_id": str(agent_id),
            "scope_id": scope_id,
            "expected_state_version": expected_state_version,
        }
        digest = _request_hash(request)
        key = f"standing-observation-resume:v1:{channel_id}:{agent_id}:{scope_id}:{client_operation_id}"
        try:
            await connection.execute("BEGIN IMMEDIATE")
            existing = await self._store.event_by_idempotency_key(key)
            if existing is not None:
                if existing.envelope.metadata.get("request_hash") != digest:
                    raise WorkshopStandingParticipationConflict("Operation identity was reused with different content")
                await connection.rollback()
                return await self.inspect(principal_id, channel_id)
            async with connection.execute(
                "SELECT considered_through_event_position, state_version, lifecycle_state "
                "FROM channel_agent_observation_states WHERE channel_id = ? AND agent_id = ? AND scope_id = ?",
                (channel_id, agent_id, scope_id),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None or str(row[2]) != "paused_overflow":
                raise WorkshopStandingParticipationConflict("Standing observation is not paused for overflow")
            if int(row[1]) != expected_state_version:
                raise WorkshopStandingParticipationConflict("Standing observation changed; refresh and retry")
            through = int(row[0])
            event = EventEnvelope.create(
                event_id=EventId.derived(
                    channel_id,
                    f"standing-observation-resume:{agent_id}:{scope_id}:{client_operation_id}",
                ),
                event_type=WorkshopEventType.CHANNEL_AGENT_STANDING_OBSERVATION_RESUMED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="channel",
                aggregate_id=channel_id,
                actor_principal_id=principal_id,
                occurred_at=datetime.now(UTC),
                idempotency_key=key,
                payload={
                    "agent_id": agent_id,
                    "scope_id": scope_id,
                    "expected_state_version": expected_state_version,
                    "state_version": expected_state_version + 1,
                    "resumed_through_event_position": through,
                },
                metadata={"source": "workshop_client", "request_hash": digest},
            )
            result = await self._store.append_in_transaction(event)
            if result.inserted:
                await self._store.project_pending_in_transaction(self._projection())
            await connection.commit()
        except WorkshopStandingParticipationError:
            await connection.rollback()
            raise
        except IdempotencyConflictError as exc:
            await connection.rollback()
            raise WorkshopStandingParticipationConflict("Operation identity conflicted") from exc
        except aiosqlite.Error as exc:
            await connection.rollback()
            raise WorkshopStandingParticipationStorageError("Standing observation could not be resumed") from exc
        return await self.inspect(principal_id, channel_id)

    async def set_channel_policy(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        *,
        enabled: object,
        expected_policy_version: object,
        client_operation_id: object,
    ) -> ChannelStandingPolicyMutation:
        if not isinstance(enabled, bool):
            raise WorkshopStandingParticipationValidationError("enabled must be a boolean")
        if (
            not isinstance(expected_policy_version, int)
            or isinstance(expected_policy_version, bool)
            or expected_policy_version < 0
        ):
            raise WorkshopStandingParticipationValidationError("expected_policy_version must be non-negative")
        if not isinstance(client_operation_id, str) or not _OPERATION_ID.fullmatch(client_operation_id):
            raise WorkshopStandingParticipationValidationError("client_operation_id is invalid")
        connection = self._store.connection
        request = {
            "enabled": enabled,
            "expected_policy_version": expected_policy_version,
        }
        digest = _request_hash(request)
        try:
            await connection.execute("BEGIN IMMEDIATE")
            from kai.workshop.standing_observation import (
                synchronize_standing_observation_host_policy_in_transaction,
            )

            await synchronize_standing_observation_host_policy_in_transaction(
                connection,
                self._host_policy,
                occurred_at=datetime.now(UTC),
            )
            workshop_id, can_manage = await self._channel_access_row(principal_id, channel_id)
            if not can_manage:
                raise WorkshopStandingParticipationAccessDenied("Only a channel owner may change this policy")
            key = f"workshop-client:standing-policy:{principal_id}:{client_operation_id}"
            existing = await self._store.event_by_idempotency_key(key)
            if existing is not None:
                if existing.envelope.metadata.get("request_hash") != digest:
                    raise WorkshopStandingParticipationConflict("Operation identity was reused with different content")
                await connection.rollback()
                return ChannelStandingPolicyMutation(await self.inspect(principal_id, channel_id), False, True)
            async with connection.execute(
                "SELECT enabled, policy_version FROM channel_standing_participation_policies WHERE channel_id = ?",
                (channel_id,),
            ) as cursor:
                current = await cursor.fetchone()
            version = int(current[1]) if current else 0
            if version != expected_policy_version:
                raise WorkshopStandingParticipationConflict("Standing policy changed; refresh and retry")
            current_enabled = bool(current[0]) if current else False
            now = datetime.now(UTC)
            event = EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_STANDING_PARTICIPATION_POLICY_SET,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="channel",
                aggregate_id=channel_id,
                actor_principal_id=principal_id,
                occurred_at=now,
                idempotency_key=key,
                payload={
                    "enabled": enabled,
                    "expected_policy_version": version,
                    "policy_version": version + 1,
                    "host_policy_version": self._host_policy.version,
                },
                metadata={"source": "workshop_client", "request_hash": digest},
            )
            appended = await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(self._projection())
            if not enabled:
                await self._end_channel_in_transaction(
                    channel_id,
                    workshop_id=workshop_id,
                    reason="channel_policy_disabled",
                    cause_event_id=appended.event.envelope.event_id,
                    actor_principal_id=principal_id,
                    occurred_at=now,
                )
            await connection.commit()
        except WorkshopStandingParticipationError:
            await connection.rollback()
            raise
        except IdempotencyConflictError as exc:
            await connection.rollback()
            raise WorkshopStandingParticipationConflict("Operation identity conflicted") from exc
        except Exception as exc:
            await connection.rollback()
            raise WorkshopStandingParticipationStorageError("Standing policy could not be persisted") from exc
        return ChannelStandingPolicyMutation(
            await self.inspect(principal_id, channel_id), current_enabled != enabled, False
        )

    async def start_from_message_in_transaction(
        self,
        message_id: MessageId,
        agent_ids: tuple[AgentId, ...],
        *,
        occurred_at: datetime,
    ) -> None:
        from kai.workshop.standing_observation import (
            synchronize_standing_observation_host_policy_in_transaction,
        )

        await synchronize_standing_observation_host_policy_in_transaction(
            self._store.connection,
            self._host_policy,
            occurred_at=occurred_at,
        )
        if CollaborationOperation.STANDING_PARTICIPATION not in self._host_policy.effective_allowed_operations:
            return
        async with self._store.connection.execute(
            "SELECT m.channel_id, c.workshop_id, c.kind, c.archived_at, m.author_principal_id, p.kind, "
            "m.mentions_json FROM messages m JOIN channels c ON c.id = m.channel_id "
            "JOIN principals p ON p.id = m.author_principal_id WHERE m.id = ?",
            (message_id,),
        ) as cursor:
            message = await cursor.fetchone()
        if message is None or tuple(message[2:4]) != ("group", None) or str(message[5]) != "human":
            return
        channel_id = ChannelId(str(message[0]))
        workshop_id = WorkshopId(str(message[1]))
        starter = PrincipalId(str(message[4]))
        await self._reconcile_current_in_transaction(channel_id, occurred_at=occurred_at)
        mentions = json.loads(str(message[6]))
        mentioned_principals = {
            str(item.get("principal_id")) for item in mentions if isinstance(item, dict) and item.get("kind") == "agent"
        }
        async with self._store.connection.execute(
            "SELECT enabled, policy_version FROM channel_standing_participation_policies WHERE channel_id = ?",
            (channel_id,),
        ) as cursor:
            policy = await cursor.fetchone()
        if policy is None or not bool(policy[0]):
            return
        active = await self._active_count(channel_id)
        for agent_id in agent_ids:
            if active >= self._host_policy.standing_participation.max_agents_per_channel:
                break
            async with self._store.connection.execute(
                "SELECT a.principal_id, d.id, d.active_revision_id, r.collaboration_operations_json, "
                "coalesce(op.policy_version, 0), op.allowed_operations_json "
                "FROM channel_agents ca JOIN agents a ON a.id = ca.agent_id "
                "JOIN agent_definitions d ON d.agent_id = ca.agent_id AND d.lifecycle_state = 'active' "
                "JOIN agent_definition_revisions r ON r.id = d.active_revision_id "
                "LEFT JOIN agent_collaboration_owner_policies op ON op.agent_definition_id = d.id "
                "WHERE ca.channel_id = ? AND ca.agent_id = ? AND ca.detached_at IS NULL",
                (channel_id, agent_id),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None or str(row[0]) not in mentioned_principals:
                continue
            requested = validate_collaboration_operations(json.loads(str(row[3])))
            allowed = validate_collaboration_operations(json.loads(str(row[5]))) if row[5] is not None else ()
            if "standing_participation" not in requested or "standing_participation" not in allowed:
                continue
            async with self._store.connection.execute(
                "SELECT lifecycle_state FROM channel_agent_standings WHERE channel_id = ? AND agent_id = ?",
                (channel_id, agent_id),
            ) as cursor:
                existing = await cursor.fetchone()
            if existing is not None and str(existing[0]) == "active":
                continue
            quiet_seconds = self._host_policy.standing_participation.quiet_expiry_seconds
            quiet_expires_at = occurred_at + timedelta(seconds=quiet_seconds) if quiet_seconds else None
            event = EventEnvelope.create(
                event_id=EventId.derived(message_id, f"standing:{agent_id}"),
                event_type=WorkshopEventType.CHANNEL_AGENT_STANDING_STARTED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="channel",
                aggregate_id=channel_id,
                actor_principal_id=starter,
                occurred_at=occurred_at,
                idempotency_key=f"standing-start:{message_id}:{agent_id}",
                payload={
                    "agent_id": agent_id,
                    "agent_definition_id": AgentDefinitionId(str(row[1])),
                    "agent_revision_id": AgentDefinitionRevisionId(str(row[2])),
                    "started_by_principal_id": starter,
                    "started_by_message_id": message_id,
                    "owner_policy_version": int(row[4]),
                    "channel_policy_version": int(policy[1]),
                    "host_policy_version": self._host_policy.version,
                    "quiet_expires_at": quiet_expires_at.isoformat() if quiet_expires_at else None,
                },
                metadata={"source": "canonical_mention"},
            )
            result = await self._store.append_in_transaction(event)
            if result.inserted:
                await self._store.project_pending_in_transaction(self._projection())
                active += 1

    async def _active_count(self, channel_id: ChannelId) -> int:
        async with self._store.connection.execute(
            "SELECT COUNT(*) FROM channel_agent_standings WHERE channel_id = ? AND lifecycle_state = 'active'",
            (channel_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        return int(row[0])

    async def _reconcile_current(self, channel_id: ChannelId) -> None:
        now = datetime.now(UTC)
        connection = self._store.connection
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await self._reconcile_current_in_transaction(channel_id, occurred_at=now)
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

    async def _reconcile_current_in_transaction(
        self,
        channel_id: ChannelId,
        *,
        occurred_at: datetime,
    ) -> None:
        async with self._store.connection.execute(
            "SELECT c.workshop_id, s.agent_id, s.quiet_expires_at, c.archived_at, ca.detached_at, "
            "d.lifecycle_state, d.active_revision_id = s.agent_definition_revision_id, cp.enabled, "
            "EXISTS(SELECT 1 FROM json_each(r.collaboration_operations_json) "
            "WHERE value = 'standing_participation'), "
            "EXISTS(SELECT 1 FROM json_each(op.allowed_operations_json) "
            "WHERE value = 'standing_participation') "
            "FROM channel_agent_standings s JOIN channels c ON c.id = s.channel_id "
            "LEFT JOIN channel_agents ca ON ca.channel_id = s.channel_id AND ca.agent_id = s.agent_id "
            "LEFT JOIN agent_definitions d ON d.id = s.agent_definition_id "
            "LEFT JOIN agent_definition_revisions r ON r.id = s.agent_definition_revision_id "
            "LEFT JOIN channel_standing_participation_policies cp ON cp.channel_id = s.channel_id "
            "LEFT JOIN agent_collaboration_owner_policies op ON op.agent_definition_id = s.agent_definition_id "
            "WHERE s.channel_id = ? AND s.lifecycle_state = 'active' "
            "ORDER BY s.started_event_position, s.agent_id",
            (channel_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        endings: list[tuple[WorkshopId, AgentId, str]] = []
        host_enabled = CollaborationOperation.STANDING_PARTICIPATION in self._host_policy.effective_allowed_operations
        for index, row in enumerate(rows):
            quiet_expires_at = _timestamp(row[2], field="quiet_expires_at") if row[2] is not None else None
            reason: str | None = None
            if not host_enabled:
                reason = "host_policy_revoked"
            elif row[3] is not None:
                reason = "channel_archived"
            elif row[4] is not None:
                reason = "detached"
            elif str(row[5]) == "archived":
                reason = "definition_archived"
            elif str(row[5]) != "active" or not bool(row[6]) or not bool(row[8]):
                reason = "access_removed"
            elif not bool(row[7]):
                reason = "channel_policy_disabled"
            elif not bool(row[9]):
                reason = "owner_policy_revoked"
            elif index >= self._host_policy.standing_participation.max_agents_per_channel:
                reason = "overflow"
            elif quiet_expires_at is not None and quiet_expires_at <= occurred_at:
                reason = "quiet_expired"
            if reason is not None:
                endings.append((WorkshopId(str(row[0])), AgentId(str(row[1])), reason))
        for workshop_id, agent_id, reason in endings:
            await self._end_one_in_transaction(
                channel_id,
                agent_id,
                workshop_id=workshop_id,
                reason=reason,
                cause_event_id=None,
                actor_principal_id=None,
                occurred_at=occurred_at,
            )

    async def _end_channel_in_transaction(
        self,
        channel_id: ChannelId,
        *,
        workshop_id: WorkshopId,
        reason: str,
        cause_event_id: EventId | None,
        actor_principal_id: PrincipalId | None,
        occurred_at: datetime,
    ) -> None:
        async with self._store.connection.execute(
            "SELECT agent_id FROM channel_agent_standings WHERE channel_id = ? AND lifecycle_state = 'active'",
            (channel_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        for row in rows:
            await self._end_one_in_transaction(
                channel_id,
                AgentId(str(row[0])),
                workshop_id=workshop_id,
                reason=reason,
                cause_event_id=cause_event_id,
                actor_principal_id=actor_principal_id,
                occurred_at=occurred_at,
            )

    async def _end_one_in_transaction(
        self,
        channel_id: ChannelId,
        agent_id: AgentId,
        *,
        workshop_id: WorkshopId,
        reason: str,
        cause_event_id: EventId | None,
        actor_principal_id: PrincipalId | None,
        occurred_at: datetime,
    ) -> None:
        if reason not in _END_REASONS:
            raise ValueError("invalid standing end reason")
        cause = str(cause_event_id) if cause_event_id is not None else occurred_at.isoformat()
        event = EventEnvelope.create(
            event_id=EventId.derived(channel_id, f"standing-end:{agent_id}:{reason}:{cause}"),
            event_type=WorkshopEventType.CHANNEL_AGENT_STANDING_ENDED,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="channel",
            aggregate_id=channel_id,
            actor_principal_id=actor_principal_id,
            occurred_at=occurred_at,
            idempotency_key=f"standing-end:{channel_id}:{agent_id}:{reason}:{cause}",
            payload={
                "agent_id": agent_id,
                "reason": reason,
                "cause_event_id": cause_event_id,
                "ended_by_principal_id": actor_principal_id,
            },
            metadata={"source": "standing_participation_authority"},
        )
        result = await self._store.append_in_transaction(event)
        if result.inserted:
            await self._store.project_pending_in_transaction(self._projection())

    async def _channel_access(self, principal_id: PrincipalId, channel_id: ChannelId) -> bool:
        _workshop_id, can_manage = await self._channel_access_row(principal_id, channel_id)
        return can_manage

    async def _channel_access_row(self, principal_id: PrincipalId, channel_id: ChannelId) -> tuple[WorkshopId, bool]:
        async with self._store.connection.execute(
            "SELECT c.workshop_id, cm.role = 'owner' FROM channels c "
            "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.principal_id = ? "
            "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
            "WHERE c.id = ? AND c.kind = 'group'",
            (principal_id, channel_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopStandingParticipationAccessDenied("Channel access denied")
        return WorkshopId(str(row[0])), bool(row[1])


async def end_active_standings_in_transaction(
    store: WorkshopEventStore,
    *,
    reason: str,
    occurred_at: datetime,
    cause_event_id: EventId,
    channel_id: ChannelId | None = None,
    agent_id: AgentId | None = None,
    definition_id: AgentDefinitionId | None = None,
    actor_principal_id: PrincipalId | None = None,
) -> int:
    """Append explicit terminal facts for active subscriptions matching a lifecycle cause."""
    selectors: list[str] = ["s.lifecycle_state = 'active'"]
    parameters: list[object] = []
    if channel_id is not None:
        selectors.append("s.channel_id = ?")
        parameters.append(channel_id)
    if agent_id is not None:
        selectors.append("s.agent_id = ?")
        parameters.append(agent_id)
    if definition_id is not None:
        selectors.append("s.agent_definition_id = ?")
        parameters.append(definition_id)
    if len(selectors) == 1:
        raise ValueError("standing end requires a bounded selector")
    async with store.connection.execute(
        "SELECT s.channel_id, s.agent_id, c.workshop_id FROM channel_agent_standings s "
        "JOIN channels c ON c.id = s.channel_id WHERE " + " AND ".join(selectors),
        tuple(parameters),
    ) as cursor:
        rows = list(await cursor.fetchall())
    service = WorkshopStandingParticipationService(store, CollaborationHostPolicy())
    for channel_raw, agent_raw, workshop_raw in rows:
        await service._end_one_in_transaction(
            ChannelId(str(channel_raw)),
            AgentId(str(agent_raw)),
            workshop_id=WorkshopId(str(workshop_raw)),
            reason=reason,
            cause_event_id=cause_event_id,
            actor_principal_id=actor_principal_id,
            occurred_at=occurred_at,
        )
    return len(rows)
