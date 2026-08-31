"""Canonical, transport-independent Workshop channel and agent attachment lifecycle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import (
    AgentId,
    ChannelAgentId,
    ChannelId,
    ChannelMembershipId,
    EventEnvelope,
    PrincipalId,
    RuntimeAssignmentId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import StoredEvent, WorkshopEventStore


class WorkshopChannelLifecycleError(RuntimeError):
    """A canonical channel lifecycle operation could not be completed."""


class WorkshopChannelLifecycleAccessDenied(WorkshopChannelLifecycleError):
    """The principal lacks the requested channel-management authority."""


class WorkshopChannelLifecycleValidationError(WorkshopChannelLifecycleError):
    """A channel lifecycle request references malformed or unavailable state."""


class WorkshopChannelLifecycleStorageError(WorkshopChannelLifecycleError):
    """A canonical channel lifecycle mutation could not be persisted atomically."""


@dataclass(frozen=True, slots=True)
class CreatedWorkshopChannel:
    """The canonical identity and immutable creation properties of a group channel."""

    channel_id: ChannelId
    workshop_id: WorkshopId
    name: str
    visibility: str
    origin_channel_id: ChannelId | None
    owner_principal_id: PrincipalId
    agent_ids: tuple[AgentId, ...]


@dataclass(frozen=True, slots=True)
class WorkshopChannelAgentAttachment:
    """One explicit, replayable active group-channel sponsorship."""

    channel_id: ChannelId
    agent_id: AgentId
    sponsor_principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId
    changed: bool


@dataclass(frozen=True, slots=True)
class WorkshopChannelLifecycleMutation:
    """One reversible canonical group-channel lifecycle transition."""

    channel_id: ChannelId
    archived: bool
    changed: bool
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class WorkshopHumanChannelMember:
    """One human eligible for or holding membership in a private group channel."""

    principal_id: PrincipalId
    display_name: str
    handle: str
    role: str | None


@dataclass(frozen=True, slots=True)
class WorkshopHumanMembershipSnapshot:
    """Principal-bounded human membership state for one group channel."""

    channel_id: ChannelId
    workshop_id: WorkshopId
    archived: bool
    can_manage: bool
    state_version: int
    members: tuple[WorkshopHumanChannelMember, ...]
    eligible_humans: tuple[WorkshopHumanChannelMember, ...]


@dataclass(frozen=True, slots=True)
class WorkshopHumanMembershipMutation:
    """One replayable human participant addition or removal."""

    channel_id: ChannelId
    member: WorkshopHumanChannelMember
    operation: str
    state_version: int
    changed: bool


@dataclass(frozen=True, slots=True)
class _AgentAttachment:
    agent_id: AgentId
    agent_principal_id: PrincipalId
    sponsor_principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId


def _normalize_name(value: object) -> str:
    if not isinstance(value, str):
        raise WorkshopChannelLifecycleValidationError("Channel name must be text")
    normalized = value.strip()
    if not normalized or len(normalized) > 200:
        raise WorkshopChannelLifecycleValidationError("Channel name must contain 1 through 200 characters")
    return normalized


def _normalize_agents(values: object) -> tuple[AgentId, ...]:
    if not isinstance(values, list) or not values or len(values) > 16:
        raise WorkshopChannelLifecycleValidationError("agent_ids must contain 1 through 16 canonical agent IDs")
    try:
        normalized = tuple(AgentId(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise WorkshopChannelLifecycleValidationError("agent_ids must contain canonical agent IDs") from exc
    if len(set(normalized)) != len(normalized):
        raise WorkshopChannelLifecycleValidationError("agent_ids must not contain duplicates")
    return normalized


def _normalize_origin(value: object) -> ChannelId | None:
    if value is None:
        return None
    try:
        return ChannelId(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise WorkshopChannelLifecycleValidationError("origin_channel_id must be a canonical channel ID") from exc


def _normalize_operation_id(value: object) -> str:
    if not isinstance(value, str):
        raise WorkshopChannelLifecycleValidationError("client_operation_id must be text")
    normalized = value.strip()
    if not normalized or len(normalized) > 200:
        raise WorkshopChannelLifecycleValidationError("client_operation_id must contain 1 through 200 characters")
    return normalized


def _normalize_membership_version(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise WorkshopChannelLifecycleValidationError("expected_state_version must be a non-negative integer")
    return value


class WorkshopChannelLifecycleService:
    """Create group channels and manage their explicit agent sponsorships."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def human_members(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
    ) -> WorkshopHumanMembershipSnapshot:
        """Return human membership and eligible Workshop humans without cross-Workshop discovery."""
        workshop_id, archived, state_version, can_manage = await self._resolve_human_membership_access(
            principal_id,
            channel_id,
        )
        async with self._store.connection.execute(
            "SELECT p.id, p.display_name, hh.handle, cm.role "
            "FROM channel_memberships cm JOIN principals p ON p.id = cm.principal_id "
            "AND p.kind = 'human' JOIN human_handles hh ON hh.workshop_id = ? "
            "AND hh.principal_id = p.id WHERE cm.channel_id = ? "
            "ORDER BY CASE cm.role WHEN 'owner' THEN 0 ELSE 1 END, lower(p.display_name), p.id",
            (workshop_id, channel_id),
        ) as cursor:
            member_rows = list(await cursor.fetchall())
        members = tuple(self._human_member_from_row(row, with_role=True) for row in member_rows)
        eligible: tuple[WorkshopHumanChannelMember, ...] = ()
        if can_manage and not archived:
            async with self._store.connection.execute(
                "SELECT p.id, p.display_name, hh.handle FROM workshop_memberships wm "
                "JOIN principals p ON p.id = wm.principal_id AND p.kind = 'human' "
                "JOIN human_handles hh ON hh.workshop_id = wm.workshop_id AND hh.principal_id = p.id "
                "WHERE wm.workshop_id = ? AND NOT EXISTS (SELECT 1 FROM channel_memberships cm "
                "WHERE cm.channel_id = ? AND cm.principal_id = p.id) "
                "ORDER BY lower(p.display_name), p.id",
                (workshop_id, channel_id),
            ) as cursor:
                eligible_rows = list(await cursor.fetchall())
            eligible = tuple(self._human_member_from_row(row, with_role=False) for row in eligible_rows)
        return WorkshopHumanMembershipSnapshot(
            channel_id,
            workshop_id,
            archived,
            can_manage and not archived,
            state_version,
            members,
            eligible,
        )

    async def add_human_member(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        member_principal_id: PrincipalId,
        *,
        expected_state_version: object,
        client_operation_id: object,
    ) -> WorkshopHumanMembershipMutation:
        """Add one Workshop human as a group participant."""
        return await self._change_human_membership(
            principal_id,
            channel_id,
            member_principal_id,
            operation="add",
            expected_state_version=expected_state_version,
            client_operation_id=client_operation_id,
        )

    async def remove_human_member(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        member_principal_id: PrincipalId,
        *,
        expected_state_version: object,
        client_operation_id: object,
    ) -> WorkshopHumanMembershipMutation:
        """Remove one human participant; immutable owners cannot be removed or transferred."""
        return await self._change_human_membership(
            principal_id,
            channel_id,
            member_principal_id,
            operation="remove",
            expected_state_version=expected_state_version,
            client_operation_id=client_operation_id,
        )

    async def _change_human_membership(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        member_principal_id: PrincipalId,
        *,
        operation: str,
        expected_state_version: object,
        client_operation_id: object,
    ) -> WorkshopHumanMembershipMutation:
        expected_version = _normalize_membership_version(expected_state_version)
        operation_id = _normalize_operation_id(client_operation_id)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            workshop_id, archived, current_version, can_manage = await self._resolve_human_membership_access(
                principal_id,
                channel_id,
            )
            if not can_manage:
                raise WorkshopChannelLifecycleAccessDenied(
                    "Human membership changes require group-channel owner or Workshop administrator authority"
                )
            if archived:
                raise WorkshopChannelLifecycleValidationError("Archived channels are read-only")
            member = await self._resolve_workshop_human(workshop_id, member_principal_id)
            idempotency_key = f"workshop-client:channel:{channel_id}:human-membership:{principal_id}:{operation_id}"
            existing = await self._store.event_by_idempotency_key(idempotency_key)
            if existing is not None:
                result = self._human_membership_result_from_event(
                    existing,
                    principal_id=principal_id,
                    workshop_id=workshop_id,
                    channel_id=channel_id,
                    member=member,
                    operation=operation,
                )
                await connection.rollback()
                return result
            if expected_version != current_version:
                raise WorkshopChannelLifecycleValidationError(
                    "Channel membership changed; reload members before saving"
                )
            async with connection.execute(
                "SELECT role FROM channel_memberships WHERE channel_id = ? AND principal_id = ?",
                (channel_id, member_principal_id),
            ) as cursor:
                current = await cursor.fetchone()
            if operation == "add":
                if current is not None:
                    raise WorkshopChannelLifecycleValidationError("Human is already a channel member")
                event_type = WorkshopEventType.CHANNEL_MEMBER_ADDED
                event_version = 2
            else:
                if current is None:
                    raise WorkshopChannelLifecycleValidationError("Human is not a channel member")
                if str(current[0]) == "owner":
                    raise WorkshopChannelLifecycleValidationError(
                        "Channel ownership is immutable; owners cannot be removed or transferred"
                    )
                if str(current[0]) != "participant":
                    raise WorkshopChannelLifecycleValidationError("Only human participants may be removed")
                event_type = WorkshopEventType.CHANNEL_MEMBER_REMOVED
                event_version = 1
            event = EventEnvelope.create(
                event_type=event_type,
                event_version=event_version,
                workshop_id=workshop_id,
                aggregate_type="channel_membership",
                aggregate_id=ChannelMembershipId.derived(channel_id, f"principal:{member_principal_id}"),
                actor_principal_id=principal_id,
                occurred_at=datetime.now(UTC),
                idempotency_key=idempotency_key,
                payload={
                    "channel_id": channel_id,
                    "principal_id": member_principal_id,
                    "role": "participant",
                },
                metadata={"source": "workshop_client"},
            )
            appended = await self._store.append_in_transaction(event)
            if not appended.inserted:
                raise WorkshopChannelLifecycleError("New membership event unexpectedly already exists")
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            await connection.commit()
            return WorkshopHumanMembershipMutation(
                channel_id,
                WorkshopHumanChannelMember(
                    member.principal_id,
                    member.display_name,
                    member.handle,
                    "participant" if operation == "add" else None,
                ),
                operation,
                appended.event.position,
                True,
            )
        except WorkshopChannelLifecycleError:
            await connection.rollback()
            raise
        except Exception as exc:
            await connection.rollback()
            raise WorkshopChannelLifecycleStorageError("Human membership change could not be persisted") from exc

    async def _resolve_human_membership_access(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
    ) -> tuple[WorkshopId, bool, int, bool]:
        async with self._store.connection.execute(
            "SELECT c.workshop_id, c.archived_at, coalesce(c.membership_event_position, 0), "
            "(coalesce(cm.role, '') = 'owner' OR wm.role = 'admin') AS can_manage "
            "FROM channels c JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
            "AND wm.principal_id = ? JOIN principals actor ON actor.id = wm.principal_id "
            "AND actor.kind = 'human' LEFT JOIN channel_memberships cm ON cm.channel_id = c.id "
            "AND cm.principal_id = wm.principal_id WHERE c.id = ? AND c.kind = 'group' "
            "AND (cm.id IS NOT NULL OR wm.role = 'admin')",
            (principal_id, channel_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopChannelLifecycleAccessDenied("Channel membership is unavailable")
        return WorkshopId(str(row[0])), row[1] is not None, int(row[2]), bool(row[3])

    async def _resolve_workshop_human(
        self,
        workshop_id: WorkshopId,
        principal_id: PrincipalId,
    ) -> WorkshopHumanChannelMember:
        async with self._store.connection.execute(
            "SELECT p.id, p.display_name, hh.handle FROM workshop_memberships wm "
            "JOIN principals p ON p.id = wm.principal_id AND p.kind = 'human' "
            "JOIN human_handles hh ON hh.workshop_id = wm.workshop_id AND hh.principal_id = p.id "
            "WHERE wm.workshop_id = ? AND wm.principal_id = ?",
            (workshop_id, principal_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopChannelLifecycleValidationError("Human is not eligible for this Workshop channel")
        return self._human_member_from_row(row, with_role=False)

    @staticmethod
    def _human_member_from_row(
        row: Sequence[object],
        *,
        with_role: bool,
    ) -> WorkshopHumanChannelMember:
        values = row
        return WorkshopHumanChannelMember(
            PrincipalId(str(values[0])),
            str(values[1]),
            str(values[2]),
            str(values[3]) if with_role else None,
        )

    @staticmethod
    def _human_membership_result_from_event(
        stored: StoredEvent,
        *,
        principal_id: PrincipalId,
        workshop_id: WorkshopId,
        channel_id: ChannelId,
        member: WorkshopHumanChannelMember,
        operation: str,
    ) -> WorkshopHumanMembershipMutation:
        envelope = stored.envelope
        position = stored.position
        expected_type = (
            WorkshopEventType.CHANNEL_MEMBER_ADDED if operation == "add" else WorkshopEventType.CHANNEL_MEMBER_REMOVED
        )
        if (
            envelope.event_type != expected_type
            or envelope.workshop_id != workshop_id
            or envelope.actor_principal_id != principal_id
            or envelope.payload.get("channel_id") != channel_id
            or envelope.payload.get("principal_id") != member.principal_id
            or envelope.payload.get("role") != "participant"
        ):
            raise WorkshopChannelLifecycleValidationError(
                "client_operation_id is already bound to a different membership operation"
            )
        return WorkshopHumanMembershipMutation(
            channel_id,
            WorkshopHumanChannelMember(
                member.principal_id,
                member.display_name,
                member.handle,
                "participant" if operation == "add" else None,
            ),
            operation,
            position,
            False,
        )

    async def archive(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        *,
        client_operation_id: object,
    ) -> WorkshopChannelLifecycleMutation:
        """Archive one idle group channel without deleting its history."""
        return await self._change_archival_state(
            principal_id,
            channel_id,
            archived=True,
            client_operation_id=client_operation_id,
        )

    async def restore(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        *,
        client_operation_id: object,
    ) -> WorkshopChannelLifecycleMutation:
        """Restore one archived group channel with its identity intact."""
        return await self._change_archival_state(
            principal_id,
            channel_id,
            archived=False,
            client_operation_id=client_operation_id,
        )

    async def _change_archival_state(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        *,
        archived: bool,
        client_operation_id: object,
    ) -> WorkshopChannelLifecycleMutation:
        operation_id = _normalize_operation_id(client_operation_id)
        operation = "archive" if archived else "restore"
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            workshop_id, archived_at = await self._resolve_managed_group_state(
                principal_id,
                channel_id,
            )
            idempotency_key = f"workshop-client:channel-lifecycle:{principal_id}:{operation_id}"
            existing = await self._store.event_by_idempotency_key(idempotency_key)
            if existing is not None:
                result = self._archival_result_from_event(
                    existing.envelope,
                    principal_id=principal_id,
                    workshop_id=workshop_id,
                    channel_id=channel_id,
                    archived=archived,
                )
                await connection.rollback()
                return result
            if archived_at is not None and archived:
                raise WorkshopChannelLifecycleValidationError("Channel is already archived")
            if archived_at is None and not archived:
                raise WorkshopChannelLifecycleValidationError("Channel is not archived")
            if archived:
                async with connection.execute(
                    "SELECT COUNT(*) FROM runs WHERE channel_id = ? AND status IN ('accepted', 'started')",
                    (channel_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None or int(row[0]) != 0:
                    raise WorkshopChannelLifecycleValidationError(
                        "Channel cannot be archived while an agent run is active"
                    )
            now = datetime.now(UTC)
            event = EventEnvelope.create(
                event_type=(WorkshopEventType.CHANNEL_ARCHIVED if archived else WorkshopEventType.CHANNEL_RESTORED),
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="channel",
                aggregate_id=channel_id,
                actor_principal_id=principal_id,
                occurred_at=now,
                idempotency_key=idempotency_key,
                payload={},
                metadata={"source": "workshop_client"},
            )
            result = await self._store.append_in_transaction(event)
            if not result.inserted:
                raise WorkshopChannelLifecycleError("New channel lifecycle event unexpectedly already exists")
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            await connection.commit()
            return WorkshopChannelLifecycleMutation(channel_id, archived, True, now)
        except WorkshopChannelLifecycleError:
            await connection.rollback()
            raise
        except Exception as exc:
            await connection.rollback()
            raise WorkshopChannelLifecycleStorageError(f"Channel {operation} could not be persisted") from exc

    @staticmethod
    def _archival_result_from_event(
        envelope: EventEnvelope,
        *,
        principal_id: PrincipalId,
        workshop_id: WorkshopId,
        channel_id: ChannelId,
        archived: bool,
    ) -> WorkshopChannelLifecycleMutation:
        expected_type = WorkshopEventType.CHANNEL_ARCHIVED if archived else WorkshopEventType.CHANNEL_RESTORED
        if (
            envelope.event_type != expected_type
            or envelope.event_version != 1
            or envelope.workshop_id != workshop_id
            or envelope.aggregate_type != "channel"
            or envelope.aggregate_id != channel_id
            or envelope.actor_principal_id != principal_id
            or envelope.payload
        ):
            raise WorkshopChannelLifecycleValidationError(
                "client_operation_id is already bound to a different channel lifecycle operation"
            )
        return WorkshopChannelLifecycleMutation(
            channel_id,
            archived,
            False,
            envelope.occurred_at,
        )

    async def create_group(
        self,
        principal_id: PrincipalId,
        *,
        name: object,
        agent_ids: object,
        origin_channel_id: object = None,
    ) -> CreatedWorkshopChannel:
        if not isinstance(principal_id, PrincipalId):
            raise WorkshopChannelLifecycleAccessDenied("Canonical human principal is required")
        normalized_name = _normalize_name(name)
        normalized_agents = _normalize_agents(agent_ids)
        normalized_origin = _normalize_origin(origin_channel_id)

        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            workshop_id = await self._resolve_workshop(principal_id, normalized_origin)
            attachments = await self._resolve_agent_attachments(
                principal_id,
                workshop_id,
                normalized_agents,
            )
            channel_id = ChannelId.new()
            now = datetime.now(UTC)
            for event in self._creation_events(
                workshop_id=workshop_id,
                channel_id=channel_id,
                principal_id=principal_id,
                name=normalized_name,
                origin_channel_id=normalized_origin,
                attachments=attachments,
                occurred_at=now,
            ):
                result = await self._store.append_in_transaction(event)
                if not result.inserted:
                    raise WorkshopChannelLifecycleError("New channel event identity unexpectedly already exists")
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            await connection.commit()
        except WorkshopChannelLifecycleError:
            await connection.rollback()
            raise
        except Exception as exc:
            await connection.rollback()
            raise WorkshopChannelLifecycleStorageError("Channel creation could not be persisted") from exc

        return CreatedWorkshopChannel(
            channel_id=channel_id,
            workshop_id=workshop_id,
            name=normalized_name,
            visibility="private",
            origin_channel_id=normalized_origin,
            owner_principal_id=principal_id,
            agent_ids=tuple(attachment.agent_id for attachment in attachments),
        )

    async def attach_agent(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        agent_id: AgentId,
        *,
        client_operation_id: object,
    ) -> WorkshopChannelAgentAttachment:
        """Attach one enabled agent using the acting owner's direct runtime."""
        operation_id = _normalize_operation_id(client_operation_id)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            workshop_id = await self._resolve_managed_group(principal_id, channel_id)
            idempotency_key = self._attachment_key(
                channel_id,
                agent_id,
                "attach",
                operation_id,
            )
            existing_event = await self._store.event_by_idempotency_key(idempotency_key)
            if existing_event is not None:
                result = self._attachment_result_from_event(
                    existing_event.envelope,
                    principal_id=principal_id,
                    workshop_id=workshop_id,
                    channel_id=channel_id,
                    agent_id=agent_id,
                    operation="attach",
                )
                await connection.rollback()
                return result

            attachment = (await self._resolve_agent_attachments(principal_id, workshop_id, (agent_id,)))[0]
            current = await self._attachment_state(channel_id, agent_id)
            if current is not None and current[2] is None:
                raise WorkshopChannelLifecycleValidationError("Agent is already attached to this channel")

            now = datetime.now(UTC)
            events: list[EventEnvelope] = []
            if current is None and not await self._has_membership(
                channel_id,
                attachment.agent_principal_id,
            ):
                events.append(
                    self._agent_membership_event(
                        workshop_id,
                        channel_id,
                        principal_id,
                        attachment.agent_principal_id,
                        occurred_at=now,
                        operation_id=operation_id,
                    )
                )
            events.append(
                self._attachment_event(
                    workshop_id,
                    channel_id,
                    principal_id,
                    attachment.agent_id,
                    attachment.sponsor_principal_id,
                    attachment.runtime_profile_id,
                    occurred_at=now,
                    operation="attach",
                    operation_id=operation_id,
                )
            )
            if current is None:
                events.append(
                    self._runtime_assignment_event(
                        workshop_id,
                        channel_id,
                        principal_id,
                        attachment,
                        occurred_at=now,
                        reassigned=False,
                        operation_id=operation_id,
                    )
                )
            elif current[1] != attachment.runtime_profile_id:
                events.append(
                    self._runtime_assignment_event(
                        workshop_id,
                        channel_id,
                        principal_id,
                        attachment,
                        occurred_at=now,
                        reassigned=True,
                        operation_id=operation_id,
                    )
                )
            for event in events:
                await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            await connection.commit()
        except WorkshopChannelLifecycleError:
            await connection.rollback()
            raise
        except Exception as exc:
            await connection.rollback()
            raise WorkshopChannelLifecycleStorageError("Agent attachment could not be persisted") from exc
        return self._attachment_result(channel_id, attachment, changed=True)

    async def detach_agent(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        agent_id: AgentId,
        *,
        client_operation_id: object,
    ) -> WorkshopChannelAgentAttachment:
        """Detach one group agent while retaining its sponsorship history."""
        operation_id = _normalize_operation_id(client_operation_id)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            workshop_id = await self._resolve_managed_group(principal_id, channel_id)
            idempotency_key = self._attachment_key(
                channel_id,
                agent_id,
                "detach",
                operation_id,
            )
            existing_event = await self._store.event_by_idempotency_key(idempotency_key)
            if existing_event is not None:
                result = self._attachment_result_from_event(
                    existing_event.envelope,
                    principal_id=principal_id,
                    workshop_id=workshop_id,
                    channel_id=channel_id,
                    agent_id=agent_id,
                    operation="detach",
                )
                await connection.rollback()
                return result

            current = await self._attachment_state(channel_id, agent_id)
            if current is None:
                raise WorkshopChannelLifecycleValidationError("Agent is not attached to this channel")
            sponsor_principal_id, runtime_profile_id, detached_at = current
            if detached_at is not None:
                raise WorkshopChannelLifecycleValidationError("Agent is not attached to this channel")
            await self._store.append_in_transaction(
                self._attachment_event(
                    workshop_id,
                    channel_id,
                    principal_id,
                    agent_id,
                    sponsor_principal_id,
                    runtime_profile_id,
                    occurred_at=datetime.now(UTC),
                    operation="detach",
                    operation_id=operation_id,
                )
            )
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            await connection.commit()
        except WorkshopChannelLifecycleError:
            await connection.rollback()
            raise
        except Exception as exc:
            await connection.rollback()
            raise WorkshopChannelLifecycleStorageError("Agent detachment could not be persisted") from exc
        return WorkshopChannelAgentAttachment(
            channel_id,
            agent_id,
            sponsor_principal_id,
            runtime_profile_id,
            True,
        )

    @staticmethod
    def _attachment_result_from_event(
        envelope: EventEnvelope,
        *,
        principal_id: PrincipalId,
        workshop_id: WorkshopId,
        channel_id: ChannelId,
        agent_id: AgentId,
        operation: str,
    ) -> WorkshopChannelAgentAttachment:
        expected_type = (
            WorkshopEventType.CHANNEL_AGENT_ATTACHED
            if operation == "attach"
            else WorkshopEventType.CHANNEL_AGENT_DETACHED
        )
        payload = envelope.payload
        if (
            envelope.event_type != expected_type
            or envelope.workshop_id != workshop_id
            or envelope.actor_principal_id != principal_id
            or payload.get("channel_id") != channel_id
            or payload.get("agent_id") != agent_id
            or payload.get("sponsor_principal_id") != principal_id
        ):
            raise WorkshopChannelLifecycleValidationError(
                "client_operation_id is already bound to a different channel-agent operation"
            )
        try:
            return WorkshopChannelAgentAttachment(
                channel_id,
                agent_id,
                PrincipalId(str(payload["sponsor_principal_id"])),
                RuntimeProfileId(str(payload["runtime_profile_id"])),
                False,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkshopChannelLifecycleValidationError(
                "Stored channel-agent operation has invalid sponsorship metadata"
            ) from exc

    async def _resolve_workshop(
        self,
        principal_id: PrincipalId,
        origin_channel_id: ChannelId | None,
    ) -> WorkshopId:
        if origin_channel_id is not None:
            async with self._store.connection.execute(
                "SELECT c.workshop_id FROM channels c "
                "JOIN channel_memberships cm ON cm.channel_id = c.id "
                "AND cm.principal_id = ? "
                "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
                "AND wm.principal_id = cm.principal_id "
                "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
                "WHERE c.id = ?",
                (principal_id, origin_channel_id),
            ) as cursor:
                rows = list(await cursor.fetchall())
        else:
            async with self._store.connection.execute(
                "SELECT wm.workshop_id FROM workshop_memberships wm "
                "JOIN principals p ON p.id = wm.principal_id AND p.kind = 'human' "
                "WHERE wm.principal_id = ? ORDER BY wm.workshop_id",
                (principal_id,),
            ) as cursor:
                rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopChannelLifecycleAccessDenied("Channel creation requires one accessible Workshop context")
        return WorkshopId(str(rows[0][0]))

    async def _resolve_managed_group(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
    ) -> WorkshopId:
        workshop_id, archived_at = await self._resolve_managed_group_state(
            principal_id,
            channel_id,
        )
        if archived_at is not None:
            raise WorkshopChannelLifecycleValidationError("Archived channels are read-only")
        return workshop_id

    async def _resolve_managed_group_state(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
    ) -> tuple[WorkshopId, str | None]:
        async with self._store.connection.execute(
            "SELECT c.workshop_id, c.archived_at FROM channels c "
            "JOIN channel_memberships cm ON cm.channel_id = c.id "
            "AND cm.principal_id = ? AND cm.role = 'owner' "
            "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
            "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
            "AND wm.principal_id = cm.principal_id "
            "WHERE c.id = ? AND c.kind = 'group'",
            (principal_id, channel_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopChannelLifecycleAccessDenied(
                "Channel lifecycle changes require group-channel owner authority"
            )
        return WorkshopId(str(row[0])), (str(row[1]) if row[1] is not None else None)

    async def _resolve_agent_attachments(
        self,
        principal_id: PrincipalId,
        workshop_id: WorkshopId,
        agent_ids: tuple[AgentId, ...],
    ) -> tuple[_AgentAttachment, ...]:
        placeholders = ",".join("?" for _ in agent_ids)
        async with self._store.connection.execute(
            "SELECT a.id, a.principal_id, ra.runtime_profile_id "
            "FROM agents a "
            "JOIN agent_definitions ad ON ad.agent_id = a.id "
            "AND ad.workshop_id = a.workshop_id "
            "AND ad.lifecycle_state = 'active' "
            "AND ad.active_revision_id IS NOT NULL "
            "JOIN workshop_memberships agent_wm ON agent_wm.workshop_id = a.workshop_id "
            "AND agent_wm.principal_id = a.principal_id AND agent_wm.role = 'agent' "
            "JOIN channel_agents ca ON ca.agent_id = a.id AND ca.detached_at IS NULL "
            "JOIN channels c ON c.id = ca.channel_id AND c.workshop_id = a.workshop_id "
            "AND c.kind = 'direct' "
            "JOIN channel_memberships cm ON cm.channel_id = c.id "
            "AND cm.principal_id = ? AND cm.role = 'owner' "
            "JOIN principal_agent_enablements pae ON pae.direct_channel_id = c.id "
            "AND pae.principal_id = cm.principal_id AND pae.agent_id = a.id "
            "AND pae.lifecycle_state = 'enabled' "
            "JOIN channel_agent_runtime_assignments ra ON ra.channel_id = c.id "
            "AND ra.agent_id = a.id AND ra.runtime_profile_id = pae.runtime_profile_id "
            f"WHERE a.workshop_id = ? AND a.id IN ({placeholders}) "
            "ORDER BY a.id",
            (principal_id, workshop_id, *agent_ids),
        ) as cursor:
            rows = list(await cursor.fetchall())
        by_agent: dict[AgentId, list[_AgentAttachment]] = {}
        for row in rows:
            attachment = _AgentAttachment(
                AgentId(str(row[0])),
                PrincipalId(str(row[1])),
                principal_id,
                RuntimeProfileId(str(row[2])),
            )
            by_agent.setdefault(attachment.agent_id, []).append(attachment)
        if any(len(by_agent.get(agent_id, ())) != 1 for agent_id in agent_ids):
            raise WorkshopChannelLifecycleValidationError(
                "Every requested agent must be enabled with one runnable sponsored runtime"
            )
        return tuple(by_agent[agent_id][0] for agent_id in agent_ids)

    async def _attachment_state(
        self,
        channel_id: ChannelId,
        agent_id: AgentId,
    ) -> tuple[PrincipalId, RuntimeProfileId, str | None] | None:
        async with self._store.connection.execute(
            "SELECT sponsor_principal_id, sponsored_runtime_profile_id, detached_at "
            "FROM channel_agents WHERE channel_id = ? AND agent_id = ?",
            (channel_id, agent_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        if row[0] is None or row[1] is None:
            raise WorkshopChannelLifecycleValidationError("Existing attachment has incomplete sponsorship metadata")
        return PrincipalId(str(row[0])), RuntimeProfileId(str(row[1])), (str(row[2]) if row[2] is not None else None)

    async def _has_membership(self, channel_id: ChannelId, principal_id: PrincipalId) -> bool:
        async with self._store.connection.execute(
            "SELECT 1 FROM channel_memberships WHERE channel_id = ? AND principal_id = ?",
            (channel_id, principal_id),
        ) as cursor:
            return await cursor.fetchone() is not None

    @staticmethod
    def _creation_events(
        *,
        workshop_id: WorkshopId,
        channel_id: ChannelId,
        principal_id: PrincipalId,
        name: str,
        origin_channel_id: ChannelId | None,
        attachments: tuple[_AgentAttachment, ...],
        occurred_at: datetime,
    ) -> tuple[EventEnvelope, ...]:
        metadata = {"source": "workshop_client"}
        channel_payload: dict[str, object] = {
            "kind": "group",
            "name": name,
            "visibility": "private",
        }
        if origin_channel_id is not None:
            channel_payload["origin_channel_id"] = origin_channel_id
        events = [
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_CREATED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="channel",
                aggregate_id=channel_id,
                actor_principal_id=principal_id,
                occurred_at=occurred_at,
                idempotency_key=f"workshop-client:channel:{channel_id}:created",
                payload=channel_payload,
                metadata=metadata,
            ),
            WorkshopChannelLifecycleService._agent_membership_event(
                workshop_id,
                channel_id,
                principal_id,
                principal_id,
                occurred_at=occurred_at,
                operation_id="created-owner",
                role="owner",
            ),
        ]
        for attachment in attachments:
            events.extend(
                (
                    WorkshopChannelLifecycleService._agent_membership_event(
                        workshop_id,
                        channel_id,
                        principal_id,
                        attachment.agent_principal_id,
                        occurred_at=occurred_at,
                        operation_id=f"created-{attachment.agent_id}",
                    ),
                    WorkshopChannelLifecycleService._attachment_event(
                        workshop_id,
                        channel_id,
                        principal_id,
                        attachment.agent_id,
                        attachment.sponsor_principal_id,
                        attachment.runtime_profile_id,
                        occurred_at=occurred_at,
                        operation="attach",
                        operation_id="created",
                    ),
                    WorkshopChannelLifecycleService._runtime_assignment_event(
                        workshop_id,
                        channel_id,
                        principal_id,
                        attachment,
                        occurred_at=occurred_at,
                        reassigned=False,
                        operation_id="created",
                    ),
                )
            )
        return tuple(events)

    @staticmethod
    def _attachment_result(
        channel_id: ChannelId,
        attachment: _AgentAttachment,
        *,
        changed: bool,
    ) -> WorkshopChannelAgentAttachment:
        return WorkshopChannelAgentAttachment(
            channel_id,
            attachment.agent_id,
            attachment.sponsor_principal_id,
            attachment.runtime_profile_id,
            changed,
        )

    @staticmethod
    def _attachment_key(
        channel_id: ChannelId,
        agent_id: AgentId,
        operation: str,
        operation_id: str,
    ) -> str:
        return f"workshop-client:channel:{channel_id}:agent:{agent_id}:{operation}:{operation_id}"

    @classmethod
    def _attachment_event(
        cls,
        workshop_id: WorkshopId,
        channel_id: ChannelId,
        actor_principal_id: PrincipalId,
        agent_id: AgentId,
        sponsor_principal_id: PrincipalId,
        runtime_profile_id: RuntimeProfileId,
        *,
        occurred_at: datetime,
        operation: str,
        operation_id: str,
    ) -> EventEnvelope:
        return EventEnvelope.create(
            event_type=(
                WorkshopEventType.CHANNEL_AGENT_ATTACHED
                if operation == "attach"
                else WorkshopEventType.CHANNEL_AGENT_DETACHED
            ),
            event_version=2 if operation == "attach" else 1,
            workshop_id=workshop_id,
            aggregate_type="channel_agent",
            aggregate_id=ChannelAgentId.derived(channel_id, f"agent:{agent_id}"),
            actor_principal_id=actor_principal_id,
            occurred_at=occurred_at,
            idempotency_key=cls._attachment_key(channel_id, agent_id, operation, operation_id),
            payload={
                "channel_id": channel_id,
                "agent_id": agent_id,
                "sponsor_principal_id": sponsor_principal_id,
                "runtime_profile_id": runtime_profile_id,
            },
            metadata={"source": "workshop_client"},
        )

    @staticmethod
    def _agent_membership_event(
        workshop_id: WorkshopId,
        channel_id: ChannelId,
        actor_principal_id: PrincipalId,
        member_principal_id: PrincipalId,
        *,
        occurred_at: datetime,
        operation_id: str,
        role: str = "participant",
    ) -> EventEnvelope:
        return EventEnvelope.create(
            event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="channel_membership",
            aggregate_id=ChannelMembershipId.derived(channel_id, f"principal:{member_principal_id}"),
            actor_principal_id=actor_principal_id,
            occurred_at=occurred_at,
            idempotency_key=(f"workshop-client:channel:{channel_id}:member:{member_principal_id}:{operation_id}"),
            payload={
                "channel_id": channel_id,
                "principal_id": member_principal_id,
                "role": role,
            },
            metadata={"source": "workshop_client"},
        )

    @staticmethod
    def _runtime_assignment_event(
        workshop_id: WorkshopId,
        channel_id: ChannelId,
        actor_principal_id: PrincipalId,
        attachment: _AgentAttachment,
        *,
        occurred_at: datetime,
        reassigned: bool,
        operation_id: str,
    ) -> EventEnvelope:
        return EventEnvelope.create(
            event_type=(
                WorkshopEventType.RUNTIME_PROFILE_REASSIGNED
                if reassigned
                else WorkshopEventType.RUNTIME_PROFILE_ASSIGNED
            ),
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="runtime_assignment",
            aggregate_id=RuntimeAssignmentId.derived(
                channel_id,
                f"runtime-profile:{attachment.agent_id}",
            ),
            actor_principal_id=actor_principal_id,
            occurred_at=occurred_at,
            idempotency_key=(f"workshop-client:channel:{channel_id}:runtime:{attachment.agent_id}:{operation_id}"),
            payload={
                "channel_id": channel_id,
                "agent_id": attachment.agent_id,
                "runtime_profile_id": attachment.runtime_profile_id,
            },
            metadata={"source": "workshop_client"},
        )
