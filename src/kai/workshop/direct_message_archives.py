"""Canonical per-principal visibility for Workshop direct messages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import ChannelId, EventEnvelope, PrincipalId, WorkshopEventType, WorkshopId
from kai.workshop.human_direct_messages import is_canonical_human_direct_channel
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import StoredEvent, WorkshopEventStore


class WorkshopDirectMessageArchiveError(RuntimeError):
    """A direct-message visibility operation could not be completed."""


class WorkshopDirectMessageArchiveAccessDenied(WorkshopDirectMessageArchiveError):
    """The principal cannot manage the requested direct conversation."""


class WorkshopDirectMessageArchiveConflict(WorkshopDirectMessageArchiveError):
    """The requested direct-message archive transition is invalid."""


class WorkshopDirectMessageArchiveStorageError(WorkshopDirectMessageArchiveError):
    """A direct-message archive transition could not be persisted."""


@dataclass(frozen=True, slots=True)
class WorkshopDirectMessageArchiveMutation:
    """One principal's effective direct-message visibility transition."""

    channel_id: ChannelId
    archived: bool
    changed: bool
    occurred_at: datetime


def _normalize_operation_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise WorkshopDirectMessageArchiveConflict("client_operation_id must be a non-empty string")
    return value.strip()


class WorkshopDirectMessageArchiveService:
    """Archive direct conversations for one principal without changing the channel."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def archive(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        *,
        client_operation_id: object,
    ) -> WorkshopDirectMessageArchiveMutation:
        return await self._change(
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
    ) -> WorkshopDirectMessageArchiveMutation:
        return await self._change(
            principal_id,
            channel_id,
            archived=False,
            client_operation_id=client_operation_id,
        )

    async def _change(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        *,
        archived: bool,
        client_operation_id: object,
    ) -> WorkshopDirectMessageArchiveMutation:
        operation_id = _normalize_operation_id(client_operation_id)
        operation = "archive" if archived else "restore"
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            workshop_id, effective_archived = await self._resolve_state(principal_id, channel_id)
            idempotency_key = f"workshop-client:direct-message:{principal_id}:{operation_id}"
            existing = await self._store.event_by_idempotency_key(idempotency_key)
            if existing is not None:
                result = self._result_from_event(
                    existing,
                    principal_id=principal_id,
                    workshop_id=workshop_id,
                    channel_id=channel_id,
                    archived=archived,
                )
                await connection.rollback()
                return result
            if effective_archived == archived:
                state = "already archived" if archived else "not archived"
                raise WorkshopDirectMessageArchiveConflict(f"Direct message is {state}")
            now = datetime.now(UTC)
            event = EventEnvelope.create(
                event_type=(
                    WorkshopEventType.PRINCIPAL_DIRECT_MESSAGE_ARCHIVED
                    if archived
                    else WorkshopEventType.PRINCIPAL_DIRECT_MESSAGE_RESTORED
                ),
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="principal_direct_message",
                aggregate_id=channel_id,
                actor_principal_id=principal_id,
                occurred_at=now,
                idempotency_key=idempotency_key,
                payload={},
                metadata={"source": "workshop_client"},
            )
            result = await self._store.append_in_transaction(event)
            if not result.inserted:
                raise WorkshopDirectMessageArchiveError("New direct-message archive event already exists")
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            await connection.commit()
            return WorkshopDirectMessageArchiveMutation(channel_id, archived, True, now)
        except WorkshopDirectMessageArchiveError:
            await connection.rollback()
            raise
        except Exception as exc:
            await connection.rollback()
            raise WorkshopDirectMessageArchiveStorageError(
                f"Direct message {operation} could not be persisted"
            ) from exc

    async def _resolve_state(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
    ) -> tuple[WorkshopId, bool]:
        if not isinstance(principal_id, PrincipalId) or not isinstance(channel_id, ChannelId):
            raise WorkshopDirectMessageArchiveAccessDenied("Canonical identities are required")
        async with self._store.connection.execute(
            "SELECT c.workshop_id, dma.archived_event_position, "
            "EXISTS(SELECT 1 FROM messages m WHERE m.channel_id = c.id "
            "AND m.author_principal_id != ? "
            "AND m.created_event_position > dma.archived_event_position), "
            "EXISTS(SELECT 1 FROM channel_agents ca "
            "JOIN agent_definitions ad ON ad.agent_id = ca.agent_id "
            "WHERE ca.channel_id = c.id AND ca.detached_at IS NULL "
            "AND ad.lifecycle_state = 'archived'), "
            "EXISTS(SELECT 1 FROM channel_agents ca "
            "JOIN agent_definitions ad ON ad.agent_id = ca.agent_id "
            "WHERE ca.channel_id = c.id AND ca.detached_at IS NULL "
            "AND ad.lifecycle_state = 'active') "
            "FROM channels c JOIN channel_memberships cm ON cm.channel_id = c.id "
            "LEFT JOIN principal_direct_message_archives dma "
            "ON dma.principal_id = cm.principal_id AND dma.channel_id = c.id "
            "WHERE c.id = ? AND c.kind = 'direct' AND c.archived_at IS NULL "
            "AND cm.principal_id = ?",
            (principal_id, channel_id, principal_id),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1 or bool(rows[0][3]):
            raise WorkshopDirectMessageArchiveAccessDenied("Direct message is unavailable")
        if not bool(rows[0][4]) and not await is_canonical_human_direct_channel(self._store, channel_id):
            raise WorkshopDirectMessageArchiveAccessDenied("Direct message is unavailable")
        archive_position = int(rows[0][1]) if rows[0][1] is not None else None
        resurfaced = bool(rows[0][2]) if archive_position is not None else False
        return WorkshopId(str(rows[0][0])), archive_position is not None and not resurfaced

    @staticmethod
    def _result_from_event(
        stored: StoredEvent,
        *,
        principal_id: PrincipalId,
        workshop_id: WorkshopId,
        channel_id: ChannelId,
        archived: bool,
    ) -> WorkshopDirectMessageArchiveMutation:
        envelope = stored.envelope
        expected = (
            WorkshopEventType.PRINCIPAL_DIRECT_MESSAGE_ARCHIVED
            if archived
            else WorkshopEventType.PRINCIPAL_DIRECT_MESSAGE_RESTORED
        )
        if (
            envelope.event_type != expected
            or envelope.event_version != 1
            or envelope.workshop_id != workshop_id
            or envelope.aggregate_type != "principal_direct_message"
            or envelope.aggregate_id != channel_id
            or envelope.actor_principal_id != principal_id
            or envelope.payload
        ):
            raise WorkshopDirectMessageArchiveConflict(
                "client_operation_id is already bound to a different direct-message operation"
            )
        return WorkshopDirectMessageArchiveMutation(channel_id, archived, False, envelope.occurred_at)
