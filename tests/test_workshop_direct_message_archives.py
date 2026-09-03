"""Contracts for principal-scoped Workshop direct-message archival."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.client_api import _read_agent_lifecycle_event_records
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.direct_message_archives import (
    WorkshopDirectMessageArchiveAccessDenied,
    WorkshopDirectMessageArchiveConflict,
    WorkshopDirectMessageArchiveService,
)
from kai.workshop.domain import ChannelId, ChannelMembershipId, PrincipalId, WorkshopId
from kai.workshop.human_direct_messages import WorkshopHumanDirectMessageService
from kai.workshop.inbound import ClientInboundMessage
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 9, 3, 15, 0, tzinfo=UTC)


async def _open_human_conversation(
    path: Path,
) -> tuple[WorkshopEventStore, WorkshopId, PrincipalId, PrincipalId, ChannelId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Alice", "admin", "telegram", "101", "101"),
            BootstrapHuman("Bob", "member", "telegram", "202", "202"),
        ),
    )
    async with store.connection.execute("SELECT id FROM workshops") as cursor:
        workshop_id = WorkshopId(str((await cursor.fetchone())[0]))
    async with store.connection.execute(
        "SELECT display_name, id FROM principals WHERE kind = 'human' ORDER BY display_name"
    ) as cursor:
        rows = list(await cursor.fetchall())
    alice_id = PrincipalId(str(rows[0][1]))
    bob_id = PrincipalId(str(rows[1][1]))
    conversation = await WorkshopHumanDirectMessageService(store).start(alice_id, workshop_id, bob_id)
    return store, workshop_id, alice_id, bob_id, conversation.channel_id


async def _effective_archived(store: WorkshopEventStore, principal_id: PrincipalId, channel_id: ChannelId) -> bool:
    async with store.connection.execute(
        "SELECT EXISTS(SELECT 1 FROM principal_direct_message_archives dma "
        "WHERE dma.principal_id = ? AND dma.channel_id = ? "
        "AND NOT EXISTS(SELECT 1 FROM messages m WHERE m.channel_id = dma.channel_id "
        "AND m.author_principal_id != dma.principal_id "
        "AND m.created_event_position > dma.archived_event_position))",
        (principal_id, channel_id),
    ) as cursor:
        return bool((await cursor.fetchone())[0])


class TestWorkshopDirectMessageArchiveAuthority:
    async def test_archive_is_principal_scoped_idempotent_and_replayable(self, tmp_path: Path) -> None:
        store, _, alice_id, bob_id, channel_id = await _open_human_conversation(tmp_path / "kai.db")
        service = WorkshopDirectMessageArchiveService(store)
        try:
            first = await service.archive(
                alice_id,
                channel_id,
                client_operation_id="alice-archive-1",
            )
            replay = await service.archive(
                alice_id,
                channel_id,
                client_operation_id="alice-archive-1",
            )
            assert first.archived is True and first.changed is True
            assert replay.archived is True and replay.changed is False
            assert await _effective_archived(store, alice_id, channel_id)
            assert not await _effective_archived(store, bob_id, channel_id)

            with pytest.raises(WorkshopDirectMessageArchiveConflict, match="not archived"):
                await service.restore(
                    bob_id,
                    channel_id,
                    client_operation_id="bob-restore-1",
                )

            await store.rebuild_projection(CanonicalConversationProjection())
            assert await _effective_archived(store, alice_id, channel_id)

            restored = await service.restore(
                alice_id,
                channel_id,
                client_operation_id="alice-restore-1",
            )
            assert restored.archived is False and restored.changed is True
            assert not await _effective_archived(store, alice_id, channel_id)
        finally:
            await store.close()

    async def test_peer_message_resurfaces_and_allows_a_new_archive(self, tmp_path: Path) -> None:
        store, _, alice_id, bob_id, channel_id = await _open_human_conversation(tmp_path / "kai.db")
        service = WorkshopDirectMessageArchiveService(store)
        try:
            await service.archive(alice_id, channel_id, client_operation_id="alice-archive-1")
            await WorkshopConversationCommandService(store).accept_client(
                ClientInboundMessage(
                    principal_id=bob_id,
                    channel_id=channel_id,
                    client_message_id="bob-resurface-1",
                    body="This should resurface for Alice",
                    occurred_at=_NOW,
                )
            )
            assert not await _effective_archived(store, alice_id, channel_id)

            second = await service.archive(
                alice_id,
                channel_id,
                client_operation_id="alice-archive-2",
            )
            assert second.changed is True
            assert await _effective_archived(store, alice_id, channel_id)
        finally:
            await store.close()

    async def test_archive_and_peer_resurface_emit_private_navigation_signals(self, tmp_path: Path) -> None:
        store, workshop_id, alice_id, bob_id, channel_id = await _open_human_conversation(tmp_path / "kai.db")
        service = WorkshopDirectMessageArchiveService(store)
        try:
            async with store.connection.execute("SELECT MAX(position) FROM event_log") as cursor:
                before_archive = int((await cursor.fetchone())[0])
            await service.archive(alice_id, channel_id, client_operation_id="alice-archive-signal")
            alice_events, archive_position, _ = await _read_agent_lifecycle_event_records(
                store,
                workshop_id=str(workshop_id),
                principal_id=alice_id,
                role="admin",
                after_position=before_archive,
            )
            bob_events, _, _ = await _read_agent_lifecycle_event_records(
                store,
                workshop_id=str(workshop_id),
                principal_id=bob_id,
                role="member",
                after_position=before_archive,
            )
            assert [(event.event_name, event.event_type) for event in alice_events] == [
                ("workshop.navigation.changed", "principal_direct_message.archived")
            ]
            assert bob_events == ()

            await WorkshopConversationCommandService(store).accept_client(
                ClientInboundMessage(
                    principal_id=bob_id,
                    channel_id=channel_id,
                    client_message_id="bob-resurface-signal",
                    body="Wake this archived conversation",
                    occurred_at=_NOW,
                )
            )
            resurfaced, _, _ = await _read_agent_lifecycle_event_records(
                store,
                workshop_id=str(workshop_id),
                principal_id=alice_id,
                role="admin",
                after_position=archive_position,
            )
            assert [(event.event_name, event.event_type) for event in resurfaced] == [
                ("workshop.navigation.changed", "message.created")
            ]
        finally:
            await store.close()

    async def test_active_agent_direct_qualifies_but_archived_agent_does_not(self, tmp_path: Path) -> None:
        store, _, alice_id, _, human_channel_id = await _open_human_conversation(tmp_path / "kai.db")
        service = WorkshopDirectMessageArchiveService(store)
        try:
            async with store.connection.execute(
                "SELECT c.id, ad.id FROM channels c "
                "JOIN channel_memberships cm ON cm.channel_id = c.id "
                "JOIN channel_agents ca ON ca.channel_id = c.id AND ca.detached_at IS NULL "
                "JOIN agent_definitions ad ON ad.agent_id = ca.agent_id "
                "WHERE c.kind = 'direct' AND cm.principal_id = ? AND c.id != ?",
                (alice_id, human_channel_id),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            agent_channel_id = ChannelId(str(row[0]))
            definition_id = str(row[1])

            archived = await service.archive(
                alice_id,
                agent_channel_id,
                client_operation_id="alice-agent-archive-1",
            )
            assert archived.archived is True and archived.changed is True
            await service.restore(
                alice_id,
                agent_channel_id,
                client_operation_id="alice-agent-restore-1",
            )

            await store.connection.execute(
                "UPDATE agent_definitions SET lifecycle_state = 'archived' WHERE id = ?",
                (definition_id,),
            )
            await store.connection.commit()
            with pytest.raises(WorkshopDirectMessageArchiveAccessDenied):
                await service.archive(
                    alice_id,
                    agent_channel_id,
                    client_operation_id="alice-archived-agent-rejected",
                )
        finally:
            await store.close()

    async def test_rejects_groups_unknown_channels_and_nonmembers(self, tmp_path: Path) -> None:
        store, workshop_id, alice_id, _, _ = await _open_human_conversation(tmp_path / "kai.db")
        service = WorkshopDirectMessageArchiveService(store)
        try:
            nondirect_id = ChannelId.new()
            await store.connection.execute(
                "INSERT INTO channels (id, workshop_id, kind, name, created_at) "
                "VALUES (?, ?, 'group', 'Not direct', ?)",
                (nondirect_id, workshop_id, _NOW.isoformat()),
            )
            await store.connection.execute(
                "INSERT INTO channel_memberships (id, channel_id, principal_id, role, created_at) "
                "VALUES (?, ?, ?, 'owner', ?)",
                (ChannelMembershipId.new(), nondirect_id, alice_id, _NOW.isoformat()),
            )
            await store.connection.commit()
            for channel_id in (nondirect_id, ChannelId.new()):
                with pytest.raises(WorkshopDirectMessageArchiveAccessDenied):
                    await service.archive(
                        alice_id,
                        channel_id,
                        client_operation_id=f"reject-{channel_id}",
                    )
        finally:
            await store.close()
