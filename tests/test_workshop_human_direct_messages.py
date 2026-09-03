"""Contracts for canonical private conversations between Workshop humans."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.workshop.authorization import CanonicalChannelAuthorizer
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.conversation_commands import (
    ConversationCommandDisposition,
    WorkshopConversationCommandService,
)
from kai.workshop.domain import ChannelId, PrincipalId, WorkshopId
from kai.workshop.human_direct_messages import (
    WorkshopHumanDirectMessageAccessDenied,
    WorkshopHumanDirectMessageService,
    is_canonical_human_direct_channel,
)
from kai.workshop.inbound import ClientInboundMessage
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


async def _open_store(
    path: Path,
) -> tuple[WorkshopEventStore, WorkshopId, PrincipalId, PrincipalId]:
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
    return store, workshop_id, PrincipalId(str(rows[0][1])), PrincipalId(str(rows[1][1]))


class TestWorkshopHumanDirectMessageAuthority:
    async def test_discovers_same_workshop_peers_and_creates_one_unordered_pair(
        self,
        tmp_path: Path,
    ) -> None:
        store, workshop_id, alice_id, bob_id = await _open_store(tmp_path / "kai.db")
        service = WorkshopHumanDirectMessageService(store)
        try:
            initial = await service.list_peers(alice_id, workshop_id)
            assert [(peer.principal_id, peer.display_name, peer.handle, peer.channel_id) for peer in initial.peers] == [
                (bob_id, "Bob", "bob", None)
            ]

            created = await service.start(alice_id, workshop_id, bob_id)
            reverse = await service.start(bob_id, workshop_id, alice_id)

            assert created.created is True
            assert reverse.created is False
            assert reverse.channel_id == created.channel_id
            assert await is_canonical_human_direct_channel(store, created.channel_id)
            assert (await service.list_peers(alice_id, workshop_id)).peers[0].channel_id == created.channel_id
            async with store.connection.execute(
                "SELECT cm.principal_id, cm.role FROM channel_memberships cm "
                "WHERE cm.channel_id = ? ORDER BY cm.principal_id",
                (created.channel_id,),
            ) as cursor:
                assert [(str(row[0]), str(row[1])) for row in await cursor.fetchall()] == sorted(
                    [(str(alice_id), "owner"), (str(bob_id), "owner")]
                )
            async with store.connection.execute(
                "SELECT (SELECT COUNT(*) FROM channel_agents WHERE channel_id = ?), "
                "(SELECT COUNT(*) FROM channel_agent_runtime_assignments WHERE channel_id = ?), "
                "(SELECT COUNT(*) FROM channel_bindings WHERE channel_id = ?)",
                (created.channel_id, created.channel_id, created.channel_id),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == (0, 0, 0)
        finally:
            await store.close()

    async def test_pair_is_private_message_only_and_survives_projection_rebuild(
        self,
        tmp_path: Path,
    ) -> None:
        store, workshop_id, alice_id, bob_id = await _open_store(tmp_path / "kai.db")
        try:
            conversation = await WorkshopHumanDirectMessageService(store).start(alice_id, workshop_id, bob_id)
            outsider = PrincipalId.new()
            await store.connection.execute(
                "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, 'human', 'Outsider', ?)",
                (outsider, _NOW.isoformat()),
            )
            await store.connection.execute(
                "INSERT INTO workshop_memberships (id, workshop_id, principal_id, role, created_at) "
                "VALUES (?, ?, ?, 'admin', ?)",
                (f"wmb_{'1' * 32}", workshop_id, outsider, _NOW.isoformat()),
            )
            await store.connection.commit()

            authorizer = CanonicalChannelAuthorizer(store)
            assert await authorizer.can_submit_command(alice_id, conversation.channel_id)
            assert await authorizer.can_submit_command(bob_id, conversation.channel_id)
            assert not await authorizer.can_read_channel(outsider, conversation.channel_id)
            assert not await authorizer.can_submit_command(outsider, conversation.channel_id)

            accepted = await WorkshopConversationCommandService(store).accept_client(
                ClientInboundMessage(
                    principal_id=alice_id,
                    channel_id=conversation.channel_id,
                    client_message_id="alice-to-bob-1",
                    body="Private hello",
                    occurred_at=_NOW,
                )
            )
            assert accepted.command.disposition == ConversationCommandDisposition.MESSAGE_ONLY
            assert accepted.command.runs == ()
            assert accepted.runtime_profile_ids == ()
            assert accepted.deliveries == ()
            async with store.connection.execute(
                "SELECT COUNT(*) FROM runs WHERE channel_id = ?", (conversation.channel_id,)
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0

            await store.rebuild_projection(CanonicalConversationProjection())
            assert await is_canonical_human_direct_channel(store, conversation.channel_id)
            assert await CanonicalChannelAuthorizer(store).can_read_channel(bob_id, conversation.channel_id)
        finally:
            await store.close()

    async def test_rejects_self_agent_unknown_and_cross_workshop_targets(self, tmp_path: Path) -> None:
        store, workshop_id, alice_id, _ = await _open_store(tmp_path / "kai.db")
        service = WorkshopHumanDirectMessageService(store)
        async with store.connection.execute("SELECT principal_id FROM agents LIMIT 1") as cursor:
            agent_id = PrincipalId(str((await cursor.fetchone())[0]))
        try:
            for target in (alice_id, agent_id, PrincipalId.new()):
                with pytest.raises(WorkshopHumanDirectMessageAccessDenied):
                    await service.start(alice_id, workshop_id, target)
            with pytest.raises(WorkshopHumanDirectMessageAccessDenied):
                await service.list_peers(alice_id, WorkshopId.new())
        finally:
            await store.close()

    async def test_malformed_direct_channel_never_becomes_a_submit_lane(self, tmp_path: Path) -> None:
        store, _, alice_id, _ = await _open_store(tmp_path / "kai.db")
        try:
            channel_id = ChannelId.new()
            async with store.connection.execute(
                "SELECT workshop_id FROM workshop_memberships WHERE principal_id = ?", (alice_id,)
            ) as cursor:
                workshop_id = WorkshopId(str((await cursor.fetchone())[0]))
            await store.connection.execute(
                "INSERT INTO channels (id, workshop_id, kind, name, created_at) VALUES (?, ?, 'direct', 'Direct', ?)",
                (channel_id, workshop_id, _NOW.isoformat()),
            )
            await store.connection.commit()
            assert not await is_canonical_human_direct_channel(store, channel_id)
            assert not await CanonicalChannelAuthorizer(store).can_submit_command(alice_id, channel_id)
        finally:
            await store.close()
