"""Contracts for canonical, explicit Workshop channel authorization."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.workshop.authorization import CanonicalChannelAuthorizer
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import ChannelId, ChannelMembershipId, PrincipalId, WorkshopId
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore
from kai.workshop.timeline import TimelineAccessDeniedError, read_channel_timeline

_NOW = datetime(2026, 8, 11, 13, 0, tzinfo=UTC)


async def _open_store(
    path: Path,
) -> tuple[WorkshopEventStore, PrincipalId, ChannelId, PrincipalId, ChannelId, PrincipalId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Admin", "admin", "telegram", "101", "101"),
            BootstrapHuman("Member", "member", "telegram", "202", "202"),
        ),
    )
    async with store.connection.execute(
        "SELECT e.external_subject, e.principal_id, b.channel_id "
        "FROM external_identities e JOIN channel_bindings b "
        "ON b.transport = e.provider AND b.external_channel_id = e.external_subject "
        "ORDER BY e.external_subject"
    ) as cursor:
        rows = list(await cursor.fetchall())
    async with store.connection.execute("SELECT id FROM principals WHERE kind = 'agent'") as cursor:
        agent_row = await cursor.fetchone()
    assert len(rows) == 2
    assert agent_row is not None
    return (
        store,
        PrincipalId(str(rows[0][1])),
        ChannelId(str(rows[0][2])),
        PrincipalId(str(rows[1][1])),
        ChannelId(str(rows[1][2])),
        PrincipalId(str(agent_row[0])),
    )


class TestCanonicalChannelAuthorizer:
    async def test_human_can_read_own_direct_channel_only(self, tmp_path: Path):
        store, admin_id, admin_channel, member_id, member_channel, _ = await _open_store(tmp_path / "kai.db")
        try:
            authorizer = CanonicalChannelAuthorizer(store)

            assert await authorizer.can_read_channel(admin_id, admin_channel) is True
            assert await authorizer.can_read_channel(member_id, member_channel) is True
            assert await authorizer.can_read_channel(admin_id, member_channel) is False
            assert await authorizer.can_read_channel(member_id, admin_channel) is False
        finally:
            await store.close()

    async def test_workshop_admin_role_does_not_grant_cross_channel_access(self, tmp_path: Path):
        store, admin_id, _, _, member_channel, _ = await _open_store(tmp_path / "kai.db")
        try:
            async with store.connection.execute(
                "SELECT role FROM workshop_memberships WHERE principal_id = ?",
                (admin_id,),
            ) as cursor:
                role = await cursor.fetchone()
            assert role is not None and role[0] == "admin"

            assert await CanonicalChannelAuthorizer(store).can_read_channel(admin_id, member_channel) is False
        finally:
            await store.close()

    async def test_attached_agent_can_read_each_channel_it_participates_in(self, tmp_path: Path):
        store, _, first_channel, _, second_channel, agent_id = await _open_store(tmp_path / "kai.db")
        try:
            authorizer = CanonicalChannelAuthorizer(store)

            assert await authorizer.can_read_channel(agent_id, first_channel) is True
            assert await authorizer.can_read_channel(agent_id, second_channel) is True
        finally:
            await store.close()

    async def test_unknown_principal_or_channel_fails_closed(self, tmp_path: Path):
        store, admin_id, admin_channel, _, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            authorizer = CanonicalChannelAuthorizer(store)

            assert await authorizer.can_read_channel(PrincipalId.new(), admin_channel) is False
            assert await authorizer.can_read_channel(admin_id, ChannelId.new()) is False
        finally:
            await store.close()

    async def test_cross_workshop_membership_record_does_not_grant_access(self, tmp_path: Path):
        store, admin_id, _, _, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            other_workshop = WorkshopId.new()
            other_channel = ChannelId.new()
            await store.connection.execute(
                "INSERT INTO workshops (id, name, created_at) VALUES (?, ?, ?)",
                (other_workshop, "Other", _NOW.isoformat()),
            )
            await store.connection.execute(
                "INSERT INTO channels (id, workshop_id, kind, name, created_at) VALUES (?, ?, ?, ?, ?)",
                (other_channel, other_workshop, "direct", "Other", _NOW.isoformat()),
            )
            await store.connection.execute(
                "INSERT INTO channel_memberships (id, channel_id, principal_id, role, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (ChannelMembershipId.new(), other_channel, admin_id, "owner", _NOW.isoformat()),
            )
            await store.connection.commit()

            assert await CanonicalChannelAuthorizer(store).can_read_channel(admin_id, other_channel) is False
        finally:
            await store.close()

    async def test_projection_rebuild_restores_explicit_memberships(self, tmp_path: Path):
        store, admin_id, admin_channel, member_id, member_channel, _ = await _open_store(tmp_path / "kai.db")
        try:
            await store.connection.execute("DELETE FROM channel_memberships")
            await store.connection.commit()

            await store.rebuild_projection(CanonicalConversationProjection())
            authorizer = CanonicalChannelAuthorizer(store)

            assert await authorizer.can_read_channel(admin_id, admin_channel) is True
            assert await authorizer.can_read_channel(member_id, member_channel) is True
            assert await authorizer.can_read_channel(admin_id, member_channel) is False
        finally:
            await store.close()

    async def test_timeline_uses_concrete_channel_policy(self, tmp_path: Path):
        store, admin_id, admin_channel, member_id, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            await record_inbound_message(
                store,
                InboundMessage("telegram", "9001", "42", "101", "101", "Private", _NOW),
            )
            authorizer = CanonicalChannelAuthorizer(store)

            own_page = await read_channel_timeline(
                store,
                principal_id=admin_id,
                channel_id=admin_channel,
                authorizer=authorizer,
            )
            assert [message.body for message in own_page.messages] == ["Private"]

            with pytest.raises(TimelineAccessDeniedError):
                await read_channel_timeline(
                    store,
                    principal_id=member_id,
                    channel_id=admin_channel,
                    authorizer=authorizer,
                )
        finally:
            await store.close()
