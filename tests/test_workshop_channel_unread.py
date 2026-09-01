"""Canonical channel unread-position authority contracts."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.channel_lifecycle import WorkshopChannelLifecycleService
from kai.workshop.channel_unread import (
    WorkshopChannelUnreadAccessDenied,
    WorkshopChannelUnreadConflict,
    WorkshopChannelUnreadService,
)
from kai.workshop.client_api import register_workshop_read_routes
from kai.workshop.domain import AgentId, ChannelId, MessageId, PrincipalId
from kai.workshop.inbound import ClientInboundMessage, record_client_inbound_message_in_transaction
from kai.workshop.outbound import OutboundMessage, record_outbound_message
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.schema import migrate_workshop_schema
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id

_NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


@dataclass
class _Authenticator:
    principals_by_token: dict[str, PrincipalId]

    async def authenticate(self, request: web.Request) -> PrincipalId | None:
        scheme, separator, token = request.headers.get("Authorization", "").partition(" ")
        return self.principals_by_token.get(token) if separator and scheme == "Bearer" else None

    async def authenticate_token(self, token: str) -> PrincipalId | None:
        return self.principals_by_token.get(token)


async def _identity_for(store: WorkshopEventStore, subject: str) -> tuple[PrincipalId, ChannelId]:
    async with store.connection.execute(
        "SELECT e.principal_id, b.channel_id FROM external_identities e "
        "JOIN channel_bindings b ON b.transport = e.provider "
        "AND b.external_channel_id = e.external_subject "
        "WHERE e.provider = 'desktop' AND e.external_subject = ?",
        (subject,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return PrincipalId(str(row[0])), ChannelId(str(row[1]))


async def _context(
    path: Path,
) -> tuple[WorkshopEventStore, PrincipalId, PrincipalId, ChannelId, AgentId]:
    store = await WorkshopEventStore.open(path)
    result = await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Daniel", "admin", "desktop", "daniel", "daniel", profile_id(101)),
            BootstrapHuman("Scott", "member", "desktop", "scott", "scott", profile_id(202)),
        ),
    )
    daniel_id, daniel_direct = await _identity_for(store, "daniel")
    scott_id, _ = await _identity_for(store, "scott")
    lifecycle = WorkshopChannelLifecycleService(store)
    group = await lifecycle.create_group(
        daniel_id,
        name="Unread authority",
        agent_ids=[result.agent_id],
        origin_channel_id=daniel_direct,
    )
    membership = await lifecycle.human_members(daniel_id, group.channel_id)
    await lifecycle.add_human_member(
        daniel_id,
        group.channel_id,
        scott_id,
        expected_state_version=membership.state_version,
        client_operation_id="add-scott-for-unread",
    )
    return store, daniel_id, scott_id, group.channel_id, result.agent_id


async def _record(
    store: WorkshopEventStore,
    principal_id: PrincipalId,
    channel_id: ChannelId,
    identity: str,
    body: str,
    *,
    occurred_at: datetime = _NOW,
    thread_root_id: MessageId | None = None,
) -> MessageId:
    try:
        await store.connection.execute("BEGIN IMMEDIATE")
        result = await record_client_inbound_message_in_transaction(
            store,
            ClientInboundMessage(
                principal_id,
                channel_id,
                identity,
                body,
                occurred_at,
                thread_root_id=thread_root_id,
            ),
        )
        await store.connection.commit()
    except Exception:
        await store.connection.rollback()
        raise
    return MessageId(str(result.event.envelope.aggregate_id))


async def _open_client(store: WorkshopEventStore, authenticator: _Authenticator) -> TestClient:
    app = web.Application()
    register_workshop_read_routes(
        app,
        store=store,
        authenticator=authenticator,
        request_lock=asyncio.Lock(),
        event_poll_interval=0.01,
        event_heartbeat_interval=0.05,
        event_authentication_recheck_interval=0.01,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def _next_sse_event(response) -> dict[str, object]:
    event_name: str | None = None
    data: str | None = None
    while True:
        line = (await asyncio.wait_for(response.content.readline(), timeout=1)).decode().rstrip("\r\n")
        if not line:
            if event_name is not None and data is not None:
                return {"event": event_name, "data": json.loads(data)}
            continue
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data = line.removeprefix("data: ")


class TestChannelUnreadAuthority:
    async def test_positions_are_private_and_own_messages_and_threads_are_excluded(self, tmp_path: Path) -> None:
        store, daniel_id, scott_id, channel_id, agent_id = await _context(tmp_path / "kai.db")
        service = WorkshopChannelUnreadService(store)
        try:
            daniel_message = await _record(store, daniel_id, channel_id, "daniel-1", "hello Scott")
            scott_message = await _record(
                store,
                scott_id,
                channel_id,
                "scott-1",
                "hello Daniel",
                occurred_at=_NOW + timedelta(seconds=1),
            )
            reply = await record_outbound_message(
                store,
                OutboundMessage(
                    daniel_message,
                    "agent response",
                    _NOW + timedelta(seconds=2),
                    agent_id,
                ),
            )
            await _record(
                store,
                daniel_id,
                channel_id,
                "daniel-thread",
                "thread reply",
                occurred_at=_NOW + timedelta(seconds=3),
                thread_root_id=daniel_message,
            )

            daniel = await service.channel(daniel_id, channel_id)
            scott = await service.channel(scott_id, channel_id)
            assert daniel.unread_count == 2
            assert daniel.first_unread_message_id == scott_message
            assert scott.unread_count == 2
            assert scott.first_unread_message_id == daniel_message
            assert reply.event.envelope.aggregate_id != scott.first_unread_message_id
        finally:
            await store.close()

    async def test_advance_is_exact_monotonic_idempotent_and_replayable(self, tmp_path: Path) -> None:
        store, daniel_id, scott_id, channel_id, _agent_id = await _context(tmp_path / "kai.db")
        service = WorkshopChannelUnreadService(store)
        try:
            first = await _record(store, daniel_id, channel_id, "first", "first")
            second = await _record(
                store,
                daniel_id,
                channel_id,
                "second",
                "second",
                occurred_at=_NOW + timedelta(seconds=1),
            )
            before = await service.channel(scott_id, channel_id)
            mutation = await service.advance(
                scott_id,
                channel_id,
                second,
                expected_state_version=before.state_version,
                client_operation_id="read-through-second",
            )
            assert mutation.replayed is False
            assert mutation.state.unread_count == 0
            assert mutation.state.read_through_message_id == second
            replay = await service.advance(
                scott_id,
                channel_id,
                second,
                expected_state_version=before.state_version,
                client_operation_id="read-through-second",
            )
            assert replay.replayed is True
            assert replay.state == mutation.state
            with pytest.raises(WorkshopChannelUnreadConflict, match="backward"):
                await service.advance(
                    scott_id,
                    channel_id,
                    first,
                    expected_state_version=mutation.state.state_version,
                    client_operation_id="backward",
                )
            with pytest.raises(WorkshopChannelUnreadConflict, match="revision"):
                await service.advance(
                    scott_id,
                    channel_id,
                    second,
                    expected_state_version=before.state_version,
                    client_operation_id="stale-version",
                )
        finally:
            await store.close()

    async def test_cross_channel_and_cross_principal_boundaries_fail_closed(self, tmp_path: Path) -> None:
        store, daniel_id, scott_id, channel_id, _agent_id = await _context(tmp_path / "kai.db")
        service = WorkshopChannelUnreadService(store)
        try:
            message = await _record(store, daniel_id, channel_id, "private", "private")
            _scott, scott_direct = await _identity_for(store, "scott")
            state = await service.channel(scott_id, scott_direct)
            with pytest.raises(WorkshopChannelUnreadAccessDenied, match="unavailable"):
                await service.advance(
                    scott_id,
                    scott_direct,
                    message,
                    expected_state_version=state.state_version,
                    client_operation_id="cross-channel",
                )
            unknown = ChannelId.new()
            with pytest.raises(WorkshopChannelUnreadAccessDenied, match="unavailable"):
                await service.channel(scott_id, unknown)
        finally:
            await store.close()

    async def test_membership_readdition_uses_a_new_baseline(self, tmp_path: Path) -> None:
        store, daniel_id, scott_id, channel_id, _agent_id = await _context(tmp_path / "kai.db")
        lifecycle = WorkshopChannelLifecycleService(store)
        service = WorkshopChannelUnreadService(store)
        try:
            members = await lifecycle.human_members(daniel_id, channel_id)
            await lifecycle.remove_human_member(
                daniel_id,
                channel_id,
                scott_id,
                expected_state_version=members.state_version,
                client_operation_id="remove-scott",
            )
            await _record(store, daniel_id, channel_id, "while-away", "while away")
            members = await lifecycle.human_members(daniel_id, channel_id)
            await lifecycle.add_human_member(
                daniel_id,
                channel_id,
                scott_id,
                expected_state_version=members.state_version,
                client_operation_id="readd-scott",
            )
            state = await service.channel(scott_id, channel_id)
            assert state.unread_count == 0
            await _record(
                store,
                daniel_id,
                channel_id,
                "after-return",
                "after return",
                occurred_at=_NOW + timedelta(seconds=1),
            )
            assert (await service.channel(scott_id, channel_id)).unread_count == 1
        finally:
            await store.close()

    async def test_archived_channels_leave_the_active_snapshot_without_losing_position(
        self,
        tmp_path: Path,
    ) -> None:
        store, daniel_id, scott_id, channel_id, _agent_id = await _context(tmp_path / "kai.db")
        lifecycle = WorkshopChannelLifecycleService(store)
        service = WorkshopChannelUnreadService(store)
        try:
            message = await _record(store, daniel_id, channel_id, "before-archive", "before archive")
            before = await service.channel(scott_id, channel_id)
            assert before.first_unread_message_id == message
            await lifecycle.archive(
                daniel_id,
                channel_id,
                client_operation_id="archive-unread-channel",
            )
            assert channel_id not in {state.channel_id for state in (await service.snapshot(scott_id)).channels}
            archived = await service.channel(scott_id, channel_id)
            assert archived.archived is True
            assert archived.first_unread_message_id == message
            await lifecycle.restore(
                daniel_id,
                channel_id,
                client_operation_id="restore-unread-channel",
            )
            restored = await service.channel(scott_id, channel_id)
            assert restored.archived is False
            assert restored.first_unread_message_id == message
        finally:
            await store.close()

    async def test_projection_rebuild_reproduces_positions_counts_and_anchors(self, tmp_path: Path) -> None:
        store, daniel_id, scott_id, channel_id, _agent_id = await _context(tmp_path / "kai.db")
        service = WorkshopChannelUnreadService(store)
        try:
            first = await _record(store, daniel_id, channel_id, "one", "one")
            await _record(
                store,
                daniel_id,
                channel_id,
                "two",
                "two",
                occurred_at=_NOW + timedelta(seconds=1),
            )
            before = await service.channel(scott_id, channel_id)
            advanced = await service.advance(
                scott_id,
                channel_id,
                first,
                expected_state_version=before.state_version,
                client_operation_id="rebuild-boundary",
            )
            expected = advanced.state
            await store.rebuild_projection(CanonicalConversationProjection())
            assert await service.channel(scott_id, channel_id) == expected
        finally:
            await store.close()

    async def test_upgrade_baseline_marks_existing_history_read(self, tmp_path: Path) -> None:
        path = tmp_path / "kai.db"
        store, daniel_id, scott_id, channel_id, _agent_id = await _context(path)
        try:
            await _record(store, daniel_id, channel_id, "historical", "historical")
            await store.connection.execute("DROP TABLE channel_read_positions")
            await store.connection.execute("DROP TABLE channel_unread_migration_baselines")
            await store.connection.execute("DELETE FROM workshop_schema_migrations WHERE version = 60")
            await store.connection.execute("DELETE FROM workshop_schema_migrations WHERE version = 61")
            await store.connection.commit()
            await migrate_workshop_schema(store.connection)
            state = await WorkshopChannelUnreadService(store).channel(scott_id, channel_id)
            assert state.unread_count == 0
            assert state.read_through_event_position == state.membership_baseline_event_position
            assert state.last_event_position == state.membership_baseline_event_position
            await store.rebuild_projection(CanonicalConversationProjection())
            assert await WorkshopChannelUnreadService(store).channel(scott_id, channel_id) == state
        finally:
            await store.close()

    async def test_version_sixty_one_repairs_an_installed_rebuild_boundary(self, tmp_path: Path) -> None:
        store, _daniel_id, scott_id, channel_id, _agent_id = await _context(tmp_path / "kai.db")
        try:
            state = await WorkshopChannelUnreadService(store).channel(scott_id, channel_id)
            assert state.membership_baseline_event_position > 0
            await store.connection.execute(
                "UPDATE channel_read_positions SET last_event_position = ? WHERE principal_id = ? AND channel_id = ?",
                (state.membership_baseline_event_position - 1, scott_id, channel_id),
            )
            await store.connection.execute("DELETE FROM workshop_schema_migrations WHERE version = 61")
            await store.connection.commit()

            await migrate_workshop_schema(store.connection)

            repaired = await WorkshopChannelUnreadService(store).channel(scott_id, channel_id)
            assert repaired.last_event_position == repaired.membership_baseline_event_position
        finally:
            await store.close()

    async def test_authenticated_api_and_private_event_stream(self, tmp_path: Path) -> None:
        store, daniel_id, scott_id, channel_id, _agent_id = await _context(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"daniel": daniel_id, "scott": scott_id}))
        try:
            await _record(store, daniel_id, channel_id, "api-message", "api message")
            unauthorized = await client.get("/v1/client/unread")
            assert unauthorized.status == 401
            snapshot = await client.get("/v1/client/unread", headers={"Authorization": "Bearer scott"})
            assert snapshot.status == 200
            payload = await snapshot.json()
            group = next(item for item in payload["channels"] if item["channel_id"] == channel_id)
            assert group["unread_count"] == 1
            stream = await client.get(
                f"/v1/client/unread/events?after_position={payload['through_position']}",
                headers={"Authorization": "Bearer scott", "X-Kai-Stream-ID": "unread-test"},
            )
            live_message = await _record(
                store,
                daniel_id,
                channel_id,
                "api-live-message",
                "live api message",
                occurred_at=_NOW + timedelta(seconds=1),
            )
            live_event = await _next_sse_event(stream)
            assert live_event["event"] == "channel_unread.changed"
            assert live_event["data"]["state"]["channel_id"] == channel_id
            assert live_event["data"]["state"]["unread_count"] == 2
            mutation = await client.post(
                f"/v1/channels/{channel_id}/read-position",
                headers={"Authorization": "Bearer scott"},
                json={
                    "message_id": live_message,
                    "expected_state_version": group["state_version"],
                    "client_operation_id": "api-read",
                },
            )
            assert mutation.status == 200
            assert (await mutation.json())["state"]["unread_count"] == 0
            event = await _next_sse_event(stream)
            assert event["event"] == "channel_unread.changed"
            assert event["data"]["state"]["channel_id"] == channel_id
            assert event["data"]["state"]["unread_count"] == 0
        finally:
            await client.close()
            await store.close()
