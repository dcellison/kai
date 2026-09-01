"""Canonical followed-thread unread authority contracts."""

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
from kai.workshop.channel_unread import WorkshopChannelUnreadService
from kai.workshop.client_api import register_workshop_read_routes
from kai.workshop.domain import AgentId, ChannelId, MessageId, PrincipalId
from kai.workshop.inbound import ClientInboundMessage, record_client_inbound_message_in_transaction
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore
from kai.workshop.thread_unread import (
    WorkshopThreadUnreadAccessDenied,
    WorkshopThreadUnreadConflict,
    WorkshopThreadUnreadService,
)
from tests.workshop_profiles import profile_id

_NOW = datetime(2026, 9, 1, 14, 0, tzinfo=UTC)


@dataclass
class _Authenticator:
    principals_by_token: dict[str, PrincipalId]

    async def authenticate(self, request: web.Request) -> PrincipalId | None:
        scheme, separator, token = request.headers.get("Authorization", "").partition(" ")
        return self.principals_by_token.get(token) if separator and scheme == "Bearer" else None

    async def authenticate_token(self, token: str) -> PrincipalId | None:
        return self.principals_by_token.get(token)


@dataclass(frozen=True, slots=True)
class _Context:
    store: WorkshopEventStore
    daniel_id: PrincipalId
    scott_id: PrincipalId
    channel_id: ChannelId
    agent_id: AgentId


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


async def _context(path: Path) -> _Context:
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
        name="Thread unread authority",
        agent_ids=[result.agent_id],
        origin_channel_id=daniel_direct,
    )
    membership = await lifecycle.human_members(daniel_id, group.channel_id)
    await lifecycle.add_human_member(
        daniel_id,
        group.channel_id,
        scott_id,
        expected_state_version=membership.state_version,
        client_operation_id="add-scott-for-thread-unread",
    )
    return _Context(store, daniel_id, scott_id, group.channel_id, result.agent_id)


async def _record(
    context: _Context,
    principal_id: PrincipalId,
    identity: str,
    body: str,
    *,
    occurred_at: datetime = _NOW,
    thread_root_id: MessageId | None = None,
) -> MessageId:
    try:
        await context.store.connection.execute("BEGIN IMMEDIATE")
        result = await record_client_inbound_message_in_transaction(
            context.store,
            ClientInboundMessage(
                principal_id,
                context.channel_id,
                identity,
                body,
                occurred_at,
                thread_root_id=thread_root_id,
            ),
        )
        await context.store.connection.commit()
    except Exception:
        await context.store.connection.rollback()
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


class TestFollowedThreadUnreadAuthority:
    async def test_start_reply_and_mention_auto_follow_at_current_boundary(self, tmp_path: Path) -> None:
        context = await _context(tmp_path / "kai.db")
        service = WorkshopThreadUnreadService(context.store)
        try:
            root = await _record(context, context.daniel_id, "root", "A root")
            daniel = await service.thread(context.daniel_id, context.channel_id, root)
            scott = await service.thread(context.scott_id, context.channel_id, root)
            assert daniel.followed is True
            assert daniel.unread_count == 0
            assert scott.followed is False

            reply = await _record(
                context,
                context.scott_id,
                "scott-reply",
                "Scott replies",
                occurred_at=_NOW + timedelta(seconds=1),
                thread_root_id=root,
            )
            assert (await service.thread(context.scott_id, context.channel_id, root)).followed is True
            daniel = await service.thread(context.daniel_id, context.channel_id, root)
            assert daniel.unread_count == 1
            assert daniel.first_unread_message_id == reply

            mentioned_root = await _record(
                context,
                context.daniel_id,
                "mentioned-root",
                "@scott please follow this",
                occurred_at=_NOW + timedelta(seconds=2),
            )
            mentioned = await service.thread(context.scott_id, context.channel_id, mentioned_root)
            assert mentioned.followed is True
            assert mentioned.unread_count == 0

            mentioned_thread = await _record(
                context,
                context.daniel_id,
                "mentioned-thread-root",
                "Another root",
                occurred_at=_NOW + timedelta(seconds=3),
            )
            mentioned_reply = await _record(
                context,
                context.daniel_id,
                "mentioned-thread-reply",
                "@scott please read this reply",
                occurred_at=_NOW + timedelta(seconds=4),
                thread_root_id=mentioned_thread,
            )
            mentioned = await service.thread(context.scott_id, context.channel_id, mentioned_thread)
            assert mentioned.followed is True
            assert mentioned.unread_count == 1
            assert mentioned.first_unread_message_id == mentioned_reply
        finally:
            await context.store.close()

    async def test_positions_are_independent_and_channel_aggregation_does_not_merge_cursors(
        self,
        tmp_path: Path,
    ) -> None:
        context = await _context(tmp_path / "kai.db")
        threads = WorkshopThreadUnreadService(context.store)
        channels = WorkshopChannelUnreadService(context.store)
        try:
            first_root = await _record(context, context.daniel_id, "first-root", "First")
            second_root = await _record(context, context.scott_id, "second-root", "Second")
            first_reply = await _record(
                context,
                context.scott_id,
                "first-reply",
                "Unread for Daniel",
                occurred_at=_NOW + timedelta(seconds=1),
                thread_root_id=first_root,
            )
            await _record(
                context,
                context.daniel_id,
                "second-reply",
                "Unread for Scott",
                occurred_at=_NOW + timedelta(seconds=2),
                thread_root_id=second_root,
            )
            aggregated = await channels.channel(context.daniel_id, context.channel_id)
            assert aggregated.unread_reply_count == 1
            assert aggregated.unread_thread_count == 1
            assert aggregated.first_unread_thread_root_id == first_root
            assert aggregated.first_unread_thread_reply_id == first_reply

            first = await threads.thread(context.daniel_id, context.channel_id, first_root)
            advanced = await threads.advance(
                context.daniel_id,
                context.channel_id,
                first_root,
                first_reply,
                expected_state_version=first.state_version,
                client_operation_id="read-first-thread",
            )
            assert advanced.state.unread_count == 0
            assert (await threads.thread(context.scott_id, context.channel_id, second_root)).unread_count == 1
            assert (await channels.channel(context.daniel_id, context.channel_id)).unread_reply_count == 0
        finally:
            await context.store.close()

    async def test_unfollow_and_refollow_use_current_boundary_and_are_idempotent(self, tmp_path: Path) -> None:
        context = await _context(tmp_path / "kai.db")
        service = WorkshopThreadUnreadService(context.store)
        try:
            root = await _record(context, context.daniel_id, "root", "Root")
            current = await service.thread(context.daniel_id, context.channel_id, root)
            unfollowed = await service.unfollow(
                context.daniel_id,
                context.channel_id,
                root,
                expected_state_version=current.state_version,
                client_operation_id="unfollow",
            )
            replay = await service.unfollow(
                context.daniel_id,
                context.channel_id,
                root,
                expected_state_version=current.state_version,
                client_operation_id="unfollow",
            )
            assert replay.replayed is True
            assert replay.state == unfollowed.state
            await _record(
                context,
                context.scott_id,
                "while-unfollowed",
                "No backlog",
                occurred_at=_NOW + timedelta(seconds=1),
                thread_root_id=root,
            )
            refollowed = await service.follow(
                context.daniel_id,
                context.channel_id,
                root,
                expected_state_version=unfollowed.state.state_version,
                client_operation_id="refollow",
            )
            assert refollowed.state.followed is True
            assert refollowed.state.unread_count == 0
            with pytest.raises(WorkshopThreadUnreadConflict, match="unchanged"):
                await service.follow(
                    context.daniel_id,
                    context.channel_id,
                    root,
                    expected_state_version=refollowed.state.state_version,
                    client_operation_id="follow-again",
                )
        finally:
            await context.store.close()

    async def test_projection_rebuild_preserves_follow_and_read_state(self, tmp_path: Path) -> None:
        context = await _context(tmp_path / "kai.db")
        service = WorkshopThreadUnreadService(context.store)
        try:
            root = await _record(context, context.daniel_id, "root", "Root")
            reply = await _record(
                context,
                context.scott_id,
                "reply",
                "Reply",
                occurred_at=_NOW + timedelta(seconds=1),
                thread_root_id=root,
            )
            current = await service.thread(context.daniel_id, context.channel_id, root)
            expected = (
                await service.advance(
                    context.daniel_id,
                    context.channel_id,
                    root,
                    reply,
                    expected_state_version=current.state_version,
                    client_operation_id="read-before-rebuild",
                )
            ).state
            await context.store.rebuild_projection(CanonicalConversationProjection())
            assert await service.thread(context.daniel_id, context.channel_id, root) == expected
        finally:
            await context.store.close()

    async def test_membership_removal_and_readd_do_not_resurrect_thread_state(self, tmp_path: Path) -> None:
        context = await _context(tmp_path / "kai.db")
        service = WorkshopThreadUnreadService(context.store)
        lifecycle = WorkshopChannelLifecycleService(context.store)
        try:
            root = await _record(context, context.daniel_id, "root", "Root")
            await _record(context, context.scott_id, "scott-reply", "Scott follows", thread_root_id=root)
            assert (await service.thread(context.scott_id, context.channel_id, root)).followed is True

            membership = await lifecycle.human_members(context.daniel_id, context.channel_id)
            await lifecycle.remove_human_member(
                context.daniel_id,
                context.channel_id,
                context.scott_id,
                expected_state_version=membership.state_version,
                client_operation_id="remove-scott-thread-state",
            )
            with pytest.raises(WorkshopThreadUnreadAccessDenied):
                await service.thread(context.scott_id, context.channel_id, root)

            membership = await lifecycle.human_members(context.daniel_id, context.channel_id)
            await lifecycle.add_human_member(
                context.daniel_id,
                context.channel_id,
                context.scott_id,
                expected_state_version=membership.state_version,
                client_operation_id="readd-scott-thread-state",
            )
            restored = await service.thread(context.scott_id, context.channel_id, root)
            assert restored.followed is False
            assert restored.unread_count == 0
        finally:
            await context.store.close()

    async def test_authenticated_api_and_private_principal_event_stream(self, tmp_path: Path) -> None:
        context = await _context(tmp_path / "kai.db")
        client = await _open_client(
            context.store,
            _Authenticator({"daniel": context.daniel_id, "scott": context.scott_id}),
        )
        try:
            root = await _record(context, context.daniel_id, "api-root", "API root")
            thread_path = f"/v1/channels/{context.channel_id}/threads/{root}"
            unauthorized = await client.get(f"{thread_path}/unread")
            assert unauthorized.status == 401
            state_response = await client.get(f"{thread_path}/unread", headers={"Authorization": "Bearer daniel"})
            assert state_response.status == 200
            initial = await state_response.json()
            assert initial["state"]["followed"] is True

            async with context.store.connection.execute("SELECT COALESCE(MAX(position), 0) FROM event_log") as cursor:
                tip_row = await cursor.fetchone()
            assert tip_row is not None
            tip = int(tip_row[0])
            stream = await client.get(
                "/v1/client/events",
                headers={
                    "Authorization": "Bearer daniel",
                    "Last-Event-ID": str(tip),
                    "X-Kai-Stream-ID": "thread-unread-test:principal",
                },
            )
            reply = await _record(
                context,
                context.scott_id,
                "api-reply",
                "Unread API reply",
                occurred_at=_NOW + timedelta(seconds=1),
                thread_root_id=root,
            )
            event = await _next_sse_event(stream)
            assert event["event"] == "workshop.principal.changed"
            data = event["data"]
            assert isinstance(data, dict)
            changes = data["changes"]
            thread_change = next(
                item for item in changes[0]["thread_changes"] if item["state"]["thread_root_id"] == root
            )
            assert thread_change["state"]["unread_count"] == 1
            mutation = await client.post(
                f"{thread_path}/read-position",
                headers={"Authorization": "Bearer daniel"},
                json={
                    "message_id": reply,
                    "expected_state_version": initial["state"]["state_version"],
                    "client_operation_id": "api-read-thread",
                },
            )
            assert mutation.status == 200
            assert (await mutation.json())["state"]["unread_count"] == 0

            scott_state = await client.get(f"{thread_path}/unread", headers={"Authorization": "Bearer scott"})
            assert scott_state.status == 200
            assert (await scott_state.json())["state"]["unread_count"] == 0
        finally:
            await client.close()
            await context.store.close()
