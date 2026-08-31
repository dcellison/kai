"""Canonical human mention notification authority contracts."""

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
from kai.workshop.client_api import register_workshop_read_routes
from kai.workshop.diagnostics import workshop_human_notification_status
from kai.workshop.domain import AgentId, ChannelId, HumanNotificationId, MessageId, PrincipalId
from kai.workshop.human_notifications import (
    HumanNotificationStateRequest,
    WorkshopHumanNotificationAccessDenied,
    WorkshopHumanNotificationConflict,
    WorkshopHumanNotificationService,
)
from kai.workshop.inbound import ClientInboundMessage, record_client_inbound_message_in_transaction
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore
from kai.workshop.wake_policy import resolve_message_wake_targets
from tests.workshop_profiles import profile_id

_NOW = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)


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


async def _notification_context(
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
    group = await WorkshopChannelLifecycleService(store).create_group(
        daniel_id,
        name="Human notifications",
        agent_ids=[result.agent_id],
        origin_channel_id=daniel_direct,
    )
    lifecycle = WorkshopChannelLifecycleService(store)
    membership = await lifecycle.human_members(daniel_id, group.channel_id)
    await lifecycle.add_human_member(
        daniel_id,
        group.channel_id,
        scott_id,
        expected_state_version=membership.state_version,
        client_operation_id="add-scott-for-notifications",
    )
    return store, daniel_id, scott_id, group.channel_id, result.agent_id


async def _record(
    store: WorkshopEventStore,
    principal_id: PrincipalId,
    channel_id: ChannelId,
    client_message_id: str,
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
                client_message_id,
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


async def _open_client(
    store: WorkshopEventStore,
    authenticator: _Authenticator,
) -> TestClient:
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


class TestHumanNotificationAuthority:
    async def test_resolved_human_mentions_create_one_private_replayable_notification(
        self,
        tmp_path: Path,
    ) -> None:
        store, daniel_id, scott_id, channel_id, agent_id = await _notification_context(tmp_path / "kai.db")
        service = WorkshopHumanNotificationService(store)
        try:
            async with store.connection.execute(
                "SELECT ad.handle FROM agent_definitions ad WHERE ad.agent_id = ?",
                (agent_id,),
            ) as cursor:
                agent_handle = str((await cursor.fetchone())[0])
            human_only = await _record(
                store,
                daniel_id,
                channel_id,
                "human-only",
                "@scott please review; repeating @SCOTT does not duplicate",
            )
            human_wake = await resolve_message_wake_targets(store, human_only)
            assert human_wake.agent_ids == ()
            mixed = await _record(
                store,
                daniel_id,
                channel_id,
                "mixed",
                f"@scott and @{agent_handle} please coordinate",
                occurred_at=_NOW + timedelta(seconds=1),
            )
            assert (await resolve_message_wake_targets(store, mixed)).agent_ids == (agent_id,)
            await _record(
                store,
                daniel_id,
                channel_id,
                "agent-only",
                f"@{agent_handle} agent-only message",
                occurred_at=_NOW + timedelta(seconds=2),
            )

            scott_page = await service.list(scott_id)
            assert scott_page.counts.total == 2
            assert scott_page.counts.unread == 2
            assert scott_page.counts.unread_by_channel == ((channel_id, 2),)
            assert {item.source_message_id for item in scott_page.notifications} == {
                human_only,
                mixed,
            }
            assert (await service.counts(daniel_id)).total == 0

            await _record(
                store,
                daniel_id,
                channel_id,
                "human-only",
                "@scott please review; repeating @SCOTT does not duplicate",
            )
            assert (await service.counts(scott_id)).total == 2
            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type = 'human_notification.created'"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 2

            await store.rebuild_projection(CanonicalConversationProjection())
            assert (await service.counts(scott_id)).total == 2
            status = workshop_human_notification_status(tmp_path / "kai.db")
            assert "active" in status
            assert "notifications=2" in status
            assert "integrity gaps=0" in status
            assert "replay gaps=0" in status
        finally:
            await store.close()

    async def test_read_state_is_revision_safe_bounded_and_persists(
        self,
        tmp_path: Path,
    ) -> None:
        store, daniel_id, scott_id, channel_id, _ = await _notification_context(tmp_path / "kai.db")
        service = WorkshopHumanNotificationService(store)
        try:
            for ordinal in range(3):
                await _record(
                    store,
                    daniel_id,
                    channel_id,
                    f"page-{ordinal}",
                    f"@scott page {ordinal}",
                    occurred_at=_NOW + timedelta(seconds=ordinal),
                )
            first_page = await service.list(scott_id, limit=2)
            assert len(first_page.notifications) == 2
            assert first_page.through_position > 0
            assert first_page.next_cursor is not None
            second_page = await service.list(scott_id, limit=2, cursor=first_page.next_cursor)
            assert len(second_page.notifications) == 1
            assert second_page.next_cursor is None

            first, second = first_page.notifications
            mutations = await service.set_many_read(
                scott_id,
                (
                    HumanNotificationStateRequest(first.notification_id, first.state_version),
                    HumanNotificationStateRequest(second.notification_id, second.state_version),
                ),
                client_operation_id="bulk-read-one",
            )
            assert [item.changed for item in mutations] == [True, True]
            assert (await service.counts(scott_id)).unread == 1
            replayed = await service.set_many_read(
                scott_id,
                (
                    HumanNotificationStateRequest(first.notification_id, first.state_version),
                    HumanNotificationStateRequest(second.notification_id, second.state_version),
                ),
                client_operation_id="bulk-read-one",
            )
            assert [item.replayed for item in replayed] == [True, True]
            with pytest.raises(WorkshopHumanNotificationConflict):
                await service.set_read_state(
                    scott_id,
                    first.notification_id,
                    read=False,
                    expected_state_version=0,
                    client_operation_id="stale-unread",
                )
            unread = await service.set_read_state(
                scott_id,
                first.notification_id,
                read=False,
                expected_state_version=1,
                client_operation_id="valid-unread",
            )
            assert unread.changed is True
            assert unread.notification.read is False
            await store.rebuild_projection(CanonicalConversationProjection())
            assert (await service.counts(scott_id)).unread == 2
        finally:
            await store.close()

    async def test_current_membership_controls_visibility_without_destroying_history(
        self,
        tmp_path: Path,
    ) -> None:
        store, daniel_id, scott_id, channel_id, _ = await _notification_context(tmp_path / "kai.db")
        lifecycle = WorkshopChannelLifecycleService(store)
        service = WorkshopHumanNotificationService(store)
        try:
            root_message_id = await _record(
                store,
                daniel_id,
                channel_id,
                "membership-root",
                "Thread root without a mention",
            )
            await _record(
                store,
                daniel_id,
                channel_id,
                "membership",
                "@scott retained notification",
                thread_root_id=root_message_id,
            )
            notification = (await service.list(scott_id)).notifications[0]
            assert notification.source_thread_root_id == root_message_id
            snapshot = await lifecycle.human_members(daniel_id, channel_id)
            await lifecycle.remove_human_member(
                daniel_id,
                channel_id,
                scott_id,
                expected_state_version=snapshot.state_version,
                client_operation_id="remove-scott-notification-access",
            )
            assert (await service.counts(scott_id)).total == 0
            with pytest.raises(WorkshopHumanNotificationAccessDenied):
                await service.set_read_state(
                    scott_id,
                    notification.notification_id,
                    read=True,
                    expected_state_version=0,
                    client_operation_id="hidden-notification-read",
                )
            async with store.connection.execute("SELECT COUNT(*) FROM human_notifications") as cursor:
                assert int((await cursor.fetchone())[0]) == 1
            snapshot = await lifecycle.human_members(daniel_id, channel_id)
            await lifecycle.add_human_member(
                daniel_id,
                channel_id,
                scott_id,
                expected_state_version=snapshot.state_version,
                client_operation_id="restore-scott-notification-access",
            )
            assert (await service.counts(scott_id)).total == 1
        finally:
            await store.close()


class TestHumanNotificationApi:
    async def test_authenticated_inbox_mutations_isolation_and_live_events(
        self,
        tmp_path: Path,
    ) -> None:
        store, daniel_id, scott_id, channel_id, _ = await _notification_context(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"daniel": daniel_id, "scott": scott_id}))
        try:
            stream = await client.get(
                "/v1/client/notifications/events",
                headers={"Authorization": "Bearer scott", "X-Kai-Stream-ID": "notification-test"},
            )
            assert stream.status == 200
            await _record(store, daniel_id, channel_id, "api", "@scott API notification")
            event = await _next_sse_event(stream)
            assert event["event"] == "human_notification.changed"
            event_data = event["data"]
            assert isinstance(event_data, dict)
            assert event_data["event_type"] == "human_notification.created"
            assert "body" not in json.dumps(event_data)

            response = await client.get(
                "/v1/client/notifications",
                headers={"Authorization": "Bearer scott"},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload["counts"] == {
                "total": 1,
                "unread": 1,
                "read": 0,
                "unread_by_channel": {str(channel_id): 1},
            }
            assert payload["through_position"] > 0
            notification = payload["notifications"][0]
            assert notification["source_channel_id"] == channel_id
            assert "body" not in notification
            notification_id = HumanNotificationId(notification["notification_id"])
            source_response = await client.get(
                f"/v1/channels/{channel_id}/messages/{notification['source_message_id']}",
                headers={"Authorization": "Bearer scott"},
            )
            assert source_response.status == 200
            source_payload = await source_response.json()
            assert source_payload["message"]["body"] == "@scott API notification"

            daniel_response = await client.get(
                "/v1/client/notifications",
                headers={"Authorization": "Bearer daniel"},
            )
            assert daniel_response.status == 200
            assert (await daniel_response.json())["notifications"] == []
            denied = await client.post(
                f"/v1/client/notifications/{notification_id}/read",
                headers={"Authorization": "Bearer daniel"},
                json={"expected_state_version": 0, "client_operation_id": "daniel-denied"},
            )
            assert denied.status == 404

            read_response = await client.post(
                f"/v1/client/notifications/{notification_id}/read",
                headers={"Authorization": "Bearer scott"},
                json={"expected_state_version": 0, "client_operation_id": "scott-read"},
            )
            assert read_response.status == 200
            read_payload = await read_response.json()
            assert read_payload["changed"] is True
            assert read_payload["notification"]["read"] is True
            unread_response = await client.post(
                f"/v1/client/notifications/{notification_id}/unread",
                headers={"Authorization": "Bearer scott"},
                json={"expected_state_version": 1, "client_operation_id": "scott-unread"},
            )
            assert unread_response.status == 200
            assert (await unread_response.json())["notification"]["read"] is False
            bulk_response = await client.post(
                "/v1/client/notifications/read",
                headers={"Authorization": "Bearer scott"},
                json={
                    "client_operation_id": "scott-bulk-read",
                    "notifications": [
                        {
                            "notification_id": notification_id,
                            "expected_state_version": 2,
                        }
                    ],
                },
            )
            assert bulk_response.status == 200
            bulk_payload = await bulk_response.json()
            assert bulk_payload["notifications"][0]["changed"] is True
            assert bulk_payload["notifications"][0]["notification"]["read"] is True
            counts = await client.get(
                "/v1/client/notifications/counts",
                headers={"Authorization": "Bearer scott"},
            )
            assert await counts.json() == {
                "version": 1,
                "total": 1,
                "unread": 0,
                "read": 1,
                "unread_by_channel": {},
            }
            stream.close()
        finally:
            await client.close()
            await store.close()
