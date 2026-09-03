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
from kai.workshop.channel_notification_policy import (
    WorkshopChannelNotificationPolicyService,
    WorkshopChannelNotificationPolicyValidationError,
)
from kai.workshop.client_api import register_workshop_read_routes
from kai.workshop.delivery_policy import DeliveryAdapterCapabilities, WorkshopDeliveryBindingPolicy
from kai.workshop.diagnostics import (
    workshop_channel_notification_policy_status,
    workshop_human_notification_status,
)
from kai.workshop.domain import AgentId, ChannelId, HumanNotificationId, MessageId, PrincipalId
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.human_notifications import (
    HUMAN_NOTIFICATION_DELIVERY_MODE,
    HumanNotificationStateRequest,
    WorkshopHumanNotificationAccessDenied,
    WorkshopHumanNotificationConflict,
    WorkshopHumanNotificationService,
)
from kai.workshop.inbound import ClientInboundMessage, record_client_inbound_message_in_transaction
from kai.workshop.outbound import OutboundMessage, record_outbound_message
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore
from kai.workshop.wake_policy import resolve_message_wake_targets
from tests.workshop_profiles import profile_id, profile_registry

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
    delivery_policy: WorkshopDeliveryBindingPolicy | None = None,
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
            delivery_policy=delivery_policy,
        )
        await store.connection.commit()
    except Exception:
        await store.connection.rollback()
        raise
    return MessageId(str(result.event.envelope.aggregate_id))


async def _open_client(
    store: WorkshopEventStore,
    authenticator: _Authenticator,
    *,
    channel_notification_policy: WorkshopChannelNotificationPolicyService | None = None,
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
        channel_notification_policy=channel_notification_policy,
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
    async def test_publication_is_safe_recipient_routed_idempotent_and_replayable(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "kai.db"
        store, daniel_id, scott_id, channel_id, _agent_id = await _notification_context(path)
        capabilities = DeliveryAdapterCapabilities("desktop")
        policy = WorkshopDeliveryBindingPolicy(
            frozenset({"desktop"}),
            (capabilities,),
            "http://workshop.example/workshop/",
        )
        secret_body = "@scott private source content must not leave Workshop"
        try:
            message_id = await _record(
                store,
                daniel_id,
                channel_id,
                "publication-safe",
                secret_body,
                delivery_policy=policy,
            )
            await _record(
                store,
                daniel_id,
                channel_id,
                "publication-safe",
                secret_body,
                delivery_policy=policy,
            )
            async with store.connection.execute(
                "SELECT p.notification_id, p.recipient_principal_id, p.policy_result, "
                "p.alert_body, p.deep_link, o.mode, o.transport, o.status "
                "FROM human_notification_publications p "
                "JOIN delivery_outbox o ON o.human_notification_id = p.notification_id "
                "WHERE p.source_message_id = ?",
                (message_id,),
            ) as cursor:
                rows = tuple(await cursor.fetchall())
            assert len(rows) == 1
            row = rows[0]
            assert str(row[1]) == scott_id
            assert tuple(str(value) for value in row[2:]) == (
                "eligible",
                "Daniel mentioned you in #Human notifications.\nOpen Workshop: http://workshop.example/workshop/",
                "http://workshop.example/workshop/",
                HUMAN_NOTIFICATION_DELIVERY_MODE,
                "desktop",
                "pending",
            )
            assert "private source content" not in str(row[3])

            await store.rebuild_projection(CanonicalConversationProjection())
            async with store.connection.execute(
                "SELECT COUNT(*) FROM human_notification_publications WHERE source_message_id = ?",
                (message_id,),
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 1
        finally:
            await store.close()

        reopened = await WorkshopEventStore.open(path)
        try:
            assert (await WorkshopHumanNotificationService(reopened).counts(scott_id)).total == 1
            status = workshop_human_notification_status(path)
            assert "publications=1 (eligible=1, DND suppressed=0, integrity gaps=0)" in status
            assert "adapter decisions=1 (eligible=1, preference suppressed=0, integrity gaps=0)" in status
            assert "adapter deliveries=1 (pending=1" in status
            assert "failure classes=none" in status
        finally:
            await reopened.close()

    async def test_adapter_suppression_never_removes_the_canonical_inbox(
        self,
        tmp_path: Path,
    ) -> None:
        store, daniel_id, scott_id, channel_id, _agent_id = await _notification_context(tmp_path / "kai.db")
        capabilities = DeliveryAdapterCapabilities("desktop")
        enabled = WorkshopDeliveryBindingPolicy(frozenset({"desktop"}), (capabilities,))
        try:
            await store.connection.execute(
                "INSERT INTO principal_human_notification_policies "
                "(principal_id, muted_mentions_notify, dnd_enabled, dnd_timezone, "
                "dnd_start_minute, dnd_end_minute) VALUES (?, 1, 1, 'UTC', 660, 780)",
                (scott_id,),
            )
            await store.connection.commit()
            await _record(
                store,
                daniel_id,
                channel_id,
                "publication-dnd",
                "@scott DND still keeps the inbox",
                delivery_policy=enabled,
            )
            assert (await WorkshopHumanNotificationService(store).counts(scott_id)).total == 1
            async with store.connection.execute("SELECT policy_result FROM human_notification_publications") as cursor:
                assert tuple(await cursor.fetchone()) == ("suppressed_dnd",)
            async with store.connection.execute(
                "SELECT COUNT(*) FROM delivery_outbox WHERE human_notification_id IS NOT NULL"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0

            await store.connection.execute(
                "UPDATE principal_human_notification_policies SET dnd_enabled = 0 WHERE principal_id = ?",
                (scott_id,),
            )
            await store.connection.execute(
                "DELETE FROM external_identities WHERE principal_id = ? AND provider = 'desktop'",
                (scott_id,),
            )
            await store.connection.commit()
            await _record(
                store,
                daniel_id,
                channel_id,
                "publication-revoked",
                "@scott revoked adapter still keeps the inbox",
                delivery_policy=enabled,
            )
            assert (await WorkshopHumanNotificationService(store).counts(scott_id)).total == 2
            async with store.connection.execute(
                "SELECT COUNT(*) FROM human_notification_publications WHERE policy_result = 'eligible'"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 1
            async with store.connection.execute(
                "SELECT COUNT(*) FROM delivery_outbox WHERE human_notification_id IS NOT NULL"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
        finally:
            await store.close()

    async def test_version_fifty_three_notification_rows_upgrade_without_loss(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from kai.workshop import schema

        path = tmp_path / "kai.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 53)
            migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:53])
            store, daniel_id, scott_id, channel_id, _agent_id = await _notification_context(path)
            await _record(store, daniel_id, channel_id, "pre-policy-mention", "@scott preserved")
            assert (await WorkshopHumanNotificationService(store).counts(scott_id)).total == 1
            await store.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 64
            async with upgraded.connection.execute(
                "SELECT kind FROM human_notifications WHERE recipient_principal_id = ?",
                (scott_id,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == ("mention",)
            assert workshop_human_notification_status(path).startswith(
                "Workshop human notifications: active; notifications=1"
            )
        finally:
            await upgraded.close()

    async def test_agent_group_reply_notifies_replied_to_human_without_direct_message_noise(
        self,
        tmp_path: Path,
    ) -> None:
        store, daniel_id, _scott_id, channel_id, agent_id = await _notification_context(tmp_path / "kai.db")
        inbox = WorkshopHumanNotificationService(store)
        try:
            inbound_id = await _record(store, daniel_id, channel_id, "agent-reply-source", "@kai reply")
            await record_outbound_message(
                store,
                OutboundMessage(
                    in_reply_to_message_id=inbound_id,
                    body="Agent group reply",
                    occurred_at=_NOW + timedelta(seconds=1),
                    agent_id=agent_id,
                ),
            )
            page = await inbox.list(daniel_id)
            assert [item.kind for item in page.notifications] == ["reply"]
        finally:
            await store.close()

    async def test_authenticated_policy_api_is_principal_isolated_and_conflict_safe(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "kai.db"
        store, daniel_id, scott_id, channel_id, _agent_id = await _notification_context(path)
        registry = await WorkshopExecutionStateRegistry.from_store(store, profile_registry(101, 202))
        policy = await WorkshopChannelNotificationPolicyService.open(path, registry)
        client = await _open_client(
            store,
            _Authenticator({"daniel": daniel_id, "scott": scott_id}),
            channel_notification_policy=policy,
        )
        try:
            scott_response = await client.get(
                "/v1/settings/channel-notifications",
                headers={"Authorization": "Bearer scott"},
            )
            assert scott_response.status == 200
            scott = await scott_response.json()
            assert scott["adapter_deliveries"] == [
                {
                    "transport": "desktop",
                    "display_name": "Desktop",
                    "enabled": True,
                    "source": "default",
                }
            ]
            assert scott["channels"] == [
                {
                    "channel_id": channel_id,
                    "channel_name": "Human notifications",
                    "level": "mentions_replies",
                    "source": "default",
                }
            ]
            scott_changed_response = await client.patch(
                "/v1/settings/channel-notifications",
                headers={"Authorization": "Bearer scott"},
                json={
                    "revision": scott["revision"],
                    "adapter_delivery": {"transport": "desktop", "enabled": False},
                },
            )
            assert scott_changed_response.status == 200
            assert (await scott_changed_response.json())["adapter_deliveries"][0]["enabled"] is False

            daniel_response = await client.get(
                "/v1/settings/channel-notifications",
                headers={"Authorization": "Bearer daniel"},
            )
            daniel = await daniel_response.json()
            assert daniel["adapter_deliveries"][0]["enabled"] is True
            changed_response = await client.patch(
                "/v1/settings/channel-notifications",
                headers={"Authorization": "Bearer daniel"},
                json={
                    "revision": daniel["revision"],
                    "channel": {"channel_id": channel_id, "level": "all"},
                },
            )
            assert changed_response.status == 200

            unchanged_response = await client.get(
                "/v1/settings/channel-notifications",
                headers={"Authorization": "Bearer scott"},
            )
            unchanged = await unchanged_response.json()
            assert unchanged["channels"][0]["level"] == "mentions_replies"
            assert unchanged["adapter_deliveries"][0]["enabled"] is False
            forged_adapter_response = await client.patch(
                "/v1/settings/channel-notifications",
                headers={"Authorization": "Bearer scott"},
                json={
                    "revision": unchanged["revision"],
                    "adapter_delivery": {"transport": "whatsapp", "enabled": False},
                },
            )
            assert forged_adapter_response.status == 403

            stale_response = await client.patch(
                "/v1/settings/channel-notifications",
                headers={"Authorization": "Bearer daniel"},
                json={
                    "revision": daniel["revision"],
                    "muted_mentions_notify": False,
                },
            )
            assert stale_response.status == 409
        finally:
            await client.close()
            await policy.close()
            await store.close()

    async def test_adapter_opt_out_survives_restart_and_reenable_does_not_replay(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "kai.db"
        store, daniel_id, scott_id, channel_id, _agent_id = await _notification_context(path)
        registry = await WorkshopExecutionStateRegistry.from_store(store, profile_registry(101, 202))
        policy = await WorkshopChannelNotificationPolicyService.open(path, registry)
        delivery_policy = WorkshopDeliveryBindingPolicy(
            frozenset({"desktop"}),
            (DeliveryAdapterCapabilities("desktop"),),
        )
        authority = policy.authority_for_principal(scott_id)
        initial = await policy.inspect(authority)
        assert [(item.transport, item.enabled, item.source) for item in initial.adapter_deliveries] == [
            ("desktop", True, "default")
        ]
        disabled = await policy.set_adapter_delivery(
            authority,
            "desktop",
            False,
            expected_revision=initial.revision,
        )
        assert disabled.adapter_deliveries[0].enabled is False
        await policy.close()

        reopened = await WorkshopChannelNotificationPolicyService.open(path, registry)
        try:
            restarted = await reopened.inspect(reopened.authority_for_principal(scott_id))
            assert restarted.adapter_deliveries[0].enabled is False
            first_message = await _record(
                store,
                daniel_id,
                channel_id,
                "adapter-disabled",
                "@scott inbox only",
                delivery_policy=delivery_policy,
            )
            assert (await WorkshopHumanNotificationService(store).counts(scott_id)).total == 1
            async with store.connection.execute(
                "SELECT d.policy_result FROM human_notification_adapter_delivery_decisions d "
                "JOIN human_notification_publications p ON p.notification_id = d.notification_id "
                "WHERE p.source_message_id = ?",
                (first_message,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == ("suppressed_preference",)
            async with store.connection.execute(
                "SELECT COUNT(*) FROM delivery_outbox WHERE human_notification_id IS NOT NULL"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0

            enabled = await reopened.set_adapter_delivery(
                reopened.authority_for_principal(scott_id),
                "desktop",
                True,
                expected_revision=restarted.revision,
            )
            assert enabled.adapter_deliveries[0].enabled is True
            async with store.connection.execute(
                "SELECT COUNT(*) FROM delivery_outbox WHERE human_notification_id IS NOT NULL"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
            second_message = await _record(
                store,
                daniel_id,
                channel_id,
                "adapter-reenabled",
                "@scott adapter alert",
                occurred_at=_NOW + timedelta(seconds=1),
                delivery_policy=delivery_policy,
            )
            async with store.connection.execute(
                "SELECT d.policy_result FROM human_notification_adapter_delivery_decisions d "
                "JOIN human_notification_publications p ON p.notification_id = d.notification_id "
                "WHERE p.source_message_id = ?",
                (second_message,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == ("eligible",)
            async with store.connection.execute(
                "SELECT COUNT(*) FROM delivery_outbox WHERE human_notification_id IS NOT NULL"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 1
        finally:
            await reopened.close()
            await store.close()

    async def test_channel_policy_controls_ordinary_reply_and_muted_mentions(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "kai.db"
        store, daniel_id, scott_id, channel_id, _agent_id = await _notification_context(path)
        registry = await WorkshopExecutionStateRegistry.from_store(store, profile_registry(101, 202))
        policy = await WorkshopChannelNotificationPolicyService.open(path, registry)
        inbox = WorkshopHumanNotificationService(store)
        authority = policy.authority_for_principal(scott_id)
        try:
            await _record(store, daniel_id, channel_id, "ordinary-default", "Ordinary default")
            assert (await inbox.counts(scott_id)).total == 0

            root_id = await _record(store, scott_id, channel_id, "scott-root", "Scott root")
            await _record(
                store,
                daniel_id,
                channel_id,
                "reply-default",
                "Reply to Scott",
                thread_root_id=root_id,
            )
            page = await inbox.list(scott_id)
            assert [item.kind for item in page.notifications] == ["reply"]
            await _record(
                store,
                scott_id,
                channel_id,
                "self-reply",
                "Scott self reply",
                thread_root_id=root_id,
            )
            assert (await inbox.counts(scott_id)).total == 1

            initial = await policy.inspect(authority)
            all_messages = await policy.set_channel_level(
                authority,
                channel_id,
                "all",
                expected_revision=initial.revision,
            )
            await _record(store, daniel_id, channel_id, "ordinary-all", "Ordinary all")
            page = await inbox.list(scott_id)
            assert [item.kind for item in page.notifications[:2]] == ["message", "reply"]

            muted = await policy.set_channel_level(
                authority,
                channel_id,
                "muted",
                expected_revision=all_messages.revision,
            )
            suppressed = await policy.set_muted_mentions_notify(
                authority,
                False,
                expected_revision=muted.revision,
            )
            await _record(store, daniel_id, channel_id, "muted-suppressed", "@scott suppressed")
            assert (await inbox.counts(scott_id)).total == 2

            await policy.set_muted_mentions_notify(
                authority,
                True,
                expected_revision=suppressed.revision,
            )
            await _record(store, daniel_id, channel_id, "muted-override", "@scott override")
            page = await inbox.list(scott_id)
            assert page.notifications[0].kind == "mention"
            assert page.counts.total == 3
        finally:
            await policy.close()
            await store.close()

    async def test_dnd_policy_is_validated_and_survives_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "kai.db"
        store, _daniel_id, scott_id, _channel_id, _agent_id = await _notification_context(path)
        registry = await WorkshopExecutionStateRegistry.from_store(store, profile_registry(101, 202))
        policy = await WorkshopChannelNotificationPolicyService.open(path, registry)
        authority = policy.authority_for_principal(scott_id)
        initial = await policy.inspect(authority)
        with pytest.raises(WorkshopChannelNotificationPolicyValidationError):
            await policy.set_do_not_disturb(
                authority,
                enabled=True,
                timezone="Not/A_Zone",
                start="22:00",
                end="07:00",
                expected_revision=initial.revision,
            )
        saved = await policy.set_do_not_disturb(
            authority,
            enabled=True,
            timezone="America/Toronto",
            start="22:30",
            end="06:45",
            expected_revision=initial.revision,
        )
        assert saved.do_not_disturb.enabled is True
        assert workshop_channel_notification_policy_status(path) == (
            "Workshop channel notification policy: active; principals=1, channel overrides=0, "
            "DND enabled=1, adapter bindings=2 (enabled=2, disabled=0, explicit=0, stale=0), "
            "invalid=0; authority=canonical, Workshop=in-app immediate"
        )
        assert (
            await policy.external_delivery_allowed(
                scott_id,
                datetime(2026, 8, 31, 3, 0, tzinfo=UTC),
            )
            is False
        )
        assert (
            await policy.external_delivery_allowed(
                scott_id,
                datetime(2026, 8, 31, 16, 0, tzinfo=UTC),
            )
            is True
        )
        await policy.close()

        reopened = await WorkshopChannelNotificationPolicyService.open(path, registry)
        try:
            restarted = await reopened.inspect(reopened.authority_for_principal(scott_id))
            assert restarted.do_not_disturb.timezone == "America/Toronto"
            assert restarted.do_not_disturb.start == "22:30"
            assert restarted.do_not_disturb.end == "06:45"
        finally:
            await reopened.close()
            await store.close()

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
    async def test_principal_stream_starts_with_checkpoint_and_rejects_future_resume(
        self,
        tmp_path: Path,
    ) -> None:
        store, _, scott_id, _, _ = await _notification_context(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"scott": scott_id}))
        stream = None
        try:
            stream = await client.get(
                "/v1/client/events",
                headers={
                    "Authorization": "Bearer scott",
                    "X-Kai-Stream-ID": "principal-checkpoint-test",
                },
            )
            assert stream.status == 200
            event = await _next_sse_event(stream)
            assert event["event"] == "workshop.principal.changed"
            data = event["data"]
            assert isinstance(data, dict)
            assert data["changes"] == []
            assert data["through_position"] > 0
            stream.close()
            stream = None

            response = await client.get(
                "/v1/client/events?after_position=999999999",
                headers={
                    "Authorization": "Bearer scott",
                    "X-Kai-Stream-ID": "principal-future-resume-test",
                },
            )
            assert response.status == 409
            assert await response.json() == {
                "error": {
                    "code": "resynchronization_required",
                    "message": "Workshop principal events resynchronization required",
                }
            }
        finally:
            if stream is not None:
                stream.close()
            await client.close()
            await store.close()

    async def test_principal_stream_groups_mention_and_unread_at_one_position(
        self,
        tmp_path: Path,
    ) -> None:
        store, daniel_id, scott_id, channel_id, _ = await _notification_context(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"scott": scott_id}))
        stream = None
        try:
            async with store.connection.execute("SELECT COALESCE(MAX(position), 0) FROM event_log") as cursor:
                tip_row = await cursor.fetchone()
            assert tip_row is not None
            stream = await client.get(
                f"/v1/client/events?after_position={int(tip_row[0])}",
                headers={
                    "Authorization": "Bearer scott",
                    "X-Kai-Stream-ID": "principal-test",
                },
            )
            assert stream.status == 200

            await _record(
                store,
                daniel_id,
                channel_id,
                "multiplexed-principal-event",
                "@scott one canonical message",
            )
            event = await _next_sse_event(stream)

            assert event["event"] == "workshop.principal.changed"
            data = event["data"]
            assert isinstance(data, dict)
            assert data["version"] == 1
            assert len(data["changes"]) == 2
            unread_change, notification_change = data["changes"]
            assert len(unread_change["unread_changes"]) == 1
            assert len(notification_change["notification_changes"]) == 1
            assert notification_change["notification_changes"][0]["event_type"] == ("human_notification.created")
            assert unread_change["unread_changes"][0]["state"]["channel_id"] == channel_id
            assert unread_change["event_position"] < notification_change["event_position"]
            assert notification_change["event_position"] == data["through_position"]
        finally:
            if stream is not None:
                stream.close()
            await client.close()
            await store.close()

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
