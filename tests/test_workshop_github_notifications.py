"""Contracts for canonical external integration notification recording."""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai import workshop_cli
from kai.workshop.bootstrap import (
    BootstrapHuman,
    BootstrapNotificationChannel,
    bootstrap_default_workshop,
)
from kai.workshop.delivery_outbox import NOTIFICATION_PURPOSE
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.domain import ChannelId
from kai.workshop.integration_notifications import (
    IntegrationNotification,
    IntegrationNotificationError,
    WorkshopIntegrationNotificationService,
)
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore
from tests.workshop_delivery import DISABLED_DELIVERY_POLICY, TELEGRAM_DELIVERY_POLICY

_NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)


async def _open_notification_store(path: Path) -> WorkshopEventStore:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Daniel",
                role="admin",
                transport="telegram",
                external_subject="101",
                external_channel_id="101",
            ),
        ),
        notification_channels=(
            BootstrapNotificationChannel(
                transport="telegram",
                external_channel_id="-100123",
                member_external_subjects=("101",),
            ),
        ),
    )
    return store


async def _notification_channel(store: WorkshopEventStore) -> ChannelId:
    async with store.connection.execute("SELECT id FROM channels WHERE kind = 'notification'") as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return ChannelId(str(row[0]))


def _notification(*, body: str = "**Push** to [kai](https://github.com/example/kai)") -> IntegrationNotification:
    return IntegrationNotification(
        delivery_id="f8112a52-7129-11f1-8e31-acde48001122",
        source="github",
        event_type="push",
        repository="example/kai",
        body=body,
        occurred_at=_NOW,
    )


class TestIntegrationNotificationInput:
    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"delivery_id": "bad delivery"}, "delivery_id"),
            ({"source": "GitHub"}, "source"),
            ({"event_type": "Pull-Request"}, "event_type"),
            ({"repository": "not-a-repository"}, "repository"),
            ({"body": ""}, "body"),
            ({"occurred_at": datetime(2026, 8, 13)}, "occurred_at"),
        ],
    )
    def test_invalid_input_fails_before_storage(self, changes, match):
        values = {
            "delivery_id": "delivery-1",
            "source": "github",
            "event_type": "push",
            "repository": "example/kai",
            "body": "Notification",
            "occurred_at": _NOW,
        }
        values.update(changes)

        with pytest.raises(ValueError, match=match):
            IntegrationNotification(**values)


class TestWorkshopIntegrationNotificationService:
    async def test_open_seeds_default_generic_route_from_canonical_admin(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store = await _open_notification_store(path)
        await store.close()

        service = await WorkshopIntegrationNotificationService.open(path, TELEGRAM_DELIVERY_POLICY)
        try:
            status = await service.route_status(source="generic", route_name="default")

            assert status.state == "active"
            assert status.channel_id is not None
            assert status.detail == "canonical destination is configured"
        finally:
            await service.close()

    async def test_route_records_without_resolving_any_transport_binding(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store = await _open_notification_store(path)
        await store.close()
        service = await WorkshopIntegrationNotificationService.open(path, TELEGRAM_DELIVERY_POLICY)
        try:
            notification = IntegrationNotification(
                delivery_id="generic-route-1",
                source="generic",
                event_type="notification",
                body="Canonical route",
                occurred_at=_NOW,
            )

            result = await service.record_for_route(notification, route_name="default")

            assert result.inserted is True
            assert len(result.deliveries) == 1
        finally:
            await service.close()

    async def test_disabled_adapter_retains_binding_without_enqueuing_notification(self, tmp_path: Path):
        store = await _open_notification_store(tmp_path / "kai.db")
        try:
            result = await WorkshopIntegrationNotificationService(
                store,
                DISABLED_DELIVERY_POLICY,
            ).record_for_channel(_notification(), await _notification_channel(store))

            assert result.inserted is True
            assert result.deliveries == ()
            async with store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
                assert int((await cursor.fetchone())[0]) == 0
            async with store.connection.execute(
                "SELECT COUNT(*) FROM channel_bindings WHERE transport = 'telegram'"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 2

            # Enabling an adapter later does not discover or enqueue historical
            # canonical messages. Only new publication work can request delivery.
            enabled_service = WorkshopIntegrationNotificationService(
                store,
                TELEGRAM_DELIVERY_POLICY,
            )
            await enabled_service.route_status(source="generic", route_name="default")
            async with store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
                assert int((await cursor.fetchone())[0]) == 0
        finally:
            await store.close()

    async def test_mixed_bindings_enqueue_only_for_enabled_transport(self, tmp_path: Path):
        store = await _open_notification_store(tmp_path / "kai.db")
        channel_id = await _notification_channel(store)
        await store.connection.execute(
            "INSERT INTO channel_bindings (id, channel_id, transport, external_channel_id, created_at) "
            "VALUES (?, ?, 'desktop', 'desktop-notifications', '2026-08-25T00:00:00Z')",
            ("cbd_00000000000000000000000000000002", channel_id),
        )
        await store.connection.commit()
        try:
            desktop_policy = WorkshopDeliveryBindingPolicy(frozenset({"desktop"}))
            result = await WorkshopIntegrationNotificationService(store, desktop_policy).record_for_channel(
                _notification(),
                channel_id,
            )

            assert len(result.deliveries) == 1
            assert result.deliveries[0].delivery.transport == "desktop"
            async with store.connection.execute(
                "SELECT transport FROM delivery_outbox WHERE message_id = ?",
                (result.message_id,),
            ) as cursor:
                assert [str(row[0]) for row in await cursor.fetchall()] == ["desktop"]
        finally:
            await store.close()

    async def test_ambiguous_admin_policy_does_not_seed_or_deliver(self, tmp_path: Path):
        store = await WorkshopEventStore.open(tmp_path / "kai.db")
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    display_name="Daniel",
                    role="admin",
                    transport="telegram",
                    external_subject="101",
                    external_channel_id="101",
                ),
                BootstrapHuman(
                    display_name="Scott",
                    role="admin",
                    transport="telegram",
                    external_subject="202",
                    external_channel_id="202",
                ),
            ),
        )
        try:
            service = WorkshopIntegrationNotificationService(store, TELEGRAM_DELIVERY_POLICY)
            status = await service.reconcile_default_generic_route()

            assert status.state == "ambiguous"
            with pytest.raises(IntegrationNotificationError, match="is missing"):
                await service.record_for_route(
                    IntegrationNotification(
                        delivery_id="generic-route-ambiguous",
                        source="generic",
                        event_type="notification",
                        body="Do not route",
                        occurred_at=_NOW,
                    ),
                    route_name="default",
                )
        finally:
            await store.close()

    async def test_operator_can_inspect_and_assign_default_route(
        self,
        tmp_path: Path,
        monkeypatch,
        capsys,
    ):
        path = tmp_path / "kai.db"
        store = await _open_notification_store(path)
        async with store.connection.execute("SELECT id FROM channels WHERE kind = 'notification'") as cursor:
            channel_id = str((await cursor.fetchone())[0])
        await store.close()
        monkeypatch.setattr(workshop_cli, "DATA_DIR", tmp_path)

        assert await workshop_cli._run(Namespace(command="integration-route", action="status")) == 2
        assert "Status: missing" in capsys.readouterr().out

        assert (
            await workshop_cli._run(
                Namespace(
                    command="integration-route",
                    action="set",
                    channel_id=channel_id,
                )
            )
            == 0
        )
        output = capsys.readouterr().out
        assert "Integration route: generic/default" in output
        assert "Status: active" in output
        assert f"Channel: {channel_id}" in output

    async def test_operator_can_set_route_to_canonical_notification_channel(self, tmp_path: Path):
        store = await _open_notification_store(tmp_path / "kai.db")
        try:
            async with store.connection.execute("SELECT id FROM channels WHERE kind = 'notification'") as cursor:
                channel_id = ChannelId(str((await cursor.fetchone())[0]))
            service = WorkshopIntegrationNotificationService(store, TELEGRAM_DELIVERY_POLICY)

            status = await service.set_route(
                source="generic",
                route_name="default",
                channel_id=channel_id,
            )

            assert status.state == "active"
            assert status.channel_id == channel_id
        finally:
            await store.close()

    async def test_records_directly_to_authorized_canonical_channel(self, tmp_path: Path):
        store = await _open_notification_store(tmp_path / "kai.db")
        try:
            async with store.connection.execute("SELECT id FROM channels WHERE kind = 'notification'") as cursor:
                channel_id = ChannelId(str((await cursor.fetchone())[0]))

            result = await WorkshopIntegrationNotificationService(store, TELEGRAM_DELIVERY_POLICY).record_for_channel(
                _notification(),
                channel_id,
            )

            async with store.connection.execute(
                "SELECT channel_id FROM messages WHERE id = ?",
                (result.message_id,),
            ) as cursor:
                assert str((await cursor.fetchone())[0]) == channel_id
            assert len(result.deliveries) == 1
        finally:
            await store.close()

    async def test_atomically_records_feed_entry_and_durable_telegram_work(
        self,
        tmp_path: Path,
    ):
        store = await _open_notification_store(tmp_path / "kai.db")
        try:
            result = await WorkshopIntegrationNotificationService(store, TELEGRAM_DELIVERY_POLICY).record_for_binding(
                _notification(),
                transport="telegram",
                external_channel_id="-100123",
            )

            assert result is not None
            assert result.inserted is True
            assert len(result.deliveries) == 1
            assert result.deliveries[0].inserted is True
            assert result.deliveries[0].delivery.purpose == NOTIFICATION_PURPOSE
            assert result.deliveries[0].delivery.status == "pending"
            async with store.connection.execute(
                "SELECT c.kind, m.body, p.display_name FROM messages m "
                "JOIN channels c ON c.id = m.channel_id "
                "JOIN principals p ON p.id = m.author_principal_id WHERE m.id = ?",
                (result.message_id,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == (
                    "notification",
                    _notification().body,
                    "Kai",
                )
            async with store.connection.execute(
                "SELECT cb.id, e.idempotency_key, e.metadata_json "
                "FROM channel_bindings cb JOIN event_log e ON e.aggregate_id = ? "
                "WHERE cb.channel_id = (SELECT channel_id FROM messages WHERE id = ?)",
                (result.message_id, result.message_id),
            ) as cursor:
                binding_id, idempotency_key, metadata_json = await cursor.fetchone()
            assert idempotency_key == (f"workshop-github-notification:v1:{binding_id}:{_notification().delivery_id}")
            assert json.loads(metadata_json) == {
                "source": "github",
                "github_delivery_id": _notification().delivery_id,
                "github_event": "push",
                "repository": "example/kai",
            }
        finally:
            await store.close()

    async def test_same_github_delivery_and_destination_is_idempotent_across_restart(
        self,
        tmp_path: Path,
    ):
        path = tmp_path / "kai.db"
        store = await _open_notification_store(path)
        first = await WorkshopIntegrationNotificationService(store, TELEGRAM_DELIVERY_POLICY).record_for_binding(
            _notification(),
            transport="telegram",
            external_channel_id="-100123",
        )
        await store.close()

        reopened = await WorkshopEventStore.open(path)
        try:
            retry = await WorkshopIntegrationNotificationService(reopened, TELEGRAM_DELIVERY_POLICY).record_for_binding(
                IntegrationNotification(
                    **{
                        **{
                            "delivery_id": _notification().delivery_id,
                            "source": _notification().source,
                            "event_type": _notification().event_type,
                            "repository": _notification().repository,
                            "body": _notification().body,
                        },
                        "occurred_at": _NOW + timedelta(minutes=1),
                    }
                ),
                transport="telegram",
                external_channel_id="-100123",
            )
            assert first is not None and retry is not None
            assert retry.message_id == first.message_id
            assert retry.inserted is False
            assert retry.deliveries[0].inserted is False
            async with reopened.connection.execute(
                "SELECT COUNT(*) FROM delivery_outbox WHERE purpose = 'notification'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await reopened.close()

    async def test_reused_delivery_identity_with_changed_content_fails_closed(
        self,
        tmp_path: Path,
    ):
        store = await _open_notification_store(tmp_path / "kai.db")
        try:
            recorder = WorkshopIntegrationNotificationService(store, TELEGRAM_DELIVERY_POLICY)
            await recorder.record_for_binding(_notification(), transport="telegram", external_channel_id="-100123")
            with pytest.raises(IdempotencyConflictError):
                await recorder.record_for_binding(
                    _notification(body="Different notification"),
                    transport="telegram",
                    external_channel_id="-100123",
                )
        finally:
            await store.close()

    async def test_unbound_destination_returns_none_without_creating_message(
        self,
        tmp_path: Path,
    ):
        store = await _open_notification_store(tmp_path / "kai.db")
        try:
            notification = IntegrationNotification(
                delivery_id="delivery-2",
                source="github",
                event_type="push",
                repository="example/kai",
                body="Unbound",
                occurred_at=_NOW,
            )
            assert (
                await WorkshopIntegrationNotificationService(store, TELEGRAM_DELIVERY_POLICY).record_for_binding(
                    notification,
                    transport="telegram",
                    external_channel_id="-100999",
                )
                is None
            )
            async with store.connection.execute("SELECT COUNT(*) FROM messages") as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await store.close()

    async def test_default_admin_records_canonical_direct_message_and_delivery(self, tmp_path: Path):
        store = await _open_notification_store(tmp_path / "kai.db")
        try:
            notification = IntegrationNotification(
                delivery_id="generic-1",
                source="generic",
                event_type="notification",
                body="Build complete",
                occurred_at=_NOW,
            )
            result = await WorkshopIntegrationNotificationService(
                store, TELEGRAM_DELIVERY_POLICY
            ).record_for_default_admin(notification)

            assert result.inserted is True
            assert len(result.deliveries) == 1
            async with store.connection.execute(
                "SELECT c.kind, m.body FROM messages m JOIN channels c ON c.id = m.channel_id WHERE m.id = ?",
                (result.message_id,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == ("direct", "Build complete")
        finally:
            await store.close()

    async def test_generic_delivery_identity_is_idempotent(self, tmp_path: Path):
        store = await _open_notification_store(tmp_path / "kai.db")
        try:
            service = WorkshopIntegrationNotificationService(store, TELEGRAM_DELIVERY_POLICY)
            notification = IntegrationNotification(
                delivery_id="generic-retry-1",
                source="generic",
                event_type="notification",
                body="Build complete",
                occurred_at=_NOW,
            )

            first = await service.record_for_default_admin(notification)
            retry = await service.record_for_default_admin(notification)

            assert first.message_id == retry.message_id
            assert first.inserted is True
            assert retry.inserted is False
            assert retry.deliveries[0].inserted is False
        finally:
            await store.close()
