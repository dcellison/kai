"""Contracts for canonical GitHub notification feed recording."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.workshop.bootstrap import (
    BootstrapHuman,
    BootstrapNotificationChannel,
    bootstrap_default_workshop,
)
from kai.workshop.delivery_outbox import NOTIFICATION_PURPOSE
from kai.workshop.github_notifications import (
    GitHubNotification,
    WorkshopGitHubNotificationRecorder,
)
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore

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


def _notification(*, body: str = "**Push** to [kai](https://github.com/example/kai)") -> GitHubNotification:
    return GitHubNotification(
        delivery_id="f8112a52-7129-11f1-8e31-acde48001122",
        event_type="push",
        repository="example/kai",
        telegram_chat_id=-100123,
        body=body,
        occurred_at=_NOW,
    )


class TestGitHubNotificationInput:
    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"delivery_id": "bad delivery"}, "delivery_id"),
            ({"event_type": "Pull-Request"}, "event_type"),
            ({"repository": "not-a-repository"}, "repository"),
            ({"telegram_chat_id": 101}, "telegram_chat_id"),
            ({"body": ""}, "body"),
            ({"occurred_at": datetime(2026, 8, 13)}, "occurred_at"),
        ],
    )
    def test_invalid_input_fails_before_storage(self, changes, match):
        values = {
            "delivery_id": "delivery-1",
            "event_type": "push",
            "repository": "example/kai",
            "telegram_chat_id": -100123,
            "body": "Notification",
            "occurred_at": _NOW,
        }
        values.update(changes)

        with pytest.raises(ValueError, match=match):
            GitHubNotification(**values)


class TestWorkshopGitHubNotificationRecorder:
    async def test_atomically_records_feed_entry_and_durable_telegram_work(
        self,
        tmp_path: Path,
    ):
        store = await _open_notification_store(tmp_path / "kai.db")
        try:
            result = await WorkshopGitHubNotificationRecorder(store).record(_notification())

            assert result is not None
            assert result.delivery.inserted is True
            assert result.delivery.delivery.purpose == NOTIFICATION_PURPOSE
            assert result.delivery.delivery.status == "pending"
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
        finally:
            await store.close()

    async def test_same_github_delivery_and_destination_is_idempotent_across_restart(
        self,
        tmp_path: Path,
    ):
        path = tmp_path / "kai.db"
        store = await _open_notification_store(path)
        first = await WorkshopGitHubNotificationRecorder(store).record(_notification())
        await store.close()

        reopened = await WorkshopEventStore.open(path)
        try:
            retry = await WorkshopGitHubNotificationRecorder(reopened).record(
                GitHubNotification(
                    **{
                        **{
                            "delivery_id": _notification().delivery_id,
                            "event_type": _notification().event_type,
                            "repository": _notification().repository,
                            "telegram_chat_id": _notification().telegram_chat_id,
                            "body": _notification().body,
                        },
                        "occurred_at": _NOW + timedelta(minutes=1),
                    }
                )
            )
            assert first is not None and retry is not None
            assert retry.message_id == first.message_id
            assert retry.delivery.inserted is False
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
            recorder = WorkshopGitHubNotificationRecorder(store)
            await recorder.record(_notification())
            with pytest.raises(IdempotencyConflictError):
                await recorder.record(_notification(body="Different notification"))
        finally:
            await store.close()

    async def test_unbound_destination_returns_none_without_creating_message(
        self,
        tmp_path: Path,
    ):
        store = await _open_notification_store(tmp_path / "kai.db")
        try:
            notification = GitHubNotification(
                delivery_id="delivery-2",
                event_type="push",
                repository="example/kai",
                telegram_chat_id=-100999,
                body="Unbound",
                occurred_at=_NOW,
            )
            assert await WorkshopGitHubNotificationRecorder(store).record(notification) is None
            async with store.connection.execute("SELECT COUNT(*) FROM messages") as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await store.close()
