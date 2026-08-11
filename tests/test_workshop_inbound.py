"""Tests for authenticated Telegram inbound Workshop shadow records."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai import sessions
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.inbound import InboundBindingNotFoundError, InboundMessage, record_inbound_message
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore


def _human(telegram_id: int) -> BootstrapHuman:
    return BootstrapHuman(
        display_name="Authorized Human",
        role="admin",
        transport="telegram",
        external_subject=str(telegram_id),
        external_channel_id=str(telegram_id),
    )


def _message(
    *,
    update_id: str = "9001",
    message_id: str = "42",
    sender: str = "101",
    channel: str = "101",
    body: str = "Hello from Telegram",
) -> InboundMessage:
    return InboundMessage(
        transport="telegram",
        update_id=update_id,
        message_id=message_id,
        sender_subject=sender,
        channel_subject=channel,
        body=body,
        occurred_at=datetime(2026, 8, 11, 12, 0, tzinfo=UTC),
    )


async def _open_bootstrapped_store(path: Path) -> WorkshopEventStore:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(store, [_human(101)])
    return store


class TestInboundMessage:
    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"transport": "Telegram"}, "transport"),
            ({"update_id": ""}, "update_id"),
            ({"message_id": ""}, "message_id"),
            ({"sender_subject": ""}, "sender_subject"),
            ({"channel_subject": ""}, "channel_subject"),
            ({"body": ""}, "body"),
        ],
    )
    def test_rejects_invalid_transport_input(self, changes, match):
        values = {
            "transport": "telegram",
            "update_id": "9001",
            "message_id": "42",
            "sender_subject": "101",
            "channel_subject": "101",
            "body": "hello",
            "occurred_at": datetime(2026, 8, 11, tzinfo=UTC),
        }
        values.update(changes)

        with pytest.raises(ValueError, match=match):
            InboundMessage(**values)

    def test_requires_timezone_aware_occurrence_time(self):
        with pytest.raises(ValueError, match="occurred_at"):
            InboundMessage(
                transport="telegram",
                update_id="9001",
                message_id="42",
                sender_subject="101",
                channel_subject="101",
                body="hello",
                occurred_at=datetime(2026, 8, 11),
            )


class TestInboundShadowRecording:
    async def test_records_canonical_author_channel_and_message(self, tmp_path: Path):
        store = await _open_bootstrapped_store(tmp_path / "kai.db")
        try:
            result = await record_inbound_message(store, _message())

            assert result.inserted is True
            async with store.connection.execute(
                "SELECT m.id, m.body, m.author_principal_id, m.channel_id, "
                "p.id, c.id FROM messages m "
                "JOIN principals p ON p.id = m.author_principal_id "
                "JOIN channels c ON c.id = m.channel_id"
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            assert row[0].startswith("msg_")
            assert row[1] == "Hello from Telegram"
            assert row[2] == row[4]
            assert row[3] == row[5]
            assert row[2].startswith("prn_")
            assert row[3].startswith("chn_")

            event = result.event.envelope
            assert event.event_type == "message.created"
            assert event.actor_principal_id == row[2]
            assert event.metadata == {
                "source": "telegram",
                "transport_message_id": "42",
                "transport_update_id": "9001",
            }
            assert "101" not in (event.idempotency_key or "")
        finally:
            await store.close()

    async def test_duplicate_delivery_is_idempotent(self, tmp_path: Path):
        store = await _open_bootstrapped_store(tmp_path / "kai.db")
        try:
            first = await record_inbound_message(store, _message())
            second = await record_inbound_message(store, _message())

            assert first.inserted is True
            assert second.inserted is False
            assert second.event == first.event
            async with store.connection.execute("SELECT COUNT(*) FROM messages") as cursor:
                assert (await cursor.fetchone())[0] == 1
            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type = 'message.created'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await store.close()

    async def test_duplicate_delivery_remains_idempotent_after_restart(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        first_store = await _open_bootstrapped_store(db_path)
        first = await record_inbound_message(first_store, _message())
        await first_store.close()

        second_store = await WorkshopEventStore.open(db_path)
        try:
            second = await record_inbound_message(second_store, _message())

            assert second.inserted is False
            assert second.event == first.event
            async with second_store.connection.execute("SELECT COUNT(*) FROM messages") as cursor:
                assert (await cursor.fetchone())[0] == 1
            async with second_store.connection.execute(
                "SELECT last_position FROM projection_checkpoints WHERE name = 'canonical_conversations'"
            ) as cursor:
                checkpoint = await cursor.fetchone()
            async with second_store.connection.execute("SELECT MAX(position) FROM event_log") as cursor:
                maximum = await cursor.fetchone()
            assert checkpoint[0] == maximum[0]
        finally:
            await second_store.close()

    async def test_same_transport_identity_with_changed_content_is_rejected(self, tmp_path: Path):
        store = await _open_bootstrapped_store(tmp_path / "kai.db")
        try:
            await record_inbound_message(store, _message(body="original"))

            with pytest.raises(IdempotencyConflictError):
                await record_inbound_message(store, _message(body="changed"))
        finally:
            await store.close()

    @pytest.mark.parametrize(
        "message",
        [
            _message(sender="999", channel="101"),
            _message(sender="101", channel="999"),
        ],
    )
    async def test_requires_bound_identity_and_matching_direct_channel(self, tmp_path: Path, message):
        store = await _open_bootstrapped_store(tmp_path / "kai.db")
        try:
            with pytest.raises(InboundBindingNotFoundError):
                await record_inbound_message(store, message)

            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type = 'message.created'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await store.close()

    async def test_rejects_another_humans_existing_direct_channel(self, tmp_path: Path):
        store = await WorkshopEventStore.open(tmp_path / "kai.db")
        await bootstrap_default_workshop(store, [_human(101), _human(202)])
        try:
            with pytest.raises(InboundBindingNotFoundError, match="Direct transport channel"):
                await record_inbound_message(store, _message(sender="101", channel="202"))
        finally:
            await store.close()


class TestSharedDatabaseInboundRecording:
    async def test_serializes_concurrent_duplicate_updates(self, tmp_path: Path):
        await sessions.init_db(tmp_path / "kai.db")
        try:
            await sessions.bootstrap_workshop_foundation((_human(101),))

            first, second = await asyncio.gather(
                sessions.record_workshop_inbound_message(_message()),
                sessions.record_workshop_inbound_message(_message()),
            )

            assert sorted((first.inserted, second.inserted)) == [False, True]
            async with sessions._get_db().execute("SELECT COUNT(*) FROM messages") as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await sessions.close_db()
