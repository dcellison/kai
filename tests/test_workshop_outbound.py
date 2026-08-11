"""Contracts for assistant-result and delivery-observation Workshop shadow writes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai import sessions
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import MessageId
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import (
    DeliveryObservation,
    OutboundMessage,
    OutboundMessageNotFoundError,
    record_delivery_observation,
    record_outbound_message,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore

_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


async def _open_with_inbound(path: Path) -> tuple[WorkshopEventStore, MessageId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Authorized Human",
                role="admin",
                transport="telegram",
                external_subject="101",
                external_channel_id="101",
            ),
        ),
    )
    inbound = await record_inbound_message(
        store,
        InboundMessage(
            transport="telegram",
            update_id="9001",
            message_id="42",
            sender_subject="101",
            channel_subject="101",
            body="Hello",
            occurred_at=_NOW,
        ),
    )
    assert isinstance(inbound.event.envelope.aggregate_id, MessageId)
    return store, inbound.event.envelope.aggregate_id


def _outbound(inbound_id: MessageId, *, body: str = "Hello back") -> OutboundMessage:
    return OutboundMessage(
        in_reply_to_message_id=inbound_id,
        body=body,
        occurred_at=_NOW + timedelta(seconds=2),
    )


class TestOutboundMessage:
    async def test_records_agent_authored_reply_in_same_channel(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            result = await record_outbound_message(store, _outbound(inbound_id))

            assert result.inserted is True
            assert isinstance(result.event.envelope.aggregate_id, MessageId)
            async with store.connection.execute(
                "SELECT m.body, m.channel_id, m.author_principal_id, m.reply_to_message_id, p.kind "
                "FROM messages m JOIN principals p ON p.id = m.author_principal_id "
                "WHERE m.id = ?",
                (result.event.envelope.aggregate_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "Hello back"
            assert str(row[1]).startswith("chn_")
            assert str(row[2]).startswith("prn_")
            assert row[3] == inbound_id
            assert row[4] == "agent"
            assert result.event.envelope.actor_principal_id == row[2]
            assert result.event.envelope.payload["channel_id"] == row[1]
        finally:
            await store.close()

    async def test_retry_is_idempotent_even_with_new_observation_time(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            first = await record_outbound_message(store, _outbound(inbound_id))
            retry = await record_outbound_message(
                store,
                OutboundMessage(
                    in_reply_to_message_id=inbound_id,
                    body="Hello back",
                    occurred_at=_NOW + timedelta(minutes=5),
                ),
            )

            assert first.inserted is True
            assert retry.inserted is False
            assert retry.event == first.event
        finally:
            await store.close()

    async def test_retry_survives_restart(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, inbound_id = await _open_with_inbound(path)
        first = await record_outbound_message(store, _outbound(inbound_id))
        await store.close()

        reopened = await WorkshopEventStore.open(path)
        try:
            retry = await record_outbound_message(reopened, _outbound(inbound_id))
            assert retry.inserted is False
            assert retry.event == first.event
        finally:
            await reopened.close()

    async def test_retry_catches_projection_up_after_interrupted_projection(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            first = await record_outbound_message(store, _outbound(inbound_id))
            assistant_id = first.event.envelope.aggregate_id
            await store.connection.execute("DELETE FROM messages WHERE id = ?", (assistant_id,))
            await store.connection.execute(
                "UPDATE projection_checkpoints SET last_position = ? WHERE name = 'canonical_conversations'",
                (first.event.position - 1,),
            )
            await store.connection.commit()

            retry = await record_outbound_message(store, _outbound(inbound_id))

            assert retry.inserted is False
            async with store.connection.execute("SELECT body FROM messages WHERE id = ?", (assistant_id,)) as cursor:
                assert (await cursor.fetchone())[0] == "Hello back"
        finally:
            await store.close()

    async def test_changed_body_for_same_inbound_message_fails_closed(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            await record_outbound_message(store, _outbound(inbound_id, body="original"))
            with pytest.raises(IdempotencyConflictError):
                await record_outbound_message(store, _outbound(inbound_id, body="changed"))
        finally:
            await store.close()

    async def test_requires_existing_canonical_inbound_message(self, tmp_path: Path):
        store, _ = await _open_with_inbound(tmp_path / "kai.db")
        try:
            with pytest.raises(OutboundMessageNotFoundError):
                await record_outbound_message(store, _outbound(MessageId.new()))
        finally:
            await store.close()


class TestDeliveryObservation:
    async def test_records_successful_telegram_text_delivery(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            outbound = await record_outbound_message(store, _outbound(inbound_id))
            message_id = MessageId(str(outbound.event.envelope.aggregate_id))
            result = await record_delivery_observation(
                store,
                DeliveryObservation(
                    message_id=message_id,
                    transport="telegram",
                    mode="text",
                    succeeded=True,
                    occurred_at=_NOW + timedelta(seconds=3),
                ),
            )

            assert result.inserted is True
            async with store.connection.execute("SELECT message_id, transport, mode, status FROM deliveries") as cursor:
                row = await cursor.fetchone()
            assert tuple(row) == (message_id, "telegram", "text", "succeeded")
        finally:
            await store.close()

    async def test_duplicate_status_is_idempotent(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            outbound = await record_outbound_message(store, _outbound(inbound_id))
            observation = DeliveryObservation(
                message_id=MessageId(str(outbound.event.envelope.aggregate_id)),
                transport="telegram",
                mode="text",
                succeeded=True,
                occurred_at=_NOW + timedelta(seconds=3),
            )
            first = await record_delivery_observation(store, observation)
            retry = await record_delivery_observation(store, observation)
            assert first.inserted is True
            assert retry.inserted is False
            assert retry.event == first.event
        finally:
            await store.close()

    async def test_failure_then_success_updates_single_projection(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            outbound = await record_outbound_message(store, _outbound(inbound_id))
            message_id = MessageId(str(outbound.event.envelope.aggregate_id))
            await record_delivery_observation(
                store,
                DeliveryObservation(message_id, "telegram", "text", False, _NOW + timedelta(seconds=3)),
            )
            await record_delivery_observation(
                store,
                DeliveryObservation(message_id, "telegram", "text", True, _NOW + timedelta(seconds=4)),
            )

            async with store.connection.execute("SELECT COUNT(*), status FROM deliveries") as cursor:
                row = await cursor.fetchone()
            assert tuple(row) == (1, "succeeded")
            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type LIKE 'delivery.%'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 2
        finally:
            await store.close()


class TestSharedDatabaseOutboundRecording:
    async def test_records_result_and_delivery_through_serialized_session_adapters(self, tmp_path: Path):
        await sessions.init_db(tmp_path / "kai.db")
        try:
            await sessions.bootstrap_workshop_foundation(
                (
                    BootstrapHuman(
                        display_name="Authorized Human",
                        role="admin",
                        transport="telegram",
                        external_subject="101",
                        external_channel_id="101",
                    ),
                )
            )
            inbound = await sessions.record_workshop_inbound_message(
                InboundMessage("telegram", "9001", "42", "101", "101", "Hello", _NOW)
            )
            inbound_id = MessageId(str(inbound.event.envelope.aggregate_id))
            outbound = await sessions.record_workshop_outbound_message(_outbound(inbound_id))
            outbound_id = MessageId(str(outbound.event.envelope.aggregate_id))
            delivery = await sessions.record_workshop_delivery_observation(
                DeliveryObservation(outbound_id, "telegram", "text", True, _NOW + timedelta(seconds=3))
            )

            assert outbound.inserted is True
            assert delivery.inserted is True
            async with sessions._get_db().execute("SELECT status FROM deliveries") as cursor:
                assert (await cursor.fetchone())[0] == "succeeded"
        finally:
            await sessions.close_db()


class TestDeliveryObservationReplay:
    async def test_text_and_voice_are_separate_delivery_modes(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            outbound = await record_outbound_message(store, _outbound(inbound_id))
            message_id = MessageId(str(outbound.event.envelope.aggregate_id))
            for mode in ("text", "voice"):
                await record_delivery_observation(
                    store,
                    DeliveryObservation(message_id, "telegram", mode, True, _NOW + timedelta(seconds=3)),
                )
            async with store.connection.execute("SELECT mode FROM deliveries ORDER BY mode") as cursor:
                assert [row[0] for row in await cursor.fetchall()] == ["text", "voice"]
        finally:
            await store.close()

    async def test_projection_rebuild_reproduces_messages_and_deliveries(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            outbound = await record_outbound_message(store, _outbound(inbound_id))
            await record_delivery_observation(
                store,
                DeliveryObservation(
                    MessageId(str(outbound.event.envelope.aggregate_id)),
                    "telegram",
                    "text",
                    True,
                    _NOW + timedelta(seconds=3),
                ),
            )
            async with store.connection.execute("SELECT * FROM messages ORDER BY id") as cursor:
                messages_before = [tuple(row) for row in await cursor.fetchall()]
            async with store.connection.execute("SELECT * FROM deliveries ORDER BY id") as cursor:
                deliveries_before = [tuple(row) for row in await cursor.fetchall()]

            await store.rebuild_projection(CanonicalConversationProjection())

            async with store.connection.execute("SELECT * FROM messages ORDER BY id") as cursor:
                assert [tuple(row) for row in await cursor.fetchall()] == messages_before
            async with store.connection.execute("SELECT * FROM deliveries ORDER BY id") as cursor:
                assert [tuple(row) for row in await cursor.fetchall()] == deliveries_before
        finally:
            await store.close()
