"""Contracts for assistant-result and delivery-observation Workshop shadow writes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest

from kai import sessions
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import (
    ChannelBindingId,
    EventEnvelope,
    MessageId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import (
    DeliveryObservation,
    OutboundDeliveryBindingError,
    OutboundDeliveryStateConflictError,
    OutboundMessage,
    OutboundMessageNotFoundError,
    record_delivery_observation,
    record_outbound_message,
    record_outbound_message_with_delivery,
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


class TestAtomicOutboundDelivery:
    async def test_commits_message_projection_request_event_and_pending_work_together(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            result = await record_outbound_message_with_delivery(store, _outbound(inbound_id))

            assert result.message.inserted is True
            assert result.delivery.inserted is True
            assert result.delivery.delivery.message_id == result.message.event.envelope.aggregate_id
            assert result.delivery.delivery.transport == "telegram"
            assert result.delivery.delivery.mode == "text"
            assert result.delivery.delivery.status == "pending"
            assert result.delivery.delivery.attempt_count == 0
            async with store.connection.execute(
                "SELECT body, reply_to_message_id FROM messages WHERE id = ?",
                (result.message.event.envelope.aggregate_id,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == ("Hello back", inbound_id)
            async with store.connection.execute(
                "SELECT event_type FROM event_log WHERE position IN (?, ?) ORDER BY position",
                (result.message.event.position, result.delivery.delivery.requested_event_position),
            ) as cursor:
                assert [row[0] for row in await cursor.fetchall()] == [
                    "message.created",
                    "delivery.requested",
                ]
            async with store.connection.execute(
                "SELECT last_position FROM projection_checkpoints WHERE name = 'canonical_conversations'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == result.delivery.delivery.requested_event_position
        finally:
            await store.close()

    async def test_retry_is_idempotent_across_restart_and_observation_time(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, inbound_id = await _open_with_inbound(path)
        first = await record_outbound_message_with_delivery(store, _outbound(inbound_id))
        await store.close()

        reopened = await WorkshopEventStore.open(path)
        try:
            retry = await record_outbound_message_with_delivery(
                reopened,
                OutboundMessage(inbound_id, "Hello back", _NOW + timedelta(minutes=5)),
            )

            assert retry.message.inserted is False
            assert retry.delivery.inserted is False
            assert retry.message.event == first.message.event
            assert retry.delivery.delivery == first.delivery.delivery
            async with reopened.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type IN ('message.created', 'delivery.requested')"
            ) as cursor:
                # One inbound message, one assistant message, and one request.
                assert (await cursor.fetchone())[0] == 3
            async with reopened.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await reopened.close()

    async def test_concurrent_retry_across_connections_creates_one_message_and_delivery(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        first_store, inbound_id = await _open_with_inbound(path)
        second_store = await WorkshopEventStore.open(path)
        try:
            first, second = await asyncio.gather(
                record_outbound_message_with_delivery(first_store, _outbound(inbound_id)),
                record_outbound_message_with_delivery(second_store, _outbound(inbound_id)),
            )

            assert sorted((first.message.inserted, second.message.inserted)) == [False, True]
            assert sorted((first.delivery.inserted, second.delivery.inserted)) == [False, True]
            assert first.message.event == second.message.event
            assert first.delivery.delivery == second.delivery.delivery
            async with first_store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type IN ('message.created', 'delivery.requested')"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 3
            async with first_store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await first_store.close()
            await second_store.close()

    async def test_changed_body_fails_closed_without_changing_delivery(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            first = await record_outbound_message_with_delivery(store, _outbound(inbound_id, body="original"))

            with pytest.raises(IdempotencyConflictError):
                await record_outbound_message_with_delivery(store, _outbound(inbound_id, body="changed"))

            async with store.connection.execute(
                "SELECT body FROM messages WHERE id = ?", (first.message.event.envelope.aggregate_id,)
            ) as cursor:
                assert (await cursor.fetchone())[0] == "original"
            async with store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await store.close()

    async def test_missing_telegram_binding_rolls_back_without_creating_reply(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            await store.connection.execute("DELETE FROM channel_bindings")
            await store.connection.commit()

            with pytest.raises(OutboundDeliveryBindingError):
                await record_outbound_message_with_delivery(store, _outbound(inbound_id))

            await self._assert_no_outbound_delivery_state(store, inbound_id)
        finally:
            await store.close()

    async def test_ambiguous_telegram_binding_rolls_back_without_guessing(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            async with store.connection.execute(
                "SELECT c.workshop_id, m.channel_id FROM messages m "
                "JOIN channels c ON c.id = m.channel_id WHERE m.id = ?",
                (inbound_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            workshop_id = WorkshopId(str(row[0]))
            channel_id = str(row[1])
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.TRANSPORT_CHANNEL_BOUND,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="channel_binding",
                    aggregate_id=ChannelBindingId.new(),
                    occurred_at=_NOW,
                    idempotency_key="test:second-telegram-binding",
                    payload={
                        "channel_id": channel_id,
                        "transport": "telegram",
                        "external_channel_id": "202",
                    },
                )
            )
            await store.project_pending(CanonicalConversationProjection())

            with pytest.raises(OutboundDeliveryBindingError):
                await record_outbound_message_with_delivery(store, _outbound(inbound_id))

            await self._assert_no_outbound_delivery_state(store, inbound_id)
        finally:
            await store.close()

    async def test_outbox_insert_failure_rolls_back_message_event_and_projection(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            await store.connection.execute(
                "CREATE TRIGGER reject_atomic_delivery BEFORE INSERT ON delivery_outbox "
                "BEGIN SELECT RAISE(ABORT, 'test delivery rejection'); END"
            )
            await store.connection.commit()

            with pytest.raises(aiosqlite.IntegrityError, match="test delivery rejection"):
                await record_outbound_message_with_delivery(store, _outbound(inbound_id))

            await self._assert_no_outbound_delivery_state(store, inbound_id)
        finally:
            await store.close()

    async def test_projection_failure_rolls_back_message_event_and_delivery(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            await store.connection.execute(
                "CREATE TRIGGER reject_atomic_projection BEFORE INSERT ON messages "
                "WHEN NEW.reply_to_message_id IS NOT NULL "
                "BEGIN SELECT RAISE(ABORT, 'test projection rejection'); END"
            )
            await store.connection.commit()

            with pytest.raises(aiosqlite.IntegrityError, match="test projection rejection"):
                await record_outbound_message_with_delivery(store, _outbound(inbound_id))

            await self._assert_no_outbound_delivery_state(store, inbound_id)
        finally:
            await store.close()

    async def test_preexisting_message_without_delivery_is_rejected_as_half_state(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            prior = await record_outbound_message(store, _outbound(inbound_id))

            with pytest.raises(OutboundDeliveryStateConflictError):
                await record_outbound_message_with_delivery(store, _outbound(inbound_id))

            async with store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
                assert (await cursor.fetchone())[0] == 0
            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type = 'delivery.requested'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0
            async with store.connection.execute(
                "SELECT body FROM messages WHERE id = ?", (prior.event.envelope.aggregate_id,)
            ) as cursor:
                assert (await cursor.fetchone())[0] == "Hello back"
        finally:
            await store.close()

    @staticmethod
    async def _assert_no_outbound_delivery_state(store: WorkshopEventStore, inbound_id: MessageId) -> None:
        async with store.connection.execute(
            "SELECT COUNT(*) FROM event_log WHERE "
            "(event_type = 'message.created' AND json_extract(payload_json, '$.reply_to_message_id') = ?) "
            "OR event_type = 'delivery.requested'",
            (inbound_id,),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0
        async with store.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE reply_to_message_id = ?",
            (inbound_id,),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0
        async with store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
            assert (await cursor.fetchone())[0] == 0


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
