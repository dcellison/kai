"""Contracts for the production-unused Workshop delivery outbox foundation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.delivery_outbox import (
    CONVERSATION_REPLY_PURPOSE,
    QUALIFICATION_PURPOSE,
    SEND_FRAGMENTS_CONTRACT,
    STREAMING_FINALIZATION_CONTRACT,
    DeliveryRequest,
    DeliveryRequestConflictError,
    DeliveryTargetNotFoundError,
    StaleDeliveryLeaseError,
    WorkshopDeliveryOutbox,
)
from kai.workshop.domain import (
    ChannelBindingId,
    ChannelId,
    DeliveryId,
    EventEnvelope,
    MessageId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import (
    DeliveryObservation,
    OutboundMessage,
    record_delivery_observation,
    record_outbound_message,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


@dataclass
class _Clock:
    now: datetime = _NOW

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


async def _open_with_outbound(path: Path) -> tuple[WorkshopEventStore, MessageId, ChannelBindingId]:
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
    outbound = await record_outbound_message(
        store,
        OutboundMessage(
            in_reply_to_message_id=MessageId(str(inbound.event.envelope.aggregate_id)),
            body="Hello back",
            occurred_at=_NOW,
        ),
    )
    message_id = MessageId(str(outbound.event.envelope.aggregate_id))
    async with store.connection.execute(
        "SELECT cb.id FROM channel_bindings cb JOIN messages m ON m.channel_id = cb.channel_id "
        "WHERE m.id = ? AND cb.transport = 'telegram'",
        (message_id,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return store, message_id, ChannelBindingId(str(row[0]))


async def _add_outbound(store: WorkshopEventStore, sequence: int) -> MessageId:
    inbound = await record_inbound_message(
        store,
        InboundMessage(
            transport="telegram",
            update_id=str(9001 + sequence),
            message_id=str(42 + sequence),
            sender_subject="101",
            channel_subject="101",
            body=f"Message {sequence}",
            occurred_at=_NOW + timedelta(seconds=sequence),
        ),
    )
    outbound = await record_outbound_message(
        store,
        OutboundMessage(
            in_reply_to_message_id=MessageId(str(inbound.event.envelope.aggregate_id)),
            body=f"Reply {sequence}",
            occurred_at=_NOW + timedelta(seconds=sequence),
        ),
    )
    return MessageId(str(outbound.event.envelope.aggregate_id))


async def _add_group_binding(
    store: WorkshopEventStore,
    message_id: MessageId,
) -> ChannelBindingId:
    async with store.connection.execute(
        "SELECT c.workshop_id, m.channel_id FROM messages m JOIN channels c ON c.id = m.channel_id WHERE m.id = ?",
        (message_id,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    workshop_id = WorkshopId(str(row[0]))
    channel_id = str(row[1])
    group_binding_id = ChannelBindingId.new()
    await store.append(
        EventEnvelope.create(
            event_type=WorkshopEventType.TRANSPORT_CHANNEL_BOUND,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="channel_binding",
            aggregate_id=group_binding_id,
            occurred_at=_NOW,
            idempotency_key=f"test:notification-group-binding:{group_binding_id}",
            payload={
                "channel_id": channel_id,
                "transport": "telegram",
                "external_channel_id": "-100123",
            },
        )
    )
    await store.project_pending(CanonicalConversationProjection())
    return group_binding_id


def _request(
    message_id: MessageId,
    binding_id: ChannelBindingId,
    *,
    mode: str = "text",
    purpose=QUALIFICATION_PURPOSE,
    execution_contract=SEND_FRAGMENTS_CONTRACT,
    max_attempts: int = 5,
) -> DeliveryRequest:
    return DeliveryRequest(
        message_id=message_id,
        channel_binding_id=binding_id,
        mode=mode,
        purpose=purpose,
        occurred_at=_NOW,
        execution_contract=execution_contract,
        max_attempts=max_attempts,
    )


class TestDeliveryRequests:
    async def test_transactional_request_requires_caller_transaction(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        try:
            with pytest.raises(RuntimeError, match="active transaction"):
                await WorkshopDeliveryOutbox(store).request_delivery_in_transaction(_request(message_id, binding_id))
        finally:
            await store.close()

    async def test_request_atomically_records_event_and_pending_work_without_delivering(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        try:
            result = await WorkshopDeliveryOutbox(store, clock=_Clock()).request_delivery(
                _request(message_id, binding_id)
            )

            assert result.inserted is True
            assert result.delivery.status == "pending"
            assert result.delivery.attempt_count == 0
            events = [
                event
                for event in await store.read_events()
                if event.envelope.event_type == WorkshopEventType.DELIVERY_REQUESTED
            ]
            assert len(events) == 1
            assert events[0].position == result.delivery.requested_event_position
            assert events[0].envelope.event_version == 2
            assert result.delivery.purpose == QUALIFICATION_PURPOSE
            assert result.delivery.execution_contract == SEND_FRAGMENTS_CONTRACT
            assert events[0].envelope.payload == {
                "message_id": message_id,
                "channel_id": result.delivery.channel_id,
                "channel_binding_id": binding_id,
                "transport": "telegram",
                "mode": "text",
                "purpose": QUALIFICATION_PURPOSE,
                "max_attempts": 5,
            }
            async with store.connection.execute("SELECT COUNT(*) FROM delivery_attempts") as cursor:
                assert (await cursor.fetchone())[0] == 0
            async with store.connection.execute("SELECT COUNT(*) FROM deliveries") as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await store.close()

    async def test_duplicate_request_is_idempotent_across_restart(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, message_id, binding_id = await _open_with_outbound(path)
        first = await WorkshopDeliveryOutbox(store).request_delivery(_request(message_id, binding_id))
        await store.close()

        reopened = await WorkshopEventStore.open(path)
        try:
            retry = await WorkshopDeliveryOutbox(reopened).request_delivery(_request(message_id, binding_id))
            assert retry.inserted is False
            assert retry.delivery == first.delivery
            async with reopened.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await reopened.close()

    async def test_duplicate_request_with_changed_policy_fails_closed(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        try:
            outbox = WorkshopDeliveryOutbox(store)
            await outbox.request_delivery(_request(message_id, binding_id, max_attempts=5))
            with pytest.raises(DeliveryRequestConflictError):
                await outbox.request_delivery(_request(message_id, binding_id, max_attempts=4))
        finally:
            await store.close()

    async def test_duplicate_request_with_changed_purpose_fails_closed(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        try:
            outbox = WorkshopDeliveryOutbox(store)
            await outbox.request_delivery(_request(message_id, binding_id))
            with pytest.raises(DeliveryRequestConflictError):
                await outbox.request_delivery(_request(message_id, binding_id, purpose=CONVERSATION_REPLY_PURPOSE))
        finally:
            await store.close()

    async def test_duplicate_request_with_changed_execution_contract_fails_closed(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        try:
            outbox = WorkshopDeliveryOutbox(store)
            await outbox.request_delivery(_request(message_id, binding_id))
            with pytest.raises(DeliveryRequestConflictError):
                await outbox.request_delivery(
                    _request(
                        message_id,
                        binding_id,
                        execution_contract=STREAMING_FINALIZATION_CONTRACT,
                    )
                )
        finally:
            await store.close()

    async def test_same_message_can_target_multiple_registered_telegram_bindings(self, tmp_path: Path):
        store, message_id, direct_binding_id = await _open_with_outbound(tmp_path / "kai.db")
        try:
            group_binding_id = await _add_group_binding(store, message_id)
            outbox = WorkshopDeliveryOutbox(store)

            direct = await outbox.request_delivery(_request(message_id, direct_binding_id))
            group = await outbox.request_delivery(_request(message_id, group_binding_id))

            assert direct.delivery.delivery_id != group.delivery.delivery_id
            async with store.connection.execute(
                "SELECT channel_binding_id FROM delivery_outbox ORDER BY requested_event_position"
            ) as cursor:
                assert [row[0] for row in await cursor.fetchall()] == [
                    direct_binding_id,
                    group_binding_id,
                ]
        finally:
            await store.close()

    async def test_request_requires_message_and_binding_in_same_canonical_channel(self, tmp_path: Path):
        store, message_id, _ = await _open_with_outbound(tmp_path / "kai.db")
        try:
            with pytest.raises(DeliveryTargetNotFoundError):
                await WorkshopDeliveryOutbox(store).request_delivery(_request(message_id, ChannelBindingId.new()))
        finally:
            await store.close()

    async def test_failed_outbox_insert_rolls_request_event_back(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        try:
            await store.connection.execute(
                "CREATE TRIGGER reject_delivery_request BEFORE INSERT ON delivery_outbox "
                "BEGIN SELECT RAISE(ABORT, 'test rejection'); END"
            )
            await store.connection.commit()

            with pytest.raises(aiosqlite.IntegrityError, match="test rejection"):
                await WorkshopDeliveryOutbox(store).request_delivery(_request(message_id, binding_id))

            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type = 'delivery.requested'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0
            async with store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await store.close()


class TestDeliveryOutcomes:
    async def test_success_appends_and_projects_exact_binding_aware_fact_idempotently(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        outbox = WorkshopDeliveryOutbox(store, clock=_Clock())
        try:
            requested = await outbox.request_delivery(_request(message_id, binding_id))
            claim = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert claim is not None

            succeeded = await outbox.mark_succeeded(claim)
            assert succeeded.status == "succeeded"
            assert await outbox.mark_succeeded(claim) == succeeded

            events = [
                event
                for event in await store.read_events()
                if event.envelope.event_type
                in {WorkshopEventType.DELIVERY_SUCCEEDED, WorkshopEventType.DELIVERY_FAILED}
            ]
            assert len(events) == 1
            event = events[0]
            assert event.envelope.event_type == WorkshopEventType.DELIVERY_SUCCEEDED
            assert event.envelope.event_version == 2
            assert event.envelope.aggregate_id == requested.delivery.delivery_id
            assert event.envelope.payload == {
                "message_id": message_id,
                "channel_id": requested.delivery.channel_id,
                "channel_binding_id": binding_id,
                "transport": "telegram",
                "mode": "text",
                "attempt_number": 1,
            }
            assert event.envelope.metadata == {"source": "delivery_outbox"}

            await store.project_pending(CanonicalConversationProjection())
            async with store.connection.execute(
                "SELECT id, channel_binding_id, status FROM deliveries WHERE id = ?",
                (requested.delivery.delivery_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert tuple(row) == (requested.delivery.delivery_id, binding_id, "succeeded")
        finally:
            await store.close()

    async def test_retry_scheduling_has_no_terminal_fact_but_exhaustion_does(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        clock = _Clock()
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        try:
            requested = await outbox.request_delivery(_request(message_id, binding_id, max_attempts=2))
            first = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert first is not None
            retry = await outbox.mark_failed(first, retryable=True, error_code="network_timeout")
            assert retry.status == "retry_wait"
            assert not [
                event
                for event in await store.read_events()
                if event.envelope.event_type
                in {WorkshopEventType.DELIVERY_SUCCEEDED, WorkshopEventType.DELIVERY_FAILED}
            ]

            clock.advance(timedelta(seconds=5))
            second = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert second is not None
            failed = await outbox.mark_failed(second, retryable=True, error_code="network_timeout")
            assert failed.status == "failed"

            events = [
                event
                for event in await store.read_events()
                if event.envelope.event_type
                in {WorkshopEventType.DELIVERY_SUCCEEDED, WorkshopEventType.DELIVERY_FAILED}
            ]
            assert len(events) == 1
            event = events[0]
            assert event.envelope.event_type == WorkshopEventType.DELIVERY_FAILED
            assert event.envelope.event_version == 2
            assert event.envelope.aggregate_id == requested.delivery.delivery_id
            assert event.envelope.payload["attempt_number"] == 2
            assert event.envelope.payload["channel_binding_id"] == binding_id
            assert event.envelope.payload["error_code"] == "network_timeout"
        finally:
            await store.close()

    async def test_failed_event_append_rolls_terminal_state_and_attempt_back(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        outbox = WorkshopDeliveryOutbox(store, clock=_Clock())
        try:
            requested = await outbox.request_delivery(_request(message_id, binding_id))
            claim = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert claim is not None
            await store.connection.execute(
                "CREATE TRIGGER reject_delivery_success BEFORE INSERT ON event_log "
                "WHEN NEW.event_type = 'delivery.succeeded' "
                "BEGIN SELECT RAISE(ABORT, 'test outcome rejection'); END"
            )
            await store.connection.commit()

            with pytest.raises(aiosqlite.IntegrityError, match="test outcome rejection"):
                await outbox.mark_succeeded(claim)

            assert (await outbox.state(requested.delivery.delivery_id)).status == "leased"
            async with store.connection.execute(
                "SELECT completed_at, outcome FROM delivery_attempts WHERE id = ?",
                (claim.attempt_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert tuple(row) == (None, None)
        finally:
            await store.close()

    async def test_same_message_projects_one_outcome_per_registered_binding(self, tmp_path: Path):
        store, message_id, direct_binding_id = await _open_with_outbound(tmp_path / "kai.db")
        outbox = WorkshopDeliveryOutbox(store, clock=_Clock())
        try:
            group_binding_id = await _add_group_binding(store, message_id)
            direct = await outbox.request_delivery(_request(message_id, direct_binding_id))
            group = await outbox.request_delivery(_request(message_id, group_binding_id))

            first = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert first is not None
            await outbox.mark_succeeded(first)
            second = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert second is not None
            await outbox.mark_succeeded(second)
            await store.project_pending(CanonicalConversationProjection())

            async with store.connection.execute(
                "SELECT id, channel_binding_id FROM deliveries WHERE message_id = ? ORDER BY id",
                (message_id,),
            ) as cursor:
                rows = await cursor.fetchall()
            assert sorted((row[0], row[1]) for row in rows) == sorted(
                (
                    (direct.delivery.delivery_id, direct_binding_id),
                    (group.delivery.delivery_id, group_binding_id),
                )
            )
        finally:
            await store.close()

    async def test_legacy_shadow_observation_and_binding_aware_outcome_replay_together(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        outbox = WorkshopDeliveryOutbox(store, clock=_Clock())
        try:
            legacy = await record_delivery_observation(
                store,
                DeliveryObservation(
                    message_id=message_id,
                    transport="telegram",
                    mode="text",
                    succeeded=True,
                    occurred_at=_NOW,
                ),
            )
            requested = await outbox.request_delivery(_request(message_id, binding_id))
            claim = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert claim is not None
            await outbox.mark_succeeded(claim)

            await store.rebuild_projection(CanonicalConversationProjection())
            async with store.connection.execute(
                "SELECT id, channel_binding_id FROM deliveries WHERE message_id = ? ORDER BY id",
                (message_id,),
            ) as cursor:
                rows = await cursor.fetchall()
            assert sorted((row[0], row[1]) for row in rows) == sorted(
                (
                    (legacy.event.envelope.aggregate_id, None),
                    (requested.delivery.delivery_id, binding_id),
                )
            )
        finally:
            await store.close()

    async def test_projection_rejects_binding_aware_fact_with_mismatched_channel(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        try:
            async with store.connection.execute(
                "SELECT c.workshop_id FROM messages m JOIN channels c ON c.id = m.channel_id WHERE m.id = ?",
                (message_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            workshop_id = WorkshopId(str(row[0]))
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.DELIVERY_SUCCEEDED,
                    event_version=2,
                    workshop_id=workshop_id,
                    aggregate_type="delivery",
                    aggregate_id=DeliveryId.new(),
                    occurred_at=_NOW,
                    payload={
                        "message_id": message_id,
                        "channel_id": ChannelId.new(),
                        "channel_binding_id": binding_id,
                        "transport": "telegram",
                        "mode": "text",
                        "attempt_number": 1,
                    },
                )
            )

            with pytest.raises(ValueError, match="message and binding must belong to the same channel"):
                await store.project_pending(CanonicalConversationProjection())
        finally:
            await store.close()


class TestDeliveryOrdering:
    async def test_purpose_lanes_do_not_claim_or_block_each_other(self, tmp_path: Path):
        store, first_message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        second_message_id = await _add_outbound(store, 1)
        outbox = WorkshopDeliveryOutbox(store, clock=_Clock())
        try:
            qualification = await outbox.request_delivery(_request(first_message_id, binding_id))
            conversation = await outbox.request_delivery(
                _request(second_message_id, binding_id, purpose=CONVERSATION_REPLY_PURPOSE)
            )

            conversation_claim = await outbox.claim_next(
                "conversation-worker",
                purposes=(CONVERSATION_REPLY_PURPOSE,),
            )
            assert conversation_claim is not None
            assert conversation_claim.delivery_id == conversation.delivery.delivery_id
            assert conversation_claim.purpose == CONVERSATION_REPLY_PURPOSE
            assert (await outbox.state(qualification.delivery.delivery_id)).status == "pending"

            qualification_claim = await outbox.claim_next(
                "qualification-worker",
                purposes=(QUALIFICATION_PURPOSE,),
            )
            assert qualification_claim is not None
            assert qualification_claim.delivery_id == qualification.delivery.delivery_id
            assert qualification_claim.purpose == QUALIFICATION_PURPOSE
        finally:
            await store.close()

    async def test_concurrent_workers_cannot_overtake_on_one_binding(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, first_message_id, binding_id = await _open_with_outbound(path)
        second_message_id = await _add_outbound(store, 1)
        outbox = WorkshopDeliveryOutbox(store, clock=_Clock())
        first_request = await outbox.request_delivery(_request(first_message_id, binding_id))
        second_request = await outbox.request_delivery(_request(second_message_id, binding_id))
        second_store = await WorkshopEventStore.open(path)
        try:
            first_claim, competing_claim = await asyncio.gather(
                outbox.claim_next("worker-1", purposes=(QUALIFICATION_PURPOSE,)),
                WorkshopDeliveryOutbox(second_store, clock=_Clock()).claim_next(
                    "worker-2", purposes=(QUALIFICATION_PURPOSE,)
                ),
            )
            claims = [claim for claim in (first_claim, competing_claim) if claim is not None]
            assert len(claims) == 1
            assert claims[0].delivery_id == first_request.delivery.delivery_id

            await outbox.mark_succeeded(claims[0])
            next_claim = await WorkshopDeliveryOutbox(second_store, clock=_Clock()).claim_next(
                "worker-2", purposes=(QUALIFICATION_PURPOSE,)
            )
            assert next_claim is not None
            assert next_claim.delivery_id == second_request.delivery.delivery_id
        finally:
            await second_store.close()
            await store.close()

    async def test_retry_wait_blocks_later_delivery_until_predecessor_is_terminal(self, tmp_path: Path):
        store, first_message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        second_message_id = await _add_outbound(store, 1)
        clock = _Clock()
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        try:
            first_request = await outbox.request_delivery(_request(first_message_id, binding_id))
            second_request = await outbox.request_delivery(_request(second_message_id, binding_id))
            first_claim = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert first_claim is not None
            await outbox.mark_failed(first_claim, retryable=True, error_code="network_timeout")

            assert await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,)) is None
            clock.advance(timedelta(seconds=5))
            retry_claim = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert retry_claim is not None
            assert retry_claim.delivery_id == first_request.delivery.delivery_id
            await outbox.mark_failed(retry_claim, retryable=False, error_code="provider_rejected")

            next_claim = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert next_claim is not None
            assert next_claim.delivery_id == second_request.delivery.delivery_id
        finally:
            await store.close()

    async def test_mode_filter_cannot_bypass_earlier_work_on_same_binding(self, tmp_path: Path):
        store, first_message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        second_message_id = await _add_outbound(store, 1)
        outbox = WorkshopDeliveryOutbox(store, clock=_Clock())
        try:
            first_request = await outbox.request_delivery(_request(first_message_id, binding_id, mode="text"))
            second_request = await outbox.request_delivery(_request(second_message_id, binding_id, mode="photo"))

            assert await outbox.claim_next("photo-worker", purposes=(QUALIFICATION_PURPOSE,), modes=("photo",)) is None
            first_claim = await outbox.claim_next("text-worker", purposes=(QUALIFICATION_PURPOSE,), modes=("text",))
            assert first_claim is not None
            assert first_claim.delivery_id == first_request.delivery.delivery_id
            await outbox.mark_succeeded(first_claim)
            second_claim = await outbox.claim_next("photo-worker", purposes=(QUALIFICATION_PURPOSE,), modes=("photo",))
            assert second_claim is not None
            assert second_claim.delivery_id == second_request.delivery.delivery_id
        finally:
            await store.close()

    async def test_different_bindings_progress_independently(self, tmp_path: Path):
        store, message_id, direct_binding_id = await _open_with_outbound(tmp_path / "kai.db")
        outbox = WorkshopDeliveryOutbox(store, clock=_Clock())
        try:
            group_binding_id = await _add_group_binding(store, message_id)
            direct = await outbox.request_delivery(_request(message_id, direct_binding_id))
            group = await outbox.request_delivery(_request(message_id, group_binding_id))

            direct_claim = await outbox.claim_next("worker-1", purposes=(QUALIFICATION_PURPOSE,))
            assert direct_claim is not None
            assert direct_claim.delivery_id == direct.delivery.delivery_id
            group_claim = await outbox.claim_next("worker-2", purposes=(QUALIFICATION_PURPOSE,))
            assert group_claim is not None
            assert group_claim.delivery_id == group.delivery.delivery_id
        finally:
            await store.close()


class TestDeliveryClaims:
    async def test_recovery_is_scoped_to_worker_purpose(self, tmp_path: Path):
        store, message_id, private_binding = await _open_with_outbound(tmp_path / "kai.db")
        group_binding = await _add_group_binding(store, message_id)
        clock = _Clock()
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        qualification = await outbox.request_delivery(_request(message_id, private_binding))
        conversation = await outbox.request_delivery(
            _request(message_id, group_binding, purpose=CONVERSATION_REPLY_PURPOSE)
        )
        try:
            assert (
                await outbox.claim_next(
                    "qualification-worker",
                    purposes=(QUALIFICATION_PURPOSE,),
                    delivery_id=qualification.delivery.delivery_id,
                    lease_duration=timedelta(seconds=5),
                )
                is not None
            )
            assert (
                await outbox.claim_next(
                    "conversation-worker",
                    purposes=(CONVERSATION_REPLY_PURPOSE,),
                    delivery_id=conversation.delivery.delivery_id,
                    lease_duration=timedelta(seconds=5),
                )
                is not None
            )
            clock.advance(timedelta(seconds=5))

            recovered = await outbox.recover_expired_leases(
                purposes=(CONVERSATION_REPLY_PURPOSE,),
            )

            assert recovered.requeued == 1
            assert (await outbox.state(conversation.delivery.delivery_id)).status == "retry_wait"
            assert (await outbox.state(qualification.delivery.delivery_id)).status == "leased"
        finally:
            await store.close()

    async def test_exact_claim_never_drains_other_due_work(self, tmp_path: Path):
        store, message_id, private_binding = await _open_with_outbound(tmp_path / "kai.db")
        group_binding = await _add_group_binding(store, message_id)
        private = await WorkshopDeliveryOutbox(store).request_delivery(_request(message_id, private_binding))
        group = await WorkshopDeliveryOutbox(store).request_delivery(_request(message_id, group_binding))
        try:
            claim = await WorkshopDeliveryOutbox(store).claim_next(
                "qualification-worker",
                purposes=(QUALIFICATION_PURPOSE,),
                delivery_id=group.delivery.delivery_id,
                transport="telegram",
                modes=("text",),
            )

            assert claim is not None
            assert claim.delivery_id == group.delivery.delivery_id
            assert (await WorkshopDeliveryOutbox(store).state(private.delivery.delivery_id)).status == "pending"
        finally:
            await store.close()

    async def test_exact_claim_recovers_only_selected_expired_lease(self, tmp_path: Path):
        store, message_id, private_binding = await _open_with_outbound(tmp_path / "kai.db")
        group_binding = await _add_group_binding(store, message_id)
        clock = _Clock()
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        private = await outbox.request_delivery(_request(message_id, private_binding))
        group = await outbox.request_delivery(_request(message_id, group_binding))
        try:
            private_claim = await outbox.claim_next(
                "private-worker",
                purposes=(QUALIFICATION_PURPOSE,),
                delivery_id=private.delivery.delivery_id,
                lease_duration=timedelta(seconds=5),
            )
            group_claim = await outbox.claim_next(
                "group-worker",
                purposes=(QUALIFICATION_PURPOSE,),
                delivery_id=group.delivery.delivery_id,
                lease_duration=timedelta(seconds=5),
            )
            assert private_claim is not None
            assert group_claim is not None
            clock.advance(timedelta(seconds=5))

            recovered_group = await outbox.claim_next(
                "qualification-worker",
                purposes=(QUALIFICATION_PURPOSE,),
                delivery_id=group.delivery.delivery_id,
            )

            assert recovered_group is not None
            assert recovered_group.attempt_number == 2
            assert (await outbox.state(private.delivery.delivery_id)).status == "leased"
        finally:
            await store.close()

    async def test_exact_claim_respects_same_binding_predecessor(self, tmp_path: Path):
        store, first_message, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        second_message = await _add_outbound(store, 1)
        first = await WorkshopDeliveryOutbox(store).request_delivery(_request(first_message, binding_id))
        second = await WorkshopDeliveryOutbox(store).request_delivery(_request(second_message, binding_id))
        try:
            claim = await WorkshopDeliveryOutbox(store).claim_next(
                "qualification-worker",
                purposes=(QUALIFICATION_PURPOSE,),
                delivery_id=second.delivery.delivery_id,
            )

            assert claim is None
            assert (await WorkshopDeliveryOutbox(store).state(first.delivery.delivery_id)).status == "pending"
            assert (await WorkshopDeliveryOutbox(store).state(second.delivery.delivery_id)).status == "pending"
        finally:
            await store.close()

    async def test_claim_is_exclusive_and_contains_only_registered_delivery_target(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, message_id, binding_id = await _open_with_outbound(path)
        clock = _Clock()
        requested = await WorkshopDeliveryOutbox(store, clock=clock).request_delivery(_request(message_id, binding_id))
        second_store = await WorkshopEventStore.open(path)
        try:
            first, second = await asyncio.gather(
                WorkshopDeliveryOutbox(store, clock=clock).claim_next("worker-1", purposes=(QUALIFICATION_PURPOSE,)),
                WorkshopDeliveryOutbox(second_store, clock=clock).claim_next(
                    "worker-2", purposes=(QUALIFICATION_PURPOSE,)
                ),
            )
            claims = [claim for claim in (first, second) if claim is not None]
            assert len(claims) == 1
            claim = claims[0]
            assert claim.delivery_id == requested.delivery.delivery_id
            assert claim.message_id == message_id
            assert claim.channel_binding_id == binding_id
            assert claim.transport == "telegram"
            assert claim.external_channel_id == "101"
            assert claim.mode == "text"
            assert claim.body == "Hello back"
            assert claim.attempt_number == 1
        finally:
            await second_store.close()
            await store.close()

    async def test_retryable_failure_uses_bounded_policy_and_then_succeeds(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        clock = _Clock()
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        try:
            requested = await outbox.request_delivery(_request(message_id, binding_id))
            first = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert first is not None
            failed = await outbox.mark_failed(first, retryable=True, error_code="network_timeout")
            assert failed.status == "retry_wait"
            assert failed.available_at == _NOW + timedelta(seconds=5)
            assert await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,)) is None

            clock.advance(timedelta(seconds=5))
            second = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert second is not None
            assert second.delivery_id == requested.delivery.delivery_id
            assert second.attempt_number == 2
            succeeded = await outbox.mark_succeeded(second)
            assert succeeded.status == "succeeded"
            assert succeeded.attempt_count == 2
            assert succeeded.completed_at == clock.now
            assert await outbox.mark_succeeded(second) == succeeded
        finally:
            await store.close()

    async def test_permanent_or_exhausted_failure_becomes_terminal(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        clock = _Clock()
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        try:
            await outbox.request_delivery(_request(message_id, binding_id, max_attempts=1))
            claim = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert claim is not None
            failed = await outbox.mark_failed(claim, retryable=True, error_code="network_timeout")
            assert failed.status == "failed"
            assert failed.completed_at == _NOW
            assert await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,)) is None
        finally:
            await store.close()

    async def test_provider_retry_floor_cannot_be_shortened_by_default_policy(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        clock = _Clock()
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        try:
            await outbox.request_delivery(_request(message_id, binding_id))
            claim = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert claim is not None

            failed = await outbox.mark_failed(
                claim,
                retryable=True,
                error_code="telegram_rate_limited",
                minimum_retry_delay=timedelta(seconds=20),
            )

            assert failed.status == "retry_wait"
            assert failed.available_at == _NOW + timedelta(seconds=20)
        finally:
            await store.close()

    async def test_stale_claim_cannot_complete_reassigned_delivery(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        clock = _Clock()
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        try:
            await outbox.request_delivery(_request(message_id, binding_id))
            stale = await outbox.claim_next(
                "worker-1", purposes=(QUALIFICATION_PURPOSE,), lease_duration=timedelta(seconds=10)
            )
            assert stale is not None
            clock.advance(timedelta(seconds=10))
            current = await outbox.claim_next("worker-2", purposes=(QUALIFICATION_PURPOSE,))
            assert current is not None
            assert current.attempt_number == 2
            with pytest.raises(StaleDeliveryLeaseError):
                await outbox.mark_succeeded(stale)
            assert (await outbox.state(current.delivery_id)).status == "leased"
        finally:
            await store.close()

    async def test_completion_after_lease_expiry_fails_and_recovers_work(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        clock = _Clock()
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        try:
            await outbox.request_delivery(_request(message_id, binding_id))
            expired = await outbox.claim_next(
                "worker-1", purposes=(QUALIFICATION_PURPOSE,), lease_duration=timedelta(seconds=10)
            )
            assert expired is not None
            clock.advance(timedelta(seconds=10))

            with pytest.raises(StaleDeliveryLeaseError, match="expired"):
                await outbox.mark_succeeded(expired)

            assert (await outbox.state(expired.delivery_id)).status == "retry_wait"
        finally:
            await store.close()


class TestDeliveryRecovery:
    async def test_restart_recovery_requeues_expired_lease_and_records_attempt_outcome(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, message_id, binding_id = await _open_with_outbound(path)
        clock = _Clock()
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        requested = await outbox.request_delivery(_request(message_id, binding_id))
        claim = await outbox.claim_next(
            "worker-before-restart", purposes=(QUALIFICATION_PURPOSE,), lease_duration=timedelta(seconds=10)
        )
        assert claim is not None
        await store.close()

        clock.advance(timedelta(seconds=10))
        reopened = await WorkshopEventStore.open(path)
        try:
            recovered = await WorkshopDeliveryOutbox(reopened, clock=clock).recover_expired_leases(
                purposes=(QUALIFICATION_PURPOSE,)
            )
            assert recovered.requeued == 1
            assert recovered.failed == 0
            state = await WorkshopDeliveryOutbox(reopened, clock=clock).state(requested.delivery.delivery_id)
            assert state.status == "retry_wait"
            assert state.last_error_code == "lease_expired"
            async with reopened.connection.execute(
                "SELECT outcome, error_code FROM delivery_attempts WHERE id = ?",
                (claim.attempt_id,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == ("lease_expired", "lease_expired")
        finally:
            await reopened.close()

    async def test_restart_recovery_appends_failure_fact_when_expired_lease_is_exhausted(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, message_id, binding_id = await _open_with_outbound(path)
        clock = _Clock()
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        requested = await outbox.request_delivery(_request(message_id, binding_id, max_attempts=1))
        claim = await outbox.claim_next(
            "worker-before-restart", purposes=(QUALIFICATION_PURPOSE,), lease_duration=timedelta(seconds=10)
        )
        assert claim is not None
        await store.close()

        clock.advance(timedelta(seconds=10))
        reopened = await WorkshopEventStore.open(path)
        try:
            recovered = await WorkshopDeliveryOutbox(reopened, clock=clock).recover_expired_leases(
                purposes=(QUALIFICATION_PURPOSE,)
            )
            assert recovered.requeued == 0
            assert recovered.failed == 1
            assert (await WorkshopDeliveryOutbox(reopened).state(requested.delivery.delivery_id)).status == "failed"
            events = [
                event
                for event in await reopened.read_events()
                if event.envelope.event_type == WorkshopEventType.DELIVERY_FAILED
            ]
            assert len(events) == 1
            assert events[0].envelope.event_version == 2
            assert events[0].envelope.aggregate_id == requested.delivery.delivery_id
            assert events[0].envelope.payload["channel_binding_id"] == binding_id
            assert events[0].envelope.payload["error_code"] == "lease_expired"
        finally:
            await reopened.close()

    async def test_projection_rebuild_preserves_outbox_and_attempt_state(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        clock = _Clock()
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        try:
            requested = await outbox.request_delivery(_request(message_id, binding_id))
            claim = await outbox.claim_next("telegram-worker", purposes=(QUALIFICATION_PURPOSE,))
            assert claim is not None

            await store.rebuild_projection(CanonicalConversationProjection())

            state = await outbox.state(requested.delivery.delivery_id)
            assert state.status == "leased"
            assert state.attempt_count == 1
            async with store.connection.execute(
                "SELECT attempt_number, outcome FROM delivery_attempts WHERE id = ?",
                (claim.attempt_id,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == (1, None)
        finally:
            await store.close()
