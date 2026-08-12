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
    DeliveryRequest,
    DeliveryRequestConflictError,
    DeliveryTargetNotFoundError,
    StaleDeliveryLeaseError,
    WorkshopDeliveryOutbox,
)
from kai.workshop.domain import (
    ChannelBindingId,
    EventEnvelope,
    MessageId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import OutboundMessage, record_outbound_message
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


def _request(
    message_id: MessageId,
    binding_id: ChannelBindingId,
    *,
    max_attempts: int = 5,
) -> DeliveryRequest:
    return DeliveryRequest(
        message_id=message_id,
        channel_binding_id=binding_id,
        mode="text",
        occurred_at=_NOW,
        max_attempts=max_attempts,
    )


class TestDeliveryRequests:
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
            assert events[0].envelope.payload == {
                "message_id": message_id,
                "channel_id": result.delivery.channel_id,
                "channel_binding_id": binding_id,
                "transport": "telegram",
                "mode": "text",
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

    async def test_same_message_can_target_multiple_registered_telegram_bindings(self, tmp_path: Path):
        store, message_id, direct_binding_id = await _open_with_outbound(tmp_path / "kai.db")
        try:
            async with store.connection.execute(
                "SELECT c.workshop_id, m.channel_id FROM messages m "
                "JOIN channels c ON c.id = m.channel_id WHERE m.id = ?",
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
                    idempotency_key="test:notification-group-binding",
                    payload={
                        "channel_id": channel_id,
                        "transport": "telegram",
                        "external_channel_id": "-100123",
                    },
                )
            )
            await store.project_pending(CanonicalConversationProjection())
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


class TestDeliveryClaims:
    async def test_claim_is_exclusive_and_contains_only_registered_delivery_target(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, message_id, binding_id = await _open_with_outbound(path)
        clock = _Clock()
        requested = await WorkshopDeliveryOutbox(store, clock=clock).request_delivery(_request(message_id, binding_id))
        second_store = await WorkshopEventStore.open(path)
        try:
            first, second = await asyncio.gather(
                WorkshopDeliveryOutbox(store, clock=clock).claim_next("worker-1"),
                WorkshopDeliveryOutbox(second_store, clock=clock).claim_next("worker-2"),
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
            first = await outbox.claim_next("telegram-worker")
            assert first is not None
            failed = await outbox.mark_failed(first, retryable=True, error_code="network_timeout")
            assert failed.status == "retry_wait"
            assert failed.available_at == _NOW + timedelta(seconds=5)
            assert await outbox.claim_next("telegram-worker") is None

            clock.advance(timedelta(seconds=5))
            second = await outbox.claim_next("telegram-worker")
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
            claim = await outbox.claim_next("telegram-worker")
            assert claim is not None
            failed = await outbox.mark_failed(claim, retryable=True, error_code="network_timeout")
            assert failed.status == "failed"
            assert failed.completed_at == _NOW
            assert await outbox.claim_next("telegram-worker") is None
        finally:
            await store.close()

    async def test_provider_retry_floor_cannot_be_shortened_by_default_policy(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        clock = _Clock()
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        try:
            await outbox.request_delivery(_request(message_id, binding_id))
            claim = await outbox.claim_next("telegram-worker")
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
            stale = await outbox.claim_next("worker-1", lease_duration=timedelta(seconds=10))
            assert stale is not None
            clock.advance(timedelta(seconds=10))
            current = await outbox.claim_next("worker-2")
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
            expired = await outbox.claim_next("worker-1", lease_duration=timedelta(seconds=10))
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
        claim = await outbox.claim_next("worker-before-restart", lease_duration=timedelta(seconds=10))
        assert claim is not None
        await store.close()

        clock.advance(timedelta(seconds=10))
        reopened = await WorkshopEventStore.open(path)
        try:
            recovered = await WorkshopDeliveryOutbox(reopened, clock=clock).recover_expired_leases()
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

    async def test_projection_rebuild_preserves_outbox_and_attempt_state(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        clock = _Clock()
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        try:
            requested = await outbox.request_delivery(_request(message_id, binding_id))
            claim = await outbox.claim_next("telegram-worker")
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
