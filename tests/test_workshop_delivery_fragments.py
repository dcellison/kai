"""Durable fragment-progress contracts for the Workshop delivery outbox."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.delivery_fragments import DeliveryFragmentPlanConflictError, WorkshopDeliveryFragments
from kai.workshop.delivery_outbox import (
    DeliveryRequest,
    IncompleteDeliveryFragmentsError,
    UnsettledDeliveryFragmentError,
    WorkshopDeliveryOutbox,
)
from kai.workshop.domain import ChannelBindingId, MessageId
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import OutboundMessage, record_outbound_message
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


@dataclass
class _Clock:
    now: datetime = _NOW

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


async def _open_with_delivery(
    path: Path,
    clock: _Clock,
) -> tuple[WorkshopEventStore, WorkshopDeliveryOutbox, WorkshopDeliveryFragments]:
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
            body="A long Workshop response",
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
    binding_id = ChannelBindingId(str(row[0]))
    outbox = WorkshopDeliveryOutbox(store, clock=clock)
    fragments = WorkshopDeliveryFragments(store, clock=clock)
    await outbox.request_delivery(
        DeliveryRequest(
            message_id=message_id,
            channel_binding_id=binding_id,
            mode="text",
            occurred_at=_NOW,
        )
    )
    return store, outbox, fragments


class TestDeliveryFragmentPlan:
    async def test_plan_is_durable_idempotent_and_immutable(self, tmp_path: Path):
        clock = _Clock()
        store, outbox, fragments = await _open_with_delivery(tmp_path / "kai.db", clock)
        try:
            claim = await outbox.claim_next("worker-1")
            assert claim is not None

            first = await fragments.prepare(claim, ("first", "second"))
            repeated = await fragments.prepare(claim, ("first", "second"))

            assert repeated == first
            assert [fragment.status for fragment in first] == ["pending", "pending"]
            with pytest.raises(DeliveryFragmentPlanConflictError):
                await fragments.prepare(claim, ("different", "plan"))
        finally:
            await store.close()

    async def test_delivery_cannot_complete_before_every_fragment_is_sent(self, tmp_path: Path):
        clock = _Clock()
        store, outbox, fragments = await _open_with_delivery(tmp_path / "kai.db", clock)
        try:
            claim = await outbox.claim_next("worker-1")
            assert claim is not None
            await fragments.prepare(claim, ("first", "second"))
            first = await fragments.begin_next(claim)
            assert first is not None
            await fragments.mark_sent(claim, first, external_message_id=1001)

            with pytest.raises(IncompleteDeliveryFragmentsError):
                await outbox.mark_succeeded(claim)
            second = await fragments.begin_next(claim)
            assert second is not None and second.fragment_index == 1
            await fragments.mark_sent(claim, second, external_message_id=1002)

            assert (await outbox.mark_succeeded(claim)).status == "succeeded"
        finally:
            await store.close()

    async def test_outbox_cannot_retry_while_a_fragment_is_in_flight(self, tmp_path: Path):
        clock = _Clock()
        store, outbox, fragments = await _open_with_delivery(tmp_path / "kai.db", clock)
        try:
            claim = await outbox.claim_next("worker-1")
            assert claim is not None
            await fragments.prepare(claim, ("first", "second"))
            assert await fragments.begin_next(claim) is not None

            with pytest.raises(UnsettledDeliveryFragmentError):
                await outbox.mark_failed(claim, retryable=True, error_code="telegram_rate_limited")
            assert (await outbox.state(claim.delivery_id)).status == "leased"
        finally:
            await store.close()


class TestDeliveryFragmentRecovery:
    async def test_restart_resumes_after_confirmed_fragments_without_resending_them(self, tmp_path: Path):
        clock = _Clock()
        store, outbox, fragments = await _open_with_delivery(tmp_path / "kai.db", clock)
        try:
            first_claim = await outbox.claim_next("worker-1", lease_duration=timedelta(seconds=10))
            assert first_claim is not None
            await fragments.prepare(first_claim, ("first", "second"))
            first = await fragments.begin_next(first_claim)
            assert first is not None
            await fragments.mark_sent(first_claim, first, external_message_id=1001)

            clock.advance(timedelta(seconds=10))
            assert (await outbox.recover_expired_leases()).requeued == 1
            second_claim = await outbox.claim_next("worker-2")
            assert second_claim is not None and second_claim.attempt_number == 2
            await fragments.prepare(second_claim, ("first", "second"))
            resumed = await fragments.begin_next(second_claim)

            assert resumed is not None and resumed.fragment_index == 1
            await fragments.mark_sent(second_claim, resumed, external_message_id=1002)
            assert (await outbox.mark_succeeded(second_claim)).status == "succeeded"
            assert [
                fragment.external_message_id for fragment in await fragments.fragments(second_claim.delivery_id)
            ] == [
                "1001",
                "1002",
            ]
        finally:
            await store.close()

    async def test_expired_in_flight_fragment_fails_uncertain_instead_of_resending(self, tmp_path: Path):
        clock = _Clock()
        store, outbox, fragments = await _open_with_delivery(tmp_path / "kai.db", clock)
        try:
            claim = await outbox.claim_next("worker-1", lease_duration=timedelta(seconds=10))
            assert claim is not None
            await fragments.prepare(claim, ("first", "second"))
            assert await fragments.begin_next(claim) is not None

            clock.advance(timedelta(seconds=10))
            recovered = await outbox.recover_expired_leases()
            state = await outbox.state(claim.delivery_id)
            persisted = await fragments.fragments(claim.delivery_id)

            assert recovered.failed == 1
            assert recovered.requeued == 0
            assert state.status == "failed"
            assert state.last_error_code == "delivery_send_uncertain"
            assert persisted[0].status == "uncertain"
            assert await outbox.claim_next("worker-2") is None
        finally:
            await store.close()

    async def test_projection_rebuild_preserves_fragment_plan_and_progress(self, tmp_path: Path):
        clock = _Clock()
        store, outbox, fragments = await _open_with_delivery(tmp_path / "kai.db", clock)
        try:
            claim = await outbox.claim_next("worker-1")
            assert claim is not None
            await fragments.prepare(claim, ("first", "second"))
            first = await fragments.begin_next(claim)
            assert first is not None
            await fragments.mark_sent(claim, first, external_message_id=1001)

            await store.rebuild_projection(CanonicalConversationProjection())

            persisted = await fragments.fragments(claim.delivery_id)
            assert [fragment.status for fragment in persisted] == ["sent", "pending"]
            assert persisted[0].external_message_id == "1001"
        finally:
            await store.close()
