"""Atomic Workshop streaming-finalization contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.delivery_fragments import EDIT_OPERATION, SEND_OPERATION, WorkshopDeliveryFragments
from kai.workshop.delivery_outbox import (
    CONVERSATION_REPLY_PURPOSE,
    SEND_FRAGMENTS_CONTRACT,
    STREAMING_FINALIZATION_CONTRACT,
    DeliveryRequest,
    WorkshopDeliveryOutbox,
)
from kai.workshop.domain import (
    ChannelBindingId,
    DeliveryId,
    EventEnvelope,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import (
    OutboundDeliveryStateConflictError,
    OutboundMessage,
    record_outbound_message,
)
from kai.workshop.outbound import (
    record_outbound_message_with_streaming_finalization as _record_outbound_message_with_streaming_finalization,
)
from kai.workshop.store import WorkshopEventStore
from kai.workshop.streaming_preview import (
    ConfirmedTelegramStreamingPreview,
    bind_confirmed_telegram_streaming_preview,
)
from kai.workshop.telegram_delivery import (
    TelegramWorkOutcome,
    WorkshopTelegramDeliveryAdapter,
    WorkshopTelegramDeliveryWorker,
    WorkshopTelegramFinalizationPlanner,
)
from tests.workshop_delivery import DISABLED_DELIVERY_POLICY, TELEGRAM_DELIVERY_POLICY

_NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


async def record_outbound_message_with_streaming_finalization(store, message):
    return await _record_outbound_message_with_streaming_finalization(
        store,
        message,
        delivery_policy=TELEGRAM_DELIVERY_POLICY,
    )


@dataclass
class _Clock:
    now: datetime = _NOW

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


async def _open_with_inbound(
    path: Path,
    *,
    activate_authority: bool = True,
) -> tuple[WorkshopEventStore, MessageId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Authorized human", "admin", "telegram", "101", "101"),),
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
    if activate_authority:
        await WorkshopConversationDeliveryAuthority(store).activate()
    return store, inbound.event.envelope.aggregate_id


async def _bind_preview(store: WorkshopEventStore, inbound_id: MessageId, message_id: int = 7001) -> None:
    await bind_confirmed_telegram_streaming_preview(
        store,
        ConfirmedTelegramStreamingPreview(
            inbound_message_id=inbound_id,
            external_message_id=message_id,
            confirmed_at=_NOW + timedelta(seconds=1),
        ),
    )


def _outbound(inbound_id: MessageId, body: str = "Final answer") -> OutboundMessage:
    return OutboundMessage(
        in_reply_to_message_id=inbound_id,
        body=body,
        occurred_at=_NOW + timedelta(seconds=2),
    )


async def _assert_no_finalization_state(store: WorkshopEventStore, inbound_id: MessageId) -> None:
    async with store.connection.execute(
        "SELECT COUNT(*) FROM messages WHERE reply_to_message_id = ?", (inbound_id,)
    ) as cursor:
        assert (await cursor.fetchone())[0] == 0


async def _claim_finalization(store: WorkshopEventStore, delivery_id: DeliveryId):
    epoch = await WorkshopConversationDeliveryAuthority(store).active_epoch()
    claim = await WorkshopDeliveryOutbox(
        store,
        clock=lambda: _NOW + timedelta(seconds=3),
    ).claim_next(
        "telegram-adapter-planner",
        purposes=(CONVERSATION_REPLY_PURPOSE,),
        execution_contracts=(STREAMING_FINALIZATION_CONTRACT,),
        delivery_id=delivery_id,
        authority_epoch_id=epoch.epoch_id,
    )
    assert claim is not None
    return claim
    async with store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
        assert (await cursor.fetchone())[0] == 0
    async with store.connection.execute("SELECT COUNT(*) FROM delivery_fragments") as cursor:
        assert (await cursor.fetchone())[0] == 0


class TestAtomicStreamingFinalization:
    async def test_disabled_telegram_adapter_records_reply_without_delivery_for_retained_binding(
        self,
        tmp_path: Path,
    ):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            result = await _record_outbound_message_with_streaming_finalization(
                store,
                _outbound(inbound_id),
                delivery_policy=DISABLED_DELIVERY_POLICY,
            )

            assert result.message.inserted is True
            assert result.delivery is None
            assert result.plan is None
            async with store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
                assert int((await cursor.fetchone())[0]) == 0
            async with store.connection.execute(
                "SELECT COUNT(*) FROM channel_bindings WHERE transport = 'telegram'"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 1
        finally:
            await store.close()

    async def test_short_reply_defers_preview_edit_planning_to_adapter_claim(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            await _bind_preview(store, inbound_id)

            result = await record_outbound_message_with_streaming_finalization(
                store,
                _outbound(inbound_id),
            )

            assert result.message.inserted is True
            assert result.delivery.inserted is True
            assert result.plan is None
            assert result.delivery.delivery.purpose == CONVERSATION_REPLY_PURPOSE
            assert result.delivery.delivery.execution_contract == STREAMING_FINALIZATION_CONTRACT
            assert result.delivery.delivery.status == "pending"
            assert result.delivery.delivery.attempt_count == 0
            assert await WorkshopDeliveryFragments(store).fragments(result.delivery.delivery.delivery_id) == ()
            claim = await _claim_finalization(store, result.delivery.delivery.delivery_id)
            operations = await WorkshopTelegramFinalizationPlanner(store).operations(claim)
            assert len(operations) == 1
            operation = operations[0]
            assert operation.operation == EDIT_OPERATION
            assert operation.target_external_message_id == 7001
            assert operation.body == "Final answer"
        finally:
            await store.close()

    async def test_terminal_text_always_has_a_distinct_edit_even_if_preview_was_final_snapshot(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            await _bind_preview(store, inbound_id, message_id=7002)

            result = await record_outbound_message_with_streaming_finalization(
                store,
                _outbound(inbound_id, body="Already published final snapshot"),
            )

            claim = await _claim_finalization(store, result.delivery.delivery.delivery_id)
            operations = await WorkshopTelegramFinalizationPlanner(store).operations(claim)
            assert [(item.operation, item.target_external_message_id, item.body) for item in operations] == [
                (EDIT_OPERATION, 7002, "Already published final snapshot")
            ]
        finally:
            await store.close()

    async def test_long_reply_edits_preview_then_sends_continuations(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            await _bind_preview(store, inbound_id)
            body = "A" * 4096 + "\n" + "B" * 20

            result = await record_outbound_message_with_streaming_finalization(store, _outbound(inbound_id, body))

            claim = await _claim_finalization(store, result.delivery.delivery.delivery_id)
            operations = await WorkshopTelegramFinalizationPlanner(store).operations(claim)
            assert [item.operation for item in operations] == [EDIT_OPERATION, SEND_OPERATION]
            assert [item.target_external_message_id for item in operations] == [7001, None]
            assert "".join(item.body for item in operations) == "A" * 4096 + "B" * 20
        finally:
            await store.close()

    async def test_reply_without_preview_is_an_all_send_plan(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            body = "A" * 4096 + "\n" + "B" * 20

            result = await record_outbound_message_with_streaming_finalization(store, _outbound(inbound_id, body))

            claim = await _claim_finalization(store, result.delivery.delivery.delivery_id)
            operations = await WorkshopTelegramFinalizationPlanner(store).operations(claim)
            assert [item.operation for item in operations] == [SEND_OPERATION, SEND_OPERATION]
            assert all(item.target_external_message_id is None for item in operations)
        finally:
            await store.close()

    async def test_retry_is_idempotent_across_restart_and_observation_time(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, inbound_id = await _open_with_inbound(path)
        await _bind_preview(store, inbound_id)
        first = await record_outbound_message_with_streaming_finalization(store, _outbound(inbound_id))
        await store.close()

        reopened = await WorkshopEventStore.open(path)
        try:
            retry = await record_outbound_message_with_streaming_finalization(
                reopened,
                OutboundMessage(inbound_id, "Final answer", _NOW + timedelta(minutes=5)),
            )
            assert retry.message.inserted is False
            assert retry.delivery.inserted is False
            assert retry.plan is None
            assert retry.message.event == first.message.event
            assert retry.delivery.delivery == first.delivery.delivery
            assert await WorkshopDeliveryFragments(reopened).fragments(retry.delivery.delivery.delivery_id) == ()
        finally:
            await reopened.close()

    async def test_preview_bound_before_adapter_claim_is_used_by_adapter_plan(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            first = await record_outbound_message_with_streaming_finalization(store, _outbound(inbound_id))
            await _bind_preview(store, inbound_id)

            retry = await record_outbound_message_with_streaming_finalization(store, _outbound(inbound_id))
            claim = await _claim_finalization(store, first.delivery.delivery.delivery_id)
            operations = await WorkshopTelegramFinalizationPlanner(store).operations(claim)
            assert retry.delivery.delivery == first.delivery.delivery
            assert [item.operation for item in operations] == [EDIT_OPERATION]
        finally:
            await store.close()

    async def test_tampered_preview_routing_fails_at_telegram_adapter_boundary(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            await _bind_preview(store, inbound_id)
            await store.connection.execute(
                "UPDATE telegram_streaming_previews SET channel_id = ? WHERE inbound_message_id = ?",
                ("chn_00000000000000000000000000000001", inbound_id),
            )
            await store.connection.commit()

            result = await record_outbound_message_with_streaming_finalization(store, _outbound(inbound_id))
            claim = await _claim_finalization(store, result.delivery.delivery.delivery_id)
            with pytest.raises(Exception, match="telegram_preview_routing_invalid"):
                await WorkshopTelegramFinalizationPlanner(store).operations(claim)
        finally:
            await store.close()

    async def test_core_commit_does_not_touch_adapter_fragment_table(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            await store.connection.execute(
                "CREATE TRIGGER reject_finalization_plan BEFORE INSERT ON delivery_fragments "
                "BEGIN SELECT RAISE(ABORT, 'test plan rejection'); END"
            )
            await store.connection.commit()

            result = await record_outbound_message_with_streaming_finalization(store, _outbound(inbound_id))
            assert result.message.inserted is True
            assert result.delivery.inserted is True
            assert result.plan is None
        finally:
            await store.close()

    async def test_preexisting_message_without_delivery_or_plan_is_rejected_as_half_state(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            prior = await record_outbound_message(store, _outbound(inbound_id))

            with pytest.raises(OutboundDeliveryStateConflictError):
                await record_outbound_message_with_streaming_finalization(store, _outbound(inbound_id))

            async with store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
                assert (await cursor.fetchone())[0] == 0
            async with store.connection.execute("SELECT COUNT(*) FROM delivery_fragments") as cursor:
                assert (await cursor.fetchone())[0] == 0
            async with store.connection.execute(
                "SELECT body FROM messages WHERE id = ?", (prior.event.envelope.aggregate_id,)
            ) as cursor:
                assert (await cursor.fetchone())[0] == "Final answer"
        finally:
            await store.close()

    async def test_preexisting_message_and_delivery_without_adapter_plan_is_valid_core_state(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            prior = await record_outbound_message(store, _outbound(inbound_id))
            message_id = MessageId(str(prior.event.envelope.aggregate_id))
            async with store.connection.execute(
                "SELECT cb.id FROM channel_bindings cb JOIN messages m ON m.channel_id = cb.channel_id "
                "WHERE m.id = ? AND cb.transport = 'telegram'",
                (message_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            delivery = await WorkshopDeliveryOutbox(store).request_delivery(
                DeliveryRequest(
                    message_id=message_id,
                    channel_binding_id=ChannelBindingId(str(row[0])),
                    mode="text",
                    purpose=CONVERSATION_REPLY_PURPOSE,
                    occurred_at=_NOW + timedelta(seconds=2),
                    execution_contract=STREAMING_FINALIZATION_CONTRACT,
                    authority_epoch_id=(await WorkshopConversationDeliveryAuthority(store).active_epoch()).epoch_id,
                )
            )

            retry = await record_outbound_message_with_streaming_finalization(store, _outbound(inbound_id))
            assert retry.delivery.delivery == delivery.delivery
            assert retry.plan is None
            assert (await WorkshopDeliveryOutbox(store).state(delivery.delivery.delivery_id)).status == "pending"
            assert await WorkshopDeliveryFragments(store).fragments(delivery.delivery.delivery_id) == ()
        finally:
            await store.close()


class TestStreamingFinalizationWorkerIsolation:
    async def test_current_send_only_worker_cannot_claim_or_send_finalization_plan(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        bot = AsyncMock()
        bot.send_message.return_value = SimpleNamespace(message_id=8001)
        try:
            finalization = await record_outbound_message_with_streaming_finalization(
                store,
                _outbound(inbound_id),
            )
            worker = WorkshopTelegramDeliveryWorker(
                WorkshopDeliveryOutbox(store),
                WorkshopDeliveryFragments(store),
                WorkshopTelegramDeliveryAdapter(bot),
                worker_id="send-only-worker",
                purpose=CONVERSATION_REPLY_PURPOSE,
            )

            result = await worker.run_delivery(finalization.delivery.delivery.delivery_id)

            assert result.outcome == TelegramWorkOutcome.IDLE
            bot.send_message.assert_not_awaited()
            state = await WorkshopDeliveryOutbox(store).state(finalization.delivery.delivery.delivery_id)
            assert state.status == "pending"
            assert state.attempt_count == 0
        finally:
            await store.close()

    async def test_claim_and_recovery_are_execution_contract_scoped(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        clock = _Clock(_NOW + timedelta(seconds=2))
        try:
            finalization = await record_outbound_message_with_streaming_finalization(
                store,
                _outbound(inbound_id),
            )
            outbox = WorkshopDeliveryOutbox(store, clock=clock)
            authority_epoch = await WorkshopConversationDeliveryAuthority(store).active_epoch()
            assert (
                await outbox.claim_next(
                    "send-only-worker",
                    purposes=(CONVERSATION_REPLY_PURPOSE,),
                    execution_contracts=(SEND_FRAGMENTS_CONTRACT,),
                    delivery_id=finalization.delivery.delivery.delivery_id,
                )
                is None
            )
            claim = await outbox.claim_next(
                "future-finalization-worker",
                purposes=(CONVERSATION_REPLY_PURPOSE,),
                execution_contracts=(STREAMING_FINALIZATION_CONTRACT,),
                lease_duration=timedelta(seconds=10),
                delivery_id=finalization.delivery.delivery.delivery_id,
                authority_epoch_id=authority_epoch.epoch_id,
            )
            assert claim is not None
            clock.advance(timedelta(seconds=10))

            untouched = await outbox.recover_expired_leases(
                purposes=(CONVERSATION_REPLY_PURPOSE,),
                execution_contracts=(SEND_FRAGMENTS_CONTRACT,),
            )
            assert untouched.requeued == 0 and untouched.failed == 0
            assert (await outbox.state(claim.delivery_id)).status == "leased"

            recovered = await outbox.recover_expired_leases(
                purposes=(CONVERSATION_REPLY_PURPOSE,),
                execution_contracts=(STREAMING_FINALIZATION_CONTRACT,),
                authority_epoch_id=authority_epoch.epoch_id,
            )
            assert recovered.requeued == 1 and recovered.failed == 0
            assert (await outbox.state(claim.delivery_id)).status == "retry_wait"
        finally:
            await store.close()


class TestStreamingFinalizationMigration:
    async def test_version_twelve_work_defaults_to_send_fragments_and_send_operations(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from kai.workshop import schema

        path = tmp_path / "kai.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 12)
            migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:12])
            version_twelve, inbound_id = await _open_with_inbound(
                path,
                activate_authority=False,
            )
            outbound = await record_outbound_message(version_twelve, _outbound(inbound_id))
            message_id = MessageId(str(outbound.event.envelope.aggregate_id))
            async with version_twelve.connection.execute(
                "SELECT c.workshop_id, m.channel_id, cb.id, m.author_principal_id "
                "FROM messages m JOIN channels c ON c.id = m.channel_id "
                "JOIN channel_bindings cb ON cb.channel_id = m.channel_id "
                "WHERE m.id = ? AND cb.transport = 'telegram'",
                (message_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            workshop_id = WorkshopId(str(row[0]))
            delivery_id = DeliveryId.derived(workshop_id, f"delivery:{message_id}:{row[2]}:text")
            requested = await version_twelve.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.DELIVERY_REQUESTED,
                    event_version=2,
                    workshop_id=workshop_id,
                    aggregate_type="delivery",
                    aggregate_id=delivery_id,
                    actor_principal_id=PrincipalId(str(row[3])),
                    occurred_at=_NOW,
                    payload={
                        "message_id": message_id,
                        "channel_id": str(row[1]),
                        "channel_binding_id": str(row[2]),
                        "transport": "telegram",
                        "mode": "text",
                        "purpose": "qualification",
                        "max_attempts": 3,
                    },
                )
            )
            timestamp = "2026-08-12T12:00:00.000000Z"
            await version_twelve.connection.execute(
                "INSERT INTO delivery_outbox "
                "(id, workshop_id, channel_id, channel_binding_id, message_id, transport, mode, purpose, "
                "status, max_attempts, attempt_count, available_at, requested_event_position, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'telegram', 'text', 'qualification', "
                "'pending', 3, 0, ?, ?, ?, ?)",
                (
                    delivery_id,
                    str(row[0]),
                    str(row[1]),
                    str(row[2]),
                    message_id,
                    timestamp,
                    requested.event.position,
                    timestamp,
                    timestamp,
                ),
            )
            await version_twelve.connection.execute(
                "INSERT INTO delivery_fragments "
                "(delivery_id, fragment_index, fragment_count, body, status, created_at, updated_at) "
                "VALUES (?, 0, 1, 'existing', 'pending', ?, ?)",
                (delivery_id, timestamp, timestamp),
            )
            await version_twelve.connection.commit()
            await version_twelve.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 53
            async with upgraded.connection.execute(
                "SELECT execution_contract FROM delivery_outbox WHERE id = ?", (delivery_id,)
            ) as cursor:
                assert (await cursor.fetchone())[0] == SEND_FRAGMENTS_CONTRACT
            async with upgraded.connection.execute(
                "SELECT operation, target_external_message_id FROM delivery_fragments WHERE delivery_id = ?",
                (delivery_id,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == (SEND_OPERATION, None)
            async with upgraded.connection.execute(
                "SELECT authority_epoch_id FROM delivery_outbox WHERE id = ?", (delivery_id,)
            ) as cursor:
                assert (await cursor.fetchone())[0] is None
        finally:
            await upgraded.close()
