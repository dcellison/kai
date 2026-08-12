"""Contracts for production-unused Workshop conversation-delivery authority epochs."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import BadRequest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.delivery_authority import (
    DeliveryAuthorityHistoricalWorkError,
    DeliveryAuthorityInactiveError,
    DeliveryAuthorityOutstandingWorkError,
    DeliveryAuthorityUnreconciledFailureError,
    WorkshopConversationDeliveryAuthority,
)
from kai.workshop.delivery_fragments import WorkshopDeliveryFragments
from kai.workshop.delivery_outbox import (
    CONVERSATION_REPLY_PURPOSE,
    STREAMING_FINALIZATION_CONTRACT,
    WorkshopDeliveryOutbox,
)
from kai.workshop.diagnostics import workshop_delivery_authority_status
from kai.workshop.domain import MessageId
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import OutboundMessage, record_outbound_message_with_streaming_finalization
from kai.workshop.store import WorkshopEventStore
from kai.workshop.telegram_delivery import (
    TelegramWorkOutcome,
    WorkshopTelegramStreamingFinalizationAdapter,
    WorkshopTelegramStreamingFinalizationWorker,
)

_NOW = datetime(2026, 8, 12, 15, 0, tzinfo=UTC)


async def _open_with_inbound(path: Path, *, sequence: int = 1) -> tuple[WorkshopEventStore, MessageId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Authorized human", "admin", "telegram", "101", "101"),),
    )
    inbound = await record_inbound_message(
        store,
        InboundMessage(
            transport="telegram",
            update_id=str(9000 + sequence),
            message_id=str(40 + sequence),
            sender_subject="101",
            channel_subject="101",
            body=f"Hello {sequence}",
            occurred_at=_NOW + timedelta(minutes=sequence),
        ),
    )
    return store, MessageId(str(inbound.event.envelope.aggregate_id))


async def _add_inbound(store: WorkshopEventStore, *, sequence: int) -> MessageId:
    inbound = await record_inbound_message(
        store,
        InboundMessage(
            transport="telegram",
            update_id=str(9000 + sequence),
            message_id=str(40 + sequence),
            sender_subject="101",
            channel_subject="101",
            body=f"Hello {sequence}",
            occurred_at=_NOW + timedelta(minutes=sequence),
        ),
    )
    return MessageId(str(inbound.event.envelope.aggregate_id))


async def _finalize(store: WorkshopEventStore, inbound_id: MessageId, *, sequence: int):
    return await record_outbound_message_with_streaming_finalization(
        store,
        OutboundMessage(
            in_reply_to_message_id=inbound_id,
            body=f"Final answer {sequence}",
            occurred_at=_NOW + timedelta(minutes=sequence, seconds=1),
        ),
    )


class TestConversationDeliveryAuthority:
    async def test_activation_is_idempotent_and_survives_restart(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, _ = await _open_with_inbound(path)
        first = await WorkshopConversationDeliveryAuthority(store).activate()
        retry = await WorkshopConversationDeliveryAuthority(store).activate()
        await store.close()

        reopened = await WorkshopEventStore.open(path)
        try:
            after_restart = await WorkshopConversationDeliveryAuthority(reopened).activate()
            assert first.inserted is True
            assert retry.inserted is False
            assert after_restart.inserted is False
            assert retry.epoch == first.epoch == after_restart.epoch
        finally:
            await reopened.close()

    async def test_concurrent_activation_creates_exactly_one_active_epoch(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        initial, _ = await _open_with_inbound(path)
        await initial.close()
        first_store = await WorkshopEventStore.open(path)
        second_store = await WorkshopEventStore.open(path)
        try:
            first, second = await asyncio.gather(
                WorkshopConversationDeliveryAuthority(first_store).activate(),
                WorkshopConversationDeliveryAuthority(second_store).activate(),
            )
            assert {first.inserted, second.inserted} == {True, False}
            assert first.epoch == second.epoch
            async with first_store.connection.execute(
                "SELECT COUNT(*) FROM delivery_authority_epochs WHERE status = 'active'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await first_store.close()
            await second_store.close()

    async def test_finalization_requires_and_internally_stamps_the_active_epoch(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            with pytest.raises(DeliveryAuthorityInactiveError):
                await _finalize(store, inbound_id, sequence=1)

            async with store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
                assert (await cursor.fetchone())[0] == 0
            activation = await WorkshopConversationDeliveryAuthority(store).activate()
            result = await _finalize(store, inbound_id, sequence=1)

            assert result.delivery.delivery.authority_epoch_id == activation.epoch.epoch_id
            assert result.delivery.delivery.status == "pending"
        finally:
            await store.close()

    async def test_nonterminal_work_blocks_deactivation_and_another_epoch_cannot_claim_it(
        self,
        tmp_path: Path,
    ):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        authority = WorkshopConversationDeliveryAuthority(store)
        activation = await authority.activate()
        result = await _finalize(store, inbound_id, sequence=1)
        try:
            with pytest.raises(DeliveryAuthorityOutstandingWorkError):
                await authority.deactivate()

            unrelated_epoch = type(activation.epoch.epoch_id).new()
            claim = await WorkshopDeliveryOutbox(store).claim_next(
                "wrong-epoch-worker",
                purposes=(CONVERSATION_REPLY_PURPOSE,),
                execution_contracts=(STREAMING_FINALIZATION_CONTRACT,),
                delivery_id=result.delivery.delivery.delivery_id,
                authority_epoch_id=unrelated_epoch,
            )
            assert claim is None
            assert (await authority.active_epoch()).epoch_id == activation.epoch.epoch_id
        finally:
            await store.close()

    async def test_deactivation_and_reactivation_never_replay_prior_epoch_work(self, tmp_path: Path):
        store, first_inbound = await _open_with_inbound(tmp_path / "kai.db")
        authority = WorkshopConversationDeliveryAuthority(store)
        first_epoch = (await authority.activate()).epoch
        first = await _finalize(store, first_inbound, sequence=1)
        first_bot = AsyncMock()
        first_bot.send_message.return_value = SimpleNamespace(message_id=8001)

        def worker_clock() -> datetime:
            return _NOW + timedelta(hours=1)

        first_worker = WorkshopTelegramStreamingFinalizationWorker(
            WorkshopDeliveryOutbox(store, clock=worker_clock),
            WorkshopDeliveryFragments(store, clock=worker_clock),
            WorkshopTelegramStreamingFinalizationAdapter(first_bot),
            worker_id="first-authority-worker",
            authority_epoch_id=first_epoch.epoch_id,
        )
        assert (await first_worker.run_delivery(first.delivery.delivery.delivery_id)).outcome == (
            TelegramWorkOutcome.SUCCEEDED
        )

        deactivated = await authority.deactivate()
        second_epoch = (await authority.activate()).epoch
        second_inbound = await _add_inbound(store, sequence=2)
        second = await _finalize(store, second_inbound, sequence=2)
        second_bot = AsyncMock()
        second_bot.send_message.return_value = SimpleNamespace(message_id=8002)
        second_worker = WorkshopTelegramStreamingFinalizationWorker(
            WorkshopDeliveryOutbox(store, clock=worker_clock),
            WorkshopDeliveryFragments(store, clock=worker_clock),
            WorkshopTelegramStreamingFinalizationAdapter(second_bot),
            worker_id="second-authority-worker",
            authority_epoch_id=second_epoch.epoch_id,
        )
        try:
            assert deactivated.status == "deactivated"
            assert second_epoch.epoch_id != first_epoch.epoch_id
            assert second.delivery.delivery.authority_epoch_id == second_epoch.epoch_id
            assert (await second_worker.run_once()).delivery_id == second.delivery.delivery.delivery_id
            assert (await second_worker.run_once()).outcome == TelegramWorkOutcome.IDLE
            assert second_bot.send_message.await_count == 1
            assert (await WorkshopDeliveryOutbox(store).state(first.delivery.delivery.delivery_id)).status == (
                "succeeded"
            )
        finally:
            await store.close()

    async def test_terminal_failure_requires_explicit_acknowledgement_before_deactivation(
        self,
        tmp_path: Path,
    ):
        path = tmp_path / "kai.db"
        store, inbound_id = await _open_with_inbound(path)
        authority = WorkshopConversationDeliveryAuthority(store)
        epoch = (await authority.activate()).epoch
        result = await _finalize(store, inbound_id, sequence=1)
        bot = AsyncMock()
        bot.send_message.side_effect = [BadRequest("rejected"), BadRequest("rejected")]

        def worker_clock() -> datetime:
            return _NOW + timedelta(hours=1)

        worker = WorkshopTelegramStreamingFinalizationWorker(
            WorkshopDeliveryOutbox(store, clock=worker_clock),
            WorkshopDeliveryFragments(store, clock=worker_clock),
            WorkshopTelegramStreamingFinalizationAdapter(bot),
            worker_id="failed-authority-worker",
            authority_epoch_id=epoch.epoch_id,
        )
        assert (await worker.run_delivery(result.delivery.delivery.delivery_id)).outcome == (TelegramWorkOutcome.FAILED)
        with pytest.raises(DeliveryAuthorityUnreconciledFailureError):
            await authority.deactivate()

        deactivated = await authority.deactivate(acknowledge_terminal_failures=True)
        await store.close()

        assert deactivated.terminal_failures_acknowledged_at is not None
        status = workshop_delivery_authority_status(path)
        assert "Workshop delivery authority: inactive" in status
        assert "prior failed=1" in status
        assert "unacknowledged epochs=0" in status


class TestDeliveryAuthorityDiagnostic:
    def test_missing_schema_is_pending(self, tmp_path: Path):
        assert workshop_delivery_authority_status(tmp_path / "missing.db") == (
            "Workshop delivery authority: pending; authority schema unavailable"
        )

    async def test_status_reports_only_aggregate_active_work(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, inbound_id = await _open_with_inbound(path)
        activation = await WorkshopConversationDeliveryAuthority(store).activate()
        result = await _finalize(store, inbound_id, sequence=1)
        await store.close()

        status = workshop_delivery_authority_status(path)
        assert status == (
            "Workshop delivery authority: active; epochs=1, unclassified=0, prior nonterminal=0, "
            "prior failed=0, prior uncertain=0, unacknowledged epochs=0, "
            "active pending=1, leased=0, retrying=0, succeeded=0, failed=0, uncertain=0"
        )
        assert str(activation.epoch.epoch_id) not in status
        assert str(result.delivery.delivery.delivery_id) not in status

    async def test_unclassified_historical_work_is_not_ready_and_blocks_activation(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, inbound_id = await _open_with_inbound(path)
        activation = await WorkshopConversationDeliveryAuthority(store).activate()
        result = await _finalize(store, inbound_id, sequence=1)
        await store.connection.execute(
            "UPDATE delivery_outbox SET authority_epoch_id = NULL WHERE id = ?",
            (result.delivery.delivery.delivery_id,),
        )
        await store.connection.execute(
            "DELETE FROM delivery_authority_epochs WHERE id = ?",
            (activation.epoch.epoch_id,),
        )
        await store.connection.commit()
        try:
            with pytest.raises(DeliveryAuthorityHistoricalWorkError):
                await WorkshopConversationDeliveryAuthority(store).activate()
        finally:
            await store.close()

        assert workshop_delivery_authority_status(path).startswith(
            "Workshop delivery authority: NOT READY; epochs=0, unclassified=1"
        )
