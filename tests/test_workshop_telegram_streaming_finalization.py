"""Production-unused Telegram execution contracts for streaming finalization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.constants import ParseMode
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.delivery_fragments import (
    EDIT_OPERATION,
    SEND_OPERATION,
    DeliveryFragment,
    WorkshopDeliveryFragments,
)
from kai.workshop.delivery_outbox import (
    CONVERSATION_REPLY_PURPOSE,
    QUALIFICATION_PURPOSE,
    STREAMING_FINALIZATION_CONTRACT,
    DeliveryClaim,
    DeliveryPurpose,
    DeliveryRequest,
    WorkshopDeliveryOutbox,
)
from kai.workshop.domain import (
    ChannelBindingId,
    ChannelId,
    DeliveryAttemptId,
    DeliveryId,
    MessageId,
    WorkshopId,
)
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import (
    OutboundMessage,
    record_outbound_message,
    record_outbound_message_with_streaming_finalization,
)
from kai.workshop.store import WorkshopEventStore
from kai.workshop.streaming_preview import (
    ConfirmedTelegramStreamingPreview,
    bind_confirmed_telegram_streaming_preview,
)
from kai.workshop.telegram_delivery import (
    TelegramDeliveryContractError,
    TelegramDeliveryFailure,
    TelegramWorkOutcome,
    WorkshopTelegramStreamingFinalizationAdapter,
    WorkshopTelegramStreamingFinalizationWorker,
)

_NOW = datetime(2026, 8, 12, 13, 30, tzinfo=UTC)


@dataclass
class _Clock:
    now: datetime = _NOW

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


def _claim(*, execution_contract=STREAMING_FINALIZATION_CONTRACT) -> DeliveryClaim:
    return DeliveryClaim(
        delivery_id=DeliveryId.new(),
        attempt_id=DeliveryAttemptId.new(),
        attempt_number=1,
        workshop_id=WorkshopId.new(),
        channel_id=ChannelId.new(),
        channel_binding_id=ChannelBindingId.new(),
        message_id=MessageId.new(),
        transport="telegram",
        external_channel_id="101",
        mode="text",
        purpose=CONVERSATION_REPLY_PURPOSE,
        execution_contract=execution_contract,
        body="Final answer",
        lease_expires_at=_NOW + timedelta(seconds=30),
    )


def _fragment(
    claim: DeliveryClaim,
    *,
    operation=EDIT_OPERATION,
    target_external_message_id: int | None = 7001,
    body: str = "Final answer",
) -> DeliveryFragment:
    return DeliveryFragment(
        delivery_id=claim.delivery_id,
        fragment_index=0,
        fragment_count=1,
        body=body,
        status="sending",
        attempt_id=claim.attempt_id,
        external_message_id=None,
        sent_at=None,
        operation=operation,
        target_external_message_id=target_external_message_id,
    )


async def _open_with_inbound(path: Path) -> tuple[WorkshopEventStore, MessageId]:
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
    return store, MessageId(str(inbound.event.envelope.aggregate_id))


async def _prepare_finalization(
    path: Path,
    *,
    body: str = "Final answer",
    preview_message_id: int | None = 7001,
) -> tuple[WorkshopEventStore, DeliveryId]:
    store, inbound_id = await _open_with_inbound(path)
    if preview_message_id is not None:
        await bind_confirmed_telegram_streaming_preview(
            store,
            ConfirmedTelegramStreamingPreview(
                inbound_message_id=inbound_id,
                external_message_id=preview_message_id,
                confirmed_at=_NOW + timedelta(seconds=1),
            ),
        )
    result = await record_outbound_message_with_streaming_finalization(
        store,
        OutboundMessage(
            in_reply_to_message_id=inbound_id,
            body=body,
            occurred_at=_NOW + timedelta(seconds=2),
        ),
    )
    return store, result.delivery.delivery.delivery_id


def _worker(
    store: WorkshopEventStore,
    bot: AsyncMock,
    *,
    clock: _Clock | None = None,
    lease_duration: timedelta = timedelta(seconds=30),
) -> WorkshopTelegramStreamingFinalizationWorker:
    effective_clock = clock or _Clock(_NOW + timedelta(seconds=3))
    return WorkshopTelegramStreamingFinalizationWorker(
        WorkshopDeliveryOutbox(store, clock=effective_clock),
        WorkshopDeliveryFragments(store, clock=effective_clock),
        WorkshopTelegramStreamingFinalizationAdapter(bot),
        worker_id="telegram-finalization-worker",
        lease_duration=lease_duration,
        poll_interval=0.01,
    )


class TestTelegramStreamingFinalizationAdapter:
    async def test_edits_the_exact_confirmed_message_with_markdown(self):
        bot = AsyncMock()
        bot.edit_message_text.return_value = SimpleNamespace(message_id=7001)
        claim = _claim()

        result = await WorkshopTelegramStreamingFinalizationAdapter(bot).deliver_fragment(
            claim,
            _fragment(claim),
        )

        assert result == 7001
        bot.edit_message_text.assert_awaited_once_with(
            chat_id=101,
            message_id=7001,
            text="Final answer",
            parse_mode=ParseMode.MARKDOWN,
        )
        bot.send_message.assert_not_awaited()

    async def test_edit_markdown_rejection_retries_once_as_plain_text(self):
        bot = AsyncMock()
        bot.edit_message_text.side_effect = [
            BadRequest("can't parse entities"),
            SimpleNamespace(message_id=7001),
        ]
        claim = _claim()

        assert (
            await WorkshopTelegramStreamingFinalizationAdapter(bot).deliver_fragment(
                claim,
                _fragment(claim),
            )
            == 7001
        )
        assert bot.edit_message_text.await_args_list[0].kwargs["parse_mode"] == ParseMode.MARKDOWN
        assert bot.edit_message_text.await_args_list[1].kwargs == {
            "chat_id": 101,
            "message_id": 7001,
            "text": "Final answer",
        }

    async def test_already_identical_terminal_snapshot_is_confirmed_without_a_second_call(self):
        bot = AsyncMock()
        bot.edit_message_text.side_effect = BadRequest("Message is not modified")
        claim = _claim()

        assert (
            await WorkshopTelegramStreamingFinalizationAdapter(bot).deliver_fragment(
                claim,
                _fragment(claim),
            )
            == 7001
        )
        assert bot.edit_message_text.await_count == 1

    async def test_deleted_or_uneditable_preview_is_a_sanitized_permanent_failure(self):
        bot = AsyncMock()
        bot.edit_message_text.side_effect = [
            BadRequest("message to edit not found"),
            BadRequest("message to edit not found"),
        ]
        claim = _claim()

        with pytest.raises(TelegramDeliveryFailure) as raised:
            await WorkshopTelegramStreamingFinalizationAdapter(bot).deliver_fragment(
                claim,
                _fragment(claim),
            )

        assert raised.value.error_code == "telegram_edit_rejected"
        assert raised.value.retryable is False
        assert raised.value.ambiguous is False
        assert str(raised.value) == "telegram_edit_rejected"

    @pytest.mark.parametrize(
        ("error", "error_code"),
        [
            (TimedOut(), "telegram_edit_timeout_uncertain"),
            (NetworkError("reset"), "telegram_edit_network_uncertain"),
        ],
    )
    async def test_edit_transport_ambiguity_is_operation_specific(self, error, error_code):
        bot = AsyncMock()
        bot.edit_message_text.side_effect = error
        claim = _claim()

        with pytest.raises(TelegramDeliveryFailure) as raised:
            await WorkshopTelegramStreamingFinalizationAdapter(bot).deliver_fragment(
                claim,
                _fragment(claim),
            )

        assert raised.value.error_code == error_code
        assert raised.value.ambiguous is True
        assert raised.value.retryable is False

    async def test_edit_success_with_mismatched_message_evidence_is_ambiguous(self):
        bot = AsyncMock()
        bot.edit_message_text.return_value = SimpleNamespace(message_id=7999)
        claim = _claim()

        with pytest.raises(TelegramDeliveryFailure) as raised:
            await WorkshopTelegramStreamingFinalizationAdapter(bot).deliver_fragment(
                claim,
                _fragment(claim),
            )

        assert raised.value.error_code == "telegram_edit_response_invalid"
        assert raised.value.ambiguous is True

    async def test_edit_rate_limit_is_retryable_only_after_a_definitive_rejection(self, monkeypatch):
        monkeypatch.setenv("PTB_TIMEDELTA", "1")
        bot = AsyncMock()
        bot.edit_message_text.side_effect = RetryAfter(timedelta(seconds=7))
        claim = _claim()

        with pytest.raises(TelegramDeliveryFailure) as raised:
            await WorkshopTelegramStreamingFinalizationAdapter(bot).deliver_fragment(
                claim,
                _fragment(claim),
            )

        assert raised.value.error_code == "telegram_rate_limited"
        assert raised.value.retryable is True
        assert raised.value.ambiguous is False
        assert raised.value.minimum_retry_delay == timedelta(seconds=7)

    async def test_adapter_rejects_another_execution_contract_before_telegram(self):
        bot = AsyncMock()
        claim = _claim(execution_contract="send_fragments")

        with pytest.raises(TelegramDeliveryContractError, match="telegram_execution_contract_mismatch"):
            await WorkshopTelegramStreamingFinalizationAdapter(bot).deliver_fragment(
                claim,
                _fragment(claim),
            )

        bot.edit_message_text.assert_not_awaited()
        bot.send_message.assert_not_awaited()


class TestTelegramStreamingFinalizationWorker:
    async def test_short_preview_is_finalized_in_place(self, tmp_path: Path):
        store, delivery_id = await _prepare_finalization(tmp_path / "kai.db")
        bot = AsyncMock()
        bot.edit_message_text.return_value = SimpleNamespace(message_id=7001)
        try:
            result = await _worker(store, bot).run_delivery(delivery_id)
            persisted = await WorkshopDeliveryFragments(store).fragments(delivery_id)

            assert result.outcome == TelegramWorkOutcome.SUCCEEDED
            assert (await WorkshopDeliveryOutbox(store).state(delivery_id)).status == "succeeded"
            assert [(item.operation, item.status, item.external_message_id) for item in persisted] == [
                (EDIT_OPERATION, "sent", "7001")
            ]
            bot.send_message.assert_not_awaited()
        finally:
            await store.close()

    async def test_long_reply_edits_then_sends_continuations_in_order(self, tmp_path: Path):
        body = "A" * 4096 + "\n" + "B" * 20
        store, delivery_id = await _prepare_finalization(tmp_path / "kai.db", body=body)
        bot = AsyncMock()
        bot.edit_message_text.return_value = SimpleNamespace(message_id=7001)
        bot.send_message.return_value = SimpleNamespace(message_id=8001)
        try:
            result = await _worker(store, bot).run_once()
            persisted = await WorkshopDeliveryFragments(store).fragments(delivery_id)

            assert result.outcome == TelegramWorkOutcome.SUCCEEDED
            assert [item.operation for item in persisted] == [EDIT_OPERATION, SEND_OPERATION]
            assert [item.external_message_id for item in persisted] == ["7001", "8001"]
            assert bot.edit_message_text.await_args.kwargs["text"] == "A" * 4096
            assert bot.send_message.await_args.kwargs["text"] == "B" * 20
        finally:
            await store.close()

    async def test_no_preview_executes_an_all_send_plan(self, tmp_path: Path):
        body = "A" * 4096 + "\n" + "B" * 20
        store, delivery_id = await _prepare_finalization(
            tmp_path / "kai.db",
            body=body,
            preview_message_id=None,
        )
        bot = AsyncMock()
        bot.send_message.side_effect = [
            SimpleNamespace(message_id=8001),
            SimpleNamespace(message_id=8002),
        ]
        try:
            assert (await _worker(store, bot).run_once()).outcome == TelegramWorkOutcome.SUCCEEDED
            persisted = await WorkshopDeliveryFragments(store).fragments(delivery_id)
            assert [item.operation for item in persisted] == [SEND_OPERATION, SEND_OPERATION]
            assert [item.external_message_id for item in persisted] == ["8001", "8002"]
            bot.edit_message_text.assert_not_awaited()
        finally:
            await store.close()

    async def test_restart_skips_confirmed_edit_and_resumes_continuation_send(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        monkeypatch.setenv("PTB_TIMEDELTA", "1")
        path = tmp_path / "kai.db"
        body = "A" * 4096 + "\n" + "B" * 20
        store, delivery_id = await _prepare_finalization(path, body=body)
        clock = _Clock(_NOW + timedelta(seconds=3))
        first_bot = AsyncMock()
        first_bot.edit_message_text.return_value = SimpleNamespace(message_id=7001)
        first_bot.send_message.side_effect = RetryAfter(timedelta(seconds=1))
        first = await _worker(store, first_bot, clock=clock).run_once()
        assert first.outcome == TelegramWorkOutcome.RETRY_SCHEDULED
        assert [item.status for item in await WorkshopDeliveryFragments(store).fragments(delivery_id)] == [
            "sent",
            "pending",
        ]
        await store.close()

        clock.advance(timedelta(seconds=5))
        reopened = await WorkshopEventStore.open(path)
        second_bot = AsyncMock()
        second_bot.send_message.return_value = SimpleNamespace(message_id=8001)
        try:
            second = await _worker(reopened, second_bot, clock=clock).run_delivery(delivery_id)

            assert second.outcome == TelegramWorkOutcome.SUCCEEDED
            assert second.attempt_number == 2
            second_bot.edit_message_text.assert_not_awaited()
            second_bot.send_message.assert_awaited_once()
            assert [
                item.external_message_id for item in await WorkshopDeliveryFragments(reopened).fragments(delivery_id)
            ] == ["7001", "8001"]
        finally:
            await reopened.close()

    async def test_deleted_preview_fails_terminally_without_a_send_fallback(self, tmp_path: Path):
        store, delivery_id = await _prepare_finalization(tmp_path / "kai.db")
        bot = AsyncMock()
        bot.edit_message_text.side_effect = [
            BadRequest("message to edit not found"),
            BadRequest("message to edit not found"),
        ]
        worker = _worker(store, bot)
        try:
            result = await worker.run_once()

            assert result.outcome == TelegramWorkOutcome.FAILED
            assert result.error_code == "telegram_edit_rejected"
            assert (await WorkshopDeliveryOutbox(store).state(delivery_id)).status == "failed"
            assert (await WorkshopDeliveryFragments(store).fragments(delivery_id))[0].status == "pending"
            assert (await worker.run_once()).outcome == TelegramWorkOutcome.IDLE
            bot.send_message.assert_not_awaited()
        finally:
            await store.close()

    async def test_ambiguous_edit_is_terminal_and_never_retried(self, tmp_path: Path):
        store, delivery_id = await _prepare_finalization(tmp_path / "kai.db")
        bot = AsyncMock()
        bot.edit_message_text.side_effect = TimedOut()
        worker = _worker(store, bot)
        try:
            result = await worker.run_once()

            assert result.outcome == TelegramWorkOutcome.FAILED
            assert result.error_code == "telegram_edit_timeout_uncertain"
            assert (await WorkshopDeliveryFragments(store).fragments(delivery_id))[0].status == "uncertain"
            assert (await worker.run_once()).outcome == TelegramWorkOutcome.IDLE
            assert bot.edit_message_text.await_count == 1
        finally:
            await store.close()

    async def test_ambiguous_continuation_send_preserves_confirmed_edit_and_never_retries(
        self,
        tmp_path: Path,
    ):
        body = "A" * 4096 + "\n" + "B" * 20
        store, delivery_id = await _prepare_finalization(tmp_path / "kai.db", body=body)
        bot = AsyncMock()
        bot.edit_message_text.return_value = SimpleNamespace(message_id=7001)
        bot.send_message.side_effect = TimedOut()
        worker = _worker(store, bot)
        try:
            result = await worker.run_once()
            persisted = await WorkshopDeliveryFragments(store).fragments(delivery_id)

            assert result.outcome == TelegramWorkOutcome.FAILED
            assert result.error_code == "telegram_timeout_uncertain"
            assert [(item.operation, item.status) for item in persisted] == [
                (EDIT_OPERATION, "sent"),
                (SEND_OPERATION, "uncertain"),
            ]
            assert (await worker.run_once()).outcome == TelegramWorkOutcome.IDLE
            assert bot.edit_message_text.await_count == 1
            assert bot.send_message.await_count == 1
        finally:
            await store.close()

    async def test_expired_inflight_edit_recovers_as_terminal_edit_uncertainty(self, tmp_path: Path):
        store, delivery_id = await _prepare_finalization(tmp_path / "kai.db")
        clock = _Clock(_NOW + timedelta(seconds=3))
        outbox = WorkshopDeliveryOutbox(store, clock=clock)
        fragments = WorkshopDeliveryFragments(store, clock=clock)
        claim = await outbox.claim_next(
            "crashed-finalization-worker",
            purposes=(CONVERSATION_REPLY_PURPOSE,),
            execution_contracts=(STREAMING_FINALIZATION_CONTRACT,),
            lease_duration=timedelta(seconds=10),
            delivery_id=delivery_id,
        )
        assert claim is not None
        started = await fragments.begin_next(claim)
        assert started is not None and started.operation == EDIT_OPERATION
        await store.close()

        clock.advance(timedelta(seconds=10))
        reopened = await WorkshopEventStore.open(tmp_path / "kai.db")
        recovered_outbox = WorkshopDeliveryOutbox(reopened, clock=clock)
        try:
            recovered = await recovered_outbox.recover_expired_leases(
                purposes=(CONVERSATION_REPLY_PURPOSE,),
                execution_contracts=(STREAMING_FINALIZATION_CONTRACT,),
            )
            state = await recovered_outbox.state(delivery_id)

            assert recovered.failed == 1 and recovered.requeued == 0
            assert state.status == "failed"
            assert state.last_error_code == "delivery_edit_uncertain"
            assert (await WorkshopDeliveryFragments(reopened).fragments(delivery_id))[0].status == "uncertain"
        finally:
            await reopened.close()

    @pytest.mark.parametrize("purpose", [CONVERSATION_REPLY_PURPOSE, QUALIFICATION_PURPOSE])
    async def test_worker_cannot_claim_send_fragment_or_qualification_work(
        self,
        tmp_path: Path,
        purpose: DeliveryPurpose,
    ):
        store, inbound_id = await _open_with_inbound(tmp_path / f"{purpose}.db")
        outbound = await record_outbound_message(
            store,
            OutboundMessage(inbound_id, "Legacy send", _NOW + timedelta(seconds=2)),
        )
        message_id = MessageId(str(outbound.event.envelope.aggregate_id))
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
                purpose=purpose,
                occurred_at=_NOW + timedelta(seconds=2),
            )
        )
        bot = AsyncMock()
        try:
            assert (await _worker(store, bot).run_once()).outcome == TelegramWorkOutcome.IDLE
            assert (await WorkshopDeliveryOutbox(store).state(delivery.delivery.delivery_id)).status == "pending"
            bot.edit_message_text.assert_not_awaited()
            bot.send_message.assert_not_awaited()
        finally:
            await store.close()
