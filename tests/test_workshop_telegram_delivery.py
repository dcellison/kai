"""Contracts for the production-unused Workshop Telegram outbox worker."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from telegram.constants import ParseMode
from telegram.error import BadRequest, Forbidden, InvalidToken, NetworkError, RetryAfter, TelegramError, TimedOut

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.delivery_outbox import DeliveryClaim, DeliveryRequest, WorkshopDeliveryOutbox
from kai.workshop.domain import (
    ChannelBindingId,
    ChannelId,
    DeliveryAttemptId,
    DeliveryId,
    EventEnvelope,
    MessageId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import OutboundMessage, record_outbound_message
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore
from kai.workshop.telegram_delivery import (
    TelegramDeliveryContractError,
    TelegramDeliveryFailure,
    TelegramWorkOutcome,
    WorkshopTelegramDeliveryAdapter,
    WorkshopTelegramDeliveryWorker,
)

_NOW = datetime(2026, 8, 12, 9, 32, tzinfo=UTC)


def _claim(
    *,
    external_channel_id: str = "101",
    transport: str = "telegram",
    mode: str = "text",
    body: str = "Hello from Workshop",
) -> DeliveryClaim:
    return DeliveryClaim(
        delivery_id=DeliveryId.new(),
        attempt_id=DeliveryAttemptId.new(),
        attempt_number=1,
        workshop_id=WorkshopId.new(),
        channel_id=ChannelId.new(),
        channel_binding_id=ChannelBindingId.new(),
        message_id=MessageId.new(),
        transport=transport,
        external_channel_id=external_channel_id,
        mode=mode,
        body=body,
        lease_expires_at=_NOW + timedelta(seconds=30),
    )


async def _open_with_outbound(
    path: Path,
    *,
    transport: str = "telegram",
) -> tuple[WorkshopEventStore, MessageId, ChannelBindingId]:
    subject = "101" if transport == "telegram" else "person@example.com"
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Authorized Human",
                role="admin",
                transport=transport,
                external_subject=subject,
                external_channel_id=subject,
            ),
        ),
    )
    inbound = await record_inbound_message(
        store,
        InboundMessage(
            transport=transport,
            update_id="9001",
            message_id="42",
            sender_subject=subject,
            channel_subject=subject,
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
        "WHERE m.id = ? AND cb.transport = ?",
        (message_id, transport),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return store, message_id, ChannelBindingId(str(row[0]))


async def _request(
    store: WorkshopEventStore,
    message_id: MessageId,
    binding_id: ChannelBindingId,
    *,
    mode: str = "text",
) -> DeliveryId:
    result = await WorkshopDeliveryOutbox(store).request_delivery(
        DeliveryRequest(
            message_id=message_id,
            channel_binding_id=binding_id,
            mode=mode,
            occurred_at=_NOW,
        )
    )
    return result.delivery.delivery_id


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
    binding_id = ChannelBindingId.new()
    await store.append(
        EventEnvelope.create(
            event_type=WorkshopEventType.TRANSPORT_CHANNEL_BOUND,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="channel_binding",
            aggregate_id=binding_id,
            occurred_at=_NOW,
            idempotency_key="test:telegram-notification-group",
            payload={
                "channel_id": str(row[1]),
                "transport": "telegram",
                "external_channel_id": "-1001234567890",
            },
        )
    )
    await store.project_pending(CanonicalConversationProjection())
    return binding_id


class TestWorkshopTelegramDeliveryAdapter:
    @pytest.mark.parametrize(
        ("external_channel_id", "expected_target"),
        [("101", 101), ("-1001234567890", -1001234567890), ("@KaiUpdates", "@KaiUpdates")],
    )
    async def test_delivers_private_group_and_named_channel_targets(
        self,
        external_channel_id,
        expected_target,
    ):
        bot = AsyncMock()
        adapter = WorkshopTelegramDeliveryAdapter(bot)

        await adapter.deliver(_claim(external_channel_id=external_channel_id))

        bot.send_message.assert_awaited_once_with(
            chat_id=expected_target,
            text="Hello from Workshop",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def test_markdown_rejection_retries_once_as_plain_text(self):
        bot = AsyncMock()
        bot.send_message.side_effect = [BadRequest("can't parse entities"), object()]
        adapter = WorkshopTelegramDeliveryAdapter(bot)

        await adapter.deliver(_claim())

        assert bot.send_message.await_args_list[0].kwargs["parse_mode"] == ParseMode.MARKDOWN
        assert bot.send_message.await_args_list[1].kwargs == {
            "chat_id": 101,
            "text": "Hello from Workshop",
        }

    @pytest.mark.parametrize(
        ("error", "retryable", "error_code"),
        [
            (TimedOut(), True, "telegram_timeout"),
            (NetworkError("reset"), True, "telegram_network_error"),
            (TelegramError("unknown"), True, "telegram_error"),
            (Forbidden("blocked"), False, "telegram_forbidden"),
            (InvalidToken(), False, "telegram_invalid_token"),
        ],
    )
    async def test_classifies_telegram_failures_without_persisting_provider_messages(
        self,
        error,
        retryable,
        error_code,
    ):
        bot = AsyncMock()
        bot.send_message.side_effect = error

        with pytest.raises(TelegramDeliveryFailure) as raised:
            await WorkshopTelegramDeliveryAdapter(bot).deliver(_claim())

        assert raised.value.retryable is retryable
        assert raised.value.error_code == error_code
        assert str(raised.value) == error_code

    async def test_rate_limit_is_retryable(self, monkeypatch):
        monkeypatch.setenv("PTB_TIMEDELTA", "1")
        bot = AsyncMock()
        bot.send_message.side_effect = RetryAfter(timedelta(seconds=7))

        with pytest.raises(TelegramDeliveryFailure) as raised:
            await WorkshopTelegramDeliveryAdapter(bot).deliver(_claim())

        assert raised.value.retryable is True
        assert raised.value.error_code == "telegram_rate_limited"
        assert raised.value.minimum_retry_delay == timedelta(seconds=7)

    async def test_second_bad_request_is_permanent(self):
        bot = AsyncMock()
        bot.send_message.side_effect = [BadRequest("markup"), BadRequest("chat not found")]

        with pytest.raises(TelegramDeliveryFailure) as raised:
            await WorkshopTelegramDeliveryAdapter(bot).deliver(_claim())

        assert raised.value.retryable is False
        assert raised.value.error_code == "telegram_bad_request"

    @pytest.mark.parametrize(
        "claim",
        [
            _claim(transport="email"),
            _claim(mode="photo"),
            _claim(external_channel_id="not a Telegram target"),
            _claim(body="x" * 4097),
        ],
    )
    async def test_rejects_unsupported_claims_before_calling_telegram(self, claim):
        bot = AsyncMock()

        with pytest.raises(TelegramDeliveryContractError):
            await WorkshopTelegramDeliveryAdapter(bot).deliver(claim)

        bot.send_message.assert_not_awaited()


class TestWorkshopTelegramDeliveryWorker:
    async def test_successfully_delivers_and_completes_private_chat_work(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        delivery_id = await _request(store, message_id, binding_id)
        bot = AsyncMock()
        worker = WorkshopTelegramDeliveryWorker(
            WorkshopDeliveryOutbox(store),
            WorkshopTelegramDeliveryAdapter(bot),
            worker_id="telegram-worker-1",
        )
        try:
            result = await worker.run_once()

            assert result.outcome == TelegramWorkOutcome.SUCCEEDED
            assert result.delivery_id == delivery_id
            assert result.attempt_number == 1
            assert (await WorkshopDeliveryOutbox(store).state(delivery_id)).status == "succeeded"
            bot.send_message.assert_awaited_once()
            assert bot.send_message.await_args.kwargs["chat_id"] == 101
        finally:
            await store.close()

    async def test_notification_group_binding_is_the_only_send_target(self, tmp_path: Path):
        store, message_id, _ = await _open_with_outbound(tmp_path / "kai.db")
        group_binding_id = await _add_group_binding(store, message_id)
        await _request(store, message_id, group_binding_id)
        bot = AsyncMock()
        worker = WorkshopTelegramDeliveryWorker(
            WorkshopDeliveryOutbox(store),
            WorkshopTelegramDeliveryAdapter(bot),
            worker_id="telegram-worker-1",
        )
        try:
            assert (await worker.run_once()).outcome == TelegramWorkOutcome.SUCCEEDED
            assert bot.send_message.await_args.kwargs["chat_id"] == -1001234567890
        finally:
            await store.close()

    async def test_retryable_failure_schedules_bounded_outbox_retry(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        delivery_id = await _request(store, message_id, binding_id)
        bot = AsyncMock()
        bot.send_message.side_effect = TimedOut()
        worker = WorkshopTelegramDeliveryWorker(
            WorkshopDeliveryOutbox(store),
            WorkshopTelegramDeliveryAdapter(bot),
            worker_id="telegram-worker-1",
        )
        try:
            result = await worker.run_once()

            assert result.outcome == TelegramWorkOutcome.RETRY_SCHEDULED
            assert result.error_code == "telegram_timeout"
            state = await WorkshopDeliveryOutbox(store).state(delivery_id)
            assert state.status == "retry_wait"
            assert state.last_error_code == "telegram_timeout"
        finally:
            await store.close()

    async def test_permanent_failure_completes_terminally(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        delivery_id = await _request(store, message_id, binding_id)
        bot = AsyncMock()
        bot.send_message.side_effect = Forbidden("blocked")
        worker = WorkshopTelegramDeliveryWorker(
            WorkshopDeliveryOutbox(store),
            WorkshopTelegramDeliveryAdapter(bot),
            worker_id="telegram-worker-1",
        )
        try:
            result = await worker.run_once()

            assert result.outcome == TelegramWorkOutcome.FAILED
            assert (await WorkshopDeliveryOutbox(store).state(delivery_id)).status == "failed"
        finally:
            await store.close()

    @pytest.mark.parametrize(
        ("transport", "mode"),
        [("email", "text"), ("telegram", "photo")],
    )
    async def test_worker_does_not_claim_other_transports_or_modes(
        self,
        tmp_path: Path,
        transport,
        mode,
    ):
        store, message_id, binding_id = await _open_with_outbound(
            tmp_path / "kai.db",
            transport=transport,
        )
        other_delivery = await _request(store, message_id, binding_id, mode=mode)
        bot = AsyncMock()
        worker = WorkshopTelegramDeliveryWorker(
            WorkshopDeliveryOutbox(store),
            WorkshopTelegramDeliveryAdapter(bot),
            worker_id="telegram-worker-1",
        )
        try:
            assert (await worker.run_once()).outcome == TelegramWorkOutcome.IDLE
            assert (await WorkshopDeliveryOutbox(store).state(other_delivery)).status == "pending"
            bot.send_message.assert_not_awaited()
        finally:
            await store.close()

    async def test_unexpected_adapter_failure_leaves_lease_for_crash_recovery(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        delivery_id = await _request(store, message_id, binding_id)
        bot = AsyncMock()
        bot.send_message.side_effect = RuntimeError("programming error")
        worker = WorkshopTelegramDeliveryWorker(
            WorkshopDeliveryOutbox(store),
            WorkshopTelegramDeliveryAdapter(bot),
            worker_id="telegram-worker-1",
        )
        try:
            with pytest.raises(RuntimeError, match="programming error"):
                await worker.run_once()
            assert (await WorkshopDeliveryOutbox(store).state(delivery_id)).status == "leased"
        finally:
            await store.close()

    async def test_cancellation_leaves_lease_for_crash_recovery(self, tmp_path: Path):
        store, message_id, binding_id = await _open_with_outbound(tmp_path / "kai.db")
        delivery_id = await _request(store, message_id, binding_id)
        entered = asyncio.Event()

        async def blocked_send(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        bot = AsyncMock()
        bot.send_message.side_effect = blocked_send
        worker = WorkshopTelegramDeliveryWorker(
            WorkshopDeliveryOutbox(store),
            WorkshopTelegramDeliveryAdapter(bot),
            worker_id="telegram-worker-1",
        )
        task = asyncio.create_task(worker.run_once())
        try:
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert (await WorkshopDeliveryOutbox(store).state(delivery_id)).status == "leased"
        finally:
            if not task.done():
                task.cancel()
            await store.close()

    async def test_idle_run_stops_without_polling_when_already_signaled(self, tmp_path: Path):
        store, _, _ = await _open_with_outbound(tmp_path / "kai.db")
        bot = AsyncMock()
        worker = WorkshopTelegramDeliveryWorker(
            WorkshopDeliveryOutbox(store),
            WorkshopTelegramDeliveryAdapter(bot),
            worker_id="telegram-worker-1",
            poll_interval=0.01,
        )
        stop_event = asyncio.Event()
        stop_event.set()
        try:
            await worker.run(stop_event)
            bot.send_message.assert_not_awaited()
        finally:
            await store.close()
