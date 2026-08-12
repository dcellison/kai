"""Production-unused Telegram adapter and worker for Workshop delivery claims."""

from __future__ import annotations

import asyncio
import re
import warnings
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Protocol

from telegram.constants import ParseMode
from telegram.error import (
    BadRequest,
    ChatMigrated,
    Conflict,
    Forbidden,
    InvalidToken,
    NetworkError,
    RetryAfter,
    TelegramError,
    TimedOut,
)
from telegram.warnings import PTBDeprecationWarning

from kai.workshop.delivery_outbox import DeliveryClaim, DeliveryState, WorkshopDeliveryOutbox
from kai.workshop.domain import DeliveryId

_NUMERIC_CHAT_ID_PATTERN = re.compile(r"^-?[1-9][0-9]{0,19}$")
_USERNAME_CHAT_ID_PATTERN = re.compile(r"^@[A-Za-z][A-Za-z0-9_]{4,31}$")
_TELEGRAM_TEXT_LIMIT = 4096


class TelegramTextBot(Protocol):
    """The Bot API surface needed by the production-unused text adapter."""

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = None,
    ) -> object: ...


class TelegramDeliveryFailure(RuntimeError):
    """A sanitized, policy-bearing Telegram failure safe for outbox storage."""

    def __init__(
        self,
        *,
        retryable: bool,
        error_code: str,
        minimum_retry_delay: timedelta | None = None,
    ) -> None:
        super().__init__(error_code)
        self.retryable = retryable
        self.error_code = error_code
        self.minimum_retry_delay = minimum_retry_delay


class TelegramDeliveryContractError(TelegramDeliveryFailure):
    """The durable claim cannot be represented by this Telegram adapter."""

    def __init__(self, error_code: str) -> None:
        super().__init__(retryable=False, error_code=error_code)


class TelegramWorkOutcome(StrEnum):
    IDLE = "idle"
    SUCCEEDED = "succeeded"
    RETRY_SCHEDULED = "retry_scheduled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TelegramWorkResult:
    outcome: TelegramWorkOutcome
    delivery_id: DeliveryId | None = None
    attempt_number: int | None = None
    error_code: str | None = None


def _telegram_target(external_channel_id: str) -> int | str:
    if _NUMERIC_CHAT_ID_PATTERN.fullmatch(external_channel_id):
        value = int(external_channel_id)
        if -(2**63) <= value <= 2**63 - 1:
            return value
    if _USERNAME_CHAT_ID_PATTERN.fullmatch(external_channel_id):
        return external_channel_id
    raise TelegramDeliveryContractError("telegram_target_invalid")


def _classify_telegram_error(error: TelegramError) -> TelegramDeliveryFailure:
    # BadRequest is a NetworkError subclass in python-telegram-bot, so all
    # permanent request/target errors must be checked before NetworkError.
    if isinstance(error, Forbidden):
        return TelegramDeliveryFailure(retryable=False, error_code="telegram_forbidden")
    if isinstance(error, InvalidToken):
        return TelegramDeliveryFailure(retryable=False, error_code="telegram_invalid_token")
    if isinstance(error, ChatMigrated):
        return TelegramDeliveryFailure(retryable=False, error_code="telegram_chat_migrated")
    if isinstance(error, BadRequest):
        return TelegramDeliveryFailure(retryable=False, error_code="telegram_bad_request")
    if isinstance(error, RetryAfter):
        # PTB 22 supports both integer and timedelta representations during a
        # deprecation window. Normalize at the adapter boundary without making
        # Kai's retry timing depend on the process-wide PTB_TIMEDELTA setting.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PTBDeprecationWarning)
            retry_after = error.retry_after
        minimum_retry_delay = retry_after if isinstance(retry_after, timedelta) else timedelta(seconds=retry_after)
        if minimum_retry_delay > timedelta(days=1):
            return TelegramDeliveryFailure(
                retryable=False,
                error_code="telegram_rate_limit_too_long",
            )
        return TelegramDeliveryFailure(
            retryable=True,
            error_code="telegram_rate_limited",
            minimum_retry_delay=minimum_retry_delay,
        )
    if isinstance(error, TimedOut):
        return TelegramDeliveryFailure(retryable=True, error_code="telegram_timeout")
    if isinstance(error, Conflict):
        return TelegramDeliveryFailure(retryable=True, error_code="telegram_conflict")
    if isinstance(error, NetworkError):
        return TelegramDeliveryFailure(retryable=True, error_code="telegram_network_error")
    return TelegramDeliveryFailure(retryable=True, error_code="telegram_error")


class WorkshopTelegramDeliveryAdapter:
    """Deliver one bounded Telegram text claim through one Bot API message."""

    def __init__(self, bot: TelegramTextBot) -> None:
        self._bot = bot

    async def deliver(self, claim: DeliveryClaim) -> None:
        if not isinstance(claim, DeliveryClaim):
            raise ValueError("claim must be a DeliveryClaim")
        if claim.transport != "telegram":
            raise TelegramDeliveryContractError("telegram_transport_mismatch")
        if claim.mode != "text":
            raise TelegramDeliveryContractError("telegram_mode_unsupported")
        if not claim.body or len(claim.body) > _TELEGRAM_TEXT_LIMIT:
            raise TelegramDeliveryContractError("telegram_text_size_unsupported")

        target = _telegram_target(claim.external_channel_id)
        try:
            await self._bot.send_message(
                chat_id=target,
                text=claim.body,
                parse_mode=ParseMode.MARKDOWN,
            )
        except BadRequest:
            # Match Kai's existing presentation behavior: Telegram Markdown
            # rejection is retried once as plain text. A second BadRequest is
            # classified as permanent by the common Telegram error policy.
            try:
                await self._bot.send_message(chat_id=target, text=claim.body)
            except TelegramError as error:
                raise _classify_telegram_error(error) from error
        except TelegramError as error:
            raise _classify_telegram_error(error) from error


class WorkshopTelegramDeliveryWorker:
    """Claim and settle Telegram text work without any production registration."""

    def __init__(
        self,
        outbox: WorkshopDeliveryOutbox,
        adapter: WorkshopTelegramDeliveryAdapter,
        *,
        worker_id: str,
        lease_duration: timedelta = timedelta(seconds=30),
        poll_interval: float = 1.0,
    ) -> None:
        if poll_interval <= 0 or poll_interval > 60:
            raise ValueError("poll_interval must be positive and at most 60 seconds")
        self._outbox = outbox
        self._adapter = adapter
        self._worker_id = worker_id
        self._lease_duration = lease_duration
        self._poll_interval = poll_interval

    async def run_once(self) -> TelegramWorkResult:
        claim = await self._outbox.claim_next(
            self._worker_id,
            lease_duration=self._lease_duration,
            transport="telegram",
            modes=("text",),
        )
        if claim is None:
            return TelegramWorkResult(outcome=TelegramWorkOutcome.IDLE)

        try:
            await self._adapter.deliver(claim)
        except TelegramDeliveryFailure as failure:
            state = await self._outbox.mark_failed(
                claim,
                retryable=failure.retryable,
                error_code=failure.error_code,
                minimum_retry_delay=failure.minimum_retry_delay,
            )
            return TelegramWorkResult(
                outcome=(
                    TelegramWorkOutcome.RETRY_SCHEDULED if state.status == "retry_wait" else TelegramWorkOutcome.FAILED
                ),
                delivery_id=claim.delivery_id,
                attempt_number=claim.attempt_number,
                error_code=failure.error_code,
            )

        state = await self._outbox.mark_succeeded(claim)
        return self._success_result(claim, state)

    async def run(self, stop_event: asyncio.Event) -> None:
        if not isinstance(stop_event, asyncio.Event):
            raise ValueError("stop_event must be an asyncio.Event")
        while not stop_event.is_set():
            result = await self.run_once()
            if result.outcome != TelegramWorkOutcome.IDLE:
                continue
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._poll_interval)
            except TimeoutError:
                pass

    @staticmethod
    def _success_result(claim: DeliveryClaim, state: DeliveryState) -> TelegramWorkResult:
        if state.status != "succeeded":
            raise RuntimeError("Successful Telegram delivery did not reach succeeded outbox state")
        return TelegramWorkResult(
            outcome=TelegramWorkOutcome.SUCCEEDED,
            delivery_id=claim.delivery_id,
            attempt_number=claim.attempt_number,
        )
