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

from kai.telegram_utils import chunk_text
from kai.workshop.delivery_fragments import DeliveryFragment, WorkshopDeliveryFragments
from kai.workshop.delivery_outbox import DeliveryClaim, DeliveryState, WorkshopDeliveryOutbox
from kai.workshop.domain import DeliveryId

_NUMERIC_CHAT_ID_PATTERN = re.compile(r"^-?[1-9][0-9]{0,19}$")
_USERNAME_CHAT_ID_PATTERN = re.compile(r"^@[A-Za-z][A-Za-z0-9_]{4,31}$")
_TELEGRAM_TEXT_LIMIT = 4096
_MAX_TELEGRAM_FRAGMENTS = 1000
_MAX_TELEGRAM_TEXT_SIZE = _TELEGRAM_TEXT_LIMIT * _MAX_TELEGRAM_FRAGMENTS


class TelegramSentMessage(Protocol):
    """The successful Bot API response evidence retained by the outbox."""

    message_id: int


class TelegramTextBot(Protocol):
    """The Bot API surface needed by the production-unused text adapter."""

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = None,
    ) -> TelegramSentMessage: ...


class TelegramDeliveryFailure(RuntimeError):
    """A sanitized, policy-bearing Telegram failure safe for outbox storage."""

    def __init__(
        self,
        *,
        retryable: bool,
        error_code: str,
        minimum_retry_delay: timedelta | None = None,
        ambiguous: bool = False,
    ) -> None:
        super().__init__(error_code)
        self.retryable = retryable
        self.error_code = error_code
        self.minimum_retry_delay = minimum_retry_delay
        self.ambiguous = ambiguous


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
        minimum_retry_delay = max(minimum_retry_delay, timedelta(seconds=1))
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
        return TelegramDeliveryFailure(
            retryable=False,
            error_code="telegram_timeout_uncertain",
            ambiguous=True,
        )
    if isinstance(error, Conflict):
        return TelegramDeliveryFailure(retryable=True, error_code="telegram_conflict")
    if isinstance(error, NetworkError):
        return TelegramDeliveryFailure(
            retryable=False,
            error_code="telegram_network_uncertain",
            ambiguous=True,
        )
    return TelegramDeliveryFailure(
        retryable=False,
        error_code="telegram_error_uncertain",
        ambiguous=True,
    )


def _telegram_fragments(body: str) -> tuple[str, ...]:
    if not body or len(body) > _MAX_TELEGRAM_TEXT_SIZE:
        raise TelegramDeliveryContractError("telegram_text_size_unsupported")
    fragments = tuple(chunk_text(body, max_len=_TELEGRAM_TEXT_LIMIT))
    if not fragments or len(fragments) > _MAX_TELEGRAM_FRAGMENTS:
        raise TelegramDeliveryContractError("telegram_text_size_unsupported")
    return fragments


class WorkshopTelegramDeliveryAdapter:
    """Deliver one durably selected Telegram text fragment."""

    def __init__(self, bot: TelegramTextBot) -> None:
        self._bot = bot

    async def deliver_fragment(self, claim: DeliveryClaim, fragment: DeliveryFragment) -> int:
        if not isinstance(claim, DeliveryClaim):
            raise ValueError("claim must be a DeliveryClaim")
        if not isinstance(fragment, DeliveryFragment) or fragment.delivery_id != claim.delivery_id:
            raise ValueError("fragment must belong to the claimed delivery")
        if claim.transport != "telegram":
            raise TelegramDeliveryContractError("telegram_transport_mismatch")
        if claim.mode != "text":
            raise TelegramDeliveryContractError("telegram_mode_unsupported")
        if not fragment.body or len(fragment.body) > _TELEGRAM_TEXT_LIMIT:
            raise TelegramDeliveryContractError("telegram_text_size_unsupported")

        target = _telegram_target(claim.external_channel_id)
        try:
            sent = await self._bot.send_message(
                chat_id=target,
                text=fragment.body,
                parse_mode=ParseMode.MARKDOWN,
            )
        except BadRequest:
            # Match Kai's existing presentation behavior: Telegram Markdown
            # rejection is retried once as plain text. A second BadRequest is
            # classified as permanent by the common Telegram error policy.
            try:
                sent = await self._bot.send_message(chat_id=target, text=fragment.body)
            except TelegramError as error:
                raise _classify_telegram_error(error) from error
        except TelegramError as error:
            raise _classify_telegram_error(error) from error
        message_id = getattr(sent, "message_id", None)
        if not isinstance(message_id, int) or isinstance(message_id, bool) or message_id <= 0:
            raise TelegramDeliveryFailure(
                retryable=False,
                error_code="telegram_response_invalid",
                ambiguous=True,
            )
        return message_id


class WorkshopTelegramDeliveryWorker:
    """Claim and settle Telegram text work without any production registration."""

    def __init__(
        self,
        outbox: WorkshopDeliveryOutbox,
        fragments: WorkshopDeliveryFragments,
        adapter: WorkshopTelegramDeliveryAdapter,
        *,
        worker_id: str,
        lease_duration: timedelta = timedelta(seconds=30),
        poll_interval: float = 1.0,
    ) -> None:
        if poll_interval <= 0 or poll_interval > 60:
            raise ValueError("poll_interval must be positive and at most 60 seconds")
        self._outbox = outbox
        self._fragments = fragments
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
        return await self._run_claim(claim)

    async def run_delivery(self, delivery_id: DeliveryId) -> TelegramWorkResult:
        """Run only one explicitly selected delivery without draining the outbox."""
        if not isinstance(delivery_id, DeliveryId):
            raise ValueError("delivery_id must be a DeliveryId")
        claim = await self._outbox.claim_next(
            self._worker_id,
            lease_duration=self._lease_duration,
            transport="telegram",
            modes=("text",),
            delivery_id=delivery_id,
        )
        return await self._run_claim(claim)

    async def _run_claim(self, claim: DeliveryClaim | None) -> TelegramWorkResult:
        if claim is None:
            return TelegramWorkResult(outcome=TelegramWorkOutcome.IDLE)

        try:
            bodies = _telegram_fragments(claim.body)
            await self._fragments.prepare(claim, bodies)
        except TelegramDeliveryFailure as failure:
            return await self._settle_failure(claim, None, failure)

        while True:
            fragment = await self._fragments.begin_next(claim)
            if fragment is None:
                state = await self._outbox.mark_succeeded(claim)
                return self._success_result(claim, state)
            try:
                external_message_id = await self._adapter.deliver_fragment(claim, fragment)
            except TelegramDeliveryFailure as failure:
                return await self._settle_failure(claim, fragment, failure)
            await self._fragments.mark_sent(
                claim,
                fragment,
                external_message_id=external_message_id,
            )

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

    async def _settle_failure(
        self,
        claim: DeliveryClaim,
        fragment: DeliveryFragment | None,
        failure: TelegramDeliveryFailure,
    ) -> TelegramWorkResult:
        if fragment is not None:
            if failure.ambiguous:
                await self._fragments.mark_uncertain(claim, fragment)
            else:
                await self._fragments.release_after_definitive_failure(claim, fragment)
        state = await self._outbox.mark_failed(
            claim,
            retryable=failure.retryable and not failure.ambiguous,
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
